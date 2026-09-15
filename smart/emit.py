"""SQL generation.

Three shapes:

* baseline      -- the user's query as written, ML predicate at the top
* flat_atomic   -- baseline + derived atomic predicates in the WHERE clause;
                   PostgreSQL pushes them to the scans by itself (Technique 1)
* fenced        -- a left-deep chain of subqueries, one per join, with composite
                   predicates at chosen levels and the running partial sum carried
                   upward (Techniques 2 + 3, progressive inference).

Each fenced level ends in OFFSET 0. That stops the planner from flattening the subquery
or pushing quals across it, so predicates stay exactly where we put them, while rows
still stream (unlike a MATERIALIZED CTE, nothing is spooled to a tuplestore).

Bounds are widened by a tiny epsilon before emission. Widening preserves validity
(a necessary condition stays necessary); narrowing would destroy it.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from .catalog import Model
from .derive import Bound, Composite, Feature
from .query import Query, cols_in, refs_in, rewrite_refs

EPS_REL = 1e-9
EPS_ABS = 1e-9


def _num(v: float) -> str:
    return f"{v:.12g}"


def _widen(lo: float | None, hi: float | None) -> tuple[float | None, float | None]:
    if lo is not None:
        lo = lo - (EPS_ABS + EPS_REL * abs(lo))
    if hi is not None:
        hi = hi + (EPS_ABS + EPS_REL * abs(hi))
    return lo, hi


def atomic_conditions(q: Query, bounds: list[Bound], ref=None) -> list[tuple[str, str]]:
    """[(alias, 'a.room_num <= 2.000000002'), ...] for every non-trivial bound."""
    ref = ref or (lambda alias, col: f"{alias}.{col}")
    out = []
    for b in bounds:
        if b.is_trivial:
            continue
        alias = q.alias_of[b.feature.table]
        col = ref(alias, b.feature.col)
        lo = b.lo if b.lo > b.feature.lo else None
        hi = b.hi if b.hi < b.feature.hi else None
        lo, hi = _widen(lo, hi)
        if b.feature.is_integer:
            # x integer and x <= hi  <=>  x <= floor(hi): still a necessary condition
            lo = None if lo is None else math.ceil(lo)
            hi = None if hi is None else math.floor(hi)
        if lo is not None:
            out.append((alias, f"{col} >= {_num(lo)}"))
        if hi is not None:
            out.append((alias, f"{col} <= {_num(hi)}"))
    return out


# --------------------------------------------------------------------------------------
# Flat shapes
# --------------------------------------------------------------------------------------

def _flat_from(q: Query) -> str:
    parts = [f"{q.tables[0].table} {q.tables[0].alias}"]
    for t in q.tables[1:]:
        parts.append(f"JOIN {t.table} {t.alias} ON {t.on}")
    return "\n  ".join(parts)


def baseline_sql(q: Query, model: Model) -> str:
    ml = q.ml.sql(model, lambda f: f"{q.alias_of[f.table]}.{f.col}")
    conds = list(q.where) + [ml]
    return (f"SELECT {', '.join(q.select)}\nFROM {_flat_from(q)}\nWHERE "
            + "\n  AND ".join(conds))


def flat_atomic_sql(q: Query, model: Model, bounds: list[Bound]) -> str:
    ml = q.ml.sql(model, lambda f: f"{q.alias_of[f.table]}.{f.col}")
    atoms = [c for _, c in atomic_conditions(q, bounds)]
    conds = list(q.where) + atoms + [ml]
    return (f"SELECT {', '.join(q.select)}\nFROM {_flat_from(q)}\nWHERE "
            + "\n  AND ".join(conds))


# --------------------------------------------------------------------------------------
# Fenced shape: one level per join, composites at chosen levels
# --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Placement:
    """Which composite (keyed by level = number of joined tables - 1) is emitted where."""
    order: tuple[str, ...]                  # join order as aliases, driving table first
    composites: dict                        # level -> Composite (only the ones to emit)
    mode: str = "inline"                    # 'inline' arithmetic | 'udf' partial inference

    @property
    def label(self) -> str:
        levels = ",".join(f"L{k}" for k in sorted(self.composites)) or "none"
        return f"order={'>'.join(self.order)} composites@{levels} ({self.mode})"


def _needed_columns(q: Query, model: Model) -> dict[str, set[str]]:
    """alias -> columns that must survive through the levels."""
    need: dict[str, set[str]] = {t.alias: set() for t in q.tables}
    for expr in list(q.select) + list(q.where) + [t.on for t in q.tables if t.on]:
        for a, c in cols_in(expr):
            if a in need:
                need[a].add(c)
    for f in model.features:
        need[q.alias_of[f.table]].add(f.col)
    return need


def fenced_sql(q: Query, model: Model, bounds: list[Bound], placement: Placement) -> str:
    order = list(placement.order)
    assert order[0] == q.tables[0].alias, "join order must start with the driving table"
    need = _needed_columns(q, model)
    feats_by_alias: dict[str, list[Feature]] = {}
    for f in model.features:
        feats_by_alias.setdefault(q.alias_of[f.table], []).append(f)
    atoms = atomic_conditions(q, bounds)
    where_left = list(q.where)
    highest_composite = max(placement.composites) if placement.composites else 0

    ctes = []
    for level in range(1, len(order)):
        new = order[level]
        prev_aliases = order[:level]
        prev = f"lvl{level - 1}"

        # how to reference a column at this level
        def ref(alias: str, col: str, _new=new, _prev=prev, _first=(level == 1)) -> str:
            if alias == _new or (_first and alias == order[0]):
                return f"{alias}.{col}"
            return f"{_prev}.{alias}__{col}"

        # FROM
        if level == 1:
            frm = f"{q.table_of[order[0]]} {order[0]}\n    JOIN {q.table_of[new]} {new} ON {rewrite_refs(q.on_clause(new), ref)}"
        else:
            frm = f"{prev}\n    JOIN {q.table_of[new]} {new} ON {rewrite_refs(q.on_clause(new), ref)}"

        # predicates available at this level
        conds = []
        available = set(prev_aliases) | {new}
        for alias, cond in atoms:
            if alias == new or (level == 1 and alias == order[0]):
                conds.append(rewrite_refs(cond, ref))
        for p in list(where_left):
            if refs_in(p) <= available:
                conds.append(rewrite_refs(p, ref))
                where_left.remove(p)

        # projection
        proj = []
        for alias in order[: level + 1]:
            for col in sorted(need[alias]):
                proj.append(f"{ref(alias, col)} AS {alias}__{col}")

        # progressive partial sum
        feats_here = [f for a in order[: level + 1] for f in feats_by_alias.get(a, [])]
        carry = level <= highest_composite
        partial_expr = None
        if carry:
            if placement.mode == "inline":
                new_terms = " + ".join(
                    f"{_num(f.w)}::float8 * {ref(new, f.col)}" for f in feats_by_alias.get(new, [])
                )
                if level == 1:
                    first_terms = " + ".join(
                        f"{_num(f.w)}::float8 * {ref(order[0], f.col)}" for f in feats_by_alias.get(order[0], [])
                    )
                    partial_expr = " + ".join(x for x in [first_terms, new_terms] if x) or "0"
                else:
                    partial_expr = " + ".join(x for x in [f"{prev}.partial_sum", new_terms] if x)
            else:  # udf: partial inference through the model, recomputed at this level
                idx = ", ".join(str(model.feature_index(f.key)) for f in feats_here)
                vals = ", ".join(ref(q.alias_of[f.table], f.col) for f in feats_here)
                partial_expr = f"lr_partial('{model.name}', ARRAY[{idx}], ARRAY[{vals}]::numeric[])"
            proj.append(f"{partial_expr} AS partial_sum")

        comp: Composite | None = placement.composites.get(level)
        inner = (f"SELECT {', '.join(proj)}\n    FROM {frm}"
                 + (("\n    WHERE " + "\n      AND ".join(conds)) if conds else ""))
        if comp is not None:
            offset = model.intercept if placement.mode == "udf" else 0.0
            lo, hi = _widen(comp.rhs_lo, comp.rhs_hi)
            cc = []
            if hi is not None:
                cc.append(f"s.partial_sum <= {_num(hi + offset)}")
            if lo is not None:
                cc.append(f"s.partial_sum >= {_num(lo + offset)}")
            body = (f"SELECT s.* FROM (\n    {inner}\n  ) s\n  WHERE " + " AND ".join(cc)
                    + f"   -- composite P({','.join(order[:level + 1])})")
        else:
            body = inner
        ctes.append(f"lvl{level} AS (\n  {body}\n  OFFSET 0\n)")

    top = f"lvl{len(order) - 1}"

    def top_ref(alias: str, col: str) -> str:
        return f"{top}.{alias}__{col}"

    select = []
    for item in q.select:
        cols = cols_in(item)
        if len(cols) == 1 and item.strip() == ".".join(next(iter(cols))):
            a, c = next(iter(cols))
            select.append(f"{top_ref(a, c)} AS {c}")
        else:
            select.append(rewrite_refs(item, top_ref))
    ml = q.ml.sql(model, lambda f: top_ref(q.alias_of[f.table], f.col))
    conds = [rewrite_refs(p, top_ref) for p in where_left] + [ml]
    return ("WITH " + ",\n".join(ctes) + f"\nSELECT {', '.join(select)}\nFROM {top}\nWHERE "
            + "\n  AND ".join(conds) + "   -- original ML predicate retained")
