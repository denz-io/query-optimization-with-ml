"""A structured SQL+ML query. The demo rewrites a query *spec*, not arbitrary SQL text:
parsing general SQL is not where the paper's ideas are."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .catalog import Model
from .derive import logistic_to_linear

_REF = re.compile(r"\b([A-Za-z_]\w*)\.([A-Za-z_]\w*)\b")
_SIMPLE_PRED = re.compile(
    r"^\s*([A-Za-z_]\w*)\.([A-Za-z_]\w*)\s*(<=|>=|<|>|=)\s*(-?\d+(?:\.\d+)?)\s*$"
)


@dataclass(frozen=True)
class JoinTable:
    table: str
    alias: str
    on: str | None = None      # e.g. "a.bid = b.bid"; None for the driving table


@dataclass(frozen=True)
class MLPredicate:
    """model(x) OP threshold.  op in {'<','<=','>','>=','between','='}.
    For 'between' threshold is (low, high). For logistic models op is '=' and threshold a bool."""
    model: str
    op: str
    threshold: float | tuple[float, float] | bool

    def linear_bounds(self, model: Model) -> tuple[float | None, float | None]:
        """(l_low, l_up) on the linear part w0 + sum(w_i x_i)."""
        if model.category == "logistic":
            if self.op != "=" or not isinstance(self.threshold, bool):
                raise ValueError("logistic predicates must be  = TRUE / = FALSE")
            return logistic_to_linear(self.threshold)
        if self.op in ("<", "<="):
            return None, float(self.threshold)
        if self.op in (">", ">="):
            return float(self.threshold), None
        if self.op == "between":
            lo, hi = self.threshold
            return float(lo), float(hi)
        raise ValueError(f"unsupported op {self.op!r} for a {model.category} model")

    def sql(self, model: Model, feature_ref) -> str:
        """Render the ORIGINAL ML predicate. `feature_ref(feature) -> str` supplies column refs."""
        arr = "ARRAY[" + ", ".join(feature_ref(f) for f in model.features) + "]::numeric[]"
        if model.category == "logistic":
            return f"logit_classify('{model.name}', {arr}) = {'TRUE' if self.threshold else 'FALSE'}"
        call = f"lr_predict('{model.name}', {arr})"
        if self.op == "between":
            lo, hi = self.threshold
            return f"{call} BETWEEN {lo} AND {hi}"
        return f"{call} {self.op} {self.threshold}"


@dataclass(frozen=True)
class Query:
    select: tuple[str, ...]            # alias-qualified columns, e.g. "a.aid"
    tables: tuple[JoinTable, ...]      # driving table first
    ml: MLPredicate
    where: tuple[str, ...] = field(default_factory=tuple)   # existing plain predicates
    key: str = "aid"                   # output column used for the validity fingerprint

    # -- helpers ----------------------------------------------------------------------
    @property
    def alias_of(self) -> dict[str, str]:
        return {t.table: t.alias for t in self.tables}

    @property
    def table_of(self) -> dict[str, str]:
        return {t.alias: t.table for t in self.tables}

    def join_edges(self) -> list[tuple[str, str]]:
        edges = []
        for t in self.tables[1:]:
            refs = {a for a, _ in _REF.findall(t.on or "")}
            refs.discard(t.alias)
            for other in refs:
                edges.append((other, t.alias))
        return edges

    def on_clause(self, alias: str) -> str:
        return next(t.on for t in self.tables if t.alias == alias)

    def seeds(self, model: Model) -> tuple[dict, dict]:
        """Existing simple predicates on feature columns become the starting bounds
        (the paper: seed Algorithm 1 from the query's own predicates)."""
        seed_lo, seed_hi = {}, {}
        feat_by_ref = {(self.alias_of[f.table], f.col): f for f in model.features}
        for p in self.where:
            m = _SIMPLE_PRED.match(p)
            if not m:
                continue
            alias, col, op, val = m.groups()
            f = feat_by_ref.get((alias, col))
            if f is None:
                continue
            v = float(val)
            if op in (">", ">="):
                seed_lo[f.key] = max(seed_lo.get(f.key, -1e300), v)
            elif op in ("<", "<="):
                seed_hi[f.key] = min(seed_hi.get(f.key, 1e300), v)
            elif op == "=":
                seed_lo[f.key] = seed_hi[f.key] = v
        return seed_lo, seed_hi


def refs_in(expr: str) -> set[str]:
    """Aliases referenced by an expression."""
    return {a for a, _ in _REF.findall(expr)}


def cols_in(expr: str) -> set[tuple[str, str]]:
    return set(_REF.findall(expr))


def rewrite_refs(expr: str, fn) -> str:
    """Replace every alias.col reference using fn(alias, col) -> str."""
    return _REF.sub(lambda m: fn(m.group(1), m.group(2)), expr)
