"""Placement search (the paper's Technique 3 / Algorithm 3, approximated).

We cannot run a DP over PostgreSQL's plan tree from outside the server, so we enumerate:

  * every left-deep join order that starts with the driving table and stays connected
  * for each order, every subset of the selected composite candidates
    (a composite sits either at the lowest level where its columns exist, or is pulled
    up into its super-predicate -- at the super-predicate's level it is implied by it,
    so "pull up" and "merge away" are the same thing)

and score each variant two ways:

  1. `est_cost`   -- our own cost model: sample-based cardinalities x PostgreSQL unit
                     costs (cpu_tuple_cost, cpu_operator_cost, function COST). This is
                     what the paper's cost model consumes: row estimates and per-predicate
                     evaluation cost.
  2. `pg_cost`    -- PostgreSQL's own EXPLAIN Total Cost for the generated SQL. Shown for
                     comparison; PostgreSQL has no selectivity estimate for a composite
                     expression (it assumes 1/3), so it undervalues selective composites.

Only the winner is executed, plus the all-down and all-up extremes for the placement
scenario.
"""
from __future__ import annotations

import itertools
import json
from dataclasses import dataclass, field

import numpy as np
import psycopg

from .catalog import Model
from .derive import Bound
from .emit import Placement, fenced_sql, flat_atomic_sql
from .query import Query
from .select import Candidate, SampleEstimator, candidates_for_order

# PostgreSQL defaults
CPU_TUPLE = 0.01
CPU_OP = 0.0025
UDF_COST = 500 * CPU_OP          # matches COST 500 on lr_predict / lr_partial
INLINE_OPS = 3                   # ~ a few arithmetic operators + a comparison


@dataclass
class Variant:
    placement: Placement | None      # None == flat atomic (no fences)
    sql: str
    candidates: list[Candidate] = field(default_factory=list)
    est_cost: float = float("nan")
    pg_cost: float = float("nan")
    est_rows_to_inference: float = float("nan")

    @property
    def label(self) -> str:
        return "flat atomic (planner decides)" if self.placement is None else self.placement.label


def join_orders(q: Query) -> list[tuple[str, ...]]:
    """All connected left-deep orders starting from the driving table."""
    edges = q.join_edges()
    adj: dict[str, set[str]] = {t.alias: set() for t in q.tables}
    for x, y in edges:
        adj[x].add(y); adj[y].add(x)
    rest = [t.alias for t in q.tables[1:]]
    first = q.tables[0].alias
    out = []
    for perm in itertools.permutations(rest):
        seen = {first}
        ok = True
        for a in perm:
            if not (adj[a] & seen):
                ok = False; break
            seen.add(a)
        if ok:
            out.append((first,) + perm)
    return out


def _est_cost(q: Query, model: Model, bounds: list[Bound], placement: Placement,
              cands: list[Candidate], est: SampleEstimator, table_rows: dict[str, int]) -> tuple[float, float]:
    """Cardinality-based cost of a fenced variant. Returns (cost, rows reaching inference)."""
    feats_by_alias: dict[str, list] = {}
    for f in model.features:
        feats_by_alias.setdefault(q.alias_of[f.table], []).append(f)
    by_key = {b.feature.key: b for b in bounds}
    order = placement.order

    # rows of the joined space surviving predicates applied so far
    mask = np.ones(len(est.X), dtype=bool)

    def apply_atomics(alias):
        nonlocal mask
        for f in feats_by_alias.get(alias, []):
            b = by_key[f.key]
            x = est.col(f.key)
            mask &= (x >= b.lo - 1e-9) & (x <= b.hi + 1e-9)

    cost = 0.0
    # driving table scan + its atomics
    cost += table_rows[q.table_of[order[0]]] * (CPU_TUPLE + CPU_OP)
    apply_atomics(order[0])
    rows = mask.sum() * est.scale

    cand_by_level = {c.level: c for c in cands}
    for level in range(1, len(order)):
        new = order[level]
        # build side scan of the new table; probe with the current stream
        cost += table_rows[q.table_of[new]] * (CPU_TUPLE + CPU_OP)
        cost += rows * (CPU_TUPLE + CPU_OP)
        apply_atomics(new)
        rows = mask.sum() * est.scale
        comp = placement.composites.get(level)
        if comp is not None:
            per_row = UDF_COST if placement.mode == "udf" else INLINE_OPS * CPU_OP
            cost += rows * per_row
            mask &= est.mask_composite(comp)
            rows = mask.sum() * est.scale
    # top-level ML inference on whatever is left
    cost += rows * UDF_COST
    return cost, rows


def enumerate_variants(
    q: Query, model: Model, bounds: list[Bound], l_low, l_up,
    est: SampleEstimator, mode: str = "inline", min_gain: float = 0.05,
) -> list[Variant]:
    variants = [Variant(None, flat_atomic_sql(q, model, bounds))]
    for order in join_orders(q):
        cands = candidates_for_order(q, model, bounds, l_low, l_up, order, est, min_gain)
        selected = [c for c in cands if c.selected]
        for r in range(len(selected) + 1):
            for subset in itertools.combinations(selected, r):
                pl = Placement(order, {c.level: c.composite for c in subset}, mode)
                variants.append(Variant(pl, fenced_sql(q, model, bounds, pl), cands))
    return variants


def score(conn: psycopg.Connection, q: Query, model: Model, bounds: list[Bound],
          variants: list[Variant], est: SampleEstimator) -> list[Variant]:
    table_rows = {
        t: int(n) for t, n in conn.execute(
            "SELECT relname, GREATEST(reltuples, 1) FROM pg_class WHERE relname = ANY(%s)",
            ([t.table for t in q.tables],))
    }
    for v in variants:
        (plan,) = conn.execute("EXPLAIN (FORMAT JSON) " + v.sql).fetchone()
        if isinstance(plan, str):
            plan = json.loads(plan)
        v.pg_cost = float(plan[0]["Plan"]["Total Cost"])
        if v.placement is None:
            # flat: model it as the best fenced order with no composites
            best = None
            for order in join_orders(q):
                c, rows = _est_cost(q, model, bounds, Placement(order, {}), [], est, table_rows)
                if best is None or c < best[0]:
                    best = (c, rows)
            v.est_cost, v.est_rows_to_inference = best
        else:
            v.est_cost, v.est_rows_to_inference = _est_cost(
                q, model, bounds, v.placement, v.candidates, est, table_rows)
    return variants


def choose(variants: list[Variant], scorer: str = "sample") -> Variant:
    key = (lambda v: v.est_cost) if scorer == "sample" else (lambda v: v.pg_cost)
    return min(variants, key=key)
