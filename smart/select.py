"""Composite predicate selection (the paper's Algorithm 2, approximated).

The paper picks composite predicates whose extra pruning justifies their cost. We do the
same with a fixed 1% sample of the joined space (`apartment_sample`): for every prefix
of a join order, derive the composite over the features available so far and measure
how many of the rows that survive the atomic predicates it removes on top.

Only composites with a gain of at least `min_gain` are kept as candidates; the placement
search (place.py) then decides where, if anywhere, each candidate is evaluated.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import psycopg

from .catalog import Model
from .derive import Bound, Composite, composite
from .query import Query


class SampleEstimator:
    """Row-level sample of the joined space with one column per feature."""

    def __init__(self, conn: psycopg.Connection, model: Model, total_rows: int | None = None):
        cols = [f.col for f in model.features]
        cur = conn.execute(f"SELECT {', '.join(cols)} FROM apartment_sample")
        self.X = np.array(cur.fetchall(), dtype=float)
        self.keys = [f.key for f in model.features]
        self.model = model
        if total_rows is None:
            total_rows = conn.execute("SELECT count(*) FROM apartment").fetchone()[0]
        self.total_rows = int(total_rows)
        self.scale = self.total_rows / len(self.X)   # sample row -> estimated real rows

    def col(self, key: str) -> np.ndarray:
        return self.X[:, self.keys.index(key)]

    def predictions(self) -> np.ndarray:
        W = np.array([f.w for f in self.model.features])
        return self.model.intercept + self.X @ W

    def threshold_for_selectivity(self, sel: float) -> float:
        """Threshold T such that  P(model(x) < T) ~= sel  over the joined space."""
        return float(np.quantile(self.predictions(), sel))

    def mask_atomic(self, bounds: list[Bound]) -> np.ndarray:
        m = np.ones(len(self.X), dtype=bool)
        for b in bounds:
            x = self.col(b.feature.key)
            m &= (x >= b.lo - 1e-9) & (x <= b.hi + 1e-9)
        return m

    def mask_composite(self, c: Composite) -> np.ndarray:
        s = sum(f.w * self.col(f.key) for f in c.features)
        m = np.ones(len(self.X), dtype=bool)
        if c.rhs_hi is not None:
            m &= s <= c.rhs_hi + 1e-9
        if c.rhs_lo is not None:
            m &= s >= c.rhs_lo - 1e-9
        return m

    def mask_ml(self, l_low: float | None, l_up: float | None) -> np.ndarray:
        p = self.predictions()
        m = np.ones(len(self.X), dtype=bool)
        if l_up is not None:
            m &= p <= l_up
        if l_low is not None:
            m &= p >= l_low
        return m


@dataclass
class Candidate:
    level: int                 # emitted after the join that completes order[:level+1]
    tables: tuple[str, ...]    # aliases covered
    composite: Composite
    rows_before: float         # estimated rows entering this level (after everything below)
    rows_after: float          # estimated rows after applying this composite
    selected: bool

    @property
    def gain(self) -> float:
        return 1.0 - self.rows_after / self.rows_before if self.rows_before else 0.0


def candidates_for_order(
    q: Query, model: Model, bounds: list[Bound], l_low, l_up,
    order: tuple[str, ...], est: SampleEstimator, min_gain: float = 0.05,
) -> list[Candidate]:
    """Composite candidates along one join order, with sample-based gains.

    Gains are cumulative: each candidate is measured on the rows left by the atomic
    predicates *and* the previously selected composites below it.
    """
    feats_by_alias: dict[str, list] = {}
    for f in model.features:
        feats_by_alias.setdefault(q.alias_of[f.table], []).append(f)

    current = est.mask_atomic(bounds)
    out = []
    for level in range(1, len(order) - 1):           # the full prefix IS the ML predicate
        covered = order[: level + 1]
        keys = {f.key for a in covered for f in feats_by_alias.get(a, []) if f.w != 0}
        if len(keys) < 2:
            continue                                  # a single feature is already atomic
        comp = composite(model.features, model.intercept, l_up, l_low, keys, bounds)
        after = current & est.mask_composite(comp)
        c = Candidate(level, tuple(covered), comp,
                      rows_before=current.sum() * est.scale,
                      rows_after=after.sum() * est.scale,
                      selected=False)
        c.selected = c.gain >= min_gain
        if c.selected:
            current = after
        out.append(c)
    return out
