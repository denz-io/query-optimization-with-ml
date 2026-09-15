"""The whole pipeline in one call: catalog -> derive -> select -> place -> SQL."""
from __future__ import annotations

from dataclasses import dataclass

import psycopg

from .catalog import Model, check_bounds, load_model
from .derive import Bound, atomic_bounds
from .emit import baseline_sql, flat_atomic_sql
from .place import Variant, choose, enumerate_variants, score
from .query import Query
from .select import SampleEstimator


class StaleStatisticsError(RuntimeError):
    pass


@dataclass
class Rewrite:
    query: Query
    model: Model
    l_low: float | None
    l_up: float | None
    bounds: list[Bound]
    trace: list[dict]
    estimator: SampleEstimator
    variants: list[Variant]
    chosen: Variant

    @property
    def baseline_sql(self) -> str:
        return baseline_sql(self.query, self.model)

    @property
    def flat_atomic_sql(self) -> str:
        return flat_atomic_sql(self.query, self.model, self.bounds)

    @property
    def smart_sql(self) -> str:
        return self.chosen.sql

    def variant(self, order: tuple[str, ...], levels: set[int], mode: str) -> Variant:
        for v in self.variants:
            if v.placement and v.placement.order == order and set(v.placement.composites) == levels \
                    and v.placement.mode == mode:
                return v
        raise KeyError((order, levels, mode))


def rewrite(conn: psycopg.Connection, q: Query, mode: str = "inline", min_gain: float = 0.05,
            scorer: str = "sample", verify_statistics: bool = True,
            estimator: SampleEstimator | None = None) -> Rewrite:
    model = load_model(conn, q.ml.model)
    if verify_statistics:
        problems = check_bounds(conn, model)
        if problems:
            raise StaleStatisticsError(
                "sys_feature is stale; derived predicates would silently drop rows: " + str(problems))
    l_low, l_up = q.ml.linear_bounds(model)
    seed_lo, seed_hi = q.seeds(model)
    bounds, trace = atomic_bounds(list(model.features), model.intercept, l_up, l_low,
                                  seed_lo=seed_lo, seed_hi=seed_hi)
    est = estimator or SampleEstimator(conn, model)
    variants = enumerate_variants(q, model, bounds, l_low, l_up, est, mode=mode, min_gain=min_gain)
    score(conn, q, model, bounds, variants, est)
    chosen = choose(variants, scorer)
    return Rewrite(q, model, l_low, l_up, bounds, trace, est, variants, chosen)
