"""Read model weights and column statistics from the two catalog tables."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass

import psycopg
from dotenv import load_dotenv

from .derive import Feature

load_dotenv()


def connect(autocommit: bool = True) -> psycopg.Connection:
    dsn = os.environ.get("SMART_DSN", "postgresql://postgres:demo@localhost:5432/smartdemo")
    return psycopg.connect(dsn, autocommit=autocommit)


@dataclass(frozen=True)
class Model:
    name: str
    category: str            # linear | logistic | tree
    intercept: float
    features: tuple[Feature, ...]   # in catalog weight order == coef[] order

    def feature_index(self, key: str) -> int:
        """1-based position of a feature in coef[], for lr_partial()."""
        for i, f in enumerate(self.features, start=1):
            if f.key == key:
                return i
        raise KeyError(key)


def load_model(conn: psycopg.Connection, name: str) -> Model:
    row = conn.execute(
        "SELECT model_category, intercept, weights FROM sys_model WHERE model_name = %s", (name,)
    ).fetchone()
    if row is None:
        raise KeyError(f"model {name!r} not in sys_model")
    category, intercept, weights = row
    if isinstance(weights, str):
        weights = json.loads(weights)
    stats = {
        (t, c): (float(lo), float(hi))
        for t, c, lo, hi in conn.execute(
            "SELECT table_name, attribute_name, min_val, max_val FROM sys_feature"
        )
    }
    dtypes = {
        (t, c): d for t, c, d in conn.execute(
            "SELECT table_name, column_name, data_type FROM information_schema.columns "
            "WHERE table_schema = 'public'")
    }
    feats = []
    for w in weights:
        lo, hi = stats[(w["table"], w["col"])]
        feats.append(Feature(table=w["table"], col=w["col"], w=float(w["w"]), lo=lo, hi=hi,
                             dtype=dtypes.get((w["table"], w["col"]), "numeric")))
    return Model(name, category, float(intercept), tuple(feats))


def check_bounds(conn: psycopg.Connection, model: Model) -> list[dict]:
    """Recompute MIN/MAX of every feature column and compare with sys_feature.

    Stale statistics are the failure mode the paper does not analyse: if any real value
    lies outside the stored [min, max], the derived predicates stop being necessary
    conditions and rows disappear from the result with no error raised.
    Returns a list of violations (empty == safe).
    """
    problems = []
    for f in model.features:
        lo, hi = conn.execute(f"SELECT MIN({f.col}), MAX({f.col}) FROM {f.table}").fetchone()
        lo, hi = float(lo), float(hi)
        if lo < f.lo or hi > f.hi:
            problems.append({"feature": f.key, "stored": (f.lo, f.hi), "actual": (lo, hi)})
    return problems
