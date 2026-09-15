"""One-shot environment build: schema -> data -> catalog -> UDFs -> train -> populate.

    python -m scripts.setup_db            # full build using row counts from .env
    python -m scripts.setup_db --skip-data  # re-train / re-populate only

Idempotent: every step drops and recreates what it owns.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from rich.console import Console
from sklearn.linear_model import LinearRegression, LogisticRegression

from smart.catalog import connect

load_dotenv()
ROOT = Path(__file__).resolve().parent.parent
SQL = ROOT / "sql"
console = Console()

# Feature order == weights JSONB order == coef[] order. Everything keys off this.
FEATURES = [
    ("apartment", "room_num"),
    ("building", "building_age"),
    ("district", "hospital_num"),
    ("landlord", "rating_score"),
]


def step(title: str):
    console.rule(f"[bold]{title}")


def run_sql_file(conn, name: str, **subs):
    text = (SQL / name).read_text()
    for k, v in subs.items():
        text = text.replace("{" + k + "}", str(v))
    t0 = time.perf_counter()
    conn.execute(text)
    console.print(f"  {name}  [dim]{time.perf_counter() - t0:.1f}s[/]")


def train_and_populate(conn):
    cur = conn.execute("SELECT * FROM apartment_sample")
    df = pd.DataFrame(cur.fetchall(), columns=[d.name for d in cur.description]).astype(float)
    cols = [c for _, c in FEATURES]
    X = df[cols].astype(float).to_numpy()
    console.print(f"  training sample: {len(df):,} rows")

    # Linear regression predicting price
    lr = LinearRegression().fit(X, df["price"].astype(float))
    console.print("  price_lr  coef =", np.round(lr.coef_, 4), " intercept =", round(float(lr.intercept_), 4))

    # Logistic regression: "pricey" = above the 80th percentile of price
    thr = float(df["price"].quantile(0.8))
    y = (df["price"].astype(float) > thr).astype(int)
    logit = LogisticRegression(max_iter=5000).fit(X, y)
    console.print("  pricey_logit coef =", np.round(logit.coef_[0], 4),
                  " intercept =", round(float(logit.intercept_[0]), 4), f" (price > {thr:.2f})")

    conn.execute("DELETE FROM sys_model")
    for name, cat, coef, b0 in [
        ("price_lr", "linear", lr.coef_, float(lr.intercept_)),
        ("pricey_logit", "logistic", logit.coef_[0], float(logit.intercept_[0])),
    ]:
        weights = [{"table": t, "col": c, "w": float(w)} for (t, c), w in zip(FEATURES, coef)]
        conn.execute(
            "INSERT INTO sys_model (model_name, model_category, intercept, weights, coef) "
            "VALUES (%s, %s, %s, %s::jsonb, %s)",
            (name, cat, b0, json.dumps(weights), [float(w) for w in coef]),
        )

    # Column statistics from the REAL data, not guesses.
    conn.execute("DELETE FROM sys_feature")
    for t, c in FEATURES:
        conn.execute(
            f"INSERT INTO sys_feature SELECT %s, %s, MIN({c}), MAX({c}) FROM {t}", (t, c)
        )
    for row in conn.execute("SELECT * FROM sys_feature ORDER BY 1, 2"):
        console.print("  sys_feature", row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-data", action="store_true", help="keep existing tables, only retrain")
    args = ap.parse_args()

    sizes = {k: int(os.environ.get(k, d)) for k, d in [
        ("APARTMENT_ROWS", 3_000_000), ("BUILDING_ROWS", 300_000),
        ("LANDLORD_ROWS", 100_000), ("DISTRICT_ROWS", 1_000)]}

    with connect() as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS pg_stat_statements")
        if not args.skip_data:
            step("schema + data")
            console.print("  sizes:", sizes)
            run_sql_file(conn, "01_schema.sql")
            run_sql_file(conn, "02_generate_data.sql", **sizes)
        step("catalog + UDFs")
        run_sql_file(conn, "03_catalog.sql")
        run_sql_file(conn, "04_baseline_udf.sql")
        step("train models, populate sys_model / sys_feature")
        train_and_populate(conn)
        step("sizes")
        for row in conn.execute("""
            SELECT relname, n_live_tup, pg_size_pretty(pg_total_relation_size(relid))
            FROM pg_stat_user_tables ORDER BY pg_total_relation_size(relid) DESC"""):
            console.print("  ", *row)
    console.print("[green bold]done[/]")


if __name__ == "__main__":
    main()
