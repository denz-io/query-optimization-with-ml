"""Timing, plan metrics, the validity assertion and terminal reporting."""
from __future__ import annotations

import json
import time
from pathlib import Path
from dataclasses import dataclass, field

import psycopg
from rich.console import Console
from rich.table import Table

import os, sys
console = Console(width=None if sys.stdout.isatty() else int(os.environ.get("SMART_WIDTH", 140)))

INFERENCE_FUNCS = ("lr_predict", "logit_classify")


def explain_analyze(conn: psycopg.Connection, sql: str) -> dict:
    (plan,) = conn.execute("EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) " + sql).fetchone()
    if isinstance(plan, str):
        plan = json.loads(plan)
    return plan[0]


def explain_text(conn: psycopg.Connection, sql: str, analyze: bool = False) -> str:
    opts = "ANALYZE, COSTS OFF, TIMING OFF, SUMMARY OFF" if analyze else "COSTS OFF"
    return "\n".join(line for (line,) in conn.execute(f"EXPLAIN ({opts}) " + sql))


@dataclass
class Measurement:
    label: str
    sql: str
    exec_ms: float                 # minimum over runs, as the paper reports
    plan_ms: float
    rows_to_inference: int         # rows entering the node that evaluates the ML UDF
    result_rows: int
    shared_hit: int
    shared_read: int
    runs: list[float] = field(default_factory=list)
    plan: dict | None = None
    valid: bool | None = None


def _walk(node: dict):
    yield node
    for child in node.get("Plans", []):
        yield from _walk(child)


def plan_metrics(plan: dict) -> dict:
    root = plan["Plan"]
    rows_in = 0
    for n in _walk(root):
        texts = " ".join(str(n.get(k, "")) for k in ("Filter", "Join Filter", "One-Time Filter"))
        if any(f in texts for f in INFERENCE_FUNCS):
            loops = n.get("Actual Loops", 1)
            rows_in += loops * (n.get("Actual Rows", 0)
                                + n.get("Rows Removed by Filter", 0)
                                + n.get("Rows Removed by Join Filter", 0))
    return {
        "rows_to_inference": int(rows_in),
        "result_rows": int(root.get("Actual Rows", 0)),
        "shared_hit": int(root.get("Shared Hit Blocks", 0)),
        "shared_read": int(root.get("Shared Read Blocks", 0)),
    }


def timed(conn: psycopg.Connection, label: str, sql: str, runs: int = 3) -> Measurement:
    times, plans = [], []
    for _ in range(runs):
        p = explain_analyze(conn, sql)
        times.append(p["Execution Time"]); plans.append(p)
    best = plans[times.index(min(times))]
    m = plan_metrics(best)
    return Measurement(label, sql, min(times), best["Planning Time"], m["rows_to_inference"],
                       m["result_rows"], m["shared_hit"], m["shared_read"], times, best)


# --------------------------------------------------------------------------------------
# Validity: the two result sets must be identical
# --------------------------------------------------------------------------------------

_FP_CACHE: dict[str, tuple] = {}
_FP_FILE = Path(__file__).resolve().parent.parent / "results" / "fingerprint_cache.json"


def fingerprint(conn: psycopg.Connection, sql: str, key: str = "aid") -> tuple:
    """Order-independent multiset fingerprint of a result: (count, sum(hash(key)), sum(key)).
    Computed inside the database so a 100%-selectivity result never crosses the wire.
    Baseline fingerprints are cached on disk (keyed by SQL + fact-table row count) because
    each one costs a full baseline run; Smart fingerprints are always recomputed."""
    n = conn.execute("SELECT count(*) FROM apartment").fetchone()[0]
    ck = f"{n}|{sql}"
    if ck in _FP_CACHE:
        return _FP_CACHE[ck]
    disk = {}
    if "lr_predict" in sql and "WITH" not in sql and os.environ.get("SMART_FRESH") != "1":
        try:
            disk = json.loads(_FP_FILE.read_text())
        except (OSError, ValueError):
            disk = {}
        if ck in disk:
            _FP_CACHE[ck] = tuple(disk[ck])
            return _FP_CACHE[ck]
    row = conn.execute(
        f"SELECT count(*), COALESCE(sum(hashtextextended(({key})::text, 0)), 0)::text, COALESCE(sum({key}), 0)::text "
        f"FROM (\n{sql}\n) q"
    ).fetchone()
    _FP_CACHE[ck] = tuple(row)
    if disk is not None and "WITH" not in sql:
        disk[ck] = list(row)
        _FP_FILE.parent.mkdir(exist_ok=True)
        _FP_FILE.write_text(json.dumps(disk))
    return tuple(row)


def validate(conn: psycopg.Connection, baseline_sql: str, smart_sql: str, key: str = "aid") -> bool:
    a, b = fingerprint(conn, baseline_sql, key), fingerprint(conn, smart_sql, key)
    ok = a == b
    if ok:
        console.print(f"  [bold white on green]  VALIDITY CHECK PASSED  [/]  "
                      f"identical result sets, {a[0]:,} rows")
    else:
        console.print(f"  [bold white on red]  VALIDITY VIOLATION  [/]  baseline {a[0]:,} rows, "
                      f"smart {b[0]:,} rows -- derived predicates dropped or added rows")
    return ok


# --------------------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------------------

def report(measurements: list[Measurement], title: str = "") -> None:
    base = measurements[0]
    t = Table(title=title or None, show_lines=False, header_style="bold")
    t.add_column("variant"); t.add_column("exec ms", justify="right")
    t.add_column("speed-up", justify="right"); t.add_column("rows -> inference", justify="right")
    t.add_column("reduction", justify="right"); t.add_column("result rows", justify="right")
    t.add_column("buffers hit+read", justify="right"); t.add_column("plan ms", justify="right")
    t.add_column("valid", justify="center")
    for m in measurements:
        speed = base.exec_ms / m.exec_ms if m.exec_ms else float("inf")
        red = base.rows_to_inference / m.rows_to_inference if m.rows_to_inference else float("inf")
        valid = "" if m.valid is None else ("[green]OK[/]" if m.valid else "[red]FAIL[/]")
        t.add_row(m.label, f"{m.exec_ms:,.0f}", f"{speed:,.1f}x", f"{m.rows_to_inference:,}",
                  f"{red:,.0f}x", f"{m.result_rows:,}",
                  f"{m.shared_hit + m.shared_read:,}", f"{m.plan_ms:.2f}", valid)
    console.print(t)


def show_sql(sql: str, title: str) -> None:
    from rich.syntax import Syntax
    console.print(f"[bold]{title}[/]")
    console.print(Syntax(sql, "sql", theme="ansi_dark", word_wrap=True))


def bounds_table(model, bounds, trace=None) -> None:
    t = Table(title="Atomic predicates (Theorem 1 / Algorithm 1)", header_style="bold")
    t.add_column("feature"); t.add_column("weight", justify="right")
    t.add_column("catalog min", justify="right"); t.add_column("catalog max", justify="right")
    t.add_column("derived lo", justify="right"); t.add_column("derived hi", justify="right")
    t.add_column("emitted")
    for b in bounds:
        f = b.feature
        lo = "" if b.lo <= f.lo else f"{b.lo:.4g}"
        hi = "" if b.hi >= f.hi else f"{b.hi:.4g}"
        pred = " AND ".join(x for x in [f"{f.col} >= {lo}" if lo else "", f"{f.col} <= {hi}" if hi else ""] if x)
        t.add_row(f.key, f"{f.w:.4f}", f"{f.lo:g}", f"{f.hi:g}", lo or "[dim]-[/]", hi or "[dim]-[/]",
                  pred or "[dim](no pruning possible)[/]")
    console.print(t)
    if trace:
        console.print(f"  Algorithm 1 converged after {len(trace) - 1} pass(es); "
                      f"max change per pass: " + ", ".join(f"{p['changed']:.3g}" for p in trace[1:]))
