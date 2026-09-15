"""Selectivity sweep: baseline vs Smart across thresholds. Writes results/sweep.csv."""
import csv

from rich.table import Table

from smart.harness import console
from smart.rewriter import rewrite

from .common import RESULTS, demo_query, measure, threshold_for


def run(conn, points: list[float], runs: int):
    console.rule("[bold]Selectivity sweep")
    rows = []
    for sel in points:
        T, est = threshold_for(conn, sel)
        q = demo_query(T)
        rw = rewrite(conn, q, mode="inline")
        console.print(f"[bold]selectivity ~{sel:.1%}[/]  threshold {T}  chosen: {rw.chosen.label}")
        base = measure(conn, "baseline", rw.baseline_sql, runs)
        smart = measure(conn, "Smart", rw.smart_sql, runs, baseline_sql=rw.baseline_sql)
        actual = base.result_rows / max(base.rows_to_inference, 1)
        rows.append({"selectivity_target": sel, "selectivity_actual": actual, "threshold": T,
                     "baseline_ms": base.exec_ms, "smart_ms": smart.exec_ms,
                     "rows_base": base.rows_to_inference, "rows_smart": smart.rows_to_inference,
                     "result_rows": base.result_rows, "valid": smart.valid, "variant": rw.chosen.label})
    out = RESULTS / "sweep.csv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    t = Table(title="Selectivity sweep", header_style="bold")
    for c in ("selectivity", "threshold", "baseline ms", "Smart ms", "speed-up", "rows -> inference (Smart)", "valid"):
        t.add_column(c, justify="right")
    for r in rows:
        t.add_row(f"{r['selectivity_actual']:.2%}", f"{r['threshold']}", f"{r['baseline_ms']:,.0f}",
                  f"{r['smart_ms']:,.0f}", f"{r['baseline_ms'] / r['smart_ms']:.1f}x",
                  f"{r['rows_smart']:,}", "OK" if r["valid"] else "FAIL")
    console.print(t)
    console.print(f"written {out}")
    return rows
