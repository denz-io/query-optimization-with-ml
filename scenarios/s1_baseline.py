"""Scenario 1 - baseline pathology: the ML UDF sits on top of the full join."""
from smart.harness import console, explain_text, show_sql
from smart.rewriter import rewrite

from .common import demo_query, measure, save, threshold_for


def run(conn, selectivity: float, runs: int):
    console.rule("[bold]Scenario 1 - baseline pathology")
    T, est = threshold_for(conn, selectivity)
    q = demo_query(T)
    rw = rewrite(conn, q)
    console.print(f"ML predicate: predicted price < {T}  (~{selectivity:.1%} of joined rows)")
    show_sql(rw.baseline_sql, "The query as written")
    console.print("[bold]Plan[/] - note where the inference call sits:")
    console.print(explain_text(conn, rw.baseline_sql))
    m = measure(conn, "baseline (UDF on top of the join)", rw.baseline_sql, runs)
    console.print(f"\n  [bold red]{m.rows_to_inference:,}[/] rows fed into inference to produce "
                  f"[bold]{m.result_rows:,}[/] results; {m.exec_ms:,.0f} ms")
    save("s1", {"threshold": T, "exec_ms": m.exec_ms, "rows_to_inference": m.rows_to_inference,
                "result_rows": m.result_rows})
    return m
