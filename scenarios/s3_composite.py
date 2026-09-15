"""Scenario 3 - Techniques 1+2: composite predicates with progressive inference."""
from smart.harness import console, explain_text, report, show_sql
from smart.rewriter import rewrite

from .common import candidates_table, demo_query, measure, save, threshold_for, variants_table


def run(conn, selectivity: float, runs: int):
    console.rule("[bold]Scenario 3 - atomic + composite + progressive inference (Techniques 1+2)")
    T, est = threshold_for(conn, selectivity)
    q = demo_query(T)
    rw = rewrite(conn, q, mode="inline")
    console.print(f"ML predicate: predicted price < {T}")
    order = ("a", "b", "d", "l")
    candidates_table(rw, order)
    variants_table(rw)
    console.print(f"[bold]Chosen:[/] {rw.chosen.label}")
    show_sql(rw.smart_sql, "Rewritten: one fenced level per join; partial_sum carried upward (progressive inference)")
    console.print("[bold]Plan[/] - composites sit as Join Filters at their own join level:")
    console.print(explain_text(conn, rw.smart_sql))
    base = measure(conn, "baseline", rw.baseline_sql, runs)
    atom = measure(conn, "Smart: atomic only", rw.flat_atomic_sql, runs, baseline_sql=rw.baseline_sql)
    comp = measure(conn, "Smart: atomic + composite", rw.smart_sql, runs, baseline_sql=rw.baseline_sql)
    report([base, atom, comp], title="Scenario 3")
    save("s3", {"threshold": T, "baseline_ms": base.exec_ms, "atomic_ms": atom.exec_ms,
                "composite_ms": comp.exec_ms, "rows_atomic": atom.rows_to_inference,
                "rows_composite": comp.rows_to_inference, "chosen": rw.chosen.label,
                "valid": comp.valid})
    return base, atom, comp
