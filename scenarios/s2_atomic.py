"""Scenario 2 - Technique 1: atomic predicates only. Large win, minimal machinery."""
from smart.harness import bounds_table, console, explain_text, report, show_sql
from smart.rewriter import rewrite

from .common import demo_query, measure, save, threshold_for


def run(conn, selectivity: float, runs: int):
    console.rule("[bold]Scenario 2 - atomic predicates (Technique 1)")
    T, est = threshold_for(conn, selectivity)
    q = demo_query(T)
    rw = rewrite(conn, q)
    console.print(f"ML predicate: predicted price < {T}")
    bounds_table(rw.model, rw.bounds, rw.trace)
    show_sql(rw.flat_atomic_sql, "Rewritten: atomic predicates appended to WHERE; PostgreSQL pushes them to the scans")
    console.print("[bold]Plan[/] - the derived predicates appear as Filter: on the Seq Scans:")
    console.print(explain_text(conn, rw.flat_atomic_sql))
    base = measure(conn, "baseline", rw.baseline_sql, runs)
    atom = measure(conn, "Smart: atomic only", rw.flat_atomic_sql, runs, baseline_sql=rw.baseline_sql)
    report([base, atom], title="Scenario 2")
    save("s2", {"threshold": T, "baseline_ms": base.exec_ms, "atomic_ms": atom.exec_ms,
                "rows_base": base.rows_to_inference, "rows_atomic": atom.rows_to_inference,
                "valid": atom.valid})
    return base, atom
