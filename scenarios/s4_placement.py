"""Scenario 4 - Technique 3: placement matters.

Here the composite predicates are evaluated *through the model* - partial inference via
lr_partial(), which reads the catalog per row exactly like the final lr_predict() does.
That is how the paper costs a composite predicate when there is no progressive reuse.

Textbook wisdom says push every predicate down. The cost model says: an expensive
predicate that prunes less than it costs should be pulled up and merged into its
super-predicate. We enumerate all four placements along a>b>d>l, rank them by the cost
model, run all four, and let the timings decide.

Contrast with scenario 3, where the same composites are inlined arithmetic on a carried
partial sum (progressive inference) and pushing them down wins.
"""
from rich.table import Table

from smart.harness import console, report, show_sql
from smart.rewriter import rewrite

from .common import candidates_table, demo_query, measure, save, threshold_for

ORDER = ("a", "b", "d", "l")


def run(conn, selectivity: float, runs: int, min_gain: float = 0.02):
    console.rule("[bold]Scenario 4 - placement (Technique 3): expensive composite predicates")
    T, est = threshold_for(conn, selectivity)
    q = demo_query(T)
    rw = rewrite(conn, q, mode="udf", min_gain=min_gain)
    console.print(f"ML predicate: predicted price < {T}   composites evaluated via lr_partial() "
                  f"(per-row catalog lookup, COST 500 - same as the final inference)")
    candidates_table(rw, ORDER)

    fenced = [v for v in rw.variants if v.placement and v.placement.order == ORDER]
    fenced.sort(key=lambda v: len(v.placement.composites), reverse=True)   # push-down first
    names = {
        frozenset({1, 2}): "push-down: P(a,b) at L1 and P(a,b,d) at L2  (textbook)",
        frozenset({1}): "P(a,b) at L1 only",
        frozenset({2}): "P(a,b) pulled up into P(a,b,d) at L2",
        frozenset(): "everything pulled up into the ML predicate (atomics only)",
    }
    by_cost = sorted(fenced, key=lambda v: v.est_cost)
    t = Table(title="Cost-model ranking (sample cardinalities x PostgreSQL unit costs)", header_style="bold")
    t.add_column("rank"); t.add_column("placement"); t.add_column("est. cost", justify="right")
    t.add_column("est. rows -> inference", justify="right"); t.add_column("EXPLAIN cost", justify="right")
    for i, v in enumerate(by_cost, 1):
        t.add_row(str(i), names[frozenset(v.placement.composites)], f"{v.est_cost:,.0f}",
                  f"{v.est_rows_to_inference:,.0f}", f"{v.pg_cost:,.0f}")
    console.print(t)
    push_down = fenced[0]
    show_sql(push_down.sql, "Push-down variant: partial inference at every level")

    base = measure(conn, "baseline", rw.baseline_sql, runs)
    ms = [base]
    for v in fenced:
        ms.append(measure(conn, names[frozenset(v.placement.composites)], v.sql, runs, rw.baseline_sql))
    report(ms, title="Scenario 4 - measured")
    fastest = min(ms[1:], key=lambda m: m.exec_ms)
    predicted = names[frozenset(by_cost[0].placement.composites)]
    agree = fastest.label == predicted
    console.print(f"  cost model predicted: [bold]{predicted}[/]\n  fastest measured:     [bold]{fastest.label}[/]  "
                  + ("[green](agree)[/]" if agree else "[yellow](disagree - say so on stage)[/]"))
    console.print("  Pushing an expensive, weakly selective predicate down costs more than it saves. "
                  "Progressive inference (scenario 3) is what makes composites cheap enough to push.")
    save("s4", {"threshold": T, "results": [{"label": m.label, "exec_ms": m.exec_ms,
                "rows_to_inference": m.rows_to_inference, "valid": m.valid} for m in ms],
                "predicted": predicted, "agree": agree})
    return ms
