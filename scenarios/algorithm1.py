"""The whiteboard moment: derive the paper's bounds by hand, then let Algorithm 1 run."""
from rich.table import Table

from smart.derive import Feature, LinearConstraint, atomic_bounds, atomic_bounds_multi, composite
from smart.harness import console

PAPER = [
    Feature("A", "a", 3.0, lo=1.0, hi=100.0),
    Feature("B", "b", 2.0, lo=2.0, hi=100.0),
    Feature("C", "c", 1.0, lo=3.0, hi=100.0),
    Feature("D", "d", 0.5, lo=2.0, hi=100.0),
]


def _trace_table(trace, title):
    t = Table(title=title, header_style="bold")
    t.add_column("pass")
    keys = list(trace[0]["lo"])
    for k in keys:
        t.add_column(k, justify="right")
    t.add_column("max change", justify="right")
    for p in trace:
        cells = []
        for k in keys:
            lo, hi = p["lo"][k], p["hi"][k]
            cells.append(f"[{lo:.4g}, {hi:.4g}]")
        t.add_row(str(p["pass"]), *cells, "" if p["changed"] is None else f"{p['changed']:.4g}")
    console.print(t)


def run():
    console.rule("[bold]Algorithm 1 on the paper's own numbers")
    console.print("f(x) = 3*A.a + 2*B.b + 1*C.c + 0.5*D.d < 14,   min(B.b)=2, min(C.c)=3, min(D.d)=2")
    console.print("By hand:  A.a < (14 - 2*2 - 1*3 - 0.5*2) / 3 = 6/3 = [bold]2[/]   "
                  "(the paper's prose says 4; its Figure 2 and the arithmetic say 2)")
    bounds, trace = atomic_bounds(PAPER, 0.0, l_up=14.0)
    _trace_table(trace, "Algorithm 1 trace - one linear predicate")
    console.print("  A single linear predicate is box-consistent after one sweep: pass 2 confirms the fixpoint.\n")
    c = composite(PAPER, 0.0, 14.0, None, {"A.a", "B.b"}, bounds)
    console.print(f"Composite (Theorem 3, as the paper's example computes it):  3*A.a + 2*B.b < "
                  f"14 - 1*min(C.c) - 0.5*min(D.d) = [bold]{c.rhs_hi:g}[/]\n")

    console.rule("[bold]When the iteration actually matters: several predicates on shared columns")
    a = Feature("A", "a", 1.0, lo=0.0, hi=100.0)
    b = Feature("B", "b", 1.0, lo=0.0, hi=100.0)
    cc = Feature("C", "c", 1.0, lo=0.0, hi=100.0)
    neg_c = Feature("C", "c", -1.0, lo=0.0, hi=100.0)
    console.print("a + b <= 10      b - c >= 3      a + c >= 6     (three models / predicates in one WHERE)")
    _, trace2 = atomic_bounds_multi([
        LinearConstraint((a, b), 0.0, l_up=10.0),
        LinearConstraint((b, neg_c), 0.0, l_up=None, l_low=3.0),
        LinearConstraint((a, cc), 0.0, l_up=None, l_low=6.0),
    ])
    _trace_table(trace2, "Algorithm 1 trace - bounds tighten across passes")
