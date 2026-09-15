"""Smart demo runner.

    python run_demo.py algorithm1                 # the paper's numbers, no database needed
    python run_demo.py s1 | s2 | s3 | s4           # scenarios (see scenarios/*.py)
    python run_demo.py sweep [--points ...]        # selectivity sweep -> results/sweep.csv + chart
    python run_demo.py all
    python run_demo.py break-it [--undo]           # stretch goal: stale statistics

Common options: --selectivity 0.01  --runs 3
"""
from __future__ import annotations

import argparse
import sys

from smart.harness import console


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["algorithm1", "s1", "s2", "s3", "s4", "sweep", "all", "break-it"])
    ap.add_argument("--selectivity", type=float, default=None,
                    help="fraction of joined rows the ML predicate keeps (default 0.01)")
    ap.add_argument("--runs", type=int, default=3, help="repetitions per measurement; the minimum is reported")
    ap.add_argument("--points", default="0.001,0.01,0.05,0.2,0.5,1.0", help="sweep selectivities")
    ap.add_argument("--sweep-runs", type=int, default=1)
    ap.add_argument("--undo", action="store_true")
    ap.add_argument("--no-chart", action="store_true")
    ap.add_argument("--fresh", action="store_true", help="ignore cached measurements and re-run everything")
    args = ap.parse_args(argv)
    if args.fresh:
        import os
        os.environ["SMART_FRESH"] = "1"

    if args.what == "algorithm1":
        from scenarios.algorithm1 import run
        run(); return

    from smart.catalog import connect
    with connect() as conn:
        sel = args.selectivity
        if args.what == "break-it":
            from scenarios.break_it import run
            run(conn, sel or 0.01, undo=args.undo); return
        if args.what in ("s1", "all"):
            from scenarios.s1_baseline import run
            run(conn, sel or 0.01, args.runs)
        if args.what in ("s2", "all"):
            from scenarios.s2_atomic import run
            run(conn, sel or 0.01, args.runs)
        if args.what in ("s3", "all"):
            from scenarios.s3_composite import run
            run(conn, sel or 0.01, args.runs)
        if args.what in ("s4", "all"):
            from scenarios.s4_placement import run
            run(conn, sel or 0.01, args.runs)
        if args.what in ("sweep", "all"):
            from scenarios.sweep import run
            run(conn, [float(x) for x in args.points.split(",")], args.sweep_runs)
            if not args.no_chart:
                from charts.selectivity import make_chart
                console.print(f"chart: {make_chart()}")


if __name__ == "__main__":
    sys.exit(main())
