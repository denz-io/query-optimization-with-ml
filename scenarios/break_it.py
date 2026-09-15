"""Stretch goal: break it deliberately.

Insert a building whose age is below the stored minimum, and an apartment in it whose
predicted price is under the threshold even though its room count fails the derived
atomic bound. The bound was computed from a minimum that is no longer true, so it is no
longer a necessary condition: the row silently disappears from the Smart result.

    python run_demo.py break-it            # insert, show check_bounds() catching it, show the violation
    python run_demo.py break-it --undo     # remove the rows again
"""
import math
import os

from smart.catalog import load_model
from smart.harness import console, validate
from smart.rewriter import StaleStatisticsError, rewrite

from .common import demo_query, threshold_for

OUTLIER_BID = 999_999_999
OUTLIER_AID = 999_999_999_999


def run(conn, selectivity: float, undo: bool = False):
    console.rule("[bold]Break it deliberately - stale sys_feature statistics")
    os.environ["SMART_FRESH"] = "1"      # this scenario mutates the data: never trust cached fingerprints
    if undo:
        conn.execute("DELETE FROM apartment WHERE aid = %s", (OUTLIER_AID,))
        conn.execute("DELETE FROM building WHERE bid = %s", (OUTLIER_BID,))
        console.print("[green]outlier rows removed; statistics are consistent again[/]")
        return
    T, est = threshold_for(conn, selectivity)
    q = demo_query(T)
    model = load_model(conn, "price_lr")
    w = {f.col: f.w for f in model.features}
    # Attach the outlier to the cheapest district and landlord so only building_age has to move.
    did, h = conn.execute("SELECT did, hospital_num FROM district ORDER BY hospital_num, did LIMIT 1").fetchone()
    lid, s_ = conn.execute("SELECT lid, rating_score FROM landlord ORDER BY rating_score, lid LIMIT 1").fetchone()
    rw_clean = rewrite(conn, q, verify_statistics=False)
    room_hi = next(b.hi for b in rw_clean.bounds if b.feature.col == "room_num")
    room = int(room_hi) + 1                  # fails the derived atomic bound room_num <= floor(hi)
    # solve for a building_age that still puts the prediction under T (it will be < stored min 2)
    age = int(math.floor((T - 0.05 - model.intercept - w["room_num"] * room
                          - w["hospital_num"] * float(h) - w["rating_score"] * float(s_)) / w["building_age"]))
    conn.execute("INSERT INTO building VALUES (%s, %s, %s, 'north') ON CONFLICT DO NOTHING", (OUTLIER_BID, did, age))
    conn.execute("INSERT INTO apartment VALUES (%s, %s, %s, %s, 0) ON CONFLICT DO NOTHING",
                 (OUTLIER_AID, OUTLIER_BID, lid, room))
    pred = conn.execute(
        "SELECT lr_predict('price_lr', ARRAY[%s, %s, %s, %s]::numeric[])", (room, age, h, s_)).fetchone()[0]
    console.print(f"inserted building {OUTLIER_BID} with building_age = {age} (stored min 2) and an apartment in it "
                  f"with room_num = {room} (derived bound: room_num <= {int(room_hi)}).\n"
                  f"Its predicted price is {float(pred):.2f} < {T}: it belongs in the result, but the atomic "
                  f"predicate on room_num - derived assuming building_age >= 2 - rejects it.")

    console.print("\n[bold]1. With the guard on:[/] check_bounds() recomputes MIN/MAX and refuses to rewrite")
    try:
        rewrite(conn, q)
        console.print("[red]guard did not fire[/]")
    except StaleStatisticsError as e:
        console.print(f"  [bold white on red] REFUSED [/] {e}")

    console.print("\n[bold]2. With the guard bypassed:[/] the rewrite runs and the validity check catches the missing row")
    rw = rewrite(conn, q, verify_statistics=False)
    validate(conn, rw.baseline_sql, rw.smart_sql)
    console.print("\n  Run with --undo to remove the outlier rows.")
