"""Shared pieces for the four scenarios: the demo query, threshold selection, reporting."""
from __future__ import annotations

import json
import os
from pathlib import Path

from rich.table import Table

from smart.catalog import connect, load_model
from smart.harness import Measurement, timed, validate
from smart.query import JoinTable, MLPredicate, Query
from smart.select import SampleEstimator

from smart.harness import console
RESULTS = Path(__file__).resolve().parent.parent / "results"
RESULTS.mkdir(exist_ok=True)

TABLES = (
    JoinTable("apartment", "a"),
    JoinTable("building", "b", "a.bid = b.bid"),
    JoinTable("district", "d", "b.did = d.did"),
    JoinTable("landlord", "l", "a.lid = l.lid"),
)


def demo_query(threshold: float, where: tuple[str, ...] = ()) -> Query:
    """The paper's running example: apartments whose predicted price is below a threshold."""
    return Query(select=("a.aid", "a.price"), tables=TABLES,
                 ml=MLPredicate("price_lr", "<", threshold), where=where)


def threshold_for(conn, selectivity: float) -> tuple[float, SampleEstimator]:
    """Pick the ML threshold from the sample so the predicate keeps ~`selectivity` of rows."""
    model = load_model(conn, "price_lr")
    est = SampleEstimator(conn, model)
    if selectivity >= 1.0:
        return float(round(est.predictions().max() + 1, 2)), est
    return float(round(est.threshold_for_selectivity(selectivity), 2)), est


# Measurements are cached on disk, keyed by the SQL text, run count and the row count of the
# fact table, so a 35 s baseline is not re-run for every scenario or rehearsal. Cached
# numbers are labelled as such; pass --fresh (FRESH=1) to ignore the cache.
_CACHE_FILE = RESULTS / "measurement_cache.json"
_FRESH = os.environ.get("SMART_FRESH") == "1"


def _cache_load() -> dict:
    try:
        return json.loads(_CACHE_FILE.read_text())
    except (OSError, ValueError):
        return {}


def _cache_key(conn, sql: str, runs: int) -> str:
    n = conn.execute("SELECT count(*) FROM apartment").fetchone()[0]
    return f"{runs}|{n}|{sql}"


def measure(conn, label: str, sql: str, runs: int, baseline_sql: str | None = None,
            key: str = "aid") -> Measurement:
    """timed() with a cache and the validity assertion against the baseline."""
    cache = _cache_load()
    ck = _cache_key(conn, sql, runs)
    if ck in cache and not _FRESH:
        d = cache[ck]
        m = Measurement(label, sql, d["exec_ms"], d["plan_ms"], d["rows_to_inference"],
                        d["result_rows"], d["shared_hit"], d["shared_read"], d["runs"])
        console.print(f"  [dim]{label}: cached measurement ({m.exec_ms:,.0f} ms; --fresh to re-run)[/]")
    else:
        console.print(f"  running [bold]{label}[/] x{runs} ...", end="")
        m = timed(conn, label, sql, runs)
        console.print(f" {m.exec_ms:,.0f} ms  (runs: {', '.join(f'{t:,.0f}' for t in m.runs)})")
        cache[ck] = {k: getattr(m, k) for k in ("exec_ms", "plan_ms", "rows_to_inference",
                                                 "result_rows", "shared_hit", "shared_read", "runs")}
        _CACHE_FILE.write_text(json.dumps(cache))
        m = Measurement(**{**m.__dict__, "label": label})
    if baseline_sql is not None:
        m.valid = validate(conn, baseline_sql, sql, key)
    return m


def save(name: str, payload) -> None:
    (RESULTS / f"{name}.json").write_text(json.dumps(payload, indent=2, default=str))


def variants_table(rw, title="Placement search: every enumerated variant") -> None:
    t = Table(title=title, header_style="bold")
    t.add_column("#", justify="right"); t.add_column("variant")
    t.add_column("est. cost (sample x unit costs)", justify="right")
    t.add_column("est. rows -> inference", justify="right")
    t.add_column("PostgreSQL EXPLAIN cost", justify="right"); t.add_column("")
    for i, v in enumerate(rw.variants):
        mark = "[green]<- chosen[/]" if v is rw.chosen else ""
        t.add_row(str(i), v.label, f"{v.est_cost:,.0f}", f"{v.est_rows_to_inference:,.0f}",
                  f"{v.pg_cost:,.0f}", mark)
    console.print(t)


def candidates_table(rw, order) -> None:
    cands = next((v.candidates for v in rw.variants if v.placement and v.placement.order == order), [])
    t = Table(title=f"Composite candidates along {'>'.join(order)} (Theorem 3 + selection)",
              header_style="bold")
    t.add_column("level"); t.add_column("predicate"); t.add_column("rows before", justify="right")
    t.add_column("rows after", justify="right"); t.add_column("extra pruning", justify="right")
    t.add_column("selected")
    for c in cands:
        terms = " + ".join(f"{f.w:.3g}*{f.col}" for f in c.composite.features)
        t.add_row(f"L{c.level}", f"{terms} <= {c.composite.rhs_hi:.4g}",
                  f"{c.rows_before:,.0f}", f"{c.rows_after:,.0f}", f"{c.gain:.1%}",
                  "[green]yes[/]" if c.selected else "[dim]no[/]")
    console.print(t)
