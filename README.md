# Smart demo: in-database query optimisation for SQL with ML predicates

A Level B ("Python rewriter middleware") demo of the paper *In-database query optimization
on SQL with ML predicates*, built from `Smart_Demo_Build_Report.md`.

The claim it proves: you can read a trained model's weights out of a catalog table, derive
plain SQL `WHERE` conditions from them, and use those conditions to discard rows **before**
the joins, without changing the query's answer.

```
sys_model + sys_feature  ──►  derive.py (Thm 1-3, Alg 1)  ──►  select.py (Alg 2, sampled)
                              ──►  place.py (Alg 3, enumerated + costed)  ──►  emit.py (SQL)
                              ──►  harness.py: EXPLAIN ANALYZE both, assert identical results
```

## Quick start

```bash
docker compose up -d                       # PostgreSQL 16, 512 MB shared_buffers, no parallelism
python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
python -m scripts.setup_db                 # schema, 3M-row synthetic data, train, populate catalog (~1 min)
python -m pytest tests                     # 31 tests: the paper's numbers, sign flips, brute-force validity

python run_demo.py algorithm1              # whiteboard moment, no DB needed
python run_demo.py s1                      # baseline pathology
python run_demo.py s2                      # Technique 1: atomic predicates
python run_demo.py s3                      # Techniques 1+2: composites + progressive inference
python run_demo.py s4                      # Technique 3: placement - pull-up beats push-down
python run_demo.py sweep                   # selectivity sweep -> results/sweep.csv + charts/selectivity.png
python run_demo.py break-it [--undo]       # stretch goal: stale statistics break correctness
```

Connection string and data sizes live in `.env`. Every scenario prints the derived
predicates, the rewritten SQL, the plan, the timings and a green/red **validity check**
that fingerprints both result sets inside the database.

## What is faithful and what is approximated

| Paper component | Here | Fidelity |
|---|---|---|
| Theorem 1 + Algorithm 1 (atomic predicates) | `smart/derive.py: atomic_bounds`, `atomic_bounds_multi` | faithful, incl. negative-weight normalisation and seeding from existing predicates |
| Theorem 3 (composite predicates) | `smart/derive.py: composite` | faithful to the paper's worked example (uses `min` of excluded columns and the *tightened* minima) |
| Theorem 2 (decision trees) | `smart/derive.py: tree_intervals` | implemented and tested; not wired into a DB scenario |
| Logistic reduction | `smart/derive.py: logistic_to_linear`, model `pricey_logit` in the catalog | faithful |
| Technique 1 (push atomics to scans) | flat `WHERE` + PostgreSQL's own pushdown | faithful |
| Technique 2 + progressive inference | fenced subqueries (`OFFSET 0`) carrying `partial_sum` | faithful in effect, but done in SQL text rather than the plan tree |
| Algorithm 2 (composite selection) | `smart/select.py`: gain measured on a fixed 1 % sample of the joined space | approximated |
| Algorithm 3 / Technique 3 (placement) | `smart/place.py`: enumerate join orders x composite subsets, score with a cardinality x unit-cost model; PostgreSQL's EXPLAIN cost shown alongside | approximated (no DP over the plan tree) |
| Baseline | `lr_predict()` SQL UDF reading `sys_model` per row, `STABLE`, `COST 500` | the paper's faster (UDF) baseline |

Placement is effectively binary per composite: a composite sits either at the lowest join
where all its columns exist, or is pulled up into its super-predicate, where it is implied
by it and therefore merges away. Join order is enumerated as well, which is the other
degree of freedom the fences give us.

## Measured on the build machine (3M apartments, single run, WSL2, PostgreSQL 16 in Docker)

Threshold at ~1 % selectivity (predicted price < 16.92, 31,196 result rows):

| Variant | exec | rows fed to inference | valid |
|---|---:|---:|:-:|
| baseline: `lr_predict()` on top of the 4-way join | 30.5 s | 3,000,000 | |
| Smart, atomic predicates only (Technique 1) | 2.4 s | 175,059 | OK |
| Smart, atomic + inlined composites with carried `partial_sum` (Techniques 1+2) | 1.3 s | 59,998 | OK |

Scenario 4, same threshold, composites evaluated through the inference UDF:

| Placement | cost-model rank | measured |
|---|:-:|---:|
| everything pulled up into the ML predicate | 1 | 2.3 s |
| P(a,b) merged into P(a,b,d) at L2 | 2 | 3.0 s |
| P(a,b) at L1 only | 3 | 5.2 s |
| textbook push-down: P(a,b) at L1, P(a,b,d) at L2 | 4 | 5.6 s |

The cost model's ranking matches the measured ranking exactly. PostgreSQL's own EXPLAIN
cost ranks these four in the opposite order, because it assumes a fixed 1/3 selectivity
for any composite expression: a concrete reason the paper needs its own cost model.

Selectivity sweep (`charts/selectivity.png`):

| selectivity | baseline | Smart | speed-up | rows fed to inference |
|---:|---:|---:|---:|---:|
| 0.11 % | 30.0 s | 0.38 s | 79x | 5,716 |
| 1.04 % | 30.5 s | 1.27 s | 24x | 59,998 |
| 5.08 % | 29.5 s | 3.67 s | 8.0x | 243,326 |
| 19.9 % | 29.3 s | 10.0 s | 2.9x | 841,365 |
| 50.3 % | 30.9 s | 20.0 s | 1.5x | 1,862,649 |
| 100 % | 32.8 s | 32.8 s | 1.0x | 3,000,000 |

At 100 % no derived predicate tightens anything, so the rewriter emits the original query
unchanged: the curves converge, they never cross.

The lesson across scenarios 3 and 4: a composite predicate is worth pushing down only when
it is cheap, and progressive inference (reusing the partial sum) is what makes it cheap.

## Two things the build report got wrong, found by implementing

1. **Algorithm 1 converges in one pass for a single linear predicate.** The report expected
   two or three passes on the paper's numbers. For one constraint the first Gauss-Seidel
   sweep already yields the exact box hull; pass 2 only confirms. The iteration genuinely
   tightens when several predicates constrain shared columns, so `algorithm1` shows both
   cases (`atomic_bounds_multi`).
2. **A composite cannot usefully be "placed higher" within the same join chain.** At its
   super-predicate's level it is implied by the super-predicate. The placement decision is
   push-down vs. merge-away, plus join order.

Both points are talking points, not defects: they show the implementation was checked
against the maths rather than paraphrased.

## Layout

```
docker-compose.yml        PostgreSQL 16 with the report's settings
sql/                      schema, data generator, catalog tables, baseline UDFs
smart/                    catalog.py derive.py select.py place.py emit.py harness.py rewriter.py query.py
scenarios/                s1_baseline s2_atomic s3_composite s4_placement sweep algorithm1 break_it
tests/test_derive.py      hand-calculated bounds, sign flips, brute-force validity, tree paths
charts/selectivity.py     the closing chart
results/                  JSON/CSV written by each run
```

## Pitfalls handled

- **Stale statistics**: `check_bounds()` recomputes MIN/MAX before every rewrite and
  refuses to proceed if `sys_feature` is stale. `break-it` demonstrates the failure.
- **Sign errors**: normalise/denormalise round trip and a 20-seed brute-force test.
- **Numeric vs float**: bounds are widened by an epsilon before emission (widening keeps
  a necessary condition necessary); integer columns get integer bounds.
- **Cache warming**: minimum of N runs for both sides, as the paper does.
- **Plan flipping**: the placement table prints PostgreSQL's cost next to ours, so a
  disagreement is visible rather than hidden.
