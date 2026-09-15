"""Predicate derivation. Pure functions, no database access.

Implements, for the paper *In-database query optimization on SQL with ML predicates*:

* Theorem 1 / Algorithm 1  -- atomic predicates for linear models, iterated to a fixpoint
* Theorem 3                -- composite predicates over a subset of feature columns
* Theorem 2                -- decision trees: per-feature disjunction of path intervals
* the logistic -> linear reduction  (sigmoid(z) >= 0.5  <=>  z >= 0)

Conventions
-----------
An ML predicate is  l_low <= f(x) <= l_up  (either side may be None).
Every derived predicate is a *necessary* condition: a row that satisfies the ML predicate
always satisfies the derived one. The reverse need not hold, which is why the original
ML predicate is kept at the top of the rewritten query.

Negative weights are handled by normalising:  (w, x, lo, hi) -> (-w, -x, -hi, -lo).
The bounds derived in normalised space are mapped back with `denormalise`.
A sign error here does not raise; it silently drops rows. See tests/test_derive.py.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable, Sequence

INF = float("inf")


@dataclass(frozen=True)
class Feature:
    """One model input: a column of a table, its weight, and its catalog min/max."""
    table: str
    col: str
    w: float
    lo: float
    hi: float
    alias: str | None = None     # query alias for the table, filled in by the rewriter
    flipped: bool = False        # True after normalisation of a negative weight
    dtype: str = "numeric"       # SQL data type; integer columns get integer bounds

    @property
    def is_integer(self) -> bool:
        return self.dtype in ("integer", "bigint", "smallint")

    @property
    def key(self) -> str:
        return f"{self.table}.{self.col}"

    @property
    def qualified(self) -> str:
        return f"{self.alias or self.table}.{self.col}"


@dataclass(frozen=True)
class Bound:
    """Derived interval on an ORIGINAL (un-normalised) column:  lo <= col <= hi."""
    feature: Feature
    lo: float
    hi: float

    @property
    def is_trivial(self) -> bool:
        """True when the bound does not tighten the catalog range at all."""
        f = self.feature
        return self.lo <= f.lo and self.hi >= f.hi


# --------------------------------------------------------------------------------------
# Normalisation of negative weights
# --------------------------------------------------------------------------------------

def normalise(feats: Sequence[Feature]) -> list[Feature]:
    """[w, x, x_min, x_max] <- (-1) * [w, x, x_max, x_min] for w < 0."""
    out = []
    for f in feats:
        if f.w < 0:
            out.append(replace(f, w=-f.w, lo=-f.hi, hi=-f.lo, flipped=True))
        else:
            out.append(f)
    return out


def denormalise(feats_norm: Sequence[Feature], lo: dict, hi: dict) -> list[Bound]:
    """Map bounds in normalised space back to the original columns."""
    out = []
    for f in feats_norm:
        l, h = lo[f.key], hi[f.key]
        orig = replace(f, w=-f.w, lo=-f.hi, hi=-f.lo, flipped=False) if f.flipped else f
        if f.flipped:
            # -x in [l, h]  <=>  x in [-h, -l]
            out.append(Bound(orig, -h, -l))
        else:
            out.append(Bound(orig, l, h))
    return out


# --------------------------------------------------------------------------------------
# Theorem 1 + Algorithm 1: atomic predicates for a linear model
# --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class LinearConstraint:
    """l_low <= w0 + sum_i w_i * x_i <= l_up over a subset of the query's feature columns."""
    feats: tuple[Feature, ...]
    w0: float
    l_up: float | None
    l_low: float | None = None


def atomic_bounds(
    feats: Sequence[Feature],
    w0: float,
    l_up: float | None,
    l_low: float | None = None,
    seed_lo: dict | None = None,
    seed_hi: dict | None = None,
    max_iter: int = 10,
    eps: float = 1e-9,
) -> tuple[list[Bound], list[dict]]:
    """Derive per-column bounds for  l_low <= w0 + sum(w_i x_i) <= l_up.

    Theorem 1:  x_i <= (l_up  - w0 - sum_{j!=i} w_j * min(x_j)) / w_i
                x_i >= (l_low - w0 - sum_{j!=i} w_j * max(x_j)) / w_i
    Algorithm 1: repeat with the tightened bounds of the other columns until nothing moves.

    `seed_lo` / `seed_hi` let existing query predicates (e.g. rating_score > 4) replace the
    catalog min/max as the starting point. Keys are "table.col" of the ORIGINAL feature.

    Returns (bounds in original space, per-pass trace for display).
    Raises ValueError if the predicate is infeasible (some interval becomes empty).

    Note: for a *single* linear predicate one Gauss-Seidel sweep already yields the exact
    box hull, so pass 2 only confirms the fixpoint. Iteration genuinely tightens when
    several predicates share columns -- see `atomic_bounds_multi`.
    """
    if l_up is None and l_low is None:
        raise ValueError("need at least one of l_up / l_low")
    return atomic_bounds_multi(
        [LinearConstraint(tuple(feats), w0, l_up, l_low)],
        seed_lo=seed_lo, seed_hi=seed_hi, max_iter=max_iter, eps=eps,
    )


def atomic_bounds_multi(
    constraints: Sequence[LinearConstraint],
    seed_lo: dict | None = None,
    seed_hi: dict | None = None,
    max_iter: int = 10,
    eps: float = 1e-9,
) -> tuple[list[Bound], list[dict]]:
    """Algorithm 1 over several linear predicates at once (e.g. two models in one WHERE).

    Each constraint is normalised separately (a column may have a positive weight in one
    model and a negative one in another), while the shared bounds are kept in ORIGINAL
    column space and translated on the fly.
    """
    seed_lo, seed_hi = seed_lo or {}, seed_hi or {}
    # union of original features, first occurrence wins for catalog range
    originals: dict[str, Feature] = {}
    for c in constraints:
        for f in c.feats:
            originals.setdefault(f.key, f)

    lo = {k: max(f.lo, seed_lo.get(k, -INF)) for k, f in originals.items()}
    hi = {k: min(f.hi, seed_hi.get(k, INF)) for k, f in originals.items()}
    trace: list[dict] = [{"pass": 0, "lo": dict(lo), "hi": dict(hi), "changed": None}]

    normed = [(c, normalise(c.feats)) for c in constraints]

    for it in range(1, max_iter + 1):
        changed = 0.0
        for c, norm in normed:
            def n_lo(g: Feature) -> float:   # bound of the normalised column
                return -hi[g.key] if g.flipped else lo[g.key]

            def n_hi(g: Feature) -> float:
                return -lo[g.key] if g.flipped else hi[g.key]

            for f in norm:
                if f.w == 0:
                    continue  # a zero weight constrains nothing
                others_min = sum(g.w * n_lo(g) for g in norm if g.key != f.key)
                others_max = sum(g.w * n_hi(g) for g in norm if g.key != f.key)
                new_hi, new_lo = n_hi(f), n_lo(f)
                if c.l_up is not None:
                    new_hi = min(new_hi, (c.l_up - c.w0 - others_min) / f.w)
                if c.l_low is not None:
                    new_lo = max(new_lo, (c.l_low - c.w0 - others_max) / f.w)
                if new_lo > new_hi + eps:
                    raise ValueError(
                        f"ML predicate is infeasible: derived {f.key} in [{new_lo}, {new_hi}]"
                    )
                # back to original space
                o_lo, o_hi = (-new_hi, -new_lo) if f.flipped else (new_lo, new_hi)
                changed = max(changed, abs(o_hi - hi[f.key]), abs(o_lo - lo[f.key]))
                lo[f.key], hi[f.key] = o_lo, o_hi
        trace.append({"pass": it, "lo": dict(lo), "hi": dict(hi), "changed": changed})
        if changed < eps:
            break

    return [Bound(f, lo[k], hi[k]) for k, f in originals.items()], trace


# --------------------------------------------------------------------------------------
# Theorem 3: composite predicates over a subset I of the features
# --------------------------------------------------------------------------------------

@dataclass(frozen=True)
class Composite:
    """sum_{i in I} w_i * x_i  in [rhs_lo, rhs_hi]   (in ORIGINAL column space)."""
    features: tuple[Feature, ...]     # original (un-normalised) features in I
    rhs_lo: float | None
    rhs_hi: float | None

    @property
    def keys(self) -> frozenset:
        return frozenset(f.key for f in self.features)


def composite(
    feats: Sequence[Feature],
    w0: float,
    l_up: float | None,
    l_low: float | None,
    subset: Iterable[str],
    bounds: Sequence[Bound],
) -> Composite:
    """Theorem 3, implemented as the paper's worked example does it:

        sum_{i in I} w_i x_i  <  l_up - w0 - sum_{d not in I} w_d * min(x_d)

    using the *tightened* minima from the atomic pass (that is the compounding effect).
    Note: the theorem as printed subtracts w_d * x_d,up for excluded columns, which would
    not be a necessary condition; the paper's own example (... < 14 - 1*min(C.c) -
    0.5*min(D.d) = 10) uses the minimum, so that is what we implement.

    Because the arithmetic is done in normalised space, negative weights come out right:
    for w_d < 0 the "minimum" of the normalised column is -max(x_d), i.e. the excluded
    term becomes  - w_d * max(x_d)  in original space, which is the tightest valid choice.
    """
    subset = set(subset)
    norm = normalise(feats)
    by_key = {b.feature.key: b for b in bounds}

    def norm_lo(f: Feature) -> float:
        b = by_key[f.key]
        return -b.hi if f.flipped else b.lo

    def norm_hi(f: Feature) -> float:
        b = by_key[f.key]
        return -b.lo if f.flipped else b.hi

    excluded = [f for f in norm if f.key not in subset]
    rhs_hi = rhs_lo = None
    if l_up is not None:
        rhs_hi = l_up - w0 - sum(f.w * norm_lo(f) for f in excluded)
    if l_low is not None:
        rhs_lo = l_low - w0 - sum(f.w * norm_hi(f) for f in excluded)
    # In original space  sum_{i in I} w_i x_i  is exactly the normalised sum, so the RHS
    # values carry over unchanged; we just report the original features.
    included = tuple(f for f in feats if f.key in subset)
    return Composite(included, rhs_lo, rhs_hi)


# --------------------------------------------------------------------------------------
# Logistic regression -> linear
# --------------------------------------------------------------------------------------

def logistic_to_linear(target: bool) -> tuple[float | None, float | None]:
    """S(z) >= 0.5  <=>  z >= 0.  Returns (l_low, l_up) for the linear part."""
    return (0.0, None) if target else (None, 0.0)


# --------------------------------------------------------------------------------------
# Theorem 2: decision trees
# --------------------------------------------------------------------------------------

def tree_intervals(tree, feature_keys: Sequence[str], target_class: int) -> dict[str, list[tuple[float, float]]]:
    """Walk an sklearn `tree_` structure and return, per feature, the disjunction of
    intervals along all root-to-leaf paths that predict `target_class`.

    A feature that is not tested on *every* qualifying path yields no predicate (its
    interval on that path is (-inf, inf), so the disjunction is the whole line).
    Intervals are (lo, hi] style: x <= hi on the "left" branch, x > lo on the "right".
    """
    t = tree.tree_ if hasattr(tree, "tree_") else tree
    n_feat = len(feature_keys)
    paths: list[list[tuple[float, float]]] = []

    def walk(node: int, lo: list[float], hi: list[float]) -> None:
        left, right = t.children_left[node], t.children_right[node]
        if left == -1:  # leaf
            if int(t.value[node][0].argmax()) == target_class:
                paths.append(list(zip(lo, hi)))
            return
        f, thr = int(t.feature[node]), float(t.threshold[node])
        # left: x <= thr
        hi2 = hi.copy(); hi2[f] = min(hi2[f], thr)
        walk(left, lo, hi2)
        # right: x > thr
        lo2 = lo.copy(); lo2[f] = max(lo2[f], thr)
        walk(right, lo2, hi)

    walk(0, [-INF] * n_feat, [INF] * n_feat)

    out: dict[str, list[tuple[float, float]]] = {}
    for i, key in enumerate(feature_keys):
        ivs = sorted({p[i] for p in paths})
        if any(lo == -INF and hi == INF for lo, hi in ivs) or not ivs:
            continue  # no constraint on this feature
        out[key] = _merge_intervals(ivs)
    return out


def _merge_intervals(ivs: list[tuple[float, float]]) -> list[tuple[float, float]]:
    merged: list[tuple[float, float]] = []
    for lo, hi in ivs:
        if merged and lo <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    return merged
