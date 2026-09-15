"""Hand-calculated bounds from the paper, sign flips, and a brute-force validity check.

Everything downstream depends on these being right. A sign error will not announce
itself; it will just quietly return the wrong rows.
"""
import math
import random

import numpy as np
import pytest

from smart.derive import (
    Feature, atomic_bounds, composite, logistic_to_linear, normalise, denormalise,
    tree_intervals,
)

# The paper's example: weights 3, 2, 1, 0.5 on A.a, B.b, C.c, D.d; intercept 0;
# predicate  f(x) < 14;  minimums  A.a=1 (unused for its own bound), B.b=2, C.c=3, D.d=2.
PAPER = [
    Feature("A", "a", 3.0, lo=1.0, hi=100.0),
    Feature("B", "b", 2.0, lo=2.0, hi=100.0),
    Feature("C", "c", 1.0, lo=3.0, hi=100.0),
    Feature("D", "d", 0.5, lo=2.0, hi=100.0),
]


def _by_key(bounds):
    return {b.feature.key: b for b in bounds}


def test_paper_atomic_bound_on_first_column_is_2():
    bounds, trace = atomic_bounds(PAPER, w0=0.0, l_up=14.0)
    b = _by_key(bounds)
    # a < (14 - 2*2 - 1*3 - 0.5*2) / 3 = 6/3 = 2   (the paper's prose says 4; Figure 2 and the
    # arithmetic both give 2 -- see the build report, section 8.3)
    assert b["A.a"].hi == pytest.approx(2.0)
    assert b["B.b"].hi == pytest.approx((14 - 3 * 1 - 3 - 1) / 2)     # 3.5
    assert b["C.c"].hi == pytest.approx(14 - 3 - 4 - 1)               # 6
    assert b["D.d"].hi == pytest.approx((14 - 3 - 4 - 3) / 0.5)       # 8
    # lower bounds untouched (no l_low)
    assert b["A.a"].lo == 1.0
    # one-sided predicate converges in a single pass
    assert trace[-1]["changed"] < 1e-9 and len(trace) <= 3


def test_paper_composite_rhs_is_10():
    bounds, _ = atomic_bounds(PAPER, w0=0.0, l_up=14.0)
    c = composite(PAPER, 0.0, 14.0, None, {"A.a", "B.b"}, bounds)
    # 3*A.a + 2*B.b < 14 - 1*min(C.c) - 0.5*min(D.d) = 14 - 3 - 1 = 10
    assert c.rhs_hi == pytest.approx(10.0)
    assert c.rhs_lo is None
    assert {f.key for f in c.features} == {"A.a", "B.b"}


def test_composite_uses_tightened_minima_when_two_sided():
    # With a lower bound, the atomic pass raises minima; composite must use the raised ones.
    feats = [
        Feature("A", "a", 1.0, lo=0.0, hi=10.0),
        Feature("B", "b", 1.0, lo=0.0, hi=10.0),
        Feature("C", "c", 1.0, lo=0.0, hi=10.0),
    ]
    bounds, trace = atomic_bounds(feats, 0.0, l_up=30.0, l_low=25.0)
    b = _by_key(bounds)
    # a >= 25 - 10 - 10 = 5
    assert b["A.a"].lo == pytest.approx(5.0)
    c = composite(feats, 0.0, 30.0, 25.0, {"A.a", "B.b"}, bounds)
    # a + b < 30 - min(c) = 30 - 5 = 25   (raw catalog min would give 30, looser)
    assert c.rhs_hi == pytest.approx(25.0)
    # a + b > 25 - max(c) = 15
    assert c.rhs_lo == pytest.approx(15.0)


def test_single_linear_predicate_converges_in_one_pass():
    # For ONE linear predicate the first Gauss-Seidel sweep is already the exact box hull;
    # pass 2 only confirms the fixpoint. (Iteration matters with several predicates.)
    feats = [
        Feature("A", "a", 1.0, lo=0.0, hi=100.0),
        Feature("B", "b", 1.0, lo=0.0, hi=100.0),
    ]
    _, trace = atomic_bounds(feats, 0.0, l_up=10.0, l_low=9.0)
    assert trace[1]["hi"]["A.a"] == pytest.approx(10.0)
    assert trace[1]["lo"]["A.a"] == 0.0            # clamped to the catalog minimum
    assert trace[2]["changed"] == 0.0 and len(trace) == 3


def test_several_predicates_on_shared_columns_tighten_over_passes():
    from smart.derive import LinearConstraint, atomic_bounds_multi
    a = Feature("A", "a", 1.0, lo=0.0, hi=100.0)
    b = Feature("B", "b", 1.0, lo=0.0, hi=100.0)
    c = Feature("C", "c", 1.0, lo=0.0, hi=100.0)
    neg_c = Feature("C", "c", -1.0, lo=0.0, hi=100.0)
    # a + b <= 10,   b - c >= 3,   a + c >= 6   -- a cycle: hi_b -> hi_c -> lo_a -> hi_b ...
    cons = [
        LinearConstraint((a, b), 0.0, l_up=10.0),
        LinearConstraint((b, neg_c), 0.0, l_up=None, l_low=3.0),
        LinearConstraint((a, c), 0.0, l_up=None, l_low=6.0),
    ]
    bounds, trace = atomic_bounds_multi(cons)
    by = {x.feature.key: x for x in bounds}
    assert trace[1]["hi"]["A.a"] == pytest.approx(10.0)   # after pass 1
    assert trace[2]["hi"]["A.a"] == pytest.approx(7.0)    # pass 2 tightens via lo_b = 3
    assert by["A.a"].hi == pytest.approx(7.0) and by["B.b"].lo == pytest.approx(3.0)
    assert by["C.c"].hi == pytest.approx(7.0)
    assert len(trace) == 4  # pass 0 seed, pass 1, pass 2, pass 3 confirms


def test_negative_weight_round_trip():
    f = Feature("T", "x", -2.0, lo=-5.0, hi=7.0)
    (n,) = normalise([f])
    assert (n.w, n.lo, n.hi, n.flipped) == (2.0, -7.0, 5.0, True)
    (back,) = denormalise([n], {n.key: -7.0}, {n.key: 5.0})
    assert (back.lo, back.hi) == (-5.0, 7.0)
    assert back.feature.w == -2.0 and not back.feature.flipped


def test_negative_weight_gives_lower_bound_direction():
    # f = 1*a - 1*b < 2, a in [0,10], b in [0,10]
    # b > a - 2 >= -2   => trivial;   a < 2 + b <= 12 => trivial ... use tighter numbers:
    feats = [Feature("A", "a", 1.0, lo=0.0, hi=10.0), Feature("B", "b", -1.0, lo=0.0, hi=10.0)]
    bounds, _ = atomic_bounds(feats, 0.0, l_up=-5.0)
    b = _by_key(bounds)
    # a - b < -5  => a < b - 5 <= 5  and  b > a + 5 >= 5
    assert b["A.a"].hi == pytest.approx(5.0)
    assert b["B.b"].lo == pytest.approx(5.0)
    assert b["B.b"].hi == 10.0


def test_seed_from_existing_predicate_tightens():
    bounds_plain, _ = atomic_bounds(PAPER, 0.0, 14.0)
    bounds_seeded, _ = atomic_bounds(PAPER, 0.0, 14.0, seed_lo={"D.d": 4.0})
    # min(D.d) rises 2 -> 4, so every other upper bound drops by 0.5*2/w
    assert _by_key(bounds_seeded)["A.a"].hi == pytest.approx(_by_key(bounds_plain)["A.a"].hi - 1.0 / 3)


def test_infeasible_predicate_raises():
    with pytest.raises(ValueError):
        atomic_bounds(PAPER, 0.0, l_up=5.0)  # minimum possible f = 3+4+3+1 = 11 > 5


def test_logistic_reduction():
    assert logistic_to_linear(True) == (0.0, None)
    assert logistic_to_linear(False) == (None, 0.0)


@pytest.mark.parametrize("seed", range(20))
def test_derived_predicates_never_drop_a_qualifying_row(seed):
    """Brute force: random model, random data inside the catalog ranges, random threshold.
    Every row satisfying the ML predicate must satisfy every atomic AND composite predicate."""
    rng = np.random.default_rng(seed)
    k = int(rng.integers(2, 5))
    feats = []
    X = []
    for i in range(k):
        lo = float(rng.uniform(-10, 0)); hi = lo + float(rng.uniform(1, 20))
        w = float(rng.uniform(-3, 3)) or 1.0
        feats.append(Feature("T", f"x{i}", w, lo, hi))
        X.append(rng.uniform(lo, hi, size=5000))
    X = np.stack(X, axis=1)
    w0 = float(rng.uniform(-5, 5))
    W = np.array([f.w for f in feats])
    y = w0 + X @ W
    l_low, l_up = np.quantile(y, [0.2, 0.6])
    two_sided = seed % 2 == 0
    bounds, _ = atomic_bounds(feats, w0, l_up=float(l_up), l_low=float(l_low) if two_sided else None)
    ok = (y <= l_up) & ((y >= l_low) if two_sided else True)
    for j, b in enumerate(bounds):
        assert np.all(X[ok, j] <= b.hi + 1e-9), f"atomic hi on {b.feature.key} drops rows"
        assert np.all(X[ok, j] >= b.lo - 1e-9), f"atomic lo on {b.feature.key} drops rows"
    # every 2-subset composite
    for i in range(k):
        for j in range(i + 1, k):
            c = composite(feats, w0, float(l_up), float(l_low) if two_sided else None,
                          {feats[i].key, feats[j].key}, bounds)
            s = X[ok, i] * feats[i].w + X[ok, j] * feats[j].w
            assert np.all(s <= c.rhs_hi + 1e-9)
            if two_sided:
                assert np.all(s >= c.rhs_lo - 1e-9)


def test_tree_intervals_disjunction_across_paths():
    from sklearn.tree import DecisionTreeClassifier
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 10, size=(4000, 2))
    # class 1 iff x0 < 3 or x0 > 8 ; x1 irrelevant
    y = ((X[:, 0] < 3) | (X[:, 0] > 8)).astype(int)
    clf = DecisionTreeClassifier(max_depth=3, random_state=0).fit(X, y)
    iv = tree_intervals(clf, ["T.x0", "T.x1"], target_class=1)
    assert "T.x0" in iv
    # every positive row must fall inside one of the x0 intervals
    for x0 in X[y == 1, 0]:
        assert any(lo < x0 <= hi for lo, hi in iv["T.x0"]), (x0, iv["T.x0"])
    # x1 is not on every path -> no predicate emitted for it
    assert "T.x1" not in iv
