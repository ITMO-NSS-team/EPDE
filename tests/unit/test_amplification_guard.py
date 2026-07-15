"""Unit tests for the RPS amplified-identity guard
(`_amplification_ratio` + `globals.set_rps_amplification_cap`).

The guard declines candidate right parts whose fit only "explains" the
target by amplifying the residual of a near-null feature combination
(huge mutually-cancelling coefficients, e.g. the LV
``Lambda*(du+dv-alpha*u+gamma*v) = d^2u`` parasite). Truth anchors
measure A in [1.0, 6.65]; parasites at ~7e2..1.6e6.
"""
import numpy as np
import pytest

import epde.globals as gv
from epde.operators.common.right_part_selection import _amplification_ratio


class _FakeTerm:
    def __init__(self, col):
        self._col = np.asarray(col, dtype=float)

    def evaluate(self, normalize, grids=None):
        assert normalize is False
        return self._col


class _FakeEq:
    def __init__(self, cols, target_idx, weights_internal):
        self.structure = [_FakeTerm(c) for c in cols]
        self.target_idx = target_idx
        self.weights_internal = np.asarray(weights_internal, dtype=float)


@pytest.fixture()
def rng():
    return np.random.default_rng(0)


def test_healthy_fit_is_order_one(rng):
    # target = 0.5*a - 0.03*b + noise-free: no cancellation -> A ~ 1
    a = rng.normal(size=300)
    b = rng.normal(size=300)
    t = 0.5 * a - 0.03 * b
    eq = _FakeEq([a, b, t], target_idx=2, weights_internal=[0.5, -0.03, 0.0])
    A = _amplification_ratio(eq)
    assert 1.0 <= A < 3.0


def test_amplified_identity_flagged(rng):
    # near-null pair: b = a + tiny noise; coefficients Lambda*(1,-1)
    # stretched to fit an unrelated target -> A ~ Lambda-scale >> 100
    a = rng.normal(size=300)
    b = a + 1e-4 * rng.normal(size=300)
    t = rng.normal(size=300)  # unrelated target, O(1)
    lam = 2.0e4
    eq = _FakeEq([a, b, t], target_idx=2,
                 weights_internal=[lam, -lam, 0.0])
    A = _amplification_ratio(eq)
    assert A > 1e3


def test_intercept_contributes(rng):
    a = rng.normal(size=300)
    t = 1.0 * a
    big_intercept = 500.0
    eq = _FakeEq([a, t], target_idx=1,
                 weights_internal=[1.0, big_intercept])
    A_with = _amplification_ratio(eq)
    eq0 = _FakeEq([a, t], target_idx=1, weights_internal=[1.0, 0.0])
    A_without = _amplification_ratio(eq0)
    assert A_with > A_without


def test_zero_weight_terms_skipped(rng):
    a = rng.normal(size=300)
    junk = rng.normal(size=300) * 1e9   # huge column, zero weight
    t = 2.0 * a
    eq = _FakeEq([a, junk, t], target_idx=2,
                 weights_internal=[2.0, 0.0, 0.0])
    assert _amplification_ratio(eq) < 3.0


def test_nonfinite_maps_to_inf(rng):
    a = rng.normal(size=300)
    t = a.copy()
    eq = _FakeEq([a, t], target_idx=1, weights_internal=[np.nan, 0.0])
    assert _amplification_ratio(eq) == np.inf


def test_zero_target_maps_to_inf(rng):
    a = rng.normal(size=300)
    eq = _FakeEq([a, np.zeros(300)], target_idx=1,
                 weights_internal=[1.0, 0.0])
    assert _amplification_ratio(eq) == np.inf


def test_cap_validator():
    old = gv.rps_amplification_cap
    try:
        gv.set_rps_amplification_cap(50.0)
        assert gv.rps_amplification_cap == 50.0
        gv.set_rps_amplification_cap(None)     # disable
        assert gv.rps_amplification_cap is None
        with pytest.raises(ValueError):
            gv.set_rps_amplification_cap(1.0)  # must exceed truth range
        with pytest.raises(ValueError):
            gv.set_rps_amplification_cap(np.inf)
    finally:
        gv.rps_amplification_cap = old
