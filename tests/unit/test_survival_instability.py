#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for the block-based instability estimators
(``epde/operators/common/survival.py``) and their dispatch through the
``Instability`` filler via ``epde.globals.instability_metric``.

Synthetic design shared by the estimator tests: a 1-D "grid" of N points,
target ``y = 2*x1 + noise-free``, features ``x1`` (true, constant
coefficient everywhere) and either a near-zero column (survival must flag
it) or an x-modulated-contribution column (tile must flag it).
"""

from types import SimpleNamespace

import numpy as np
import pytest

import epde.globals as global_var
from epde.operators.common.survival import (
    block_gram_partition, survival_scores, tile_scores,
)
from epde.operators.common.objectives import FitContext, Instability


def _ctx(n):
    return FitContext(g_fun_vals=None, data_shape=(n,), penalty_coeff=0.2)


class TestBlockGramPartition:

    def test_blocks_sum_to_full_gram(self):
        rng = np.random.default_rng(0)
        X = rng.normal(size=(120, 3))
        y = rng.normal(size=120)
        G_blocks, Gy_blocks = block_gram_partition(X, y, None, (120,), 8)
        p = 3
        G_full = G_blocks.sum(axis=0)
        assert np.allclose(G_full[:p, :p], X.T @ X)
        assert np.allclose(Gy_blocks.sum(axis=0)[:p], X.T @ y)
        assert np.allclose(G_full[-1, -1], 120.0)

    def test_clamps_block_count(self):
        X = np.ones((10, 1))
        G_blocks, _ = block_gram_partition(X, np.ones(10), None, (10,), 64)
        assert G_blocks.shape[0] <= 10


class TestSurvivalScores:

    def test_true_column_stable_near_zero_column_unstable(self):
        n = 400
        rng = np.random.default_rng(1)
        x = np.linspace(0, 4 * np.pi, n)
        f_true = np.sin(x) + 0.5 * np.cos(3 * x)
        f_junk = rng.normal(size=n)          # uncorrelated -> ~0 coefficient
        y = 2.0 * f_true
        X = np.column_stack([f_true, f_junk])
        scores = survival_scores(X, y, None, (n,), fit_intercept=True)
        assert scores[0] < 0.1               # true term: no flips, tiny spread
        assert scores[1] > 0.4               # near-zero term: flips ~half the draws

    def test_deterministic(self):
        n = 200
        x = np.linspace(0, 2 * np.pi, n)
        X = np.column_stack([np.sin(x), np.cos(x)])
        y = X @ np.array([1.5, -0.7])
        a = survival_scores(X, y, None, (n,))
        b = survival_scores(X, y, None, (n,))
        assert np.array_equal(a, b)

    def test_does_not_touch_global_rng(self):
        n = 200
        x = np.linspace(0, 2 * np.pi, n)
        X = np.column_stack([np.sin(x)])
        y = 2.0 * X[:, 0]
        np.random.seed(42)
        before = np.random.get_state()[1].copy()
        survival_scores(X, y, None, (n,))
        after = np.random.get_state()[1].copy()
        assert np.array_equal(before, after)


class TestTileScores:

    def test_flags_x_modulated_column_keeps_constant_column(self):
        # y = 2*f1 + x*f2: fitting {f1, f2} with CONSTANT coefficients
        # forces f2's per-tile coefficient to track the local mean of x
        # (varies tile to tile), while f1's stays at 2 everywhere.
        # Zero-mean features so within-tile collinearity with the constant
        # doesn't blur the attribution.
        n = 600
        x = np.linspace(0.5, 3.0, n)         # bounded away from 0
        f1 = np.sin(5 * x)
        f2 = np.cos(7 * x)
        y = 2.0 * f1 + x * f2
        X = np.column_stack([f1, f2])
        scores = tile_scores(X, y, None, (n,), fit_intercept=False, n_tiles=8)
        assert scores[0] < 0.1                # constant-coefficient term
        assert scores[1] > 5 * scores[0]      # modulated term clearly flagged

    def test_constant_coefficients_score_near_zero(self):
        n = 400
        x = np.linspace(0, 2 * np.pi, n)
        X = np.column_stack([np.sin(x), np.cos(2 * x)])
        y = X @ np.array([1.0, -3.0])
        scores = tile_scores(X, y, None, (n,))
        assert np.all(scores < 1e-6)


class TestInstabilityDispatch:

    def teardown_method(self):
        global_var.set_instability_metric(None)

    def test_survival_metric_dispatch_and_memo(self):
        global_var.set_instability_metric('survival')
        n = 200
        x = np.linspace(0, 2 * np.pi, n)
        X = np.column_stack([np.sin(x)])
        y = 2.0 * X[:, 0]
        eq = SimpleNamespace(
            evaluate=lambda normalize, return_val: (None, y, X),
            weights_internal=np.array([2.0, 0.0]),
            _cached_vc_score=np.array([123.0]),   # must be IGNORED here
        )
        value = Instability().compute(eq, _ctx(n))
        assert value == pytest.approx(float(np.sum(
            survival_scores(X, y, None, (n,), fit_intercept=False))))
        assert eq._cached_alt_instability == ('survival', value)
        # Memo short-circuits the recompute.
        eq.evaluate = lambda normalize, return_val: (_ for _ in ()).throw(
            AssertionError('must not re-evaluate'))
        assert Instability().compute(eq, _ctx(n)) == pytest.approx(value)

    def test_default_resolution_keeps_vcoef_fast_path(self):
        assert global_var.resolve_instability_metric() == 'vcoef'
        eq = SimpleNamespace(_cached_vc_score=np.array([0.1, 0.2]))
        assert Instability().compute(eq, _ctx(10)) == pytest.approx(0.3)

    def test_invalid_metric_rejected(self):
        with pytest.raises(ValueError):
            global_var.set_instability_metric('bogus')
