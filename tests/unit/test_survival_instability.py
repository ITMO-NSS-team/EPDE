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
    block_gram_partition, chi2_scores, heterogeneity_scores, survival_scores,
    tile_scores,
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


class TestHeterogeneityScores:

    def test_constant_coefficients_score_near_zero(self):
        n = 400
        x = np.linspace(0, 2 * np.pi, n)
        rng = np.random.default_rng(3)
        X = np.column_stack([np.sin(x), np.cos(2 * x)])
        y = X @ np.array([1.0, -3.0]) + 1e-3 * rng.normal(size=n)
        scores = heterogeneity_scores(X, y, None, (n,), fit_intercept=False)
        assert np.all(scores <= 1e-3)
        assert np.all(scores >= 0.0)

    def test_flags_drifting_coefficient(self):
        # y = 2*f1 + x*f2: f2's per-block coefficient must track the local
        # mean of x (genuine drift, far beyond the block standard errors);
        # f1's stays at 2 with scatter consistent with its standard errors.
        n = 600
        x = np.linspace(0.5, 3.0, n)
        f1 = np.sin(5 * x)
        f2 = np.cos(7 * x)
        y = 2.0 * f1 + x * f2
        X = np.column_stack([f1, f2])
        scores = heterogeneity_scores(X, y, None, (n,), fit_intercept=False)
        assert scores[1] > 0.05               # genuine drift flagged
        assert scores[1] > 10.0 * scores[0]   # clearly separated

    def test_noise_column_not_flagged(self):
        # A pure-noise column's block scatter matches its own standard
        # errors -> calibrated score ~0 (insignificance is the t-stat
        # anchor's job, not heterogeneity's). Raw-dispersion scores would
        # explode here (MAD/|median| with median ~ 0).
        n = 400
        rng = np.random.default_rng(1)
        x = np.linspace(0, 4 * np.pi, n)
        f_true = np.sin(x) + 0.5 * np.cos(3 * x)
        f_junk = rng.normal(size=n)
        y = 2.0 * f_true + 1e-2 * rng.normal(size=n)
        X = np.column_stack([f_true, f_junk])
        scores = heterogeneity_scores(X, y, None, (n,), fit_intercept=False)
        assert np.all(np.isfinite(scores))
        assert np.all((scores >= 0.0) & (scores <= 1.0))
        assert scores[1] <= 0.01

    def test_unexcited_column_bounded(self):
        # A near-zero column makes the block Grams near-degenerate; the
        # relative ridge turns that into huge standard errors (~zero
        # evidence), not fake consistency or a LinAlgError.
        n = 400
        x = np.linspace(0, 2 * np.pi, n)
        f_true = np.sin(x)
        f_dead = np.full(n, 1e-9)
        y = 2.0 * f_true
        X = np.column_stack([f_true, f_dead])
        scores = heterogeneity_scores(X, y, None, (n,), fit_intercept=False)
        assert np.all(np.isfinite(scores))
        assert np.all((scores >= 0.0) & (scores <= 1.0))

    def test_fails_loudly_on_too_few_samples(self):
        n = 20
        X = np.ones((n, 2))
        with pytest.raises(ValueError, match='>= 3 blocks'):
            heterogeneity_scores(X, np.ones(n), None, (n,))

    def test_scale_invariance(self):
        n = 400
        x = np.linspace(0.5, 3.0, n)
        f1 = np.sin(5 * x)
        f2 = np.cos(7 * x)
        y = 2.0 * f1 + x * f2
        X = np.column_stack([f1, f2])
        a = heterogeneity_scores(X, y, None, (n,), fit_intercept=False)
        b = heterogeneity_scores(X * np.array([3.0, 0.2]), y * 7.0, None,
                                 (n,), fit_intercept=False)
        assert np.allclose(a, b, atol=1e-6)


class TestChi2Scores:
    """``chi2_scores`` keeps Theta from OLS on ALL the data and watches the
    per-term cumulative score ``S_j(t) = sum_{i<=t} w_i X_ij r_i``, which the
    global normal equations pin to zero at both ends. A constant coefficient
    leaves an aimless bridge; a drifting one leaves a one-sided bulge.
    Shared design: ``x`` is the time axis, ``f1``/``f2`` two oscillatory
    terms."""

    @staticmethod
    def _fields(n=600):
        x = np.linspace(0.5, 3.0, n)
        return x, np.sin(5 * x), np.cos(7 * x)

    def test_returns_one_score_per_term(self):
        n = 600
        x, f1, f2 = self._fields(n)
        scores = chi2_scores(np.column_stack([f1, f2]), 2.0 * f1 + x * f2,
                             None, (n,), fit_intercept=False)
        assert scores.shape == (2,)
        assert np.all(scores >= 0.0)

    def test_exact_fit_scores_zero(self):
        n = 600
        _, f1, f2 = self._fields(n)
        X = np.column_stack([f1, f2])
        scores = chi2_scores(X, 2.0 * f1 - 3.0 * f2, None, (n,),
                             fit_intercept=False)
        assert np.all(scores == 0.0)          # no residual -> nothing to test

    def test_constant_coefficients_score_small(self):
        # iid error under a correct constant-coefficient model: each path is
        # a noise bridge whose bulge is tiny against the term's own signal
        # energy (the residual magnitude enters -- it does NOT cancel).
        n = 600
        rng = np.random.default_rng(5)
        _, f1, f2 = self._fields(n)
        X = np.column_stack([f1, f2])
        y = X @ np.array([1.0, -3.0]) + 1e-2 * rng.normal(size=n)
        scores = chi2_scores(X, y, None, (n,), fit_intercept=False)
        assert np.all(scores < 0.1)

    def test_flags_the_drifting_term_only(self):
        # y = 2*f1 + x*f2 fitted with CONSTANT coefficients: f2 is
        # under-fitted early and over-fitted late, so its score path takes
        # one big excursion. f1's coefficient is genuinely constant, but the
        # separation is ~6x rather than total: this is the DIAGONAL form of
        # the statistic (one V_j per term, no cross-covariance), so a term
        # correlated with the drifting one leaks part of the bulge. The
        # multivariate S^T V^-1 S form would decorrelate that, at the cost
        # of inverting V.
        n = 600
        x, f1, f2 = self._fields(n)
        X = np.column_stack([f1, f2])
        scores = chi2_scores(X, 2.0 * f1 + x * f2, None, (n,),
                             fit_intercept=False)
        assert scores[1] > 4.0 * scores[0]

    def test_spurious_column_costs_something(self):
        # THE property the residual-only revision could not have: a junk
        # column whose fitted coefficient is ~0 leaves the residual field
        # untouched, so any residual-only statistic ties with the truth and
        # loses on discrepancy. Its own score path still bulges, so a
        # per-term score charges for it.
        n = 600
        rng = np.random.default_rng(1)
        x, f1, f2 = self._fields(n)
        y = 2.0 * f1 - 3.0 * f2 + 1e-2 * rng.normal(size=n)
        truth = chi2_scores(np.column_stack([f1, f2]), y, None, (n,),
                            fit_intercept=False)
        overfit = chi2_scores(np.column_stack([f1, f2, rng.normal(size=n)]),
                              y, None, (n,), fit_intercept=False)
        assert overfit[2] > 0.0
        assert float(np.sum(overfit)) > float(np.sum(truth))

    def test_scale_invariance(self):
        # S_j scales with X_j*r and V_j with its square, so L_j is invariant
        # to rescaling any column or the target.
        n = 600
        x, f1, f2 = self._fields(n)
        X = np.column_stack([f1, f2])
        y = 2.0 * f1 + x * f2
        a = chi2_scores(X, y, None, (n,), fit_intercept=False)
        b = chi2_scores(X * np.array([3.0, 0.2]), y * 7.0, None, (n,),
                        fit_intercept=False)
        assert np.allclose(a, b, rtol=1e-9, atol=1e-12)

    def test_deterministic(self):
        n = 600
        x, f1, f2 = self._fields(n)
        X = np.column_stack([f1, f2])
        y = 2.0 * f1 + x * f2
        assert np.array_equal(chi2_scores(X, y, None, (n,)),
                              chi2_scores(X, y, None, (n,)))

    def test_dead_column_carries_no_evidence(self):
        # An all-zero column has zero path AND zero signal energy; the 0/0
        # maps to 0 (no evidence), not NaN and not the degenerate-form
        # explosion (which is reserved for a LIVE column fitted to ~0).
        n = 600
        x, f1, _ = self._fields(n)
        X = np.column_stack([f1, np.zeros(n)])
        scores = chi2_scores(X, 2.0 * f1 + 0.1 * x, None, (n,),
                             fit_intercept=False)
        assert np.all(np.isfinite(scores))
        assert scores[1] == 0.0

    def test_fails_loudly_without_three_levels(self):
        X = np.ones((2, 1))
        with pytest.raises(ValueError, match='>= 3 levels'):
            chi2_scores(X, np.ones(2), None, (2,), fit_intercept=False)

    def test_two_axis_catches_spatially_modulated_term(self):
        # The wave mechanism, reconstructed: on a separable standing wave
        # u = sin(x)cos(t) (so u_tt = u_xx exactly), the x-modulated term
        # (1 + 0.8 sin 2x) * u_xx has separable misfit s = g(x) h(t) with
        # h = cos^2(t) >= 0. The normal equations pin sum_x g ~ 0, so every
        # TIME-level increment vanishes and the t-path lies flat -- the
        # t-only revision scored this family 0.54x BELOW the truth. The
        # x-path is the running sum of g(x) and bulges. The axis-averaged
        # score must therefore flag the form while the exact truth floors.
        nt, nx = 60, 64
        t = np.linspace(0.0, 2 * np.pi, nt, endpoint=False)
        x = np.linspace(0.0, 2 * np.pi, nx, endpoint=False)
        tt, xx = np.meshgrid(t, x, indexing='ij')
        core = -np.sin(xx) * np.cos(tt)          # = u_xx = u_tt
        y = core.reshape(-1)
        f_mod = ((1.0 + 0.8 * np.sin(2 * xx)) * core).reshape(-1)
        wrong = chi2_scores(f_mod[:, None], y, None, (nt, nx),
                            fit_intercept=False)
        truth = chi2_scores(y[:, None], y, None, (nt, nx),
                            fit_intercept=False)
        assert np.all(truth == 0.0)              # exact fit -> floor
        assert wrong[0] > 0.1                    # x-path bulge is charged

    def test_two_axis_constant_coefficients_small(self):
        # Correct 2-D constant-coefficient model with iid error: both the
        # t-path and the x-path are noise bridges -> tiny on both axes.
        nt, nx = 60, 64
        rng = np.random.default_rng(7)
        t = np.linspace(0.0, 2 * np.pi, nt, endpoint=False)
        x = np.linspace(0.0, 2 * np.pi, nx, endpoint=False)
        tt, xx = np.meshgrid(t, x, indexing='ij')
        f1 = (np.sin(xx) * np.cos(tt)).reshape(-1)
        f2 = (np.cos(2 * xx) * np.sin(3 * tt)).reshape(-1)
        X = np.column_stack([f1, f2])
        y = X @ np.array([2.0, -1.5]) + 1e-2 * rng.normal(size=nt * nx)
        scores = chi2_scores(X, y, None, (nt, nx), fit_intercept=False)
        assert np.all(scores < 0.1)


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

    def test_het_metric_dispatch_and_memo(self):
        global_var.set_instability_metric('het')
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
            heterogeneity_scores(X, y, None, (n,), fit_intercept=False))))
        assert eq._cached_alt_instability == ('het', value)

    def test_chi2_metric_dispatch_and_memo(self):
        global_var.set_instability_metric('chi2')
        n = 600
        x = np.linspace(0.0, 1.0, n)
        X = np.column_stack([np.sin(20 * x)])
        y = 2.0 * X[:, 0] + (x >= 0.5) * 3.0 * np.cos(13 * x)
        eq = SimpleNamespace(
            evaluate=lambda normalize, return_val: (None, y, X),
            weights_internal=np.array([2.0, 0.0]),
            _cached_vc_score=np.array([123.0]),   # must be IGNORED here
        )
        value = Instability().compute(eq, _ctx(n))
        assert value == pytest.approx(float(np.sum(
            chi2_scores(X, y, None, (n,), fit_intercept=False))))
        assert eq._cached_alt_instability == ('chi2', value)
        eq.evaluate = lambda normalize, return_val: (_ for _ in ()).throw(
            AssertionError('must not re-evaluate'))
        assert Instability().compute(eq, _ctx(n)) == pytest.approx(value)

    def test_default_resolution_is_chi2(self):
        assert global_var.resolve_instability_metric() == 'chi2'

    def test_explicit_vcoef_keeps_fast_path(self):
        global_var.set_instability_metric('vcoef')
        eq = SimpleNamespace(_cached_vc_score=np.array([0.1, 0.2]))
        assert Instability().compute(eq, _ctx(10)) == pytest.approx(0.3)

    def test_invalid_metric_rejected(self):
        with pytest.raises(ValueError):
            global_var.set_instability_metric('bogus')
