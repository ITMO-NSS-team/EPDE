"""Properties of the basis-free instability estimators.

Covers the two additions to the family -- ``chi2_centered`` and the unbounded
``het_raw`` -- plus the degenerate-form guard that ``heterogeneity_scores``
was missing, and the per-column rescaling invariance every member of the
family claims but nothing in the suite currently checks.

The guard has a concrete provenance. On Allen-Cahn, appending a
coordinate-modulated copy of ``u_xx`` to the true equation produced a fitted
coefficient of -9.7e-07 which ``het`` scored exactly 0.0 -- its BEST score --
while also shaving 0.8% off the real term's score. The summed objective
therefore DROPPED when a junk column was added, the discrepancy improved too,
and the true equation was dominated on both axes. Across the 13-system panel
that was 9/13 truths non-dominated for ``het`` against 13/13 for every other
estimator.
"""
import numpy as np
import pytest

from epde.operators.common.objectives import _BASIS_FREE_METRICS
from epde.operators.common.survival import (chi2_centered_scores, chi2_scores,
                                            heterogeneity_raw_scores,
                                            heterogeneity_scores)

GRID = (40, 30)
N = GRID[0] * GRID[1]


def _library(seed=0, offset=0.0):
    """Three well-excited columns and a target they explain exactly."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(N, 3))
    X[:, 0] += offset
    y = X @ np.array([1.0, 2.0, -1.0]) + 0.01 * rng.normal(size=N)
    return X, y, np.ones(N)


def _hitchhiked():
    """The AC failure shape, on a smooth field so it is faithful.

    A RANDOM library will not reproduce it: an i.i.d. junk column is
    well-excited, so its coefficient is resolved (t^2 ~ 3) even though it is
    tiny, and the guard correctly stays out of the way. The failure needs a
    column that is nearly collinear with a true one -- ``sin(2x) * u_xx`` on a
    smooth field -- so the fit cannot identify its coefficient at all.
    """
    nt, nx = GRID
    t = np.linspace(0, 2, nt)
    x = np.linspace(0, 2 * np.pi, nx, endpoint=False)
    T, Xg = np.meshgrid(t, x, indexing='ij')
    u = np.exp(-0.3 * T) * np.sin(Xg) + 0.5 * np.exp(-0.1 * T) * np.sin(2 * Xg)
    uxx = -np.exp(-0.3 * T) * np.sin(Xg) - 2.0 * np.exp(-0.1 * T) * np.sin(2 * Xg)
    F = np.stack([u.reshape(-1), uxx.reshape(-1), (u ** 3).reshape(-1)], axis=1)
    eps = (1e-3 * np.sin(5 * Xg) * (0.5 + T)).reshape(-1)   # out-of-span model error
    y = 5.0 * F[:, 0] + 1e-4 * F[:, 1] - 5.0 * F[:, 2] + eps
    hitch = (np.sin(2 * Xg) * uxx).reshape(-1)
    return F, np.column_stack([F, hitch]), y, np.ones(N)


class TestHetDegenerateGuard:
    def test_an_unresolved_coefficient_is_not_certified_stable(self):
        """A column whose LEVEL is indistinguishable from zero must not score
        0.0. Absence of evidence is not evidence of homogeneity."""
        _, Xh, y, w = _hitchhiked()
        score = heterogeneity_scores(Xh, y, w, GRID, fit_intercept=False)
        coef = np.linalg.lstsq(Xh, y, rcond=None)[0]
        assert abs(coef[3]) < 1e-6, 'fixture must drive the hitchhiker to ~0'
        assert score[3] == pytest.approx(1.0), (
            'the hitchhiker scored its best possible value -- the guard is gone')

    def test_a_hitchhiker_cannot_lower_the_summed_objective(self):
        """The regression proper: adding a column must never make the summed
        instability axis BETTER, or overfitting is free."""
        X, Xh, y, w = _hitchhiked()
        base = float(np.sum(heterogeneity_scores(X, y, w, GRID, fit_intercept=False)))
        with_junk = float(np.sum(heterogeneity_scores(Xh, y, w, GRID, fit_intercept=False)))
        assert with_junk > base

    def test_a_resolved_constant_coefficient_still_scores_near_zero(self):
        """The guard must not cost het its actual purpose: a genuine,
        well-determined, homogeneous term stays at ~0."""
        X, y, w = _library()
        score = heterogeneity_scores(X, y, w, GRID, fit_intercept=False)
        assert np.all(score < 1e-3)

    def test_the_guard_is_scale_invariant(self):
        """It compares theta_bar^2 against its own sampling variance, so
        rescaling a column must not move a term across it."""
        _, Xh, y, w = _hitchhiked()
        a = heterogeneity_scores(Xh, y, w, GRID, fit_intercept=False)
        Xs = Xh.copy()
        Xs[:, 3] *= 7.0
        b = heterogeneity_scores(Xs, y, w, GRID, fit_intercept=False)
        assert np.allclose(a, b, rtol=1e-6, atol=1e-12)


class TestHetRaw:
    def test_agrees_with_the_bounded_form_when_excess_variance_is_small(self):
        """tau2 / (tau2 + mu^2) -> tau2 / mu^2 as tau2 << mu^2, which is the
        regime every clean system sits in."""
        X, y, w = _library()
        b = heterogeneity_scores(X, y, w, GRID, fit_intercept=False)
        r = heterogeneity_raw_scores(X, y, w, GRID, fit_intercept=False)
        assert np.allclose(b, r, rtol=1e-3, atol=1e-12)

    def test_shares_the_degenerate_guard(self):
        """The bound was never what caused the hitchhiking failure -- tau2
        clipping to zero was, and the unbounded form clips identically."""
        _, Xh, y, w = _hitchhiked()
        r = heterogeneity_raw_scores(Xh, y, w, GRID, fit_intercept=False)
        assert r[3] > 1.0, 'unbounded form must exceed the bounded supremum'
        assert np.isfinite(r).all(), 'nan_to_num must render the guard finite'

    def test_is_not_capped_at_one(self):
        """The whole point of the variant: it can order two forms that both
        saturate the bounded score."""
        _, Xh, y, w = _hitchhiked()
        assert float(np.max(heterogeneity_raw_scores(
            Xh, y, w, GRID, fit_intercept=False))) > 1.0


class TestChi2Centered:
    def test_is_a_no_op_on_already_centered_data(self):
        """Centering subtracts a weighted mean; when there is none to
        subtract the statistic must be chi2 itself."""
        X, y, w = _library()
        X = X - X.mean(axis=0)
        y = y - y.mean()
        a = chi2_scores(X, y, w, GRID, fit_intercept=False)
        b = chi2_centered_scores(X, y, w, GRID, fit_intercept=False)
        assert np.allclose(a, b, rtol=1e-9, atol=1e-14)

    def test_differs_from_chi2_when_a_column_carries_a_mean(self):
        """The motivating case: an uncentered offset accumulates into the
        score path as c0 * cumsum(w X), which is not coefficient drift."""
        X, y, w = _library(offset=5.0)
        a = chi2_scores(X, y, w, GRID, fit_intercept=False)
        b = chi2_centered_scores(X, y, w, GRID, fit_intercept=False)
        assert not np.allclose(a, b, rtol=1e-6)

    def test_scores_a_constant_column_zero_and_keeps_alignment(self):
        """The keep-rule appends its own ones column and calls with
        fit_intercept=False, so a constant column reaches the estimator as an
        ordinary feature. Centering it would zero it out and make the Gram
        singular; it is excluded and scored 0.0 instead, and the returned
        vector still has one entry per input column."""
        X, y, w = _library()
        Xa = np.hstack([X, np.ones((N, 1))])
        score = chi2_centered_scores(Xa, y, w, GRID, fit_intercept=False)
        assert score.shape == (4,)
        assert score[3] == 0.0
        bare = chi2_centered_scores(X, y, w, GRID, fit_intercept=False)
        assert np.allclose(score[:3], bare, rtol=1e-9, atol=0.0)

    def test_a_constant_column_of_any_magnitude_is_caught(self):
        """The constant test is scale-free, so a large constant column is
        recognised as one rather than centered into zeros."""
        X, y, w = _library()
        Xa = np.hstack([X, np.full((N, 1), 1e6)])
        score = chi2_centered_scores(Xa, y, w, GRID, fit_intercept=False)
        assert np.isfinite(score).all()
        assert score[3] == 0.0

    def test_all_constant_library_returns_zeros_rather_than_raising(self):
        X, y, w = _library()
        score = chi2_centered_scores(np.ones((N, 2)), y, w, GRID,
                                     fit_intercept=False)
        assert np.array_equal(score, np.zeros(2))

    def test_zero_weights_fail_loudly(self):
        X, y, _ = _library()
        with pytest.raises(ValueError, match='sample weights'):
            chi2_centered_scores(X, y, np.zeros(N), GRID, fit_intercept=False)


@pytest.mark.parametrize('metric', sorted(_BASIS_FREE_METRICS))
class TestFamilyInvariants:
    def test_one_score_per_column(self, metric):
        X, y, w = _library()
        score = _BASIS_FREE_METRICS[metric](X, y, w, GRID, fit_intercept=False)
        assert np.shape(score) == (3,)

    def test_column_rescale_invariance(self, metric):
        """Rescaling a column by ``a`` sends its fitted coefficient to
        ``c/a``; every estimator in the family claims to be blind to that, so
        the physical units of a term cannot change its score."""
        X, y, w = _library()
        a = _BASIS_FREE_METRICS[metric](X, y, w, GRID, fit_intercept=False)
        Xs = X.copy()
        Xs[:, 1] *= 7.0
        b = _BASIS_FREE_METRICS[metric](Xs, y, w, GRID, fit_intercept=False)
        assert np.allclose(a, b, rtol=1e-6, atol=1e-12)

    def test_scores_are_finite_and_non_negative(self, metric):
        X, y, w = _library()
        score = np.asarray(_BASIS_FREE_METRICS[metric](
            X, y, w, GRID, fit_intercept=False), dtype=float)
        assert np.isfinite(score).all()
        assert (score >= 0.0).all()

    def test_deterministic(self, metric):
        X, y, w = _library()
        fn = _BASIS_FREE_METRICS[metric]
        assert np.array_equal(fn(X, y, w, GRID, fit_intercept=False),
                              fn(X, y, w, GRID, fit_intercept=False))

    def test_does_not_disturb_the_global_numpy_stream(self, metric):
        """``survival`` bootstraps; it must use a local generator so a search
        stays reproducible."""
        X, y, w = _library()
        np.random.seed(1234)
        before = np.random.random()
        np.random.seed(1234)
        _BASIS_FREE_METRICS[metric](X, y, w, GRID, fit_intercept=False)
        assert np.random.random() == before


@pytest.mark.parametrize('metric', sorted(_BASIS_FREE_METRICS))
def test_keep_rule_and_objective_agree_numerically(metric):
    """One statistic, both sides -- asserted on the NUMBERS.

    Nothing pinned this before. The nearest test grepped
    ``inspect.getsource(Instability.compute)`` for the estimator's name, which
    could only ever see that it was mentioned. The two call shapes genuinely
    differ: the keep-rule (``sparsity.instability_scores``) appends its own
    ones column and passes ``fit_intercept=False``, so the intercept is an
    ordinary scored column and the vector is one longer; the objective passes
    ``fit_intercept=True`` and lets the estimator append and then drop it.
    Both build the same augmented design, so the FEATURE scores must match.

    ``survival`` failed this until its bootstrap stream stopped depending on
    ``p``, which differs between the two shapes (nf + 1 vs nf).
    """
    from epde.operators.common.sparsity import instability_scores

    X, y, w = _library(offset=2.0)
    n_features = X.shape[1]
    active = np.ones(n_features + 1, dtype=bool)      # every feature + intercept

    keep = np.asarray(instability_scores(metric, X, y, w, GRID, active,
                                         n_features), dtype=float)
    objective = np.asarray(_BASIS_FREE_METRICS[metric](
        X, y, w, GRID, fit_intercept=True), dtype=float)

    assert keep.shape == (n_features + 1,), 'keep-rule also scores the intercept'
    assert objective.shape == (n_features,), 'the objective drops it'
    assert np.allclose(keep[:n_features], objective, rtol=1e-9, atol=1e-15)
