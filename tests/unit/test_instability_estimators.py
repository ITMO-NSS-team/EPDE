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
import itertools

import numpy as np
import pytest

from conftest import using_config
from epde.operators.common.objectives import _BASIS_FREE_METRICS
from epde.operators.common.survival import (_dyadic_slab_edges,
                                            _reference_width,
                                            chi2_centered_scores, chi2_scores,
                                            dyad_t_scores,
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

    def test_scores_a_constant_column_as_the_intercept_and_keeps_alignment(self):
        """The keep-rule appends its own ones column and calls with
        fit_intercept=False, so a constant column reaches the estimator as an
        ordinary feature. Centering it would zero it out and make the Gram
        singular, so it is excluded from the centered design and scored with
        the chi2 intercept functional instead -- the entry ``chi2_scores``
        gives the same ones column (it used to be 0.0, which made the
        intercept unprunable). The returned vector still has one entry per
        input column, and the feature entries are untouched."""
        X, y, w = _library()
        Xa = np.hstack([X, np.ones((N, 1))])
        score = chi2_centered_scores(Xa, y, w, GRID, fit_intercept=False)
        assert score.shape == (4,)
        assert np.isfinite(score[3]) and score[3] > 0.0
        uncentred = chi2_scores(Xa, y, w, GRID, fit_intercept=False)
        assert score[3] == pytest.approx(uncentred[3], rel=1e-8)
        bare = chi2_centered_scores(X, y, w, GRID, fit_intercept=False)
        assert np.allclose(score[:3], bare, rtol=1e-9, atol=0.0)

    def test_a_constant_column_of_any_magnitude_is_caught(self):
        """The constant test is exact, so a large constant column is
        recognised as one rather than centered into zeros, and it scores what
        the unit ones column scores -- the functional is blind to the value."""
        X, y, w = _library()
        ones = chi2_centered_scores(np.hstack([X, np.ones((N, 1))]), y, w, GRID,
                                    fit_intercept=False)
        Xa = np.hstack([X, np.full((N, 1), 1e6)])
        score = chi2_centered_scores(Xa, y, w, GRID, fit_intercept=False)
        assert np.isfinite(score).all()
        assert score[3] > 0.0
        assert score[3] == pytest.approx(ones[3], rel=1e-9)

    def test_all_constant_library_scores_the_offset_rather_than_raising(self):
        """No column survives centering: the fit is intercept-only, and the
        two collinear constants tie on its score instead of the empty centered
        design raising."""
        X, y, w = _library()
        score = chi2_centered_scores(np.ones((N, 2)), y, w, GRID,
                                     fit_intercept=False)
        assert np.isfinite(score).all() and score[0] > 0.0
        assert score[0] == score[1]

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
    """One statistic, both call shapes -- asserted on the NUMBERS.

    Nothing pinned this before. The nearest test grepped
    ``inspect.getsource(Instability.compute)`` for the estimator's name, which
    could only ever see that it was mentioned. The two call shapes genuinely
    differ: the keep-rule (``sparsity.instability_scores``) appends its own
    ones column and passes ``fit_intercept=False``, so the intercept is an
    ordinary scored column and the vector is one longer; the FOLDED call
    passes ``fit_intercept=True`` and lets the estimator append and then drop
    it. Both build the same augmented design, so the FEATURE scores must
    match. (The ``Instability`` objective itself now uses the keep-rule shape
    under the regularized-intercept rule -- see
    ``test_intercept_scoring_rule.py`` -- so this pins the estimators for the
    offline callers that still fold.)

    ``survival`` failed this until its bootstrap stream stopped depending on
    ``p``, which differs between the two shapes (nf + 1 vs nf).
    """
    from epde.operators.common.sparsity import instability_scores

    X, y, w = _library(offset=2.0)
    n_features = X.shape[1]
    active = np.ones(n_features + 1, dtype=bool)      # every feature + intercept

    keep = np.asarray(instability_scores(metric, X, y, w, GRID, active,
                                         n_features), dtype=float)
    folded = np.asarray(_BASIS_FREE_METRICS[metric](
        X, y, w, GRID, fit_intercept=True), dtype=float)

    assert keep.shape == (n_features + 1,), 'keep-rule also scores the intercept'
    assert folded.shape == (n_features,), 'the folded call drops it'
    assert np.allclose(keep[:n_features], folded, rtol=1e-9, atol=1e-15)


def _drifting_library():
    """Two constant-coefficient columns and one whose coefficient drifts in t."""
    nt, nx = GRID
    t = np.linspace(0, 1, nt)
    rng = np.random.default_rng(3)
    X = rng.normal(size=(N, 3))
    T = np.repeat(t, nx)
    y = 1.0 * X[:, 0] + (1.0 + 2.0 * T) * X[:, 1] - 1.0 * X[:, 2]
    return X, y, np.ones(N)


class TestDyadT:
    def test_partitions_are_dyadic_nested_and_respect_the_floor(self):
        min_rows = 4 * (_reference_width(3) + 1)
        parts = _dyadic_slab_edges(N, GRID, min_rows)
        stride = GRID[1]
        assert [len(e) - 1 for e in parts] == [2 ** s for s in range(1, len(parts) + 1)]
        fine = set(parts[-1].tolist())
        for e in parts:
            assert e[0] == 0 and e[-1] == N
            assert set(e.tolist()) <= fine, 'coarser edges must be finer edges'
            assert np.all(np.diff(e) % stride == 0), 'whole time steps only'
            assert np.diff(e).min() >= min_rows
        assert 2 ** (len(parts) + 1) * (-(-min_rows // stride)) > GRID[0], 'no finer partition fits'

    def test_at_least_one_partition_on_a_tiny_grid(self):
        assert [len(e) - 1 for e in _dyadic_slab_edges(12, (3, 4), 1000)] == [2]

    def test_constant_coefficients_score_round_off(self):
        X, y, w = _library()
        y = X @ np.array([1.0, 2.0, -1.0])
        assert np.all(dyad_t_scores(X, y, w, GRID, fit_intercept=False) < 1e-10)

    def test_a_drifting_coefficient_is_the_unstable_term(self):
        X, y, w = _drifting_library()
        score = dyad_t_scores(X, y, w, GRID, fit_intercept=False)
        # the within-slab drift is unmodelled, so it leaks a little into the
        # other terms' slab fits (~0.007 here) -- far below the drifting term
        assert score[1] > 0.1
        assert score[1] > 10 * max(score[0], score[2])

    def test_matches_an_explicit_per_slab_refit(self):
        """Reference: lstsq on every slab of every partition, MAD / |median|
        per term, averaged over the partitions."""
        X, y, w = _drifting_library()
        min_rows = 4 * (_reference_width(3) + 1)
        total = np.zeros(3)
        parts = _dyadic_slab_edges(N, GRID, min_rows)
        for e in parts:
            coefs = np.array([np.linalg.lstsq(X[a:b], y[a:b], rcond=None)[0]
                              for a, b in zip(e[:-1], e[1:])])
            med = np.median(coefs, axis=0)
            total += np.median(np.abs(coefs - med), axis=0) / np.abs(med)
        assert np.allclose(dyad_t_scores(X, y, w, GRID, fit_intercept=False),
                           total / len(parts), rtol=1e-8, atol=1e-14)

    @pytest.mark.parametrize('grid', [(40, 30), (6, 5, 40), (1200,)])
    def test_batched_level_grams_equal_the_slab_loop(self, grid):
        """The one-multiply per-level Grams summed over the slabs are the
        slab loop's Grams (weights and the folded intercept included)."""
        from epde.operators.common.survival import (_edge_gram_blocks,
                                                    _level_gram_blocks)
        rng = np.random.default_rng(7)
        X = rng.normal(size=(N, 3)) * np.array([0.01, 1.0, 100.0])
        y = rng.normal(size=N)
        w = rng.uniform(0.5, 1.5, size=N)
        edges = _dyadic_slab_edges(N, grid, 28)[-1]
        G, g = _level_gram_blocks(X, y, w, grid, edges)
        G0, g0 = _edge_gram_blocks(X, y, w, edges)
        assert np.allclose(G, G0, rtol=1e-12, atol=0.0)
        assert np.allclose(g, g0, rtol=1e-12, atol=1e-12 * np.abs(g0).max())

    def test_rank_deficient_slabs_get_the_min_norm_fit(self):
        """On a one-level time slab ``t * f(x)`` is ``f(x)`` times a constant,
        so the slab design is exactly singular. ``np.linalg.solve`` does not
        raise there and returned huge coefficients (a 46x error against the
        study's dyad_t on Allen-Cahn); the min-norm solution matches an
        explicit column-normalised lstsq."""
        nt, nx = GRID
        rng = np.random.default_rng(5)
        f = np.tile(rng.normal(size=nx), nt)
        g = rng.normal(size=N)
        T = np.repeat(np.linspace(0, 1, nt), nx)
        X = np.column_stack([f, g, T * f])
        y = 2.0 * f + 3.0 * g
        parts = _dyadic_slab_edges(N, GRID, 4 * (_reference_width(3) + 1))
        assert np.diff(parts[-1]).min() == nx, 'fixture needs one-level slabs'
        total = np.zeros(3)
        for e in parts:
            coefs = []
            for a, b in zip(e[:-1], e[1:]):
                nrm = np.linalg.norm(X[a:b], axis=0)
                nrm = np.where(nrm > 0, nrm, 1.0)
                coefs.append(np.linalg.lstsq(X[a:b] / nrm, y[a:b], rcond=None)[0] / nrm)
            coefs = np.array(coefs)
            med = np.median(coefs, axis=0)
            with np.errstate(divide='ignore', invalid='ignore'):
                rel = np.median(np.abs(coefs - med), axis=0) / np.abs(med)
            total += np.where(np.isnan(rel), 0.0, rel)
        ref = np.nan_to_num(total / len(parts))
        score = dyad_t_scores(X, y, np.ones(N), GRID, fit_intercept=False)
        assert np.all(np.isfinite(score))
        assert score[1] < 1e-10, 'g is identified in every slab'
        assert np.allclose(score[:2], ref[:2], rtol=1e-8, atol=1e-10)

    def test_is_tile_when_only_one_partition_fits(self):
        """Two slabs, nothing finer: the same refits and statistic as a
        two-tile ``tile``."""
        from epde.operators.common.survival import tile_scores
        X, y, w = _drifting_library()
        grid, n = (2, 600), 1200
        assert len(_dyadic_slab_edges(n, grid, 4 * (_reference_width(3) + 1))) == 1
        assert np.allclose(dyad_t_scores(X, y, w, grid, fit_intercept=False),
                           tile_scores(X, y, w, grid, fit_intercept=False, n_tiles=2),
                           rtol=1e-9, atol=1e-14)

    def test_a_zero_median_with_spread_is_capped(self):
        """A column irrelevant on average but not per slab: |median| ~ 0
        under a nonzero spread scores far above every real term."""
        X, y, w = _drifting_library()
        T = np.repeat(np.linspace(-1, 1, GRID[0]), GRID[1])
        y2 = X[:, 0] + T * X[:, 1]
        score = dyad_t_scores(X, y2, w, GRID, fit_intercept=False)
        assert score[1] > 1e6
        assert score[0] < 0.1


def test_the_default_dyad_t_solve_is_csne():
    from epde.operators.common import survival
    assert survival._DYAD_SOLVE == 'csne'


class TestDyadTEigh(TestDyadT):
    """Every dyad_t property above under the eigh slab solve (the default
    before csne, and every other solve's reference)."""

    @pytest.fixture(autouse=True)
    def _eigh(self, monkeypatch):
        from epde.operators.common import survival
        monkeypatch.setattr(survival, '_DYAD_SOLVE', 'eigh')


class TestDyadTCholesky(TestDyadT):
    """Every dyad_t property above under ``survival._DYAD_SOLVE =
    'cholesky'`` (the certified vectorised slab solve), plus the solve's own
    contract against the eigh min-norm solve."""

    @pytest.fixture(autouse=True)
    def _cholesky(self, monkeypatch):
        from epde.operators.common import survival
        monkeypatch.setattr(survival, '_DYAD_SOLVE', 'cholesky')

    @staticmethod
    def _stack(kind, m=400, k=5, rows=31, seed=0):
        rng = np.random.default_rng(seed)
        X = rng.normal(size=(m, rows, k)) * np.array([1e-3, 1.0, 10.0, 1.0, 1e3])[:k]
        if kind == 'dependent':
            X[::3, :, 1] = 2.0 * X[::3, :, 0]                  # every third slab exactly singular
        elif kind == 'near':
            X[:, :, 1] = X[:, :, 0] * 1e3 + 1e-9 * rng.normal(size=(m, rows))
        elif kind == 'zero':
            X[::2, :, 2] = 0.0
        y = rng.normal(size=(m, rows))
        return np.einsum('mri,mrj->mij', X, X), np.einsum('mri,mr->mi', X, y)

    def test_well_conditioned_slabs_take_the_eigh_solution(self):
        from epde.operators.common.survival import (_solve_min_norm_cholesky,
                                                    _solve_min_norm_eigh)
        G, Gy = self._stack('gauss')
        a, b = _solve_min_norm_eigh(G, Gy), _solve_min_norm_cholesky(G, Gy)
        np.testing.assert_allclose(b, a, rtol=1e-11, atol=1e-13 * np.abs(a).max())

    @pytest.mark.parametrize('kind', ['dependent', 'near', 'zero'])
    def test_slabs_near_the_cut_go_to_the_eigh_solve(self, kind):
        """Rank-deficient, nearly dependent and zero-column slabs are not
        certified: they get the eigh min-norm solution bit for bit, the rest
        the eigh solution to round-off."""
        from epde.operators.common.survival import (_solve_min_norm_cholesky,
                                                    _solve_min_norm_eigh)
        G, Gy = self._stack(kind)
        a, b = _solve_min_norm_eigh(G, Gy), _solve_min_norm_cholesky(G, Gy)
        bad = {'dependent': np.arange(len(G)) % 3 == 0, 'near': np.ones(len(G), bool),
               'zero': np.arange(len(G)) % 2 == 0}[kind]
        assert np.array_equal(b[bad], _solve_min_norm_eigh(G[bad], Gy[bad]))
        np.testing.assert_allclose(b[~bad], a[~bad], rtol=1e-10, atol=1e-12 * np.abs(a).max())

    def test_non_finite_slabs_behave_as_under_eigh(self):
        from epde.operators.common.survival import (_solve_min_norm_cholesky,
                                                    _solve_min_norm_eigh)
        G, Gy = self._stack('gauss', m=10)
        G[3, 1, 1] = np.nan

        def outcome(solve):
            try:
                return solve(G, Gy)
            except np.linalg.LinAlgError as err:
                return type(err)
        a, b = outcome(_solve_min_norm_eigh), outcome(_solve_min_norm_cholesky)
        if isinstance(a, np.ndarray):
            assert isinstance(b, np.ndarray)
            assert np.array_equal(np.isnan(a[3]), np.isnan(b[3]))
            np.testing.assert_allclose(np.delete(b, 3, 0), np.delete(a, 3, 0), rtol=1e-11)
        else:
            assert b is a

    def test_subset_scores_agree_with_the_eigh_scores(self, monkeypatch):
        from epde.operators.common import survival
        X, y, w = _drifting_library()
        subsets = [[0], [1], [0, 1], [0, 2], [0, 1, 2]]
        chol = survival.dyad_t_subset_scores(X, y, w, GRID, subsets)
        monkeypatch.setattr(survival, '_DYAD_SOLVE', 'eigh')
        eigh = survival.dyad_t_subset_scores(X, y, w, GRID, subsets)
        np.testing.assert_allclose(chol, eigh, rtol=1e-9, atol=1e-14)


def _smooth_ode_library(n=4000, seed=3):
    """A lorenz-like 1-D record: smooth columns whose ~30-row slabs are
    near-singular (a polynomial eigenvalue ladder), the case where the Gram
    solve's round-off decides the cut."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 20.0, n)
    u = np.sin(t) + 0.3 * np.sin(2.7 * t)
    v = np.cos(1.3 * t)
    X = np.column_stack([u, v, u * v, u ** 2, v ** 2, t, u ** 3])
    y = 2.0 * u - 0.5 * u * v + 1e-6 * rng.normal(size=n)
    return X, y, np.ones(n)


class TestDyadTHybrid(TestDyadT):
    """Every dyad_t property above under ``survival._DYAD_SOLVE = 'hybrid'``
    (the Cholesky solve where a bound proves a well-conditioned slab, the
    same rule from the slab design elsewhere), plus its contract: the eigh
    solution on well-conditioned slabs, the design's min-norm solution on the
    rest, and scores that round-off no longer moves."""

    @pytest.fixture(autouse=True)
    def _hybrid(self, monkeypatch):
        from epde.operators.common import survival
        monkeypatch.setattr(survival, '_DYAD_SOLVE', 'hybrid')

    def test_well_conditioned_scores_are_the_eigh_scores(self, monkeypatch):
        from epde.operators.common import survival
        X, y, w = _drifting_library()
        subsets = [[0], [1], [0, 1], [0, 2], [0, 1, 2]]
        hybrid = survival.dyad_t_subset_scores(X, y, w, GRID, subsets)
        monkeypatch.setattr(survival, '_DYAD_SOLVE', 'eigh')
        np.testing.assert_allclose(hybrid, survival.dyad_t_subset_scores(X, y, w, GRID, subsets),
                                   rtol=1e-10, atol=1e-14)

    def test_near_singular_slabs_take_the_design_solution(self):
        """On a smooth 1-D record the scores are the statistic evaluated from
        per-slab ``lstsq`` of the column-normalised weighted design with the
        same cut (``rcond = sqrt(_DYAD_RCOND)`` on singular values)."""
        from epde.operators.common import survival
        X, y, w = _smooth_ode_library()
        n, p = X.shape
        grid = (n,)
        parts = _dyadic_slab_edges(n, grid, 4 * (_reference_width(p) + 1))
        total = np.zeros(p)
        for e in parts:
            coefs = []
            for a, b in zip(e[:-1], e[1:]):
                nrm = np.linalg.norm(X[a:b], axis=0)
                coefs.append(np.linalg.lstsq(X[a:b] / nrm, y[a:b],
                                             rcond=np.sqrt(survival._DYAD_RCOND))[0] / nrm)
            coefs = np.array(coefs)
            med = np.median(coefs, axis=0)
            with np.errstate(divide='ignore', invalid='ignore'):
                rel = np.median(np.abs(coefs - med), axis=0) / np.abs(med)
            total += np.where(np.isnan(rel), 0.0, rel)
        ref = np.nan_to_num(total / len(parts))
        score = dyad_t_scores(X, y, w, grid, fit_intercept=False)
        # Two accurate evaluations of ill-conditioned slabs agree to ~kappa*eps
        # (2.8e-5 here); the Gram solve is 27% off on this fixture.
        np.testing.assert_allclose(score, ref, rtol=1e-4, atol=1e-12)
        survival._DYAD_SOLVE = 'eigh'
        gram = dyad_t_scores(X, y, w, grid, fit_intercept=False)
        assert np.max(np.abs(gram - ref) / np.abs(ref)) > 1e-2

    def test_round_off_no_longer_moves_the_scores(self):
        """A 1e-14 relative perturbation of the columns moves no subset score
        by more than 1e-4 relative (measured: 7e-6; the Gram solve moves them
        by O(1) here)."""
        from epde.operators.common import survival
        X, y, w = _smooth_ode_library()
        n = X.shape[0]
        Xa = np.hstack([X, np.ones((n, 1))])
        subsets = [list(S) for k in (2, 3, 4) for S in itertools.combinations(range(Xa.shape[1]), k)]
        Xp = Xa * (1.0 + 1e-14 * np.random.default_rng(0).normal(size=Xa.shape))
        a = survival.dyad_t_subset_scores(Xa, y, w, (n,), subsets)
        b = survival.dyad_t_subset_scores(Xp, y, w, (n,), subsets)
        ok = (a < 1e300) & (b < 1e300)
        assert ok.mean() > 0.9
        np.testing.assert_allclose(b[ok], a[ok], rtol=1e-4, atol=1e-12)

    def test_without_the_slab_designs_it_refuses(self):
        from epde.operators.common import survival
        G = np.eye(2)[None]
        with pytest.raises(ValueError, match=survival._DYAD_SOLVE):
            survival._solve_min_norm(G, np.ones((1, 2)))


def _banded_designs(m=300, rows=31, k=4, seed=0):
    """Slab designs whose unit-diagonal Gram ratios ``lambda_min / lambda_max``
    fall in the three bands of ``_solve_min_norm_csne`` (column 1 is column 0
    plus a relative perturbation ``delta``, so the ratio is about
    ``delta**2 / (2 k)``): the Gram stack, its cross moments, a ``design``
    callable on the R factors of ``[A | y]``, and each slab's delta."""
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(m, rows, k)) * np.array([1e-3, 1.0, 10.0, 1e3])[:k]
    delta = np.geomspace(1e-1, 1e-7, m)
    A[:, :, 1] = 1e3 * A[:, :, 0] + delta[:, None] * rng.normal(size=(m, rows))
    y = (A @ rng.normal(size=k)[:, None])[:, :, 0] + 1e-3 * rng.normal(size=(m, rows))
    R = np.linalg.qr(np.concatenate([A, y[:, :, None]], axis=2), mode='r')

    def design(rows_):
        return R[rows_][:, :, :k], R[rows_][:, :, k]
    return np.einsum('mri,mrj->mij', A, A), np.einsum('mri,mr->mi', A, y), design


class TestDyadTCSNE(TestDyadTHybrid):
    """Every dyad_t property above (and the hybrid's contract) under
    ``survival._DYAD_SOLVE = 'csne'``: the Cholesky solution down to
    ``_CSNE_RATIO``, refined once from the slab design below
    ``_HYBRID_RATIO``, the design solve under ``_CSNE_RATIO``."""

    @pytest.fixture(autouse=True)
    def _hybrid(self, monkeypatch):
        from epde.operators.common import survival
        monkeypatch.setattr(survival, '_DYAD_SOLVE', 'csne')

    @staticmethod
    def _bands(G):
        from epde.operators.common import survival
        _Linv, _d, ok, trace_inv = survival._cholesky_factors(G)
        k = G.shape[1]
        with np.errstate(all='ignore'):
            plain = ok & (trace_inv * (survival._HYBRID_RATIO * k) < 1.0)
            certified = ok & (trace_inv * (survival._CSNE_RATIO * k) < 1.0)
        return plain, certified & ~plain, ~certified

    def test_the_fixture_fills_every_band(self):
        G, _Gy, _design = _banded_designs()
        assert all(band.sum() >= 20 for band in self._bands(G))

    def test_each_band_takes_its_solve(self):
        """Well-conditioned slabs: the hybrid's Cholesky solution bit for
        bit. Uncertified slabs: the design solve bit for bit, and only they
        reach it -- the band between is refined, not handed to the design
        SVD (which would cost the speed this solve exists for). That band
        gets the design solve's solution to about the level at which the two
        agree on the well-conditioned slabs (~1e-8 of the largest coefficient
        on this fixture), where the bare Cholesky solution is off by up to
        ~3e-5."""
        from epde.operators.common import survival
        G, Gy, design = _banded_designs()
        plain, refined, rest = self._bands(G)
        to_design, solve_design = [], survival._solve_min_norm_design

        def counting(M, rhs, total=None):
            to_design.append(len(M))
            return solve_design(M, rhs, total=total)
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(survival, '_solve_min_norm_design', counting)
            theta = survival._solve_min_norm_csne(G, Gy, design)
        assert to_design == [int(rest.sum())]
        chol, _ = survival._cholesky_certified(G, Gy, survival._CSNE_RATIO)
        assert np.array_equal(theta[plain], chol[plain])
        M, rhs = design(np.flatnonzero(rest))
        assert np.array_equal(theta[rest], survival._solve_min_norm_design(M, rhs))
        M, rhs = design(np.flatnonzero(refined))
        accurate = survival._solve_min_norm_design(M, rhs)
        assert not np.array_equal(theta[refined], accurate)
        scale = np.abs(accurate).max(axis=1, keepdims=True)
        err_csne = np.max(np.abs(theta[refined] - accurate) / scale)
        err_chol = np.max(np.abs(chol[refined] - accurate) / scale)
        assert err_csne < 1e-7
        assert err_chol > 100 * err_csne

    def test_the_solution_is_the_hybrid_one_to_round_off(self, monkeypatch):
        """To the two accurate solves' agreement level (see above)."""
        from epde.operators.common import survival
        G, Gy, design = _banded_designs()
        csne = survival._solve_min_norm(G, Gy, design)
        monkeypatch.setattr(survival, '_DYAD_SOLVE', 'hybrid')
        hybrid = survival._solve_min_norm(G, Gy, design)
        scale = np.abs(hybrid).max(axis=1, keepdims=True)
        assert np.max(np.abs(csne - hybrid) / scale) < 1e-7

    def test_non_finite_slabs_fail_as_under_the_hybrid(self, monkeypatch):
        """A non-finite slab is uncertified and goes to the design solve,
        whose SVD does not converge -- the hybrid's loud failure."""
        from epde.operators.common import survival
        G, Gy, design = _banded_designs(m=10)
        G[3, 1, 1] = np.nan

        def bad_design(rows_):
            M, rhs = design(rows_)
            M = M.copy()
            M[np.flatnonzero(rows_ == 3)] = np.nan
            return M, rhs
        with pytest.raises(np.linalg.LinAlgError):
            survival._solve_min_norm(G, Gy, bad_design)


def _grid_library(shape, p=3, seed=0):
    """A weighted random design on ``shape``, flattened C-order."""
    rng = np.random.default_rng(seed)
    n = int(np.prod(shape))
    X = rng.normal(size=(n, p))
    y = X @ rng.normal(size=p) + 0.1 * rng.normal(size=n)
    return X, y, rng.uniform(0.5, 1.5, size=n)


class TestLevelSlabCount:
    """``tile`` and ``het`` take their slab count from the levels of storage
    axis 0, as ``dyad_t`` does, not from the row total. Their slabs are cut at
    whole levels, so on a coarse axis 0 a count read off the row total cut
    them below the ``4 (w + 1)``-row floor."""

    W = 6                       # a block of 4 * (6 + 1) = 28 rows

    @staticmethod
    def _recorded_cuts(monkeypatch):
        """Every slab edge array the block estimators cut from here on."""
        from epde.operators.common import survival
        cuts, slab_edges = [], survival._slab_edges

        def recording(*args):
            edges = slab_edges(*args)
            cuts.append(np.diff(edges).tolist())
            return edges
        monkeypatch.setattr(survival, '_slab_edges', recording)
        return cuts

    @pytest.mark.parametrize('estimator', ['tile_scores', 'heterogeneity_scores'])
    def test_a_coarse_axis0_keeps_whole_blocks(self, estimator, monkeypatch):
        """(13, 3, 3): 117 rows are four blocks of 28 in total, but nine-row
        levels cut four slabs of 27, 27, 27 and 36 rows. Counted in levels
        (four to a block) it is three slabs of 36, 36 and 45."""
        from epde.operators.common import survival
        X, y, w = _grid_library((13, 3, 3))
        cuts = self._recorded_cuts(monkeypatch)
        with using_config(equation_terms_max_number=self.W):
            getattr(survival, estimator)(X, y, w, (13, 3, 3))
        assert cuts == [[36, 36, 45]]

    def test_tile_keeps_its_two_tile_floor(self, monkeypatch):
        """Three 20-row levels hold one 28-row block of whole levels; tile
        still cuts two tiles (20 and 40 rows), which the row floor reports."""
        from epde.operators.common import survival
        X, y, w = _grid_library((3, 20))
        cuts = self._recorded_cuts(monkeypatch)
        with using_config(equation_terms_max_number=self.W):
            survival.tile_scores(X, y, w, (3, 20))
        assert cuts == [[20, 40]]

    def test_het_refuses_fewer_than_three_blocks_of_levels(self):
        """(4, 21) holds 84 rows -- three blocks of 28 in total -- but whole
        21-row levels make only two: het fails loudly instead of cutting
        blocks of 21, 21 and 42 rows."""
        X, y, w = _grid_library((4, 21))
        with using_config(equation_terms_max_number=self.W):
            with pytest.raises(ValueError, match='whole levels'):
                heterogeneity_scores(X, y, w, (4, 21))

    @pytest.mark.parametrize('estimator', ['heterogeneity_scores', 'heterogeneity_raw_scores'])
    def test_one_level_is_cut_flat_and_two_levels_are_refused(self, estimator, monkeypatch):
        """One axis-0 level of 100 rows is cut as one row per level: three flat
        blocks of 33, 33 and 34. Two whole levels make at most two blocks, so
        het raises before cutting; three levels are three blocks of 100."""
        from epde.operators.common import survival
        fn = getattr(survival, estimator)
        cuts = self._recorded_cuts(monkeypatch)
        with using_config(equation_terms_max_number=self.W):
            fn(*_grid_library((1, 100)), (1, 100))
            assert cuts == [[33, 33, 34]]
            cuts.clear()
            with pytest.raises(ValueError, match='whole levels'):
                fn(*_grid_library((2, 100)), (2, 100))
            assert cuts == []
            fn(*_grid_library((3, 100)), (3, 100))
            assert cuts == [[100, 100, 100]]

    @pytest.mark.parametrize('metric, estimator', [('tile', 'tile_scores'),
                                                   ('het', 'heterogeneity_scores'),
                                                   ('het_raw', 'heterogeneity_raw_scores')])
    def test_the_row_floor_predicts_the_estimator(self, metric, estimator, monkeypatch):
        """``row_floor`` passes a grid exactly when the estimator scores it on
        blocks of >= 4 (w + 1) rows -- one axis-0 level (cut flat) and two (at
        most two whole-level blocks) included."""
        from epde.operators.common import survival
        fn = getattr(survival, estimator)
        cuts = self._recorded_cuts(monkeypatch)
        rng = np.random.default_rng(23)
        shapes = [(n0, k) for k in (40, 100) for n0 in (1, 2, 3)] + [
            (int(rng.integers(1, 12)),) + tuple(int(k) for k in rng.integers(1, 40, size=int(rng.integers(1, 3))))
            for _ in range(120)]
        verdicts = {True: 0, False: 0}
        for trial, shape in enumerate(shapes):
            width = int(rng.integers(3, 9))
            cuts.clear()
            with using_config(equation_terms_max_number=width):
                X, y, w = _grid_library(shape, seed=trial)
                try:
                    fn(X, y, w, shape)
                    whole = min(min(cut) for cut in cuts) >= 4 * (_reference_width(X.shape[1]) + 1)
                except ValueError:
                    whole = False
                accepted = survival.row_floor(metric, shape, _reference_width(X.shape[1])) is None
            assert accepted == whole, (shape, width, cuts)
            verdicts[accepted] += 1
        assert min(verdicts.values()) >= 20, verdicts

    def test_every_slab_up_to_the_count_keeps_a_block(self):
        """Any count up to the level count (the ``n_tiles`` / ``n_blocks``
        caps included) cuts every slab at >= ``min_rows`` rows; a flat sample
        count that does not match the grid keeps one row per level."""
        from epde.operators.common import survival
        rng = np.random.default_rng(5)
        for _ in range(300):
            shape = (int(rng.integers(2, 50)),) + tuple(
                int(k) for k in rng.integers(1, 9, size=int(rng.integers(0, 3))))
            n = int(np.prod(shape))
            block = int(rng.integers(4, 60))
            count = survival._level_slab_capacity(n, shape, block)
            for k in range(2, count + 1):
                assert np.diff(survival._slab_edges(n, shape, k)).min() >= block
            assert survival._level_slab_capacity(n + 3, shape, block) == (n + 3) // block

    def test_unchanged_where_the_row_total_kept_whole_blocks(self, monkeypatch):
        """Where the count read off the row total already cut every slab at
        >= 4 (w + 1) rows it IS the level count, so the scores are the old ones
        bit for bit. The old rule is the row total (``n_samples // block``)
        put back in place of the level count."""
        from epde.operators.common import survival
        level_count = survival._level_slab_capacity

        def row_total(n_samples, grid_shape, min_rows):
            return n_samples // min_rows

        def scored(fn, *args):
            try:
                return fn(*args)
            except ValueError:
                return None

        rng = np.random.default_rng(11)
        kept = moved = 0
        for trial in range(60):
            shape = (int(rng.integers(2, 40)),) + tuple(
                int(k) for k in rng.integers(1, 8, size=int(rng.integers(1, 3))))
            n = int(np.prod(shape))
            width = int(rng.integers(3, 9))
            block = 4 * (width + 1)
            X, y, w = _grid_library(shape, seed=trial)
            for name, least, cap in (('tile_scores', 2, survival._DEFAULT_N_TILES),
                                     ('heterogeneity_scores', 3, survival._DEFAULT_N_BLOCKS_HET)):
                old_count = max(least, min(cap, n // block))
                old_whole = (n // block >= least
                             and np.diff(survival._slab_edges(n, shape, old_count)).min() >= block)
                fn = getattr(survival, name)
                with using_config(equation_terms_max_number=width):
                    new = scored(fn, X, y, w, shape)
                    monkeypatch.setattr(survival, '_level_slab_capacity', row_total)
                    old = scored(fn, X, y, w, shape)
                    monkeypatch.setattr(survival, '_level_slab_capacity', level_count)
                if old_whole:
                    assert new is not None and np.array_equal(new, old), (name, shape, width)
                    kept += 1
                elif old is not None and (new is None or not np.array_equal(new, old)):
                    moved += 1
        assert kept >= 30 and moved >= 5, (kept, moved)
