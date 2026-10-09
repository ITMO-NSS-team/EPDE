"""The regularized-intercept scoring rule, and the reductions around it.

RULE 1. The Instability objective charges the intercept its own instability
score iff the intercept survived regularization (``weights_internal[-1] != 0``)
and charges nothing for it when the sparsity step drove it to zero. The
keep-rule scores and prunes the intercept for every metric regardless.

Before this rule only ``vcoef`` and ``cv`` followed it. The six basis-free
estimators fitted the intercept and then dropped its entry, so their objective
discarded a score the keep-rule had just computed for the same column -- and
``chi2_centered`` scored the keep-rule's ones column 0.0, so its intercept had
an L1 threshold of zero and could never be pruned. The objective now calls the
basis-free estimators in the keep-rule's own call shape (explicit ones column,
``fit_intercept=False``), so the two sides are one formula.

Also pinned here: ``chi2_centered``'s exact constancy test and its intercept
functional, the cap-absorbing reductions (a capped term used to surface as
``inf`` or as half the cap), and the basis-free memo's intercept key.
"""
import numpy as np
import pytest

from conftest import using_config
from epde.operators.common import objectives as objectives_mod
from epde.operators.common.objectives import (FitContext, Instability,
                                              _mean_of_sums)
from epde.operators.common.sparsity import (PhysicsInformedLasso,
                                            instability_scores)
from epde.operators.common.stability import (calculate_weights, cv_scores,
                                             vc_stability_total_lr)
from epde.operators.common.survival import (_BASIS_FREE_METRICS,
                                            chi2_centered_scores, chi2_scores)

GRID = (40, 30)
N = GRID[0] * GRID[1]
CAP = np.finfo(float).max
BASIS_FREE = sorted(_BASIS_FREE_METRICS)


def _fields():
    nt, nx = GRID
    t = np.linspace(0, 2, nt)
    x = np.linspace(0, 2 * np.pi, nx, endpoint=False)
    T, Xg = np.meshgrid(t, x, indexing='ij')
    u = np.exp(-0.3 * T) * np.sin(Xg) + 0.5 * np.exp(-0.1 * T) * np.sin(2 * Xg)
    ux = np.exp(-0.3 * T) * np.cos(Xg) + np.exp(-0.1 * T) * np.cos(2 * Xg)
    uxx = -np.exp(-0.3 * T) * np.sin(Xg) - 2.0 * np.exp(-0.1 * T) * np.sin(2 * Xg)
    return T.reshape(-1), Xg.reshape(-1), u.reshape(-1), ux.reshape(-1), uxx.reshape(-1)


def _library(offset=0.0, column_mean=0.0, offset_drift=0.0):
    """``[u, u_x, u_xx]`` with a gently DRIFTING ``u_xx`` coefficient, so every
    score is resolved rather than a round-off ratio while the keep-rule still
    keeps the three true terms, plus an optional genuine constant ``offset``
    in the target (drifting in t by ``offset_drift``, which makes the
    intercept's own score resolved for every estimator) and a mean carried by
    column 0."""
    T, _, u, ux, uxx = _fields()
    X = np.column_stack([u + column_mean, ux, uxx])
    y = (0.5 * u - ux + (1.0 + 0.01 * (T - 1.0)) * uxx
         + offset + offset_drift * (T - 1.0))
    return X, y, np.ones(N)


# The intercept-bearing library the rule-1 tests score.
_RULE_ONE_LIBRARY = dict(offset=0.7, column_mean=1.5, offset_drift=0.05)
# The keep-rule slices its active columns by fancy indexing, which hands the
# estimator an F-ordered copy; on a block-refit estimator that layout change
# alone moves the sum by up to ~1e-9 relative (het on this design). The
# intercept entry this rule adds is many orders larger.
_LAYOUT_RTOL = 1e-8


def _with_ones(X, value=1.0):
    return np.hstack([X, np.full((X.shape[0], 1), value)])


class _Eq:
    """What ``Instability.compute`` reads off an equation, and nothing else."""

    def __init__(self, feats, tgts, intercept):
        self._feats, self._tgts = feats, tgts
        nf = next(iter(feats.values())).shape[1]
        self.weights_internal = np.r_[np.ones(nf), 0.7 if intercept else 0.0]
        self.main_var_to_explain = 'u'
        self._cached_vc_score = None
        self._cached_sw_weights = None
        self._cached_alt_instability = None

    def evaluate(self, active_only=False):
        return self._tgts, self._feats


def _ctx(keys=(0,), grid=GRID):
    # ``Domain.inner_shape`` hands the grid over as an ndarray.
    return FitContext(g_fun_vals={k: np.ones(int(np.prod(grid))) for k in keys},
                      data_shape={k: np.asarray(grid) for k in keys}, penalty_coeff=0.2)


def _objective(metric, X, y, intercept):
    with using_config(instability_metric=metric):
        return Instability().compute(_Eq({0: X}, {0: y}, intercept), _ctx())


# --------------------------------------------------------------------------- #
#  Rule 1 on the basis-free estimators                                         #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize('metric', BASIS_FREE)
class TestRuleOneBasisFree:

    def test_a_surviving_intercept_is_charged_the_keep_rule_score(self, metric):
        """The objective is the SUM of the keep-rule's own per-term vector,
        intercept entry included -- one formula, both sides."""
        X, y, w = _library(**_RULE_ONE_LIBRARY)
        active = np.ones(X.shape[1] + 1, dtype=bool)
        with using_config(instability_metric=metric):
            keep = np.asarray(instability_scores(metric, X, y, w, GRID, active,
                                                 X.shape[1]), dtype=float)
        assert keep.shape == (X.shape[1] + 1,)
        assert keep[-1] > 0.0, 'fixture must give the intercept a resolved score'
        value = _objective(metric, X, y, intercept=True)
        assert value == pytest.approx(float(np.sum(keep)), rel=_LAYOUT_RTOL, abs=0.0)

    def test_the_intercept_entry_is_not_dropped(self, metric):
        """Bitwise the keep-rule call shape on the objective's own matrix:
        explicit ones column, ``fit_intercept=False``, every entry summed."""
        X, y, w = _library(**_RULE_ONE_LIBRARY)
        full = np.asarray(_BASIS_FREE_METRICS[metric](
            _with_ones(X), y, w, GRID, fit_intercept=False), dtype=float)
        value = _objective(metric, X, y, intercept=True)
        assert value == float(np.sum(full))

    def test_a_regularized_intercept_is_not_charged(self, metric):
        """Intercept driven to zero: features only, no ones column at all --
        the keep-rule's view of a mask whose intercept slot is off."""
        X, y, w = _library(**_RULE_ONE_LIBRARY)
        mask = np.ones(X.shape[1] + 1, dtype=bool)
        mask[-1] = False
        with using_config(instability_metric=metric):
            keep = np.asarray(instability_scores(metric, X, y, w, GRID, mask,
                                                 X.shape[1]), dtype=float)
        assert keep.shape == (X.shape[1],)
        bare = np.asarray(_BASIS_FREE_METRICS[metric](
            X, y, w, GRID, fit_intercept=False), dtype=float)
        value = _objective(metric, X, y, intercept=False)
        assert value == float(np.sum(bare))
        assert value == pytest.approx(float(np.sum(keep)), rel=_LAYOUT_RTOL, abs=0.0)


# --------------------------------------------------------------------------- #
#  Rule 1 on vcoef / cv -- they already followed it; pin that                  #
# --------------------------------------------------------------------------- #
class TestRuleOneGramMetrics:

    @pytest.mark.parametrize('intercept', [True, False])
    def test_vcoef_cold_path_scores_the_intercept_iff_it_survived(self, intercept):
        X, y, w = _library(offset=0.7, column_mean=1.5)
        expected = vc_stability_total_lr(X, y, w, GRID, main_var='u',
                                         fit_intercept=intercept, sample_key=0)
        other = vc_stability_total_lr(X, y, w, GRID, main_var='u',
                                      fit_intercept=not intercept, sample_key=0)
        value = _objective('vcoef', X, y, intercept)
        assert value == expected
        assert value != other

    def test_vcoef_fast_path_sums_the_converged_mask_vector(self):
        """The sparsity cache already holds the intercept's entry iff its
        slot survived (``score(active_mask)``); the objective sums it whole."""
        X, y, _ = _library()
        eq = _Eq({0: X}, {0: y}, intercept=True)
        eq._cached_vc_score = {0: np.array([0.25, 0.5, 2.0])}
        with using_config(instability_metric='vcoef'):
            assert Instability().compute(eq, _ctx()) == 2.75

    @pytest.mark.parametrize('intercept', [True, False])
    def test_cv_cold_path_scores_the_intercept_iff_it_survived(self, intercept):
        X, y, w = _library(offset=0.7, column_mean=1.5)
        vec = np.nan_to_num(cv_scores(calculate_weights(
            X, y, w, GRID, fit_intercept=intercept)))
        assert vec.size == X.shape[1] + int(intercept)
        value = _objective('cv', X, y, intercept)
        assert value == float(np.sum(vec)) / len(GRID)


# --------------------------------------------------------------------------- #
#  chi2_centered: exact constancy test and a real intercept score               #
# --------------------------------------------------------------------------- #
class TestChi2CenteredIntercept:

    @pytest.mark.parametrize('value', [0.1, 3.7, 1e-8, 1e-20, 1.0, 1e6, -2.5])
    def test_every_exact_constant_is_recognised(self, value):
        """The old round-off test ``spread > eps*max(|mu|, 1)`` missed 0.1, 3.7
        and 1e-8 and centred them into garbage. A constant column's score is
        the intercept functional, which is blind to the constant's value."""
        X, y, w = _library(offset=0.7)
        ref = chi2_centered_scores(_with_ones(X), y, w, GRID, fit_intercept=False)
        got = chi2_centered_scores(_with_ones(X, value), y, w, GRID,
                                   fit_intercept=False)
        assert np.isfinite(got).all()
        assert got[-1] == pytest.approx(ref[-1], rel=1e-9)
        assert np.allclose(got[:-1], ref[:-1], rtol=1e-12, atol=0.0)

    @pytest.mark.parametrize('scale', [1e-17, 1e-12, 1e8])
    def test_a_tiny_non_constant_column_is_not_declared_constant(self, scale):
        """Magnitude is not constancy: chi2 is column-scale invariant, and so
        must its centred variant be, however small the column."""
        X, y, w = _library()
        ref = chi2_centered_scores(X, y, w, GRID, fit_intercept=False)
        Xs = X.copy()
        Xs[:, 2] *= scale
        got = chi2_centered_scores(Xs, y, w, GRID, fit_intercept=False)
        assert got[2] > 0.0
        assert np.allclose(got, ref, rtol=1e-6, atol=0.0)

    @pytest.mark.parametrize('weights', ['unit', 'scaled', 'bump'])
    def test_the_constant_entry_is_the_chi2_intercept_functional(self, weights):
        """Centring is an implicit intercept (Frisch-Waugh-Lovell), so the
        centred residual IS the uncentred fit-with-intercept residual, and the
        ones column's score is exactly the entry ``chi2_scores`` returns for
        it. Non-unit weights used to slip the ones column past the old test
        (scores 1e28-3e29 instead of the intended value)."""
        X, y, w = _library(offset=0.7, column_mean=1.5)
        T, Xg, *_ = _fields()
        w = {'unit': w, 'scaled': 0.1 * w,
             'bump': 0.5 + np.exp(-((Xg - np.pi) ** 2))}[weights]
        Xa = _with_ones(X)
        uncentred = chi2_scores(Xa, y, w, GRID, fit_intercept=False)
        centred = chi2_centered_scores(Xa, y, w, GRID, fit_intercept=False)
        assert np.isfinite(centred[-1]) and centred[-1] > 0.0
        assert centred[-1] == pytest.approx(uncentred[-1], rel=1e-8)

    def test_collinear_constants_tie_on_the_single_constant_score(self):
        """Several constant columns cannot split the offset identifiably; each
        is scored as the sole carrier of it, so they tie."""
        X, y, w = _library(offset=0.7)
        single = chi2_centered_scores(_with_ones(X), y, w, GRID, fit_intercept=False)
        both = chi2_centered_scores(_with_ones(_with_ones(X), 3.7), y, w, GRID,
                                    fit_intercept=False)
        assert both[-1] == pytest.approx(both[-2], rel=1e-12)
        assert both[-1] == pytest.approx(single[-1], rel=1e-9)

    def test_a_no_constant_library_prunes_the_intercept(self):
        """The keep-rule's ones column used to score 0.0 -> L1 threshold 0 ->
        a round-off intercept (|c0| 6e-33..1.6e-17) survived every fit."""
        X, y, w = _library()
        with using_config(instability_metric='chi2_centered'):
            est = PhysicsInformedLasso().fit(X, y, grid_shape=GRID, sample_weights=w)
        assert est.full_coef_[-1] == 0.0
        assert np.all(est.full_coef_[:-1] != 0.0)

    def test_a_genuine_constant_is_kept_and_scored_finitely(self):
        X, y, w = _library(offset=0.7)
        with using_config(instability_metric='chi2_centered'):
            est = PhysicsInformedLasso().fit(X, y, grid_shape=GRID, sample_weights=w)
            assert est.full_coef_[-1] == pytest.approx(0.7, rel=1e-6)
            score = instability_scores('chi2_centered', X, y, w, GRID,
                                       np.ones(X.shape[1] + 1, dtype=bool), X.shape[1])
        assert np.isfinite(score[-1]) and score[-1] > 0.0


# --------------------------------------------------------------------------- #
#  Cap-absorbing reductions (A6)                                               #
# --------------------------------------------------------------------------- #
class TestCapAbsorbingReduction:

    @pytest.mark.parametrize('per_sample', [
        {0: np.array([CAP, CAP])},                       # overflowed to inf
        {0: np.array([CAP]), 1: np.array([1.0])},        # was CAP / 2
        {0: np.array([CAP]), 1: np.array([CAP])},        # mean overflowed
        {0: np.array([np.inf, 1.0])},
        {0: np.array([np.nan, 1.0])},
        {0: np.array([1e308, 1e308])},                   # finite terms, sum overflows
    ])
    def test_a_capped_term_caps_the_equation(self, per_sample):
        assert _mean_of_sums(per_sample) == CAP

    def test_uncapped_values_reduce_as_before(self):
        per_sample = {0: np.array([0.1, 0.2, 0.3]), 1: np.array([1e-3]),
                      2: None, 3: np.zeros(0)}
        expected = float(np.mean([float(np.sum(v)) for v in per_sample.values()
                                  if v is not None]))
        assert _mean_of_sums(per_sample) == expected
        assert _mean_of_sums({0: None}) == 0.0

    @pytest.mark.parametrize('vectors', [
        {0: [CAP, CAP]},
        {0: [CAP], 1: [1.0]},
        {0: [np.inf]},
    ])
    def test_basis_free_objective_absorbs_the_cap(self, vectors, monkeypatch):
        calls = iter([np.asarray(v, dtype=float) for v in vectors.values()])
        monkeypatch.setitem(_BASIS_FREE_METRICS, 'het_raw',
                            lambda *a, **k: next(calls))
        X, y, _ = _library()
        keys = tuple(vectors)
        eq = _Eq({k: X for k in keys}, {k: y for k in keys}, intercept=False)
        with using_config(instability_metric='het_raw'):
            assert Instability().compute(eq, _ctx(keys)) == CAP

    @pytest.mark.parametrize('totals', [{0: np.inf}, {0: CAP, 1: 1.0},
                                        {0: CAP, 1: CAP}])
    def test_vcoef_cold_objective_absorbs_the_cap(self, totals, monkeypatch):
        calls = iter(totals.values())
        monkeypatch.setattr(objectives_mod, 'vc_stability_total_lr',
                            lambda *a, **k: next(calls))
        X, y, _ = _library()
        keys = tuple(totals)
        eq = _Eq({k: X for k in keys}, {k: y for k in keys}, intercept=True)
        with using_config(instability_metric='vcoef'):
            assert Instability().compute(eq, _ctx(keys)) == CAP

    def test_vcoef_fast_path_absorbs_the_cap(self):
        X, y, _ = _library()
        eq = _Eq({0: X, 1: X}, {0: y, 1: y}, intercept=True)
        eq._cached_vc_score = {0: np.array([CAP, 0.1]), 1: np.array([0.2])}
        with using_config(instability_metric='vcoef'):
            assert Instability().compute(eq, _ctx((0, 1))) == CAP

    def test_cv_objective_absorbs_the_cap_before_the_axis_division(self, monkeypatch):
        """A capped trajectory is CAP, not CAP / n_dims."""
        monkeypatch.setattr(objectives_mod, 'cv_scores',
                            lambda weights: np.array([CAP, 1.0, 0.5, 0.25]))
        X, y, _ = _library()
        eq = _Eq({0: X}, {0: y}, intercept=True)
        with using_config(instability_metric='cv'):
            assert Instability().compute(eq, _ctx()) == CAP

    def test_two_unresolved_het_raw_terms_cap_rather_than_overflow(self):
        """The live shape of the bug: het_raw's unresolved supremum is the
        cap, and two of them on one trajectory summed to ``inf``."""
        X, y, w = _library()
        dead = np.zeros(N)
        dead[:3] = 1e-9                               # barely excited -> unresolved
        Xh = np.column_stack([X, dead, np.roll(dead, 600)])
        per_term = _BASIS_FREE_METRICS['het_raw'](Xh, y, w, GRID, fit_intercept=False)
        assert np.sum(per_term >= CAP) == 2, per_term
        assert _objective('het_raw', Xh, y, intercept=False) == CAP


# --------------------------------------------------------------------------- #
#  The basis-free memo is keyed by the intercept flag                          #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize('metric', ['chi2', 'tile'])
def test_the_memo_does_not_serve_a_value_scored_under_the_other_intercept_flag(metric):
    X, y, _ = _library(offset=0.7, column_mean=1.5)
    eq = _Eq({0: X}, {0: y}, intercept=True)
    with using_config(instability_metric=metric):
        with_icpt = Instability().compute(eq, _ctx())
        eq.weights_internal[-1] = 0.0                 # no reset: the memo survives
        stale_or_fresh = Instability().compute(eq, _ctx())
        fresh = Instability().compute(_Eq({0: X}, {0: y}, intercept=False), _ctx())
    assert with_icpt != fresh
    assert stale_or_fresh == fresh


def test_a_memo_in_the_old_layout_is_rescored_not_misread():
    # The pre-rule memo was (metric, value); a value of 0.0 compares equal to
    # fit_intercept=False, so a key test on the first two entries alone would
    # pass and then index a third entry that is not there.
    X, y, _ = _library(offset=0.7, column_mean=1.5)
    eq = _Eq({0: X}, {0: y}, intercept=False)
    eq._cached_alt_instability = ('chi2', 0.0)
    with using_config(instability_metric='chi2'):
        got = Instability().compute(eq, _ctx())
        fresh = Instability().compute(_Eq({0: X}, {0: y}, intercept=False), _ctx())
    assert got == fresh
