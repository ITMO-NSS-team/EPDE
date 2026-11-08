"""One instability statistic drives both the Pareto axis and the L1 threshold.

``PhysicsInformedLasso.fit`` scales each active column's L1 penalty by a
per-term instability score (``active_thresholds = score * max_corr``). That
score and the ``Instability`` objective must be the SAME statistic: a search
that prunes by one measure and then ranks the survivors by another is
optimizing against its own regularizer. ``objectives.instability_metric``
therefore selects both, and the Gram is built only when the selected estimator
is one that needs it.
"""

import numpy as np
import pytest

from conftest import using_config
from epde.interface.search_config import load_search_config
from epde.operators.common import sparsity as sparsity_mod
from epde.operators.common.sparsity import (_KEEP_RULE_ESTIMATORS,
                                            PhysicsInformedLasso)
from epde.operators.common.survival import chi2_scores

VALID_METRICS = ('vcoef', 'cv', 'survival', 'tile', 'het', 'chi2')


@pytest.fixture
def problem():
    """u_tt = -u on a 120-point grid, plus a hitchhiker column."""
    t = np.linspace(0, 4 * np.pi, 120)
    u = np.sin(t) + 0.3 * np.cos(t)
    X = np.column_stack([u, np.cos(3 * t), t / t.max()])
    y = -u
    return X, y, np.ones(t.size), (t.size,)


class TestOneStatisticBothSides:

    def test_every_selectable_metric_can_score_the_threshold(self):
        """No metric may be selectable for the objective but unscorable for
        the keep-rule -- that combination is what 'same metric' forbids."""
        for metric in VALID_METRICS:
            assert (metric in _KEEP_RULE_ESTIMATORS
                    or metric in ('vcoef', 'cv')), metric

    def test_chi2_threshold_is_the_canonical_estimator(self, problem, monkeypatch):
        """Delegation, asserted on the CALL rather than the values: a dead
        column's score is a ratio of two near-zero quantities (its own signal
        energy collapses with its fitted coefficient), so the last ulp of a
        ~1e-17 coefficient moves it -- and that shifts with array layout, not
        with the statistic. What must hold is that the keep-rule calls the
        objective's estimator, on the active columns, with the intercept
        handled explicitly."""
        X, y, sw, grid_shape = problem
        seen = {}

        def spy(features, target, sample_weights, gshape, fit_intercept=True):
            seen['features'] = np.array(features)
            seen['fit_intercept'] = fit_intercept
            return chi2_scores(features, target, sample_weights, gshape,
                               fit_intercept)

        monkeypatch.setitem(_KEEP_RULE_ESTIMATORS, 'chi2', spy)
        estimator = PhysicsInformedLasso()
        active = np.ones(X.shape[1] + 1, dtype=bool)
        estimator._keep_rule_scores('chi2', None, None, X, y, sw, grid_shape,
                                    active, X.shape[1])
        assert seen['fit_intercept'] is False
        assert np.array_equal(seen['features'],
                              np.hstack([X, np.ones((X.shape[0], 1))]))

    def test_a_dropped_column_is_not_scored(self, problem, monkeypatch):
        X, y, sw, grid_shape = problem
        seen = {}

        def spy(features, target, sample_weights, gshape, fit_intercept=True):
            seen['features'] = np.array(features)
            return chi2_scores(features, target, sample_weights, gshape,
                               fit_intercept)

        monkeypatch.setitem(_KEEP_RULE_ESTIMATORS, 'chi2', spy)
        active = np.array([True, False, True, True])       # column 1 is gone
        PhysicsInformedLasso()._keep_rule_scores(
            'chi2', None, None, X, y, sw, grid_shape, active, X.shape[1])
        assert np.array_equal(seen['features'],
                              np.hstack([X[:, [0, 2]],
                                         np.ones((X.shape[0], 1))]))

    def test_the_score_covers_every_active_column(self, problem):
        """Including the intercept: the caller indexes this by position within
        the active mask, so a short vector silently misaligns the thresholds."""
        X, y, sw, grid_shape = problem
        estimator = PhysicsInformedLasso()
        for active in (np.ones(X.shape[1] + 1, dtype=bool),
                       np.array([True, False, True, True]),
                       np.array([True, True, True, False])):   # no intercept
            got = estimator._keep_rule_scores('chi2', None, None, X, y, sw,
                                              grid_shape, active, X.shape[1])
            assert len(got) == int(active.sum()), active

    def test_vcoef_reads_the_gram_setup(self, problem):
        X, y, sw, grid_shape = problem
        estimator = PhysicsInformedLasso()
        active = np.ones(X.shape[1] + 1, dtype=bool)

        class Spy:
            def score(self, mask):
                return np.arange(int(mask.sum()), dtype=float)

        got = estimator._keep_rule_scores('vcoef', Spy(), None, X, y, sw,
                                          grid_shape, active, X.shape[1])
        assert np.array_equal(got, np.arange(4, dtype=float))

    def test_cv_reduces_the_window_stack(self, problem):
        X, y, sw, grid_shape = problem
        estimator = PhysicsInformedLasso()
        weights = np.array([[1.0, 2.0], [1.0, 4.0]])
        got = estimator._keep_rule_scores('cv', None, weights, X, y, sw,
                                          grid_shape,
                                          np.ones(X.shape[1] + 1, dtype=bool),
                                          X.shape[1])
        assert np.array_equal(got, estimator.get_cv(weights))


class TestGramIsBuiltOnlyWhenNeeded:

    def _spy_setups(self, monkeypatch):
        built = []
        for name in ('GramSetup', 'VaryingCoefSetup'):
            def make(name):
                def boom(*args, **kwargs):
                    built.append(name)
                    raise AssertionError('%s must not be built' % name)
                return boom
            monkeypatch.setattr(sparsity_mod, name, make(name))
        return built

    def test_the_default_builds_no_setup(self, problem, monkeypatch):
        """chi2 scores straight from the active columns, so neither the
        varying-coefficient mode solve nor the sliding-window stack is built."""
        self._spy_setups(monkeypatch)
        X, y, sw, grid_shape = problem
        with using_config(instability_metric=None) as cfg:
            assert cfg.objectives.gram_mode is None
            PhysicsInformedLasso().fit(X, y, grid_shape=grid_shape,
                                       sample_weights=sw)  # must not raise

    @pytest.mark.parametrize('metric', ['het', 'tile', 'dyad', 'survival'])
    def test_the_other_basis_free_metrics_build_nothing_either(
            self, metric, problem, monkeypatch):
        self._spy_setups(monkeypatch)
        X, y, sw, grid_shape = problem
        with using_config(instability_metric=metric) as cfg:
            assert cfg.objectives.gram_mode is None
            PhysicsInformedLasso().fit(X, y, grid_shape=grid_shape,
                                       sample_weights=sw)

    def test_explicit_vcoef_still_builds_the_vc_machinery(self, problem):
        X, y, sw, grid_shape = problem
        with using_config(instability_metric='vcoef') as cfg:
            assert cfg.objectives.gram_mode == 'vcoef'
            estimator = PhysicsInformedLasso().fit(X, y, grid_shape=grid_shape,
                                                   sample_weights=sw)
        assert estimator.cached_vc_score_ is not None

    def test_explicit_cv_still_builds_the_window_stack(self, problem):
        X, y, sw, grid_shape = problem
        with using_config(instability_metric='cv') as cfg:
            assert cfg.objectives.gram_mode == 'axis'
            estimator = PhysicsInformedLasso().fit(X, y, grid_shape=grid_shape,
                                                   sample_weights=sw)
        assert estimator.cached_weights_ is not None


def _probe_library(*names):
    """``u = sin 2t + 0.3 t`` on the 117 inner points of linspace(0, 8, 119),
    its exact derivative as the target, and the named columns of the library
    u, x_0, x_0^2, u*x_0."""
    t = np.linspace(0, 8, 119)[1:-1]
    u = np.sin(2 * t) + 0.3 * t
    columns = {'u': u, 'x_0': t, 'x_0^2': t ** 2, 'u*x_0': u * t}
    return (np.column_stack([columns[name] for name in names]),
            2 * np.cos(2 * t) + 0.3)


def _stub_scores(monkeypatch, feature_scores, intercept_score):
    """Fixed per-column scores: these tests check what the CD/RFE loop does
    with a score, not how the score is computed."""
    full = np.append(np.asarray(feature_scores, dtype=float), intercept_score)

    def scores(self, metric, gram_setup, weights, X, y, sw, grid_shape,
               active_mask, n_features):
        return full[active_mask]

    monkeypatch.setattr(PhysicsInformedLasso, '_keep_rule_scores', scores)


def _ols(X, y, support, intercept):
    """Unpenalized least squares on ``support`` (plus the ones column when
    ``intercept``), laid out like ``full_coef_``."""
    cols = [X[:, j] for j in support] + ([np.ones(X.shape[0])] if intercept else [])
    beta = np.linalg.lstsq(np.column_stack(cols), y, rcond=None)[0]
    full = np.zeros(X.shape[1] + 1)
    full[list(support) + ([X.shape[1]] if intercept else [])] = beta
    return full


def _fit_probe(X, y):
    n = X.shape[0]
    return PhysicsInformedLasso().fit(X, y, grid_shape=(n,),
                                      sample_weights=np.ones(n))


class TestTheEliminationLoop:
    """What the CD/RFE loop does with a score, for every metric (the score is
    stubbed): a column whose threshold reaches its correlation is zeroed, the
    loop restarts on the smaller set, and the survivors carry the unpenalized
    refit (relaxed-refit normal equations against lstsq, hence 1e-9)."""

    def test_a_kill_beside_another_feature_leaves_the_refit(self, monkeypatch):
        _stub_scores(monkeypatch, [1e3, 0.0], 0.0)
        X, du = _probe_library('u', 'x_0')
        np.testing.assert_allclose(_fit_probe(X, du).full_coef_,
                                   _ols(X, du, [1], intercept=True), rtol=1e-9)

    def test_two_survivors_carry_the_refit(self, monkeypatch):
        _stub_scores(monkeypatch, [1e3, 1e3, 0.0, 0.0], 0.0)
        X, du = _probe_library('u', 'x_0', 'x_0^2', 'u*x_0')
        np.testing.assert_allclose(_fit_probe(X, du).full_coef_,
                                   _ols(X, du, [2, 3], intercept=True), rtol=1e-9)

    @pytest.mark.parametrize('score', [1.5, 1e3])
    def test_the_last_feature_column_has_no_exemption(self, monkeypatch, score):
        """The rule may end a fit with no term: the last column standing faces
        the same threshold as any other. That fit reaches the host, which
        declines it, and the population operators redraw the equation
        (``regenerate_degenerate_equations``). A last-feature exemption was
        tried and dropped: it let one-term junk through."""
        _stub_scores(monkeypatch, [score], 1e3)
        X, du = _probe_library('x_0')
        assert not np.any(_fit_probe(X, du).full_coef_[:-1])


class TestNoExactFitFloor:
    """The floor returned all-zero scores for a machine-precision fit. As a
    diagnostic that reads 'nothing left to test'; as a threshold it is a TIE
    that switches sparsity off entirely -- 0/5 vs 5/5 recoveries on the ODE
    benchmark."""

    def test_an_exact_fit_still_ranks_the_columns(self):
        t = np.linspace(0, 4 * np.pi, 120)
        u = np.sin(t)
        # y is EXACTLY -1 * u: the first column fits to machine precision and
        # the second is a dead passenger.
        X = np.column_stack([u, np.cos(3 * t)])
        y = -u
        scores = chi2_scores(X, y, np.ones(t.size), (t.size,),
                             fit_intercept=False)
        assert not np.all(scores == 0), 'the floor is back: every score tied'
        assert scores[1] > scores[0], 'the dead column must score worse'

    def test_a_true_term_still_scores_negligibly(self):
        t = np.linspace(0, 4 * np.pi, 120)
        u = np.sin(t)
        scores = chi2_scores(u[:, None], -u, np.ones(t.size), (t.size,),
                             fit_intercept=False)
        assert scores[0] < 1e-12, scores
