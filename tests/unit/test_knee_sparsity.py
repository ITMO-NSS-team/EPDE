"""The knee sparsity operator: selection by fit geometry, not by a threshold.

``sparsity_cls='knee'`` replaces ``PhysicsInformedLasso``'s shrinkage rule
(kill feature *j* when ``|rho_j| <= score_j * max_corr``) with an exhaustive
subset search over the weighted Gram, scored by the scree-knee family in
``epde.operators.common.subset_selection``.

Two properties carry the design and are pinned here:

* the RULES are self-calibrated -- every condition in ``extend2`` compares this
  equation's own curve against itself, so there is no constant to tune and no
  estimator-dependent scale;
* the OUTPUT obeys the same coefficient contract as every other sparsity
  producer (``Equation._validate_weight_layout``), including the intercept
  being an ordinary selectable column that can end up exactly zero. Three
  readers depend on that -- ``Instability.compute``'s intercept rule,
  ``Complexity('terms')`` and ``amplification_ratio`` -- so an intercept that
  were non-zero by construction would quietly lie to all three.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from conftest import using_config

import epde.globals as global_var
from epde.interface.search_config import (load_search_config, resolve_sparsity,
                                          sparsity_settings,
                                          validate_sparsity_kwargs)
from epde.operators.common.sparsity import (VWSRSparsity,
                                            build_sparsity_operator,
                                            initial_sparsity_interval)
from epde.operators.common.objectives import Discrepancy, FitContext
from epde.operators.common.subset_selection import (KneeSparsity, REALIZATIONS,
                                                    SOLVER_FREE_DISCREPANCIES,
                                                    build_chain,
                                                    discrepancy_value, extend2,
                                                    extend2_stab, greedy_chain,
                                                    knee_size, mask_of,
                                                    select_from_chain,
                                                    select_support,
                                                    subset_coefficients,
                                                    subset_table)
from epde.structure.main_structures import Equation


# ---------------------------------------------------------------------------
# The chain, the elbow and the extension
# ---------------------------------------------------------------------------
class TestChainArithmetic:
    """``extend2``'s four exit conditions, each isolated.

    The base chain below is built so that the step from size 1 to size 2 is
    accepted and the step from 2 to 3 is refused on the machine floor; each
    other test then breaks exactly one condition and asserts the walk stops one
    size earlier. Passing ``d`` explicitly (rather than deriving it from
    ``best_rss``) is what makes the conditions separable -- otherwise a change
    meant to break one of them moves all four.
    """

    #: nested supports: {}, {0}, {0,1}, {0,1,2}, {0,1,2,3}
    BEST_SUB = np.array([0b0000, 0b0001, 0b0011, 0b0111, 0b1111])
    #: cumulative log-RSS 0 -> 1 -> 7 -> 9 -> 9.5, i.e. the drops in ``D``.
    #: The last size lands on the global minimum -- the machine-floor case.
    BEST_RSS = np.exp(-np.array([0.0, 1.0, 7.0, 9.0, 9.5]))
    D = np.array([1.0, 6.0, 2.0, 0.01])
    N_COLS = 4
    ELBOW = 2

    def test_knee_size_is_the_drop_then_flat_elbow(self):
        # drops 3.0, 2.0, then flat: the curve turns after the second column.
        assert knee_size(np.array([3.0, 2.0, 0.1, 0.05])) == 2
        # a single dominant drop puts the elbow at size 1.
        assert knee_size(np.array([5.0, 0.1, 0.05])) == 1
        assert knee_size(self.D) == self.ELBOW

    def test_base_chain_extends_once_then_hits_the_floor(self):
        assert extend2(self.D, self.BEST_RSS, self.BEST_SUB,
                       self.N_COLS, self.ELBOW) == 3

    def test_a_non_nested_next_subset_is_not_an_extension(self):
        best_sub = self.BEST_SUB.copy()
        best_sub[3] = 0b1101                    # drops column 1: not nested
        assert extend2(self.D, self.BEST_RSS, best_sub, self.N_COLS,
                       self.ELBOW) == 2

    def test_landing_on_the_machine_floor_is_not_evidence(self):
        best_rss = self.BEST_RSS.copy()
        best_rss[3] = best_rss.min()            # an exact fit
        assert extend2(self.D, best_rss, self.BEST_SUB, self.N_COLS,
                       self.ELBOW) == 2

    def test_a_drop_the_tail_could_match_is_refused(self):
        d = np.array([1.0, 6.0, 2.0, 3.0])      # tail beats the candidate
        assert extend2(d, self.BEST_RSS, self.BEST_SUB, self.N_COLS,
                       self.ELBOW) == 2

    def test_a_drop_weaker_than_the_weakest_accepted_is_refused(self):
        d = np.array([3.0, 6.0, 2.0, 0.01])     # 2.0 < the accepted 3.0
        assert extend2(d, self.BEST_RSS, self.BEST_SUB, self.N_COLS,
                       self.ELBOW) == 2

    def test_knee_alone_never_extends(self):
        chain = (self.BEST_RSS, self.BEST_SUB, self.D)
        plain, _ = select_from_chain('knee', *chain, self.N_COLS)
        extended, _ = select_from_chain('knee_ext2', *chain, self.N_COLS)
        assert plain == int(self.BEST_SUB[self.ELBOW])
        assert extended == int(self.BEST_SUB[self.ELBOW + 1])

    def test_unknown_realization_fails_loud(self):
        with pytest.raises(ValueError, match='Unknown knee realization'):
            select_from_chain('knee_ext9', self.BEST_RSS, self.BEST_SUB,
                              self.D, self.N_COLS)

    def test_ke2_stab_needs_a_stability_callback(self):
        with pytest.raises(ValueError, match='stability callback'):
            select_from_chain('ke2_stab', self.BEST_RSS, self.BEST_SUB,
                              self.D, self.N_COLS)


class TestStabilityVeto:
    """``extend2_stab`` vetoes an extension the geometry would have taken."""

    BEST_SUB = TestChainArithmetic.BEST_SUB
    BEST_RSS = TestChainArithmetic.BEST_RSS
    D = TestChainArithmetic.D
    N_COLS = TestChainArithmetic.N_COLS
    ELBOW = TestChainArithmetic.ELBOW

    def _run(self, scores):
        """The extension under test adds column 2 to the core {0, 1}."""
        return extend2_stab(self.D, self.BEST_RSS, self.BEST_SUB, self.N_COLS,
                            self.ELBOW,
                            lambda mask: np.asarray(scores)[mask])

    def test_a_stable_extension_is_accepted(self):
        subset, vetoed = self._run([1.0, 1.0, 0.1, 0.0])  # the calmest column
        assert subset == int(self.BEST_SUB[self.ELBOW + 1])
        assert vetoed is None

    def test_an_extension_less_stable_than_every_member_is_vetoed(self):
        subset, vetoed = self._run([0.1, 0.1, 9.0, 0.0])  # the worst column
        assert subset == int(self.BEST_SUB[self.ELBOW])
        assert vetoed == 2

    def test_an_unstable_core_does_not_shrink_the_result(self):
        """Fit geometry owns the CORE: stability is asked only about the term
        being ADDED, so a core of wildly unstable members is neither
        re-examined nor pruned -- and it also raises the bar the newcomer is
        measured against, which is the self-calibration working as designed."""
        subset, vetoed = self._run([50.0, 50.0, 0.1, 0.0])
        assert subset == int(self.BEST_SUB[self.ELBOW + 1])
        assert vetoed is None

    def test_a_veto_never_falls_below_the_elbow(self):
        subset, _ = self._run([0.0, 0.0, 9.0, 0.0])
        assert bin(subset).count('1') == self.ELBOW


# ---------------------------------------------------------------------------
# The subset table
# ---------------------------------------------------------------------------
@pytest.fixture
def design():
    """200 points, 3 informative columns and 2 pure junk ones."""
    rng = np.random.default_rng(20260831)
    n = 200
    X = rng.normal(size=(n, 5))
    y = 3.0 * X[:, 0] - 2.0 * X[:, 1] + 0.5 * X[:, 2] + 0.01 * rng.normal(size=n)
    w = rng.uniform(0.5, 1.5, size=n)
    return X, y, w


def _reference_rss(X, y, w, cols, p):
    """Weighted RSS of an explicit lstsq fit on ``cols`` (intercept = col p)."""
    Xa = np.hstack([X, np.ones((X.shape[0], 1))])[:, cols]
    root = np.sqrt(w)
    coef, *_ = np.linalg.lstsq(root[:, None] * Xa, root * y, rcond=None)
    resid = y - Xa @ coef
    return float(resid @ (w * resid))


class TestSubsetTable:

    def test_rss_matches_an_independent_least_squares_fit(self, design):
        X, y, w = design
        rss, _, n_cols = subset_table(X, y, w)
        assert n_cols == X.shape[1] + 1
        for S in (0b000001, 0b100011, 0b111111, 0b010101):
            cols = [j for j in range(n_cols) if (S >> j) & 1]
            assert rss[S] == pytest.approx(
                _reference_rss(X, y, w, cols, X.shape[1]), rel=1e-8)

    def test_the_two_intercept_modes_agree_where_they_overlap(self, design):
        X, y, w = design
        p = X.shape[1]
        rss_sel, amp_sel, _ = subset_table(X, y, w, intercept='selectable')
        rss_alw, amp_alw, n_alw = subset_table(X, y, w, intercept='always')
        assert n_alw == p                      # intercept is not enumerated
        for S in range(1 << p):
            # 'always' subset S is the 'selectable' subset with the intercept.
            assert rss_alw[S] == pytest.approx(rss_sel[S | (1 << p)], rel=1e-12)
            # ...but 'always' leaves the intercept out of the guard ratio.
            assert amp_alw[S] <= amp_sel[S | (1 << p)] + 1e-12

    def test_an_exactly_singular_subset_is_not_a_fake_exact_fit(self):
        """The Cholesky guard. LU raises only on an exactly zero pivot, so a
        duplicated column used to yield garbage coefficients and a negative
        pre-floor RSS that read as a perfect fit."""
        rng = np.random.default_rng(7)
        n = 120
        col = rng.normal(size=n)
        X = np.column_stack([col, col, rng.normal(size=n)])   # cols 0 == 1
        y = 2.0 * col + 0.1 * rng.normal(size=n)
        w = np.ones(n)
        rss, _, n_cols = subset_table(X, y, w)
        singular = 0b00011                       # both copies, no intercept
        one_copy = 0b00001
        floor = np.finfo(float).eps * n * float(y @ y)
        assert rss[singular] > floor * 10        # not a fake exact fit
        assert rss[singular] == pytest.approx(rss[one_copy], rel=1e-8)

    def test_the_empty_model_anchors_the_curve(self, design):
        X, y, w = design
        rss, amp, _ = subset_table(X, y, w)
        assert rss[0] == pytest.approx(float(y @ (w * y)))
        assert amp[0] == 0.0

    def test_amplification_matches_the_production_guard_ratio(self, design):
        X, y, w = design
        p = X.shape[1]
        rss, amp, n_cols = subset_table(X, y, w)
        S = 0b100111
        support = mask_of(S, n_cols)
        coef = subset_coefficients(X, y, w, support)
        Xa = np.hstack([X, np.ones((X.shape[0], 1))])
        # A = sum |c_j| * ||col_j||_W / ||y||_W, the amplification_ratio form.
        norms = np.sqrt((w[:, None] * Xa ** 2).sum(axis=0))
        expected = float(np.abs(coef) @ norms) / np.sqrt(float(y @ (w * y)))
        assert amp[S] == pytest.approx(expected, rel=1e-8)

    def test_rejects_an_unknown_intercept_mode(self, design):
        X, y, w = design
        with pytest.raises(ValueError, match='intercept must be one of'):
            subset_table(X, y, w, intercept='sometimes')


class TestSubsetCoefficients:

    def test_coefficients_are_a_weighted_ols_on_the_support(self, design):
        X, y, w = design
        support = np.array([True, True, False, False, False, False])
        coef = subset_coefficients(X, y, w, support)
        assert np.all(coef[~support] == 0.0)
        Xs = X[:, :2]
        root = np.sqrt(w)
        expected, *_ = np.linalg.lstsq(root[:, None] * Xs, root * y, rcond=None)
        assert coef[:2] == pytest.approx(expected, rel=1e-8)

    def test_the_true_support_is_recovered_with_no_intercept(self, design):
        """The selectable-intercept decision, end to end: a design with no
        constant term must leave the trailing slot exactly zero."""
        X, y, w = design
        rss, amp, n_cols = subset_table(X, y, w)
        for realization in ('knee', 'knee_ext2', 'knee_ext2_amp'):
            subset, _ = select_support(realization, rss, amp, n_cols,
                                       amp_cap=100.0)
            support = mask_of(subset, n_cols)
            assert list(np.flatnonzero(support)) == [0, 1, 2], realization
            assert support[-1] == False, realization


# ---------------------------------------------------------------------------
# The greedy fallback
# ---------------------------------------------------------------------------
class TestGreedyFallback:

    def test_greedy_matches_exhaustive_on_an_orthogonal_design(self):
        """Forward selection is optimal when the columns are orthogonal, so
        the two paths must agree there -- the check that the fallback runs the
        SAME rule rather than a different one."""
        rng = np.random.default_rng(11)
        n = 256
        Q = np.linalg.qr(rng.normal(size=(n, 6)))[0]
        y = 4.0 * Q[:, 0] + 3.0 * Q[:, 1] + 0.001 * rng.normal(size=n)
        w = np.ones(n)

        rss, amp, n_cols = subset_table(Q, y, w)
        exact, _ = select_support('knee_ext2', rss, amp, n_cols)

        best_rss, best_sub, d, n_greedy = greedy_chain(Q, y, w)
        approx, _ = select_from_chain('knee_ext2', best_rss, best_sub, d,
                                      n_greedy)
        assert n_greedy == n_cols
        assert mask_of(approx, n_cols).tolist() == mask_of(exact, n_cols).tolist()

    def test_the_greedy_chain_is_nested_by_construction(self):
        rng = np.random.default_rng(3)
        X = rng.normal(size=(80, 4))
        y = X[:, 0] - X[:, 3]
        _, best_sub, _, _ = greedy_chain(X, y, np.ones(80))
        for k in range(len(best_sub) - 1):
            assert int(best_sub[k + 1]) & int(best_sub[k]) == int(best_sub[k])


# ---------------------------------------------------------------------------
# The operator
# ---------------------------------------------------------------------------
class StubEquation:
    """Minimal Equation stand-in: the surface ``KneeSparsity.apply`` touches.

    ``_validate_weight_layout`` is borrowed from the real class so an
    assignment that breaks the coefficient contract fails here exactly as it
    would in a live search.
    """

    _validate_weight_layout = Equation._validate_weight_layout

    def __init__(self, features, target):
        self._features = features
        self._target = target
        n_features = next(iter(features.values())).shape[1]
        self.structure = [SimpleNamespace(name=i) for i in range(n_features + 1)]
        self.target_idx = 0
        self.main_var_to_explain = 'u'
        self._gram_super = None
        self.weights_internal = self.weights_final = None
        self.weights_internal_evald = self.weights_final_evald = False
        self._cached_sw_weights = 'stale'
        self._cached_vc_score = 'stale'

    def evaluate(self, *, active_only=False):
        return self._target, self._features


@pytest.fixture
def stub_samples(monkeypatch):
    """Install a samples_manager over the trajectories a test declares."""
    def install(shapes):
        manager = SimpleNamespace(
            gFunc=lambda kind: {key: np.ones(int(np.prod(shape)))
                                for key, shape in shapes.items()},
            inner_shapes=dict(shapes))
        # ``raising=False``: the module has no such attribute until a
        # real run installs one.
        monkeypatch.setattr(global_var, 'samples_manager', manager,
                            raising=False)
        return manager
    return install


def _library(seed=0, n=240):
    """3 true columns + 3 junk ones, no constant term."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 6))
    y = 5.0 * X[:, 0] - 4.0 * X[:, 1] + 3.0 * X[:, 2] + 0.005 * rng.normal(size=n)
    return X, y


class TestKneeSparsityOperator:

    @pytest.mark.parametrize('realization', REALIZATIONS)
    def test_recovers_the_true_support_in_every_realization(
            self, realization, stub_samples):
        X, y = _library()
        stub_samples({0: (X.shape[0],)})
        equation = StubEquation({0: X}, {0: y})
        operator = KneeSparsity()
        operator.realization = realization
        with using_config(instability_metric='chi2'):
            operator.apply(equation, {})

        weights = np.asarray(equation.weights_internal)
        assert weights.shape == (len(equation.structure),)
        equation._validate_weight_layout(weights, 'weights_internal')
        assert list(np.flatnonzero(weights)) == [0, 1, 2], realization
        assert weights[-1] == 0.0                      # no constant needed
        assert weights[:3] == pytest.approx([5.0, -4.0, 3.0], abs=2e-3)

    def test_final_and_internal_weights_agree(self, stub_samples):
        """No legacy refit follows, so the magnitudes ARE the internal ones --
        and the marker that requests that refit must stay unset."""
        X, y = _library()
        stub_samples({0: (X.shape[0],)})
        equation = StubEquation({0: X}, {0: y})
        with using_config(instability_metric='chi2'):
            KneeSparsity().apply(equation, {})
        assert np.array_equal(equation.weights_internal, equation.weights_final)
        assert equation.weights_internal_evald is True
        assert equation.weights_final_evald is True
        assert getattr(equation, '_legacy_refit_pending', False) is False

    def test_the_stale_instability_caches_are_always_overwritten(
            self, stub_samples):
        """EqRightPartSelector copies both out of the equation during its term
        sweep, so a skipped assignment would carry the previous candidate
        target's value forward."""
        X, y = _library()
        stub_samples({0: (X.shape[0],)})
        equation = StubEquation({0: X}, {0: y})
        with using_config(instability_metric='chi2'):
            KneeSparsity().apply(equation, {})
        assert equation._cached_sw_weights is None      # basis-free metric
        assert equation._cached_vc_score is None

    def test_a_needed_constant_keeps_the_intercept_slot(self, stub_samples):
        X, y = _library()
        y = y + 40.0                                    # a large offset
        stub_samples({0: (X.shape[0],)})
        equation = StubEquation({0: X}, {0: y})
        with using_config(instability_metric='chi2'):
            KneeSparsity().apply(equation, {})
        weights = np.asarray(equation.weights_internal)
        assert weights[-1] != 0.0
        assert weights[-1] == pytest.approx(40.0, abs=1e-2)

    def test_multisample_averages_the_full_vector(self, stub_samples):
        """Each trajectory converges to its own support; averaging the FULL
        vector keeps every slot aligned and carries the union."""
        X0, y0 = _library(seed=1, n=200)
        X1, y1 = _library(seed=2, n=160)
        y1 = y1 + 2.0 * X1[:, 4]                  # sample 1 needs one more term
        stub_samples({0: (200,), 1: (160,)})
        equation = StubEquation({0: X0, 1: X1}, {0: y0, 1: y1})
        with using_config(instability_metric='chi2'):
            KneeSparsity().apply(equation, {})
        active = list(np.flatnonzero(equation.weights_internal))
        assert active == [0, 1, 2, 4]
        # Present in one of two samples -> half its per-sample magnitude.
        assert equation.weights_internal[4] == pytest.approx(1.0, abs=5e-3)

    def test_non_finite_features_give_an_empty_equation(self, stub_samples):
        X, y = _library()
        X[3, 2] = np.nan
        stub_samples({0: (X.shape[0],)})
        equation = StubEquation({0: X}, {0: y})
        with using_config(instability_metric='chi2'):
            KneeSparsity().apply(equation, {})
        assert np.all(np.asarray(equation.weights_internal) == 0.0)

    def test_no_features_gives_an_empty_equation(self, stub_samples):
        X, y = _library()
        stub_samples({0: (X.shape[0],)})
        equation = StubEquation({0: X}, {0: y})
        equation._features = None
        with using_config(instability_metric='chi2'):
            KneeSparsity().apply(equation, {})
        assert np.all(np.asarray(equation.weights_internal) == 0.0)

    def test_the_greedy_path_is_taken_above_the_column_cap(self, stub_samples):
        X, y = _library()
        stub_samples({0: (X.shape[0],)})
        operator = KneeSparsity()
        operator.max_exhaustive_columns = 3        # 7 columns -> fallback
        exhaustive = KneeSparsity()
        eq_greedy = StubEquation({0: X}, {0: y})
        eq_exact = StubEquation({0: X}, {0: y})
        with using_config(instability_metric='chi2'):
            operator.apply(eq_greedy, {})
            exhaustive.apply(eq_exact, {})
        assert list(np.flatnonzero(eq_greedy.weights_internal)) == [0, 1, 2]
        assert np.allclose(eq_greedy.weights_internal, eq_exact.weights_internal)

    def test_the_veto_is_built_from_the_configured_instability_metric(
            self, stub_samples, monkeypatch):
        """``ke2_stab`` has no ``_chi2`` / ``_vc`` variants on purpose: it
        scores with whatever ``objectives.instability_metric`` selects, which
        is the same statistic the Instability Pareto axis reads.

        Asserted on the CALLBACK's construction rather than on a score, since
        the veto only runs when the geometry proposes an extension -- see
        ``test_a_clean_library_never_reaches_the_veto``."""
        X, y = _library()
        stub_samples({0: (X.shape[0],)})
        seen = []
        original = KneeSparsity._stability_callback

        def spy(metric, *args, **kwargs):
            seen.append(metric)
            return original(metric, *args, **kwargs)

        monkeypatch.setattr(KneeSparsity, '_stability_callback',
                            staticmethod(spy))
        operator = KneeSparsity()
        operator.realization = 'ke2_stab'
        with using_config(instability_metric='tile'):
            operator.apply(StubEquation({0: X}, {0: y}), {})
        assert seen == ['tile']

        # ...and no other realization builds a veto at all.
        seen.clear()
        plain = KneeSparsity()
        plain.realization = 'knee_ext2_amp'
        with using_config(instability_metric='tile'):
            plain.apply(StubEquation({0: X}, {0: y}), {})
        assert seen == []

    def test_the_veto_callback_aligns_with_the_selected_columns(
            self, stub_samples):
        """One score per SELECTED column, in ``np.flatnonzero(mask)`` order --
        the alignment ``extend2_stab`` indexes the new term by. The intercept
        is scored like any other column when the mask selects it, which is the
        same treatment it gets everywhere else in this operator."""
        X, y = _library()
        n_features = X.shape[1]
        callback = KneeSparsity._stability_callback(
            'chi2', X, y, np.ones(X.shape[0]), (X.shape[0],), n_features, None)
        mask = np.zeros(n_features + 1, dtype=bool)
        mask[[0, 2, n_features]] = True             # two columns + intercept
        scores = np.asarray(callback(mask))
        assert scores.shape == (3,)
        assert np.all(np.isfinite(scores))

    def test_a_clean_library_never_reaches_the_veto(self, stub_samples):
        """On a well-separated library the elbow already sits on the true
        support, so the geometry proposes no extension and ``ke2_stab``
        reduces to ``knee_ext2_amp``. Worth pinning: stability is a tie-breaker
        at one decision point, not a second selection rule."""
        X, y = _library()
        stub_samples({0: (X.shape[0],)})
        results = {}
        for realization in ('knee_ext2_amp', 'ke2_stab'):
            operator = KneeSparsity()
            operator.realization = realization
            equation = StubEquation({0: X}, {0: y})
            with using_config(instability_metric='chi2'):
                operator.apply(equation, {})
            results[realization] = np.asarray(equation.weights_internal)
        assert np.array_equal(results['knee_ext2_amp'], results['ke2_stab'])


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
class TestRegistry:

    def test_knee_resolves_to_the_operator(self):
        assert resolve_sparsity('knee') is KneeSparsity

    def test_realization_is_a_configurable_setting(self):
        assert 'realization' in sparsity_settings(KneeSparsity)
        assert validate_sparsity_kwargs(
            KneeSparsity, {'realization': 'ke2_stab'}) == {
                'realization': 'ke2_stab'}

    def test_an_unknown_setting_name_fails_loud(self):
        with pytest.raises(ValueError, match='Unknown sparsity_kwargs'):
            validate_sparsity_kwargs(KneeSparsity, {'realisation': 'knee'})

    def test_an_unknown_realization_value_fails_loud(self):
        """``validate_sparsity_kwargs`` checks names, not values, so the
        operator itself has to refuse -- otherwise a typo sits there silently
        selecting the default."""
        with pytest.raises(ValueError, match='Unknown knee realization'):
            build_sparsity_operator(KneeSparsity, {'realization': 'knee_ext9'})

    def test_build_sparsity_operator_applies_the_realization(self):
        operator = build_sparsity_operator(KneeSparsity,
                                           {'realization': 'knee'})
        assert isinstance(operator, KneeSparsity)
        assert operator.realization == 'knee'
        assert KneeSparsity().realization == 'knee_ext2_amp'   # not leaked

    def test_knee_reads_no_sparsity_alpha(self):
        """The degenerate interval is the convention for 'this operator does
        not tune a sparsity constant' -- the same declaration VWSR makes."""
        assert initial_sparsity_interval(KneeSparsity) == (1.0, 1.0)
        assert (initial_sparsity_interval(KneeSparsity)
                == initial_sparsity_interval(VWSRSparsity))

    def test_the_shipped_default_is_still_vwsr(self):
        """Adding an operator to the registry must not change what a search
        does when nobody asks for it."""
        assert load_search_config().objectives.sparsity_cls is VWSRSparsity

    def test_the_config_layer_carries_knee_end_to_end(self):
        config = load_search_config(overrides={
            'sparsity_cls': 'knee',
            'sparsity_kwargs': {'realization': 'knee_ext2'}})
        operator = build_sparsity_operator(config.objectives.sparsity_cls,
                                           config.objectives.sparsity_kwargs)
        assert isinstance(operator, KneeSparsity)
        assert operator.realization == 'knee_ext2'


# ---------------------------------------------------------------------------
# The fit curve
# ---------------------------------------------------------------------------
class TestDiscrepancyCurve:
    """The knee reads the Discrepancy OBJECTIVE, not a Gram by-product.

    The point of the swap is that the elbow should be measured on the curve
    the search is actually optimising. That only holds if the number this
    module computes per subset is the number the filler would report for the
    same fitted equation -- so parity against ``Discrepancy.compute`` is the
    load-bearing test here, not the recovery ones.
    """

    def _fitted(self, stub_samples, curve, metric):
        X, y = _library()
        stub_samples({0: (X.shape[0],)})
        equation = StubEquation({0: X}, {0: y})
        operator = KneeSparsity()
        operator.fit_curve = curve
        with using_config(instability_metric='chi2',
                          discrepancy_metric=metric):
            operator.apply(equation, {})
        return X, y, equation

    @pytest.mark.parametrize('metric', SOLVER_FREE_DISCREPANCIES)
    def test_the_curve_equals_what_the_filler_would_report(
            self, metric, stub_samples):
        """For the support the rule selected, the subset table's entry must
        equal ``Discrepancy(metric).compute`` on the fitted equation."""
        X, y, equation = self._fitted(stub_samples, 'discrepancy', metric)
        weights = np.asarray(equation.weights_internal)

        ctx = FitContext(g_fun_vals={0: np.ones(X.shape[0])},
                         data_shape={0: (X.shape[0],)},
                         penalty_coeff=0.2, for_rps=False)
        from_filler = Discrepancy(metric).compute(equation, ctx)

        curve, _, n_cols = subset_table(X, y, np.ones(X.shape[0]),
                                        discrepancy=metric)
        selected = sum(1 << j for j in np.flatnonzero(weights != 0))
        assert curve[selected] == pytest.approx(from_filler, rel=1e-9)

    def test_wape_matches_its_definition(self):
        """Spot-check the arithmetic against the formula rather than against
        another implementation of it."""
        rng = np.random.default_rng(5)
        t = rng.normal(size=50)
        resid = rng.normal(size=50) * 0.1
        got = discrepancy_value('wape', resid, t, np.ones(50))
        assert got == pytest.approx(np.sum(np.abs(resid)) / np.sum(np.abs(t)))

    def test_wape_ignores_the_sample_weighting(self):
        """The filler's WAPE is unweighted; only its ``l2`` option applies the
        g_func. A curve that quietly weighted WAPE would drift from the axis
        on any run with a non-uniform g_func."""
        rng = np.random.default_rng(6)
        t, resid = rng.normal(size=40), rng.normal(size=40)
        flat = discrepancy_value('wape', resid, t, np.ones(40))
        shaped = discrepancy_value('wape', resid, t, rng.uniform(.5, 2, 40))
        assert flat == pytest.approx(shaped)
        # ...whereas l2 must move with it.
        assert (discrepancy_value('l2', resid, t, np.ones(40))
                != pytest.approx(discrepancy_value('l2', resid, t,
                                                   rng.uniform(2, 3, 40))))

    def test_a_solver_only_option_is_refused_by_name(self):
        """``solver_l2`` / ``pic`` / ``deepxde`` need a solved system; there is
        no subset-level form, so this must fail loud rather than fall back to
        RSS behind the caller's back."""
        rng = np.random.default_rng(7)
        X = rng.normal(size=(30, 3))
        with pytest.raises(ValueError, match='without solving'):
            subset_table(X, rng.normal(size=30), np.ones(30),
                         discrepancy='solver_l2')

    def test_the_default_curve_is_the_discrepancy_objective(self):
        assert KneeSparsity().fit_curve == 'discrepancy'

    def test_fit_curve_validates(self):
        operator = KneeSparsity()
        operator.fit_curve = 'rss'
        assert operator.fit_curve == 'rss'
        with pytest.raises(ValueError, match='fit_curve must be'):
            operator.fit_curve = 'wape'          # a metric, not a curve choice

    def test_fit_curve_is_configurable_through_sparsity_kwargs(self):
        operator = build_sparsity_operator(KneeSparsity, {'fit_curve': 'rss'})
        assert operator.fit_curve == 'rss'
        assert KneeSparsity().fit_curve == 'discrepancy'      # not leaked

    def test_an_rss_curve_needs_no_accumulation_but_a_discrepancy_may(self):
        """Best-RSS-per-size is monotone in the size by construction; a curve
        the least-squares fit does not minimise is not, and the raw log-diff
        would then go negative and corrupt extend2's tail-dominance sums."""
        rng = np.random.default_rng(8)
        X = rng.normal(size=(120, 5))
        y = 2.0 * X[:, 0] + 0.05 * rng.normal(size=120)
        w = np.ones(120)
        rss, _, n_cols = subset_table(X, y, w)
        best = np.minimum.accumulate(
            [rss[min((S for S in range(1 << n_cols)
                      if bin(S).count('1') == k), key=lambda S: rss[S])]
             for k in range(n_cols + 1)])
        assert np.all(np.diff(best) <= 0)          # RSS: monotone

        wape, _, _ = subset_table(X, y, w, discrepancy='wape')
        per_size = [wape[min((S for S in range(1 << n_cols)
                              if bin(S).count('1') == k),
                             key=lambda S: wape[S])]
                    for k in range(n_cols + 1)]
        # The accumulated chain is what build_chain uses; it is monotone even
        # when the raw per-size curve is not.
        assert np.all(np.diff(np.minimum.accumulate(per_size)) <= 0)

    def test_both_curves_recover_the_true_support(self, stub_samples):
        """The swap must not cost recovery on a clean library."""
        for curve in ('discrepancy', 'rss'):
            X, y, equation = self._fitted(stub_samples, curve, 'wape')
            assert list(np.flatnonzero(equation.weights_internal)) == [0, 1, 2]
