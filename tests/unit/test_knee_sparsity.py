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

import warnings
from types import SimpleNamespace

import numpy as np
import pytest

from conftest import using_config

import epde.globals as global_var
import epde.operators.common.subset_selection as subset_selection
from epde.interface.search_config import (load_search_config, resolve_sparsity,
                                          sparsity_settings,
                                          validate_sparsity_kwargs)
from epde.operators.common.sparsity import (VWSRSparsity,
                                            build_sparsity_operator,
                                            initial_sparsity_interval)
from epde.operators.common.coeff_calculation import LinRegBasedCoeffsEquation
from epde.operators.common.objectives import Discrepancy, FitContext
from epde.operators.common.subset_selection import (KneeSparsity, REALIZATIONS,
                                                    BUDGET_REALIZATIONS,
                                                    INTERCEPT_TIE_LABEL,
                                                    SOLVER_FREE_DISCREPANCIES,
                                                    budget_prefix, build_chain,
                                                    column_costs,
                                                    complexity_budget,
                                                    discrepancy_value, extend2,
                                                    extend2_stab, greedy_chain,
                                                    knee_size, mask_of,
                                                    resolve_budget,
                                                    select_by_budget,
                                                    select_from_chain,
                                                    select_support,
                                                    subset_coefficients,
                                                    subset_costs,
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


def _promote(equation):
    """Run the coefficient step as ``EqRightPartSelector`` wires it for a
    physical-scale sparsity operator: it promotes ``weights_internal`` and is
    the sole author of ``weights_final_evald``."""
    coeff_calc = LinRegBasedCoeffsEquation([])
    coeff_calc.sparsity_fits_physical = True
    coeff_calc.apply(equation, {})
    return equation


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

    def test_sparsity_alone_does_not_claim_final_weights(self, stub_samples):
        """The support decision is this operator's whole output.

        ``weights_final_evald`` has exactly one author -- the coefficient
        calculation -- so an equation that has only been through sparsity must
        still read as having no fitted magnitudes.
        """
        X, y = _library()
        stub_samples({0: (X.shape[0],)})
        equation = StubEquation({0: X}, {0: y})
        with using_config(instability_metric='chi2'):
            KneeSparsity().apply(equation, {})
        assert equation.weights_internal_evald is True
        assert equation.weights_final_evald is False

    def test_the_coefficient_step_promotes_them_unchanged(self, stub_samples):
        """This operator declares ``fits_physical_scale``, so the coefficient
        step promotes its vector rather than refitting: the magnitudes ARE the
        internal ones."""
        X, y = _library()
        stub_samples({0: (X.shape[0],)})
        equation = StubEquation({0: X}, {0: y})
        with using_config(instability_metric='chi2'):
            KneeSparsity().apply(equation, {})
        _promote(equation)
        assert np.array_equal(equation.weights_internal, equation.weights_final)
        assert equation.weights_final_evald is True

    def test_the_operator_declares_its_scale(self):
        assert KneeSparsity.fits_physical_scale is True

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
        # The filler reads ``weights_final``; only the coefficient step writes
        # it. For this operator that is a promotion of the vector above.
        _promote(equation)
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


# ---------------------------------------------------------------------------
# The complexity budget ('complexity' realization)
# ---------------------------------------------------------------------------
class BudgetStub(StubEquation):
    """``StubEquation`` plus what the budget reads.

    Every term gets a factor list, whose length is its atom count, and factor
    labels for the order-independent tie-break; the equation carries its
    ``('complexity', 'u')`` budget, or no such key for ``budget=None``.
    ``atoms`` and ``names`` list the TARGET first (``target_idx`` is 0), then
    the feature columns in order.
    """

    def __init__(self, features, target, *, budget=None, atoms=None,
                 names=None):
        super().__init__(features, target)
        n_terms = len(self.structure)
        atoms = [1] * n_terms if atoms is None else list(atoms)
        names = (['t%d' % i for i in range(n_terms)] if names is None
                 else list(names))
        self.structure = [SimpleNamespace(structure=[None] * k,
                                          factors_labels=(name,))
                          for k, name in zip(atoms, names)]
        self.metaparameters = {}
        if budget is not None:
            self.metaparameters[('complexity', 'u')] = {
                'optimizable': True, 'value': budget, 'bounds': (1, 99)}


def _budgeted(stub_samples, budget, *, X=None, y=None, atoms=None,
              metric='wape', operator=None):
    """One ``'complexity'`` fit of the synthetic library (or ``X``, ``y``)."""
    if X is None:
        X, y = _library()
    stub_samples({0: (X.shape[0],)})
    equation = BudgetStub({0: X}, {0: y}, budget=budget, atoms=atoms)
    operator = KneeSparsity() if operator is None else operator
    operator.realization = 'complexity'
    with using_config(instability_metric='chi2', discrepancy_metric=metric):
        operator.apply(equation, {})
    return equation


def _bits(weights):
    return sum(1 << int(j) for j in np.flatnonzero(weights))


class TestComplexityBudget:
    """The best subset at the largest complexity a per-equation budget allows.

    The budget is the equation's own ``('complexity', var)`` metaparameter,
    never the ``('sparsity', var)`` alpha; a term costs its number of factors,
    the intercept costs 1, the target nothing.
    """

    #: Unit costs over two features and the intercept (bit 2). Subsets by
    #: bitmask: 1={0} 2={1} 3={0,1} 4={b} 5={0,b} 6={1,b} 7={0,1,b}.
    UNIT = subset_costs([1, 1, 1])

    # -- pricing ----------------------------------------------------------
    def test_column_costs_follow_the_feature_layout(self):
        """Non-target terms in structure order, then the intercept at 1."""
        equation = SimpleNamespace(
            structure=[SimpleNamespace(structure=[None] * k)
                       for k in (2, 1, 3, 1)],
            target_idx=0)
        assert column_costs(equation).tolist() == [1, 3, 1, 1]
        equation.target_idx = 2
        assert column_costs(equation).tolist() == [2, 1, 1, 1]

    def test_subset_costs_are_indexed_by_bitmask(self):
        col_costs = [1, 3, 1, 1]
        costs = subset_costs(col_costs)
        assert costs.size == 16
        for S in range(16):
            assert costs[S] == sum(c for j, c in enumerate(col_costs)
                                   if (S >> j) & 1)
        assert costs[0b1000] == 1                    # the intercept alone

    @pytest.mark.parametrize('value, expected', [
        (1, 1), (3, 3), (2.4, 2), (2.5, 3), (2.6, 3), (0.2, 1),
        (99, 7), (np.inf, 7),                          # clipped to the total
        (None, 7), (np.nan, 7), (0, 7), (0.0, 7), (-3, 7), ('three', 7),
    ])
    def test_a_value_resolves_to_an_integer_budget(self, value, expected):
        """Missing, zero, negative or non-finite means NO budget (the total),
        never one atom."""
        assert resolve_budget(value, 7) == expected

    def test_only_the_complexity_key_is_read(self):
        equation = SimpleNamespace(main_var_to_explain='u', metaparameters={
            ('sparsity', 'u'): {'optimizable': True, 'value': 2.0}})
        assert complexity_budget(equation) is None   # the alpha is no budget
        equation.metaparameters[('complexity', 'u')] = {
            'optimizable': True, 'value': 4, 'bounds': (1, 6)}
        assert complexity_budget(equation) == 4
        equation.main_var_to_explain = 'v'
        assert complexity_budget(equation) is None   # another equation's
        assert complexity_budget(
            SimpleNamespace(main_var_to_explain='u')) is None
        from epde.structure.main_structures import \
            _DEFAULT_EQUATION_METAPARAMETERS as defaults
        assert complexity_budget(SimpleNamespace(
            main_var_to_explain='u', metaparameters=dict(defaults))) is None

    # -- selection --------------------------------------------------------
    def test_the_largest_cost_wins_over_a_better_cheaper_fit(self):
        curve = np.array([9., 0.1, 0.8, 0.5, 0.2, 0.3, 0.7, 0.05])
        # {0} alone fits best of all, but a cost-2 subset is achievable.
        assert select_by_budget(curve, None, self.UNIT, 2, 3) == 0b101
        assert select_by_budget(curve, None, self.UNIT, 1, 3) == 0b001
        assert select_by_budget(curve, None, self.UNIT, 3, 3) == 0b111
        assert select_by_budget(curve, None, self.UNIT, 50, 3) == 0b111

    def test_never_empty_never_intercept_only(self):
        """The host declines an equation whose term weights are all zero."""
        curve = np.array([-1., 5., 5., 5., -1., 5., 5., 5.])
        for budget in (1, 2, 3):
            assert select_by_budget(curve, None, self.UNIT, budget, 3) & 0b011

    def test_with_nothing_affordable_the_cheapest_feature_subset(self):
        """No single-atom term at a budget of 1: the cheapest subset that
        carries a feature, whose cost then exceeds the budget."""
        costs = subset_costs([2, 3, 1])
        curve = np.linspace(1.0, 0.1, 8)          # bigger subsets fit better
        pick = select_by_budget(curve, None, costs, 1, 3)
        assert pick == 0b001
        assert costs[pick] == 2

    def test_amplified_subsets_are_skipped_then_fallen_back_to(self):
        curve = np.array([9., 0.4, 0.6, 0.1, 0.9, 0.8, 0.7, 0.05])
        amp = np.array([0., 1., 1., 500., 1., 1., 1., 500.])
        # {0,1} fits best at cost 2 but is amplified: next admissible one.
        assert select_by_budget(curve, amp, self.UNIT, 2, 3,
                                amp_cap=100.) == 0b110
        # The cost-3 level is all amplified, so the level drops to 2.
        assert select_by_budget(curve, amp, self.UNIT, 3, 3,
                                amp_cap=100.) == 0b110
        # Nothing admissible at all: knee_chain's fallback, unrestricted.
        assert select_by_budget(curve, np.full(8, 500.), self.UNIT, 2, 3,
                                amp_cap=100.) == 0b011

    def test_ties_break_by_amp_then_by_term_labels(self):
        """Not by column position: an equation's structure order is random."""
        curve = np.array([9., 0.5, 0.5, 0.1, 0.9, 0.9, 0.9, 0.2])
        labels = [('b_term',), ('a_term',), ('<intercept>',)]
        assert select_by_budget(curve, np.zeros(8), self.UNIT, 1, 3,
                                labels=labels) == 0b010
        swapped = [labels[1], labels[0], labels[2]]
        assert select_by_budget(curve, np.zeros(8), self.UNIT, 1, 3,
                                labels=swapped) == 0b001
        amp = np.array([0., 1., 2., 0., 0., 0., 0., 0.])
        assert select_by_budget(curve, amp, self.UNIT, 1, 3,
                                labels=labels) == 0b001

    def test_rounding_level_differences_are_ties(self):
        """On clean data the floored subsets' amplifications differ only in
        their last ulps -- rounding that depends on the column order through
        the Cholesky solve. Compared exactly, those ulps picked the subset and
        the order-free label key was never reached."""
        curve = np.array([9., 0.5, 0.5, 0.1, 0.9, 0.9, 0.9, 0.2])
        labels = [('b_term',), ('a_term',), ('<intercept>',)]
        a_term, b_term = 0b010, 0b001
        for amp_a, amp_b in ((1.0 + 4e-16, 1.0), (1.0, 1.0 + 4e-16)):
            amp = np.zeros(8)
            amp[a_term], amp[b_term] = amp_a, amp_b
            assert select_by_budget(curve, amp, self.UNIT, 1, 3,
                                    labels=labels) == a_term
        # The same for a curve that ties up to rounding without being floored.
        ulp = curve.copy()
        ulp[a_term] = 0.5 + 2e-16
        assert select_by_budget(ulp, np.zeros(8), self.UNIT, 1, 3,
                                labels=labels) == a_term
        # ...while a real difference, however small next to the values, still
        # decides before the labels do.
        amp = np.zeros(8)
        amp[a_term], amp[b_term] = 1.0 + 1e-6, 1.0
        assert select_by_budget(curve, amp, self.UNIT, 1, 3,
                                labels=labels) == b_term
        real = curve.copy()
        real[a_term] = 0.5 * (1 + 1e-6)
        assert select_by_budget(real, np.zeros(8), self.UNIT, 1, 3,
                                labels=labels) == b_term

    def test_an_infinite_curve_still_selects(self):
        """A zero target makes WAPE infinite for every subset; the tie band
        must neither overflow nor empty the pool."""
        with warnings.catch_warnings():
            warnings.simplefilter('error', RuntimeWarning)
            for value in (np.inf, np.nan, np.finfo(float).max):
                pick = select_by_budget(np.full(8, value), None, self.UNIT,
                                        2, 3)
                assert self.UNIT[pick] == 2 and pick & 0b011

    def test_the_greedy_pick_is_the_largest_fitting_prefix(self):
        best_sub = np.array([0, 0b0001, 0b0011, 0b1011, 0b1111])
        col_costs = [1, 2, 1, 1]                   # prefix costs 1, 3, 4, 5
        single = [0.1, 0.2, 0.3, 0.4]
        assert budget_prefix(best_sub, col_costs, 3, single) == 0b0011
        assert budget_prefix(best_sub, col_costs, 2, single) == 0b0001
        assert budget_prefix(best_sub, col_costs, 99, single) == 0b1111

    def test_with_no_fitting_prefix_the_best_affordable_column(self):
        """The chain's first pick is too dear, but a cheaper column fits on
        its own: it is chosen by its one-column curve, never the dear prefix
        that would exceed the budget."""
        # First pick: column 1 at two atoms; column 0 alone affords a budget
        # of 1, as do columns 2 and 3, of which column 3 fits best.
        best_sub = np.array([0, 0b00010, 0b00011, 0b00111, 0b01111, 0b11111])
        col_costs = [1, 2, 1, 1, 1]
        assert budget_prefix(best_sub, col_costs, 1,
                             [0.5, 0.1, 0.6, 0.4, 0.9]) == 0b01000
        # An intercept-only prefix is never the pick, and the intercept is
        # never the fallback column, however well it fits alone.
        assert budget_prefix(np.array([0, 0b100, 0b101, 0b111]), [1, 1, 1], 1,
                             [0.3, 0.2, 0.0]) == 0b010
        # A NaN curve is never the best column.
        assert budget_prefix(np.array([0, 0b0100, 0b0101, 0b0111, 0b1111]),
                             [1, 1, 2, 1], 1, [np.nan, 0.8, 0.1, 0.0]) == 0b010

    def test_with_no_affordable_column_the_cheapest_one(self):
        """Only when no feature column costs <= budget is it exceeded, and
        then by the cheapest feature subset -- one column of the lowest cost,
        the best-fitting of those."""
        best_sub = np.array([0, 0b0100, 0b0110, 0b0111, 0b1111])
        assert budget_prefix(best_sub, [3, 2, 2, 1], 1,
                             [0.0, 0.5, 0.2, 0.0]) == 0b0100
        assert budget_prefix(best_sub, [3, 2, 2, 1], 1,
                             [0.0, 0.1, 0.2, 0.0]) == 0b0010

    # -- the operator ------------------------------------------------------
    def test_every_budget_on_the_synthetic_library(self, stub_samples):
        X, y = _library()
        curve, _, n_cols = subset_table(X, y, np.ones(X.shape[0]),
                                        discrepancy='wape')
        feature_bits = (1 << (n_cols - 1)) - 1
        weights = {}
        for budget in (1, 2, 3, 7):
            equation = _budgeted(stub_samples, budget)
            w = np.asarray(equation.weights_internal)
            equation._validate_weight_layout(w, 'weights_internal')
            assert np.count_nonzero(w) == budget, budget  # unit costs
            weights[budget] = w
        for budget in (1, 2):
            level = [S for S in range(1 << n_cols)
                     if bin(S).count('1') == budget and S & feature_bits]
            assert _bits(weights[budget]) == min(level, key=lambda S: curve[S])
        assert list(np.flatnonzero(weights[3])) == [0, 1, 2]
        assert weights[3][:3] == pytest.approx([5.0, -4.0, 3.0], abs=2e-3)
        assert weights[3][-1] == 0.0
        assert np.count_nonzero(weights[7]) == 7      # everything, intercept too

    def test_a_two_atom_term_costs_two(self, stub_samples):
        rng = np.random.default_rng(4)
        n = 240
        X = rng.normal(size=(n, 4))
        atoms = [1, 1, 1, 2, 1]                 # feature 2 has two factors
        noise = 0.005 * rng.normal(size=n)
        y = 2.0 * X[:, 0] + 6.0 * X[:, 2] + noise
        # At 2 the two-factor term alone beats every pair of one-factor ones.
        assert list(np.flatnonzero(_budgeted(
            stub_samples, 2, X=X, y=y, atoms=atoms).weights_internal)) == [2]
        # The truth costs 3: one one-factor term plus the two-factor one.
        assert list(np.flatnonzero(_budgeted(
            stub_samples, 3, X=X, y=y, atoms=atoms).weights_internal)) == [0, 2]
        # Shares flipped: now a pair of one-factor terms wins at 2.
        y = 6.0 * X[:, 0] + 2.0 * X[:, 2] + noise
        w = _budgeted(stub_samples, 2, X=X, y=y, atoms=atoms).weights_internal
        assert w[2] == 0.0 and w[0] != 0.0 and np.count_nonzero(w) == 2

    def test_a_fractional_budget_reads_as_its_nearest_integer(self, stub_samples):
        assert np.array_equal(_budgeted(stub_samples, 2.6).weights_internal,
                              _budgeted(stub_samples, 3).weights_internal)

    def test_without_its_own_key_there_is_no_budget(self, stub_samples):
        """The ('sparsity', var) alpha beside it does not stand in for one."""
        X, y = _library()
        stub_samples({0: (X.shape[0],)})
        equation = BudgetStub({0: X}, {0: y}, budget=None)
        equation.metaparameters[('sparsity', 'u')] = {'optimizable': True,
                                                      'value': 1.0}
        operator = KneeSparsity()
        operator.realization = 'complexity'
        with using_config(instability_metric='chi2'):
            operator.apply(equation, {})
        assert np.count_nonzero(equation.weights_internal) == 7

    def test_the_chain_realizations_ignore_the_budget(self, stub_samples):
        X, y = _library()
        stub_samples({0: (X.shape[0],)})
        equation = BudgetStub({0: X}, {0: y}, budget=1)
        with using_config(instability_metric='chi2'):
            KneeSparsity().apply(equation, {})
        assert list(np.flatnonzero(equation.weights_internal)) == [0, 1, 2]

    def test_multisample_shares_one_support_within_the_budget(
            self, stub_samples):
        """One subset for every trajectory, each refit on it, then averaged
        -- unlike the chain realizations, whose union of per-trajectory
        supports (``test_multisample_averages_the_full_vector``) can exceed
        any budget."""
        X0, y0 = _library(seed=1, n=200)
        X1, y1 = _library(seed=2, n=160)
        y1 = y1 + 2.0 * X1[:, 4]                  # sample 1 needs one more term
        stub_samples({0: (200,), 1: (160,)})
        expected_support = {3: [0, 1, 2], 4: [0, 1, 2, 4]}
        for budget, support in expected_support.items():
            equation = BudgetStub({0: X0, 1: X1}, {0: y0, 1: y1},
                                  budget=budget)
            operator = KneeSparsity()
            operator.realization = 'complexity'
            with using_config(instability_metric='chi2'):
                operator.apply(equation, {})
            w = np.asarray(equation.weights_internal)
            assert list(np.flatnonzero(w)) == support
            mask = w != 0
            refits = [subset_coefficients(X0, y0, np.ones(200), mask),
                      subset_coefficients(X1, y1, np.ones(160), mask)]
            assert np.array_equal(w, np.mean(np.stack(refits, axis=1), axis=1))
        assert w[4] == pytest.approx(1.0, abs=5e-3)   # mean of ~0 and ~2

    def test_the_greedy_path_respects_the_budget(self, stub_samples):
        for budget in (1, 2, 3):
            operator = KneeSparsity()
            operator.max_exhaustive_columns = 3       # 7 columns -> fallback
            w = _budgeted(stub_samples, budget,
                          operator=operator).weights_internal
            assert np.count_nonzero(w) == budget
        assert list(np.flatnonzero(w)) == [0, 1, 2]

    #: Feature-column orders for the permutation test, identity included.
    ORDERS = ([0, 1, 2, 3, 4, 5], [5, 4, 3, 2, 1, 0], [3, 0, 5, 1, 4, 2],
              [1, 2, 0, 4, 5, 3], [2, 5, 1, 3, 0, 4], [4, 3, 0, 5, 2, 1],
              [0, 4, 2, 5, 3, 1], [5, 1, 3, 0, 2, 4])

    def test_the_pick_does_not_depend_on_the_column_order(
            self, stub_samples, monkeypatch):
        """Exact-fit data and a budget above the truth: every "truth + junk"
        subset of the level is floored to one curve value and their
        amplifications differ only by order-dependent rounding, so the pick
        must come from the term labels. Through the real ``subset_table`` and
        the real label construction, every column order must select ONE label
        set -- and the labels ``select_by_budget`` receives must be the
        permuted columns' own, target excluded and intercept last (the
        alignment a misplaced label list would break)."""
        rng = np.random.default_rng(0)
        n = 240
        X = rng.normal(size=(n, 6))
        y = 5.0 * X[:, 0] - 4.0 * X[:, 1] + 3.0 * X[:, 2]       # no noise
        names = ['a', 'b', 'c', 'd', 'e', 'f']
        stub_samples({0: (n,)})
        picks = []
        original = subset_selection.select_by_budget

        def spy(*args, **kwargs):
            subset = original(*args, **kwargs)
            picks.append((subset, kwargs['labels']))
            return subset

        monkeypatch.setattr(subset_selection, 'select_by_budget', spy)
        for budget in (4, 5, 6):
            chosen = set()
            for order in self.ORDERS:
                equation = BudgetStub({0: X[:, order]}, {0: y}, budget=budget,
                                      names=['target'] + [names[j]
                                                          for j in order])
                operator = KneeSparsity()
                operator.realization = 'complexity'
                with using_config(instability_metric='chi2'):
                    operator.apply(equation, {})
                subset, labels = picks[-1]
                assert labels == ([("'%s'" % names[j],) for j in order]
                                  + [INTERCEPT_TIE_LABEL])
                chosen.add(frozenset(labels[j] for j in range(7)
                                     if (subset >> j) & 1))
                # The refit carries nothing outside the selected subset.
                weights = np.asarray(equation.weights_internal)
                assert not np.any(weights[~mask_of(subset, 7)])
            assert len(chosen) == 1, (budget, chosen)
            picked = next(iter(chosen))
            assert len(picked) == budget                    # unit costs
            assert {("'a'",), ("'b'",), ("'c'",)} <= picked  # truth inside

    def test_the_greedy_path_falls_back_to_an_affordable_column(
            self, stub_samples):
        """The forward chain's first pick is a two-atom term, which a budget
        of 1 cannot afford. A one-atom column can, and must be the pick --
        the same one the exhaustive table chooses -- not the dear prefix."""
        rng = np.random.default_rng(4)
        n = 240
        X = rng.normal(size=(n, 4))
        y = 6.0 * X[:, 0] + 2.0 * X[:, 1] + 0.005 * rng.normal(size=n)
        atoms = [1, 2, 1, 1, 1]                 # feature 0 has two factors
        greedy = KneeSparsity()
        greedy.max_exhaustive_columns = 3       # 5 columns -> fallback
        _, best_sub, _, _ = greedy_chain(X, y, np.ones(n), amp_cap=100.,
                                         discrepancy='wape')
        assert int(best_sub[1]) == 0b00001      # the dear column leads
        w = _budgeted(stub_samples, 1, X=X, y=y, atoms=atoms,
                      operator=greedy).weights_internal
        assert list(np.flatnonzero(w)) == [1]
        exact = _budgeted(stub_samples, 1, X=X, y=y, atoms=atoms)
        assert list(np.flatnonzero(exact.weights_internal)) == [1]

    def test_the_greedy_path_reads_the_stacked_trajectories(
            self, stub_samples):
        """Each trajectory carries an impostor column equal to its target, so
        a forward chain on either one alone leads with the impostor and its
        budget-2 prefix misses the truth; on the row-stacked data no impostor
        explains every row and the true pair comes first."""
        rng = np.random.default_rng(31)
        n = 400
        samples = {}
        for key, impostor in ((0, 2), (1, 3)):
            X = rng.normal(size=(n, 4))
            y = 3.0 * X[:, 0] + X[:, 1]
            X[:, impostor] = y
            samples[key] = (X, y)
        truth = 0b00011
        unit = [1] * 5
        for X, y in samples.values():
            _, best_sub, _, _, (single, _) = greedy_chain(
                X, y, np.ones(n), amp_cap=100., discrepancy='wape',
                singles=True)
            assert budget_prefix(best_sub, unit, 2, single) != truth
        stub_samples({0: (n,), 1: (n,)})
        equation = BudgetStub({key: X for key, (X, _) in samples.items()},
                              {key: y for key, (_, y) in samples.items()},
                              budget=2)
        operator = KneeSparsity()
        operator.realization = 'complexity'
        operator.max_exhaustive_columns = 3     # 5 columns -> fallback
        with using_config(instability_metric='chi2'):
            operator.apply(equation, {})
        assert list(np.flatnonzero(equation.weights_internal)) == [0, 1]

    def test_the_amplification_cap_reads_the_mean_table(
            self, stub_samples, monkeypatch):
        """Two trajectories whose amplifications disagree: the cap applies to
        their MEAN, the table the curve is averaged into -- not to either
        trajectory alone, and not to the worse of the two."""
        X0, y0 = _library(seed=1, n=200)
        X1, y1 = _library(seed=2, n=160)
        stub_samples({0: (200,), 1: (160,)})
        n_cols = 7
        curve = np.full(1 << n_cols, 0.9)
        curve[0b0000001] = 0.1                  # {0}: best single column
        curve[0b0000010] = 0.2                  # {1}: runner-up

        def install(amp_first, amp_second):
            amps = {200: amp_first, 160: amp_second}

            def fake_table(X, y, w, discrepancy=None):
                amp = np.ones(1 << n_cols)
                amp[0b0000001] = amps[X.shape[0]]
                return curve.copy(), amp, n_cols

            monkeypatch.setattr(subset_selection, 'subset_table', fake_table)

        def picked():
            equation = BudgetStub({0: X0, 1: X1}, {0: y0, 1: y1}, budget=1)
            operator = KneeSparsity()
            operator.realization = 'complexity'
            with using_config(instability_metric='chi2'):
                operator.apply(equation, {})
            return list(np.flatnonzero(equation.weights_internal))

        install(150.0, 10.0)        # mean 80: admissible, though one is not
        assert picked() == [0]
        install(30.0, 190.0)        # mean 110: amplified, though one is not
        assert picked() == [1]

    @pytest.mark.parametrize('metric, cache, other', [
        ('vcoef', '_cached_vc_score', '_cached_sw_weights'),
        ('cv', '_cached_sw_weights', '_cached_vc_score')])
    def test_the_gram_caches_are_the_chain_paths_own(
            self, metric, cache, other, stub_samples):
        """Under a Gram-coupled instability metric the budget path must leave
        the caches the Instability filler reads exactly as the chain path does
        for the same support: per trajectory, scored on the selected support."""
        X0, y0 = _library(seed=1, n=200)
        X1, y1 = _library(seed=2, n=160)
        stub_samples({0: (200,), 1: (160,)})
        fitted = {}
        for realization, budget in (('knee_ext2_amp', None), ('complexity', 3)):
            equation = BudgetStub({0: X0, 1: X1}, {0: y0, 1: y1},
                                  budget=budget)
            operator = KneeSparsity()
            operator.realization = realization
            with using_config(instability_metric=metric), \
                    warnings.catch_warnings():
                # vcoef falls back to a fixed basis without a samples manager.
                warnings.simplefilter('ignore', global_var.EPDEUsageWarning)
                operator.apply(equation, {})
            fitted[realization] = equation
        chain, budgeted = fitted['knee_ext2_amp'], fitted['complexity']
        for equation in (chain, budgeted):
            assert list(np.flatnonzero(equation.weights_internal)) == [0, 1, 2]
            assert getattr(equation, other) is None
        want, got = getattr(chain, cache), getattr(budgeted, cache)
        assert want is not None and got is not None
        assert set(got) == set(want) == {0, 1}
        for key in want:
            assert np.array_equal(np.asarray(got[key]), np.asarray(want[key]))

    @pytest.mark.parametrize('metric', SOLVER_FREE_DISCREPANCIES)
    def test_the_curve_equals_what_the_filler_would_report(
            self, metric, stub_samples):
        """At a budget below the truth, so the selected curve value is a real
        residual rather than the machine floor."""
        X, y = _library()
        equation = _promote(_budgeted(stub_samples, 2, metric=metric))
        ctx = FitContext(g_fun_vals={0: np.ones(X.shape[0])},
                         data_shape={0: (X.shape[0],)},
                         penalty_coeff=0.2, for_rps=False)
        from_filler = Discrepancy(metric).compute(equation, ctx)
        curve, _, _ = subset_table(X, y, np.ones(X.shape[0]),
                                   discrepancy=metric)
        assert curve[_bits(equation.weights_internal)] == pytest.approx(
            from_filler, rel=1e-9)

    def test_the_fit_contract_is_the_chains_own(self, stub_samples):
        """No claim on the final weights, and both caches overwritten."""
        equation = _budgeted(stub_samples, 3)
        assert equation.weights_internal_evald is True
        assert equation.weights_final_evald is False
        assert equation._cached_sw_weights is None
        assert equation._cached_vc_score is None

    def test_non_finite_features_give_an_empty_equation(self, stub_samples):
        X, y = _library()
        X[3, 2] = np.nan
        w = _budgeted(stub_samples, 3, X=X, y=y).weights_internal
        assert np.all(np.asarray(w) == 0.0)

    def test_a_misaligned_structure_fails_loud(self, stub_samples):
        X, y = _library()
        stub_samples({0: (X.shape[0],)})
        equation = BudgetStub({0: X}, {0: y}, budget=3)
        equation.structure.append(SimpleNamespace(structure=[None],
                                                  factors_labels=('extra',)))
        operator = KneeSparsity()
        operator.realization = 'complexity'
        with using_config(instability_metric='chi2'):
            with pytest.raises(ValueError, match='non-target terms'):
                operator.apply(equation, {})

    # -- configuration ----------------------------------------------------
    def test_complexity_is_a_valid_realization_value(self):
        assert validate_sparsity_kwargs(
            KneeSparsity, {'realization': 'complexity'}) == {
                'realization': 'complexity'}
        operator = build_sparsity_operator(KneeSparsity,
                                           {'realization': 'complexity'})
        assert operator.realization == 'complexity'
        assert BUDGET_REALIZATIONS == ('complexity',)
        assert 'complexity' not in REALIZATIONS

    def test_select_from_chain_still_refuses_it(self):
        """A budget has no chain; listing it there would fall through to
        extend2 silently."""
        chain = (TestChainArithmetic.BEST_RSS, TestChainArithmetic.BEST_SUB,
                 TestChainArithmetic.D)
        with pytest.raises(ValueError, match='Unknown knee realization'):
            select_from_chain('complexity', *chain, TestChainArithmetic.N_COLS)

    def test_the_budget_range_follows_the_term_and_factor_maxima(self):
        """(T - 1) * F + 1: the target is free, the intercept costs 1."""
        kwargs = {'realization': 'complexity'}
        assert KneeSparsity.complexity_interval(kwargs, 6, 2) == (1, 11)
        assert KneeSparsity.complexity_interval(kwargs, 6, 1) == (1, 6)
        assert KneeSparsity.complexity_interval(
            kwargs, 6, {'factors_num': [1, 2], 'probas': [0.65, 0.35]}) == (1, 11)
        with pytest.raises(ValueError, match='term and factor maxima'):
            KneeSparsity.complexity_interval(kwargs, None, 2)

    def test_no_budget_range_outside_the_budget_realization(self):
        for kwargs in (None, {}, {'realization': 'knee'},
                       {'realization': 'knee_ext2_amp'}):
            assert KneeSparsity.complexity_interval(kwargs, 6, 2) is None

    def test_the_hook_is_no_setting_and_the_alpha_seeding_is_untouched(self):
        assert 'complexity_interval' not in sparsity_settings(KneeSparsity)
        assert initial_sparsity_interval(KneeSparsity) == (1.0, 1.0)
