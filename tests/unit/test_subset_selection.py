"""Exhaustive subset fits in Gram space.

``system_rps_mode='assignment'`` and ``'plain'`` fit subsets of an equation's
terms with the primitives of ``epde.operators.common.subset_selection``
instead of running a sparsity operator. Three properties carry them and are
pinned here:

* the TABLE is the objective's own number -- ``subset_table``'s entry for a
  support equals ``Discrepancy.compute`` on the equation fitted on it, so the
  subset search and the Pareto axis cannot disagree about what fits better;
* the OUTPUT obeys the same coefficient contract as every sparsity producer
  (``Equation._validate_weight_layout``), including the intercept being an
  ordinary selectable column that can end up exactly zero. Three readers
  depend on that -- ``Instability.compute``'s intercept rule,
  ``Complexity('terms')`` and ``amplification_ratio`` -- so an intercept that
  were non-zero by construction would quietly lie to all three;
* ``store_fit`` writes a fit the way the coefficient step and the Instability
  filler read it: ``weights_internal`` only, never ``weights_final_evald``,
  and both instability caches on every call.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from epde.operators.common.coeff_calculation import LinRegBasedCoeffsEquation
from epde.operators.common.objectives import Discrepancy, FitContext
from epde.operators.common.subset_selection import (SOLVER_FREE_DISCREPANCIES,
                                                    discrepancy_value, mask_of,
                                                    store_fit,
                                                    subset_coefficients,
                                                    subset_table,
                                                    support_scores,
                                                    trajectory_gram)
from epde.structure.main_structures import Equation


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

    @pytest.mark.parametrize('kind', ['random', 'collinear', 'constant_parallel'])
    @pytest.mark.parametrize('n, p', [(320, 1), (320, 4), (3000, 6), (20000, 3)])
    @pytest.mark.parametrize('intercept', ['selectable', 'always'])
    def test_the_returned_coefficients_are_subset_coefficients_bytes(self, kind, n, p, intercept):
        """``return_coefs`` hands back the table's own solve, which plain mode
        installs instead of solving again: it must be the same bytes, the
        lstsq fallback (an exactly singular subset) included."""
        rng = np.random.default_rng(n + 10 * p)
        X = rng.normal(size=(n, p))
        if kind == 'collinear' and p > 1:
            X[:, 1] = X[:, 0]
        if kind == 'constant_parallel':
            X[:, 0] = 2.5
        y = X @ rng.normal(size=p) + 0.3 + 0.01 * rng.normal(size=n)
        w = rng.uniform(0.5, 1.5, size=n)
        n_cols = p if intercept == 'always' else p + 1
        masks = list(range(1 << n_cols))
        rss, amp, n_out, coefs = subset_table(X, y, w, intercept=intercept, subsets=masks, return_coefs=True)
        plain = subset_table(X, y, w, intercept=intercept, subsets=masks)
        assert n_out == n_cols and len(plain) == 3
        np.testing.assert_array_equal(rss, plain[0])
        np.testing.assert_array_equal(amp, plain[1])
        for S, full in zip(masks, coefs):
            support = [bool((S >> j) & 1) for j in range(p + 1)]
            expected = subset_coefficients(X, y, w, support, intercept=intercept)
            assert full.tobytes() == expected.tobytes(), S


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

    def test_the_constant_is_a_selectable_column(self, design):
        """A support without the constant leaves the trailing slot EXACTLY
        zero -- the "no constant term" verdict three readers take from it --
        and a support with it fits the offset a target carries."""
        X, y, w = design
        n_cols = X.shape[1] + 1
        without = subset_coefficients(X, y, w, mask_of(0b000111, n_cols))
        assert list(np.flatnonzero(without)) == [0, 1, 2]
        assert without[-1] == 0.0
        assert without[:3] == pytest.approx([3.0, -2.0, 0.5], abs=1e-2)

        offset = subset_coefficients(X, y + 40.0, w, mask_of(0b100111, n_cols))
        assert list(np.flatnonzero(offset)) == [0, 1, 2, n_cols - 1]
        assert offset[-1] == pytest.approx(40.0, abs=1e-2)
        assert offset[:3] == pytest.approx([3.0, -2.0, 0.5], abs=1e-2)


# ---------------------------------------------------------------------------
# Writing a fit onto an equation
# ---------------------------------------------------------------------------
class StubEquation:
    """Minimal Equation stand-in: the surface ``store_fit``, the coefficient
    step and the ``Discrepancy`` filler touch.

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
    physical-scale sparsity class: it promotes ``weights_internal`` and is the
    sole author of ``weights_final_evald``."""
    coeff_calc = LinRegBasedCoeffsEquation([])
    coeff_calc.sparsity_fits_physical = True
    coeff_calc.apply(equation, {})
    return equation


def _library(seed=0, n=240):
    """3 true columns + 3 junk ones, no constant term."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 6))
    y = 5.0 * X[:, 0] - 4.0 * X[:, 1] + 3.0 * X[:, 2] + 0.005 * rng.normal(size=n)
    return X, y


def _stored(support_bits, y_shift=0.0):
    """A StubEquation with the fit on ``support_bits`` stored on it."""
    X, y = _library()
    y = y + y_shift
    w = np.ones(X.shape[0])
    equation = StubEquation({0: X}, {0: y})
    full = subset_coefficients(X, y, w, mask_of(support_bits, X.shape[1] + 1))
    store_fit(equation, full, {0: None}, {0: None})
    return X, y, w, equation


class TestStoreFit:

    def test_the_stored_vector_keeps_the_layout_and_its_zero_slots(self):
        """One slot per non-target term, then the intercept; a term the fit
        left out -- the constant included -- stays an exact zero."""
        _, _, _, equation = _stored(0b0000111)
        weights = np.asarray(equation.weights_internal)
        assert weights.shape == (len(equation.structure),)
        equation._validate_weight_layout(weights, 'weights_internal')
        assert list(np.flatnonzero(weights)) == [0, 1, 2]
        assert weights[-1] == 0.0
        assert weights[:3] == pytest.approx([5.0, -4.0, 3.0], abs=2e-3)

    def test_storing_alone_does_not_claim_final_weights(self):
        """The support's fit is all this writes: ``weights_final_evald`` has
        exactly one author -- the coefficient step -- so an equation that has
        only been stored on must still read as having no final magnitudes."""
        _, _, _, equation = _stored(0b0000111)
        assert equation.weights_internal_evald is True
        assert equation.weights_final_evald is False

    def test_the_coefficient_step_promotes_them_unchanged(self):
        """The fit is on the physical scale, so the coefficient step wired for
        a physical-scale class promotes the vector rather than refitting: the
        final magnitudes ARE the internal ones."""
        _, _, _, equation = _stored(0b1000111, y_shift=40.0)
        _promote(equation)
        assert np.array_equal(equation.weights_internal, equation.weights_final)
        assert equation.weights_final_evald is True

    def test_the_stale_instability_caches_are_always_overwritten(self):
        """A deep copy of a fitted equation, or EqRightPartSelector's term
        sweep, leaves an earlier fit's caches on the equation, so a skipped
        assignment would carry that fit's value forward."""
        _, _, _, equation = _stored(0b0000111)     # per-trajectory Nones
        assert equation._cached_sw_weights is None
        assert equation._cached_vc_score is None

        equation._cached_sw_weights = equation._cached_vc_score = 'stale'
        store_fit(equation, equation.weights_internal, None, None)
        assert equation._cached_sw_weights is None
        assert equation._cached_vc_score is None

        windows = {0: np.ones(3), 1: None}
        store_fit(equation, equation.weights_internal, windows, {0: None})
        assert equation._cached_sw_weights is windows   # kept whole
        assert equation._cached_vc_score is None

    def test_a_basis_free_metric_builds_no_gram_and_writes_no_cache(self):
        """Under a basis-free instability metric there is no Gram mode: no
        view is built, and neither cache the Gram-coupled metrics read gets a
        value."""
        X, y = _library()
        w = np.ones(X.shape[0])
        gram = trajectory_gram(None, X, y, w, (X.shape[0],), None, 'u', 0)
        assert gram is None
        active = mask_of(0b0000111, X.shape[1] + 1)
        assert support_scores(gram, active, 'chi2') == (None, None)


# ---------------------------------------------------------------------------
# The fit curve
# ---------------------------------------------------------------------------
class TestDiscrepancyCurve:
    """The table reads the Discrepancy OBJECTIVE, not a Gram by-product.

    The assignment mode ranks subsets on the table and the Pareto axis scores
    the fitted equation; that only agrees if the number computed per subset
    is the number the filler reports for the same fit -- so parity against
    ``Discrepancy.compute`` is the load-bearing test here.
    """

    @pytest.mark.parametrize('support_bits', [0b0000011, 0b1000011],
                             ids=['no_constant', 'with_constant'])
    @pytest.mark.parametrize('metric', SOLVER_FREE_DISCREPANCIES)
    def test_the_table_equals_what_the_filler_would_report(
            self, metric, support_bits):
        """For a support that misses a true term -- a residual well off the
        machine floor -- the table's entry must equal
        ``Discrepancy(metric).compute`` on the equation fitted on it, with
        and without the constant."""
        X, y, w, equation = _stored(support_bits)
        # The filler reads ``weights_final``; only the coefficient step writes
        # it, here as a promotion of the stored vector.
        _promote(equation)

        ctx = FitContext(g_fun_vals={0: w}, data_shape={0: (X.shape[0],)},
                         penalty_coeff=0.2, for_rps=False)
        from_filler = Discrepancy(metric).compute(equation, ctx)

        curve, _, _ = subset_table(X, y, w, discrepancy=metric)
        assert curve[support_bits] == pytest.approx(from_filler, rel=1e-9)

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
        g_func. A table that quietly weighted WAPE would drift from the axis
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
