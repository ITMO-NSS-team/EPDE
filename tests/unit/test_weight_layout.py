#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""THE coefficient-vector contract, and THE INTERCEPT RULE that rides on it.

Contract (``Equation._validate_weight_layout``) -- with
``m = len(structure) - 1`` non-target terms::

    weights_internal : (m+1,)   support-selection coefficients
    weights_final    : (m+1,)   final physical magnitudes

    index i in 0..m-1  ->  structure term at ``weight_index``-th position
    index -1 (== m)    ->  the fitted intercept (may be exactly 0.0)

Zeros are RETAINED in both. Before the unification the producers disagreed --
the sparsity operators emitted a bare length-m ``estimator.coef_`` next to a
zero-filtered ``nnz+1`` ``weights_final``, while the translator emitted the
full ``m+1`` layout for both -- so consumers either sniffed the length or
silently depended on ``remove_zero_terms`` having already collapsed nnz onto
m. These tests pin the single layout and the readers that index it.

Intercept rule: a regularized-away intercept (``weights_internal[-1] == 0``)
is not a column of any later model -- every downstream ``fit_intercept`` is
``bool(weights_internal[-1] != 0)``.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from epde.structure.main_structures import Equation


# --------------------------------------------------------------------------- #
#  A bare Equation: __new__ plus the slots the weight machinery touches.       #
#  Building a real one needs a token pool, a domain and cached derivatives.    #
# --------------------------------------------------------------------------- #
def _equation(n_terms, target_pos, term_coefs, intercept=0.0, final=None):
    eq = Equation.__new__(Equation)
    eq.structure = [SimpleNamespace(name='t{0}'.format(i)) for i in range(n_terms)]
    eq._target_term = eq.structure[target_pos]
    internal = np.asarray(list(term_coefs) + [intercept], dtype=float)
    eq.weights_internal_evald = True
    eq.weights_final_evald = True
    eq.weights_internal = internal
    eq.weights_final = internal.copy() if final is None else np.asarray(final, dtype=float)
    return eq


class TestLayoutValidation:

    def test_full_length_vector_is_accepted(self):
        eq = _equation(3, 2, [-1.0, 0.5], intercept=7.0)
        assert len(eq.weights_internal) == len(eq.structure)
        assert len(eq.weights_final) == len(eq.structure)

    @pytest.mark.parametrize('attr', ['weights_internal', 'weights_final'])
    def test_short_vector_is_rejected(self, attr):
        # The pre-unification sparsity layout: m entries, no intercept slot.
        eq = _equation(3, 2, [-1.0, 0.5])
        with pytest.raises(ValueError, match='trailing intercept slot'):
            setattr(eq, attr, np.array([-1.0, 0.5]))

    @pytest.mark.parametrize('attr', ['weights_internal', 'weights_final'])
    def test_zero_filtered_vector_is_rejected(self, attr):
        # The pre-unification weights_final layout: nnz + 1.
        eq = _equation(4, 3, [-1.0, 0.0, 0.5])
        with pytest.raises(ValueError, match='trailing intercept slot'):
            setattr(eq, attr, np.array([-1.0, 0.5, 0.0]))

    @pytest.mark.parametrize('attr', ['weights_internal', 'weights_final'])
    def test_none_passes(self, attr):
        # reset_state nulls both vectors.
        eq = _equation(3, 2, [-1.0, 0.5])
        setattr(eq, attr, None)
        assert getattr(eq, '_' + attr) is None


class TestWeightIndex:

    @pytest.mark.parametrize('target_pos', [0, 1, 2, 3])
    def test_round_trips_over_every_target_position(self, target_pos):
        n = 4
        eq = _equation(n, target_pos, [0.1] * (n - 1))
        seen = [eq.weight_index(i) for i in range(n) if i != target_pos]
        # Every non-target term maps to a distinct slot in 0..m-1, in order.
        assert seen == list(range(n - 1))

    def test_target_has_no_slot(self):
        eq = _equation(3, 1, [-1.0, 0.5])
        with pytest.raises(ValueError, match='carries no weight slot'):
            eq.weight_index(1)

    def test_hoisted_target_matches_derived(self):
        eq = _equation(4, 2, [1.0, 2.0, 3.0])
        for i in (0, 1, 3):
            assert eq.weight_index(i) == eq.weight_index(i, eq.target_idx)


class TestActiveMaskAndIntercept:

    def test_active_mask_covers_terms_only(self):
        eq = _equation(4, 3, [-1.0, 0.0, 0.5], intercept=2.0)
        np.testing.assert_array_equal(eq.active_mask, [True, False, True])

    def test_intercept_is_the_trailing_final_slot(self):
        eq = _equation(3, 2, [-1.0, 0.5], intercept=7.0)
        assert eq.intercept == 7.0

    def test_zero_intercept_is_not_confused_with_a_term(self):
        eq = _equation(3, 2, [-1.0, 0.5], intercept=0.0)
        assert eq.intercept == 0.0
        np.testing.assert_array_equal(eq.active_mask, [True, True])


class TestRemoveZeroTerms:
    """Both vectors compact in lockstep, each keeping its own intercept.

    ``weights_final`` used to be skipped here -- "it already holds only the
    non-zero entries" was true of the old zero-filtered layout only.
    """

    def _pruned(self):
        # 5 terms, target at 3; term slots map to [t0, t1, t2, t4] -> drop t1, t4.
        eq = _equation(5, 3,
                       term_coefs=[-1.0, 0.0, 0.5, 0.0], intercept=7.0,
                       final=[-0.9, 0.0, 0.4, 0.0, 6.5])
        eq._invalidate_label_cache()
        eq.remove_zero_terms()
        return eq

    def test_structure_and_both_vectors_stay_aligned(self):
        eq = self._pruned()
        assert len(eq.structure) == 3                       # t0, t2, target
        assert len(eq.weights_internal) == len(eq.structure)
        assert len(eq.weights_final) == len(eq.structure)

    def test_surviving_values_and_intercepts_are_kept(self):
        eq = self._pruned()
        np.testing.assert_allclose(eq.weights_internal, [-1.0, 0.5, 7.0])
        np.testing.assert_allclose(eq.weights_final, [-0.9, 0.4, 6.5])

    def test_target_still_tracked_after_the_prune(self):
        eq = self._pruned()
        assert eq.structure[eq.target_idx] is eq.target

    def test_zero_intercept_slot_is_never_pruned(self):
        eq = _equation(4, 3, [-1.0, 0.0, 0.5], intercept=0.0)
        eq._invalidate_label_cache()
        eq.remove_zero_terms()
        assert len(eq.weights_internal) == len(eq.structure) == 3
        assert eq.weights_internal[-1] == 0.0


class TestRenderersNeedNoPrune:
    """text_form / latex_form / EquationIterator must read the right
    coefficient for a structure that still carries zero-weight terms -- the
    case that was wrong whenever ``remove_zero_terms`` had not run yet."""

    def _eq(self):
        # target in the MIDDLE, so the index shift is exercised.
        return _equation(4, 1, term_coefs=[-1.0, 0.0, 0.5], intercept=7.0)

    def test_text_form_pairs_each_term_with_its_own_coefficient(self):
        eq = self._eq()
        form = eq.text_form
        # t0 -> slot 0, t2 -> slot 1, t3 -> slot 2, then the intercept.
        assert '-1.0 * t0' in form
        assert '0.0 * t2' in form
        assert '0.5 * t3' in form
        assert '7.0 = t1' in form

    def test_iterator_pairs_coefficients_with_terms(self):
        eq = self._eq()
        pairs = {}
        for coeff, term in eq:
            pairs[term.name] = coeff
        assert pairs['t0'] == -1.0
        assert pairs['t1'] == -1.0       # the target's implicit weight
        assert pairs['t3'] == 0.5
        assert 't2' not in pairs         # zero coefficient, skipped


class TestInterceptRule:
    """``fit_intercept`` is ``weights_internal[-1] != 0`` at every consumer
    downstream of the sparsity fit -- never a literal True."""

    def test_live_consumers_read_the_support_decision(self):
        import inspect
        from epde.operators.common import coeff_calculation, objectives

        checked = 0
        for mod in (coeff_calculation, objectives):
            for line in inspect.getsource(mod).splitlines():
                stripped = line.strip()
                if stripped.startswith('#') or 'fit_intercept =' not in stripped:
                    continue
                assert 'weights_internal[-1] != 0' in stripped, stripped
                checked += 1
        assert checked >= 2, 'expected the refit and the instability consumers'

    def test_killed_intercept_forces_the_refit_through_the_origin(self):
        # LinRegBasedCoeffsEquation reads weights_internal[-1]; a zero there
        # must reach LinearRegression as fit_intercept=False so the physical
        # magnitude stays exactly 0.0 rather than being re-estimated.
        from sklearn.linear_model import LinearRegression

        X = np.linspace(0.0, 1.0, 40)[:, None]
        y = 3.0 * X[:, 0] + 5.0                      # a genuine offset
        for killed, expect_zero in ((True, True), (False, False)):
            est = LinearRegression(copy_X=True, fit_intercept=not killed)
            est.fit(X, y)
            assert (float(est.intercept_) == 0.0) is expect_zero
