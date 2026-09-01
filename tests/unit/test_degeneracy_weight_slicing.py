#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Degeneracy tests must look at TERM weights, not the whole weight vector.

Under the unified layout (``Equation._validate_weight_layout``) both weight
vectors are ``[*one coef per non-target term, intercept]``. ``_term_weights``
is the named ``[:-1]`` accessor, and the distinction is load-bearing twice
over:

* the raw ``np.all(weights_internal == 0)`` would additionally demand a zero
  intercept, so an all-zero-terms equation carrying a surviving constant would
  escape the degeneracy verdict;
* the older ``weights_internal[:-1]`` slice ran against vectors that had NO
  intercept slot (the sparsity operators stored a bare ``estimator.coef_``),
  so it dropped a real coefficient -- and for a TWO-term equation it produced
  the EMPTY array, making ``np.all(... == 0)`` vacuously True. Two-term
  equations are the shape of most true laws (``u_tt = -u``), so they were
  declined by the right-part sweep and stamped LOSS_NAN_VAL in place:
  unfindable by construction.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from epde.operators.common.objectives import (
    Discrepancy, _term_weights)


def _equation(n_terms, term_coefs, intercept=0.0, fitness_value=0.0):
    """Equation stand-in: ``n_terms`` structure entries (target included), so
    ``weights_internal`` is ``n_terms`` long -- ``n_terms - 1`` term slots plus
    the trailing intercept."""
    weights = np.asarray(list(term_coefs) + [intercept], dtype=float)
    assert len(weights) == n_terms, 'test fixture violates the layout it pins'
    return SimpleNamespace(
        structure=[object() for _ in range(n_terms)],
        weights_internal=weights,
        weights_final=weights.copy(),
        fitness_value=fitness_value)


class TestTermWeights:

    def test_intercept_slot_is_dropped(self):
        # 3 terms -> 2 non-target coefficients + intercept.
        eq = _equation(3, [-1.0, 0.5], intercept=7.0)
        np.testing.assert_allclose(_term_weights(eq), [-1.0, 0.5])

    def test_two_term_equation_keeps_its_single_coefficient(self):
        eq = _equation(2, [-0.9997])
        assert len(_term_weights(eq)) == 1
        np.testing.assert_allclose(_term_weights(eq), [-0.9997])

    def test_zero_intercept_does_not_shorten_the_slice(self):
        eq = _equation(3, [0.0, 0.42], intercept=0.0)
        np.testing.assert_allclose(_term_weights(eq), [0.0, 0.42])


class TestTwoTermEquationIsNotDegenerate:

    @pytest.mark.parametrize('filler', [Discrepancy()])
    def test_fitted_two_term_law_survives(self, filler):
        # u_tt = -0.9997 * u with a near-exact fit: the truth of the
        # multisample ODE benchmark. It must reach the Pareto front.
        eq = _equation(2, [-0.9997], fitness_value=8.6e-05)
        assert filler.is_degenerate(eq) is False

    @pytest.mark.parametrize('filler', [Discrepancy()])
    def test_genuinely_zeroed_two_term_equation_still_degenerate(self, filler):
        # The check must keep FIRING when the sole coefficient really is zero.
        eq = _equation(2, [0.0], fitness_value=8.6e-05)
        assert filler.is_degenerate(eq) is True

    @pytest.mark.parametrize('filler', [Discrepancy()])
    def test_surviving_intercept_does_not_rescue_zeroed_terms(self, filler):
        # All TERMS zero but the constant survived: still degenerate. The raw
        # ``np.all(weights_internal == 0)`` would return False here.
        eq = _equation(3, [0.0, 0.0], intercept=1.7, fitness_value=8.6e-05)
        assert filler.is_degenerate(eq) is True

    def test_last_coefficient_alone_keeps_equation_alive(self):
        # 3 terms, only the LAST non-target coefficient non-zero: the old
        # slice dropped exactly that one and called the equation degenerate.
        eq = _equation(3, [0.0, 0.42], fitness_value=1e-3)
        assert Discrepancy().is_degenerate(eq) is False

    def test_threshold_branch_unaffected(self):
        # A poor fit is still degenerate regardless of the weights.
        eq = _equation(2, [-0.9997], fitness_value=5.0)
        assert Discrepancy().is_degenerate(eq) is True
