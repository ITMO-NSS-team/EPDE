#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""THE fitted-state contract: who may claim an equation is fitted, and when.

An ``Equation`` carries two validity flags -- ``weights_internal_evald`` (a
support decision exists) and ``weights_final_evald`` (fitted magnitudes exist).
The contract around them:

* **selection hands downstream operators copies that own no fit.** Both flags
  and both weight vectors survive ``__deepcopy__`` verbatim, so the deepcopy
  site in ``ParetoLevelsCrossover`` resets wholesale. Pinned in
  ``test_solver_discrepancy_scale.py``;
* **``weights_internal_evald`` goes up only inside right-part selection.** The
  three sparsity operators are reachable only as ``EqRightPartSelector``
  suboperators;
* **``weights_final_evald`` has exactly ONE author**, the coefficient
  calculation -- whether it refits (legacy LASSO, whose support decision came
  from a min-max-rescaled fit) or promotes ``weights_internal`` (VWSR, Knee,
  which already fit on the physical scale);
* **fitness requires the fitted magnitudes**, not merely the support decision,
  because every filler reads ``weights_final`` through ``Equation.residual``.

The reset side is registry-driven: ``_EQ_STATE_GROUPS`` declares each slot and
the value it clears to, ``_EQ_CACHE_FIELDS`` does the same for the caches, and
the two public resets are compositions of those groups. The sync test below is
what keeps a newly added slot from silently escaping every reset -- the failure
that left ``_aic`` readable after its ``aic_calculated`` flag went down.
"""

from types import SimpleNamespace

import numpy as np
import pytest

from epde.operators.common.coeff_calculation import LinRegBasedCoeffsEquation
from epde.operators.common.sparsity import LASSOSparsity, VWSRSparsity
from epde.operators.common.subset_selection import KneeSparsity
from epde.structure.main_structures import (Equation, _EQ_CACHE_FIELDS,
                                            _EQ_STATE_GROUPS,
                                            _EQ_STATE_GROUP_SLOTS)


#: Slots that are the equation's identity or its inputs, not derived state, so
#: no reset may touch them. Everything else must belong to a state group or a
#: cache -- that is the whole point of TestStateRegistrySync.
_NOT_DERIVED_STATE = frozenset({
    '_history', 'structure', 'interelement_operator', 'n_immutable', 'pool',
    'metaparameters', 'main_var_to_explain',
})


def _fitted_equation(n_terms=3, target_pos=2):
    """A bare Equation carrying a complete, self-consistent fitted state.

    ``__new__`` rather than ``__init__``: building a real one needs a token
    pool, a domain and cached derivatives, and none of the reset machinery
    reads any of that.
    """
    eq = Equation.__new__(Equation)
    eq.structure = [SimpleNamespace(name='t{0}'.format(i)) for i in range(n_terms)]
    eq._target_term = eq.structure[target_pos]
    eq.main_var_to_explain = 'u'
    eq.right_part_selected = True
    weights = np.arange(1.0, n_terms + 1.0)
    eq.weights_internal_evald = True
    eq.weights_final_evald = True
    eq.weights_internal = weights
    eq.weights_final = weights.copy()
    eq.fitness_calculated = True
    eq._fitness_value = 0.5
    eq.stability_calculated = True
    eq._coefficients_stability = 0.25
    eq.complexity_calculated = True
    eq._complexity_value = 3
    eq.aic_calculated = True
    eq._aic = 12.0
    eq.solver_form_defined = True
    for field, _policy in _EQ_CACHE_FIELDS:
        setattr(eq, field, {'stale': True} if field == '_eval_cache' else 'stale')
    return eq


# --------------------------------------------------------------------------- #
#  The registry is the single source of truth                                  #
# --------------------------------------------------------------------------- #
class TestStateRegistrySync:

    def test_every_slot_is_accounted_for(self):
        """A slot that is in neither registry nor the exempt set is derived
        state no reset clears -- exactly how ``_aic`` came to outlive its own
        ``aic_calculated`` flag behind a getter with no guard."""
        grouped = {slot for slots in _EQ_STATE_GROUP_SLOTS.values()
                   for slot, _ in slots}
        cached = {field for field, _ in _EQ_CACHE_FIELDS}
        unaccounted = set(Equation.__slots__) - grouped - cached - _NOT_DERIVED_STATE
        assert not unaccounted, unaccounted

    def test_no_slot_is_claimed_twice(self):
        grouped = [slot for slots in _EQ_STATE_GROUP_SLOTS.values()
                   for slot, _ in slots]
        cached = [field for field, _ in _EQ_CACHE_FIELDS]
        assert len(grouped) == len(set(grouped))
        assert not set(grouped) & set(cached)

    def test_every_registered_slot_is_declared_in_slots(self):
        declared = set(Equation.__slots__)
        for group, slots in _EQ_STATE_GROUPS:
            for slot, _ in slots:
                assert slot in declared, (group, slot)

    def test_rps_loop_state_is_not_equation_state(self):
        """``simplified`` / ``is_correct_right_part`` are locals of
        ``EqRightPartSelector.apply``. As attributes they let an offspring
        inherit "already simplified", skip the loop body -- and with it the
        ``reset_state`` on its first line -- and leave RPS still carrying its
        parent's coefficients."""
        assert 'simplified' not in Equation.__slots__
        assert 'is_correct_right_part' not in Equation.__slots__

    def test_every_flag_is_paired_with_the_value_it_guards(self):
        """A flag whose value outlives it is readable-but-stale."""
        for group in ('weights_internal', 'weights_final'):
            slots = dict(_EQ_STATE_GROUP_SLOTS[group])
            assert slots['{0}_evald'.format(group)] is False
            assert slots['_{0}'.format(group)] is None
        scores = dict(_EQ_STATE_GROUP_SLOTS['scores'])
        for flag, value in (('fitness_calculated', '_fitness_value'),
                            ('stability_calculated', '_coefficients_stability'),
                            ('complexity_calculated', '_complexity_value'),
                            ('aic_calculated', '_aic')):
            assert scores[flag] is False
            assert scores[value] is None


# --------------------------------------------------------------------------- #
#  The two resets are compositions of those groups                             #
# --------------------------------------------------------------------------- #
class TestResetCompositions:

    def test_structure_change_keeps_the_target_and_drops_the_rest(self):
        eq = _fitted_equation()
        target = eq._target_term
        eq.reset_for_structure_change()
        assert eq._target_term is target
        assert eq.weights_internal_evald is False
        assert eq.weights_final_evald is False
        assert eq._weights_internal is None
        assert eq._weights_final is None
        assert eq.right_part_selected is False
        assert eq.fitness_calculated is False
        assert eq.aic_calculated is False

    def test_a_hard_reset_drops_the_target_too(self):
        eq = _fitted_equation()
        eq.reset_state(True)
        assert eq._target_term is None
        assert eq.right_part_selected is False

    def test_the_soft_reset_is_the_structure_change_reset(self):
        soft, named = _fitted_equation(), _fitted_equation()
        soft.reset_state(False)
        named.reset_for_structure_change()
        for slot in Equation.__slots__:
            if slot in _NOT_DERIVED_STATE:
                continue
            assert getattr(soft, slot, None) == getattr(named, slot, None), slot

    def test_the_support_decision_does_not_survive_a_soft_reset(self):
        """The retired asymmetry: a soft reset used to keep
        ``weights_internal_evald`` up with its data intact while dropping only
        the ``weights_final`` flag. Every call site of it followed a real
        structural change, so the surviving support decision described a
        structure that no longer existed -- and on the one path where the outer
        RPS loop did not re-enter, it reached ``remove_zero_terms``."""
        eq = _fitted_equation()
        eq.reset_state(False)
        assert eq.weights_internal_evald is False
        assert eq._weights_internal is None

    def test_a_stale_value_never_outlives_its_flag(self):
        eq = _fitted_equation()
        eq.reset_state(True)
        assert eq.aic is None          # the getter has no flag guard
        assert eq.fitness_value is None
        assert eq.coefficients_stability is None

    def test_every_cache_is_wiped_by_both_resets(self):
        for reset in (lambda e: e.reset_state(True),
                      lambda e: e.reset_for_structure_change()):
            eq = _fitted_equation()
            reset(eq)
            for field, _policy in _EQ_CACHE_FIELDS:
                expected = {} if field == '_eval_cache' else None
                assert getattr(eq, field) == expected, field

    def test_the_eval_cache_is_wiped_to_a_fresh_dict(self):
        """Not ``None``: ``evaluate`` indexes it without a hasattr guard on the
        hot path."""
        eq = _fitted_equation()
        eq.reset_state(True)
        assert eq._eval_cache == {}
        assert isinstance(eq._eval_cache, dict)


# --------------------------------------------------------------------------- #
#  Structure-keyed caches survive a prune; fit-derived ones are recomputed     #
# --------------------------------------------------------------------------- #
class TestSparsityCachePreservation:
    """``remove_zero_terms`` compacts the weights and drops the structure-keyed
    caches, but ``_cached_sw_weights`` / ``_cached_vc_score`` deliberately
    survive: ``PhysicsInformedLasso.fit`` recomputed them on the CONVERGED
    active mask, so they already describe the zero-pruned structure, and
    ``Instability.compute`` reads them immediately after this call."""

    @staticmethod
    def _with_a_zero_weight():
        eq = _fitted_equation(n_terms=4, target_pos=3)
        # weight_index maps structure position -> coefficient slot; with the
        # target last the mapping is the identity on 0..2.
        eq.weights_internal = np.array([1.0, 0.0, 3.0, 0.0])
        eq.weights_final = np.array([1.0, 0.0, 3.0, 0.0])
        return eq

    def test_the_zero_weighted_term_is_pruned(self):
        eq = self._with_a_zero_weight()
        eq.remove_zero_terms()
        assert len(eq.structure) == 3
        assert len(eq.weights_internal) == len(eq.structure)
        assert len(eq.weights_final) == len(eq.structure)

    def test_the_sparsity_caches_survive_the_prune(self):
        eq = self._with_a_zero_weight()
        eq._cached_sw_weights = 'converged'
        eq._cached_vc_score = 'converged'
        eq.remove_zero_terms()
        assert eq._cached_sw_weights == 'converged'
        assert eq._cached_vc_score == 'converged'

    def test_the_structure_keyed_caches_do_not(self):
        eq = self._with_a_zero_weight()
        eq._terms_labels_cache = 'stale'
        eq._gram_super = 'stale'
        eq._eval_cache = {0: 'stale'}
        eq.remove_zero_terms()
        assert eq._terms_labels_cache is None
        assert eq._gram_super is None
        assert eq._eval_cache == {}


# --------------------------------------------------------------------------- #
#  One author for the fitted magnitudes                                        #
# --------------------------------------------------------------------------- #
class TestCoeffCalcOwnsFinalWeights:

    @pytest.mark.parametrize('cls, physical', [(LASSOSparsity, False),
                                               (VWSRSparsity, True),
                                               (KneeSparsity, True)])
    def test_each_operator_declares_the_scale_it_fits_on(self, cls, physical):
        """The declaration is what ``LinRegBasedCoeffsEquation`` reads to
        choose between refitting and promoting."""
        assert cls.fits_physical_scale is physical

    def test_the_declaration_is_not_a_configurable_setting(self):
        """Overriding it from ``sparsity_kwargs`` would run the legacy min-max
        refit over physical coefficients, or skip it where LASSO needs it."""
        from epde.interface.search_config import sparsity_settings
        for cls in (LASSOSparsity, VWSRSparsity, KneeSparsity):
            assert 'fits_physical_scale' not in sparsity_settings(cls)

    def test_the_selector_pushes_the_declaration_onto_coeff_calc(self):
        from epde.operators.common.right_part_selection import EqRightPartSelector
        for cls, expected in ((LASSOSparsity, False), (VWSRSparsity, True)):
            selector = EqRightPartSelector([])
            coeff_calc = LinRegBasedCoeffsEquation([])
            selector.set_suboperators({'sparsity': cls([]),
                                       'coeff_calc': coeff_calc,
                                       'fitness_calculation': coeff_calc})
            assert coeff_calc.sparsity_fits_physical is expected

    def test_an_unwired_operator_defaults_to_the_legacy_refit(self):
        assert LinRegBasedCoeffsEquation.sparsity_fits_physical is False

    def test_promotion_copies_the_support_vector_unchanged(self):
        eq = _fitted_equation()
        eq.weights_final_evald = False
        eq._weights_final = None
        coeff_calc = LinRegBasedCoeffsEquation([])
        coeff_calc.sparsity_fits_physical = True
        coeff_calc.apply(eq, {})
        assert eq.weights_final_evald is True
        assert np.array_equal(eq.weights_final, eq.weights_internal)
        # A copy, not an alias: a later in-place edit of one must not move the
        # other.
        assert eq.weights_final is not eq.weights_internal

    def test_it_refuses_to_run_before_the_support_decision(self):
        eq = _fitted_equation()
        eq.weights_internal_evald = False
        coeff_calc = LinRegBasedCoeffsEquation([])
        coeff_calc.sparsity_fits_physical = True
        with pytest.raises(AssertionError, match='before evaluating'):
            coeff_calc.apply(eq, {})


# --------------------------------------------------------------------------- #
#  The invariant check, armed by EPDE_LOOP_STATS                               #
# --------------------------------------------------------------------------- #
class TestStateInvariants:
    """``structure`` is a plain slot -- no property setter -- and
    ``_validate_weight_layout`` fires only when a weight vector is ASSIGNED, so
    nothing stops an operator from reshaping the term list under weights that
    still claim to describe it. These checks run where the contract is consumed.
    """

    @pytest.fixture
    def armed(self, monkeypatch):
        from epde import _loop_stats
        monkeypatch.setattr(_loop_stats, 'enabled', lambda: True)

    def test_a_consistent_equation_passes(self, armed):
        _fitted_equation().assert_state_invariants('test')

    def test_it_is_a_no_op_while_disarmed(self, monkeypatch):
        """Disarmed explicitly, not by relying on the ambient default -- the
        A/B harness runs the whole suite with EPDE_LOOP_STATS=1."""
        from epde import _loop_stats
        monkeypatch.setattr(_loop_stats, 'enabled', lambda: False)
        eq = _fitted_equation()
        eq.structure.pop()                      # weights now describe 3 of 2
        eq.assert_state_invariants('test')

    def test_a_structure_shrunk_under_a_live_fit_is_caught(self, armed):
        eq = _fitted_equation()
        eq.structure.pop()
        with pytest.raises(AssertionError, match='structure changed'):
            eq.assert_state_invariants('test')

    def test_magnitudes_may_not_outlive_the_support_decision(self, armed):
        eq = _fitted_equation()
        eq.weights_internal_evald = False
        with pytest.raises(AssertionError, match='without a support decision'):
            eq.assert_state_invariants('test')

    def test_a_raised_flag_with_no_vector_is_caught(self, armed):
        eq = _fitted_equation()
        eq._weights_final = None
        with pytest.raises(AssertionError, match='no vector behind it'):
            eq.assert_state_invariants('test')

    def test_weights_without_an_installed_target_are_caught(self, armed):
        """Both vectors are indexed relative to the target position, so a
        weight claim with no target is not merely stale, it is unreadable."""
        eq = _fitted_equation()
        eq._target_term = None
        with pytest.raises(AssertionError, match='no installed target'):
            eq.assert_state_invariants('test')

    def test_a_reset_equation_passes(self, armed):
        """The contract must hold at BOTH ends: fully fitted, and fully
        cleared. A reset drops the target with the weights, so nothing is left
        claiming an index into a structure it cannot address."""
        eq = _fitted_equation()
        eq.reset_state(True)
        eq.assert_state_invariants('test')
