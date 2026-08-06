#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Characterization tests for ``epde/structure/main_structures.py``.

These tests pin CURRENT behavior (correct or buggy) so the upcoming refactoring
phases can detect regressions. See ``PLAN_main_structures_refinement.md`` for
the staged roadmap they support.

Some tests pin observed bugs (most prominently the mutable default
metaparameters in ``Equation.__init__`` at l.391-395). Phase 2 fixes those
bugs; the relevant test expectations will flip in the same commit that lands
each fix.
"""

import copy
from collections import OrderedDict

import numpy as np
import pytest

import epde.globals as global_var
from epde.cache.cache import upload_grids, upload_simple_tokens
from epde.evaluators import simple_function_evaluator
from epde.interface.equation_translator import translate_equation
from epde.interface.token_family import TFPool, TokenFamily
from epde.structure.main_structures import Equation


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def basic_pool():
    """A minimal pool with two derivative-family tokens (``u`` and ``du/dx0``).

    Avoids ANN training and heavy preprocessing — we only need a pool whose
    factors have valid cache labels, evaluator linkage, and pool back-references
    so that ``Term``/``Equation``/``SoEq`` construction and deepcopy succeed.
    """
    grid = np.linspace(0.0, 4.0 * np.pi, 50)
    u = np.sin(grid)
    du = np.cos(grid)

    global_var.init_caches(set_grids=True)
    global_var.set_time_axis(0)
    global_var.init_verbose(show_warnings=False)
    global_var.tensor_cache.memory_usage_properties(
        obj_test_case=u, mem_for_cache_frac=5)
    global_var.grid_cache.memory_usage_properties(
        obj_test_case=grid, mem_for_cache_frac=5)

    upload_grids([grid], global_var.grid_cache)

    deriv_names = ['u', 'du/dx0']
    deriv_orders = [[None,], [0,]]
    deriv_tensors = np.stack([u, du], axis=0)
    upload_simple_tokens(deriv_names, global_var.tensor_cache, deriv_tensors)
    global_var.tensor_cache.use_structural()

    u_family = TokenFamily('u', variable='u', family_of_derivs=True)
    u_family.set_status(demands_equation=True, unique_specific_token=False,
                        unique_token_type=False, s_and_d_merged=False,
                        meaningful=True)
    u_family.set_params(deriv_names, OrderedDict([('power', (1, 1))]),
                        {'power': 0}, deriv_orders)
    u_family.set_evaluator(simple_function_evaluator)

    return TFPool([u_family])


def _build_soeq(pool):
    text = '1.0 * u{power: 1} + 0.0 = du/dx0{power: 1}'
    soeq = translate_equation(text, pool, all_vars=['u'])
    # translate_equation assigns weights via the setter but does not flip
    # weights_internal_evald. Set it so terms_labels_without_power can run
    # without raising AttributeError ("Internal weights called before init").
    eq = soeq.vals['u']
    eq.weights_internal_evald = True
    return soeq


@pytest.fixture
def soeq(basic_pool):
    return _build_soeq(basic_pool)


@pytest.fixture
def equation(soeq):
    return soeq.vals['u']


@pytest.fixture
def term(equation):
    return equation.structure[0]


# ---------------------------------------------------------------------------
# 1. TestTermDeepcopy
# ---------------------------------------------------------------------------

class TestTermDeepcopy:
    def test_returns_distinct_object(self, term):
        copy_t = copy.deepcopy(term)
        assert id(copy_t) != id(term)

    def test_equal_to_original(self, term):
        copy_t = copy.deepcopy(term)
        assert copy_t == term

    def test_structure_is_fresh(self, term):
        copy_t = copy.deepcopy(term)
        assert copy_t.structure is not term.structure
        for c_factor, o_factor in zip(copy_t.structure, term.structure):
            assert c_factor is not o_factor

    def test_preserves_name(self, term):
        copy_t = copy.deepcopy(term)
        assert copy_t.name == term.name

    def test_preserves_cache_label(self, term):
        copy_t = copy.deepcopy(term)
        assert copy_t.cache_label == term.cache_label


# ---------------------------------------------------------------------------
# 2. TestEquationDeepcopy
# ---------------------------------------------------------------------------

class TestEquationDeepcopy:
    def test_returns_distinct_object(self, equation):
        copy_e = copy.deepcopy(equation)
        assert id(copy_e) != id(equation)

    def test_equal_to_original(self, equation):
        copy_e = copy.deepcopy(equation)
        assert copy_e == equation

    def test_structure_is_fresh(self, equation):
        copy_e = copy.deepcopy(equation)
        assert copy_e.structure is not equation.structure
        for c_term, o_term in zip(copy_e.structure, equation.structure):
            assert c_term is not o_term

    def test_eval_cache_after_deepcopy_is_fresh_dict(self, equation):
        """Pin: __deepcopy__ traverses the _eval_cache slot, so the copy
        owns its own dict (initially empty, equal to source's empty dict).
        """
        copy_e = copy.deepcopy(equation)
        assert copy_e._eval_cache is not equation._eval_cache
        assert copy_e._eval_cache == equation._eval_cache


# ---------------------------------------------------------------------------
# 3. TestSoEqDeepcopy
# ---------------------------------------------------------------------------

class TestSoEqDeepcopy:
    def test_returns_distinct_object(self, soeq):
        copy_s = copy.deepcopy(soeq)
        assert id(copy_s) != id(soeq)

    def test_dict_attrs_present(self, soeq):
        """Pin current dual-traversal: __dict__ keys are all carried over."""
        copy_s = copy.deepcopy(soeq)
        for key in soeq.__dict__:
            assert hasattr(copy_s, key)

    def test_vals_independent(self, soeq):
        """The chromosome is itself deepcopied, not aliased."""
        copy_s = copy.deepcopy(soeq)
        assert copy_s.vals is not soeq.vals


# ---------------------------------------------------------------------------
# 4. TestEquationLabelProperties
# ---------------------------------------------------------------------------

class TestEquationLabelProperties:
    def test_terms_labels_is_frozenset_of_frozensets(self, equation):
        labels = equation.terms_labels
        assert isinstance(labels, frozenset)
        for inner in labels:
            assert isinstance(inner, frozenset)

    def test_terms_labels_count_matches_unique_terms(self, equation):
        # Two distinct terms (u and du/dx0) → two frozenset entries.
        assert len(equation.terms_labels) == len(equation.structure)

    def test_terms_labels_without_power_is_frozenset(self, equation):
        labels = equation.terms_labels_without_power
        assert isinstance(labels, frozenset)

    def test_terms_labels_stable_across_calls(self, equation):
        # Calling twice in a row returns equal results (no hidden state).
        first = equation.terms_labels
        second = equation.terms_labels
        assert first == second


# ---------------------------------------------------------------------------
# 7. TestEquationLabelsAfterTermMutation
#
# terms_labels / terms_labels_without_power are memoized in slot caches
# (_terms_labels_cache, _terms_labels_without_power_cache). Mutation paths
# that touch self.structure or its terms must call _invalidate_label_cache()
# afterward (15 known call sites cover this). These tests pin the new
# contract: fresh result on first access populates the cache, repeated
# access returns the same frozenset, and invalidation drops the cache.
# ---------------------------------------------------------------------------

class TestEquationLabelsAfterTermMutation:
    def test_terms_labels_reflect_structure_append(self, equation):
        before = equation.terms_labels
        equation.structure.append(copy.deepcopy(equation.structure[0]))
        # Manual structure append bypasses Equation's mutation API and the
        # cache; an explicit invalidation is the contract for callers that
        # touch self.structure directly.
        equation._invalidate_label_cache()
        after = equation.terms_labels
        # frozenset of frozensets — appending a duplicate keeps the frozenset
        # the same size (set semantics) but len(structure) grows.
        assert len(after) <= len(before) + 1
        assert len(after) <= len(equation.structure)

    def test_terms_labels_populates_cache(self, equation):
        # First access computes and stores; subsequent accesses return the
        # identical frozenset (cache hit, not a recomputation).
        assert equation._terms_labels_cache is None
        first = equation.terms_labels
        assert equation._terms_labels_cache is first
        second = equation.terms_labels
        assert second is first

    def test_invalidate_helper_drops_cache(self, equation):
        # Calling the helper on a populated equation drops both caches, so
        # the next read recomputes from the current structure.
        _ = equation.terms_labels
        _ = equation.terms_labels_without_power
        assert equation._terms_labels_cache is not None
        assert equation._terms_labels_without_power_cache is not None
        equation._invalidate_label_cache()
        assert equation._terms_labels_cache is None
        assert equation._terms_labels_without_power_cache is None


# ---------------------------------------------------------------------------
# 5. TestEquationDefaultMetaparameters
#
# After Phase 2: each Equation gets its OWN deep-copied default metaparameters
# dict, so mutating one cannot leak into another. Pre-Phase-2 this test
# asserted the opposite (shared mutation). The flip is the visible artifact
# that the bug at the old l.391-395 has been fixed.
# ---------------------------------------------------------------------------

class TestEquationDefaultMetaparameters:
    def test_two_equations_have_independent_default_metaparameters(self, basic_pool):
        # Default max_terms_number is 5; passing five basic terms skips the
        # random-padding loop entirely (range(5, 5) is empty).
        eq1 = Equation(basic_pool, basic_structure=['u'] * 5,
                       var_to_explain='u')
        eq2 = Equation(basic_pool, basic_structure=['u'] * 5,
                       var_to_explain='u')
        # Each sees the documented default value.
        assert eq1.metaparameters['sparsity']['value'] == 1.0
        assert eq2.metaparameters['sparsity']['value'] == 1.0

        eq1.metaparameters['sparsity']['value'] = 999.0
        # Mutation MUST stay local — the dict objects are independent.
        assert eq2.metaparameters['sparsity']['value'] == 1.0
        assert eq1.metaparameters is not eq2.metaparameters


# ---------------------------------------------------------------------------
# 6. TestTargetTerm
#
# The right-part target is an identity-tracked Term (``_target_term`` slot),
# exposed via the ``target`` (Term-or-None) and ``target_idx`` (derived int-or-
# None) properties. Pins: identity survives drops/reorders of other terms, a
# dropped/orphaned target degrades to None (no stale index -> the IndexError
# regression is gone), deepcopy preserves identity, and the target resets on
# structure change.
# ---------------------------------------------------------------------------

class TestTargetTerm:
    def _pos(self, equation, term):
        return next(i for i, x in enumerate(equation.structure) if x is term)

    def test_target_is_a_structure_term(self, equation):
        assert equation.target is not None
        assert equation.target is equation.structure[equation.target_idx]

    def test_target_tracks_identity_through_drop(self, equation):
        t = equation.target
        victim = next(term for term in equation.structure if term is not t)
        before = self._pos(equation, victim) < self._pos(equation, t)
        old_idx = equation.target_idx
        equation.structure = [x for x in equation.structure if x is not victim]
        equation._invalidate_label_cache()
        assert equation.target is t                       # identity preserved
        assert equation.target_idx == (old_idx - 1 if before else old_idx)

    def test_dropping_target_yields_none(self, equation):
        t = equation.target
        equation.structure = [x for x in equation.structure if x is not t]
        equation._invalidate_label_cache()
        assert equation.target is None
        assert equation.target_idx is None               # no dangling index

    def test_orphan_target_degrades_to_none(self, equation):
        # Regression for the original IndexError: a target Term that is no
        # longer in the structure (e.g. left over after randomize() rebuilt it
        # smaller) must NOT surface as an out-of-range integer index.
        equation._target_term = copy.deepcopy(equation.structure[0])  # not ``is`` any term
        assert equation.target is None
        assert equation.target_idx is None

    def test_deepcopy_preserves_target_identity(self, equation):
        eq2 = copy.deepcopy(equation)
        assert eq2.target is not equation.target          # deep-copied
        assert eq2.target is eq2.structure[eq2.target_idx]  # identity within the clone
        assert eq2.target_idx == equation.target_idx

    def test_reset_state_nulls_target_only_with_right_part(self, equation):
        assert equation.target is not None
        equation.reset_state(reset_right_part=False)
        assert equation.target is not None
        equation.reset_state(reset_right_part=True)
        assert equation.target is None

    def test_randomize_nulls_target(self, equation):
        assert equation.target is not None
        equation.randomize()
        assert equation.target is None

    def test_clone_shell_has_no_target(self, equation):
        shell = equation.clone_shell()
        assert shell.structure == []
        assert shell.target is None
        assert shell.target_idx is None

    def test_target_idx_setter_anchors_to_term(self, equation):
        equation.target_idx = 0
        t0 = equation.structure[0]
        assert equation.target is t0
        equation.structure.insert(0, copy.deepcopy(equation.structure[-1]))
        equation._invalidate_label_cache()
        assert equation.target is t0                      # re-anchored, not literal 0
        assert equation.target_idx == 1

    def test_remove_zero_terms_keeps_target_identity(self, equation):
        t = equation.target
        assert len(equation.structure) >= 2
        # Zero the single non-target term's internal weight (a 2-term equation
        # maps that term to weights_internal[0] regardless of target side).
        equation.weights_internal = np.array([0.0])
        equation.weights_internal_evald = True
        equation.remove_zero_terms()
        assert equation.target is t                       # target Term survived by identity
        assert equation.target_idx == self._pos(equation, t)
        assert t in equation.structure


# ---------------------------------------------------------------------------
# 9. TestSparsityWeightsFinalConventions (refactoring plan, Phase 1)
#
# The two sparsity operators emit DIFFERENT weights_final layouts:
#   * LASSOSparsity (sparsity.py, ``np.append(nonzero, intercept)``): the
#     intercept slot is ALWAYS present -- len == nnz + 1 even for a zero
#     intercept. weights_internal holds coefficients only (no intercept slot).
#   * VWSRSparsity (``[w for w in weights_internal if w != 0]``): a ZERO
#     intercept is dropped -- weights_final is then nnz-length with no
#     intercept slot. weights_internal = [*coef, intercept] and its LAST entry
#     acts as the intercept-presence flag the discrepancy fillers branch on
#     (see Discrepancy._compute_wape).
# ``Discrepancy._compute_l2`` reads ``weights_final[-1]`` as the intercept
# unconditionally, which is coherent only under the LASSO pairing. These tests
# pin both conventions so any future unification must consciously flip them.
# ---------------------------------------------------------------------------

class TestSparsityWeightsFinalConventions:

    @staticmethod
    def _prime_grid_globals():
        # Neither fixture path sets the weak-form weighting nor the boundary-
        # trimmed shape; the sparsity operators read both from the grid cache.
        global_var.grid_cache.g_func = np.ones(50)
        global_var.grid_cache.inner_shape = np.array([50])

    def test_lasso_weights_final_always_has_intercept_slot(self, equation, monkeypatch):
        import epde.operators.common.sparsity as sparsity_mod

        class _StubLasso:
            def __init__(self, **kwargs):
                pass

            def fit(self, X, y, sample_weight):
                self.coef_ = np.array([0.7])
                self.intercept_ = 0.0          # ZERO intercept

        monkeypatch.setattr(sparsity_mod, 'Lasso', _StubLasso)
        self._prime_grid_globals()
        equation.metaparameters[('sparsity', 'u')] = {'value': 1.0}

        sparsity_mod.LASSOSparsity().apply(equation, arguments={})

        # Legacy layout: weights_internal = coefficients only (no intercept).
        assert len(equation.weights_internal) == 1
        # Intercept slot survives even at 0.0 -> len == nnz + 1.
        assert len(equation.weights_final) == 2
        assert equation.weights_final[-1] == 0.0

    def test_vwsr_weights_final_omits_zero_intercept(self, equation, monkeypatch):
        import epde.operators.common.sparsity as sparsity_mod

        class _StubPIL:
            def __init__(self, **kwargs):
                pass

            def fit(self, X, y, sample_weights, gram_setup=None):
                self.coef_ = np.array([0.7])
                self.intercept_ = 0.0          # ZERO intercept
                self.cached_weights_ = None    # vcoef mode leaves this None
                self.cached_vc_score_ = np.array([0.01])

        monkeypatch.setattr(sparsity_mod, 'PhysicsInformedLasso', _StubPIL)
        self._prime_grid_globals()

        sparsity_mod.VWSRSparsity().apply(equation, arguments={})

        # VWSR layout: weights_internal = [*coef, intercept].
        assert np.array_equal(equation.weights_internal, np.array([0.7, 0.0]))
        # Zero intercept dropped -> nnz-length weights_final, NO intercept slot.
        assert np.array_equal(equation.weights_final, np.array([0.7]))
        # weights_internal[-1] is the presence flag the fillers consume.
        assert equation.weights_internal[-1] == 0.0
        # The stability caches land on the equation (sparsity.py:651-652).
        assert equation._cached_sw_weights is None
        assert np.array_equal(equation._cached_vc_score, np.array([0.01]))

    def test_vwsr_weights_final_keeps_nonzero_intercept(self, equation, monkeypatch):
        import epde.operators.common.sparsity as sparsity_mod

        class _StubPIL:
            def __init__(self, **kwargs):
                pass

            def fit(self, X, y, sample_weights, gram_setup=None):
                self.coef_ = np.array([0.7])
                self.intercept_ = 0.3          # NON-zero intercept
                self.cached_weights_ = None
                self.cached_vc_score_ = np.array([0.01])

        monkeypatch.setattr(sparsity_mod, 'PhysicsInformedLasso', _StubPIL)
        self._prime_grid_globals()

        sparsity_mod.VWSRSparsity().apply(equation, arguments={})

        assert np.array_equal(equation.weights_final, np.array([0.7, 0.3]))
        assert equation.weights_internal[-1] == 0.3


# ---------------------------------------------------------------------------
# 10. TestSparsityCachePreservation (refactoring plan, Phase 1 / CHECK #1)
#
# ``_cached_sw_weights`` / ``_cached_vc_score`` are recomputed by
# ``PhysicsInformedLasso.fit`` on the CONVERGED active mask (sparsity.py:
# 449-471) -- their columns cover only the surviving features (+ intercept).
# ``remove_zero_terms`` drops exactly the zero-weight terms, so the caches
# still align with the pruned structure and are DELIBERATELY preserved across
# both ``remove_zero_terms`` and ``_invalidate_label_cache``; wiping them
# there would force a recompute on a differently-windowed path right before
# ``Instability.compute`` reads them (fitness.py:141 -> 155-157). These tests
# pin that preservation policy.
# ---------------------------------------------------------------------------

class TestSparsityCachePreservation:

    def test_invalidate_label_cache_preserves_sparsity_caches(self, equation):
        sw_sentinel = np.ones((5, 2))
        vc_sentinel = np.array([0.01, 0.02])
        equation._cached_sw_weights = sw_sentinel
        equation._cached_vc_score = vc_sentinel
        equation._gram_super = {'mode': 'vcoef'}
        _ = equation.terms_labels
        _ = equation.terms_labels_without_power
        equation._eval_cache = {(True, False, True, 0): 'sentinel'}

        equation._invalidate_label_cache()

        # Structure-keyed caches wiped ...
        assert equation._terms_labels_cache is None
        assert equation._terms_labels_without_power_cache is None
        assert equation._eval_cache == {}
        assert equation._gram_super is None
        # ... the sparsity caches survive by identity (CHECK #1 policy).
        assert equation._cached_sw_weights is sw_sentinel
        assert equation._cached_vc_score is vc_sentinel

    def test_remove_zero_terms_preserves_sparsity_caches(self, equation):
        sw_sentinel = np.ones((5, 1))
        vc_sentinel = np.array([0.02])
        equation._cached_sw_weights = sw_sentinel
        equation._cached_vc_score = vc_sentinel
        # VWSR layout on the 2-term fixture: [feature_u, intercept]; the
        # zero feature weight makes remove_zero_terms drop the u term.
        equation.weights_internal = np.array([0.0, 0.5])
        equation.weights_internal_evald = True

        equation.remove_zero_terms()

        assert len(equation.structure) == 1               # u dropped, target kept
        assert equation._cached_sw_weights is sw_sentinel
        assert equation._cached_vc_score is vc_sentinel

    def test_cached_sw_weights_column_alignment_post_prune(self, basic_pool):
        from epde.structure.main_structures import Term

        soeq = _build_soeq(basic_pool)
        equation = soeq.vals['u']
        equation.weights_internal_evald = True
        # Third term (u * du/dx0) so the prune leaves a non-degenerate
        # feature set: structure = [u, u*du/dx0, du/dx0(target)].
        extra = Term(basic_pool, passed_term=['u', 'du/dx0'],
                     max_factors_in_term=2)
        equation.structure.insert(1, extra)
        equation._invalidate_label_cache()

        # Converged VWSR state: u kept (0.5), extra zeroed, intercept 0.3.
        equation.weights_internal = np.array([0.5, 0.0, 0.3])
        equation.weights_internal_evald = True
        # Axis-mode cache on the converged ACTIVE mask: columns = surviving
        # features + intercept = 2.
        equation._cached_sw_weights = np.ones((5, 2))

        equation.remove_zero_terms()

        # Preserved cache still aligns: columns == (features + intercept)
        # == len(structure) after the prune ([u, target] -> 1 feature + 1).
        assert len(equation.structure) == 2
        assert equation._cached_sw_weights.shape[-1] == len(equation.structure)
        assert np.array_equal(equation.weights_internal, np.array([0.5, 0.3]))


# ---------------------------------------------------------------------------
# 10b. TestCacheFieldRegistry (refactoring plan, Phase 3)
#
# ``_EQ_CACHE_FIELDS`` is the single source of truth for Equation cache slots
# and their invalidation policy; reset_state/_invalidate_label_cache/
# __deepcopy__/clone_shell all derive from it. These tests hold the registry,
# __slots__, and the four consumers in sync.
# ---------------------------------------------------------------------------

class TestCacheFieldRegistry:

    def test_cache_registry_covers_all_cache_slots(self):
        from epde.structure.main_structures import _EQ_CACHE_FIELDS
        cache_slots = {
            s for s in Equation.__slots__
            if s.startswith('_cached_') or s in (
                '_eval_cache', '_gram_super',
                '_terms_labels_cache', '_terms_labels_without_power_cache')
        }
        assert {f for f, _ in _EQ_CACHE_FIELDS} == cache_slots
        assert all(p in ('structure', 'reset-only') for _, p in _EQ_CACHE_FIELDS)

    def test_reset_state_wipes_every_registry_field(self, equation):
        from epde.structure.main_structures import _EQ_CACHE_FIELDS
        for field, _ in _EQ_CACHE_FIELDS:
            setattr(equation, field, {'sentinel': 1} if field == '_eval_cache'
                    else np.array([1.0]))
        equation.reset_state(True)
        for field, _ in _EQ_CACHE_FIELDS:
            expected = {} if field == '_eval_cache' else None
            assert getattr(equation, field) == expected if field == '_eval_cache' \
                else getattr(equation, field) is None

    def test_invalidate_label_cache_wipes_exactly_structure_policy_fields(self, equation):
        from epde.structure.main_structures import _EQ_CACHE_FIELDS
        sentinels = {}
        for field, _ in _EQ_CACHE_FIELDS:
            val = {'sentinel': 1} if field == '_eval_cache' else np.array([1.0])
            setattr(equation, field, val)
            sentinels[field] = val
        equation._invalidate_label_cache()
        for field, policy in _EQ_CACHE_FIELDS:
            if policy == 'structure':
                expected = {} if field == '_eval_cache' else None
                got = getattr(equation, field)
                assert (got == expected) if field == '_eval_cache' else (got is None)
            else:
                assert getattr(equation, field) is sentinels[field]

    def test_deepcopy_and_clone_shell_skip_all_registry_fields(self, equation):
        from epde.structure.main_structures import _EQ_CACHE_FIELDS
        for field, _ in _EQ_CACHE_FIELDS:
            setattr(equation, field, {'sentinel': 1} if field == '_eval_cache'
                    else np.array([1.0]))
        for cloned in (copy.deepcopy(equation), equation.clone_shell()):
            for field, _ in _EQ_CACHE_FIELDS:
                got = getattr(cloned, field)
                if field == '_eval_cache':
                    assert got == {}           # fresh dict, never the source's
                    assert got is not equation._eval_cache
                else:
                    assert got is None


# ---------------------------------------------------------------------------
# 10c. Phase-5 flagged fixes
# ---------------------------------------------------------------------------

class TestPhase5Fixes:

    def test_chromosome_eq_key_mismatch_returns_false(self):
        # Pre-fix, Chromosome.__eq__ compared self's keys with THEMSELVES,
        # so mismatched chromosomes fell through to a KeyError. Now a key
        # mismatch is an ordinary inequality.
        from epde.structure.encoding import Chromosome
        a = Chromosome.__new__(Chromosome)
        b = Chromosome.__new__(Chromosome)
        a.chromosome = {'u': 1}
        b.chromosome = {'v': 1}
        assert (a == b) is False
        b.chromosome = {'u': 1}
        assert (a == b) is True

    def test_alt_instability_cache_wiped_on_structure_change(self, equation):
        # survival/tile memo is keyed on the OLD structure -- unlike the
        # sparsity caches, nothing recomputes it on the converged mask, so
        # _invalidate_label_cache must wipe it.
        equation._cached_alt_instability = ('survival', 0.5)
        equation._invalidate_label_cache()
        assert equation._cached_alt_instability is None


# ---------------------------------------------------------------------------
# 11. TestResetStateSoftAsymmetry (refactoring plan, Phase 1 / item 2)
#
# ``reset_state(reset_right_part=False)`` clears ``weights_final_evald``
# UNCONDITIONALLY but leaves the ``_weights_final`` data in place (only the
# reset_right_part=True branch nulls it). The state is safe-by-guard: the
# ``weights_final`` property getter raises while the flag is down. Pinning
# this asymmetry -- "cleaning it up" by moving the flag reset into the
# if-block would CHANGE behavior (the flag would survive a soft reset).
# ---------------------------------------------------------------------------

class TestResetStateSoftAsymmetry:

    def test_soft_reset_keeps_final_weights_data_but_flags_unevald(self, equation):
        data = np.array([1.0, 0.0])
        equation.weights_final = data
        equation.weights_final_evald = True

        equation.reset_state(reset_right_part=False)

        assert equation.weights_final_evald is False
        assert equation._weights_final is data            # data survives
        with pytest.raises(AttributeError):
            _ = equation.weights_final                    # guard holds

    def test_hard_reset_also_clears_final_weights_data(self, equation):
        equation.weights_final = np.array([1.0, 0.0])
        equation.weights_final_evald = True

        equation.reset_state(reset_right_part=True)

        assert equation.weights_final_evald is False
        assert equation._weights_final is None
