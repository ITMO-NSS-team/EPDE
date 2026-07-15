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
# 6. TestRenameAliases (Phase 3)
#
# Pin the alias contract: deprecated old names delegate to new names with
# identical results. If a future commit drops an alias, this test catches it.
# ---------------------------------------------------------------------------

class TestRenameAliases:
    # Term-level term_label / term_label_without_power aliases were
    # removed; all callers route through factors_labels(_without_power)
    # directly. SoEq still aliases equations_labels(_without_power) to
    # keep the MOEA/D-side history-membership API stable.
    def test_soeq_alias_equations_labels(self, soeq):
        assert soeq.equations_labels == soeq.terms_labels

    def test_soeq_alias_equations_labels_without_power(self, soeq):
        assert soeq.equations_labels_without_power == soeq.terms_labels_without_power


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

    def test_factors_labels_alias_on_equation(self, equation):
        # Phase 3 added factors_labels on Term; mutations.py:127 also reads
        # it on Equation (treating the names as interchangeable). Pin the alias.
        assert equation.factors_labels == equation.terms_labels
        assert equation.factors_labels_without_power == equation.terms_labels_without_power


# ---------------------------------------------------------------------------
# 8. TestFilterTokensByRightPartExhaustion (Phase 6)
#
# Pin: filter_tokens_by_right_part raises RuntimeError when it cannot find
# a unique term within the retry budget. Pre-Phase-6 the function looped
# forever (or warned and continued); Phase 6 caps retries with a hard fail.
# ---------------------------------------------------------------------------

class TestFilterTokensByRightPartExhaustion:
    def test_raises_runtimeerror_on_exhaustion(self, equation):
        import warnings as _w

        # The deprecated function reads factor.status['unique_for_right_part'];
        # patch it onto our test factors (the fixture uses the modern token-
        # family schema where this key is absent).
        for t in equation.structure:
            for f in t.structure:
                f.status['unique_for_right_part'] = False

        # Force a duplicate so terms_labels never matches len(structure)
        # — guaranteeing the loop never breaks out via success.
        equation.structure.append(copy.deepcopy(equation.structure[0]))
        equation._invalidate_label_cache()

        target = equation.structure[equation.target_idx]
        candidate = equation.structure[0]

        with _w.catch_warnings():
            _w.simplefilter('ignore', DeprecationWarning)
            with pytest.raises(RuntimeError, match='filter_tokens_by_right_part'):
                candidate.filter_tokens_by_right_part(
                    target, equation, equation_position=0, max_retries=1)


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
