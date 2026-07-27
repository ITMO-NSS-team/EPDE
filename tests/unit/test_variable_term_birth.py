#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Tests for the variable per-equation birth size, the ``term_number``
property, and the ``terms_number`` -> ``max_terms_number`` metaparameter-key
back-compat shim.

Pins the post-refactor contract:
  * ``Equation.__init__`` draws a per-equation birth term count in
    ``[2, max_terms_number]`` instead of always filling to the max;
  * ``Equation.term_number`` reports the live ``len(self.structure)``;
  * ``_normalize_metaparameters`` accepts the legacy ``terms_number`` key with a
    one-time DeprecationWarning and is idempotent.
"""

import copy
import warnings
from collections import OrderedDict

import numpy as np
import pytest

import epde.globals as global_var
from epde.cache.cache import upload_grids, upload_simple_tokens
from epde.evaluators import simple_function_evaluator
from epde.interface.token_family import TFPool, TokenFamily
from epde.structure.main_structures import Equation, _normalize_metaparameters


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Six distinct single-factor tokens => up to six unique single-factor terms, so
# a birth draw anywhere in [2, 6] is always satisfiable from the pool (no
# early exhaustion break) and the realised term count equals the draw.
_DERIV_NAMES = ['u', 'du/dx0', 'd2u/dx0', 'd3u/dx0', 'd4u/dx0', 'd5u/dx0']
_DERIV_ORDERS = [[None], [0], [0, 0], [0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0, 0]]
_MAX = 6


@pytest.fixture(scope="module")
def rich_pool():
    grid = np.linspace(0.0, 4.0 * np.pi, 50)
    tensors = np.stack([
        np.sin(grid), np.cos(grid), -np.sin(grid),
        -np.cos(grid), np.sin(2.0 * grid), np.cos(2.0 * grid),
    ], axis=0)

    global_var.init_caches(set_grids=True)
    global_var.set_time_axis(0)
    global_var.init_verbose(show_warnings=False)
    global_var.tensor_cache.memory_usage_properties(
        obj_test_case=tensors[0], mem_for_cache_frac=5)
    global_var.grid_cache.memory_usage_properties(
        obj_test_case=grid, mem_for_cache_frac=5)

    upload_grids([grid], global_var.grid_cache)
    upload_simple_tokens(_DERIV_NAMES, global_var.tensor_cache, tensors)
    global_var.tensor_cache.use_structural()

    u_family = TokenFamily('u', variable='u', family_of_derivs=True)
    u_family.set_status(demands_equation=True, unique_specific_token=False,
                        unique_token_type=False, s_and_d_merged=False,
                        meaningful=True)
    u_family.set_params(_DERIV_NAMES, OrderedDict([('power', (1, 1))]),
                        {'power': 0}, _DERIV_ORDERS)
    u_family.set_evaluator(simple_function_evaluator)

    return TFPool([u_family])


def _metaparams(max_terms=_MAX):
    return {
        'sparsity':            {'optimizable': True,  'value': 1.0},
        'max_terms_number':    {'optimizable': False, 'value': max_terms},
        'max_factors_in_term': {'optimizable': False, 'value': 1},
    }


# ---------------------------------------------------------------------------
# 1. Variable per-equation birth size
# ---------------------------------------------------------------------------

class TestVariableBirth:
    def test_birth_count_within_bounds(self, rich_pool):
        np.random.seed(12345)
        for _ in range(40):
            eq = Equation(rich_pool, basic_structure=[], var_to_explain='u',
                          metaparameters=copy.deepcopy(_metaparams()))
            assert 2 <= eq.term_number <= _MAX

    def test_birth_count_varies_across_equations(self, rich_pool):
        np.random.seed(2024)
        counts = {
            Equation(rich_pool, basic_structure=[], var_to_explain='u',
                     metaparameters=copy.deepcopy(_metaparams())).term_number
            for _ in range(40)
        }
        # Not pinned to the max any more: the draw spreads across [2, 6].
        assert len(counts) > 1
        assert max(counts) <= _MAX and min(counts) >= 2

    def test_term_number_tracks_structure_length(self, rich_pool):
        np.random.seed(7)
        eq = Equation(rich_pool, basic_structure=[], var_to_explain='u',
                      metaparameters=copy.deepcopy(_metaparams()))
        assert eq.term_number == len(eq.structure)

    def test_floor_never_below_immutable_head(self, rich_pool):
        # Three mandatory basic terms => birth count is drawn in [3, 6].
        np.random.seed(99)
        for _ in range(20):
            eq = Equation(rich_pool,
                          basic_structure=['u', 'du/dx0', 'd2u/dx0'],
                          var_to_explain='u',
                          metaparameters=copy.deepcopy(_metaparams()))
            assert 3 <= eq.term_number <= _MAX

    def test_max_equal_immutable_gives_exact_count(self, rich_pool):
        # n_immutable == max => empty fill loop, exact-size equation (the
        # translated-equation / current-behaviour preservation path).
        eq = Equation(rich_pool,
                      basic_structure=['u', 'du/dx0', 'd2u/dx0'],
                      var_to_explain='u',
                      metaparameters=copy.deepcopy(_metaparams(max_terms=3)))
        assert eq.term_number == 3


# ---------------------------------------------------------------------------
# 2. Back-compat shim: terms_number -> max_terms_number
# ---------------------------------------------------------------------------

class TestMetaparameterShim:
    def test_legacy_key_renamed_with_warning(self):
        mp = {'terms_number':        {'optimizable': False, 'value': 4},
              'max_factors_in_term': {'optimizable': False, 'value': 1}}
        with pytest.warns(DeprecationWarning):
            _normalize_metaparameters(mp)
        assert 'terms_number' not in mp
        assert mp['max_terms_number']['value'] == 4

    def test_idempotent_no_warning_second_time(self):
        mp = {'max_terms_number': {'optimizable': False, 'value': 4}}
        with warnings.catch_warnings():
            warnings.simplefilter('error', DeprecationWarning)
            _normalize_metaparameters(mp)  # must NOT warn
        assert mp['max_terms_number']['value'] == 4
        assert 'terms_number' not in mp

    def test_new_key_wins_when_both_present(self):
        mp = {'terms_number':     {'optimizable': False, 'value': 1},
              'max_terms_number': {'optimizable': False, 'value': 9}}
        # No warning: nothing was renamed, only the stale legacy key dropped.
        with warnings.catch_warnings():
            warnings.simplefilter('error', DeprecationWarning)
            _normalize_metaparameters(mp)
        assert 'terms_number' not in mp
        assert mp['max_terms_number']['value'] == 9

    def test_equation_accepts_legacy_key(self, rich_pool):
        mp = {'sparsity':            {'optimizable': True,  'value': 1.0},
              'terms_number':        {'optimizable': False, 'value': 4},
              'max_factors_in_term': {'optimizable': False, 'value': 1}}
        with pytest.warns(DeprecationWarning):
            eq = Equation(rich_pool, basic_structure=[], var_to_explain='u',
                          metaparameters=mp)
        assert 'max_terms_number' in eq.metaparameters
        assert 'terms_number' not in eq.metaparameters
        assert 2 <= eq.term_number <= 4
