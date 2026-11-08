#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""``'assignment'`` mode's ``selected_terms_max_number``
(``search_space.system_rps_params``): what an equation holds and what its right
part keeps are two numbers.

``equation_terms_max_number`` bounds the structure -- the terms an equation
holds to choose from -- and the new cap bounds the columns the right-hand side
keeps: each term counts 1 whatever its factors, the constant counts 1, the
target is not counted. Only ``system_rps_mode='assignment'`` has it. Pinned
here:

* the settings and their refusals, shared by the config load and ``fit``;
* the listed-subset tables hold the dense tables' values at those bitmasks;
* no equation keeps more columns than the cap -- no variant on any equation's
  front, so it holds whichever variant the instability metric picks -- and a cap at or above
  ``equation_terms_max_number`` IS the uncapped search, bit for bit (a cap at
  or above the column count alone gives the same subsets, but the partition
  metrics then read the narrower reference width);
* the cost ceiling counts subsets, so a structure may hold more than 12
  columns under the cap;
* the reference width of the partition-based instability metrics follows the
  smaller of the two caps.
"""

import numpy as np
import pytest

import epde.operators.common.right_part_selection as rps
from epde.interface.search_config import (ASSIGNMENT_MAX_SUBSETS, assignment_subset_count,
                                          check_rps_params, load_search_config,
                                          resolve_rps_params)
from epde.operators.common.subset_selection import instability_table, subset_table
from epde.operators.common.survival import _reference_width
from epde.operators.multiobjective.moeadd_specific import (_debug_assert_rps_unique,
                                                           has_degenerate_equation)

from conftest import using_config
from test_assignment_rps import _fresh, _soeq_selector
from test_duplicate_equation_guard import uv_pool  # noqa: F401

CAP = 'selected_terms_max_number'


def _cap(value):
    """The selected-terms cap as config overrides (``'assignment'`` mode's setting)."""
    return {'system_rps_params': {CAP: value}}


def _run(system, seed=0, **overrides):
    np.random.seed(seed)
    with using_config(system_rps_mode='assignment', **overrides):
        _soeq_selector().apply(system, {})
    return system


def _kept_columns(eq):
    """Non-zero right-hand-side columns, the constant included."""
    tgt = eq.target_idx
    wi = np.asarray(eq.weights_internal, dtype=float)
    terms = sum(wi[eq.weight_index(i, tgt)] != 0 for i in range(len(eq.structure)) if i != tgt)
    return int(terms) + int(wi[-1] != 0)


def _labels_and_weights(system):
    return (system.equations_labels,
            [np.asarray(eq.weights_final).tolist() for eq in system.vals])


def _run_with_fronts(pool, monkeypatch, seed=0, **overrides):
    """:func:`_run` on a fresh uv system, and the kept columns of EVERY variant
    the assignment search put on an equation's front (the candidates
    ``_equation_variants`` returns), each built as the pick would be -- inside
    the same config, so it is fitted under the settings that scored it. A
    variant the fitness host declines (None) is left out."""
    real, fronts = rps._equation_variants, []

    def recording(*args, **kwargs):
        candidates = real(*args, **kwargs)
        fronts.extend(candidates)
        return candidates
    system = _fresh(pool)
    np.random.seed(seed)
    with monkeypatch.context() as patch:
        patch.setattr(rps, '_equation_variants', recording)
        with using_config(system_rps_mode='assignment', **overrides):
            _soeq_selector().apply(system, {})
            built = [candidate.materialize() for candidate in fronts]
    return system, [_kept_columns(variant) for variant in built if variant is not None]


# ---------------------------------------------------------------------------
# Settings.
# ---------------------------------------------------------------------------

class TestTheSettings:

    @staticmethod
    def _space(terms=None, **params):
        overrides = {'system_rps_mode': 'assignment', 'system_rps_params': params}
        if terms is not None:
            overrides['equation_terms_max_number'] = terms
        return load_search_config(overrides=overrides).search_space

    def test_the_default_is_no_cap(self):
        assert self._space().system_rps_params[CAP] is None

    @pytest.mark.parametrize('value', [0, -2, 2.0, True, '3'])
    def test_anything_but_a_positive_integer_is_refused(self, value):
        with pytest.raises(ValueError, match=CAP):
            self._space(**{CAP: value})

    @pytest.mark.parametrize('mode', ['repair', 'plain'])
    def test_only_the_assignment_mode_has_it(self, mode):
        with pytest.raises(ValueError, match=f"{CAP} is a setting of system_rps_mode 'assignment'"):
            load_search_config(overrides={'system_rps_mode': mode, 'system_rps_params': {CAP: 3}})

    @pytest.mark.parametrize('old', ['equation_selected_terms_max_number', 'equation_screen_size',
                                     'variant_pick'])
    def test_the_old_names_raise_with_directions(self, old):
        import epde
        with pytest.raises(TypeError, match='system_rps_params'):
            epde.EpdeSearch(use_solver=False, **{old: 3})
        search = epde.EpdeSearch(use_solver=False, verbose_params={'show_iter_idx': False})
        with pytest.raises(TypeError, match='system_rps_params'):
            search.fit(data=None, **{old: 3})

    def test_a_structure_may_hold_more_than_twelve_terms_under_the_cap(self):
        space = self._space(20, **{CAP: 3})
        assert (space.equation_terms_max_number, space.system_rps_params[CAP]) == (20, 3)

    def test_the_ceiling_counts_subsets(self):
        # 20 columns, at most 6 kept: 60459 subsets.
        with pytest.raises(ValueError, match='subsets'):
            self._space(20, **{CAP: 6})

    def test_the_uncapped_rule_is_unchanged(self):
        with pytest.raises(ValueError, match='at most'):
            self._space(13)

    def test_a_numpy_integer_is_a_cap(self):
        assert self._space(**{CAP: np.int64(3)}).system_rps_params[CAP] == 3

    def test_fit_keeps_its_positional_order(self):
        """The caps are the mode's settings, not ``fit`` arguments; the old
        arguments still bind by position."""
        import inspect
        import epde
        names = list(inspect.signature(epde.EpdeSearch.fit).parameters)
        assert names[:6] == ['self', 'data', 'equation_terms_max_number',
                             'equation_factors_max_number', 'eq_sparsity_interval',
                             'additional_tokens']
        assert not {'equation_selected_terms_max_number', 'equation_screen_size'} & set(names)

    def test_the_subset_count(self):
        assert assignment_subset_count(12) == ASSIGNMENT_MAX_SUBSETS == 4095
        assert assignment_subset_count(20, 3) == 20 + 190 + 1140
        assert assignment_subset_count(5, 9) == 31

    def test_fit_shares_the_refusals(self):
        """``fit``'s own ``equation_terms_max_number`` meets the mode's cap. It
        publishes through ``with_overrides``, which does not validate, so it
        runs the same check itself."""
        import epde
        search = epde.EpdeSearch(use_solver=False, multiobjective_mode=True,
                                 system_rps_mode='assignment', system_rps_params={CAP: 6},
                                 verbose_params={'show_iter_idx': False})
        with pytest.raises(ValueError, match='subsets'):
            search.fit(data=None, equation_terms_max_number=20)
        check_rps_params('assignment', 20, resolve_rps_params('assignment', {CAP: 3}))
        with pytest.raises(ValueError, match='subsets'):
            check_rps_params('assignment', 20, resolve_rps_params('assignment', {CAP: 6}))


# ---------------------------------------------------------------------------
# The listed-subset tables.
# ---------------------------------------------------------------------------

class TestTheTables:

    @staticmethod
    def _data():
        rng = np.random.default_rng(0)
        X = rng.standard_normal((80, 4))
        y = X @ np.array([1.0, -2.0, 0.0, 0.5]) + 0.1 * rng.standard_normal(80)
        return X, y, np.ones(80)

    @pytest.mark.parametrize('discrepancy', [None, 'wape'])
    def test_subset_table_holds_the_dense_values(self, discrepancy):
        X, y, w = self._data()
        masks = [0, 3, 5, 9, 17, 30, 31, 1]
        dense_curve, dense_amp, n_cols = subset_table(X, y, w, discrepancy=discrepancy)
        curve, amp, n_listed = subset_table(X, y, w, discrepancy=discrepancy, subsets=masks)
        assert n_listed == n_cols == 5
        assert np.array_equal(curve, dense_curve[masks])
        assert np.array_equal(amp, dense_amp[masks])

    def test_instability_table_holds_the_dense_values(self):
        X, y, w = self._data()
        masks = [0, 3, 5, 17, 31]
        dense = instability_table('chi2', X, y, w, (80,), 5)
        listed = instability_table('chi2', X, y, w, (80,), 5, subsets=masks)
        assert np.array_equal(listed, dense[masks])

    def test_dyad_t_scores_in_chunks_are_the_one_batch_scores(self, monkeypatch):
        import epde.operators.common.survival as survival
        rng = np.random.default_rng(1)
        n = 400
        X = np.hstack([rng.standard_normal((n, 4)), np.ones((n, 1))])
        y = X[:, :4] @ np.array([1.0, -0.5, 0.0, 2.0]) + 0.05 * rng.standard_normal(n)
        subsets = [[j] for j in range(5)] + [[0, 1], [0, 3], [1, 4], [2, 3]] + [[0, 1, 4], [0, 3, 4]]
        whole = survival.dyad_t_subset_scores(X, y, np.ones(n), (n,), subsets)
        monkeypatch.setattr(survival, '_DYAD_SUBSET_CHUNK_DOUBLES', 1)
        chunked = survival.dyad_t_subset_scores(X, y, np.ones(n), (n,), subsets)
        assert np.all(np.isfinite(whole))
        assert np.array_equal(chunked, whole)

    def test_the_capped_masks(self):
        masks = rps._capped_masks(5, 2)
        assert masks == sorted(masks)
        assert len(masks) == assignment_subset_count(5, 2) == 15
        assert all(1 <= bin(S).count('1') <= 2 for S in masks)
        assert rps._capped_masks(4, 9) == list(range(1, 16))


# ---------------------------------------------------------------------------
# The capped search.
# ---------------------------------------------------------------------------

class TestTheCappedSearch:

    @pytest.mark.parametrize('cap', [1, 2])
    def test_no_equation_keeps_more_than_the_cap(self, uv_pool, monkeypatch, cap):
        """Checked on every variant the search puts on an equation's front, not
        only on the one the system takes: which variant that is depends on the
        instability metric, and the cap must bind whichever it is. Without the
        cap some front variant keeps more columns, so the cap is what binds."""
        _, uncapped = _run_with_fronts(uv_pool, monkeypatch)
        assert max(uncapped) > cap, 'some front variant must keep more than the cap without it'
        system, capped = _run_with_fronts(uv_pool, monkeypatch, **_cap(cap))
        assert capped and 1 <= min(capped) and max(capped) <= cap
        assert not any(_debug_assert_rps_unique(system))
        assert not has_degenerate_equation(system)
        for var in ('u', 'v'):
            assert 1 <= _kept_columns(system.vals[var]) <= cap

    @pytest.mark.parametrize('metric', ['chi2', 'dyad'])
    def test_a_cap_at_the_structure_size_is_the_uncapped_search(self, uv_pool, metric):
        uncapped = _run(_fresh(uv_pool), instability_metric=metric,
                        equation_terms_max_number=6)
        capped = _run(_fresh(uv_pool), instability_metric=metric,
                      equation_terms_max_number=6, **_cap(6))
        assert _labels_and_weights(capped) == _labels_and_weights(uncapped)

    def test_the_ceiling_is_enforced_at_run_time(self, uv_pool, monkeypatch):
        monkeypatch.setattr(rps, 'MAX_ASSIGNMENT_SUBSETS', 3)
        with pytest.raises(ValueError, match='subsets'):
            _run(_fresh(uv_pool), **_cap(2))

    def test_the_column_limit_does_not_apply_under_the_cap(self, uv_pool, monkeypatch):
        monkeypatch.setattr(rps, 'MAX_ASSIGNMENT_COLUMNS', 3)
        system = _run(_fresh(uv_pool), **_cap(2))
        assert not has_degenerate_equation(system)


# ---------------------------------------------------------------------------
# The reference width.
# ---------------------------------------------------------------------------

class TestTheReferenceWidth:

    def test_it_is_the_structure_cap_without_a_selected_cap(self):
        with using_config(equation_terms_max_number=8):
            assert _reference_width(2) == 8

    def test_it_follows_the_smaller_cap(self):
        with using_config(system_rps_mode='assignment', equation_terms_max_number=8, **_cap(3)):
            assert _reference_width(2) == 3
            assert _reference_width(5) == 5
        with using_config(system_rps_mode='assignment', equation_terms_max_number=6, **_cap(9)):
            assert _reference_width(2) == 6
