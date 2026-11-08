#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""``search_space.system_rps_mode='assignment'``: a system's targets and
supports are chosen together.

Every assignment of one own-derivative target per equation is enumerated, the
terms equal to another equation's chosen target are removed, every subset of
the remaining terms is fitted, and the system takes one non-dominated variant
that is not already in the search history. Pinned here:

* the settings and their refusals;
* the cross-equation rules hold by construction (no equation carries another's
  target, no equation is left unfitted or unscored);
* variants already in the history are skipped while fresh ones remain;
* the dominance between system variants follows ``check_dominance``'s rule that
  an equation with the same structure in both systems is not compared;
* a system with no admissible target is left degenerate for the population
  operators to re-roll;
* the default ``'repair'`` mode does not read the history.
"""

from types import SimpleNamespace

import numpy as np
import pytest

import epde.operators.common.right_part_selection as rps
from epde.interface.search_config import load_search_config
from epde.operators.multiobjective.moeadd_specific import (_debug_assert_rps_unique,
                                                           has_degenerate_equation)

from conftest import using_config
from test_duplicate_equation_guard import (DU, DV, U_TOKEN, V_TOKEN,  # noqa: F401
                                           _system, uv_pool)

UV = f'{U_TOKEN} * {V_TOKEN}'


def _soeq_selector():
    """The chromosome-level selector over a real per-equation selector, with
    the live solver-free host (discrepancy + instability)."""
    from epde.interface.search_config import active_config
    from epde.operators.common.coeff_calculation import LinRegBasedCoeffsEquation
    from epde.operators.common.fitness import SolverFreeFitness
    from epde.operators.common.objectives import Discrepancy, Instability
    from epde.operators.common.sparsity import build_sparsity_operator
    from epde.operators.utils.template import add_base_param_to_operator
    disc = Discrepancy()
    host = SolverFreeFitness(['penalty_coeff'], objectives=[disc, Instability()], primary=disc)
    add_base_param_to_operator(host, {})
    cfg = active_config().objectives
    selector = rps.EqRightPartSelector()
    selector.set_suboperators({'fitness_calculation': host,
                               'sparsity': build_sparsity_operator(cfg.sparsity_cls,
                                                                   cfg.sparsity_kwargs),
                               'coeff_calc': LinRegBasedCoeffsEquation()})
    system_selector = rps.SoEqRightPartSelector()
    system_selector.set_suboperators({'eq_right_part_selector': selector})
    return system_selector


#: u carries v's target dv/dx0 and v carries u's target du/dx0: both leaks the
#: removal must clear. Extra terms give each equation several subsets.
LEAKY = ({'u': f'0.5 * {DV} + 0.3 * {V_TOKEN} + 0.2 * {U_TOKEN} + 0.1 * {UV} + 0.0 = {DU}',
          'v': f'0.4 * {DU} + 0.7 * {U_TOKEN} + 0.2 * {V_TOKEN} + 0.1 * {UV} + 0.0 = {DV}'},
         {'u': [0.5, 0.3, 0.2, 0.1, 0.0], 'v': [0.4, 0.7, 0.2, 0.1, 0.0]})


def _fresh(pool):
    system = _system(pool, *LEAKY)
    for eq in system.vals:
        eq.right_part_selected = False
    return system


def _run(system, history=None, seed=0):
    np.random.seed(seed)
    with using_config(system_rps_mode='assignment'):
        if history is None:
            _soeq_selector().apply(system, {})
        else:
            with rps.system_history(history):
                _soeq_selector().apply(system, {})
    return system


# ---------------------------------------------------------------------------
# Settings.
# ---------------------------------------------------------------------------

class TestTheSettings:

    def test_defaults(self):
        cfg = load_search_config().search_space
        assert cfg.system_rps_mode == 'repair' and cfg.system_rps_params == {}
        params = load_search_config(overrides={'system_rps_mode': 'assignment'}
                                    ).search_space.system_rps_params
        assert params == {'selected_terms_max_number': None, 'screen_size': None,
                          'variant_pick': 'random'}

    def test_assignment_is_accepted(self):
        cfg = load_search_config(overrides={'system_rps_mode': 'assignment'})
        assert cfg.search_space.system_rps_mode == 'assignment'

    @pytest.mark.parametrize('overrides, match', [
        ({'system_rps_mode': 'sequential'}, 'system_rps_mode'),
        ({'system_rps_mode': 'assignment', 'system_rps_params': {'variant_pick': 'closest'}},
         'variant_pick'),
        ({'system_rps_mode': 'assignment', 'system_rps_params': {'variant': 'sector'}},
         "it accepts \\['screen_size', 'selected_terms_max_number', 'variant_pick'\\]")])
    def test_unknown_values_are_rejected(self, overrides, match):
        with pytest.raises(ValueError, match=match):
            load_search_config(overrides=overrides)

    def test_the_solver_is_refused(self):
        with pytest.raises(ValueError, match='solver'):
            load_search_config(overrides={'system_rps_mode': 'assignment',
                                          'use_solver': True})

    @pytest.mark.parametrize('overrides', [{'multiobjective_mode': False},
                                           {'second_objective': 'complexity'},
                                           {'equation_terms_max_number': 13}])
    def test_configurations_it_cannot_serve_are_refused(self, overrides):
        with pytest.raises(ValueError, match='assignment'):
            load_search_config(overrides=dict(overrides, system_rps_mode='assignment'))


# ---------------------------------------------------------------------------
# The selection.
# ---------------------------------------------------------------------------

class TestTheAssignment:

    def test_the_rules_hold_by_construction(self, uv_pool):
        system = _run(_fresh(uv_pool))
        assert not any(_debug_assert_rps_unique(system))
        for var in ('u', 'v'):
            eq = system.vals[var]
            assert eq.target is not None and eq.target.contains_deriv(var)
            assert eq.right_part_selected
            assert eq.fitness_calculated and eq.stability_calculated
            assert eq.weights_internal_evald and eq.weights_final_evald
            # Pruned: every non-target term left has a non-zero coefficient.
            tgt = eq.target_idx
            wi = np.asarray(eq.weights_internal)
            assert all(wi[eq.weight_index(i, tgt)] != 0
                       for i in range(len(eq.structure)) if i != tgt)
        assert not has_degenerate_equation(system)

    def test_the_installed_equations_share_the_systems_metaparameters(self, uv_pool):
        system = _fresh(uv_pool)
        before = {var: system.vals[var].metaparameters for var in ('u', 'v')}
        _run(system)
        assert all(system.vals[var].metaparameters is before[var] for var in ('u', 'v'))

    def test_variants_already_in_history_are_skipped(self, uv_pool, monkeypatch):
        pools = []
        original = rps._nondominated_systems

        def spy(variants, value_attrs):
            pools.append([tuple(eq.terms_labels for eq in v) for v in variants])
            return original(variants, value_attrs)
        monkeypatch.setattr(rps, '_nondominated_systems', spy)

        first = _run(_fresh(uv_pool), history=set()).equations_labels
        assert len(pools[-1]) > 1, 'the fixture must offer several variants'
        _run(_fresh(uv_pool), history={first})
        assert first not in pools[-1]

    def test_when_everything_is_known_all_variants_compete(self, uv_pool, monkeypatch):
        pools = []
        original = rps._nondominated_systems

        def spy(variants, value_attrs):
            pools.append(len(variants))
            return original(variants, value_attrs)
        monkeypatch.setattr(rps, '_nondominated_systems', spy)

        class Everything(set):
            def __contains__(self, item):
                return True
        _run(_fresh(uv_pool), history=Everything())
        _run(_fresh(uv_pool), history=set())
        assert pools[0] == pools[1]

    def test_no_admissible_target_leaves_the_system_degenerate(self, uv_pool, monkeypatch):
        monkeypatch.setattr(rps, '_ensure_own_derivative', lambda eq, max_iter=100: False)
        system = _run(_fresh(uv_pool))
        assert has_degenerate_equation(system)
        assert all(eq.right_part_selected for eq in system.vals)

    def test_a_multiple_of_the_target_is_never_a_feature(self, uv_pool):
        # u*du/dx0 is a MULTIPLE of u's target du/dx0: the ratio-fit soak shape.
        texts = {'u': f'0.5 * {U_TOKEN} * {DU} + 0.3 * {V_TOKEN} + 0.2 * {U_TOKEN} + 0.0 = {DU}',
                 'v': f'0.7 * {U_TOKEN} + 0.2 * {V_TOKEN} + 0.0 = {DV}'}
        system = _system(uv_pool, texts, {'u': [0.5, 0.3, 0.2, 0.0], 'v': [0.7, 0.2, 0.0]})
        _run(system)
        u_eq = system.vals['u']
        target = u_eq.target
        assert not any(rps._related_to_target(term, target)
                       for i, term in enumerate(u_eq.structure) if i != u_eq.target_idx)

    def test_only_the_equations_without_variants_are_marked(self, uv_pool, monkeypatch):
        original = rps._equation_variants

        def no_v(equation, *args, **kwargs):
            return [] if equation.main_var_to_explain == 'v' else original(equation, *args, **kwargs)
        monkeypatch.setattr(rps, '_equation_variants', no_v)
        from epde.operators.common.objectives import LOSS_NAN_VAL
        system = _run(_fresh(uv_pool))
        assert system.vals['v'].fitness_value >= LOSS_NAN_VAL
        assert system.vals['u'].fitness_value < LOSS_NAN_VAL
        assert system.vals['u'].weights_final_evald

    def test_too_many_columns_fail_loud(self, uv_pool, monkeypatch):
        monkeypatch.setattr(rps, 'MAX_ASSIGNMENT_COLUMNS', 3)
        with pytest.raises(ValueError, match='column'):
            _run(_fresh(uv_pool))

    def test_a_sampled_variant_set_still_obeys_the_rules(self, uv_pool, monkeypatch):
        monkeypatch.setattr(rps, 'MAX_SYSTEM_VARIANTS', 2)
        system = _run(_fresh(uv_pool))
        assert not any(_debug_assert_rps_unique(system))
        assert not has_degenerate_equation(system)

    def test_the_repair_mode_does_not_read_the_history(self, uv_pool):
        def labels_and_weights(history):
            system = _fresh(uv_pool)
            np.random.seed(3)
            if history is None:
                _soeq_selector().apply(system, {})
            else:
                with rps.system_history(history):
                    _soeq_selector().apply(system, {})
            return (system.equations_labels,
                    [np.asarray(eq.weights_final).tolist() for eq in system.vals])
        assert labels_and_weights(None) == labels_and_weights({('anything',)})


# ---------------------------------------------------------------------------
# Dominance between system variants.
# ---------------------------------------------------------------------------

def _eq(label, disc, inst):
    return SimpleNamespace(terms_labels=frozenset({label}), fitness_value=disc,
                           coefficients_stability=inst)


class TestTheDominance:

    ATTRS = ['fitness_value', 'coefficients_stability']

    def test_plain_dominance_between_different_structures(self):
        good = (_eq('a', 1.0, 1.0), _eq('x', 1.0, 1.0))
        bad = (_eq('b', 2.0, 2.0), _eq('y', 2.0, 2.0))
        trade = (_eq('c', 0.5, 3.0), _eq('z', 1.0, 1.0))
        assert rps._nondominated_systems([good, bad, trade], self.ATTRS) == [0, 2]

    def test_the_same_structure_is_not_compared(self):
        # Same u-structure, noisy u values: only v decides, and v ties.
        first = (_eq('a', 1.0, 1.0), _eq('x', 1.0, 1.0))
        noisy = (_eq('a', 1.1, 1.1), _eq('x', 1.0, 1.0))
        assert rps._nondominated_systems([first, noisy], self.ATTRS) == [0, 1]

    def test_the_same_structure_does_not_shield_a_worse_partner(self):
        first = (_eq('a', 1.0, 1.0), _eq('x', 1.0, 1.0))
        worse_v = (_eq('a', 1.0, 1.0), _eq('y', 2.0, 2.0))
        assert rps._nondominated_systems([first, worse_v], self.ATTRS) == [0]

    def test_the_two_dimensional_front(self):
        mask = rps._pareto_mask_2d([1.0, 2.0, 0.5, 2.0], [1.0, 2.0, 3.0, 0.5])
        assert mask.tolist() == [True, False, True, True]

    @staticmethod
    def _pairwise(a, b):
        a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
        return np.array([not np.any((a <= a[i]) & (b <= b[i]) & ((a < a[i]) | (b < b[i])))
                         for i in range(a.size)], dtype=bool)

    @pytest.mark.parametrize('seed', range(40))
    def test_the_sorted_front_is_the_pairwise_front(self, seed):
        """Ties in either coordinate, duplicates, -0.0 against 0.0, and sizes
        0 and 1: the sort-based mask is the pairwise definition's."""
        rng = np.random.default_rng(seed)
        n = int(rng.integers(0, 60))
        levels = rng.choice([2, 5, 1000])
        a = rng.integers(0, levels, n).astype(float)
        b = rng.integers(0, levels, n).astype(float)
        if n and seed % 4 == 0:
            a[rng.integers(n)] = -0.0
        assert rps._pareto_mask_2d(a, b).tolist() == self._pairwise(a, b).tolist()

    def test_a_non_finite_value_takes_the_pairwise_loop(self):
        a = [1.0, np.nan, 0.5, np.inf, 2.0]
        b = [1.0, 0.1, np.nan, 0.2, -np.inf]
        assert rps._pareto_mask_2d(a, b).tolist() == self._pairwise(a, b).tolist()

    @pytest.mark.parametrize('seed', range(40))
    def test_the_admissible_front_checks_only_front_members(self, seed):
        """The lazily checked front equals the front of the admissible
        positions computed after checking every one, and asks less."""
        rng = np.random.default_rng(seed)
        n = int(rng.integers(1, 80))
        a = rng.integers(0, 6, n).astype(float)
        b = rng.integers(0, 6, n).astype(float)
        bad = set(rng.choice(n, size=int(rng.integers(0, n + 1)), replace=False).tolist())
        eligible = sorted(rng.choice(n, size=int(rng.integers(0, n + 1)), replace=False).tolist())
        asked = []

        def admissible(i):
            asked.append(i)
            return i not in bad
        lazy = rps._admissible_front(eligible, a, b, admissible)
        kept = [i for i in eligible if i not in bad]
        eager = [i for i, k in zip(kept, self._pairwise(a[kept], b[kept])) if k] if kept else []
        assert lazy == eager
        assert len(asked) == len(set(asked)) <= len(eligible)


class TestTheSimplifyGuards:

    def test_exactly_dependent_supports_are_not_candidates(self):
        rng = np.random.default_rng(0)
        a, b = rng.normal(size=50), rng.normal(size=50)
        cols = np.stack([a, b, a + 2 * b], axis=1)
        unit = cols / np.linalg.norm(cols, axis=0)
        constant = np.full(50, 1 / np.sqrt(50))
        assert rps._subset_rank_deficient(unit, constant, [0, 1, 2], False)
        assert not rps._subset_rank_deficient(unit, constant, [0, 1], True)
        shifted = np.stack([a, a + 1.0], axis=1)
        unit_s = shifted / np.linalg.norm(shifted, axis=0)
        assert rps._subset_rank_deficient(unit_s, constant, [0, 1], True)   # a+1 = a + const

    def test_a_shared_factor_makes_a_support_reducible(self, uv_pool):
        texts = {'u': f'0.5 * {U_TOKEN} * {V_TOKEN} + 0.3 * {V_TOKEN} + 0.0 = {U_TOKEN} * {DU}',
                 'v': f'0.7 * {U_TOKEN} + 0.0 = {DV}'}
        system = _system(uv_pool, texts, {'u': [0.5, 0.3, 0.0], 'v': [0.7, 0.0]})
        eq = system.vals['u']
        features = [t for i, t in enumerate(eq.structure) if i != eq.target_idx]
        # u*v alone shares u with the target u*du: reducible, with the constant
        # too (u*du = a*u*v + c is du = a*v multiplied through, c round-off); v
        # alone does not.
        assert rps._subset_reducible(eq.target, features, [0], False)
        assert rps._subset_reducible(eq.target, features, [0], True)
        assert not rps._subset_reducible(eq.target, features, [1], False)
        assert not rps._subset_reducible(eq.target, features, [1], True)

    def test_a_constant_blocks_the_root(self, uv_pool):
        # du^2 = a*u^2 is du = sqrt(a)*u squared; du^2 = a*u^2 + c (the energy
        # identity's shape) is not a square of anything.
        texts = {'u': '0.5 * u{power: 2.0} + 0.0 = du/dx0{power: 2.0}',
                 'v': f'0.7 * {U_TOKEN} + 0.0 = {DV}'}
        system = _system(uv_pool, texts, {'u': [0.5, 0.0], 'v': [0.7, 0.0]})
        eq = system.vals['u']
        features = [t for i, t in enumerate(eq.structure) if i != eq.target_idx]
        assert rps._subset_reducible(eq.target, features, [0], False)
        assert not rps._subset_reducible(eq.target, features, [0], True)


class TestTheSectorPick:

    ATTRS = ['fitness_value', 'coefficients_stability']

    @pytest.fixture(autouse=True)
    def _clear_sector(self):
        rps._MATING_SECTOR['value'] = None
        yield
        rps._MATING_SECTOR['value'] = None

    def test_no_sector_yet_means_no_pick(self):
        variants = [(_eq('a', 0.1, 0.9),), (_eq('b', 0.9, 0.1),)]
        assert rps._sector_pick(variants, [0, 1], self.ATTRS) is None

    def test_clearing_and_a_foreign_normalizer_mean_no_pick(self):
        from epde.optimizers.moeadd.moeadd import ObjFunNormalizer
        variants = [(_eq('a', 0.1, 0.9),), (_eq('b', 0.9, 0.1),)]
        rps.set_mating_sector([1.0, 0.0], ObjFunNormalizer(np.ones(6)))   # 3-equation layout
        assert rps._sector_pick(variants, [0, 1], self.ATTRS) is None
        rps.set_mating_sector(None)
        assert rps._MATING_SECTOR['value'] is None

    def test_the_variant_along_the_direction_wins(self):
        variants = [(_eq('a', 0.1, 0.9), _eq('x', 0.1, 0.9)),
                    (_eq('b', 0.9, 0.1), _eq('y', 0.9, 0.1))]
        rps.set_mating_sector([1.0, 0.0], None)            # along the discrepancy axis
        assert rps._sector_pick(variants, [0, 1], self.ATTRS) == 1
        rps.set_mating_sector([0.0, 1.0], None)            # along the instability axis
        assert rps._sector_pick(variants, [0, 1], self.ATTRS) == 0

    def test_the_served_equation_decides_under_per_equation_niching(self):
        # u favours variant 0 on the instability axis, v favours variant 1.
        variants = [(_eq('a', 0.1, 0.9), _eq('x', 0.9, 0.1)),
                    (_eq('b', 0.9, 0.1), _eq('y', 0.1, 0.9))]
        rps.set_mating_sector([0.0, 1.0], None, focus=0)
        assert rps._sector_pick(variants, [0, 1], self.ATTRS) == 0
        rps.set_mating_sector([0.0, 1.0], None, focus=1)
        assert rps._sector_pick(variants, [0, 1], self.ATTRS) == 1
