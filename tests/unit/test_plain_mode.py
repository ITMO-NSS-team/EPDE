#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""``search_space.system_rps_mode='plain'``: one fit per candidate, no subset search.

Pinned here:

* the settings and their refusals (no subset settings), and the plain cap on
  kept terms, ``equation_terms_max_number - 1``;
* the fit: every equation keeps exactly its own terms, minus another equation's
  target and the terms related to its own target by division; the target is a
  gene re-found after a reset; no two equations share a target; an equation
  without a target is the only one marked degenerate; a fitted equation is not
  refitted;
* the constant: S and S+{1} are both fitted, the kept versions are mutually
  non-dominated, and an S+{1} version leaves as a sibling system;
* parity: a support scores the same objectives and coefficients as in
  ``'assignment'`` mode;
* the labels: unchanged outside plain mode, the constant part of the structure
  and the target part of the candidate in it;
* the mutation: one move, the target untouched, the cap and the term
  uniqueness kept;
* the initial population keeps at most half its members on the first level
  (every mode).
"""
import warnings
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

import epde
import epde.globals as global_var
import epde.operators.common.plain_fit as plain_fit
import epde.operators.common.right_part_selection as rps
import epde.operators.multiobjective.moeadd_specific as moeadd_specific
from epde.interface.prepared_tokens import GridTokens
from epde.interface.search_config import (collect_overrides, load_search_config,
                                          plain_kept_terms_cap)
from epde.operators.multiobjective.moeadd_specific import (_debug_assert_rps_unique,
                                                           has_degenerate_equation)
from epde.operators.multiobjective.plain_moves import plain_mutation
from epde.optimizers.moeadd.supplementary import check_dominance, fast_non_dominated_sorting

from conftest import using_config
from test_assignment_rps import _soeq_selector
from test_duplicate_equation_guard import (DU, DV, U_TOKEN, V_TOKEN,  # noqa: F401
                                           _system, uv_pool)

UV = f'{U_TOKEN} * {V_TOKEN}'
#: u carries v's target and v carries u's: plain mode drops both.
LEAKY = ({'u': f'0.5 * {DV} + 0.3 * {V_TOKEN} + 0.2 * {U_TOKEN} + 0.1 * {UV} + 0.0 = {DU}',
          'v': f'0.4 * {DU} + 0.7 * {U_TOKEN} + 0.2 * {V_TOKEN} + 0.1 * {UV} + 0.0 = {DV}'},
         {'u': [0.5, 0.3, 0.2, 0.1, 0.0], 'v': [0.4, 0.7, 0.2, 0.1, 0.0]})
PLAIN = {'system_rps_mode': 'plain', 'equation_terms_max_number': 6}


def _fresh(pool, texts=LEAKY):
    system = _system(pool, *texts)
    for eq in system.vals:
        eq.reset_state(True)
    return system


def _run(system, seed=0, **overrides):
    np.random.seed(seed)
    with using_config(**dict(PLAIN, **overrides)):
        _soeq_selector().apply(system, {})
    return system


def _signatures(equation):
    return {term.factors_labels for term in equation.structure}


# ---------------------------------------------------------------------------
# Settings.
# ---------------------------------------------------------------------------

class TestTheSettings:

    def test_plain_has_no_settings(self):
        cfg = load_search_config(overrides={'system_rps_mode': 'plain'})
        assert cfg.search_space.system_rps_mode == 'plain'
        assert cfg.search_space.system_rps_params == {}

    @pytest.mark.parametrize('key, value', [('selected_terms_max_number', 4),
                                            ('screen_size', 5), ('variant_pick', 'random')])
    def test_the_subset_settings_belong_to_assignment(self, key, value):
        with pytest.raises(ValueError, match=f"{key} is a setting of system_rps_mode 'assignment'"):
            load_search_config(overrides={'system_rps_mode': 'plain',
                                          'system_rps_params': {key: value}})

    @pytest.mark.parametrize('overrides, match', [
        ({'use_solver': True}, 'solver'),
        ({'multiobjective_mode': False}, 'plain'),
        ({'second_objective': 'complexity'}, 'plain'),
        ({'sparsity_cls': 'lasso'}, 'refitted'),
    ])
    def test_configurations_it_cannot_serve_are_refused(self, overrides, match):
        with pytest.raises(ValueError, match=match):
            load_search_config(overrides=collect_overrides(**dict(overrides, system_rps_mode='plain')))

    def test_per_equation_niching_keys_the_planes_on_the_structure_with_the_constant(self, uv_pool):
        from epde.operators.multiobjective.moeadd_specific import equation_planes
        cfg = load_search_config(overrides=collect_overrides(system_rps_mode='plain', niching='per_equation'))
        assert cfg.evolution.niching == 'per_equation'
        system = _run(_fresh(uv_pool))
        _, labels = equation_planes([system])
        for e, key in enumerate(system.vals.equation_keys):
            assert labels[e][0] == system.vals[key].structure_label
            assert labels[e][0] != system.vals[key].terms_labels      # S and S+{1} are two structures

    @pytest.mark.parametrize('terms, cap', [(12, 11), (6, 5), (2, 1)])
    def test_the_kept_terms_cap_is_the_structure_bound(self, terms, cap):
        assert plain_kept_terms_cap(terms) == cap


# ---------------------------------------------------------------------------
# The fit.
# ---------------------------------------------------------------------------

class TestTheFit:

    def test_each_equation_keeps_its_own_terms(self, uv_pool):
        system = _run(_fresh(uv_pool))
        assert not any(_debug_assert_rps_unique(system))
        assert not has_degenerate_equation(system)
        targets = {var: system.vals[var].target.factors_labels for var in ('u', 'v')}
        for var, other in (('u', 'v'), ('v', 'u')):
            eq = system.vals[var]
            assert eq.target.contains_deriv(var)
            assert targets[other] not in _signatures(eq)
            assert eq.right_part_selected and eq.fitness_calculated and eq.stability_calculated
            # Nothing is sparsified away: every remaining term carries a coefficient.
            wi = np.asarray(eq.weights_internal)
            assert all(wi[eq.weight_index(i, eq.target_idx)] != 0
                       for i in range(len(eq.structure)) if i != eq.target_idx)
            assert eq._plain_genes['target'] == targets[var]

    def test_the_target_gene_is_refound_after_a_reset(self, uv_pool):
        system = _run(_fresh(uv_pool))
        before = {var: system.vals[var].target.factors_labels for var in ('u', 'v')}
        for eq in system.vals:
            eq.reset_state(True)
            assert eq.target is None and eq._plain_genes is not None
        _run(system, seed=7)
        assert {var: system.vals[var].target.factors_labels for var in ('u', 'v')} == before

    def test_a_gene_matches_by_factor_functions_when_the_signature_moved(self, uv_pool):
        system = _fresh(uv_pool)
        eq = system.vals['u']
        du = next(term for term in eq.structure if term.contains_deriv('u'))
        eq._plain_genes = {'target': frozenset({('nowhere',)}),
                           'names': plain_fit.target_gene(du)['names']}
        assert plain_fit.gene_term(eq) is du

    def test_two_equations_never_share_a_target(self, uv_pool):
        # du*dv holds a derivative of both variables, so both may take it.
        texts = ({'u': f'0.5 * {DU} * {DV} + 0.3 * {V_TOKEN} + 0.2 * {U_TOKEN} + 0.0 = {DU}',
                  'v': f'0.4 * {DU} * {DV} + 0.7 * {U_TOKEN} + 0.2 * {V_TOKEN} + 0.0 = {DV}'},
                 {'u': [0.5, 0.3, 0.2, 0.0], 'v': [0.4, 0.7, 0.2, 0.0]})
        system = _fresh(uv_pool, texts)
        product = next(term for term in system.vals['u'].structure
                       if term.contains_deriv('u') and term.contains_deriv('v'))
        for eq in system.vals:
            eq._plain_genes = plain_fit.target_gene(product)
        _run(system)
        u_target, v_target = system.vals['u'].target, system.vals['v'].target
        assert u_target.factors_labels == product.factors_labels   # the first equation keeps it
        assert v_target.factors_labels != product.factors_labels
        assert product.factors_labels not in _signatures(system.vals['v'])

    def test_a_term_related_to_the_target_is_dropped(self, uv_pool):
        texts = ({'u': f'0.5 * {U_TOKEN} * {DU} + 0.3 * {V_TOKEN} + 0.2 * {U_TOKEN} + 0.0 = {DU}',
                  'v': f'0.7 * {U_TOKEN} + 0.2 * {V_TOKEN} + 0.0 = {DV}'},
                 {'u': [0.5, 0.3, 0.2, 0.0], 'v': [0.7, 0.2, 0.0]})
        system = _run(_fresh(uv_pool, texts))
        eq = system.vals['u']
        assert not any(rps._related_to_target(term, eq.target)
                       for i, term in enumerate(eq.structure) if i != eq.target_idx)

    def test_only_the_equation_without_a_target_is_marked(self, uv_pool, monkeypatch):
        original = rps._ensure_own_derivative
        monkeypatch.setattr(rps, '_ensure_own_derivative',
                            lambda eq, max_iter=100: eq.main_var_to_explain != 'v' and original(eq))
        from epde.operators.common.objectives import LOSS_NAN_VAL
        system = _run(_fresh(uv_pool))
        assert system.vals['v'].fitness_value >= LOSS_NAN_VAL
        assert system.vals['u'].fitness_value < LOSS_NAN_VAL

    def test_a_fitted_equation_is_not_refitted(self, uv_pool):
        system = _run(_fresh(uv_pool))
        u_fit = system.vals['u']
        system.vals['v'].reset_state(True)
        _run(system, seed=3)
        assert system.vals['u'] is u_fit
        assert system.vals['v'].weights_final_evald


# ---------------------------------------------------------------------------
# The constant.
# ---------------------------------------------------------------------------

def _host_and_coeffs():
    selector = _soeq_selector().suboperators['eq_right_part_selector']
    return selector.suboperators['fitness_calculation'], selector.suboperators['coeff_calc']


def _prepared(system, var):
    """``system``'s equation ``var`` with plain mode's removals applied and its
    target installed, as ``apply_plain`` hands it to ``_versions``."""
    other = 'v' if var == 'u' else 'u'
    eq = system.vals[var]
    target = next(term for term in eq.structure if term.contains_deriv(var))
    banned = next(term for term in system.vals[other].structure
                  if term.contains_deriv(other)).factors_labels
    eq.structure = [term for term in eq.structure if term is target or term.factors_labels != banned]
    eq._invalidate_label_cache()
    eq.reset_state(True)
    plain_fit._install_target(eq, target)
    return eq


class TestTheConstant:

    def test_the_kept_versions_are_mutually_non_dominated(self, uv_pool):
        host, coeff_calc = _host_and_coeffs()
        with using_config(**PLAIN):
            eq = _prepared(_fresh(uv_pool), 'u')
            versions = plain_fit._versions(eq, 5, coeff_calc, {}, host, {})
        assert 1 <= len(versions) <= 2
        attrs = [filler.value_attr for filler in host.objectives]
        if len(versions) == 2:
            assert np.asarray(versions[0].weights_internal)[-1] == 0
            assert np.asarray(versions[1].weights_internal)[-1] != 0
            assert not plain_fit._dominates(versions[0], versions[1], attrs)
            assert not plain_fit._dominates(versions[1], versions[0], attrs)
            assert versions[0].structure_label != versions[1].structure_label

    def test_a_version_reads_the_parent_memo_read_only(self, uv_pool, monkeypatch):
        """A version starts from its parent's evaluate() memo -- the same
        structure and target -- as read-only views in a dict of its own, so
        the version's release (or a prune) leaves the parent's intact."""
        import epde.operators.common.subset_selection as subset_selection
        host, coeff_calc = _host_and_coeffs()
        seen, store = [], subset_selection.store_fit

        def spy(variant, *args, **kwargs):
            seen.append(variant._eval_cache)
            return store(variant, *args, **kwargs)
        monkeypatch.setattr(subset_selection, 'store_fit', spy)
        with using_config(**PLAIN):
            eq = _prepared(_fresh(uv_pool), 'u')
            versions = plain_fit._versions(eq, 5, coeff_calc, {}, host, {})
        parent = eq._eval_cache
        assert versions and seen and parent
        for memo in seen:
            assert memo is not parent and memo.keys() == parent.keys()
            for tgt, (targets, features) in memo.items():
                for part, parent_part in ((targets, parent[tgt][0]), (features, parent[tgt][1])):
                    for key, array in part.items():
                        assert not array.flags.writeable
                        assert np.shares_memory(array, parent_part[key])
                        assert parent_part[key].flags.writeable
        assert all(version._eval_cache == {} for version in versions)     # released after scoring

    def test_no_room_for_the_constant_fits_only_the_terms(self, uv_pool):
        host, coeff_calc = _host_and_coeffs()
        with using_config(**PLAIN):
            eq = _prepared(_fresh(uv_pool), 'u')
            features = len(eq.structure) - 1
            versions = plain_fit._versions(eq, features, coeff_calc, {}, host, {})
        assert all(np.asarray(v.weights_internal)[-1] == 0 for v in versions)

    def test_an_undominated_constant_leaves_as_a_sibling(self, uv_pool, monkeypatch):
        monkeypatch.setattr(plain_fit, '_dominates', lambda a, b, attrs: False)
        system = _run(_fresh(uv_pool))
        siblings = plain_fit.take_siblings(system)
        assert 1 <= len(siblings) <= 3, 'two equations, at most two versions each'
        assert plain_fit.take_siblings(system) == []
        labels = {system.equations_labels} | {sibling.equations_labels for sibling in siblings}
        assert len(labels) == len(siblings) + 1
        for sibling in siblings:
            assert sibling.structure_labels != system.structure_labels
            assert sibling.vals['u'].terms_labels == system.vals['u'].terms_labels
            assert sibling.vals['v'].terms_labels == system.vals['v'].terms_labels

    def test_another_mode_hands_over_nothing(self, uv_pool):
        system = _fresh(uv_pool)
        np.random.seed(0)
        with using_config(system_rps_mode='assignment'):
            _soeq_selector().apply(system, {})
        assert plain_fit.take_siblings(system) == []


# ---------------------------------------------------------------------------
# Parity with 'assignment'.
# ---------------------------------------------------------------------------

class TestParityWithAssignment:

    def test_the_same_support_scores_the_same(self, uv_pool, monkeypatch):
        host, coeff_calc = _host_and_coeffs()
        monkeypatch.setattr(plain_fit, '_dominates', lambda a, b, attrs: False)
        # Every admissible subset becomes a candidate, not only the front.
        monkeypatch.setattr(rps, '_admissible_front',
                            lambda eligible, a, b, admissible: [i for i in eligible if admissible(i)])
        attrs = [filler.value_attr for filler in host.objectives]
        with using_config(**PLAIN):
            eq = _prepared(_fresh(uv_pool), 'u')
            plain = plain_fit._versions(eq, 5, coeff_calc, {}, host, {})
            candidates = rps._equation_variants(eq, eq.target, frozenset(), coeff_calc, {}, host, {})
        full = frozenset(term.factors_labels for term in eq.structure)
        built = [c.materialize() for c in candidates if c.terms_labels == full]
        built = [b for b in built if b is not None]
        assert plain and built
        for version in plain:
            constant = np.asarray(version.weights_internal)[-1] != 0
            twin = next(b for b in built if (np.asarray(b.weights_internal)[-1] != 0) == constant)
            for attr in attrs:
                assert getattr(version, attr) == pytest.approx(getattr(twin, attr), rel=1e-12, abs=1e-15)
            np.testing.assert_allclose(version.weights_final, twin.weights_final, rtol=1e-12, atol=1e-15)


# ---------------------------------------------------------------------------
# Labels.
# ---------------------------------------------------------------------------

class TestTheLabels:

    def test_outside_plain_mode_nothing_changes(self, uv_pool):
        system = _fresh(uv_pool)
        for eq in system.vals:
            assert eq._plain_genes is None
            assert eq.structure_label is eq.terms_labels
            assert eq.candidate_label is eq.terms_labels
        assert system.equations_labels == tuple(eq.terms_labels for eq in system.vals)
        assert system.structure_labels == system.equations_labels

    def test_plain_labels_carry_the_constant_but_not_the_target(self, uv_pool):
        # One arrangement per structure (USER, Oct 23): the target is not part of the candidate.
        system = _run(_fresh(uv_pool))
        for eq in system.vals:
            has_constant = np.asarray(eq.weights_internal)[-1] != 0
            assert eq.structure_label == (eq.terms_labels, has_constant)
            assert eq.candidate_label == eq.structure_label
        assert system.equations_labels == system.structure_labels
        assert system.targets_labels == tuple(eq.target.factors_labels for eq in system.vals)

    def test_the_same_terms_never_dominate_whatever_the_values(self, uv_pool):
        system = _run(_fresh(uv_pool))
        twin = deepcopy(system)
        for eq in twin.vals:
            eq.fitness_value = eq.fitness_value / 10.0
        assert not check_dominance(twin, system) and not check_dominance(system, twin)


# ---------------------------------------------------------------------------
# One arrangement per structure: a target-swapped twin replaces the arrangement
# held only when its own objectives dominate.
# ---------------------------------------------------------------------------

class _Arrangement(SimpleNamespace):
    def __init__(self, targets, objectives):
        super().__init__(equations_labels=(('u-terms', False),), targets_labels=tuple(targets),
                         obj_fun=np.asarray(objectives, dtype=float), vals=[])

    def reset_moeadd_state(self):
        pass


class _Levels(SimpleNamespace):
    """MOEA/DD's ParetoLevels as the swap uses it."""

    def __init__(self, population):
        super().__init__(population=list(population), levels=[list(population)], calls=[])

    def delete_point(self, point):
        self.calls.append(('delete', point))
        self.population = [m for m in self.population if m is not point]
        self.levels = [[m for m in level if m is not point] for level in self.levels]

    def update(self, point):
        self.calls.append(('update', point))
        self.population.append(point)
        self.levels[-1].append(point)                   # ParetoLevels places it by ndl_update


def _updater():
    from epde.operators.utils.template import CompoundOperator

    class _Scored(CompoundOperator):                    # the stubs carry their objectives already
        key = 'Scored'

        def apply(self, objective, arguments):
            return objective

    updater = moeadd_specific.OffspringUpdater()
    updater.set_suboperators({'chromosome_fitness': _Scored([])})
    return updater


class TestOneArrangementPerStructure:

    @pytest.mark.parametrize('engine', ['moeadd', 'nsga2'])
    def test_a_dominating_twin_replaces_the_arrangement_held(self, engine):
        held, better = _Arrangement(['du'], [2.0, 2.0]), _Arrangement(['uv'], [1.0, 1.5])
        levels = _Levels([held])
        with using_config(**dict(PLAIN, engine=engine)):
            assert _updater()._swap_arrangement(better, levels, {'chromosome_fitness': {}})
        assert levels.population == [better] and better in levels.levels[-1]
        if engine == 'moeadd':
            assert levels.calls == [('delete', held), ('update', better)]
        else:
            assert levels.calls == []                           # swapped in place; truncation re-sorts

    @pytest.mark.parametrize('objectives', [[1.0, 2.5], [2.0, 2.0]])
    def test_a_twin_that_does_not_dominate_is_a_duplicate(self, objectives):
        held, twin = _Arrangement(['du'], [2.0, 2.0]), _Arrangement(['uv'], objectives)
        levels = _Levels([held])
        with using_config(**PLAIN):
            assert not _updater()._swap_arrangement(twin, levels, {'chromosome_fitness': {}})
        assert levels.population == [held] and levels.calls == []

    def test_the_same_arrangement_and_other_modes_are_plain_duplicates(self):
        held = _Arrangement(['du'], [2.0, 2.0])
        with using_config(**PLAIN):
            assert not _updater()._swap_arrangement(_Arrangement(['du'], [1.0, 1.0]), _Levels([held]),
                                                    {'chromosome_fitness': {}})
        with using_config(system_rps_mode='assignment'):
            assert not _updater()._swap_arrangement(_Arrangement(['uv'], [1.0, 1.0]), _Levels([held]),
                                                    {'chromosome_fitness': {}})


# ---------------------------------------------------------------------------
# The mutation.
# ---------------------------------------------------------------------------

class TestTheMoves:

    @pytest.mark.parametrize('seed', range(12))
    def test_one_move_keeps_the_target_the_cap_and_unique_terms(self, uv_pool, seed):
        system = _run(_fresh(uv_pool))
        with using_config(**dict(PLAIN, equation_terms_max_number=4)):      # 3 kept terms
            for eq in system.vals:
                target = plain_fit.gene_term(eq)
                before = len(eq.structure) - 1
                np.random.seed(seed)
                plain_mutation(eq)
                assert len(eq.terms_labels) == len(eq.structure)
                found = plain_fit.gene_term(eq)
                assert found is not None
                # The target is untouched unless the move was the target's own.
                assert found is target or found.contains_deriv(eq.main_var_to_explain)
                assert any(term is target for term in eq.structure)
                assert 1 <= len(eq.structure) - 1 <= max(3, before)

    def test_add_respects_the_cap(self, uv_pool):
        system = _run(_fresh(uv_pool))
        eq = system.vals['u']
        cap = len(eq.structure) - 1
        with using_config(**dict(PLAIN, equation_terms_max_number=cap + 1)):
            for seed in range(10):
                np.random.seed(seed)
                plain_mutation(eq)
                assert len(eq.structure) - 1 <= cap


# ---------------------------------------------------------------------------
# The crossover: offspring lengths follow the split of the unique terms.
# ---------------------------------------------------------------------------

UDV = f'{U_TOKEN} * {DV}'
#: Two parents sharing du = a v + b u, each with its own extra term.
PARENTS = ({'u': f'0.5 * {V_TOKEN} + 0.3 * {U_TOKEN} + 0.1 * {UV} + 0.0 = {DU}',
            'v': f'0.7 * {U_TOKEN} + 0.0 = {DV}'},
           {'u': f'0.5 * {V_TOKEN} + 0.3 * {U_TOKEN} + 0.1 * {UDV} + 0.0 = {DU}',
            'v': f'0.7 * {U_TOKEN} + 0.0 = {DV}'})
WEIGHTS = {'u': [0.5, 0.3, 0.1, 0.0], 'v': [0.7, 0.0]}


def _equation_crossover():
    from epde.operators.multiobjective.variation import get_basic_variation
    chromosome = get_basic_variation().suboperators['chromosome_crossover']
    return chromosome.suboperators['equation_crossover']


def _offspring_lengths(uv_pool, seeds, **overrides):
    parents = [_system(uv_pool, texts, WEIGHTS).vals['u'] for texts in PARENTS]
    for parent in parents:
        for term in parent.structure:           # the translator leaves it unset; the term exchange reads it
            term.descr_variable_marker = False
    core = {term.factors_labels for term in parents[0].structure} & \
           {term.factors_labels for term in parents[1].structure}
    lengths, cores = set(), 0
    with using_config(**overrides):
        operator = _equation_crossover()
        for seed in seeds:
            np.random.seed(seed)
            children = operator.apply(objective=(deepcopy(parents[0]), deepcopy(parents[1])), arguments={})
            for child in children:
                labels = [term.factors_labels for term in child.structure]
                assert len(set(labels)) == len(labels)
                lengths.add(len(labels))
                cores += set(labels) == core
    return lengths, cores


class TestTheCrossover:

    def test_an_offspring_can_shrink_to_the_shared_terms_or_grow(self, uv_pool):
        lengths, cores = _offspring_lengths(uv_pool, range(40), **PLAIN)
        assert lengths == {3, 4, 5} and cores > 0

    def test_the_cap_bounds_the_split(self, uv_pool):
        # At most 4 terms: 3 kept besides the target, so the 5-term offspring is never made.
        lengths, cores = _offspring_lengths(uv_pool, range(40),
                                            **dict(PLAIN, equation_terms_max_number=4))
        assert lengths == {3, 4} and cores > 0

    def test_other_modes_keep_the_parents_lengths(self, uv_pool):
        lengths, cores = _offspring_lengths(uv_pool, range(20))
        assert lengths == {4} and cores == 0


# ---------------------------------------------------------------------------
# At most half the initial population on the first level.
# ---------------------------------------------------------------------------

class _Vals(dict):
    @property
    def equation_keys(self):
        return list(self.keys())


def _stub(labels, objectives):
    return SimpleNamespace(vals=_Vals({key: SimpleNamespace(structure_label=label, terms_labels=label)
                                       for key, label in zip('uv', labels)}),
                           obj_fun=np.asarray(objectives, dtype=float))


class TestTheFirstLevelRule:

    def test_the_running_count_matches_the_sort(self):
        rng = np.random.default_rng(0)
        systems = [_stub([f'a{k}', f'b{k}'], rng.random(4)) for k in range(25)]
        count = moeadd_specific._FirstLevelCount()
        for system in systems:
            count.place(system)
        assert count.size == len(fast_non_dominated_sorting(systems)[0])

    def test_growth_only_past_the_last_slot_while_crowded(self):
        class Candidate:
            created = 0

            def create(self):
                Candidate.created += 1

        crowded = SimpleNamespace(holds=lambda: False)
        relaxed = SimpleNamespace(holds=lambda: True)
        objective = SimpleNamespace(unplaced_candidates=[Candidate(), Candidate()])
        grow = moeadd_specific.InitialParetoLevelSorting._grow_if_crowded
        extension = set()
        grow(objective, 0, objective.unplaced_candidates[0], crowded, extension, False)
        grow(objective, 1, objective.unplaced_candidates[1], relaxed, extension, False)
        grow(objective, 1, objective.unplaced_candidates[1], crowded, extension, True)
        assert len(objective.unplaced_candidates) == 2 and not extension
        grow(objective, 1, objective.unplaced_candidates[1], crowded, extension, False)
        assert len(objective.unplaced_candidates) == 3
        assert extension == {id(objective.unplaced_candidates[2])} and Candidate.created == 1

    def test_an_explicit_population_grows_until_the_rule_holds(self, monkeypatch):
        # Pretend the rule holds only from six members on: an explicit four grows. The
        # pool yields ten distinct systems whatever the right-part selection leaves of
        # them (how many a real pool yields changes with the metric and the data).
        from test_auto_settings import _controlled_pool
        _controlled_pool(monkeypatch, 10)
        monkeypatch.setattr(moeadd_specific._FirstLevelCount, 'holds',
                            lambda self: len(self.members) >= 6)
        np.random.seed(0)
        search = epde.EpdeSearch(verbose_params={'show_iter_idx': False}, population_size=4,
                                 training_epochs=0, equation_terms_max_number=3,
                                 equation_factors_max_number=2, neighbors_number=1)
        search.set_preprocessor(default_preprocessor_type='FD', preprocessor_kwargs={})
        t = np.linspace(0, 4, 80)
        domain = search.createDomain(t, ID=0)[1]
        trajectory = search.createTrajectory({'u': np.sin(2 * t) + 0.3 * t}, domain, cache_id=0)[1]
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                search.fit(data=[trajectory], max_deriv_order=(1,),
                           additional_tokens=[GridTokens(['x_0'], max_power=2)])
        finally:
            global_var.samples_manager.reset()
        size = len(search.optimizer.pareto_levels.population)
        assert size == 6
        assert search.optimizer.weights.shape[0] == size == len(search.optimizer.neighborhood_lists)
        assert any('extended from 4' in str(w.message) for w in caught)

    def test_a_dry_pool_does_not_grow(self, monkeypatch):
        """Growth draws only new systems: with the pool's two systems placed,
        an explicit four fills its slots with duplicates and stops there,
        saying the rule could not be met."""
        from test_auto_settings import _controlled_pool
        _controlled_pool(monkeypatch, 2)
        monkeypatch.setattr(moeadd_specific._FirstLevelCount, 'holds',
                            lambda self: len(self.members) >= 6)
        np.random.seed(0)
        search = epde.EpdeSearch(verbose_params={'show_iter_idx': False}, population_size=4,
                                 training_epochs=0, equation_terms_max_number=3,
                                 equation_factors_max_number=2, neighbors_number=1)
        search.set_preprocessor(default_preprocessor_type='FD', preprocessor_kwargs={})
        t = np.linspace(0, 4, 80)
        domain = search.createDomain(t, ID=0)[1]
        trajectory = search.createTrajectory({'u': np.sin(2 * t) + 0.3 * t}, domain, cache_id=0)[1]
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                search.fit(data=[trajectory], max_deriv_order=(1,),
                           additional_tokens=[GridTokens(['x_0'], max_power=2)])
        finally:
            global_var.samples_manager.reset()
        population = search.optimizer.pareto_levels.population
        assert len(population) == 4
        assert len({sol.equations_labels for sol in population}) == 2
        assert not any('extended from' in str(w.message) for w in caught)
        assert any('the pool yields no further system' in str(w.message) for w in caught)


# ---------------------------------------------------------------------------
# The candidate rules hold for the support that is kept.
# ---------------------------------------------------------------------------

@pytest.fixture
def cubic_on_sinusoid():
    """u^2 u'' = a u^3 + b u'' u' on a pure sinusoid, its target installed.

    The finite-difference u'' is exactly proportional to u there (the boundary
    of 4 drops the one-sided rows of the fourth-order stencil), so u'' * u'
    gets a coefficient at round-off, 1e-15, in the fit without the constant,
    and again beside a constant of 2e-15 in the fit with it: both are
    u^2 u'' = a u^3, the law multiplied through by u^2."""
    from epde.interface.equation_translator import translate_equation
    search = epde.EpdeSearch(verbose_params={'show_iter_idx': False})
    search.set_preprocessor(default_preprocessor_type='FD', preprocessor_kwargs={})
    t = np.linspace(0, 4 * np.pi, 200)
    domain = search.createDomain(t, ID=0, boundary_width=4)[1]
    trajectory = search.createTrajectory({'u': np.sin(t) + 1.3 * np.cos(t)}, domain, cache_id=0)[1]
    try:
        search.create_pool(data=[trajectory], max_deriv_order=(2,), data_fun_pow=3)
        system = translate_equation(
            '1.0 * u{power: 3.0} + 1.0 * d^2u/dx0^2{power: 1.0} * du/dx0{power: 1.0} + 0.0 = '
            'u{power: 2.0} * d^2u/dx0^2{power: 1.0}', search.pool, all_vars=['u'])
        eq = system.vals['u']
        target = eq.target
        eq.reset_state(True)
        plain_fit._install_target(eq, target)
        yield eq
    finally:
        global_var.samples_manager.reset()


def _recording_refusals(monkeypatch):
    """Record the structure of every variant ``_pruned_to_reducible`` refuses
    (plain mode imports it from the module at call time)."""
    refused, real = [], rps._pruned_to_reducible

    def recording(variant):
        verdict = real(variant)
        if verdict:
            refused.append(frozenset(term.factors_labels for term in variant.structure))
        return verdict
    monkeypatch.setattr(rps, '_pruned_to_reducible', recording)
    return refused


def _cubic_supports(eq):
    """{target, u^3}, what an exact zero would leave, and {target, u^3, u'' u'}."""
    others = [t.factors_labels for t in eq.structure if t is not eq.target]
    cubic = next(labels for labels in others if len(labels) == 1)
    return {eq.target.factors_labels, cubic}, {eq.target.factors_labels, *others}


class TestPrunedSupports:

    def test_round_off_zeros_cannot_leave_a_reducible_support(self, cubic_on_sinusoid,
                                                              monkeypatch):
        # Both versions, S and S+{1}, keep u'' u' at round-off: both are refused.
        refused = _recording_refusals(monkeypatch)
        host, coeff_calc = _host_and_coeffs()
        with using_config(**PLAIN):
            versions = plain_fit._versions(cubic_on_sinusoid, 5, coeff_calc, {}, host, {})
        assert versions == []
        _, round_off = _cubic_supports(cubic_on_sinusoid)
        assert refused == [round_off, round_off]

    def test_nor_in_assignment_mode(self, cubic_on_sinusoid, monkeypatch):
        eq = cubic_on_sinusoid
        refused = _recording_refusals(monkeypatch)
        host, coeff_calc = _host_and_coeffs()
        # The round-off support is dominated here; every admissible subset
        # is built instead, as one on a front would be.
        monkeypatch.setattr(rps, '_admissible_front',
                            lambda eligible, a, b, admissible: [i for i in eligible if admissible(i)])
        with using_config(system_rps_mode='assignment', equation_terms_max_number=6):
            candidates = rps._equation_variants(eq, eq.target, frozenset(), coeff_calc, {}, host, {})
            built = [c.materialize() for c in candidates]
        _, round_off = _cubic_supports(eq)
        assert round_off in refused
        # The other admissible support, u^2 u'' = c, the host declines.
        assert all(variant is None for variant in built)

    @pytest.mark.parametrize('coefficient, reducible', [(1e-6, False), (1e-14, True), (0.0, True)])
    def test_a_term_counts_above_round_off(self, cubic_on_sinusoid, coefficient, reducible):
        # u'' u' and the target are of one magnitude: its coefficient is its share.
        variant = deepcopy(cubic_on_sinusoid)
        features = [t for i, t in enumerate(variant.structure) if i != variant.target_idx]
        mixed = next(j for j, t in enumerate(features) if len(t.structure) == 2)
        weights = np.zeros(len(features) + 1)
        weights[1 - mixed], weights[mixed] = -1.0, coefficient
        variant.weights_internal = weights
        variant.weights_internal_evald = True
        assert rps._pruned_to_reducible(variant) == reducible
