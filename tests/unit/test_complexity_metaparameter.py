#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""The ``('complexity', var)`` budget: a metaparameter of its own.

``KneeSparsity``'s opt-in ``'complexity'`` realization returns the best subset
at the largest complexity an equation's integer budget allows. That budget is
NOT the ``('sparsity', var)`` alpha: it is a separate per-equation key,

* present only when the configured sparsity operator selects by a budget, so a
  default run keeps exactly its metaparameters, genes and random stream;
* seeded uniformly over every level ``1 .. (T - 1) * F + 1`` with its bounds
  stored beside the value;
* stepped by ``c + {-1, 0, +1}`` reflected into those bounds -- never by
  ``MetaparameterMutation``, whose proportional walk drifts a budget down to
  an absorbing 1 -- and an equation whose budget moved is re-opened for
  right-part selection, as a structural change re-opens it.

The ``('sparsity', var)`` side -- its seeding, its warning, its mutation -- is
pinned unchanged here as well as in ``test_sparsity_interval.py``.
"""

import copy
import warnings
from types import SimpleNamespace

import numpy as np
import pytest

import epde
import epde.globals as global_var
import epde.optimizers.moeadd.population_constr as mo_constr
import epde.optimizers.single_criterion.population_constr as so_constr
import epde.operators.multiobjective.variation as variation
from epde.operators.multiobjective.mutations import (MetaparameterMutation,
                                                     SystemMutation,
                                                     step_complexity_budget)
from epde.operators.multiobjective.variation import ChromosomeCrossover
from epde.operators.utils.template import CompoundOperator
from epde.structure.encoding import Chromosome
from epde.structure.main_structures import (COMPLEXITY_METAPARAMETER,
                                            check_metaparameters,
                                            is_complexity_key)

U_BUDGET = (COMPLEXITY_METAPARAMETER, 'u')


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------
class _RecordingSoEq:
    """Stands in for ``SoEq`` inside the constructors: validates and keeps the
    metaparameters it is built with, and skips the term generation that needs
    a real token pool."""

    def __init__(self, pool, metaparameters):
        check_metaparameters(metaparameters)
        self.metaparameters = metaparameters

    def set_objective_functions(self, obj_funs):
        pass

    def use_default_multiobjective_function(self, second_objective=None):
        pass

    def use_default_singleobjective_function(self):
        pass

    def create(self):
        pass


def _pool(*variables):
    return SimpleNamespace(families_demand_equation=[
        SimpleNamespace(variable=variable) for variable in variables])


@pytest.fixture
def recording(monkeypatch):
    monkeypatch.setattr(mo_constr, 'SoEq', _RecordingSoEq)
    monkeypatch.setattr(so_constr, 'SoEq', _RecordingSoEq)


def _same_state(first, second):
    return (first[0] == second[0] and np.array_equal(first[1], second[1])
            and first[2:] == second[2:])


class TestSeeding:

    def test_a_default_population_draws_exactly_what_it_always_did(
            self, recording):
        """No interval, no key, no extra random number."""
        constructor = mo_constr.SystemsPopulationConstructor(
            _pool('u', 'v'), sparsity_interval=(1e-4, 2.5))
        np.random.seed(3)
        solution = constructor.create()
        after = np.random.get_state()

        np.random.seed(3)
        alpha = 10 ** np.random.uniform(np.log10(1e-4), np.log10(2.5), size=2)
        assert _same_state(after, np.random.get_state())
        assert set(solution.metaparameters) == {
            'max_terms_number', 'max_factors_in_term',
            ('sparsity', 'u'), ('sparsity', 'v')}
        assert [solution.metaparameters[('sparsity', variable)]['value']
                for variable in constructor.vars_demand_equation] == list(alpha)

    def test_the_budget_is_drawn_after_the_alpha(self, recording):
        """The alpha draw is the same one with or without budgets."""
        constructor = mo_constr.SystemsPopulationConstructor(
            _pool('u', 'v'), sparsity_interval=(1e-4, 2.5),
            complexity_interval=(1, 6))
        np.random.seed(3)
        solution = constructor.create()

        np.random.seed(3)
        alpha = 10 ** np.random.uniform(np.log10(1e-4), np.log10(2.5), size=2)
        budgets = np.random.randint(1, 7, size=2)
        for idx, variable in enumerate(constructor.vars_demand_equation):
            assert solution.metaparameters[('sparsity', variable)] == {
                'optimizable': True, 'value': alpha[idx]}
            entry = solution.metaparameters[(COMPLEXITY_METAPARAMETER, variable)]
            assert entry == {'optimizable': True, 'value': int(budgets[idx]),
                             'bounds': (1, 6)}
            assert type(entry['value']) is int      # the gene's type

    def test_every_level_is_equally_likely(self, recording):
        constructor = mo_constr.SystemsPopulationConstructor(
            _pool('u'), complexity_interval=(1, 6))
        np.random.seed(11)
        draws = np.array([constructor.create().metaparameters[U_BUDGET]['value']
                          for _ in range(6000)])
        assert draws.min() == 1 and draws.max() == 6
        frequencies = np.bincount(draws, minlength=7)[1:] / draws.size
        assert np.allclose(frequencies, 1 / 6, atol=0.02)

    def test_each_equation_draws_its_own_budget(self, recording):
        constructor = mo_constr.SystemsPopulationConstructor(
            _pool('u', 'v'), complexity_interval=(1, 6))
        np.random.seed(12)
        pairs = np.array([[solution.metaparameters[(COMPLEXITY_METAPARAMETER,
                                                    variable)]['value']
                           for variable in ('u', 'v')]
                          for solution in (constructor.create()
                                           for _ in range(3600))])
        assert len({tuple(pair) for pair in pairs}) == 36
        assert abs(np.corrcoef(pairs.T)[0, 1]) < 0.05

    def test_single_objective_seeds_it_without_a_gene(self, recording):
        """No metaparameter genes in single-objective mode, so not
        optimizable -- each equation keeps its seeded budget."""
        constructor = so_constr.SystemsPopulationConstructor(
            _pool('u'), complexity_interval=(1, 5))
        np.random.seed(4)
        values = set()
        for _ in range(200):
            entry = constructor.create().metaparameters[U_BUDGET]
            assert entry['optimizable'] is False
            assert entry['bounds'] == (1, 5)
            values.add(entry['value'])
        assert values == {1, 2, 3, 4, 5}
        plain = so_constr.SystemsPopulationConstructor(_pool('u'))
        assert U_BUDGET not in plain.create().metaparameters

    def test_check_metaparameters_accepts_the_key_with_its_bounds(self):
        good = {('sparsity', 'u'): {'optimizable': True, 'value': 1.0},
                U_BUDGET: {'optimizable': True, 'value': 3, 'bounds': (1, 6)}}
        assert check_metaparameters(good) is True
        for bounds in (None, (0, 3), (4, 3), (1,)):
            entry = {'optimizable': True, 'value': 3}
            if bounds is not None:
                entry['bounds'] = bounds
            with pytest.raises(ValueError, match='needs integer bounds'):
                check_metaparameters({U_BUDGET: entry})

    def test_the_key_predicate(self):
        assert is_complexity_key(('complexity', 'u'))
        assert not is_complexity_key(('sparsity', 'u'))
        assert not is_complexity_key('complexity')
        assert not is_complexity_key('max_terms_number')


class _PassedSystem:
    """A translated-truth-like system: alpha keys, no budget, and one
    metaparameters dict per equation (as the multi-equation translator
    builds them)."""

    def __init__(self, variables):
        self.vars_to_describe = list(variables)
        self.metaparameters = {('sparsity', variable): {'optimizable': True,
                                                        'value': 0.}
                               for variable in variables}
        self.vals = {variable: SimpleNamespace(metaparameters={
            ('sparsity', variable): {'optimizable': True, 'value': 0.}})
            for variable in variables}
        self.created_with = None

    def set_objective_functions(self, obj_funs):
        pass

    def use_default_multiobjective_function(self, second_objective=None):
        pass

    def create(self, passed_equations=None):
        self.created_with = passed_equations


class TestPassedSystems:
    """``ChromosomeCrossover`` asserts one encoding for both parents, so a
    passed system must carry the budget gene the seeded ones carry."""

    def test_a_passed_system_joins_at_the_ceiling(self):
        constructor = mo_constr.SystemsPopulationConstructor(
            _pool('u', 'v'), complexity_interval=(1, 6))
        system = _PassedSystem(['u', 'v'])
        constructor.applyToPassed(system)
        for variable in ('u', 'v'):
            key = (COMPLEXITY_METAPARAMETER, variable)
            expected = {'optimizable': True, 'value': 6, 'bounds': (1, 6)}
            assert system.metaparameters[key] == expected
            assert system.vals[variable].metaparameters[key] == expected
        # ...and the chromosome is rebuilt, so the key becomes a gene.
        assert system.created_with == [system.vals['u'], system.vals['v']]

    def test_nothing_changes_without_a_budget(self):
        constructor = mo_constr.SystemsPopulationConstructor(_pool('u'))
        system = _PassedSystem(['u'])
        constructor.applyToPassed(system)
        assert U_BUDGET not in system.metaparameters
        assert system.created_with is None

    def test_a_system_that_has_the_key_is_left_alone(self):
        """Already within this search's bounds: no rebuild, and the equation
        -- which had its own dict without the key -- is handed the entry its
        gene now writes into."""
        constructor = mo_constr.SystemsPopulationConstructor(
            _pool('u'), complexity_interval=(1, 6))
        system = _PassedSystem(['u'])
        system.metaparameters[U_BUDGET] = {'optimizable': True, 'value': 2,
                                           'bounds': (1, 6)}
        constructor.applyToPassed(system)
        assert system.metaparameters[U_BUDGET]['value'] == 2
        assert system.created_with is None
        assert system.vals['u'].metaparameters[U_BUDGET] == \
            system.metaparameters[U_BUDGET]

    @pytest.mark.parametrize('value, expected', [
        (9, 4), (2, 2), (2.6, 3), (0, 4), (np.nan, 4)])
    def test_an_earlier_runs_budget_is_rebounded_and_clipped(self, value,
                                                             expected):
        """A system from a budget run with a wider range takes THIS search's
        bounds -- the step reflects into the stored ones -- and its value is
        read as the level it stands for, clipped into them ("no budget" reads
        as the ceiling, as everywhere else)."""
        constructor = mo_constr.SystemsPopulationConstructor(
            _pool('u', 'v'), complexity_interval=(1, 4))
        system = _PassedSystem(['u', 'v'])
        for variable in ('u', 'v'):
            entry = {'optimizable': True, 'value': value, 'bounds': (1, 11)}
            key = (COMPLEXITY_METAPARAMETER, variable)
            system.metaparameters[key] = entry
            system.vals[variable].metaparameters[key] = dict(entry)
        shared = system.metaparameters[U_BUDGET]
        constructor.applyToPassed(system)
        for variable in ('u', 'v'):
            key = (COMPLEXITY_METAPARAMETER, variable)
            want = {'optimizable': True, 'value': expected, 'bounds': (1, 4)}
            assert system.metaparameters[key] == want
            assert type(system.metaparameters[key]['value']) is int
            assert system.vals[variable].metaparameters[key] == want
        # A new entry, not an edit of one the caller may share elsewhere.
        assert shared['bounds'] == (1, 11)
        assert system.created_with == [system.vals['u'], system.vals['v']]

    def test_a_budget_is_dropped_in_a_run_without_one(self):
        """The seeded systems carry no budget gene, so a passed one that does
        would fail ``ChromosomeCrossover``'s encoding assertion."""
        constructor = mo_constr.SystemsPopulationConstructor(_pool('u', 'v'))
        system = _PassedSystem(['u', 'v'])
        for variable in ('u', 'v'):
            entry = {'optimizable': True, 'value': 3, 'bounds': (1, 6)}
            key = (COMPLEXITY_METAPARAMETER, variable)
            system.metaparameters[key] = entry
            system.vals[variable].metaparameters[key] = dict(entry)
        constructor.applyToPassed(system)
        for mp in [system.metaparameters] + [system.vals[variable].metaparameters
                                             for variable in ('u', 'v')]:
            assert not any(is_complexity_key(key) for key in mp)
            assert ('sparsity', 'u') in mp or ('sparsity', 'v') in mp
        assert system.created_with == [system.vals['u'], system.vals['v']]


# ---------------------------------------------------------------------------
# The integer step
# ---------------------------------------------------------------------------
@pytest.fixture
def fixed_step(monkeypatch):
    """Make the step's single random draw deterministic."""
    def install(step):
        def randint(low, high, *args, **kwargs):
            assert (low, high) == (-1, 2)          # {-1, 0, +1}
            return step
        monkeypatch.setattr(np.random, 'randint', randint)
    return install


class TestStep:

    def test_the_transition_matrix_is_symmetric_and_keeps_the_uniform(
            self, fixed_step):
        """Exactly, by enumerating the three equally likely steps: symmetric,
        so no drift, and the uniform seeding distribution is stationary."""
        lo, hi = 1, 5
        n = hi - lo + 1
        P = np.zeros((n, n))
        for step in (-1, 0, 1):
            fixed_step(step)
            for current in range(lo, hi + 1):
                new = step_complexity_budget(current, (lo, hi))
                assert lo <= new <= hi and abs(new - current) <= 1
                P[current - lo, new - lo] += 1 / 3
        assert np.allclose(P, P.T)
        assert np.allclose(P.sum(axis=1), 1.0)
        assert np.allclose(np.full(n, 1 / n) @ P, 1 / n)
        assert P[0, 0] == pytest.approx(2 / 3)          # reflected at the edge

    def test_a_long_walk_stays_in_bounds_and_visits_every_level(self):
        np.random.seed(21)
        value, seen = 3, set()
        for _ in range(3000):
            value = step_complexity_budget(value, (1, 6))
            assert 1 <= value <= 6
            seen.add(value)
        assert seen == set(range(1, 7))

    def test_a_degenerate_range_stays_put(self):
        np.random.seed(0)
        assert {step_complexity_budget(3, (3, 3)) for _ in range(50)} == {3}

    @pytest.mark.parametrize('value, expected', [
        (2.6, 3), (2.4, 2), (99, 6), (np.nan, 6), (0, 6), (-2, 6)])
    def test_the_current_level_is_read_like_the_budget(self, value, expected,
                                                       fixed_step):
        """Nearest integer, clipped; "no budget" steps from the ceiling."""
        fixed_step(0)
        assert step_complexity_budget(value, (1, 6)) == expected


class _NoOpEquationMutation(CompoundOperator):
    key = 'NoOpEquationMutation'

    def apply(self, objective, arguments):
        return objective

    def use_default_tags(self):
        self._tags = {'mutation', 'gene level', 'no suboperators'}


class _Equation:
    """What SystemMutation touches on an equation."""

    def __init__(self, metaparameters):
        self.metaparameters = metaparameters
        self.resets = []

    def reset_state(self, reset_right_part=True):
        self.resets.append(reset_right_part)


def _system(params, equation_params=None):
    """One-equation system; the equation shares the system's metaparameters
    dict unless given its own (the state a gene swap leaves)."""
    metaparameters = {key: dict(value) for key, value in params.items()}
    equation = _Equation(metaparameters if equation_params is None
                         else equation_params)
    vals = Chromosome({'u': equation},
                      {key: value for key, value in metaparameters.items()
                       if value['optimizable']})
    return SimpleNamespace(vals=vals, metaparameters=metaparameters), equation


def _mutation():
    param_mutation = MetaparameterMutation(['std', 'mean'])
    param_mutation.params = {'std': 1.0, 'mean': 0.0}
    mutation = SystemMutation(['indiv_mutation_prob'])
    mutation.params = {'indiv_mutation_prob': 1}
    mutation.set_suboperators({'equation_mutation': _NoOpEquationMutation(),
                               'param_mutation': param_mutation})
    return mutation


ALPHA = {('sparsity', 'u'): {'optimizable': True, 'value': 0.5}}
BUDGET = {U_BUDGET: {'optimizable': True, 'value': 3, 'bounds': (1, 6)}}


class TestSystemMutation:

    def test_the_alpha_keeps_metaparameter_mutation_exactly(self):
        """A default run's chromosome: one alpha gene, one normal draw."""
        system, equation = _system(ALPHA)
        np.random.seed(5)
        _mutation().apply(system, {})
        after = np.random.get_state()

        np.random.seed(5)
        expected = 0.5 + np.random.normal(0.0, 1.0 * 0.5)
        expected = np.float64(-expected if expected < 0 else expected)
        assert _same_state(after, np.random.get_state())
        assert system.vals[('sparsity', 'u')] == expected
        assert equation.metaparameters[('sparsity', 'u')]['value'] == expected
        assert equation.resets == []

    def test_the_budget_takes_the_integer_step(self):
        system, equation = _system({**ALPHA, **BUDGET})
        np.random.seed(5)
        _mutation().apply(system, {})

        np.random.seed(5)
        alpha = 0.5 + np.random.normal(0.0, 0.5)
        step = np.random.randint(-1, 2)
        assert system.vals[('sparsity', 'u')] == abs(alpha)
        assert system.vals[U_BUDGET] == 3 + step
        assert type(system.vals[U_BUDGET]) is int
        assert equation.metaparameters[U_BUDGET]['value'] == 3 + step
        assert equation.metaparameters[U_BUDGET]['bounds'] == (1, 6)

    def test_the_fit_is_reopened_exactly_when_the_budget_moves(self):
        outcomes = set()
        for seed in range(60):
            system, equation = _system(BUDGET)
            np.random.seed(seed)
            _mutation().apply(system, {})
            moved = system.vals[U_BUDGET] != 3
            outcomes.add(moved)
            assert equation.resets == ([True] if moved else [])
        assert outcomes == {True, False}

    def test_many_generations_stay_in_bounds(self):
        system, _ = _system({U_BUDGET: {'optimizable': True, 'value': 2,
                                        'bounds': (1, 4)}})
        mutation = _mutation()
        np.random.seed(8)
        seen = set()
        for _ in range(500):
            mutation.apply(system, {})
            seen.add(system.vals[U_BUDGET])
        assert seen == {1, 2, 3, 4}

    def test_moved_is_judged_against_the_equations_own_fit(self, fixed_step):
        """After a gene swap the equation was fitted under ITS dict's budget;
        a zero step on a gene that disagrees with it is still a change."""
        fixed_step(0)
        own = {U_BUDGET: {'optimizable': True, 'value': 5, 'bounds': (1, 6)}}
        system, equation = _system(BUDGET, equation_params=own)
        _mutation().apply(system, {})
        assert system.vals[U_BUDGET] == 3
        assert own[U_BUDGET]['value'] == 3
        assert equation.resets == [True]

    def test_an_equation_without_the_key_receives_it(self, fixed_step):
        fixed_step(1)
        own = {}
        system, equation = _system(BUDGET, equation_params=own)
        _mutation().apply(system, {})
        assert own[U_BUDGET] == {'optimizable': True, 'value': 4,
                                 'bounds': (1, 6)}
        assert equation.resets == [True]

    def test_a_default_chromosome_keeps_its_dict_objects(self):
        """No budget gene, no rebuild: the system and its equation still
        share the very dict they started with."""
        system, equation = _system(ALPHA)
        before = system.metaparameters
        assert equation.metaparameters is before
        _mutation().apply(system, {})
        assert system.metaparameters is before
        assert equation.metaparameters is before

    def test_the_budget_path_restores_one_private_dict(self, fixed_step):
        """After a gene swap the equation holds ANOTHER system's dict; the
        mutation leaves the system and its equation sharing one fresh dict
        whose values are the genes, and the other system's dict untouched by
        anything this system does afterwards."""
        fixed_step(1)
        foreign = {key: dict(value) for key, value in {**ALPHA, **BUDGET}.items()}
        foreign[U_BUDGET]['value'] = 5
        system, equation = _system({**ALPHA, **BUDGET}, equation_params=foreign)
        stale = system.metaparameters
        _mutation().apply(system, {})
        private = system.metaparameters
        assert private is not stale and private is not foreign
        assert equation.metaparameters is private
        assert private[U_BUDGET] == {'optimizable': True, 'value': 4,
                                     'bounds': (1, 6)}
        assert private[('sparsity', 'u')]['value'] == system.vals[('sparsity', 'u')]
        # The next step writes into the private dict only.
        foreign_value = foreign[U_BUDGET]['value']
        _mutation().apply(system, {})
        assert system.metaparameters[U_BUDGET]['value'] == 5
        assert foreign[U_BUDGET]['value'] == foreign_value


# ---------------------------------------------------------------------------
# Real systems
# ---------------------------------------------------------------------------
@pytest.fixture(scope='module')
def u_pool():
    search = epde.EpdeSearch(verbose_params={'show_iter_idx': False})
    t = np.linspace(0.0, 4 * np.pi, 200)
    _, domain = search.createDomain(t, boundary_width=10, ID=91)
    search.set_preprocessor(default_preprocessor_type='FD',
                            preprocessor_kwargs={})
    _, trajectory = search.createTrajectory({'u': np.sin(t)}, domain,
                                            cache_id=91)
    search.create_pool(data=[trajectory], max_deriv_order=(1,), data_fun_pow=1)
    return search.pool


class TestRealSystems:
    """Every place that encodes, copies or prints metaparameters copes."""

    def _seeded(self, pool, complexity_interval=None):
        constructor = mo_constr.SystemsPopulationConstructor(
            pool, terms_number=4, max_factors_in_term=1,
            complexity_interval=complexity_interval)
        return constructor.create()

    def test_a_default_system_has_only_the_alpha_gene(self, u_pool):
        np.random.seed(0)
        system = self._seeded(u_pool)
        assert system.vals.params_keys == [('sparsity', 'u')]
        assert U_BUDGET not in system.vals['u'].metaparameters

    def test_the_budget_is_a_gene_that_survives_copies(self, u_pool):
        np.random.seed(0)
        system = self._seeded(u_pool, complexity_interval=(1, 4))
        assert system.vals.params_keys == [('sparsity', 'u'), U_BUDGET]
        value = system.vals[U_BUDGET]
        assert type(value) is int and 1 <= value <= 4
        equation = system.vals['u']
        assert equation.metaparameters[U_BUDGET]['value'] == value

        clone = copy.deepcopy(system)
        assert clone.vals[U_BUDGET] == value
        assert clone.vals['u'].metaparameters[U_BUDGET] == \
            equation.metaparameters[U_BUDGET]
        # Copied, not shared: the offspring's walk is its own.
        assert clone.vals['u'].metaparameters is not equation.metaparameters
        assert "('complexity', 'u')" in system.text_form
        assert system.vals.same_encoding(
            self._seeded(u_pool, complexity_interval=(1, 4)).vals)

    def test_the_mutation_writes_the_step_into_the_equation(self, u_pool):
        np.random.seed(1)
        system = self._seeded(u_pool, complexity_interval=(1, 4))
        equation = system.vals['u']
        mutation = _mutation()
        outcomes = set()
        for seed in range(30):
            equation.right_part_selected = True      # as RPS leaves it
            before = system.vals[U_BUDGET]
            np.random.seed(seed)
            mutation.apply(system, {})
            after = system.vals[U_BUDGET]
            assert abs(after - before) <= 1 and 1 <= after <= 4
            assert equation.metaparameters[U_BUDGET]['value'] == after
            assert equation.right_part_selected is (after == before)
            outcomes.add(after == before)
        assert outcomes == {True, False}


@pytest.fixture(scope='module')
def uv_pool():
    search = epde.EpdeSearch(verbose_params={'show_iter_idx': False})
    t = np.linspace(0.0, 4 * np.pi, 200)
    _, domain = search.createDomain(t, boundary_width=10, ID=95)
    search.set_preprocessor(default_preprocessor_type='FD',
                            preprocessor_kwargs={})
    _, trajectory = search.createTrajectory(
        {'u': np.sin(t), 'v': np.cos(t)}, domain, cache_id=95)
    search.create_pool(data=[trajectory], max_deriv_order=(1,), data_fun_pow=1)
    return search.pool


class _CloneShellCrossover(CompoundOperator):
    """Stands in for ``EquationCrossover``'s output stage: each offspring
    equation is its parent's ``clone_shell`` refilled with the parent's terms
    -- the step that deep-copies the metaparameters into a fresh dict."""
    key = 'CloneShellCrossover'

    def apply(self, objective, arguments):
        children = []
        for parent in objective:
            child = parent.clone_shell()
            child.structure = copy.deepcopy(parent.structure)
            child._invalidate_label_cache()
            child.reset_state(reset_right_part=True)
            children.append(child)
        return tuple(children)

    def use_default_tags(self):
        self._tags = {'crossover', 'gene level', 'no suboperators'}


V_BUDGET = (COMPLEXITY_METAPARAMETER, 'v')


class TestCrossoverThenMutation:
    """The system-level dict follows the genes again after a mutation.

    Crossover breaks ``SoEq.create``'s one-dict-per-system layout: the gene
    swap hands an equation over with its old system's dict, the equation
    crossover deep-copies a fresh dict per equation. The genes and the
    equations' own dicts stay right; ``system.metaparameters`` -- what
    ``text_form`` prints and ``create`` rebuilds from -- did not.
    """

    BOUNDS = (1, 4)

    def _parents(self, uv_pool):
        constructor = mo_constr.SystemsPopulationConstructor(
            uv_pool, terms_number=4, max_factors_in_term=1,
            complexity_interval=self.BOUNDS)
        np.random.seed(2)
        parents = constructor.create(), constructor.create()
        for system, budgets in zip(parents, ((1, 2), (3, 4))):
            for key, value in zip((U_BUDGET, V_BUDGET), budgets):
                system.vals.replace_gene(gene_key=key, value=value)
                system.vals.pass_parametric_gene(key=key, value=value)
        return parents

    def _offspring(self, uv_pool, monkeypatch):
        """As ``ParetoLevelsCrossover`` builds them: deep copies, reset, then
        ``ChromosomeCrossover`` swapping 'u' and recombining 'v'."""
        first, second = (copy.deepcopy(parent)
                         for parent in self._parents(uv_pool))
        first.reset_state(True)
        second.reset_state(True)
        crossover = ChromosomeCrossover(['equation_exchange_prob'])
        crossover.params = {'equation_exchange_prob': 0.5}
        crossover.set_suboperators({'equation_crossover': _CloneShellCrossover()})
        # One draw per equation gene, in gene order: swap 'u', recombine 'v'.
        draws = iter([0.0 if key == 'u' else 0.99
                      for key in first.vals.equation_keys])
        with monkeypatch.context() as patch:
            patch.setattr(variation.random, 'random', lambda: next(draws))
            first, second = crossover.apply((first, second), {})
        # The defect's precondition: the swapped 'u' equations hold each
        # other's system dict, the recombined 'v' ones a fresh copy.
        assert first.vals['u'].metaparameters is second.metaparameters
        assert second.vals['u'].metaparameters is first.metaparameters
        assert first.vals['v'].metaparameters is not first.metaparameters
        return first, second

    @staticmethod
    def _budget_text(system, key):
        return '{0!r}: {1!r}'.format(key, {'optimizable': True,
                                           'value': system.vals[key],
                                           'bounds': TestCrossoverThenMutation.BOUNDS})

    def test_text_form_and_create_follow_each_systems_genes(
            self, uv_pool, monkeypatch):
        first, second = self._offspring(uv_pool, monkeypatch)
        mutation = _mutation()
        np.random.seed(7)
        for system in (second, first):          # OffspringUpdater pops LIFO
            mutation.apply(system, {})
        for system in (first, second):
            # What is printed (and snapshotted into _pareto_history) is the
            # system's own budget...
            for key in (U_BUDGET, V_BUDGET):
                assert self._budget_text(system, key) in system.text_form
                assert system.metaparameters[key]['value'] == system.vals[key]
            # ...because there is one dict per system again, shared with no
            # other system.
            assert all(equation.metaparameters is system.metaparameters
                       for equation in system.vals)
        assert first.metaparameters is not second.metaparameters

        genes = {key: second.vals[key] for key in (U_BUDGET, V_BUDGET)}
        u_eq_of_first = first.vals['u']
        u_budget_of_first = u_eq_of_first.metaparameters[U_BUDGET]['value']
        np.random.seed(8)
        second.create()                         # the updater's fallback
        for key, value in genes.items():
            assert second.vals[key] == value
            assert type(second.vals[key]) is int
        assert all(equation.metaparameters is second.metaparameters
                   for equation in second.vals)
        # ...and the rebuilt system's next steps stay its own.
        for _ in range(5):
            mutation.apply(second, {})
        assert u_eq_of_first.metaparameters[U_BUDGET]['value'] == \
            u_budget_of_first == first.vals[U_BUDGET]


class TestRealPassedSystems:
    """``applyToPassed`` on real systems: the encoding a crossover asserts."""

    def _seeded(self, pool, complexity_interval):
        constructor = mo_constr.SystemsPopulationConstructor(
            pool, terms_number=4, max_factors_in_term=1,
            complexity_interval=complexity_interval)
        return constructor.create()

    def test_a_budget_system_joins_a_run_without_budgets(self, uv_pool):
        np.random.seed(3)
        passed = self._seeded(uv_pool, (1, 11))
        seeded = self._seeded(uv_pool, None)
        assert not passed.vals.same_encoding(seeded.vals)
        mo_constr.SystemsPopulationConstructor(uv_pool).applyToPassed(passed)
        assert passed.vals.same_encoding(seeded.vals)
        for mp in [passed.metaparameters] + [eq.metaparameters
                                             for eq in passed.vals]:
            assert not any(is_complexity_key(key) for key in mp)

    def test_a_wider_budget_is_rebounded_into_this_run(self, uv_pool):
        np.random.seed(4)
        passed = self._seeded(uv_pool, (1, 11))
        for key in (U_BUDGET, V_BUDGET):
            passed.vals.replace_gene(gene_key=key, value=9)
            passed.vals.pass_parametric_gene(key=key, value=9)
        passed.metaparameters[U_BUDGET]['value'] = 9
        passed.metaparameters[V_BUDGET]['value'] = 9
        constructor = mo_constr.SystemsPopulationConstructor(
            uv_pool, terms_number=4, max_factors_in_term=1,
            complexity_interval=(1, 4))
        constructor.applyToPassed(passed)
        assert passed.vals.same_encoding(constructor.create().vals)
        for key in (U_BUDGET, V_BUDGET):
            assert passed.vals[key] == 4
            assert passed.metaparameters[key]['bounds'] == (1, 4)
            assert passed.vals[key[1]].metaparameters[key] == {
                'optimizable': True, 'value': 4, 'bounds': (1, 4)}


# ---------------------------------------------------------------------------
# EpdeSearch.fit
# ---------------------------------------------------------------------------
def _search(**kwargs):
    search = epde.EpdeSearch(verbose_params={'show_iter_idx': False}, **kwargs)
    search.set_preprocessor(default_preprocessor_type='FD',
                            preprocessor_kwargs={})
    return search


def _trajectory(search, cache_id):
    grid = np.linspace(0, 4 * np.pi, 60)
    domain = search.createDomain(grid, boundary_width=5, ID=cache_id)[1]
    return search.createTrajectory({'u': np.sin(grid)}, domain,
                                   cache_id=cache_id)[1]


BUDGET_KWARGS = dict(sparsity_cls='knee',
                     sparsity_kwargs={'realization': 'complexity'})


class TestFit:

    def _instruct(self, monkeypatch, search, **fit_kwargs):
        """The population constructor's arguments, with no search run."""
        captured = {}

        def create_optimizer(multiobjective_mode, init_params, director,
                             population=None, second_objective=None):
            captured.update(init_params['population_instruct'])
            return SimpleNamespace(optimize=lambda **kwargs: None)

        monkeypatch.setattr(type(search), '_create_optimizer',
                            staticmethod(create_optimizer))
        search.fit(data=[_trajectory(search, 93)], max_deriv_order=(1,),
                   **fit_kwargs)
        return captured

    def test_the_constructor_gets_the_range_from_the_fit_maxima(
            self, monkeypatch):
        instruct = self._instruct(monkeypatch, _search(**BUDGET_KWARGS),
                                  equation_terms_max_number=4,
                                  equation_factors_max_number=2)
        assert instruct['complexity_interval'] == (1, 7)
        # The alpha's seeding is untouched: Knee's degenerate interval.
        assert instruct['sparsity_interval'] == (1.0, 1.0)

    def test_the_dict_factor_form_takes_its_largest_entry(self, monkeypatch):
        instruct = self._instruct(
            monkeypatch, _search(**BUDGET_KWARGS),
            equation_terms_max_number=6,
            equation_factors_max_number={'factors_num': [1, 2],
                                         'probas': [0.65, 0.35]})
        assert instruct['complexity_interval'] == (1, 11)

    @pytest.mark.parametrize('kwargs', [
        {}, {'sparsity_cls': 'knee'},
        {'sparsity_cls': 'knee', 'sparsity_kwargs': {'realization': 'knee'}},
        {'sparsity_cls': 'lasso'}])
    def test_no_budget_reaches_any_other_run(self, monkeypatch, kwargs):
        instruct = self._instruct(monkeypatch, _search(**kwargs))
        assert set(instruct) == {'pool', 'terms_number', 'max_factors_in_term',
                                 'sparsity_interval', 'second_objective'}

    def _fit_without_search(self, monkeypatch, search):
        monkeypatch.setattr(type(search), '_run_optimization',
                            lambda self, *args, **kwargs: None)
        search.fit(data=[_trajectory(search, 94)], max_deriv_order=(1,))

    def test_single_objective_mode_warns_that_budgets_stay_put(
            self, monkeypatch):
        search = _search(multiobjective_mode=False, **BUDGET_KWARGS)
        with pytest.warns(global_var.EPDEUsageWarning,
                          match='single-objective mode cannot evolve') as caught:
            self._fit_without_search(monkeypatch, search)
        mine = [w for w in caught if 'complexity budget' in str(w.message)]
        assert len(mine) == 1
        assert mine[0].filename == __file__, mine[0].filename

    @pytest.mark.parametrize('kwargs', [
        dict(BUDGET_KWARGS), {'multiobjective_mode': False}])
    def test_no_warning_where_the_budget_evolves_or_does_not_exist(
            self, monkeypatch, kwargs):
        search = _search(**kwargs)
        with warnings.catch_warnings():
            warnings.simplefilter('error', global_var.EPDEUsageWarning)
            self._fit_without_search(monkeypatch, search)
