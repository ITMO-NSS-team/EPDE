"""The sparsity interval belongs to the operator that reads it.

``eq_sparsity_interval`` seeds the per-equation ``('sparsity', var)``
metaparameter, which is the ``alpha`` of the legacy LASSO estimator. It is an
*initial* range only -- metaparameter mutation and crossover move alpha outside
it during the search -- and ``LASSOSparsity.apply`` is the only reader of the
value in the tree. So it is a parameter of that estimator, not a description of
the space of equations being searched, and it is no longer a ``search_space``
config key: the default lives on the class, and its config home is
``objectives.sparsity_kwargs['initial_sparsity_interval']``. ``fit``'s
``eq_sparsity_interval`` is deprecated: it still overrides the operator's
interval for that fit, with an ``EPDEDeprecationWarning``.

These tests pin the three claims that justify the placement: the class owns the
historical default, the default (VWSR) pipeline declares that it ignores the
value, and nothing else reads it.

They also pin what a default run does with its metaparameters, which the
removed knee operator's ``('complexity', var)`` budget never touched: the
population draws exactly the alpha and no other random number, the system
mutation moves the alpha with ``MetaparameterMutation`` alone, and the
population constructor is handed exactly its five arguments. A passed system
that still carries the retired budget is stripped of it, with a warning.
"""

import re
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import epde
import epde.globals as global_var
import epde.optimizers.moeadd.population_constr as mo_constr
from epde.interface.search_config import (KEY_GROUP, SearchSpaceConfig,
                                          load_search_config,
                                          sparsity_settings)
from epde.operators.common.sparsity import (LASSOSparsity, VWSRSparsity,
                                            build_sparsity_operator,
                                            initial_sparsity_interval)
from epde.operators.multiobjective.mutations import (MetaparameterMutation,
                                                     SystemMutation)
from epde.operators.utils.template import CompoundOperator
from epde.structure.encoding import Chromosome
from epde.structure.main_structures import check_metaparameters

EPDE_ROOT = Path(epde.__file__).parent


class TestOwnership:

    def test_lasso_owns_the_historical_default(self):
        """(1e-4, 2.5) -- what ``fit``'s signature used to hardcode."""
        assert LASSOSparsity.initial_sparsity_interval == (1e-4, 2.5)

    def test_vwsr_declares_that_it_ignores_the_value(self):
        lo, hi = VWSRSparsity.initial_sparsity_interval
        assert lo == hi == 1.0

    def test_helper_reads_the_class(self):
        assert initial_sparsity_interval(LASSOSparsity) == (1e-4, 2.5)
        assert initial_sparsity_interval(VWSRSparsity) == (1.0, 1.0)

    def test_unknown_operator_gets_the_neutral_interval(self):
        class CustomSparsity:
            pass

        assert initial_sparsity_interval(CustomSparsity) == (1.0, 1.0)
        assert initial_sparsity_interval(None) == (1.0, 1.0)

    def test_the_neutral_value_matches_the_metaparameter_default(self):
        """1.0 is what an SoEq carries when nobody seeds it."""
        from epde.structure.main_structures import             _DEFAULT_EQUATION_METAPARAMETERS as defaults

        assert defaults['sparsity']['value'] == 1.0


class TestNotASearchSpaceSetting:

    def test_it_is_not_a_config_key(self):
        assert 'eq_sparsity_interval' not in KEY_GROUP

    def test_it_is_not_a_search_space_field(self):
        from dataclasses import fields

        assert 'eq_sparsity_interval' not in [f.name for f in
                                              fields(SearchSpaceConfig)]

    def test_setting_it_in_a_config_is_refused(self):
        with pytest.raises(ValueError, match='Unknown key'):
            load_search_config({'search_space':
                                {'eq_sparsity_interval': [1e-3, 1.0]}})

    def test_passing_it_to_the_constructor_is_refused(self):
        with pytest.raises(ValueError, match='Unknown search-config parameter'):
            epde.EpdeSearch(eq_sparsity_interval=(1e-3, 1.0),
                            verbose_params={'show_iter_idx': False})


class TestSoleReader:
    """If a second consumer of the metaparameter ever appears, the value stops
    being one operator's parameter and this placement stops being right."""

    def _metaparameter_lines(self):
        pattern = re.compile(r"metaparameters\s*\[\s*\(\s*['\"]sparsity['\"]")
        for path in EPDE_ROOT.rglob('*.py'):
            for lineno, line in enumerate(
                    path.read_text(encoding='utf-8', errors='ignore')
                        .splitlines(), 1):
                if pattern.search(line) and not line.lstrip().startswith('#'):
                    yield path, lineno, line

    def test_only_the_lasso_operator_reads_the_value(self):
        readers = [(path, lineno) for path, lineno, line
                   in self._metaparameter_lines()
                   if "['value']" in line and '=' not in line.split(']')[-1]]
        assert readers, 'the reader disappeared -- did the seeding change?'
        for path, lineno in readers:
            assert path.name == 'sparsity.py', '%s:%s' % (path, lineno)

    def test_the_writers_are_the_population_constructors(self):
        writers = {path.name for path, _, line in self._metaparameter_lines()
                   if line.lstrip().startswith('metaparameters[')}
        assert writers <= {'population_constr.py', 'equation_translator.py'}


class TestSparsityKwargs:
    """``objectives.sparsity_kwargs`` is the config home for it: a map that
    travels with ``sparsity_cls`` and configures that operator."""

    def test_the_group_carries_an_empty_map_by_default(self):
        assert load_search_config().objectives.sparsity_kwargs == {}

    def test_it_configures_the_chosen_operator(self):
        cfg = load_search_config({'objectives': {
            'sparsity_cls': 'lasso',
            'sparsity_kwargs': {'initial_sparsity_interval': [1e-3, 1.0]}}})
        assert cfg.objectives.sparsity_kwargs == {
            'initial_sparsity_interval': (1e-3, 1.0)}      # list -> tuple

    def test_a_kwarg_overrides_a_file(self):
        cfg = load_search_config(
            {'objectives': {'sparsity_cls': 'lasso',
                            'sparsity_kwargs': {
                                'initial_sparsity_interval': [1e-3, 1.0]}}},
            overrides={'sparsity_kwargs': {
                'initial_sparsity_interval': (1e-6, 9.0)}})
        assert cfg.objectives.sparsity_kwargs['initial_sparsity_interval']             == (1e-6, 9.0)

    def test_an_unknown_setting_is_refused(self):
        """A typo would otherwise sit on the operator doing nothing -- the
        failure this config layer exists to remove."""
        with pytest.raises(ValueError, match='initial_sparsity_interval'):
            load_search_config({'objectives': {
                'sparsity_kwargs': {'initial_sparsity_intervals': [1, 2]}}})

    def test_the_error_names_the_operator(self):
        with pytest.raises(ValueError, match='LASSOSparsity'):
            load_search_config({'objectives': {
                'sparsity_cls': 'lasso',
                'sparsity_kwargs': {'alpha': 0.1}}})

    def test_the_settable_names_come_from_the_class(self):
        assert 'initial_sparsity_interval' in sparsity_settings(LASSOSparsity)
        assert 'key' not in sparsity_settings(LASSOSparsity)

    def test_the_builder_applies_them_to_the_instance(self):
        operator = build_sparsity_operator(
            LASSOSparsity, {'initial_sparsity_interval': (1e-3, 1.0)})
        assert isinstance(operator, LASSOSparsity)
        assert operator.initial_sparsity_interval == (1e-3, 1.0)
        assert LASSOSparsity.initial_sparsity_interval == (1e-4, 2.5)

    def test_the_builder_defaults_to_vwsr(self):
        assert isinstance(build_sparsity_operator(), VWSRSparsity)

    @pytest.mark.parametrize('director_cls, extra', [
        ('epde.optimizers.moeadd.strategy:MOEADDDirector', {}),
        ('epde.optimizers.single_criterion.strategy:BaselineDirector', {}),
    ])
    def test_both_directors_accept_the_map(self, director_cls, extra):
        """The single-objective branch used to hardcode VWSRSparsity, so
        sparsity_cls was silently ignored there."""
        import importlib
        import inspect

        module, name = director_cls.split(':')
        cls = getattr(importlib.import_module(module), name)
        params = inspect.signature(cls.use_baseline).parameters
        assert 'sparsity_cls' in params
        assert 'sparsity_kwargs' in params


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


class TestFitResolution:
    """``fit`` takes its default from the sparsity operator in use, and says so
    when the operator in use cannot act on an explicitly passed one."""

    def _captured_interval(self, monkeypatch, search, **fit_kwargs):
        captured = {}

        def recorder(self, terms, factors, interval, optimizer=None,
                     population=None):
            captured['interval'] = interval

        monkeypatch.setattr(type(search), '_run_optimization', recorder)
        search.fit(data=[_trajectory(search, 70)], max_deriv_order=(2,),
                   **fit_kwargs)
        return captured['interval']

    def test_default_comes_from_the_lasso_class(self, monkeypatch):
        search = _search(sparsity_cls='lasso')
        assert self._captured_interval(monkeypatch, search) == (1e-4, 2.5)

    def test_a_configured_interval_is_used(self, monkeypatch):
        search = _search(sparsity_cls='lasso',
                         sparsity_kwargs={'initial_sparsity_interval':
                                          (1e-3, 1.0)})
        assert self._captured_interval(monkeypatch, search) == (1e-3, 1.0)

    def test_the_fit_argument_still_wins_over_the_config(self, monkeypatch):
        search = _search(sparsity_cls='lasso',
                         sparsity_kwargs={'initial_sparsity_interval':
                                          (1e-3, 1.0)})
        with pytest.warns(DeprecationWarning, match='sparsity_kwargs'):
            got = self._captured_interval(monkeypatch, search,
                                          eq_sparsity_interval=(1e-6, 9.0))
        assert got == (1e-6, 9.0)

    def test_default_is_neutral_under_vwsr(self, monkeypatch):
        search = _search()
        assert self._captured_interval(monkeypatch, search) == (1.0, 1.0)

    def test_an_explicit_interval_still_wins_for_lasso(self, monkeypatch):
        search = _search(sparsity_cls='lasso')
        with pytest.warns(DeprecationWarning, match='sparsity_kwargs'):
            got = self._captured_interval(monkeypatch, search,
                                          eq_sparsity_interval=(1e-3, 1.0))
        assert got == (1e-3, 1.0)

    def test_an_explicit_interval_warns_under_vwsr(self):
        search = _search()
        with pytest.warns(UserWarning, match='VWSRSparsity never reads it'):
            with pytest.raises(ValueError):      # no data, no pool
                search.fit(eq_sparsity_interval=(1e-3, 1.0))

    def test_the_warning_points_at_the_caller(self):
        """``fit`` is wrapped by @_loop_stats.timed, so the naive stacklevel
        blames _loop_stats.py -- and a diagnostic filed against a file the
        reader does not own is not a diagnostic."""
        search = _search()
        with pytest.warns(UserWarning) as caught:
            with pytest.raises(ValueError):
                search.fit(eq_sparsity_interval=(1e-3, 1.0))
        mine = [w for w in caught if 'eq_sparsity_interval' in str(w.message)]
        assert mine, 'no warning emitted'
        assert mine[0].filename == __file__, mine[0].filename

    def test_only_a_deprecation_when_the_operator_reads_it(self):
        """The interval is LASSO's parameter: passed to ``fit`` it is deprecated
        in favour of ``sparsity_kwargs`` -- at the caller's line -- but not
        flagged as ignored."""
        import warnings

        search = _search(sparsity_cls='lasso')
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            with pytest.raises(ValueError):
                search.fit(eq_sparsity_interval=(1e-3, 1.0))
        mine = [w for w in caught if 'eq_sparsity_interval' in str(w.message)]
        assert [w.category for w in mine] == [global_var.EPDEDeprecationWarning]
        assert "sparsity_kwargs={'initial_sparsity_interval': (0.001, 1.0)}" \
            in str(mine[0].message)
        assert mine[0].filename == __file__, mine[0].filename


# ---------------------------------------------------------------------------
# A default run's metaparameters
# ---------------------------------------------------------------------------
class _RecordingSoEq:
    """Stands in for ``SoEq`` inside the constructor: validates and keeps the
    metaparameters it is built with, and skips the term generation that needs
    a real token pool."""

    def __init__(self, pool, metaparameters):
        check_metaparameters(metaparameters)
        self.metaparameters = metaparameters

    def set_objective_functions(self, obj_funs):
        pass

    def use_default_multiobjective_function(self, second_objective=None):
        pass

    def create(self):
        pass


def _pool(*variables):
    return SimpleNamespace(families_demand_equation=[
        SimpleNamespace(variable=variable) for variable in variables])


@pytest.fixture
def recording(monkeypatch):
    monkeypatch.setattr(mo_constr, 'SoEq', _RecordingSoEq)


def _same_state(first, second):
    return (first[0] == second[0] and np.array_equal(first[1], second[1])
            and first[2:] == second[2:])


class TestDefaultSeeding:

    def test_a_default_population_draws_exactly_the_alpha(self, recording):
        """The alpha's log-uniform draw, one per equation, and no other
        random number; no metaparameter beyond the caps and the alphas."""
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


def _system(params):
    """One-equation system whose equation shares the system's
    metaparameters dict, as ``SoEq.create`` builds it."""
    metaparameters = {key: dict(value) for key, value in params.items()}
    equation = _Equation(metaparameters)
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


class TestDefaultSystemMutation:

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

    def test_a_default_chromosome_keeps_its_dict_objects(self):
        """The mutation writes the alpha into the dict in place: the system
        and its equation still share the very dict they started with."""
        system, equation = _system(ALPHA)
        before = system.metaparameters
        assert equation.metaparameters is before
        _mutation().apply(system, {})
        assert system.metaparameters is before
        assert equation.metaparameters is before


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


def _seeded(pool):
    return mo_constr.SystemsPopulationConstructor(
        pool, terms_number=4, max_factors_in_term=1).create()


class TestDefaultRealSystems:

    def test_a_default_system_has_only_the_alpha_gene(self, u_pool):
        np.random.seed(0)
        system = _seeded(u_pool)
        assert system.vals.params_keys == [('sparsity', 'u')]


#: The key the removed complexity budget was stored under.
RETIRED_BUDGET = ('complexity', 'u')


class TestRetiredBudget:
    """A system saved from a ``sparsity_cls='knee'`` budget run still carries
    ``('complexity', var)`` as a gene. Nothing reads it any more, but the
    seeded systems do not have it, so ``ChromosomeCrossover``'s encoding
    assertion would end the search at their first crossover: the population
    constructor strips it from a passed system, and says so."""

    def _budget_system(self, pool):
        """A seeded system rebuilt with the budget gene, the way the removed
        constructor gave one to a passed system."""
        system = _seeded(pool)
        entry = {'optimizable': True, 'value': 3, 'bounds': (1, 4)}
        system.metaparameters[RETIRED_BUDGET] = dict(entry)
        system.vals['u'].metaparameters[RETIRED_BUDGET] = dict(entry)
        system.create(passed_equations=[system.vals['u']])
        assert RETIRED_BUDGET in system.vals.params_keys
        return system

    def test_a_budget_system_is_stripped_with_a_warning(self, u_pool):
        np.random.seed(3)
        passed = self._budget_system(u_pool)
        seeded = _seeded(u_pool)
        assert not passed.vals.same_encoding(seeded.vals)
        equation = passed.vals['u']
        with pytest.warns(global_var.EPDEUsageWarning,
                          match='removed complexity budget'):
            mo_constr.SystemsPopulationConstructor(u_pool).applyToPassed(passed)
        assert passed.vals.same_encoding(seeded.vals)
        assert passed.vals['u'] is equation            # the structure is kept
        for mp in (passed.metaparameters, equation.metaparameters):
            assert RETIRED_BUDGET not in mp
            assert ('sparsity', 'u') in mp

    def test_a_system_without_it_is_not_touched(self, u_pool):
        np.random.seed(4)
        passed = _seeded(u_pool)
        vals, metaparameters = passed.vals, dict(passed.metaparameters)
        with warnings.catch_warnings():
            warnings.simplefilter('error', global_var.EPDEUsageWarning)
            mo_constr.SystemsPopulationConstructor(u_pool).applyToPassed(passed)
        assert passed.vals is vals                     # no rebuild
        assert passed.metaparameters == metaparameters


class TestDefaultFit:

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

    @pytest.mark.parametrize('kwargs', [{}, {'sparsity_cls': 'lasso'}],
                             ids=['default', 'lasso'])
    def test_the_constructor_gets_exactly_its_five_arguments(self, monkeypatch,
                                                              kwargs):
        instruct = self._instruct(monkeypatch, _search(**kwargs))
        assert set(instruct) == {'pool', 'terms_number', 'max_factors_in_term',
                                 'sparsity_interval', 'second_objective'}
