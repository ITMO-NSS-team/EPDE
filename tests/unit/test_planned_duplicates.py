"""``evolution.planned_duplicates`` of per-variable niching: what an assembled child whose mating plan
predicted a repeat (the plan's redraws exhausted) does before its first fit.

Pinned here:

* the configuration -- the default, the menu, the refusal of ``'mutate'`` outside per-variable niching, the
  ``EpdeSearch`` keyword;
* ``per_variable.select`` -- the per-pair flags, and the global random draws it makes;
* ``per_variable.assemble`` -- the first mutation skipped or not, and the plan status the child carries;
* the loop counters of a short fit -- planned pairs and first fits of the assembled children;
* a seeded fit with ``planned_duplicates='fit'`` equals the same fit without the key, and one with
  ``'mutate'`` finishes.
"""
import warnings
from contextlib import contextmanager
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

import epde
import epde.globals as global_var
from epde import _loop_stats
from epde.interface.search_config import (PLANNED_DUPLICATES, active_config, collect_overrides,
                                          load_search_config, set_active_config)
from epde.operators.common import plain_fit as PF
from epde.optimizers.nsga import per_variable as PV

UNIVERSAL = {'system_rps_mode': 'plain', 'engine': 'nsga2'}


# ---------------------------------------------------------------------------
# Helpers.
# ---------------------------------------------------------------------------

def _config(**overrides):
    return load_search_config(overrides=collect_overrides(**overrides))


@contextmanager
def _active(config):
    """Run the block under ``config``, putting back the configuration active before."""
    previous = active_config()
    set_active_config(config)
    try:
        yield config
    finally:
        set_active_config(previous)


def _search(**extra):
    """A small universal search: two variables, four terms, population 8, two epochs."""
    search = epde.EpdeSearch(verbose_params={'show_iter_idx': False}, engine='nsga2', system_rps_mode='plain',
                             discrepancy_metric='wape', instability_metric='dyad',
                             equation_terms_max_number=4, equation_factors_max_number=1,
                             data_fun_pow=1, population_size=8, training_epochs=2, **extra)
    search.set_preprocessor(default_preprocessor_type='FD', preprocessor_kwargs={})
    return search


def _trajectory(search):
    """u = sin t, v = cos t: u' = v, v' = -u."""
    t = np.linspace(0, 4 * np.pi, 300)
    domain = search.createDomain(t, ID=0)[1]
    return search.createTrajectory({'u': np.sin(t), 'v': np.cos(t)}, domain, cache_id=0)[1]


def _run(seed=4, **extra):
    """A seeded fit; what it reports: the front, the per-epoch history and the final population, as text and
    objective values, plus the finished search's population."""
    previous = active_config()
    search = _search(**extra)
    try:
        np.random.seed(seed)
        trajectory = _trajectory(search)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            search.fit(data=[trajectory], max_deriv_order=(1,))
        front = [(s.text_form, [float(v) for v in s.obj_fun]) for s in search.optimizer.reported_front()]
        history = [[(e['text_form'], [float(v) for v in e['obj_fun']]) for e in epoch]
                   for epoch in search.pareto_history]
        population = [(s.text_form, [float(v) for v in s.obj_fun])
                      for s in search.optimizer.pareto_levels.population]
        return SimpleNamespace(front=front, history=history, population=population,
                               systems=list(search.optimizer.pareto_levels.population))
    finally:
        global_var.samples_manager.reset()
        PF.release()
        set_active_config(previous)


def _entries(prefix):
    """The number of loop records under the sites starting with ``prefix``, by site."""
    return {site: bucket['entries'] for site, bucket in _loop_stats._stats.items()
            if site.startswith(prefix) and bucket['entries']}


@pytest.fixture(scope='module')
def fitted():
    """The systems of a finished seeded fit, read only."""
    return _run().systems


def _two_apart(systems):
    """Two fitted systems that differ in the structure of both equations."""
    keys = list(systems[0].vals.equation_keys)
    for i, A in enumerate(systems):
        for B in systems[i + 1:]:
            if all(PV._eq_label(A.vals[k]) != PV._eq_label(B.vals[k]) for k in keys):
                return A, B
    pytest.skip('the fit holds no two systems that differ in every equation')


# ---------------------------------------------------------------------------
# The configuration.
# ---------------------------------------------------------------------------

class TestConfig:

    def test_the_default_is_fit(self):
        assert load_search_config().evolution.planned_duplicates == 'fit'
        assert PLANNED_DUPLICATES == ('fit', 'mutate')

    def test_a_value_outside_the_menu_is_refused(self):
        with pytest.raises(ValueError, match='planned_duplicates must be one of'):
            _config(planned_duplicates='skip', **UNIVERSAL)

    @pytest.mark.parametrize('extra', [{}, {'system_rps_mode': 'plain'},
                                       {'system_rps_mode': 'plain', 'engine': 'nsga2', 'niching': 'system'},
                                       {'system_rps_mode': 'plain', 'engine': 'nsga2', 'niching': 'per_equation'}])
    def test_mutate_is_refused_outside_per_variable_niching(self, extra):
        with pytest.raises(ValueError, match="planned_duplicates='mutate'.*niching"):
            _config(planned_duplicates='mutate', **extra)

    def test_mutate_is_accepted_under_per_variable_niching(self):
        assert _config(planned_duplicates='mutate', **UNIVERSAL).evolution.planned_duplicates == 'mutate'
        assert _config(planned_duplicates='mutate', niching='per_variable',
                       **UNIVERSAL).evolution.planned_duplicates == 'mutate'
        assert _config(planned_duplicates='fit').evolution.planned_duplicates == 'fit'

    def test_the_search_keyword_reaches_the_config(self):
        assert _search(planned_duplicates='mutate')._config.evolution.planned_duplicates == 'mutate'
        assert _search()._config.evolution.planned_duplicates == 'fit'
        with pytest.raises(ValueError, match='planned_duplicates'):
            epde.EpdeSearch(planned_duplicates='mutate')


# ---------------------------------------------------------------------------
# select.
# ---------------------------------------------------------------------------

class TestSelect:

    def test_a_population_of_copies_plans_only_duplicates(self, fitted):
        n = 4
        population = [deepcopy(fitted[0]) for _ in range(n)]
        levels = SimpleNamespace(population=population, history=set())
        operator = SimpleNamespace(params={'parents_fraction': 1.0})
        np.random.seed(11)
        PV.select(operator, levels)
        assert len(levels.pv_plan) == n // 2
        assert levels.pv_duplicates == [True] * (n // 2)

    def test_a_novel_assembled_child_is_not_a_duplicate(self, fitted, monkeypatch):
        A, B = _two_apart(fitted)
        monkeypatch.setattr(PV, '_tournament', lambda T, e, n: e)      # parent A from plane 0, B from plane 1
        levels = SimpleNamespace(population=[deepcopy(A), deepcopy(B)], history=set())
        operator = SimpleNamespace(params={'parents_fraction': 1.0})
        np.random.seed(11)
        PV.select(operator, levels)
        assert len(levels.pv_plan) == 1 and levels.pv_duplicates == [False]

    def test_the_global_draws_are_the_plan_s_own(self, fitted):
        """Per pair: parent B's plane, A's tournament, then every redraw of B (the coins are empty for two
        equations); the draws are the same whatever the redraws end with."""
        n = 4
        population = [deepcopy(fitted[0]) for _ in range(n)]
        levels = SimpleNamespace(population=population, history=set())
        operator = SimpleNamespace(params={'parents_fraction': 1.0})
        np.random.seed(11)
        PV.select(operator, levels)
        after = np.random.get_state()
        np.random.seed(11)
        for _ in range(n // 2):
            np.random.randint(1)
            np.random.randint(n), np.random.randint(n)
            for _ in range(PV.PLAN_REDRAWS + 1):
                np.random.randint(n), np.random.randint(n)
        expected = np.random.get_state()
        assert after[0] == expected[0] and np.array_equal(after[1], expected[1]) and after[2:] == expected[2:]


# ---------------------------------------------------------------------------
# assemble.
# ---------------------------------------------------------------------------

class TestAssemble:

    @staticmethod
    def _assemble(fitted, planned_duplicate, setting):
        A, B = fitted[:2]
        keys = list(A.vals.equation_keys)
        with _active(_config(planned_duplicates=setting, **UNIVERSAL)):
            return PV.assemble(A, keys[0], B, keys[1], keys, frozenset(), planned_duplicate)

    def test_under_fit_every_assembled_child_skips_its_first_mutation(self, fitted):
        for flagged in (True, False):
            child = self._assemble(fitted, flagged, 'fit')
            assert PV.skip_first_mutation(child) is True

    def test_under_mutate_a_planned_duplicate_is_mutated_first(self, fitted):
        child = self._assemble(fitted, True, 'mutate')
        assert PV.skip_first_mutation(child) is False
        child = self._assemble(fitted, False, 'mutate')
        assert PV.skip_first_mutation(child) is True

    def test_the_child_carries_its_plan_status_until_taken(self, fitted):
        child = self._assemble(fitted, True, 'fit')
        assert PV.take_plan_status(child) == 'duplicate'
        assert PV.take_plan_status(child) is None
        child = self._assemble(fitted, False, 'mutate')
        assert PV.take_plan_status(child) == 'new'
        assert PV.take_plan_status(child) is None
        assert PV.take_plan_status(deepcopy(fitted[0])) is None

    def test_crossover_passes_the_flags_to_the_assembled_children(self, fitted, monkeypatch):
        A, B = fitted[:2]
        keys = list(A.vals.equation_keys)
        seen = []
        monkeypatch.setattr(PV, 'assemble', lambda *args: seen.append(args[-1]) or deepcopy(args[0]))
        crossed = SimpleNamespace(apply=lambda objective, arguments: (deepcopy(objective[0]), None))
        operator = SimpleNamespace(suboperators={'chromosome_crossover': crossed})
        objective = SimpleNamespace(population=[A, B])
        subop_args = {'chromosome_crossover': {}}
        plan = [(A, keys[0], B, keys[1])] * 2
        PV.crossover(operator, objective, plan, [frozenset()] * 2, subop_args, [True, False])
        assert seen == [True, False]
        PV.crossover(operator, objective, plan, [frozenset()] * 2, subop_args)
        assert seen[2:] == [False, False]


# ---------------------------------------------------------------------------
# The counters and the fits.
# ---------------------------------------------------------------------------

class TestFits:

    def test_the_counters_count_planned_pairs_and_first_fits(self, monkeypatch):
        monkeypatch.setattr(_loop_stats, '_ENABLED', True)
        _loop_stats.reset()
        planned, assembled = [], []
        original_select, original_assemble = PV.select, PV.assemble

        def select(operator, objective):
            original_select(operator, objective)
            planned.append(len(objective.pv_plan))

        def assemble(*args, **kwargs):
            assembled.append(1)
            return original_assemble(*args, **kwargs)

        monkeypatch.setattr(PV, 'select', select)
        monkeypatch.setattr(PV, 'assemble', assemble)
        try:
            _run()
            plans, firsts = _entries('PerVariable.plan.'), _entries('PerVariable.first_fit.')
        finally:
            _loop_stats.reset()
        assert sum(planned) > 0 and sum(planned) == len(assembled)
        assert sum(plans.values()) == sum(planned)
        assert set(plans) <= {'PerVariable.plan.duplicate', 'PerVariable.plan.new'}
        assert sum(firsts.values()) == len(assembled)
        statuses = {'duplicate', 'new'}
        outcomes = {'placed', 'swapped', 'sibling', 'nothing'}
        assert all(site.split('.')[2] in statuses and site.split('.')[3] in outcomes for site in firsts)

    def test_fit_given_explicitly_equals_the_key_absent(self):
        absent, given = _run(), _run(planned_duplicates='fit')
        assert absent.front and absent.front == given.front
        assert absent.history == given.history
        assert absent.population == given.population

    def test_a_fit_under_mutate_finishes_with_a_front(self):
        assert _run(planned_duplicates='mutate').front
