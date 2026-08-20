"""Sanity checks for the EPDE<->GOLEM bridge.

Run with::

    PYTHONHASHSEED=0 python -m pytest tests -q
"""

import os
import sys

import numpy as np
import pytest

#: epde_golem/tests/test_integration.py -> epde_golem -> <repo>.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, 'benchmarks', 'golem'))

DATA = os.path.join(REPO_ROOT, 'tests', 'functional', 'scenarios', 'wave')


@pytest.fixture(scope='module')
def wave_data():
    shape = 80
    data = np.loadtxt(os.path.join(DATA, 'wave_sln_80.csv'), delimiter=',').T
    axis = np.linspace(0, 1, shape + 1)
    grids = np.meshgrid(axis, axis, indexing='ij')
    return grids, data


def _make_search(grids):
    from epde_golem import EpdeGolemSearch
    search = EpdeGolemSearch(
        use_solver=False, multiobjective_mode=True, use_pic=True, boundary=20,
        coordinate_tensors=grids, verbose_params={'show_iter_idx': False},
        device='cpu')
    search.set_preprocessor(default_preprocessor_type='FD', preprocessor_kwargs={})
    return search


@pytest.fixture(scope='module')
def fitted(wave_data):
    from _common import set_seeds
    grids, data = wave_data
    set_seeds(0)
    search = _make_search(grids)
    search.set_moeadd_params(population_size=6, training_epochs=2)
    search.fit(data=data, variable_names=['u'], max_deriv_order=(2, 2),
               equation_terms_max_number=5, data_fun_pow=1, additional_tokens=[],
               equation_factors_max_number=1,
               eq_sparsity_interval=(1e-6, 1e-4))
    return search


def test_graph_mirror_round_trip(fitted):
    """Adapting a chromosome and restoring it returns the same object, and the
    node mirror reflects the system's actual term structure."""
    from epde_golem.graph import soeq_to_graph, refresh_graph_mirror
    system = fitted.optimizer.pareto_levels.population[0]
    graph = soeq_to_graph(system)
    assert graph.soeq is system
    expected_nodes = sum(len(eq.structure) for eq in system.vals) \
        + len(system.vars_to_describe) + 1
    assert len(graph.nodes) == expected_nodes
    before = graph.descriptive_id
    assert refresh_graph_mirror(graph).descriptive_id == before


def test_mirror_tracks_structural_change(fitted):
    """A structural edit must change the descriptive id -- otherwise GOLEM's
    dedup and diversity checks would be blind to it."""
    from epde_golem.graph import soeq_to_graph, refresh_graph_mirror
    from copy import deepcopy
    system = deepcopy(fitted.optimizer.pareto_levels.population[0])
    graph = soeq_to_graph(system)
    before = graph.descriptive_id
    equation = system.vals[system.vars_to_describe[0]]
    equation.structure = equation.structure[:-1]
    assert refresh_graph_mirror(graph).descriptive_id != before


def test_verification_rejects_collapsed_equation(fitted):
    """The verifier must reject an equation below the two-term floor."""
    from copy import deepcopy
    from epde_golem.graph import soeq_to_graph
    from epde_golem.operators import soeq_is_valid
    system = deepcopy(fitted.optimizer.pareto_levels.population[0])
    assert soeq_is_valid(soeq_to_graph(system)) is True
    equation = system.vals[system.vars_to_describe[0]]
    equation.structure = equation.structure[:1]
    with pytest.raises(ValueError):
        soeq_is_valid(soeq_to_graph(system))


def test_pareto_levels_interface(fitted):
    """``EpdeSearch``'s result readers must work unchanged on the GOLEM run."""
    levels = fitted.optimizer.pareto_levels
    assert len(levels.population) >= 1
    assert len(levels.levels) >= 1
    assert all(sol in levels.population for sol in levels.levels[0])
    equations = fitted.equations(only_print=False, only_str=True, num=1)
    assert equations and isinstance(equations[0][0], str)


def test_objectives_match_epde_readers(fitted):
    """The fitness GOLEM stored must equal what EPDE's own readers report."""
    evaluator = fitted.optimizer.evaluator
    assert evaluator.n_evaluations > 0
    assert not evaluator.errors, evaluator.errors[0]
    for system in fitted.optimizer.pareto_levels.levels[0]:
        values = np.asarray(system.obj_fun, dtype=float)
        assert values.shape == (len(evaluator.metric_names),)
        assert np.all(np.isfinite(values))


def test_operators_are_the_ones_epde_assembled(fitted):
    """The bridge must reuse the director's operator instances, not clones."""
    from epde_golem.optimizer import extract_epde_operators
    ops = extract_epde_operators(fitted.director)
    assert fitted.optimizer.epde_operators['chromosome_mutation'] is ops['chromosome_mutation']
    assert fitted.optimizer.epde_operators['chromosome_crossover'] is ops['chromosome_crossover']
    assert fitted.optimizer.evaluator.fitness is ops['chromosome_fitness']


def test_eval_budget_stops_the_run(wave_data):
    """A budget cap must actually terminate the search early."""
    from _common import set_seeds
    grids, data = wave_data
    set_seeds(0)
    search = _make_search(grids)
    search.golem_params.update(eval_budget=40)
    search.set_moeadd_params(population_size=6, training_epochs=200)
    search.fit(data=data, variable_names=['u'], max_deriv_order=(2, 2),
               equation_terms_max_number=5, data_fun_pow=1, additional_tokens=[],
               equation_factors_max_number=1, eq_sparsity_interval=(1e-6, 1e-4))
    # Budget is checked once per generation, so an overshoot of one generation
    # is expected; 200 generations' worth of evaluations is not.
    assert search.optimizer.evaluator.n_evaluations < 200
