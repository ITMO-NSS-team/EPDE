"""The factor move of plain-mode mutation stays inside the equation's factors setting.

* ``fill_prune.factor_cap`` reads the setting of the equation (an int, a float or a probability dict whose
  zero-probability counts do not count) and falls back to the active config when the equation has none;
* ``plain_moves._add_factor`` refuses a term that already holds ``cap`` factors, and lets a term grow up to it;
* a plain NSGA-II fit with ``equation_factors_max_number=1`` and ``factor_move_probability=1`` never holds a
  term of two factors, in the reported front or anywhere in the Pareto levels.
"""
from types import SimpleNamespace

import numpy as np

import epde
import epde.globals as global_var
from epde.interface.search_config import active_config
from epde.operators.common import fill_prune as FP
from epde.operators.multiobjective import plain_moves as PM
from epde.structure.main_structures import Term


def _search(**kwargs):
    search = epde.EpdeSearch(verbose_params={'show_iter_idx': False}, engine='nsga2', system_rps_mode='plain',
                             discrepancy_metric='wape', instability_metric='dyad',
                             equation_terms_max_number=4, equation_factors_max_number=1,
                             data_fun_pow=1, population_size=8, training_epochs=2, **kwargs)
    search.set_preprocessor(default_preprocessor_type='FD', preprocessor_kwargs={})
    return search


def _trajectory(search, names):
    t = np.linspace(0, 4 * np.pi, 300)
    funs = {'u': np.sin(t), 'v': np.cos(t), 'w': np.sin(2 * t)}
    domain = search.createDomain(t, ID=0)[1]
    return search.createTrajectory({name: funs[name] for name in names}, domain, cache_id=0)[1]


def _equation(setting):
    return SimpleNamespace(metaparameters={'max_factors_in_term': {'value': setting}})


def test_factor_cap_reads_the_equations_setting():
    assert FP.factor_cap(_equation(1)) == 1
    assert FP.factor_cap(_equation(1.0)) == 1
    assert FP.factor_cap(_equation(3)) == 3
    assert FP.factor_cap(_equation({'factors_num': [1, 2], 'probas': [0.65, 0.35]})) == 2
    assert FP.factor_cap(_equation({'factors_num': [1, 2], 'probas': [1.0, 0.0]})) == 1


def test_factor_cap_falls_back_to_the_active_config():
    search = _search()
    try:
        expected = active_config().search_space.equation_factors_max_number
        assert FP.factor_cap(SimpleNamespace(metaparameters={})) == int(expected)
    finally:
        global_var.samples_manager.reset()
    del search


def test_add_factor_honours_the_cap():
    search = _search()
    try:
        search.create_pool(data=[_trajectory(search, ('u', 'v'))], max_deriv_order=(1,))
        pool = search.pool

        for seed in range(20):
            np.random.seed(seed)
            term = Term(pool, max_factors_in_term=1)
            before = list(term.structure)
            assert PM._add_factor(term, 1) is False
            assert term.structure == before

        grown = None
        for seed in range(50):
            np.random.seed(seed)
            term = Term(pool, max_factors_in_term=1)
            if PM._add_factor(term, 2):
                grown = term
                break
        assert grown is not None and len(grown.structure) == 2
        two = list(grown.structure)
        assert PM._add_factor(grown, 2) is False
        assert grown.structure == two

        np.random.seed(0)
        for _ in range(100):
            if PM._add_factor(grown, 3):
                break
        assert len(grown.structure) == 3
        assert PM._add_factor(grown, 3) is False
    finally:
        global_var.samples_manager.reset()


def test_a_one_factor_fit_never_builds_a_two_factor_term():
    search = _search(system_rps_params={'factor_move_probability': 1.0})
    try:
        trajectory = _trajectory(search, ('u', 'v'))
        np.random.seed(7)
        search.fit(data=[trajectory], max_deriv_order=(1,))
        systems = list(search.optimizer.reported_front()) + list(search.optimizer.pareto_levels.population)
        assert systems
        for system in systems:
            for equation in system.vals:
                assert all(len(term.structure) == 1 for term in equation.structure)
    finally:
        global_var.samples_manager.reset()
