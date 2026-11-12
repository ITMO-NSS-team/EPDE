"""A seeded plain-mode NSGA-II fit depends on the numpy seed alone.

* the equation-exchange coin of the chromosome crossover follows the numpy
  stream, not the standard library's;
* a second fit on one ``EpdeSearch`` equals the same fit on a fresh one (the
  per-variable selection's rotation counter lives on the fit's population);
* ``plain_fit.start_fit`` empties every cache plain mode keeps between calls
  (``release`` leaves the variable order's generator unset);
* ``_check_plain_tokens`` raises on a cached negative verdict as well.
"""
import random
from types import SimpleNamespace

import numpy as np
import pytest

import epde
import epde.globals as global_var
from epde.operators.common import fill_prune as FP
from epde.operators.common import plain_fit as PF
from epde.operators.common import product_rule as PR


def _search():
    search = epde.EpdeSearch(verbose_params={'show_iter_idx': False}, engine='nsga2', system_rps_mode='plain',
                             discrepancy_metric='wape', instability_metric='dyad',
                             equation_terms_max_number=4, equation_factors_max_number=1,
                             data_fun_pow=1, population_size=8, training_epochs=2)
    search.set_preprocessor(default_preprocessor_type='FD', preprocessor_kwargs={})
    return search


def _trajectory(search, names):
    t = np.linspace(0, 4 * np.pi, 300)
    funs = {'u': np.sin(t), 'v': np.cos(t), 'w': np.sin(2 * t)}
    domain = search.createDomain(t, ID=0)[1]
    return search.createTrajectory({name: funs[name] for name in names}, domain, cache_id=0)[1]


def _fit(search, trajectory, seed, py_seed=None):
    """Fit ``search`` from numpy seed ``seed`` (and, when given, a standard-library seed) and return what the
    fit reports: the reported front and the per-epoch history, as text and objective values."""
    np.random.seed(seed)
    if py_seed is not None:
        random.seed(py_seed)
    search.fit(data=[trajectory], max_deriv_order=(1,))
    front = [(system.text_form, [float(v) for v in system.obj_fun]) for system in search.optimizer.reported_front()]
    history = [[(entry['text_form'], [float(v) for v in entry['obj_fun']]) for entry in epoch]
               for epoch in search.pareto_history]
    return front, history


def test_the_exchange_coin_follows_the_numpy_seed():
    results = []
    for py_seed in (1, 2):
        search = _search()
        try:
            results.append(_fit(search, _trajectory(search, ('u', 'v')), seed=5, py_seed=py_seed))
        finally:
            global_var.samples_manager.reset()
    assert results[0][0] and len(results[0][1]) == 2
    assert results[0] == results[1]


def test_a_second_fit_equals_a_fresh_search():
    search = _search()
    try:
        trajectory = _trajectory(search, ('u', 'v', 'w'))
        first = _fit(search, trajectory, seed=3)
        again = _fit(search, trajectory, seed=3)
    finally:
        global_var.samples_manager.reset()
    fresh_search = _search()
    try:
        fresh = _fit(fresh_search, _trajectory(fresh_search, ('u', 'v', 'w')), seed=3)
    finally:
        global_var.samples_manager.reset()
    assert first[0] and len(first[1]) == 2
    assert first == fresh
    assert again == fresh


def test_start_fit_empties_the_caches(monkeypatch):
    for module, name, empty in ((PF, '_SIBLINGS', {}), (PF, '_PLAIN_TOKENS', {}), (FP, '_LIBRARY', {}),
                                (FP, '_STORES', {}), (FP, '_WARNED', set()), (PR, '_GROUPS', {})):
        monkeypatch.setattr(module, name, empty)
    monkeypatch.setitem(PF._ORDER, 'rng', None)
    PF._SIBLINGS['s'] = 1
    PF._PLAIN_TOKENS['t'] = 1
    FP._LIBRARY['l'] = 1
    FP._STORES['g'] = 1
    FP._WARNED.add('w')
    PR._GROUPS['p'] = 1
    PF.start_fit(3)
    assert not (PF._SIBLINGS or PF._PLAIN_TOKENS or FP._LIBRARY or FP._STORES or FP._WARNED or PR._GROUPS)
    assert isinstance(PF._ORDER['rng'], np.random.Generator)
    PF.release()
    assert PF._ORDER['rng'] is None


def test_a_cached_negative_verdict_raises_again(monkeypatch):
    monkeypatch.setattr(PF, '_PLAIN_TOKENS', {})
    monkeypatch.setattr(FP, 'library_terms', lambda pool, factors: [])
    class Pool:                         # the cache keeps a weak reference to the pool
        pass

    equation = SimpleNamespace(pool=Pool(), main_var_to_explain='u',
                               metaparameters={'max_factors_in_term': {'value': 1}})
    for _ in range(2):
        with pytest.raises(ValueError, match="plain derivative of 'u'"):
            PF._check_plain_tokens(equation)
