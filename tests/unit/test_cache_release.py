"""Releasing the cached tensors when a search finishes.

``delete_cache`` was called by nothing, and would have done the wrong thing
twice if it had been: it ``del``-ed ``tensor_cache`` and ``grid_cache``
outright -- leaving the ~40 unguarded ``global_var.tensor_cache.…`` readers
raising ``AttributeError`` rather than degrading -- and it never touched
``samples_manager`` or ``initial_data_cache``, which is where the trajectory
tensors, the actual bulk, are held.

Two levels of release, because "finish" cannot simply mean "end of ``fit``":
``predict``, ``solver_forms``, ``visualize_solutions`` and the ``cache``
property all read the caches AFTER ``fit`` returns.

* end of ``fit`` -> the evaluated-term cache only (the bulk, and rebuildable),
  gated by ``runtime.free_tensor_cache_after_fit``;
* ``close()`` / the context-manager form -> everything.
"""

import numpy as np
import pytest

import epde
import epde.globals as global_var
from epde.interface.search_config import load_search_config

CACHES = ('tensor_cache', 'grid_cache', 'initial_data_cache')


def _entries(cache):
    if cache is None:
        return 0
    return sum(len(sub) for sub in cache.memory_default.values())


def _search(**kwargs):
    kwargs.setdefault('verbose_params', {'show_iter_idx': False})
    search = epde.EpdeSearch(**kwargs)
    search.set_preprocessor(default_preprocessor_type='FD',
                            preprocessor_kwargs={})
    return search


def _fitted(**kwargs):
    """A completed single-objective search -- the cheapest thing that fills
    every cache."""
    search = _search(multiobjective_mode=False, **kwargs)
    search.set_singleobjective_params(population_size=4, training_epochs=1)
    grid = np.linspace(0, 4 * np.pi, 100)
    data = np.sin(grid) + 1.3 * np.cos(grid)
    _, domain = search.createDomain(grid, boundary_width=10, ID=0)
    _, traj = search.createTrajectory({'u': data}, domain, cache_id=0)
    search.fit(data=[traj], max_deriv_order=(2,), data_fun_pow=1,
               equation_terms_max_number=3,
               equation_factors_max_number=1)
    return search


class TestDeleteCacheReleasesEverything:

    def test_every_name_stays_defined(self):
        """The old version ``del``-ed the names. Most readers do a plain
        attribute access, so a deleted name is an AttributeError at the next
        evaluation rather than a graceful empty cache."""
        _search()
        global_var.delete_cache()
        for name in CACHES:
            assert hasattr(global_var, name), name
        assert hasattr(global_var, 'samples_manager')

    def test_the_tensors_are_actually_dropped(self):
        search = _fitted()
        assert _entries(global_var.grid_cache) > 0
        search.close()
        for name in CACHES:
            assert _entries(getattr(global_var, name)) == 0, name

    def test_the_samples_go_too(self):
        """``samples_manager`` holds the trajectory tensors -- the bulk the old
        version left behind entirely."""
        search = _fitted()
        assert global_var.samples_manager._traj
        search.close()
        assert not global_var.samples_manager._traj

    def test_the_basis_cache_goes_too(self):
        global_var.vc_modes_cache['stale'] = object()
        global_var.delete_cache()
        assert not global_var.vc_modes_cache


class TestContextManager:

    def test_exit_releases(self):
        with _search() as search:
            assert search is not None
            grid = np.linspace(0, 4 * np.pi, 60)
            search.createDomain(grid, boundary_width=5, ID=0)
            assert _entries(global_var.grid_cache) > 0
        assert _entries(global_var.grid_cache) == 0

    def test_exit_releases_even_when_the_body_raises(self):
        with pytest.raises(RuntimeError):
            with _search() as search:
                grid = np.linspace(0, 4 * np.pi, 60)
                search.createDomain(grid, boundary_width=5, ID=0)
                raise RuntimeError('boom')
        assert _entries(global_var.grid_cache) == 0


class TestEndOfFitRelease:

    def test_the_tensor_cache_is_released(self):
        _fitted()
        assert _entries(global_var.tensor_cache) == 0

    def test_the_other_caches_survive(self):
        """This is the property that makes the end-of-fit release safe: the
        post-fit API reads them."""
        _fitted()
        assert _entries(global_var.grid_cache) > 0
        assert global_var.samples_manager._traj

    def test_the_post_fit_api_still_works(self):
        """The post-fit API reads the caches the end-of-fit release leaves
        alone -- that is what makes releasing the tensor cache there safe."""
        search = _fitted()
        assert search.equations(only_print=False, num=1)
        assert search.cache[1] is global_var.tensor_cache
        # the default grid predict() would hand the solver
        assert global_var.grid_cache.get_all()[1]
        # solver_forms reads samples_manager (grids) and rebuilds term tensors
        forms = search.solver_forms()
        assert forms and forms[0][0][0] == 'u'

    def test_it_can_be_switched_off(self):
        search = _fitted(free_tensor_cache_after_fit=False)
        assert search.config.runtime.free_tensor_cache_after_fit is False
        assert _entries(global_var.tensor_cache) > 0

    def test_the_default_is_on(self):
        assert load_search_config().runtime.free_tensor_cache_after_fit is True

    def test_releasing_keeps_the_memory_budget(self):
        """``release_tensor_cache`` empties the cache without replacing it, so
        the properties ``set_memory_properties`` configured survive."""
        _fitted()
        cache = global_var.tensor_cache
        budget = cache.max_allowed_tensors
        global_var.release_tensor_cache()
        assert global_var.tensor_cache is cache
        assert cache.max_allowed_tensors == budget
