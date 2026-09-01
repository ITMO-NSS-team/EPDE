"""The solver path, after the multisample refactor went past it.

Both PDE backends were dead. The consumer half had been migrated -- the
solver options of ``Discrepancy`` (``'solver_l2'`` / ``'pic'`` / ``'deepxde'``)
already index ``sctx.solution`` and ``sctx.g_fun_vals`` by trajectory key --
while the producing half, ``SolverBasedFitness``, still spoke the
single-domain API:

* ``samples_manager.grids`` became a METHOD returning
  ``{trajectory: [grids]}``; three sites still subscripted it as a property,
  and ``DeepXDEAdapter.solve`` read ``len(grids)`` as the dimensionality --
  getting the trajectory count.
* ``grid_cache.g_func_mask`` / ``get_all(mode=...)`` do not exist on the
  post-refactor ``Cache``; the mask belongs to a trajectory now.
* ``samples_manager.getSingleSample`` never existed
  (``getSingleTrajectory``).
* ``Cache.getKeys`` read an undeclared ``subcache_ID`` -- ``UnboundLocalError``
  on every call.
* the net was sampled on the FULL grid and compared against inner-domain
  reference data; ``Domain.getGrids`` has always had a ``mode='solver'``
  (boundary trimmed) that the trajectory accessors never passed through.
* ``solution_guess_nn`` is normally undefined, and reading a missing MODULE
  attribute raises ``AttributeError``; the guard caught ``NameError``.
* the device was chosen by ``torch.cuda.is_available`` -- the function object,
  never called, so always truthy -- ignoring ``solver.device`` outright.

The end-to-end gate for both backends is ``tests/system/solver_backends.py``
(~45 s + ~95 s); these are the fast pins.
"""

import inspect
from pathlib import Path

import numpy as np
import pytest
import torch

import epde
import epde.globals as global_var
from conftest import using_config
from epde.cache.cache_refactored import Cache
from epde.integrate.deepxde_integration import DeepXDEAdapter
from epde.operators.common.fitness import SolverBasedFitness
from epde.structure.domain import TrajectoriesManager

EPDE_ROOT = Path(epde.__file__).parent


@pytest.fixture(scope='module')
def fitted():
    """A finished two-trajectory 1-D search, so the trajectory accessors have
    something to answer with."""
    search = epde.EpdeSearch(multiobjective_mode=False,
                             verbose_params={'show_iter_idx': False})
    search.set_preprocessor(default_preprocessor_type='FD',
                            preprocessor_kwargs={})
    search.set_singleobjective_params(population_size=4, training_epochs=1)
    grid = np.linspace(0, 4 * np.pi, 100)
    trajectories = []
    for idx in range(2):
        data = np.sin(grid) + (1.0 + idx) * np.cos(grid)
        _, domain = search.createDomain(grid, boundary_width=10, ID=idx)
        trajectories.append(
            search.createTrajectory({'u': data}, domain, cache_id=idx)[1])
    search.fit(data=trajectories, max_deriv_order=(2,), data_fun_pow=1,
               equation_terms_max_number=3, equation_factors_max_number=1)
    return search


def _scan(needles):
    """Offending source lines anywhere under ``epde/``.

    Prose is skipped by the reST inline-literal marker: every docstring here
    quotes identifiers that way, so a line carrying one is documentation
    about the old API rather than a use of it.

    Scans the whole package with no exemptions. ``fitness_refactored.py`` --
    a complete pre-multisample duplicate of these hosts, ~19 stale cache
    sites, imported by nothing -- used to need one; it has since been
    deleted.
    """
    found = []
    for path in EPDE_ROOT.rglob('*.py'):
        for lineno, line in enumerate(
                path.read_text(encoding='utf-8', errors='ignore').splitlines(), 1):
            code = line.split('#')[0]
            if '``' in code:
                continue
            for needle in needles:
                if needle in code:
                    found.append('%s:%s %s' % (path.name, lineno, code.strip()))
    return found


class TestTheCacheKeyAccessor:

    def test_getkeys_declares_the_subcache_it_reads(self):
        """It read ``subcache_ID`` without taking it -- ``UnboundLocalError``
        on the only call site (``TrajectoriesManager.grid_keys``)."""
        params = inspect.signature(Cache.getKeys).parameters
        assert 'subcache_ID' in params
        assert params['subcache_ID'].default is None

    def test_grid_keys_answers(self, fitted):
        keys = global_var.samples_manager.grid_keys
        assert keys and all(isinstance(key, str) for key in keys)


class TestSolverModeGrids:
    """``Domain.getGrids(mode='solver')`` trims the boundary; the trajectory
    accessors never forwarded the argument, so the solver got full grids."""

    def test_the_accessors_take_the_mode(self):
        for owner in (TrajectoriesManager,):
            params = inspect.signature(owner.grids).parameters
            assert 'mode' in params
            assert params['mode'].default == 'full'

    def test_the_default_is_still_the_full_grid(self, fitted):
        samples = global_var.samples_manager
        for key, grids in samples.grids().items():
            assert np.asarray(grids[0]).size == 100

    def test_solver_mode_is_the_inner_domain(self, fitted):
        samples = global_var.samples_manager
        inner = samples.inner_shapes
        for key, grids in samples.grids(mode='solver').items():
            assert np.asarray(grids[0]).size == int(np.prod(inner[key]))

    def test_it_agrees_with_the_mask_and_with_evaluate(self, fitted):
        """The three things the solver compares: grid, weighting, and the
        reference data ``Equation.evaluate`` produces."""
        samples = global_var.samples_manager
        for key in samples.trajecatoryIDs:
            n_solver = np.asarray(samples.grids(mode='solver')[key]).size
            assert int(np.asarray(samples.gFunc('m')[key]).sum()) == n_solver
            assert np.asarray(samples.gFunc('dmf')[key]).size == n_solver


class TestTheTrajectoryAccessors:

    def test_the_single_sample_accessor_is_spelled_for_trajectories(self):
        assert not hasattr(TrajectoriesManager, 'getSingleSample')
        assert hasattr(TrajectoriesManager, 'getSingleTrajectory')

    def test_nothing_subscripts_grids_as_a_property(self):
        assert not _scan(['samples_manager.grids['])

    def test_nothing_reads_the_removed_cache_attributes(self):
        assert not _scan(['grid_cache.g_func_mask', 'grid_cache.get_all(mode',
                          'samples_manager.getSingleSample'])


class TestThePretrainedNetGuard:

    def test_a_missing_global_yields_none(self):
        """``solution_guess_nn`` is written only by ``reset_data_repr_nn``,
        which has no live caller -- so normally the name does not exist."""
        assert not hasattr(global_var, 'solution_guess_nn')
        assert SolverBasedFitness._pretrained_net() is None

    def test_no_guard_still_waits_for_a_nameerror(self):
        """A missing MODULE attribute is an ``AttributeError``; ``NameError``
        is what a missing LOCAL raises, so the old guard never fired."""
        source = inspect.getsource(SolverBasedFitness)
        assert 'except NameError' not in source


class TestTheDeviceChoice:

    def test_is_available_is_actually_called(self):
        """Read as a bare function object it is always truthy, so this said
        'cuda' on a CPU-only machine and the solve died on a device
        mismatch."""
        code = [line for line in inspect.getsource(SolverBasedFitness).splitlines()
                if '``' not in line and not line.lstrip().startswith('#')]
        assert any('torch.cuda.is_available()' in line for line in code)
        assert not any('explicit_cpu' in line for line in code)

    def test_the_configured_device_is_honoured(self):
        host = SolverBasedFitness(param_keys=[])
        with using_config(device='cpu'):
            host.set_adapter(net=None)
        assert host.adapter._device == 'cpu'

    @pytest.mark.skipif(torch.cuda.is_available(), reason='CUDA present')
    def test_an_unavailable_cuda_falls_back_and_says_so(self):
        host = SolverBasedFitness(param_keys=[])
        with using_config(device='cuda'):
            with pytest.warns(global_var.EPDEUsageWarning, match='cpu'):
                host.set_adapter(net=None)
        assert host.adapter._device == 'cpu'

    def test_the_host_remembers_the_device_for_its_grids(self):
        """``_apply_autograd`` builds the grid stack itself and has to put it
        where SolverAdapter.solve puts the net; the original never moved the
        grids at all, so a cuda run died in the first matmul."""
        host = SolverBasedFitness(param_keys=[])
        with using_config(device='cpu'):
            host.set_adapter(net=None)
        assert host.solver_device == host.adapter._device
        assert '.to(self.solver_device)' in inspect.getsource(
            SolverBasedFitness._apply_autograd)


class TestTheDeepXDEAdapter:

    def test_solve_takes_the_domain_it_solves(self):
        params = inspect.signature(DeepXDEAdapter.solve).parameters
        assert 'domain_key' in params
        assert params['domain_key'].default is None

    def test_the_mask_comes_from_the_trajectory(self, fitted):
        adapter = DeepXDEAdapter()
        for key in global_var.samples_manager.trajecatoryIDs:
            adapter.domain_key = key
            expected = np.asarray(global_var.samples_manager.gFunc('m')[key])
            np.testing.assert_array_equal(adapter.domain_mask, expected)

    def test_the_mask_defaults_to_the_first_trajectory(self, fitted):
        adapter = DeepXDEAdapter()
        assert adapter.domain_key is None
        first = global_var.samples_manager.trajecatoryIDs[0]
        np.testing.assert_array_equal(
            adapter.domain_mask,
            np.asarray(global_var.samples_manager.gFunc('m')[first]))

    def test_the_mask_is_grid_shaped(self, fitted):
        """The strategies index the GRIDS with it (``g[mask]``), and those
        keep their grid shape -- a flattened mask breaks every 2-D solve."""
        adapter = DeepXDEAdapter()
        samples = global_var.samples_manager
        first = samples.trajecatoryIDs[0]
        assert adapter.domain_mask.shape == \
            np.asarray(samples.grids()[first][0]).shape

    def test_an_unsupported_dimensionality_fails_loudly(self, fitted):
        """``len(grids)`` used to count trajectories, so ``_solvers.get`` could
        return None and the next line raised ``AttributeError: 'NoneType'``."""
        adapter = DeepXDEAdapter()
        with pytest.raises(NotImplementedError, match='1-D to 3-D'):
            adapter.solve(equation_or_system=None,
                          grids=[np.zeros(3)] * 4, data=[])


class TestTheHostsProducePerTrajectoryProducts:
    """``Discrepancy``'s solver options were migrated to per-trajectory dicts
    ahead of the hosts; these pin that the hosts now build what they read."""

    @pytest.mark.parametrize('name', ['_apply_autograd', '_apply_deepxde'])
    def test_the_solve_loops_over_trajectories(self, name):
        source = inspect.getsource(getattr(SolverBasedFitness, name))
        assert 'trajecatoryIDs' in source
        assert 'SolverContext(' in source

    def test_the_deepxde_branch_unpacks_evaluate_correctly(self):
        source = inspect.getsource(SolverBasedFitness._apply_deepxde)
        assert 'target.reshape(-1)' not in source
        assert 'active_only=True' in source
