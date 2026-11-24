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
from epde.operators.common.fitness import SolverBasedFitness
from epde.structure.domain import TrajectoriesManager

EPDE_ROOT = Path(epde.__file__).parent


@pytest.fixture(scope='module', autouse=True)
def _restore_torch_default_device():
    """``import deepxde`` switches torch's PROCESS-WIDE default device to cuda
    when a GPU is present, so every later test that builds a tensor and calls
    ``.numpy()`` failed once this module had run. DeepXDE is imported lazily
    (``_deepxde_adapter``) so this fixture sees the device from before the
    import, and puts it back when the module is done."""
    before = torch.get_default_device()
    yield
    torch.set_default_device(before)


def _deepxde_adapter():
    from epde.integrate.deepxde_integration import DeepXDEAdapter
    return DeepXDEAdapter


def _finished_search(preprocessor_kwargs=None):
    """A finished two-trajectory 1-D search, so the trajectory accessors have
    something to answer with. The tensor cache is kept after fit: the DeepXDE
    tests translate and fit an equation afterwards, and the post-fit release
    empties the term cache (``Cache.get`` then raises KeyError).

    ``preprocessor_kwargs`` go to the finite-difference preprocessor; None
    takes its defaults. ``test_lr_free_solve`` passes the difference order its
    bitwise pins were captured under."""
    search = epde.EpdeSearch(multiobjective_mode=False,
                             free_tensor_cache_after_fit=False,
                             verbose_params={'show_iter_idx': False})
    search.set_preprocessor(default_preprocessor_type='FD',
                            preprocessor_kwargs=dict(preprocessor_kwargs or {}))
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


@pytest.fixture(scope='module')
def fitted():
    """The finished search on the preprocessor's default settings."""
    return _finished_search()


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
        params = inspect.signature(_deepxde_adapter().solve).parameters
        assert 'domain_key' in params
        assert params['domain_key'].default is None

    def test_the_mask_comes_from_the_trajectory(self, fitted):
        adapter = _deepxde_adapter()()
        for key in global_var.samples_manager.trajecatoryIDs:
            adapter.domain_key = key
            expected = np.asarray(global_var.samples_manager.gFunc('m')[key])
            np.testing.assert_array_equal(adapter.domain_mask, expected)

    def test_the_mask_defaults_to_the_first_trajectory(self, fitted):
        adapter = _deepxde_adapter()()
        assert adapter.domain_key is None
        first = global_var.samples_manager.trajecatoryIDs[0]
        np.testing.assert_array_equal(
            adapter.domain_mask,
            np.asarray(global_var.samples_manager.gFunc('m')[first]))

    def test_the_mask_is_grid_shaped(self, fitted):
        """The strategies index the GRIDS with it (``g[mask]``), and those
        keep their grid shape -- a flattened mask breaks every 2-D solve."""
        adapter = _deepxde_adapter()()
        samples = global_var.samples_manager
        first = samples.trajecatoryIDs[0]
        assert adapter.domain_mask.shape == \
            np.asarray(samples.grids()[first][0]).shape

    def test_an_unsupported_dimensionality_fails_loudly(self, fitted):
        """``len(grids)`` used to count trajectories, so ``_solvers.get`` could
        return None and the next line raised ``AttributeError: 'NoneType'``."""
        adapter = _deepxde_adapter()()
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

    def test_the_deepxde_branch_feeds_the_observed_field(self):
        """The data handed to the solver used to be the equation's TARGET
        term (``evaluate()[0]``, e.g. du/dt), which the solver treated as
        ``u``. It has to be the observed variable."""
        code = [line for line in
                inspect.getsource(SolverBasedFitness._apply_deepxde).splitlines()
                if not line.lstrip().startswith('#') and '``' not in line]
        assert not any('evaluate(' in line for line in code)
        assert any('samples.get((eq.main_var_to_explain, (1.0,)))' in line
                   for line in code)
        assert any('time_split(' in line for line in code)


def _fitted_oscillator(search):
    """``u'' = -u`` translated on the fixture's pool and fitted -- the true
    law of the fixture data (``sin + k cos``)."""
    from epde.interface.equation_translator import translate_equation
    from epde.operators.common.coeff_calculation import LinRegBasedCoeffsEquation
    from epde.operators.common.sparsity import build_sparsity_operator
    from epde.interface.search_config import active_config

    system = translate_equation('-1.0 * u{power: 1.0} + 0.0 = d^2u/dx0^2{power: 1.0}',
                                search.pool, all_vars=['u'])
    system.vals['u'].main_var_to_explain = 'u'
    system.use_default_singleobjective_function()
    cfg = active_config().objectives
    sparsity = build_sparsity_operator(cfg.sparsity_cls, cfg.sparsity_kwargs)
    coeffs = LinRegBasedCoeffsEquation()
    for equation in system.vals:
        sparsity.apply(equation, {})
        coeffs.apply(equation, {})
    return system


class TestTheResidualFactors:
    """``_factor_value_with_map`` builds every factor of the PINN residual."""

    @staticmethod
    def _factor(power, deriv_code):
        from types import SimpleNamespace
        return SimpleNamespace(is_deriv=True, deriv_code=deriv_code, variable='u',
                               params=np.array([power]),
                               params_description={0: {'name': 'power', 'bounds': (1, 3)}},
                               structure=[], name='u', cache_label=None)

    def test_a_powered_variable_keeps_its_power(self):
        """A plain variable token carries ``is_deriv=True, deriv_code=[None]``.
        The truthiness test sent it down the derivative branch, which dropped
        the power -- ``5u - 5u^3`` cancelled and Allen-Cahn became the heat
        equation inside the PINN."""
        adapter = _deepxde_adapter()()
        y = torch.tensor([[2.0], [-1.5], [0.5]])
        x = torch.zeros(3, 2)
        value = adapter._factor_value_with_map(None, self._factor(3.0, [None]), x, y,
                                               coord_map={}, var_idx_map={'u': 0})
        assert torch.allclose(value, y ** 3)

    def test_the_power_is_read_by_name(self):
        adapter = _deepxde_adapter()
        assert adapter._factor_power(self._factor(2.0, [None])) == 2.0
        from types import SimpleNamespace
        assert adapter._factor_power(SimpleNamespace()) == 1.0

    @staticmethod
    def _token(family, label, **params):
        """A grid/trig/const factor as EPDE builds it: ``variable`` holds the
        FAMILY name, parameters are named in ``params_description``."""
        from types import SimpleNamespace
        names = list(params)
        return SimpleNamespace(ftype=family, label=label, variable=family,
                               is_deriv=False, deriv_code=None,
                               params=np.array([params[n] for n in names]),
                               params_description={i: {'name': n} for i, n in enumerate(names)})

    @staticmethod
    def _adapter_2d():
        """EPDE axes '0' (t) and '1' (x); DeepXDE input columns [x, t]."""
        adapter = _deepxde_adapter()()
        adapter._set_coordinate_info(['0', '1'])
        return adapter

    def _xy(self):
        x = torch.tensor([[0.5, 0.1], [-1.0, 0.7], [2.0, 0.3]])     # columns [x, t]
        y = torch.tensor([[1.0], [2.0], [3.0]])
        return x, y

    def test_a_grid_token_is_its_axis_to_the_power(self):
        """Grid and trig factors carry their family name in ``variable``, so
        the old ``variable is not None`` test evaluated them as ``u**power``."""
        adapter, (x, y) = self._adapter_2d(), self._xy()
        space = self._token('grids', 'x_0', power=2.0, dim=1.0)     # EPDE axis 1 = x
        time = self._token('grids', 'x_1', power=1.0, dim=0.0)      # EPDE axis 0 = t
        got_x = adapter._factor_value_with_map(None, space, x, y, adapter.coord_map, {'u': 0})
        got_t = adapter._factor_value_with_map(None, time, x, y, adapter.coord_map, {'u': 0})
        assert torch.allclose(got_x, x[:, 0:1] ** 2)
        assert torch.allclose(got_t, x[:, 1:2])

    def test_a_trig_token_uses_its_frequency_and_axis(self):
        import deepxde as dde
        adapter, (x, y) = self._adapter_2d(), self._xy()
        cos_t = self._token('trigonometric', 'cos', power=1.0, freq=2.0, dim=0.0)
        sin_x = self._token('trigonometric', 'sin', power=2.0, freq=0.5, dim=1.0)
        got_cos = adapter._factor_value_with_map(dde, cos_t, x, y, adapter.coord_map, {'u': 0})
        got_sin = adapter._factor_value_with_map(dde, sin_x, x, y, adapter.coord_map, {'u': 0})
        assert torch.allclose(got_cos, torch.cos(2.0 * x[:, 1:2]))
        assert torch.allclose(got_sin, torch.sin(0.5 * x[:, 0:1]) ** 2)

    def test_a_constant_token_is_its_value(self):
        adapter, (x, y) = self._adapter_2d(), self._xy()
        const = self._token('constants', 'const', power=1.0, value=3.5)
        got = adapter._factor_value_with_map(None, const, x, y, adapter.coord_map, {'u': 0})
        assert torch.allclose(got, torch.full_like(y, 3.5))

    def test_an_inexpressible_factor_fails_loudly(self):
        """It used to become ``u`` (or 1.0) silently."""
        adapter, (x, y) = self._adapter_2d(), self._xy()
        custom = self._token('custom family', 'thing', power=1.0)
        with pytest.raises(NotImplementedError, match='custom family'):
            adapter._factor_value_with_map(None, custom, x, y, adapter.coord_map, {'u': 0})
        stranger = self._factor(1.0, [None])
        stranger.variable = 'v'
        with pytest.raises(NotImplementedError, match="'v'"):
            adapter._factor_value_with_map(None, stranger, x, y, adapter.coord_map, {'u': 0})


class TestTheTimeSplit:
    """One rule decides what the observation term sees and where the
    candidate is scored."""

    def test_partitions_by_time_level(self):
        from epde.integrate.deepxde_integration import time_split
        t = np.repeat(np.arange(10.0), 3)            # 10 levels, 3 points each
        split = time_split(t, 0.8)
        assert split.t_fit == 7.0                    # ceil(0.8 * 10) = 8 levels
        assert split.train.sum() == 8 * 3 and split.test.sum() == 2 * 3
        assert t[split.train].max() < t[split.test].min()

    def test_rounds_the_training_share_up(self):
        from epde.integrate.deepxde_integration import time_split
        split = time_split(np.arange(7.0), 0.5)      # ceil(3.5) = 4
        assert split.t_fit == 3.0 and split.train.sum() == 4

    def test_no_validation_block_by_default(self):
        from epde.integrate.deepxde_integration import time_split
        split = time_split(np.arange(10.0), 0.8)
        assert not split.val.any()
        assert split.n_levels == (8, 0, 2)
        # The two-way contract: with no val block, test is exactly ~train.
        assert np.array_equal(split.test, ~split.train)

    def test_the_validation_block_is_carved_out_of_training(self):
        """The scored block must not move when a val block is asked for --
        otherwise no number measured under the two-way split is comparable."""
        from epde.integrate.deepxde_integration import time_split
        t = np.arange(40.0)
        base = time_split(t, 0.8)
        with_val = time_split(t, 0.8, 0.1)
        assert np.array_equal(base.test, with_val.test)    # the invariant
        assert with_val.n_levels == (28, 4, 8)
        assert with_val.train.sum() + with_val.val.sum() == base.train.sum()
        # train < val < test, contiguous and disjoint.
        assert t[with_val.train].max() < t[with_val.val].min()
        assert t[with_val.val].max() < t[with_val.test].min()
        assert not (with_val.train & with_val.val).any()
        assert (with_val.train | with_val.val | with_val.test).all()

    def test_rejects_a_validation_block_that_starves_training(self):
        from epde.integrate.deepxde_integration import (
            DeepXDEConfigError, time_split)
        with pytest.raises(DeepXDEConfigError, match='training time level'):
            time_split(np.arange(10.0), 0.8, 0.8)

    @pytest.mark.parametrize('frac', [1.0, -0.2, 1.5])
    def test_rejects_a_validation_fraction_outside_its_range(self, frac):
        from epde.integrate.deepxde_integration import (
            DeepXDEConfigError, time_split)
        with pytest.raises(DeepXDEConfigError, match='val_frac'):
            time_split(np.arange(10.0), 0.8, frac)

    def test_warns_when_the_validation_block_is_too_short_to_extrapolate(self):
        from epde.globals import EPDEUsageWarning
        from epde.integrate.deepxde_integration import time_split
        with pytest.warns(EPDEUsageWarning, match='continuity'):
            time_split(np.arange(40.0), 0.8, 0.02)     # ceil(0.8) = 1 level

    @pytest.mark.parametrize('frac', [0.0, 1.0, -0.2, 1.5])
    def test_rejects_a_fraction_outside_the_open_interval(self, frac):
        from epde.integrate.deepxde_integration import time_split
        with pytest.raises(ValueError, match='train_frac'):
            time_split(np.arange(10.0), frac)

    def test_rejects_a_split_that_holds_nothing_out(self):
        from epde.integrate.deepxde_integration import time_split
        with pytest.raises(ValueError, match='no held-out'):
            time_split(np.arange(3.0), 0.9)            # ceil(2.7) = 3 of 3

    def test_the_adapter_takes_a_learning_rate_schedule(self):
        adapter = _deepxde_adapter()(decay=['cosine', 1000, 1e-5])
        assert adapter.decay == ('cosine', 1000, 1e-5)
        assert _deepxde_adapter()().decay is None

    def test_the_adapter_validates_its_fraction(self):
        with pytest.raises(ValueError, match='train_frac'):
            _deepxde_adapter()(train_frac=1.0)
        assert _deepxde_adapter()().train_frac == 0.8


def _tiny_deepxde_model(seed=0):
    """A minimal 2-term DeepXDE problem (one PDE residual + one PointSetBC)."""
    import deepxde as dde
    dde.config.set_random_seed(seed)
    gt = dde.geometry.GeometryXTime(dde.geometry.Interval(-1, 1),
                                    dde.geometry.TimeDomain(0, 1))

    def pde(x, y):
        return [dde.grad.jacobian(y, x, i=0, j=1)
                - 0.1 * dde.grad.hessian(y, x, i=0, j=0)]

    pts = np.stack([np.linspace(-1, 1, 20), np.zeros(20)], axis=1)
    bc = dde.icbc.PointSetBC(pts, np.sin(np.pi * pts[:, 0:1]), component=0)
    data = dde.data.TimePDE(gt, pde, [bc], num_domain=64, num_boundary=0,
                            num_initial=0, num_test=16)
    return dde.Model(data, dde.nn.FNN([2, 8, 8, 1], 'tanh', 'Glorot normal'))


class TestTheGradientWeightGuards:
    """``normalised_grad_weights`` / ``_grad_norms`` -- the numerical guards the
    per-step mode needs, since a satisfied term's gradient goes to zero."""

    def _helpers(self):
        from epde.integrate import deepxde_integration as dxi
        return dxi

    def test_weights_are_inverse_norms_rescaled_to_mean_one(self):
        dxi = self._helpers()
        w = dxi.normalised_grad_weights([1.0, 2.0, 4.0])
        assert float(np.mean(w)) == pytest.approx(1.0)
        ratios = np.array(w) * np.array([1.0, 2.0, 4.0])
        assert ratios.max() - ratios.min() < 1e-9      # w_i proportional to 1/g_i

    def test_the_floor_is_relative_not_absolute(self):
        """An ABSOLUTE floor (finfo.tiny) turns a vanishing gradient into
        1/tiny = 4.5e307, which overflows to inf in the float32 loss and is not
        caught by any isnan gate."""
        dxi = self._helpers()
        w = dxi.normalised_grad_weights([1.0, 1e-30], floor_rel=1e-3)
        assert np.isfinite(w).all()
        assert w[1] / w[0] == pytest.approx(1e3, rel=1e-6)

    def test_unusable_vectors_are_rejected_not_installed(self):
        dxi = self._helpers()
        assert dxi.normalised_grad_weights([0.0, 0.0]) is None
        assert dxi.normalised_grad_weights([1.0, float('inf')]) is None
        assert dxi.normalised_grad_weights([]) is None

    def test_a_detached_term_gives_zero_not_a_typeerror(self):
        """``sum()`` over an all-None gradient tuple returns int 0, and
        ``torch.sqrt(0)`` raises TypeError -- swallowed by the solve's blanket
        except and reported as a NaN solve."""
        dxi = self._helpers()
        used = torch.nn.Parameter(torch.ones(3))
        unused = torch.nn.Parameter(torch.ones(3))
        # a real loss whose graph simply does not reach ``unused``
        loss = (used ** 2).sum()
        norms = dxi._grad_norms([loss], [unused])
        assert norms == [0.0]
        assert dxi._grad_norms([loss], [used, unused])[0] > 0.0

    def test_the_adapter_validates_the_mode_and_the_ema(self):
        with pytest.raises(ValueError, match='loss_weight_mode'):
            _deepxde_adapter()(loss_weight_mode='gradnorm')
        with pytest.raises(ValueError, match='grad_weight_ema'):
            _deepxde_adapter()(loss_weight_mode='gradient_per_step', grad_weight_ema=1.0)
        adapter = _deepxde_adapter()(loss_weight_mode='gradient_per_step',
                                     grad_weight_period=25)
        assert adapter.grad_weight_period == 25 and adapter.grad_weight_ema == 0.9


class TestThePerStepGradientStep:
    """The dynamic-weight step must be a drop-in for DeepXDE's own."""

    def test_at_fixed_weights_it_reproduces_the_stock_update(self):
        """It assembles ``p.grad = sum_i w_i g_i`` from per-term gradients
        instead of backwarding the weighted sum; same update, no extra
        forward."""
        from epde.integrate.deepxde_integration import _PerStepGradNormStep
        weights, iters = [3.0, 0.5], 5

        stock = _tiny_deepxde_model(0)
        stock.compile('adam', lr=1e-3, loss_weights=weights)
        stock.train(iterations=iters, display_every=iters)

        ours = _tiny_deepxde_model(0)
        ours.compile('adam', lr=1e-3)
        _PerStepGradNormStep(ours, weights, ema=1.0, period=1).install()  # ema=1 freezes
        ours.train(iterations=iters, display_every=iters)

        for a, b in zip(stock.net.parameters(), ours.net.parameters()):
            np.testing.assert_allclose(a.detach().cpu().numpy(),
                                       b.detach().cpu().numpy(), atol=1e-6)

    def test_it_keeps_the_learning_rate_schedule_running(self):
        """A train_step override that forgets ``lr_scheduler.step()`` silently
        disables ``decay``."""
        from epde.integrate.deepxde_integration import _PerStepGradNormStep
        model = _tiny_deepxde_model(0)
        model.compile('adam', lr=1e-3, decay=('step', 5, 0.5))
        _PerStepGradNormStep(model, [1.0, 1.0], ema=1.0, period=1).install()
        model.train(iterations=10, display_every=10)
        assert model.opt.param_groups[0]['lr'] == pytest.approx(2.5e-4)

    def test_the_weights_move_and_throttling_refreshes_less_often(self):
        from epde.integrate.deepxde_integration import _PerStepGradNormStep
        model = _tiny_deepxde_model(0)
        model.compile('adam', lr=1e-3)
        step = _PerStepGradNormStep(model, [3.0, 0.5], ema=0.9, period=1).install()
        model.train(iterations=10, display_every=10)
        assert step.refreshes == 10 and step.w != [3.0, 0.5]

        throttled = _tiny_deepxde_model(0)
        throttled.compile('adam', lr=1e-3)
        step5 = _PerStepGradNormStep(throttled, [3.0, 0.5], ema=0.9, period=5).install()
        throttled.train(iterations=10, display_every=10)
        assert step5.refreshes == 2

    def test_deepxde_does_not_weight_twice(self):
        """``install`` clears ``model.loss_weights``: the step applies the
        weights itself, and leaving DeepXDE's own weighting on would square
        them (and make losshistory a moving yardstick)."""
        from epde.integrate.deepxde_integration import _PerStepGradNormStep
        model = _tiny_deepxde_model(0)
        model.compile('adam', lr=1e-3, loss_weights=[2.0, 3.0])
        _PerStepGradNormStep(model, [2.0, 3.0]).install()
        assert model.loss_weights is None


class TestTheLBFGSBudget:

    def test_the_configured_cap_reaches_deepxde(self):
        """``set_LBFGS_options(maxiter=...)`` alone leaves ``iter_per_step`` at
        its import-time 1000, so torch's LBFGS is built with max_iter=1000 and
        a smaller cap overshoots (measured: 37 iterations for maxiter=20)."""
        from deepxde.optimizers import config as dde_opt_config
        from epde.integrate.deepxde_integration import DeepXDEAdapter
        before = dict(dde_opt_config.LBFGS_options)
        try:
            DeepXDEAdapter._set_lbfgs_budget(20)
            assert dde_opt_config.LBFGS_options['maxiter'] == 20
            assert dde_opt_config.LBFGS_options['iter_per_step'] == 20
            DeepXDEAdapter._set_lbfgs_budget(5000)
            assert dde_opt_config.LBFGS_options['iter_per_step'] == 1000
        finally:
            dde_opt_config.LBFGS_options.update(before)

    def test_the_budget_does_not_depend_on_call_order(self):
        """One solve must not inherit the previous one's L-BFGS budget.

        ``set_LBFGS_options`` takes every option as a keyword with a default and
        writes all six unconditionally, so naming only ``maxiter`` restores the
        other five rather than leaving them as the last call set them. That is
        what makes the option state a pure function of the arguments -- but only
        while every knob is passed on EVERY call. A knob added conditionally
        (``if x is not None: ... set(x)``) would silently reintroduce the
        dependence, which is what this pins."""
        from deepxde.optimizers import config as dde_opt_config
        from epde.integrate.deepxde_integration import DeepXDEAdapter
        before = dict(dde_opt_config.LBFGS_options)
        try:
            DeepXDEAdapter._set_lbfgs_budget(2000, None)
            fresh = dict(dde_opt_config.LBFGS_options)
            # ... now take a detour through a different budget AND a gtol ...
            DeepXDEAdapter._set_lbfgs_budget(37, 1e-3)
            assert dde_opt_config.LBFGS_options['gtol'] == 1e-3
            DeepXDEAdapter._set_lbfgs_budget(2000, None)
            assert dict(dde_opt_config.LBFGS_options) == fresh
        finally:
            dde_opt_config.LBFGS_options.update(before)


class TestTheAdapterTracksItsInputs:
    """``SolverBasedFitness.set_adapter`` must not freeze the first call's
    ``pretrained_net`` / ``deepxde_config`` for the whole run.

    Both are inert today -- ``deepxde_config`` is a per-search operator
    parameter and ``solution_guess_nn`` has no live writer -- so these pin the
    contract rather than a live symptom."""

    @staticmethod
    def _host(cfg):
        from epde.operators.common.fitness import SolverBasedFitness
        host = SolverBasedFitness.__new__(SolverBasedFitness)
        host.backend = 'deepxde'
        host.adapter = None
        host.param_keys = ['deepxde_config']
        host.params = {'deepxde_config': cfg}
        return host

    def test_the_adapter_is_reused_when_nothing_changed(self):
        host = self._host({'epochs': 5})
        host.set_adapter()
        first = host.adapter
        host.set_adapter()
        assert host.adapter is first

    def test_a_changed_config_rebuilds_the_adapter(self):
        host = self._host({'epochs': 5})
        host.set_adapter()
        first = host.adapter
        assert first.epochs == 5
        host.params['deepxde_config'] = {'epochs': 9}
        host.set_adapter()
        assert host.adapter is not first
        assert host.adapter.epochs == 9

    def test_a_supplied_net_rebuilds_the_adapter(self):
        import torch
        host = self._host({'epochs': 5})
        host.set_adapter()
        first = host.adapter
        assert first.pretrained_net is None
        net = torch.nn.Linear(1, 1)
        host.set_adapter(pretrained_net=net)
        assert host.adapter is not first
        assert host.adapter.pretrained_net is net


class TestTheTorchDefaultDevice:
    """``import deepxde`` calls ``torch.set_default_device('cuda')`` whenever a
    GPU is visible and never puts it back -- process-wide, permanent, and it
    silently overrode ``solver.device='cpu'``. The adapter module must contain
    that: the GPU belongs to a solve, not to the caller's process."""

    def test_importing_the_adapter_does_not_move_the_default_device(self):
        """An import side effect can only be tested from a fresh process."""
        import subprocess
        import sys
        code = (
            'import torch;'
            'before = torch.get_default_device();'
            'import epde.integrate.deepxde_integration as dxi;'
            'after = torch.get_default_device();'
            'print(before, after, dxi.DDE_SOLVE_DEVICE)'
        )
        out = subprocess.run([sys.executable, '-c', code], capture_output=True,
                             text=True, timeout=600)
        assert out.returncode == 0, out.stderr[-2000:]
        before, after, solve_device = out.stdout.strip().splitlines()[-1].split()
        assert before == after, (
            f'importing the adapter moved the default device {before} -> {after}')
        # On a GPU machine the module still REMEMBERS deepxde's choice, so the
        # containment is a restore rather than a refusal; on a CPU-only one
        # there was nothing to contain and it records None.
        assert solve_device != before or solve_device == 'None'

    def test_the_solve_device_is_entered_and_given_back(self):
        import torch
        from epde.integrate.deepxde_integration import (DDE_SOLVE_DEVICE,
                                                        _solve_device)
        outside = torch.get_default_device()
        with _solve_device():
            inside = torch.get_default_device()
        assert torch.get_default_device() == outside
        if DDE_SOLVE_DEVICE is None:
            assert inside == outside          # CPU-only machine: a no-op
        else:
            assert inside == DDE_SOLVE_DEVICE

    def test_the_public_solve_is_wrapped(self):
        """The context manager is worth nothing if the entry point forgets it."""
        from epde.integrate.deepxde_integration import DeepXDEAdapter
        assert getattr(DeepXDEAdapter.solve, '__wrapped__', None) is not None


@pytest.fixture
def lbfgs_on_cpu():
    """The tiny L-BFGS models run on CPU with DeepXDE's global L-BFGS options
    restored afterwards. These tests compare separately built models bit for
    bit, which CPU guarantees across machines; importing deepxde otherwise
    makes CUDA the default device wherever one exists."""
    import torch
    from deepxde.optimizers import config as dde_opt_config
    before = dict(dde_opt_config.LBFGS_options)
    device = torch.get_default_device()
    torch.set_default_device('cpu')
    try:
        yield
    finally:
        torch.set_default_device(device)
        dde_opt_config.LBFGS_options.update(before)


def _torch_n_iter(model):
    return model.opt.state_dict()['state'][0]['n_iter']


def _train_guarded_lbfgs(model, budget, gtol=None, loss_weights=None,
                         max_steps=50):
    """Compile ``model`` for L-BFGS under ``budget``, install the stall guard
    and train. ``model.step_n_iters`` records torch's ``n_iter`` after every
    ``step`` call; past ``max_steps`` calls the run raises, so a broken guard
    fails the test instead of hanging the suite."""
    from epde.integrate.deepxde_integration import (
        DeepXDEAdapter, _install_lbfgs_stall_guard)
    DeepXDEAdapter._set_lbfgs_budget(budget, gtol)
    model.compile('L-BFGS', loss_weights=loss_weights)
    _install_lbfgs_stall_guard(model)
    stock_step = model.opt.step
    model.step_n_iters = []

    def counted(closure=None, *args, **kwargs):
        if len(model.step_n_iters) >= max_steps:
            raise RuntimeError(f'L-BFGS loop still running after {max_steps} steps')
        out = stock_step(closure, *args, **kwargs)
        model.step_n_iters.append(_torch_n_iter(model))
        return out

    model.opt.step = counted
    model.train(verbose=0)
    return model


@pytest.mark.usefixtures('lbfgs_on_cpu')
class TestTheLBFGSStallGuard:
    """DeepXDE's L-BFGS loop cannot terminate when torch converges AT ENTRY:
    ``step`` returns without incrementing ``n_iter``, so the
    ``prev_n_iter == n_iter - 1`` test reads ``prev == prev - 1`` (False), the
    step counter advances by 0 and ``while prev_n_iter < maxiter`` spins
    forever. ``lbfgs_gtol_rel`` is a knob on precisely that condition."""

    def test_convergence_at_entry_terminates_instead_of_hanging(self):
        # gtol this large makes max|grad| <= tolerance_grad true before the
        # first iteration -- the condition that hangs the stock loop.
        model = _train_guarded_lbfgs(_tiny_deepxde_model(), 200, gtol=1e10)
        # torch returns without iterating only when the gradient test holds.
        assert model.step_n_iters == [0]
        assert model.lbfgs_exit == {'reason': 'gtol', 'iterations': 0}

    def test_a_normal_run_is_untouched_and_records_why_it_stopped(self):
        from epde.integrate.deepxde_integration import DeepXDEAdapter
        probe = np.stack([np.linspace(-1, 1, 16), np.full(16, 0.5)], axis=1)
        stock = _tiny_deepxde_model(1)
        DeepXDEAdapter._set_lbfgs_budget(200)
        stock.compile('L-BFGS')
        stock.train(verbose=0)                  # DeepXDE's own loop

        guarded = _train_guarded_lbfgs(_tiny_deepxde_model(1), 200)
        # Bit-identical: the guard adds a break that a healthy run never takes.
        assert np.array_equal(stock.predict(probe), guarded.predict(probe))
        assert guarded.lbfgs_exit['reason'] in ('converged', 'maxiter')
        assert guarded.lbfgs_exit['iterations'] == _torch_n_iter(guarded) > 0


@pytest.mark.usefixtures('lbfgs_on_cpu')
class TestTheLBFGSExitRecord:
    """``lbfgs_exit`` names the test that ended the phase and counts the
    iterations torch actually took. A one-iteration ``step`` is where DeepXDE
    breaks before counting, and where the gradient test, the budgets, a
    stuck line search and a divergence all look alike."""

    BUDGET = 10

    @staticmethod
    def _recording_line_search(monkeypatch):
        """Patch torch's line search to record max|grad| at x_0 and at every
        accepted iterate -- the value torch's own test compares."""
        import torch.optim.lbfgs as torch_lbfgs
        stock = torch_lbfgs._strong_wolfe
        grads = []

        def recording(obj_func, x, t, d, f, g, gtd, **kwargs):
            if not grads:
                grads.append(float(g.abs().max()))          # x_0
            out = stock(obj_func, x, t, d, f, g, gtd, **kwargs)
            grads.append(float(out[1].abs().max()))
            return out

        monkeypatch.setattr(torch_lbfgs, '_strong_wolfe', recording)
        return grads

    def _run_recorded(self, monkeypatch, gtol=None, loss_weights=None):
        grads = self._recording_line_search(monkeypatch)
        model = _train_guarded_lbfgs(_tiny_deepxde_model(0), self.BUDGET,
                                     gtol=gtol, loss_weights=loss_weights)
        return model, list(grads)

    def test_the_last_step_is_bounded_to_what_the_budget_allows(
            self, lbfgs_on_cpu, monkeypatch):
        """The cap is tested only BETWEEN ``step`` calls, so a call that ends
        early -- on torch's own per-call evaluation budget, or simply because
        the budget is not a multiple of ``iter_per_step`` -- used to be followed
        by a fresh FULL chunk. Measured before the fix: a budget of 9 chunked at
        4 ran 12 iterations, and seed 1 at the shipped chunk size ran 18 for a
        budget of 10. The shipped ``lbfgs_maxiter=2000`` could reach 2999.

        The chunk is forced small here so the overrun route is deterministic
        rather than dependent on which seed happens to exhaust its line
        search."""
        from deepxde.optimizers import config as dde_opt_config
        from epde.integrate.deepxde_integration import DeepXDEAdapter
        stock = DeepXDEAdapter._set_lbfgs_budget

        def chunked(maxiter, gtol=None):
            stock(maxiter, gtol)
            dde_opt_config.LBFGS_options['iter_per_step'] = 4
            dde_opt_config.LBFGS_options['fun_per_step'] = 5

        monkeypatch.setattr(DeepXDEAdapter, '_set_lbfgs_budget',
                            staticmethod(chunked))
        model = _train_guarded_lbfgs(_tiny_deepxde_model(0), 9)
        assert len(model.step_n_iters) > 1, 'the chunk size should force a refill'
        assert max(model.step_n_iters) == 9, model.step_n_iters
        assert model.step_n_iters[-1] == 9, model.step_n_iters
        assert model.lbfgs_exit == {'reason': 'maxiter', 'iterations': 9}

    def test_a_gradient_stop_mid_step_is_gtol_not_a_stall(self, monkeypatch):
        """torch breaks on the gradient inside one ``step``; the next ``step``
        returns at entry, which the guard used to call 'stalled'."""
        _, trace = self._run_recorded(monkeypatch)
        # The first iterate k >= 2 to reach a new low; a tolerance strictly
        # between it and every earlier gradient holds at k and nowhere before.
        k = next(i for i in range(2, len(trace)) if trace[i] < min(trace[:i]))
        assert k < self.BUDGET, 'the fixture needs the low inside the first step'
        model, grads = self._run_recorded(
            monkeypatch, gtol=0.5 * (trace[k] + min(trace[:k])))
        assert grads == trace[:k + 1], 'the gtol run did not reproduce the trace'
        # Broke at k inside step 1; step 2 returned at entry.
        assert model.step_n_iters == [k, k]
        assert model.lbfgs_exit == {'reason': 'gtol', 'iterations': k}

    def test_a_gradient_stop_on_the_first_iteration_of_a_step(self, monkeypatch):
        """One iteration then a gradient break: DeepXDE's 'converged' test fires,
        and only re-running the gradient test tells it apart. Weighted, like the
        adapter's own compile, so the re-test must apply the weights too."""
        weights = [2.0, 0.5]
        _, trace = self._run_recorded(monkeypatch, loss_weights=weights)
        assert trace[1] < trace[0], 'the fixture needs a first step that lowers max|grad|'
        model, grads = self._run_recorded(
            monkeypatch, gtol=0.5 * (trace[0] + trace[1]), loss_weights=weights)
        assert grads == trace[:2], 'the gtol run did not reproduce the trace'
        assert model.step_n_iters == [1]
        assert model.lbfgs_exit == {'reason': 'gtol', 'iterations': 1}

    def test_a_one_iteration_budget_is_maxiter_and_counts_the_iteration(self):
        model = _train_guarded_lbfgs(_tiny_deepxde_model(0), 1)
        assert model.step_n_iters == [1]
        assert model.lbfgs_exit == {'reason': 'maxiter', 'iterations': 1}

    def test_an_evaluation_budget_spent_in_one_iteration_is_maxeval(self):
        """``lbfgs_maxiter=2`` leaves ``fun_per_step = int(2.5) = 2``: the
        first line search spends the call's evaluations after one of its two
        iterations. That is a budget stop, not convergence."""
        from deepxde.optimizers import config as dde_opt_config
        model = _train_guarded_lbfgs(_tiny_deepxde_model(0), 2)
        assert dde_opt_config.LBFGS_options['fun_per_step'] == 2
        assert model.step_n_iters == [1]
        assert model.lbfgs_exit == {'reason': 'maxeval', 'iterations': 1}

    def test_a_stuck_line_search_is_converged_and_counts_every_iteration(
            self, monkeypatch):
        """The measured natural end of a solve: the float32 line search returns
        a zero step, torch breaks, and the next ``step`` makes one more
        (zero) iteration -- which DeepXDE's test detects but did not count."""
        import torch.optim.lbfgs as torch_lbfgs
        stock = torch_lbfgs._strong_wolfe
        calls = []
        stuck_from = 3

        def stuck(obj_func, x, t, d, f, g, gtd, **kwargs):
            calls.append(None)
            if len(calls) >= stuck_from:
                return f, g, 0.0, 1                 # no move: keep x, f, g
            return stock(obj_func, x, t, d, f, g, gtd, **kwargs)

        monkeypatch.setattr(torch_lbfgs, '_strong_wolfe', stuck)
        model = _train_guarded_lbfgs(_tiny_deepxde_model(0), self.BUDGET)
        # Both zero steps went through the line search (not a gtd break).
        assert len(calls) == stuck_from + 1
        assert model.step_n_iters == [stuck_from, stuck_from + 1]
        assert model.lbfgs_exit == {'reason': 'converged',
                                    'iterations': stuck_from + 1}

    def test_a_phase_that_starts_diverged_is_nonfinite(self):
        """NaN at entry: torch's test is False on NaN, so it iterates, and the
        one-iteration exit used to read 'converged'."""
        import torch
        model = _tiny_deepxde_model(0)
        with torch.no_grad():
            for p in model.net.parameters():
                p.fill_(float('nan'))
        model = _train_guarded_lbfgs(model, self.BUDGET)
        assert model.lbfgs_exit['reason'] == 'nonfinite'
        assert model.lbfgs_exit['iterations'] == _torch_n_iter(model)

    def test_a_divergence_mid_phase_is_nonfinite_not_stop_training(
            self, monkeypatch):
        """A NaN step inside a ``step`` call, which then ends on a zero step:
        DeepXDE's ``_test`` sees the NaN loss and sets ``stop_training``, which
        on its own would read as a callback's stop."""
        import torch.optim.lbfgs as torch_lbfgs
        stock = torch_lbfgs._strong_wolfe
        calls = []

        def diverging(obj_func, x, t, d, f, g, gtd, **kwargs):
            calls.append(None)
            if len(calls) == 3:
                return f, g, float('nan'), 1        # the move poisons x
            if len(calls) > 3:
                return f, g, 0.0, 1                 # then the call ends
            return stock(obj_func, x, t, d, f, g, gtd, **kwargs)

        monkeypatch.setattr(torch_lbfgs, '_strong_wolfe', diverging)
        model = _train_guarded_lbfgs(_tiny_deepxde_model(0), self.BUDGET)
        assert model.step_n_iters == [4]            # multi-iteration step
        assert model.lbfgs_exit == {'reason': 'nonfinite', 'iterations': 4}


class TestTheLossWeights:

    def test_variance_weight_is_the_inverse_population_variance(self):
        from epde.integrate.deepxde_integration import variance_weight
        values = np.array([1.0, 2.0, 4.0, 7.0])
        assert variance_weight(values) == pytest.approx(1.0 / np.var(values))

    @pytest.mark.parametrize('scale', [1e-3, 1e3])
    def test_a_weighted_mse_is_invariant_to_rescaling_the_field(self, scale):
        from epde.integrate.deepxde_integration import variance_weight
        rng = np.random.default_rng(0)
        obs = rng.standard_normal(200)
        pred = obs + 0.1 * rng.standard_normal(200)
        base = np.mean((pred - obs) ** 2) * variance_weight(obs)
        scaled = np.mean((scale * (pred - obs)) ** 2) * variance_weight(scale * obs)
        assert scaled == pytest.approx(base, rel=1e-9)

    def test_a_constant_channel_gets_a_finite_weight(self):
        from epde.integrate.deepxde_integration import variance_weight
        assert np.isfinite(variance_weight(np.full(50, 3.0)))

    def test_order_is_residuals_then_observations(self, fitted):
        system = _fitted_oscillator(fitted)
        samples = global_var.samples_manager
        key = samples.trajecatoryIDs[0]
        adapter = _deepxde_adapter()()
        adapter.domain_key = key
        eq = system.vals['u']
        observed = np.asarray(samples.get(('u', (1.0,)))[key]).reshape(-1)
        t_inner = np.asarray(samples.grids()[key][0])[adapter.domain_mask]
        from epde.integrate.deepxde_integration import time_split
        train = time_split(t_inner, adapter.train_frac).train
        weights = adapter.loss_weights([eq], [observed], train)
        target = np.asarray(eq.evaluate(active_only=True)[0][key]).reshape(-1)
        assert weights == pytest.approx([1.0 / np.var(target[train]),
                                         1.0 / np.var(observed[train])])


class TestTheRelativeLBFGSTolerance:
    """``lbfgs_gtol_rel`` scales the stopping test by the gradient the phase
    STARTS with, instead of DeepXDE's absolute ``gtol=1e-8`` on max|grad| --
    which two arms whose losses differ by an overall factor stop against at
    different degrees of convergence."""

    def test_it_works_without_a_first_order_phase(self, fitted):
        """'variance' weighting with ``epochs=0`` compiles nothing before the
        probe, so ``_max_abs_grad`` raised a TypeError that the solve's blanket
        handler turned into a silent all-NaN 'solve failed' -- a config that
        looks like a search finding nothing."""
        from epde.operators.common.objectives import Discrepancy
        system = _fitted_oscillator(fitted)
        host = SolverBasedFitness(['penalty_coeff', 'error_metric', 'deepxde_config'],
                                  primary=Discrepancy('deepxde'), backend='deepxde')
        host.params = {'penalty_coeff': 0.2, 'error_metric': 'rmse',
                       'deepxde_config': {'net': [16, 16], 'num_domain': 64,
                                          'num_test': 16, 'epochs': 0,
                                          'lbfgs_maxiter': 20, 'train_frac': 0.8,
                                          'loss_weight_mode': 'variance',
                                          'lbfgs_gtol_rel': 1e-3}}
        host.apply(system, {})
        eq = system.vals['u']
        assert eq.fitness_calculated
        assert np.isfinite(eq.fitness_value), 'the solve silently produced NaN'
        assert host.adapter.last_lbfgs_exit['iterations'] >= 0

    def test_a_bad_configuration_raises_instead_of_scoring_nan(self, fitted):
        """Every candidate would fail identically, so a NaN here reads as an
        empty search rather than as the setup error it is."""
        from epde.integrate.deepxde_integration import DeepXDEConfigError
        from epde.operators.common.objectives import Discrepancy
        system = _fitted_oscillator(fitted)
        host = SolverBasedFitness(['penalty_coeff', 'error_metric', 'deepxde_config'],
                                  primary=Discrepancy('deepxde'), backend='deepxde')
        host.params = {'penalty_coeff': 0.2, 'error_metric': 'rmse',
                       'deepxde_config': {'net': [16, 16], 'num_domain': 64,
                                          'num_test': 16, 'epochs': 20,
                                          'train_frac': 0.8, 'val_frac': 0.95}}
        with pytest.raises(DeepXDEConfigError, match='training time level'):
            host.apply(system, {})


class TestTheValidationCheckpoint:
    """The val block chooses the iterate; the test block scores it. The
    checkpointer must be a PURE OBSERVER -- a run with it installed has to
    follow the identical trajectory to one without."""

    def test_no_validation_block_installs_nothing(self):
        """The back-compat guarantee: val_frac=0 builds no checkpointer, so the
        solve is the two-way split it always was, bit for bit."""
        from epde.integrate.deepxde_integration import time_split
        adapter = _deepxde_adapter()()
        assert adapter.val_frac == 0.0 and adapter.last_val_stats is None
        split = time_split(np.arange(10.0), 0.8)
        assert adapter._make_checkpoint(None, None, [np.zeros(10)], split) is None

    def test_the_closure_hook_does_not_change_the_trajectory(self):
        """Wrapping ``opt.step`` must add a read and nothing else. Chunking the
        phase via ``iter_per_step`` would NOT have this property: torch's
        ``max_ls`` budget is per ``step()`` call, so short chunks truncate
        individual line searches and change which step is accepted."""
        from deepxde.optimizers import config as dde_opt_config
        from epde.integrate.deepxde_integration import (
            DeepXDEAdapter, _install_val_closure_hook)
        probe = np.stack([np.linspace(-1, 1, 16), np.full(16, 0.5)], axis=1)
        before = dict(dde_opt_config.LBFGS_options)

        class _Counter:
            def __init__(self):
                self.n = 0

            def record(self, force=False):
                self.n += 1

        try:
            plain = _tiny_deepxde_model(2)
            DeepXDEAdapter._set_lbfgs_budget(100)
            plain.compile('L-BFGS')
            plain.train(verbose=0)

            watched = _tiny_deepxde_model(2)
            DeepXDEAdapter._set_lbfgs_budget(100)
            watched.compile('L-BFGS')
            counter = _Counter()
            uninstall = _install_val_closure_hook(watched, counter)
            watched.train(verbose=0)
            uninstall()
        finally:
            dde_opt_config.LBFGS_options.update(before)

        np.testing.assert_array_equal(plain.predict(probe), watched.predict(probe))
        assert counter.n > 1, 'the hook never fired'

    def test_it_keeps_the_best_and_restores_it(self):
        import torch
        from epde.integrate.deepxde_integration import _ValCheckpoint

        net = torch.nn.Linear(1, 1)
        with torch.no_grad():
            net.weight.fill_(1.0)
            net.bias.fill_(0.0)
        x = torch.ones(4, 1)
        ckpt = _ValCheckpoint(net, x, torch.zeros(4, 1), [1.0],
                              lambda r: float(torch.sqrt(torch.mean(r ** 2))),
                              period_adam=1, period_lbfgs=1)
        ckpt.record()                                   # error 1.0
        with torch.no_grad():
            net.bias.fill_(-0.5)
        ckpt.record()                                   # error 0.5  <- the best
        with torch.no_grad():
            net.bias.fill_(2.0)
        ckpt.record(force=True)                         # error 3.0, the terminal
        assert ckpt.restore() == 'best'
        assert float(net.bias.detach()) == pytest.approx(-0.5)
        stats = ckpt.summary('best')
        assert stats['best_val'] == pytest.approx(0.5)
        assert stats['last_val'] == pytest.approx(3.0)
        assert stats['best_step'] == 2 and stats['n_checkpoints'] == 3

    def test_the_earliest_of_tied_checkpoints_wins(self):
        """The step()-entry closure re-evaluates the point it just moved to, so
        duplicate scores are guaranteed and '<=' would prefer the later copy."""
        import torch
        from epde.integrate.deepxde_integration import _ValCheckpoint

        net = torch.nn.Linear(1, 1)
        ckpt = _ValCheckpoint(net, torch.ones(2, 1), torch.zeros(2, 1), [1.0],
                              lambda r: float(torch.sqrt(torch.mean(r ** 2))),
                              period_adam=1, period_lbfgs=1)
        ckpt.record()
        ckpt.record()                                   # identical parameters
        assert ckpt.best_step == 1 and ckpt.n_snapshots == 1

    def test_a_real_solve_selects_on_validation_and_scores_on_test(self, fitted):
        from epde.operators.common.objectives import Discrepancy
        system = _fitted_oscillator(fitted)
        host = SolverBasedFitness(['penalty_coeff', 'error_metric', 'deepxde_config'],
                                  primary=Discrepancy('deepxde'), backend='deepxde')
        host.params = {'penalty_coeff': 0.2, 'error_metric': 'rmse',
                       'deepxde_config': {'net': [16, 16], 'num_domain': 64,
                                          'num_test': 16, 'epochs': 50,
                                          'train_frac': 0.8, 'val_frac': 0.1,
                                          'val_checkpoints': 10}}
        host.apply(system, {})
        eq = system.vals['u']
        assert eq.fitness_calculated and np.isfinite(eq.fitness_value)
        stats = host.adapter.last_val_stats
        assert stats is not None and stats['n_checkpoints'] > 0
        assert stats['selected'] == 'best'
        # The terminal iterate is always recorded, so the chosen model can never
        # be worse on validation than the one today's code would have returned.
        assert stats['best_val'] <= stats['last_val']
        assert stats['improvement_ratio'] >= 1.0
        assert stats['n_val_points'] > 0


class TestAFixedCoefficientDeepXDESolve:
    """A small real solve through the host: observation term on the training
    window, fixed coefficients everywhere, scored on the held-out tail."""

    def test_scores_the_held_out_tail(self, fitted):
        from epde.operators.common.objectives import Discrepancy
        system = _fitted_oscillator(fitted)
        host = SolverBasedFitness(['penalty_coeff', 'error_metric', 'deepxde_config'],
                                  primary=Discrepancy('deepxde'), backend='deepxde')
        host.params = {'penalty_coeff': 0.2, 'error_metric': 'rmse',
                       'deepxde_config': {'net': [16, 16], 'num_domain': 64,
                                          'num_test': 16, 'epochs': 50,
                                          'train_frac': 0.8}}
        seen = {}
        original = Discrepancy._compute_deepxde

        def spy(self, eq, eq_idx, sctx):
            seen['solution'], seen['data'] = sctx.solution, sctx.g_fun_vals
            return original(self, eq, eq_idx, sctx)

        Discrepancy._compute_deepxde = spy
        try:
            host.apply(system, {})
        finally:
            Discrepancy._compute_deepxde = original

        eq = system.vals['u']
        assert eq.fitness_calculated and np.isfinite(eq.fitness_value)
        samples = global_var.samples_manager
        for key in samples.trajecatoryIDs:
            n_inner = int(np.asarray(samples.gFunc('m')[key]).sum())
            n_held_out = n_inner - int(np.ceil(0.8 * n_inner))    # 1-D: one point per level
            assert seen['solution'][key][0].shape == (n_held_out,)
            observed = np.asarray(samples.get(('u', (1.0,)))[key]).reshape(-1)
            np.testing.assert_array_equal(seen['data'][key][0], observed[-n_held_out:])
