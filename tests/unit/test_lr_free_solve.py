"""Learning-rate-free options of the DeepXDE fitness solve.

The pins in ``TestTheShippedSolveIsPinned`` were captured from the code BEFORE
any lr-free option existed (Sep 28 2026), on CPU. They cover the two seams the
new options touch and that no earlier test reached bitwise:

* the gradient-weight probe plus Adam plus L-BFGS (``epochs > 0``);
* the gradient-weight probe plus L-BFGS alone (``epochs = 0``).

``TestAFixedCoefficientDeepXDESolve`` uses 'variance' weights and no L-BFGS,
so it never passes through either. Every new option is opt-in, so the pins
must survive all of them unchanged.
"""
import hashlib

import numpy as np
import pytest
import torch

from test_solver_path import (_fitted_oscillator, fitted,          # noqa: F401
                              lbfgs_on_cpu)


@pytest.fixture(scope='module', autouse=True)
def _restore_torch_default_device():
    before = torch.get_default_device()
    yield
    torch.set_default_device(before)


@pytest.fixture
def solve_on_cpu(lbfgs_on_cpu, monkeypatch):                     # noqa: F811
    """Keep the solve on the CPU: ``_solve_device`` would otherwise re-enter
    DeepXDE's CUDA default for the length of the solve."""
    import epde.integrate.deepxde_integration as dxi
    monkeypatch.setattr(dxi, 'DDE_SOLVE_DEVICE', None)
    yield dxi


def _solve(dxi, search, config, seed=0):
    """One adapter solve on the fixture's first trajectory; returns the
    full-grid prediction, the reported loss and the adapter."""
    import deepxde as dde
    import epde.globals as global_var
    system = _fitted_oscillator(search)
    samples = global_var.samples_manager
    key = samples.trajecatoryIDs[0]
    observed = np.asarray(samples.get(('u', (1.0,)))[key]).reshape(-1)
    adapter = dxi.DeepXDEAdapter(**config)
    dde.config.set_random_seed(seed)
    solutions, loss = adapter.solve(system, grids=samples.grids()[key],
                                    data=[observed], domain_key=key)
    return np.asarray(solutions[0]), float(loss), adapter


def _digest(solution, loss):
    h = hashlib.sha256(np.ascontiguousarray(solution, dtype=np.float64).tobytes())
    h.update(repr(loss).encode())
    return h.hexdigest()[:24]


PIN_CONFIGS = {
    'adam+lbfgs': {'net': [16, 16], 'num_domain': 64, 'num_test': 16, 'epochs': 20,
                   'lr': 1e-4, 'lbfgs_maxiter': 20, 'train_frac': 0.8,
                   'loss_weight_mode': 'gradient'},
    'lbfgs-only': {'net': [16, 16], 'num_domain': 64, 'num_test': 16, 'epochs': 0,
                   'lr': 1e-4, 'lbfgs_maxiter': 20, 'train_frac': 0.8,
                   'loss_weight_mode': 'gradient'},
}
#: Captured Sep 28 2026 from the pre-change adapter, CPU, seed 0.
PIN_DIGESTS = {
    'adam+lbfgs': '6f9cc57457ed66c62a35fddb',
    'lbfgs-only': '361e35ab2ff3ef152e604911',
}


@pytest.mark.usefixtures('solve_on_cpu')
class TestTheShippedSolveIsPinned:

    @pytest.mark.parametrize('name', sorted(PIN_CONFIGS))
    def test_the_prediction_and_loss_are_bitwise_unchanged(self, name, fitted,   # noqa: F811
                                                            solve_on_cpu):
        solution, loss, adapter = _solve(solve_on_cpu, fitted, PIN_CONFIGS[name])
        assert np.isfinite(solution).all() and np.isfinite(loss)
        digest = _digest(solution, loss)
        print(f'PIN {name} {digest} loss={loss!r} exit={adapter.last_lbfgs_exit}')
        assert PIN_DIGESTS[name] is not None, f'capture: {name} -> {digest}'
        assert digest == PIN_DIGESTS[name]

    def test_a_repeat_in_the_same_process_is_bitwise_equal(self, fitted,          # noqa: F811
                                                           solve_on_cpu):
        """What the pins rest on: on the CPU the solve is a pure function of
        the config and the seed, whatever ran before it in the process."""
        first = _solve(solve_on_cpu, fitted, PIN_CONFIGS['lbfgs-only'])
        second = _solve(solve_on_cpu, fitted, PIN_CONFIGS['lbfgs-only'])
        assert _digest(first[0], first[1]) == _digest(second[0], second[1])


# ============================================================ the lr-free options
LBFGS_ONLY = PIN_CONFIGS['lbfgs-only']


@pytest.mark.usefixtures('solve_on_cpu')
class TestTheNullChecks:
    """Options that must change NOTHING when set to their null value -- the
    evidence that each arm's seam is clean, checked bitwise."""

    def test_an_lbfgs_only_run_needs_no_learning_rate(self, fitted, solve_on_cpu):  # noqa: F811
        with_lr = _solve(solve_on_cpu, fitted, LBFGS_ONLY)
        without = _solve(solve_on_cpu, fitted, dict(LBFGS_ONLY, lr=None))
        assert _digest(with_lr[0], with_lr[1]) == _digest(without[0], without[1])

    def test_a_zero_iteration_prefit_changes_nothing(self, fitted, solve_on_cpu):   # noqa: F811
        base = _solve(solve_on_cpu, fitted, LBFGS_ONLY)
        null = _solve(solve_on_cpu, fitted, dict(LBFGS_ONLY, init='data_prefit',
                                                 prefit_maxiter=0))
        assert _digest(base[0], base[1]) == _digest(null[0], null[1])

    def test_the_pairing_digest_changes_nothing(self, fitted, solve_on_cpu):        # noqa: F811
        base = _solve(solve_on_cpu, fitted, LBFGS_ONLY)
        paired = _solve(solve_on_cpu, fitted, dict(LBFGS_ONLY, pairing_digest=True))
        assert _digest(base[0], base[1]) == _digest(paired[0], paired[1])
        assert paired[2].last_pairing['theta0'] and paired[2].last_pairing['loss_weights']


class TestTheConfigValidation:

    @pytest.mark.parametrize('cfg', [{'lr': 'fast'}, {'lr': -1.0}, {'lr': None, 'epochs': 5},
                                     {'init': 'magic'}, {'precision': 'float16'},
                                     {'pde_loss': 'l1'}, {'input_transform': 'log'}])
    def test_a_bad_value_raises_instead_of_scoring_nan(self, cfg):
        from epde.integrate.deepxde_integration import DeepXDEAdapter, DeepXDEConfigError
        with pytest.raises(DeepXDEConfigError):
            DeepXDEAdapter(**cfg)


@pytest.mark.usefixtures('solve_on_cpu')
class TestTheStartOptions:

    def test_the_last_layer_is_the_least_squares_fit(self, solve_on_cpu):
        import deepxde as dde
        from epde.integrate.heldout import time_split
        dxi = solve_on_cpu
        adapter = dxi.DeepXDEAdapter(init='lstsq_last_layer')
        dde.config.set_random_seed(0)
        net = dde.nn.FNN([1, 8, 8, 1], 'tanh', 'Glorot normal')
        t = np.linspace(0.0, 2.0, 40)
        u = np.sin(3 * t)
        split = time_split(t, 0.8)
        stats = adapter._apply_init(net, t[:, None], [u], split)
        with torch.no_grad():
            h = torch.as_tensor(t[split.train][:, None], dtype=torch.float32)
            for lin in net.linears[:-1]:
                h = torch.tanh(lin(h))
        A = np.hstack([h.double().numpy(), np.ones((int(split.train.sum()), 1))])
        want = np.linalg.lstsq(A, u[split.train], rcond=None)[0]
        got = np.concatenate([net.linears[-1].weight.detach().double().numpy().ravel(),
                              net.linears[-1].bias.detach().double().numpy()])
        np.testing.assert_allclose(got, want, rtol=1e-4, atol=1e-5)
        assert stats['rank'] == 9 and np.isfinite(stats['cond'])

    def test_float64_starts_from_exactly_the_float32_weights_and_restores(self, solve_on_cpu):
        import deepxde as dde
        dxi = solve_on_cpu
        dde.config.set_random_seed(3)
        f32 = dxi.DeepXDEAdapter()._build_net([2, 8, 1])
        adapter = dxi.DeepXDEAdapter(precision='float64')
        before = (dde.config.real.precision, torch.get_default_dtype())
        dde.config.set_random_seed(3)
        with adapter._precision_scope():
            f64 = adapter._build_net([2, 8, 1])
            assert dde.config.real.precision == 64
        assert (dde.config.real.precision, torch.get_default_dtype()) == before
        for a, b in zip(f32.parameters(), f64.parameters()):
            assert b.dtype == torch.float64
            assert torch.equal(a.detach().double(), b.detach())

    def test_a_float64_solve_leaves_float32_behind(self, fitted, solve_on_cpu):     # noqa: F811
        import deepxde as dde
        before = (dde.config.real.precision, torch.get_default_dtype())
        _, loss, _ = _solve(solve_on_cpu, fitted, dict(LBFGS_ONLY, precision='float64'))
        assert np.isfinite(loss)
        assert (dde.config.real.precision, torch.get_default_dtype()) == before


class TestTheCancellationRatio:

    def test_it_is_bounded_and_has_a_finite_gradient_where_every_term_vanishes(self):
        from epde.integrate.residual_terms import cancellation_ratio
        a = torch.tensor([0.0, 1.0, -2.0, 0.5], requires_grad=True)
        b = torch.tensor([0.0, 1.0, 3.0, -0.5], requires_grad=True)
        r = a - b                                      # an equation a = b
        mass = a.abs() + b.abs()
        rho = cancellation_ratio(r, mass, torch)
        assert rho[0] == 0.0                           # 0/0: the equation holds
        assert bool((rho.abs() <= 1.0).all())
        (rho ** 2).sum().backward()
        assert torch.isfinite(a.grad).all() and torch.isfinite(b.grad).all()

    def test_the_inverse_mass_ignores_points_with_no_information(self):
        from epde.integrate.residual_terms import inverse_mass, mass_diagnostics
        np.testing.assert_array_equal(inverse_mass(np.array([0.0, 2.0])), [0.0, 0.5])
        d = mass_diagnostics(np.array([0.0, 1.0, 4.0]))
        assert d['zero_share'] == pytest.approx(1 / 3)

    @pytest.mark.parametrize('loss', ['sinv_live', 'sinv_data'])
    def test_a_scale_invariant_solve_is_finite(self, loss, fitted, solve_on_cpu):  # noqa: F811
        solution, value, adapter = _solve(solve_on_cpu, fitted, dict(LBFGS_ONLY, pde_loss=loss))
        assert np.isfinite(solution).all() and np.isfinite(value)
        assert adapter.last_sinv_stats['floors'] == [0.0]


@pytest.mark.usefixtures('solve_on_cpu')
class TestTheSharedDataFit:
    """One observation-only fit per dataset, reused as every candidate's start."""

    @pytest.fixture(autouse=True)
    def _empty_cache(self, solve_on_cpu):
        solve_on_cpu.DeepXDEAdapter._SHARED_FITS.clear()
        yield
        solve_on_cpu.DeepXDEAdapter._SHARED_FITS.clear()

    CFG = dict(LBFGS_ONLY, init='shared_data_fit', shared_fit_maxiter=50)

    def _setup(self, fitted):
        import epde.globals as global_var
        system = _fitted_oscillator(fitted)
        samples = global_var.samples_manager
        key = samples.trajecatoryIDs[0]
        observed = np.asarray(samples.get(('u', (1.0,)))[key]).reshape(-1)
        return system, samples, key, observed

    def test_the_second_solve_loads_the_first_ones_fit(self, fitted, solve_on_cpu):  # noqa: F811
        first = _solve(solve_on_cpu, fitted, self.CFG)
        second = _solve(solve_on_cpu, fitted, self.CFG)
        assert first[2].last_init_stats['source'] == 'fitted'
        assert second[2].last_init_stats['source'] == 'memory'
        assert _digest(first[0], first[1]) == _digest(second[0], second[1])

    def test_a_fit_loaded_from_disk_is_bitwise_the_fitted_one(self, fitted, solve_on_cpu,  # noqa: F811
                                                              tmp_path):
        cfg = dict(self.CFG, shared_fit_dir=str(tmp_path))
        fitted_run = _solve(solve_on_cpu, fitted, cfg)
        solve_on_cpu.DeepXDEAdapter._SHARED_FITS.clear()
        loaded_run = _solve(solve_on_cpu, fitted, cfg)
        assert loaded_run[2].last_init_stats['source'] == 'disk'
        assert _digest(fitted_run[0], fitted_run[1]) == _digest(loaded_run[0], loaded_run[1])

    def test_the_fit_never_reads_the_tail(self, fitted, solve_on_cpu):          # noqa: F811
        import deepxde as dde
        from epde.integrate.heldout import time_split
        dxi = solve_on_cpu
        system, samples, key, observed = self._setup(fitted)
        mask = np.asarray(samples.gFunc('m')[key]).reshape(-1)
        t_inner = np.asarray(samples.grids()[key][0]).reshape(-1)[mask]
        blank = observed.copy()
        blank[~time_split(t_inner, 0.8).train] = np.nan
        keys = []
        for data in (observed, blank):
            dxi.DeepXDEAdapter._SHARED_FITS.clear()
            adapter = dxi.DeepXDEAdapter(**self.CFG)
            dde.config.set_random_seed(0)
            adapter.prepare_shared_fit(system, samples.grids()[key], [data], key)
            keys.append(adapter.last_init_stats['key'])
        assert keys[0] == keys[1]

    def test_a_fixed_seed_start_is_shared_and_leaves_the_global_rng_alone(self, fitted,  # noqa: F811
                                                                          solve_on_cpu):
        import deepxde as dde
        dxi = solve_on_cpu
        system, samples, key, observed = self._setup(fitted)
        adapter = dxi.DeepXDEAdapter(**dict(self.CFG, shared_fit_seed=7))
        dde.config.set_random_seed(0)
        adapter.prepare_shared_fit(system, samples.grids()[key], [observed], key)
        k0 = adapter.last_init_stats['key']
        dde.config.set_random_seed(123)               # another candidate's own init
        adapter.prepare_shared_fit(system, samples.grids()[key], [observed], key)
        assert adapter.last_init_stats['key'] == k0   # same start whatever the global RNG
        net = dde.nn.FNN([1, 16, 16, 16, 16, 1], 'tanh', 'Glorot normal')
        state = torch.get_rng_state()
        adapter._shared_data_fit(net, torch.tensor([[0.0], [0.5], [1.0]]),
                                 np.array([[0.0], [1.0], [4.0]]))
        assert torch.equal(state, torch.get_rng_state())


# ============================================================ Stage 1b/1c mechanics
class TestTheLrFreeOptimisers:

    def test_dog_takes_the_distance_over_gradients_step(self):
        from epde.integrate.lr_free import DoG
        x = torch.tensor([3.0, -4.0], requires_grad=True)
        target = torch.tensor([1.0, 1.0])
        opt = DoG([x], reps_rel=1e-6)

        def closure():
            opt.zero_grad()
            loss = 0.5 * ((x - target) ** 2).sum()
            loss.backward()
            return loss

        x0 = x.detach().clone()
        g = (x0 - target)
        opt.step(closure)
        eta = 1e-6 * (1.0 + float(x0.norm())) / float(g.norm())   # rbar / sqrt(G)
        torch.testing.assert_close(x.detach(), x0 - eta * g)
        for _ in range(300):
            opt.step(closure)
        assert float(((x - target) ** 2).sum()) < 1e-6 * float((g ** 2).sum())

    def test_prodigy_grows_its_step_from_d0_and_converges(self):
        from epde.integrate.lr_free import Prodigy
        x = torch.tensor([3.0, -4.0], requires_grad=True)
        target = torch.tensor([1.0, 1.0])
        opt = Prodigy([x], d0=1e-6)

        def closure():
            opt.zero_grad()
            loss = 0.5 * ((x - target) ** 2).sum()
            loss.backward()
            return loss

        start = float(closure())
        for _ in range(500):
            opt.step(closure)
        assert opt.param_groups[0]['d'] > 1e-3            # grew six orders from d0
        assert float(closure()) < 1e-3 * start

    @pytest.mark.parametrize('cls, kw', [('DoG', {'reps_rel': 0.0}), ('Prodigy', {'d0': -1.0})])
    def test_a_non_positive_constant_is_refused(self, cls, kw):
        from epde.integrate import lr_free
        with pytest.raises(ValueError):
            getattr(lr_free, cls)([torch.zeros(2, requires_grad=True)], **kw)


class TestTheStage1bConfig:

    @pytest.mark.parametrize('cfg', [
        {'lr': None, 'epochs': 5},                                    # adam needs an lr
        {'lr_rule': 'prefit_distance', 'first_order': 'dog', 'lr': None},
        {'lr_rule': 'data_selected', 'epochs': 0, 'lr': None},
        {'second_order': 'lm', 'lbfgs_maxiter': 10},                  # LM replaces L-BFGS
        {'first_order': 'sgd'}, {'second_order': 'newton'}, {'lr_rule': 'guess'}])
    def test_an_incoherent_config_raises(self, cfg):
        from epde.integrate.deepxde_integration import DeepXDEAdapter, DeepXDEConfigError
        with pytest.raises(DeepXDEConfigError):
            DeepXDEAdapter(**cfg)

    @pytest.mark.parametrize('cfg', [
        {'first_order': 'dog', 'lr': None}, {'first_order': 'prodigy', 'lr': None},
        {'lr_rule': 'prefit_distance', 'lr': None}, {'lr_rule': 'data_selected', 'lr': None},
        {'second_order': 'lm', 'lbfgs_maxiter': 0, 'epochs': 0, 'lr': None}])
    def test_the_lr_free_configs_are_accepted(self, cfg):
        from epde.integrate.deepxde_integration import DeepXDEAdapter
        DeepXDEAdapter(**cfg)


SMALL_ADAM = {'net': [16, 16], 'num_domain': 64, 'num_test': 16, 'epochs': 20,
              'lbfgs_maxiter': 20, 'train_frac': 0.8, 'loss_weight_mode': 'gradient'}


@pytest.mark.usefixtures('solve_on_cpu')
class TestTheStage1bSolves:

    @pytest.mark.parametrize('extra', [{'first_order': 'dog', 'lr': None},
                                       {'first_order': 'prodigy', 'lr': None}])
    def test_an_lr_free_first_order_phase_solves(self, extra, fitted, solve_on_cpu):  # noqa: F811
        solution, loss, adapter = _solve(solve_on_cpu, fitted, dict(SMALL_ADAM, **extra))
        assert np.isfinite(solution).all() and np.isfinite(loss)

    @pytest.mark.parametrize('rule', ['prefit_distance', 'data_selected'])
    def test_a_data_driven_lr_is_finite_and_leaves_the_rng_alone(self, rule, fitted,  # noqa: F811
                                                                 solve_on_cpu):
        cfg = dict(SMALL_ADAM, lr_rule=rule, lr=None, prefit_maxiter=20)
        solution, loss, adapter = _solve(solve_on_cpu, fitted, cfg)
        stats = adapter.last_lr_stats
        assert stats['rule'] == rule and stats['lr'] > 0 and np.isfinite(loss)
        # the rule runs on copies inside fork_rng: calling it again draws nothing
        import deepxde as dde
        from epde.integrate.heldout import time_split
        dde.config.set_random_seed(0)
        net = dde.nn.FNN([1, 8, 1], 'tanh', 'Glorot normal')
        t = np.linspace(0.0, 1.0, 20)
        before = torch.get_rng_state()
        adapter._data_driven_lr(net, t[:, None], [np.sin(t)], time_split(t, 0.8))
        assert torch.equal(before, torch.get_rng_state())

    def test_levenberg_marquardt_reports_the_loss_deepxde_reports(self, fitted,      # noqa: F811
                                                                  solve_on_cpu):
        cfg = {'net': [8, 8], 'num_domain': 64, 'num_test': 16, 'epochs': 0,
               'lbfgs_maxiter': 0, 'second_order': 'lm', 'lm_maxiter': 15, 'lr': None,
               'train_frac': 0.8, 'loss_weight_mode': 'gradient'}
        solution, loss, adapter = _solve(solve_on_cpu, fitted, cfg)
        stats = adapter.last_lm_stats
        assert np.isfinite(solution).all()
        assert stats['accepted'] >= 1
        assert stats['history'][-1][1] < stats['history'][0][1]        # LM lowered the loss
        # ||R||^2 is DeepXDE's weighted loss at the same parameters
        assert stats['loss'] == pytest.approx(loss, rel=1e-4)
        assert adapter.last_lbfgs_exit['reason'].startswith('lm_')


#: Captured Sep 29 2026, before the Stage-2 transform refactor (CPU, seed 0).
S1_PIN = 'c28c0dc36d3b02622e576293'


@pytest.mark.usefixtures('solve_on_cpu')
class TestTheS1PathIsPinned:
    """The affine-input (S1) solve, bitwise, so the Stage-2 transform
    composition (periodic embedding) can be shown not to move it."""

    def test_the_affine_input_solve_is_unchanged(self, fitted, solve_on_cpu):      # noqa: F811
        solution, loss, _ = _solve(solve_on_cpu, fitted,
                                   dict(LBFGS_ONLY, input_transform='affine'))
        digest = _digest(solution, loss)
        print(f'S1PIN {digest}')
        assert S1_PIN is not None, f'capture: {digest}'
        assert digest == S1_PIN



# ============================================================ Stage 2 mechanisms
@pytest.mark.usefixtures('solve_on_cpu')
class TestThePeriodicEmbedding:

    def test_it_makes_the_network_exactly_periodic(self, solve_on_cpu):
        import deepxde as dde
        from epde.integrate.heldout import time_split
        dxi = solve_on_cpu
        adapter = dxi.DeepXDEAdapter(periodic_embedding='auto', input_transform='affine')
        t = np.linspace(0.0, 1.0, 11)
        x = np.linspace(-1.0, 1.0, 16, endpoint=False)              # period 2
        grids = list(np.meshgrid(t, x, indexing='ij'))
        adapter._embedding = [(1, -1.0, 2.0)]                        # EPDE axis 1 periodic
        dde.config.set_random_seed(0)
        net = adapter._build_net([3, 8, 1])                          # 2 inputs + 1 extra
        adapter._apply_transforms(net, grids, [np.zeros(grids[0].size)],
                                  time_split(grids[0].reshape(-1), 0.8))
        pts = torch.tensor([[0.3, 0.2], [-0.7, 0.9]])                # columns [x, t]
        shifted = pts.clone()
        shifted[:, 0] += 2.0
        with torch.no_grad():
            torch.testing.assert_close(net(pts), net(shifted), rtol=0, atol=1e-6)

    def test_it_is_inert_without_a_periodic_axis(self, fitted, solve_on_cpu):   # noqa: F811
        """The 1-D fixture has no spatial axis, so the embedding must change
        nothing: bitwise the affine-input (S1) pin."""
        solution, loss, _ = _solve(solve_on_cpu, fitted,
                                   dict(LBFGS_ONLY, input_transform='affine',
                                        periodic_embedding='auto'))
        assert _digest(solution, loss) == S1_PIN


@pytest.mark.usefixtures('solve_on_cpu')
class TestThePairingDigest:

    def test_a_float64_net_pairs_with_its_float32_draw(self, solve_on_cpu):
        """float64 builds the float32 Glorot draw and upcasts it, so the digest
        (taken on float32 bytes) is the float32 net's."""
        import copy
        import deepxde as dde
        dxi = solve_on_cpu
        dde.config.set_random_seed(0)
        net = dxi.DeepXDEAdapter()._build_net([2, 8, 1])
        wide = copy.deepcopy(net).double()
        assert dxi.DeepXDEAdapter._param_digest(net) == \
            dxi.DeepXDEAdapter._param_digest(wide)


@pytest.mark.usefixtures('solve_on_cpu')
class TestTheCollocationWarp:

    @staticmethod
    def _setup(dxi, kind):
        from epde.integrate.heldout import time_split
        adapter = dxi.DeepXDEAdapter(collocation_density=kind)
        t = np.linspace(0.0, 1.0, 21)
        x = np.linspace(-1.0, 1.0, 64)
        T, X = np.meshgrid(t, x, indexing='ij')
        u = np.tanh(X / 0.05) + 0 * T                               # a sharp front at x=0
        adapter._raw = {'u': u}
        split = time_split(T.reshape(-1), 0.8)
        rng = np.random.default_rng(0)
        pts = np.stack([rng.uniform(-1, 1, 500), rng.uniform(0, 1, 500)], axis=1)
        return adapter, [T, X], pts, split

    def test_a_flat_density_is_the_identity(self, solve_on_cpu):
        adapter, grids, pts, split = self._setup(solve_on_cpu, 'flat')
        out = adapter._warped_collocation(pts, grids, ['u'], split)
        np.testing.assert_allclose(out, pts, rtol=0, atol=1e-12)

    def test_the_gradient_density_moves_points_to_the_front_deterministically(self,
                                                                              solve_on_cpu):
        adapter, grids, pts, split = self._setup(solve_on_cpu, 'data_gradient')
        out = adapter._warped_collocation(pts, grids, ['u'], split)
        again = adapter._warped_collocation(pts, grids, ['u'], split)
        np.testing.assert_array_equal(out, again)
        assert out.shape == pts.shape
        assert out[:, 0].min() >= -1.0 and out[:, 0].max() <= 1.0
        assert out[:, 1].min() >= 0.0 and out[:, 1].max() <= 1.0
        near_front = lambda p: float(np.mean(np.abs(p[:, 0]) < 0.1))
        assert near_front(out) > 2 * near_front(pts)

    def test_the_scored_tail_keeps_its_share_whatever_the_last_train_level(self,
                                                                          solve_on_cpu):
        """1-D: copying the last train level into the tail let ONE number decide
        how many physics points the scored tail gets (a front at t_train, last
        level density 38 against a train mean of 2, would have sent ~0.8 of the
        points into a tail that is 0.2 of the time). The tail now carries the
        train-window mean density. What remains is the one trapezoid segment
        straddling t_train, which rightly shares the front's density (0.287)."""
        from epde.integrate.heldout import time_split
        adapter = solve_on_cpu.DeepXDEAdapter(collocation_density='data_gradient')
        t = np.linspace(0.0, 1.0, 101)
        split = time_split(t, 0.8)
        pts = np.random.default_rng(0).uniform(0, 1, (4000, 1))
        tail = float(np.mean(pts[:, 0] > split.t_train))
        shares = []
        for u in (np.sin(4 * t), np.sin(4 * t) + 3.0 * np.tanh((t - split.t_train) / 0.004)):
            adapter._raw = {'u': u}
            out = adapter._warped_collocation(pts, [t], ['u'], split)
            shares.append(float(np.mean(out[:, 0] > split.t_train)))
        smooth, front = shares
        assert abs(smooth - tail) < 0.02, (smooth, tail)
        assert front < 1.5 * tail, (front, tail)


@pytest.mark.usefixtures('solve_on_cpu')
class TestTheCollocationWarpOnThreeAxes:
    """NS-shaped grids: EPDE axes (t, y, x), DeepXDE columns [y, x, t]."""

    @staticmethod
    def _setup(dxi, kind, x=None):
        from epde.integrate.heldout import time_split
        adapter = dxi.DeepXDEAdapter(collocation_density=kind)
        t = np.linspace(0.0, 1.0, 11)
        y = np.linspace(-1.0, 1.0, 17)
        x = np.linspace(-1.0, 1.0, 64) if x is None else x
        T, Y, X = np.meshgrid(t, y, x, indexing='ij')
        u = np.tanh((X - 0.4 * Y) / 0.05) + 0 * T               # a front slanted in y
        adapter._raw = {'u': u}
        split = time_split(T.reshape(-1), 0.8)
        rng = np.random.default_rng(0)
        pts = np.stack([rng.uniform(-1, 1, 800), rng.uniform(-1, 1, 800),
                        rng.uniform(0, 1, 800)], axis=1)        # columns [y, x, t]
        return adapter, [T, Y, X], pts, split

    def test_it_stays_in_the_box_keeps_its_faces_and_never_reads_the_tail(self,
                                                                          solve_on_cpu):
        adapter, grids, pts, split = self._setup(solve_on_cpu, 'data_gradient')
        pts = pts.astype(np.float32)                            # DeepXDE's default dtype
        pts[:3] = [[-1.0, 1.0, 0.5], [0.3, -1.0, 0.0], [1.0, 0.2, 1.0]]   # on faces
        out = adapter._warped_collocation(pts, grids, ['u'], split)
        assert out.dtype == np.float32 and out.shape == pts.shape
        for col, (lo, hi) in enumerate([(-1.0, 1.0), (-1.0, 1.0), (0.0, 1.0)]):
            assert out[:, col].min() >= lo and out[:, col].max() <= hi
        for row, col in [(0, 0), (0, 1), (1, 1), (1, 2), (2, 0), (2, 2)]:
            assert out[row, col] == pts[row, col]
        tail = grids[0] > split.t_train
        adapter._raw = {'u': np.where(tail, np.nan, adapter._raw['u'])}
        np.testing.assert_array_equal(adapter._warped_collocation(pts, grids, ['u'], split),
                                      out)

    def test_a_front_that_moves_with_y_draws_the_points_at_their_own_y(self, solve_on_cpu):
        """x is warped conditional on the y cell: the front sits at x = 0.4 y."""
        adapter, grids, pts, split = self._setup(solve_on_cpu, 'data_gradient')
        out = adapter._warped_collocation(pts, grids, ['u'], split)
        near_front = lambda p: float(np.mean(np.abs(p[:, 1] - 0.4 * p[:, 0]) < 0.1))
        assert near_front(out) > 2 * near_front(pts)

    def test_a_flat_density_is_the_identity_on_any_grid(self, solve_on_cpu):
        """A flat density makes every trapezoid CDF the normalised coordinate
        itself, on a non-uniform axis too; what is left is rounding."""
        x = np.sin(np.linspace(-0.5 * np.pi, 0.5 * np.pi, 33))  # non-uniform, ends at +-1
        adapter, grids, pts, split = self._setup(solve_on_cpu, 'flat', x=x)
        out = adapter._warped_collocation(pts, grids, ['u'], split)
        np.testing.assert_allclose(out, pts, rtol=0, atol=1e-12)

    def test_a_flat_warp_of_float32_points_returns_them_bitwise(self, solve_on_cpu):
        """Why a flat float32 cell is inert: the float64 rounding of the warp is
        far below half a float32 ulp, so the cast back returns the points."""
        adapter, grids, pts, split = self._setup(solve_on_cpu, 'flat')
        pts = pts.astype(np.float32)
        out = adapter._warped_collocation(pts, grids, ['u'], split)
        assert out.tobytes() == pts.tobytes()


class TestNoBiasUnlessTheEquationHasOne:
    """The PDE carries a free coefficient only when the equation's support keeps
    one (``weights_internal[-1] != 0``), whatever ``weights_final[-1]`` holds."""

    @staticmethod
    def _equation(support_free, free_value=5.0):
        from types import SimpleNamespace
        u = SimpleNamespace(is_deriv=True, deriv_code=[None], variable='u', name='u',
                            params=np.array([1.0]), ftype='u', label='u',
                            params_description={0: {'name': 'power', 'bounds': (1, 3)}},
                            structure=[], cache_label=None)
        term, target = SimpleNamespace(structure=[u]), SimpleNamespace(structure=[u])
        return SimpleNamespace(
            structure=[term, target], target_idx=1, target=target,
            main_var_to_explain='u', weights_final_evald=True,
            weights_final=np.array([2.0, free_value]),
            weights_internal=np.array([1.0, 1.0 if support_free else 0.0]),
            weight_index=lambda i, tgt: i if i < tgt else i - 1)

    @pytest.mark.usefixtures('solve_on_cpu')
    def test_the_pinn_residual(self, solve_on_cpu):
        adapter = solve_on_cpu.DeepXDEAdapter()
        adapter.coord_map = {}
        y = torch.tensor([[1.0], [2.0]])
        x = torch.zeros(2, 1)
        for support, want in ((False, y), (True, y + 5.0)):     # 2u - u (+ 5)
            pde = adapter._equation_system_to_pde_func(None, [self._equation(support)], ['u'])
            torch.testing.assert_close(pde(x, y)[0], want, rtol=0, atol=0)

    def test_the_basis_residual(self):
        from epde.integrate.residual_terms import equation_spec
        assert equation_spec(self._equation(False), {'u': 0}, 1).intercept == 0.0
        assert equation_spec(self._equation(True), {'u': 0}, 1).intercept == 5.0


@pytest.fixture
def _keep_the_module_search(monkeypatch):
    """A test that builds its own ``EpdeSearch`` rebinds the runtime globals
    (caches, trajectory manager, verbose, history) and the active config; put
    the module ``fitted`` search's back afterwards, or every later test that
    evaluates it reads another dataset's cache (KeyError on its labels)."""
    import types
    import epde.globals as global_var
    from epde.interface import search_config
    for name, value in list(vars(global_var).items()):
        if not (name.startswith('__') or callable(value) or isinstance(value, types.ModuleType)):
            monkeypatch.setattr(global_var, name, value)
    monkeypatch.setattr(search_config, '_ACTIVE', search_config._ACTIVE)
    vc_modes = dict(global_var.vc_modes_cache)                    # cleared in place
    yield
    global_var.vc_modes_cache.clear()
    global_var.vc_modes_cache.update(vc_modes)


@pytest.mark.usefixtures('solve_on_cpu', '_keep_the_module_search')
class TestTheSystemOrderGuard:

    def test_an_equation_explaining_another_variable_is_a_setup_error(self, solve_on_cpu):
        import epde
        from epde.integrate.heldout import DeepXDEConfigError
        from epde.interface.equation_translator import translate_equation
        from epde.operators.common.fitness import SolverBasedFitness
        from epde.operators.common.objectives import Discrepancy
        search = epde.EpdeSearch(use_solver=False, verbose_params={'show_iter_idx': False},
                                 device='cpu')
        t = np.linspace(0.0, 4 * np.pi, 120)
        _, domain = search.createDomain(t, boundary_width=10, ID=0)
        search.set_preprocessor(default_preprocessor_type='FD', preprocessor_kwargs={})
        _, trajectory = search.createTrajectory({'u': np.sin(t), 'v': np.cos(t)},
                                                domain, cache_id=0)
        search.create_pool(data=[trajectory], max_deriv_order=(1,), data_fun_pow=1)
        system = translate_equation(
            {'u': '1.0 * v{power: 1.0} + 0.0 = du/dx0{power: 1.0}',
             'v': '-1.0 * u{power: 1.0} + 0.0 = dv/dx0{power: 1.0}'},
            search.pool, all_vars=['u', 'v'])
        system.use_default_singleobjective_function()
        system.vals['v'].main_var_to_explain = 'u'                   # the mismatch
        host = SolverBasedFitness(['penalty_coeff', 'error_metric', 'deepxde_config'],
                                  primary=Discrepancy('deepxde'), backend='deepxde')
        host.params = {'penalty_coeff': 0.2, 'error_metric': 'rmse',
                       'deepxde_config': dict(LBFGS_ONLY)}
        with pytest.raises(DeepXDEConfigError, match='main_var_to_explain'):
            host.apply(system, {})



# ============================================================ scale-invariant observation loss
@pytest.mark.usefixtures('solve_on_cpu')
class TestTheScaleInvariantObservationLoss:
    """The observation equation u = u_obs as a cancellation ratio."""

    @staticmethod
    def _error(dxi, mode, y, v):
        import deepxde as dde
        adapter = dxi.DeepXDEAdapter(obs_loss=mode)
        points = np.zeros((len(v), 1))
        bc = adapter._observation_bc(dde, points, np.asarray(v, dtype=np.float32).reshape(-1, 1), 0)
        out = torch.tensor(np.asarray(y, dtype=np.float32).reshape(-1, 1), requires_grad=True)
        return bc.error(None, None, out, 0, len(v)), out, adapter

    def test_live_is_the_bounded_ratio_and_zero_at_zero_over_zero(self, solve_on_cpu):
        err, out, _ = self._error(solve_on_cpu, 'sinv_live', [2.0, -1.0, 0.0, 3.0], [1.0, 1.0, 0.0, -3.0])
        want = torch.tensor([[1.0 / 3.0], [-2.0 / 2.0], [0.0], [6.0 / 6.0]])
        torch.testing.assert_close(err, want, rtol=0, atol=1e-7)
        assert float(err.abs().max()) <= 1.0
        (err ** 2).sum().backward()
        assert torch.isfinite(out.grad).all()

    def test_data_divides_by_the_frozen_data_mass(self, solve_on_cpu):
        err, _, adapter = self._error(solve_on_cpu, 'sinv_data', [2.0, 0.5, 1.0], [1.0, 0.0, -2.0])
        want = torch.tensor([[1.0 / 2.0], [0.0], [3.0 / 4.0]])
        torch.testing.assert_close(err, want, rtol=0, atol=1e-7)
        stats = adapter.last_obs_stats
        assert stats['obs_loss'] == 'sinv_data'
        assert abs(stats['data_mass'][0]['zero_share'] - 1.0 / 3.0) < 1e-12

    def test_mse_keeps_deepxdes_own_error(self, solve_on_cpu):
        err, _, _ = self._error(solve_on_cpu, 'mse', [2.0, 0.5], [1.0, 0.0])
        torch.testing.assert_close(err, torch.tensor([[1.0], [0.5]]), rtol=0, atol=0)

    def test_the_options_it_cannot_combine_with_are_refused(self, solve_on_cpu):
        from epde.integrate.heldout import DeepXDEConfigError
        with pytest.raises(DeepXDEConfigError, match='obs_loss'):
            solve_on_cpu.DeepXDEAdapter(obs_loss='sinv_live', second_order='lm', lr=None,
                                        epochs=0, lbfgs_maxiter=0)

    def test_a_solve_with_it_is_finite(self, fitted, solve_on_cpu):               # noqa: F811
        solution, loss, _ = _solve(solve_on_cpu, fitted, dict(LBFGS_ONLY, input_transform='affine',
                                                              obs_loss='sinv_live'))
        assert np.isfinite(loss) and np.all(np.isfinite(solution))

    def test_the_default_path_is_the_s1_pin(self, fitted, solve_on_cpu):         # noqa: F811
        solution, loss, _ = _solve(solve_on_cpu, fitted, dict(LBFGS_ONLY, input_transform='affine',
                                                              obs_loss='mse'))
        assert _digest(solution, loss) == S1_PIN
