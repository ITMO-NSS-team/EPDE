"""Time marching on route S1 (``epde.integrate.time_marching``).

The plan (tau_min, the window edges, the budget split, the time orders) is
numpy and pinned exactly; the hard continuity of a hand-off is pinned at
float32 bit level; the solves are tiny CPU smokes on the ``fitted`` fixture's
oscillator (``u'' = -u``, so the second-order hand-off path runs), on a coupled
pair and on a space-time advection problem (GeometryXTime windows). DE's last
stage is pinned to BE S1's problem, and DE with a single stage to BE S1,
bitwise; CW's chunk losses are pinned against DeepXDE's own term losses on a
hand-built problem, and CW with its weights dropped from the start to BE S1,
bitwise. The default path is pinned elsewhere (``test_lr_free_solve``) and
must not move.
"""
import contextlib
import importlib.util
import math
import re
import os

import numpy as np
import pytest
import torch

from test_solver_path import _fitted_oscillator, fitted, lbfgs_on_cpu   # noqa: F401

from epde.integrate import time_marching as tm
from epde.integrate.heldout import DeepXDEConfigError, time_split

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))


@pytest.fixture(scope='module', autouse=True)
def _restore_torch_default_device():
    before = torch.get_default_device()
    yield
    torch.set_default_device(before)


@contextlib.contextmanager
def _search_globals_kept():
    """Building an ``EpdeSearch`` rebinds the runtime globals, the active config
    and the operator-parameter singleton; put all three back on exit."""
    import types
    import epde.globals as global_var
    from epde.interface import search_config
    from epde.operators.utils.default_parameter_loader import EvolutionaryParams
    with pytest.MonkeyPatch.context() as mp:
        for name, value in list(vars(global_var).items()):
            if not (name.startswith('__') or callable(value)
                    or isinstance(value, types.ModuleType)):
                mp.setattr(global_var, name, value)
        mp.setattr(search_config, '_ACTIVE', search_config._ACTIVE)
        mp.setattr(EvolutionaryParams, '_container_instances',
                   dict(EvolutionaryParams._container_instances))
        vc_modes = dict(global_var.vc_modes_cache)                # cleared in place
        try:
            yield
        finally:
            global_var.vc_modes_cache.clear()
            global_var.vc_modes_cache.update(vc_modes)


@pytest.fixture(scope='module', autouse=True)
def _leave_the_search_globals_as_found():
    """This module builds its own ``fitted`` search (single-objective). A later
    module that assembles a multi-objective strategy would then read
    single-objective operator parameters ('SortingBasedNeighborSelector is
    missing'; measured: ``test_use_pic_removal`` fails the same way right after
    ``test_solver_path``). Put the search state back when the module is done
    (the module-scoped ``_keep_the_module_search`` of ``test_lr_free_solve``)."""
    with _search_globals_kept():
        yield


@pytest.fixture
def solve_on_cpu(lbfgs_on_cpu, monkeypatch):                     # noqa: F811
    import epde.integrate.deepxde_integration as dxi
    monkeypatch.setattr(dxi, 'DDE_SOLVE_DEVICE', None)
    yield dxi


@pytest.fixture
def problems(monkeypatch):
    """The data object of every ``dde.Model`` built, in order."""
    import deepxde as dde
    built = []
    init = dde.Model.__init__

    def recording(self, data, net):
        built.append(data)
        init(self, data, net)

    monkeypatch.setattr(dde.Model, '__init__', recording)
    return built


@pytest.fixture
def phases(monkeypatch):
    """Every L-BFGS phase of a march (``_March.fit``), in order: its problem,
    the weights handed to it and the weights DeepXDE compiled with."""
    seen = []
    fit = tm._March.fit

    def recording(self, model, weights, cap, record):
        seen.append({'data': model.data, 'weights': [float(w) for w in weights]})
        out = fit(self, model, weights, cap, record)
        seen[-1]['compiled'] = [float(w) for w in model.loss_weights]
        return out

    monkeypatch.setattr(tm._March, 'fit', recording)
    return seen


def _time_derivative(net, x):
    """d net / dt (DeepXDE's last input column) at ``x``, per output."""
    x = x.clone().requires_grad_(True)
    y = net(x)
    return torch.stack([torch.autograd.grad(y[:, v].sum(), x)[0][:, -1]
                        for v in range(y.shape[1])], dim=1)


def _recording_hand_offs(monkeypatch, digest):
    """Every hand-off of a march: window ``k``, its trained predecessor, the
    network handed to window ``k`` and both digests at hand-off time (before
    window ``k`` trains)."""
    handed, real = [], tm._March.hand_off

    def recording(self, nets, k):
        net = real(self, nets, k)
        handed.append({'k': k, 'before': nets[-1], 'net': net,
                       'digests': (digest(nets[-1]), digest(net))})
        return net

    monkeypatch.setattr(tm._March, 'hand_off', recording)
    return handed


def _assert_the_chain(handed, solution, stats, X, first):
    """A march's hand-offs and its field: window k is a warm copy of the
    TRAINED window k-1 (same digest at hand-off time), equal to it at e_k at
    every spatial point (bitwise in float32), and each grid point of ``X``
    (DeepXDE's input columns, time last) is predicted by the window that owns
    its time. ``first``: the first marched window (None: TM-obs's, at
    t_train)."""
    edges = stats['edges']
    if first is None:
        first = edges.index(stats['t_train'])
    assert [h['k'] for h in handed] == list(range(first + 1, len(edges) - 1))
    space = np.unique(X[:, :-1], axis=0) if X.shape[1] > 1 else np.zeros((1, 0))
    for h in handed:
        before, after = h['digests']
        assert before == after                            # a warm start, not a fresh net
        at = torch.as_tensor(np.hstack([space, np.full((len(space), 1), edges[h['k']])]),
                             dtype=torch.float32)
        with torch.no_grad():
            assert torch.equal(h['net'](at), h['before'](at))
    nets = [handed[0]['before']] + [h['net'] for h in handed]
    owner = tm.owners(X[:, -1], edges[1:-1])
    counts, rest = np.divmod(np.bincount(owner, minlength=len(edges) - 1), len(space))
    assert not rest.any()                                 # whole time levels
    sizes = np.diff(stats['edge_levels'])
    np.testing.assert_array_equal(counts[:-1], sizes[:-1])
    assert counts[-1] == sizes[-1] + 1                    # the last window closed
    for k, net in enumerate(nets, start=first):
        rows = np.flatnonzero(owner == k)
        with torch.enable_grad():
            y = net(torch.as_tensor(X[rows], dtype=torch.float32))
        np.testing.assert_array_equal(solution[rows], y.detach().numpy()[:, 0])


#: S1 (G-0000) at fixture size, lr-free, and the marching arms on it
S1 = {'net': [16, 16], 'num_domain': 64, 'num_test': 16, 'epochs': 0, 'lr': None,
      'lbfgs_maxiter': 20, 'train_frac': 0.8, 'loss_weight_mode': 'gradient',
      'input_transform': 'affine', 'pairing_digest': True}
TM = dict(S1, time_marching='full')
TM_OBS = dict(S1, time_marching='tail_obs')
DE = dict(S1, time_marching='expand')
CW = dict(S1, causal_weights='frozen')


def _setup(search):
    import epde.globals as global_var
    system = _fitted_oscillator(search)
    samples = global_var.samples_manager
    key = samples.trajecatoryIDs[0]
    observed = np.asarray(samples.get(('u', (1.0,)))[key]).reshape(-1)
    return system, samples, key, observed


def _solve(dxi, search, config, observed=None, seed=0):
    import deepxde as dde
    system, samples, key, data = _setup(search)
    adapter = dxi.DeepXDEAdapter(**config)
    dde.config.set_random_seed(seed)
    solutions, loss = adapter.solve(system, grids=samples.grids()[key],
                                    data=[data if observed is None else observed],
                                    domain_key=key)
    return np.asarray(solutions[0]), float(loss), adapter


def _held_out(samples, key, train_frac=0.8):
    mask = np.asarray(samples.gFunc('m')[key]).reshape(-1)
    t_inner = np.asarray(samples.grids()[key][0]).reshape(-1)[mask]
    return mask, time_split(t_inner, train_frac)


# ============================================================ the time scale
class TestTauMin:

    def test_a_sinusoid_gives_its_period_over_two_pi(self):
        t = np.linspace(0.0, 20.0, 4001)
        period = 4.0
        x = 3.0 * np.sin(2 * np.pi * t / period) + 1.0
        t_train = time_split(t, 0.8).t_train
        tau, taus = tm.tau_min(t, [x], t_train)
        assert tau == pytest.approx(period / (2 * np.pi), rel=1e-3)
        assert taus == [tau]

    def test_the_shortest_variable_sets_it_and_a_constant_one_imposes_none(self):
        t = np.linspace(0.0, 2.0 * np.pi, 2001)                  # t[1600] = 1.6 pi
        slow, fast = np.sin(t), np.cos(5.0 * t)                   # fast: 4 whole periods
        tau, taus = tm.tau_min(t, [slow, fast, np.full_like(t, 2.0)], t[1600])
        assert tau == taus[1] == pytest.approx(0.2, rel=1e-3)
        assert taus[2] == math.inf

    def test_a_pde_field_uses_every_space_time_point(self):
        t = np.linspace(0.0, 2.5 * np.pi, 1001)                  # t[800] = 2 pi: two periods
        x = np.linspace(0.0, 2 * np.pi, 33)
        T, X = np.meshgrid(t, x, indexing='ij')
        u = np.sin(X) * np.cos(2.0 * T)
        tau, _ = tm.tau_min(t, [u], t[800])
        assert tau == pytest.approx(0.5, rel=2e-3)

    def test_it_is_bitwise_blind_to_the_held_out_levels(self):
        t = np.linspace(0.0, 10.0, 501)
        x = np.sin(1.3 * t) + 0.2 * t
        split = time_split(t, 0.8)
        blank = np.where(split.test, np.nan, x)
        assert tm.tau_min(t, [x], split.t_train) == tm.tau_min(t, [blank], split.t_train)

    def test_snapping_gives_whole_levels_and_at_least_one(self):
        levels = np.linspace(0.0, 1.0, 101)                          # step 0.01
        assert tm.snap(0.0712, levels) == 7
        assert tm.snap(1e-6, levels) == 1


# ============================================================ the windows
class TestTheWindowEdges:

    @pytest.mark.parametrize('n_levels, split, n_delta, first_obs', [
        (100, 73, 8, 10), (100, 73, 7, 0), (1041, 826, 20, 10), (50, 39, 100, 5),
        (60, 47, 1, 3), (100, 73, 8, None), (30, 4, 8, 4)])
    def test_they_cover_the_extent_hold_t_train_and_count_whole_levels(self, n_levels, split,
                                                                     n_delta, first_obs):
        levels = np.linspace(0.0, 5.2, n_levels)
        t_first = None if first_obs is None else levels[first_obs]
        edges = tm.window_edges(levels, levels[split], n_delta, t_first)
        assert all(isinstance(e, int) for e in edges)
        assert edges[0] == 0 and edges[-1] == n_levels - 1          # S1's whole extent
        assert split in edges                                       # no window mixes
        assert all(b > a for a, b in zip(edges[:-1], edges[1:]))
        sizes = np.diff(edges)
        k = edges.index(split)
        # every window but the outermost one on each side of t_train is Delta,
        # and a remainder joins the outermost one instead of standing alone
        assert all(s == n_delta for s in sizes[1:k])
        assert all(s == n_delta for s in sizes[k:-1])
        assert sizes[-1] < 2 * n_delta
        # window 0 holds an observed level (the boundary strip joins it)
        if t_first is not None:
            assert levels[edges[1]] >= t_first

    def test_partial_windows_join_their_neighbours(self):
        levels = np.arange(100.0)
        # left of 73 by 8 from the first observation (10): 73, 65, ..., 17; 17 is
        # only 7 past 10, so it joins window 0. Right: 81, 89, 97; 97 leaves 2.
        assert tm.window_edges(levels, 73.0, 8, 10.0) == [0, 25, 33, 41, 49, 57, 65, 73,
                                                          81, 89, 99]

    def test_t_train_must_be_a_level(self):
        with pytest.raises(DeepXDEConfigError):
            tm.window_edges(np.arange(10.0), 4.5, 2)

    def test_the_budget_is_split_by_length_and_never_rounds_an_integer_up(self):
        assert tm.window_caps([0.25, 0.75], 1.0, 20) == [5, 15]
        caps = tm.window_caps([0.1] * 10 + [0.05], 1.05, 4000)
        assert all(c >= 1 for c in caps) and sum(caps) >= 4000
        assert caps[:10] == [math.ceil(4000 * 0.1 / 1.05)] * 10
        assert tm.window_caps([1e-6, 1.0], 1.0, 20)[0] == 1

    def test_the_points_follow_the_length_with_a_floor(self):
        assert tm.window_points(0.1, 5.2, 4000, 32) == round(4000 * 0.1 / 5.2)
        assert tm.window_points(0.001, 5.2, 4000, 32) == 32

    def test_every_grid_level_is_owned_by_exactly_one_window(self):
        levels = np.linspace(0.0, 5.2, 1041)
        edges = tm.window_edges(levels, levels[826], 20, levels[10])
        t = levels[edges]
        owner = tm.owners(levels, t[1:-1])
        assert owner.min() == 0 and owner.max() == len(edges) - 2
        counts = np.bincount(owner)
        assert counts.sum() == len(levels)
        np.testing.assert_array_equal(counts[:-1], np.diff(edges)[:-1])  # [e_k, e_{k+1})
        assert counts[-1] == np.diff(edges)[-1] + 1                       # the last closed
        for k, e in enumerate(t[:-1]):
            assert owner[edges[k]] == k                                   # left-closed


class TestTheTimeOrders:

    @pytest.mark.parametrize('text, want', [
        ('-1.0 * u{power: 1.0} + 0.0 = du/dx0{power: 1.0}', [1]),
        ('-1.0 * u{power: 1.0} + 0.0 = d^2u/dx0^2{power: 1.0}', [2]),
        ('-1.0 * d^2u/dx0^2{power: 1.0} + 0.0 = du/dx0{power: 1.0}', [2]),     # a term counts
        ('-1.0 * u{power: 1.0} + 0.0 = u{power: 1.0} * du/dx0{power: 1.0}', [1])])
    def test_they_are_read_from_the_candidate(self, text, want, fitted):           # noqa: F811
        from epde.interface.equation_translator import translate_equation
        system = translate_equation(text, fitted.pool, all_vars=['u'])
        eq = system.vals['u']
        eq.main_var_to_explain = 'u'
        assert tm.time_orders([eq], ['u'], 1) == want


# ============================================================ the hand-off
def _fnn(sizes, seed):
    import deepxde as dde
    dde.config.set_random_seed(seed)
    return dde.nn.FNN(sizes, 'tanh', 'Glorot normal')


@pytest.mark.usefixtures('solve_on_cpu')
class TestTheHardContinuity:
    """At t = e_k the window IS its predecessor, and for m_v = 2 so is its
    first time derivative -- bitwise in float32."""

    E, LENGTH = 0.7, 0.3

    def _derivative(self, net, x, column, var):
        x = x.clone().requires_grad_(True)
        y = net(x)
        return torch.autograd.grad(y[:, var].sum(), x)[0][:, column]

    def test_an_ode_window_starts_from_its_predecessor(self):
        P = tm._frozen(_fnn([1, 8, 2], 1))
        net = _fnn([1, 8, 2], 2)
        ic = tm.net_edge_state(P, self.E, 2, 0)
        net.apply_output_transform(tm.hard_ic_transform(ic, self.E, self.LENGTH, [1, 2], 0))
        x = torch.full((5, 1), self.E)
        one = torch.full((1, 1), self.E)
        with torch.no_grad():
            torch.testing.assert_close(net(x), P(one).expand(5, 2), rtol=0, atol=0)
        torch.testing.assert_close(self._derivative(net, x, 0, 1),
                                   self._derivative(P, one, 0, 1).expand(5), rtol=0, atol=0)
        # away from the edge the window is its own network's
        later = torch.full((1, 1), self.E + 0.1)
        assert not torch.equal(net(later), P(later))

    def test_a_pde_window_follows_its_predecessor_in_space(self):
        P = tm._frozen(_fnn([2, 8, 2], 3))                         # columns [x, t]
        net = _fnn([2, 8, 2], 4)
        ic = tm.net_edge_state(P, self.E, 2, 1)
        net.apply_output_transform(tm.hard_ic_transform(ic, self.E, self.LENGTH, [2, 1], 1))
        x = torch.stack([torch.linspace(-1.0, 1.0, 7), torch.full((7,), self.E)], dim=1)
        torch.testing.assert_close(net(x), P(x), rtol=0, atol=0)
        # d/dt (m = 2 for var 0) and d/dx (the edge state is differentiable in x)
        for column, var in ((1, 0), (0, 0), (0, 1)):
            torch.testing.assert_close(self._derivative(net, x, column, var),
                                       self._derivative(P, x, column, var),
                                       rtol=0, atol=2e-7)

    def test_the_observed_state_is_a_c2_spline_extended_linearly(self):
        nodes = np.linspace(0.0, 1.0, 9)
        y = np.sin(3.0 * nodes)
        state = tm._SplineEdgeState([nodes], [[y]], torch.float64, 'cpu')
        at_nodes = state(torch.as_tensor(nodes)[:, None])[0][:, 0]
        np.testing.assert_allclose(at_nodes.numpy(), y, rtol=0, atol=1e-12)
        x = torch.linspace(-0.3, 1.3, 161, dtype=torch.float64)[:, None].requires_grad_(True)
        s = state(x)[0][:, 0]
        ds = torch.autograd.grad(s.sum(), x, create_graph=True)[0][:, 0]
        d2s = torch.autograd.grad(ds.sum(), x)[0][:, 0]
        outside = ((x[:, 0] < 0.0) | (x[:, 0] > 1.0)).detach().numpy()
        assert float(d2s[torch.as_tensor(outside)].abs().max()) == 0.0   # linear past the ends
        inside = x.detach()[:, 0].numpy()[~outside]
        err = s.detach().numpy()[~outside] - np.sin(3.0 * inside)
        assert float(np.abs(err).max()) < 1e-2

    def test_the_backward_differences_are_fornbergs(self):
        h = 0.1
        w = tm.fd_weights(0.0, [-2 * h, -h, 0.0], 1)
        np.testing.assert_allclose(w[1], [1 / (2 * h), -4 / (2 * h), 3 / (2 * h)], rtol=1e-12)
        t = np.linspace(0.0, 1.0, 101)
        state = tm.observed_edge_state(t, [np.sin(t)], t[80], 2)
        assert state[0][0] == np.sin(t[80])                               # the value itself
        assert float(state[1][0]) == pytest.approx(np.cos(t[80]), abs=1e-4)


# ============================================================ the causal weights
def _tiny_problem(obs_loss='mse'):
    """``y' = y`` on [0, 2], observations of sin(t) on [0, 1.4] through the
    adapter's own observation term: one residual and one observation term,
    built by hand, compiled so DeepXDE's own term losses can be read."""
    import deepxde as dde
    from epde.integrate.deepxde_integration import DeepXDEAdapter
    dde.config.set_random_seed(0)

    def pde(x, y):
        return [dde.grad.jacobian(y, x, i=0, j=0) - y]

    t_obs = np.linspace(0.0, 1.4, 29)[:, None]
    bc = DeepXDEAdapter(obs_loss=obs_loss)._observation_bc(dde, t_obs, np.sin(t_obs), 0)
    data = dde.data.PDE(dde.geometry.TimeDomain(0.0, 2.0), pde, [bc], num_domain=50,
                        num_boundary=0, num_test=10)
    net = dde.nn.FNN([1, 8, 1], 'tanh', 'Glorot normal')
    model = dde.Model(data, net)
    model.compile('L-BFGS')
    return data, net, model


@pytest.mark.usefixtures('solve_on_cpu')
class TestTheCausalWeights:
    """w_i = exp(-eps sum_{j<i} L_j), the per-chunk losses L_j, and the problem
    they weigh: S1's own points, each error times sqrt(w) of its chunk."""

    EDGES = np.linspace(0.0, 2.0, 5)                                  # four chunks of 0.5

    def test_they_start_at_one_never_rise_and_follow_the_formula(self):
        L = np.random.default_rng(0).exponential(size=40)
        eps = 40 / L.sum()
        w = tm.causal_weights(L, eps)
        assert w[0] == 1.0 and np.all(np.diff(w) <= 0.0)
        np.testing.assert_allclose(w[1:], np.exp(-eps * np.cumsum(L)[:-1]), rtol=1e-12)
        assert np.all(tm.causal_weights(np.zeros(5), 3.0) == 1.0)      # all fitted: w == 1
        assert np.all(tm.causal_weights(L, 0.0) == 1.0)

    def test_the_chunk_count_is_the_extent_over_the_raw_tau_min(self):
        # Lorenz: T = 5.2, tau_min = 0.1088513 (47.77 chunks); the snapped
        # 22-level Delta would give 47
        assert tm.chunk_count(5.2, 0.1088513) == 48
        assert tm.chunk_count(1.0, 0.0101) == 99
        assert tm.chunk_count(1.0, 5.0) == 1                          # longer than the extent

    def test_one_chunk_is_s1s_own_loss(self):
        from epde.integrate.deepxde_integration import DeepXDEAdapter
        data, net, model = _tiny_problem()
        weights = [0.7, 3.0]
        scale = tm._CausalScale(np.array([0.0, 2.0]), torch.float32, 'cpu')
        L = tm.chunk_losses(net, data, weights, scale)
        assert L.shape == (1,)
        assert L[0] == pytest.approx(DeepXDEAdapter._reported_loss(model, weights), rel=1e-6)

    def test_the_chunks_split_every_term_by_time(self):
        from epde.integrate.deepxde_integration import DeepXDEAdapter
        data, net, model = _tiny_problem()
        scale = tm._CausalScale(self.EDGES, torch.float32, 'cpu')
        whole = DeepXDEAdapter._unweighted_losses(model).detach()
        t = torch.as_tensor(data.train_x[:, -1])
        n_obs = data.num_bcs[0]
        for term, rows in ((0, slice(n_obs, None)), (1, slice(0, n_obs))):
            L = tm.chunk_losses(net, data, [float(i == term) for i in range(2)], scale)
            counts = np.bincount(scale.chunk(t[rows]).numpy(), minlength=4)
            assert float(np.sum(counts * L) / counts.sum()) == pytest.approx(float(whole[term]),
                                                                            rel=1e-5)
        assert counts[3] == 0 and L[3] == 0.0              # no observation after t = 1.4

    @pytest.mark.parametrize('obs_loss', ['mse', 'sinv_live'])
    def test_the_weighted_problem_scales_every_point_and_moves_none(self, obs_loss):
        import deepxde as dde
        from epde.integrate.deepxde_integration import DeepXDEAdapter
        data, net, model = _tiny_problem(obs_loss)
        scale = tm._CausalScale(self.EDGES, torch.float32, 'cpu')
        weighted = tm.causal_problem(data, scale)
        assert weighted is not data and weighted.pde is not data.pde
        for attr in ('train_x', 'train_x_all', 'train_x_bc', 'test_x'):
            assert getattr(weighted, attr) is getattr(data, attr)
        for ours, s1s in zip(weighted.bcs, data.bcs, strict=True):
            assert np.array_equal(ours.points, s1s.points) and ours.values is s1s.values
        wmodel = dde.Model(weighted, net)
        wmodel.compile('L-BFGS')
        scale.set(np.ones(4))                         # w == 1: S1's own term losses, bitwise
        assert torch.equal(DeepXDEAdapter._unweighted_losses(wmodel),
                           DeepXDEAdapter._unweighted_losses(model))
        w = np.array([1.0, 0.5, 0.25, 0.0])
        scale.set(w)
        t = torch.as_tensor(data.train_x[:, -1])
        n_obs = data.num_bcs[0]
        losses = DeepXDEAdapter._unweighted_losses(wmodel).detach()
        for term, rows in ((0, slice(n_obs, None)), (1, slice(0, n_obs))):
            L = tm.chunk_losses(net, data, [float(i == term) for i in range(2)], scale)
            counts = np.bincount(scale.chunk(t[rows]).numpy(), minlength=4)
            assert float(losses[term]) == pytest.approx(np.sum(counts * w * L) / counts.sum(),
                                                        rel=1e-5)

    def test_the_weighted_terms_bin_by_the_time_column(self):
        """Inputs are ``[space..., t]``: a point's weight follows its TIME,
        whatever its space coordinate. On an ODE column 0 IS time, so only a
        space-time input can tell the two apart."""
        import deepxde as dde
        scale = tm._CausalScale(np.array([0.0, 1.0, 2.0]), torch.float32, 'cpu')
        scale.set([1.0, 0.0])                         # w = 1 before t = 1, 0 after
        x = torch.tensor([[1.5, 0.2], [0.2, 1.5]])    # (x, t): early t at a large x, and back
        y = torch.tensor([[3.0], [3.0]])
        residual = tm._causal_pde(lambda x, y: [y + 0.0], scale)(x, y)[0]
        np.testing.assert_array_equal(residual.numpy()[:, 0], [3.0, 0.0])
        bc = dde.icbc.PointSetBC(x.numpy(), np.zeros((2, 1), dtype=np.float32), component=0)
        error = tm._causal_bc(bc, scale).error(x.numpy(), x, y, 0, 2)
        np.testing.assert_array_equal(error.numpy()[:, 0], [3.0, 0.0])
        # the chunk losses: x in [0, 0.4] would put every point in chunk 0
        dde.config.set_random_seed(0)
        geom = dde.geometry.GeometryXTime(dde.geometry.Interval(0.0, 0.4),
                                          dde.geometry.TimeDomain(0.0, 2.0))
        data = dde.data.TimePDE(geom, lambda x, y: y, [], num_domain=64, num_boundary=0,
                                num_initial=0)
        net = dde.nn.FNN([2, 8, 1], 'tanh', 'Glorot normal')
        L = tm.chunk_losses(net, data, [1.0], tm._CausalScale(self.EDGES, torch.float32, 'cpu'))
        assert L.shape == (4,) and np.all(L > 0.0)

    def test_the_scale_never_rises_in_time_and_bins_left_closed(self):
        scale = tm._CausalScale(self.EDGES, torch.float32, 'cpu')
        L = np.random.default_rng(1).exponential(size=4)
        scale.set(tm.causal_weights(L, 4 / L.sum()))
        s = scale(torch.linspace(0.0, 2.0, 401)).reshape(-1).numpy()
        assert s[0] == 1.0 and np.all(np.diff(s) <= 0.0)
        np.testing.assert_array_equal(scale.chunk(torch.tensor([0.0, 0.5, 1.999, 2.0])).numpy(),
                                      [0, 1, 3, 3])                 # [c_j, c_{j+1}), last closed


# ============================================================ the solves
@pytest.mark.usefixtures('solve_on_cpu')
class TestTheMarchingSolves:

    def test_tm_shares_s1s_start_and_weights_and_is_finite(self, fitted, solve_on_cpu):  # noqa: F811
        _, _, s1 = _solve(solve_on_cpu, fitted, S1)
        solution, loss, adapter = _solve(solve_on_cpu, fitted, TM)
        assert np.isfinite(solution).all() and np.isfinite(loss)
        assert adapter.last_pairing == s1.last_pairing             # theta0 AND the weights
        stats = adapter.last_march_stats
        assert stats['mode'] == 'full' and stats['time_orders'] == {'u': 2}
        assert stats['n_windows'] == len(stats['windows']) == len(stats['edges']) - 1
        assert stats['t_train'] in stats['edges']
        assert stats['loss_s1'] == loss and np.isfinite(stats['loss_windows'])
        assert [w['cap'] for w in stats['windows']] == stats['caps']
        assert all(w['iters'] is not None and w['iters'] <= w['cap'] for w in stats['windows'])
        assert stats['windows'][0]['observed_points'] > 0
        tail = [w for w in stats['windows'] if w['t0'] >= stats['t_train']]
        assert tail and all(w['observed_points'] == 0 for w in tail)
        assert adapter.last_lbfgs_exit['iterations'] == sum(w['iters'] for w in stats['windows'])

    def test_tm_obs_marches_the_tail_from_the_observed_state(self, fitted, solve_on_cpu):  # noqa: F811
        solution, loss, adapter = _solve(solve_on_cpu, fitted, TM_OBS)
        _, samples, key, observed = _setup(fitted)
        mask, split = _held_out(samples, key)
        t = np.asarray(samples.grids()[key][0]).reshape(-1)
        assert np.isfinite(loss)
        assert np.isfinite(solution[t >= split.t_train]).all()
        np.testing.assert_array_equal(solution[mask][split.train & (t[mask] < split.t_train)],
                                      observed[split.train & (t[mask] < split.t_train)])
        assert np.isnan(solution[~mask & (t < split.t_train)]).all()       # no data, no window
        stats = adapter.last_march_stats
        assert stats['windows'][0]['t0'] == stats['t_train']
        assert all(w['observed_points'] == 0 for w in stats['windows'])
        # the first tail window starts exactly at the observed value
        at = np.flatnonzero(mask)[np.flatnonzero(t[mask] == split.t_train)]
        assert solution[at] == pytest.approx(observed[t[mask] == split.t_train], rel=1e-6)

    @pytest.mark.parametrize('config', [TM, TM_OBS, DE, CW], ids=['TM', 'TM-obs', 'DE', 'CW'])
    def test_every_phase_runs_on_s1s_frozen_weights(self, config, fitted, solve_on_cpu,
                                                    phases):                           # noqa: F811
        """R1: the weights are probed ONCE, on S1's full problem, and every
        window / stage / restart compiles with them (the residual's alone where
        the problem holds no observation)."""
        _, _, s1 = _solve(solve_on_cpu, fitted, S1)
        assert phases == []                               # S1 has no march
        w = s1.last_pairing['loss_weights']
        assert len(w) == 2                                # one residual, one observation term
        _, _, adapter = _solve(solve_on_cpu, fitted, config)
        assert adapter.last_march_stats['loss_weights'] == w
        assert len(phases) == len(adapter.last_march_stats['windows'])
        for phase in phases:
            want = w if phase['data'].bcs else w[:1]
            assert phase['weights'] == phase['compiled'] == want
        observed = [bool(p['data'].bcs) for p in phases]
        if config is TM_OBS:
            assert not any(observed)
        else:
            assert observed[0]                            # S1's observation weight in use

    @pytest.mark.parametrize('config', [TM, TM_OBS], ids=['TM', 'TM-obs'])
    def test_each_window_is_handed_off_from_its_trained_predecessor(
            self, config, fitted, solve_on_cpu, monkeypatch):                        # noqa: F811
        """The chain itself: window k is a warm copy of the TRAINED window k-1,
        held to it at e_k (the value and, m = 2 here, d/dt -- bitwise in
        float32), and every grid level is predicted by the window owning it."""
        handed = _recording_hand_offs(monkeypatch, solve_on_cpu.DeepXDEAdapter._param_digest)
        solution, _, adapter = _solve(solve_on_cpu, fitted, config)
        stats = adapter.last_march_stats
        assert stats['time_orders'] == {'u': 2}
        _, samples, key, _ = _setup(fitted)
        X = solve_on_cpu._input_columns(samples.grids()[key])
        _assert_the_chain(handed, solution, stats, X, first=0 if config is TM else None)
        for h in handed:
            e = torch.full((1, 1), stats['edges'][h['k']], dtype=torch.float32)
            assert torch.equal(_time_derivative(h['net'], e), _time_derivative(h['before'], e))

    @pytest.mark.parametrize('config', [TM, TM_OBS, DE, CW], ids=['TM', 'TM-obs', 'DE', 'CW'])
    def test_the_windows_leave_s1s_observation_record_alone(self, config, fitted,
                                                           solve_on_cpu):              # noqa: F811
        """A window's observation terms are built through the adapter's own
        ``_observation_bc``, which appends to ``last_obs_stats``; the record a
        row carries must still be S1's (one entry per observed variable)."""
        _, _, s1 = _solve(solve_on_cpu, fitted, dict(S1, obs_loss='sinv_live'))
        _, _, adapter = _solve(solve_on_cpu, fitted, dict(config, obs_loss='sinv_live'))
        assert len(s1.last_obs_stats['data_mass']) == 1
        assert adapter.last_obs_stats == s1.last_obs_stats

    def test_de_grows_s1s_network_over_the_edges_and_ends_on_s1s_problem(
            self, fitted, solve_on_cpu, problems):                                   # noqa: F811
        _, _, s1 = _solve(solve_on_cpu, fitted, S1)
        assert len(problems) == 1                         # S1 builds one problem: data_obj
        s1_problem = problems.pop()
        solution, loss, adapter = _solve(solve_on_cpu, fitted, DE)
        stats = adapter.last_march_stats
        assert np.isfinite(solution).all() and np.isfinite(loss)     # one net, every level
        assert adapter.last_pairing == s1.last_pairing             # theta0 AND the weights
        assert stats['mode'] == 'expand' and 'loss_windows' not in stats
        stages, edges = stats['windows'], stats['edges']
        assert stats['t_train'] in edges
        assert [w['stage'] for w in stages] == list(range(1, len(edges)))
        assert all(w['t0'] == edges[0] for w in stages)
        assert [w['t1'] for w in stages] == edges[1:]                # the last: S1's extent
        # the budget: each curriculum stage the share of the time it adds; the
        # last stage (S1's problem) S1's whole budget
        shares = tm.window_caps(np.diff(edges), edges[-1] - edges[0], DE['lbfgs_maxiter'])
        assert [w['cap'] for w in stages] == stats['caps'] == (
            shares[:-1] + [DE['lbfgs_maxiter']])
        assert all(w['iters'] is not None and w['iters'] <= w['cap'] for w in stages)
        assert adapter.last_lbfgs_exit['iterations'] == sum(w['iters'] for w in stages)
        # the last stage IS S1's problem: the probe's own data object, and the
        # same points as S1's solve (geometry, collocation, observations)
        assert len(problems) == len(stages) + 1
        assert problems[-1] is problems[0]
        for attr in ('train_x_all', 'train_x_bc', 'train_x', 'test_x'):
            np.testing.assert_array_equal(getattr(problems[-1], attr), getattr(s1_problem, attr))
        assert problems[-1].num_domain == s1_problem.num_domain == DE['num_domain']
        assert [w['s1_problem'] for w in stages] == [False] * (len(stages) - 1) + [True]
        assert loss == stats['loss_s1'] == stages[-1]['loss']
        # every earlier stage: [t_lo, e_s], S1's point density, train observations t <= e_s
        _, samples, key, _ = _setup(fitted)
        mask, split = _held_out(samples, key)
        t_inner = np.asarray(samples.grids()[key][0]).reshape(-1)[mask]
        total = edges[-1] - edges[0]
        for data, stage in zip(problems[1:-1], stages[:-1]):
            t1 = stage['t1']
            assert (data.geom.t0, data.geom.t1) == (edges[0], t1)
            assert data.num_domain == stage['num_domain'] == max(
                1, round(DE['num_domain'] * (t1 - edges[0]) / total))
            assert data.train_x_all[:, -1].min() >= edges[0] and data.train_x_all[:, -1].max() <= t1
            n_obs = int(np.sum(split.train & (t_inner <= t1)))
            assert n_obs > 0 and stage['observed_points'] == n_obs
            assert all(len(bc.points) == n_obs and bc.points[:, -1].max() <= t1
                       for bc in data.bcs)
            assert len(data.bcs) == 1                               # one per variable

    def test_de_with_a_single_stage_is_s1_bitwise(self, fitted, solve_on_cpu, monkeypatch):  # noqa: F811
        """The mechanism is the expansion and nothing else: with one stage
        (the whole extent) DE runs S1's problem from S1's start on S1's budget."""
        first, s1_loss, s1 = _solve(solve_on_cpu, fitted, S1)
        monkeypatch.setattr(tm, 'window_edges', lambda levels, *a, **k: [0, len(levels) - 1])
        second, de_loss, de = _solve(solve_on_cpu, fitted, DE)
        assert de.last_march_stats['n_windows'] == 1
        np.testing.assert_array_equal(first, second)
        assert s1_loss == de_loss
        assert de.last_lbfgs_exit == {'reason': s1.last_lbfgs_exit['reason'],
                                      'iterations': s1.last_lbfgs_exit['iterations']}

    def test_cw_weighs_s1s_own_problem_from_s1s_start(self, fitted, solve_on_cpu, problems,
                                                      monkeypatch):                   # noqa: F811
        _, _, s1 = _solve(solve_on_cpu, fitted, S1)
        s1_problem = problems.pop()
        digests = []
        measure = tm.chunk_losses

        def recording(net, *args, **kwargs):
            digests.append(solve_on_cpu.DeepXDEAdapter._param_digest(net))
            return measure(net, *args, **kwargs)

        monkeypatch.setattr(tm, 'chunk_losses', recording)
        solution, loss, adapter = _solve(solve_on_cpu, fitted, CW)
        stats = adapter.last_march_stats
        assert np.isfinite(solution).all() and np.isfinite(loss)     # one net, every level
        assert adapter.last_pairing == s1.last_pairing             # theta0 AND the weights
        assert stats['mode'] == 'causal_frozen' and 'loss_windows' not in stats
        # M = round(T / tau_min) equal chunks of S1's whole extent (raw tau_min)
        _, samples, key, _ = _setup(fitted)
        levels = np.unique(np.asarray(samples.grids()[key][0]))
        extent = levels[-1] - levels[0]
        assert stats['n_chunks'] == max(1, round(extent / stats['tau_min'])) > 1
        assert stats['chunk_length'] == extent / stats['n_chunks']
        edges = np.asarray(stats['chunk_edges'])
        assert (edges[0], edges[-1]) == (levels[0], levels[-1])
        assert len(edges) == stats['n_chunks'] + 1
        np.testing.assert_allclose(np.diff(edges), extent / stats['n_chunks'], rtol=1e-12)
        # eps from the chunk losses at the Glorot init: S1's theta0
        assert digests[0] == s1.last_pairing['theta0']
        assert stats['eps'] == stats['n_chunks'] / np.sum(stats['chunk_loss0'])
        assert len(stats['chunk_loss_final']) == stats['n_chunks']
        # the budget: restarts of at most CAUSAL_RESTART, the TOTAL within lbfgs_maxiter
        restarts = stats['windows']
        assert [r['restart'] for r in restarts] == list(range(len(restarts)))
        assert all(r['iters'] <= r['cap'] <= tm.CAUSAL_RESTART for r in restarts)
        assert sum(r['iters'] for r in restarts) <= CW['lbfgs_maxiter']
        assert adapter.last_lbfgs_exit['iterations'] == sum(r['iters'] for r in restarts)
        # S1's own points throughout: its data object or the weighted shallow copy
        cw_problem, trained = problems[0], problems[1:]
        for attr in ('train_x_all', 'train_x_bc', 'train_x', 'test_x'):
            np.testing.assert_array_equal(getattr(cw_problem, attr), getattr(s1_problem, attr))
        assert trained and all(d is cw_problem or (d.train_x is cw_problem.train_x
                                                   and d.test_x is cw_problem.test_x)
                               for d in trained)
        assert loss == stats['loss_s1']                            # S1's objective, scored last

    def test_cw_holds_its_weights_through_a_restart(self, fitted, solve_on_cpu,
                                                    monkeypatch):                    # noqa: F811
        monkeypatch.setattr(tm, 'CAUSAL_RESTART', 5)
        log = []
        read, fit = tm._CausalScale.__call__, tm._March.fit

        def reading(self, t):
            log.append(self.root.clone())
            return read(self, t)

        def restart(self, *args, **kwargs):
            log.append('restart')
            return fit(self, *args, **kwargs)

        measured, measure = [], tm.chunk_losses

        def measuring(*args, **kwargs):
            measured.append(measure(*args, **kwargs))
            return measured[-1]

        monkeypatch.setattr(tm._CausalScale, '__call__', reading)
        monkeypatch.setattr(tm._March, 'fit', restart)
        monkeypatch.setattr(tm, 'chunk_losses', measuring)
        solution, _, adapter = _solve(solve_on_cpu, fitted, CW)
        assert np.isfinite(solution).all()
        stats = adapter.last_march_stats
        restarts = stats['windows']
        assert [r['cap'] for r in restarts] == [5] * 4 and not any(r['plain'] for r in restarts)
        phases = []
        for item in log:
            if isinstance(item, str):
                phases.append([])
            else:
                phases[-1].append(item)
        assert len(phases) == len(restarts)
        for roots, record in zip(phases, restarts):
            assert roots                                   # the weighted terms were read
            assert all(torch.equal(r, roots[0]) for r in roots)    # constant in a restart
            w = roots[0].double().numpy() ** 2
            assert w[0] == 1.0 and np.all(np.diff(w) <= 0.0)       # never rising in time
            assert w.min() == pytest.approx(record['min_w'], rel=1e-5, abs=1e-30)
            assert record['front'] == (int(np.argmax(w < 0.5)) if (w < 0.5).any() else len(w))
        # and re-read between restarts, from the chunk losses the training left
        assert not torch.equal(phases[0][0], phases[1][0])
        # restart r weighs with ITS chunk losses (read r: the init, then one
        # after each restart) but the INIT's eps
        assert len(measured) > len(restarts)              # the last read: the final record
        assert stats['eps'] == stats['n_chunks'] / float(np.sum(measured[0]))
        for roots, L in zip(phases, measured):
            want = torch.as_tensor(np.sqrt(tm.causal_weights(L, stats['eps'])),
                                   dtype=roots[0].dtype)
            assert torch.equal(roots[0], want)

    def test_cw_with_its_weights_dropped_from_the_start_is_s1_bitwise(
            self, fitted, solve_on_cpu, monkeypatch):                                 # noqa: F811
        """w == 1 is S1's own objective, on S1's own problem, for the whole
        budget: measuring the chunk losses changes nothing."""
        first, s1_loss, s1 = _solve(solve_on_cpu, fitted, S1)
        monkeypatch.setattr(tm, 'CAUSAL_DONE', 0.0)            # every chunk counts as fitted
        second, cw_loss, cw = _solve(solve_on_cpu, fitted, CW)
        restarts = cw.last_march_stats['windows']
        assert len(restarts) == 1 and restarts[0]['plain']
        assert restarts[0]['cap'] == CW['lbfgs_maxiter'] and cw.last_march_stats['plain_from'] == 0
        np.testing.assert_array_equal(first, second)
        assert s1_loss == cw_loss
        assert cw.last_lbfgs_exit == {'reason': s1.last_lbfgs_exit['reason'],
                                      'iterations': s1.last_lbfgs_exit['iterations']}

    def test_once_every_chunk_is_fitted_cw_spends_the_rest_on_s1s_objective(
            self, fitted, solve_on_cpu, monkeypatch, problems):                       # noqa: F811
        monkeypatch.setattr(tm, 'CAUSAL_RESTART', 5)
        real, calls = tm.causal_weights, []

        def fitted_after_one_restart(losses, eps):
            calls.append(eps)
            return real(losses, eps) if len(calls) == 1 else np.ones(len(losses))

        monkeypatch.setattr(tm, 'causal_weights', fitted_after_one_restart)
        solution, loss, adapter = _solve(solve_on_cpu, fitted, CW)
        stats = adapter.last_march_stats
        assert np.isfinite(solution).all()
        first, second = stats['windows']                       # the plain phase is the last
        assert not first['plain'] and second['plain'] and second['min_w'] == 1.0
        assert first['cap'] == 5 and second['cap'] == CW['lbfgs_maxiter'] - first['iters']
        assert stats['plain_from'] == 1
        # the weighted restart on the shallow copy, the plain one on S1's data object
        s1_problem, weighted, plain = problems
        assert plain is s1_problem and weighted is not s1_problem
        assert weighted.train_x is s1_problem.train_x
        assert loss == stats['loss_s1'] == second['loss']

    @pytest.mark.parametrize('config', [TM, TM_OBS, DE, CW], ids=['TM', 'TM-obs', 'DE', 'CW'])
    def test_the_result_never_reads_the_held_out_data(self, config, fitted, solve_on_cpu):  # noqa: F811
        _, samples, key, observed = _setup(fitted)
        _, split = _held_out(samples, key)
        blank = np.where(split.test, np.nan, observed)
        first = _solve(solve_on_cpu, fitted, config)
        second = _solve(solve_on_cpu, fitted, config, observed=blank)
        np.testing.assert_array_equal(first[0], second[0])
        assert first[1] == second[1]

    def test_the_default_path_carries_no_march_record(self, fitted, solve_on_cpu):   # noqa: F811
        _, _, adapter = _solve(solve_on_cpu, fitted, S1)
        assert adapter.last_march_stats is None


@pytest.fixture
def _own_search():
    """A test that builds its own ``EpdeSearch``; the module's ``fitted``
    search is put back afterwards."""
    with _search_globals_kept():
        yield


@pytest.mark.usefixtures('solve_on_cpu', '_own_search')
class TestACoupledSystem:
    """u' = v, v' = -u: every variable shares the windows (CW: the chunks), one
    net, one output column per variable (as Lorenz, LV and the cylinder wake)."""

    @pytest.mark.parametrize('config', [TM, TM_OBS, DE, CW], ids=['TM', 'TM-obs', 'DE', 'CW'])
    def test_it_marches_every_variable_through_the_same_windows(self, config, solve_on_cpu):
        import deepxde as dde
        import epde
        import epde.globals as global_var
        from epde.interface.equation_translator import translate_equation
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
        for var in ('u', 'v'):
            system.vals[var].main_var_to_explain = var
        samples = global_var.samples_manager
        key = samples.trajecatoryIDs[0]
        observed = [np.asarray(samples.get((v, (1.0,)))[key]).reshape(-1) for v in ('u', 'v')]
        adapter = solve_on_cpu.DeepXDEAdapter(**config)
        dde.config.set_random_seed(0)
        solutions, loss = adapter.solve(system, grids=samples.grids()[key], data=observed,
                                        domain_key=key)
        stats = adapter.last_march_stats
        assert len(solutions) == 2 and np.isfinite(loss)
        t_full = np.asarray(samples.grids()[key][0]).reshape(-1)
        for solution in solutions:
            assert np.isfinite(np.asarray(solution)[t_full >= stats['t_train']]).all()
        # period 2 pi; the train window holds ~1.3 periods, so not exactly 1
        assert stats['tau_min'] == pytest.approx(1.0, rel=0.1)
        assert set(stats['tau']) == {'u', 'v'}
        if config is CW:                              # both terms of both variables, chunked
            assert len(stats['loss_weights']) == 4 and stats['n_chunks'] > 1
            assert len(stats['chunk_loss0']) == stats['n_chunks']
            return
        assert stats['time_orders'] == {'u': 1, 'v': 1}
        if config is DE:                              # every stage holds observations
            assert all(w['observed_points'] > 0 for w in stats['windows'])
            assert stats['windows'][-1]['s1_problem']


# ============================================================ a space-time problem
@pytest.fixture(scope='class')
def advection():
    """``u = sin(x - 12 t)`` on a 41 x 32 space-time grid (boundary strips of
    3) with the candidate ``u_t = c * u_x`` fitted on it: DeepXDE's inputs are
    ``[x, t]``, so every window or stage is a GeometryXTime and a space / time
    column mix-up shows (on an ODE column 0 IS time). tau_min ~ 1/12 gives
    3-level windows, so even TM-obs's tail holds several. Its own search, its
    globals put back when the class is done."""
    import types
    import epde
    import epde.globals as global_var
    from epde.interface.equation_translator import translate_equation
    from epde.operators.common.coeff_calculation import LinRegBasedCoeffsEquation
    with _search_globals_kept():
        t = np.linspace(0.0, 1.0, 41)
        x = np.linspace(0.0, 2 * np.pi, 32, endpoint=False)
        T, X = np.meshgrid(t, x, indexing='ij')
        search = epde.EpdeSearch(use_solver=False, verbose_params={'show_iter_idx': False},
                                 device='cpu')
        _, domain = search.createDomain((T, X), boundary_width=(3, 3), ID=0)
        search.set_preprocessor(default_preprocessor_type='FD', preprocessor_kwargs={})
        _, trajectory = search.createTrajectory({'u': np.sin(X - 12.0 * T)}, domain,
                                                cache_id=0)
        search.create_pool(data=[trajectory], max_deriv_order=(1, 1), additional_tokens=[])
        system = translate_equation('1.0 * du/dx1{power: 1.0} + 0.0 = du/dx0{power: 1.0}',
                                    search.pool, all_vars=['u'])
        eq = system.vals['u']
        eq.main_var_to_explain = 'u'
        system.use_default_singleobjective_function()
        support = np.ones(len(eq.structure))
        support[-1] = 0.0
        eq.weights_internal = support
        eq.weights_internal_evald = True
        LinRegBasedCoeffsEquation().apply(eq, {})
        samples = global_var.samples_manager
        key = samples.trajecatoryIDs[0]
        grids = samples.grids()[key]
        mask = np.asarray(samples.gFunc('m')[key]).reshape(-1)
        t_full = np.asarray(grids[0]).reshape(-1)
        yield types.SimpleNamespace(
            system=system, grids=grids, key=key, mask=mask, t=t_full,
            observed=np.asarray(samples.get(('u', (1.0,)))[key]).reshape(-1),
            split=time_split(t_full[mask], 0.8))


def _solve_pde(dxi, problem, config, seed=0):
    import deepxde as dde
    adapter = dxi.DeepXDEAdapter(**dict(config, **PDE))
    dde.config.set_random_seed(seed)
    solutions, loss = adapter.solve(problem.system, grids=problem.grids,
                                    data=[problem.observed], domain_key=problem.key)
    return np.asarray(solutions[0]), float(loss), adapter


#: the space-time problem's size (on top of the arm's config)
PDE = {'num_domain': 200, 'num_test': 20, 'lbfgs_maxiter': 8}


@pytest.mark.usefixtures('solve_on_cpu')
class TestASpaceTimeProblem:
    """Every mode on a 1-D-in-space PDE: windows (stages) are GeometryXTime
    over S1's spatial geometry and their own time span, the hand-off holds at
    every spatial point, and with the mechanism off DE and CW are S1."""

    @pytest.mark.parametrize('config', [TM, TM_OBS, DE, CW], ids=['TM', 'TM-obs', 'DE', 'CW'])
    def test_every_mode_marches_on_the_space_time_geometry(self, config, advection,
                                                           solve_on_cpu, problems, phases,
                                                           monkeypatch):
        import deepxde as dde
        handed = _recording_hand_offs(monkeypatch, solve_on_cpu.DeepXDEAdapter._param_digest)
        solution, loss, adapter = _solve_pde(solve_on_cpu, advection, config)
        stats = adapter.last_march_stats
        split, mask, t = advection.split, advection.mask, advection.t
        assert np.isfinite(loss)
        assert np.isfinite(solution[t >= split.t_train] if config is TM_OBS else solution).all()
        s1_problem = problems[0]                          # built before the dispatch
        assert isinstance(s1_problem.geom, dde.geometry.GeometryXTime)
        space = s1_problem.geom.geometry
        # R1 on a PDE too: every phase on S1's frozen weights
        w = adapter.last_pairing['loss_weights']
        assert len(phases) == len(stats['windows'])
        for phase in phases:
            assert phase['compiled'] == (w if phase['data'].bcs else w[:1])
            geom = phase['data'].geom
            assert isinstance(geom, dde.geometry.GeometryXTime)
            for ours, s1s in zip(geom.geometry.bbox, space.bbox, strict=True):
                np.testing.assert_array_equal(ours, s1s)
        if config is CW:                                  # S1's own points throughout
            assert all(p['data'].train_x is s1_problem.train_x for p in phases)
            # M = round(T / tau_min) on the RAW tau_min; here the snapped
            # Delta (whole levels) would give another count
            levels = np.unique(t)
            extent, step = levels[-1] - levels[0], (levels[-1] - levels[0]) / (len(levels) - 1)
            assert stats['n_chunks'] == round(extent / stats['tau_min']) > 1
            assert stats['n_chunks'] != round(extent / (stats['n_delta_levels'] * step))
            return
        edges = stats['edges']
        spans = [(p['data'].geom.timedomain.t0, p['data'].geom.timedomain.t1) for p in phases]
        if config is DE:                                  # [t_lo, e_s], the last S1's problem
            assert spans == [(edges[0], e) for e in edges[1:]]
            assert phases[-1]['data'] is s1_problem
            return
        first = 0 if config is TM else edges.index(stats['t_train'])
        assert spans == list(zip(edges[first:-1], edges[first + 1:]))
        assert len(handed) >= 2                           # the PDE hand-off ran, twice over
        X = solve_on_cpu._input_columns(advection.grids)
        _assert_the_chain(handed, solution, stats, X, first)
        if config is TM_OBS:                              # the observed state, through x
            at = mask & (t == split.t_train)
            np.testing.assert_allclose(solution[at],
                                       advection.observed[t[mask] == split.t_train],
                                       rtol=1e-6, atol=1e-6)

    @pytest.mark.parametrize('config', [DE, CW], ids=['DE', 'CW'])
    def test_with_the_mechanism_off_it_is_s1_bitwise(self, config, advection, solve_on_cpu,
                                                     monkeypatch):
        """DE with a single stage, CW with its weights dropped from the start:
        S1's problem from S1's start on S1's budget, on GeometryXTime too."""
        first, s1_loss, s1 = _solve_pde(solve_on_cpu, advection, S1)
        if config is DE:
            monkeypatch.setattr(tm, 'window_edges', lambda levels, *a, **k: [0, len(levels) - 1])
        else:
            monkeypatch.setattr(tm, 'CAUSAL_DONE', 0.0)
        second, loss, adapter = _solve_pde(solve_on_cpu, advection, config)
        assert len(adapter.last_march_stats['windows']) == 1
        np.testing.assert_array_equal(first, second)
        assert s1_loss == loss
        assert adapter.last_lbfgs_exit == {'reason': s1.last_lbfgs_exit['reason'],
                                           'iterations': s1.last_lbfgs_exit['iterations']}


# ============================================================ the refusals
#: every marching mode, with the name its refusals carry
MODES = [(TM, "time_marching='full'"), (TM_OBS, "time_marching='tail_obs'"),
         (DE, "time_marching='expand'"), (CW, "causal_weights='frozen'")]
MODE_IDS = ['TM', 'TM-obs', 'DE', 'CW']
#: (a change to the mode's config, the reason its refusal must give). The
#: epochs cases with an lr-free first order keep lr=None, which the adapter
#: itself accepts: only the march's own epochs rule refuses them.
REFUSED = [
    ({'epochs': 5, 'first_order': 'dog'}, 'set epochs=0'),
    ({'epochs': 5, 'first_order': 'prodigy'}, 'set epochs=0'),
    ({'epochs': 5, 'lr': 1e-3}, 'set epochs=0'),
    ({'lr': 1e-4}, 'set lr=None'),
    ({'second_order': 'lm', 'lbfgs_maxiter': 0}, "second_order must be 'lbfgs'"),
    ({'init': 'data_prefit'}, 'init must be None'),
    ({'val_frac': 0.1}, 'use val_frac=0'),
    ({'lbfgs_gtol_rel': 1e-3}, 'lbfgs_gtol_rel must be None'),
    ({'lbfgs_maxiter': 0}, 'lbfgs_maxiter >= 1')]


def _refusal(name, reason):
    return re.escape(name) + '.*' + re.escape(reason)


class TestTheRefusals:

    @pytest.mark.parametrize('extra, reason', REFUSED,
                             ids=[','.join(f'{k}={v}' for k, v in e.items()) for e, _ in REFUSED])
    @pytest.mark.parametrize('config, name', MODES, ids=MODE_IDS)
    def test_what_no_mode_can_honour_is_refused_for_its_own_reason(self, config, name,
                                                                    extra, reason):
        from epde.integrate.deepxde_integration import DeepXDEAdapter
        with pytest.raises(DeepXDEConfigError, match=_refusal(name, reason)):
            DeepXDEAdapter(**dict(config, **extra))

    @pytest.mark.parametrize('epochs, reason', [(0, 'lr_rule sets the Adam lr'),
                                                (5, 'set epochs=0')])
    @pytest.mark.parametrize('config', [c for c, _ in MODES], ids=MODE_IDS)
    def test_a_learning_rate_rule_is_refused(self, config, epochs, reason):
        """An lr_rule sets Adam's lr: at epochs=0 the adapter itself refuses it
        (before the march's checks run), with epochs > 0 the march does."""
        from epde.integrate.deepxde_integration import DeepXDEAdapter
        with pytest.raises(DeepXDEConfigError, match=re.escape(reason)):
            DeepXDEAdapter(**dict(config, lr_rule='prefit_distance', epochs=epochs))

    @pytest.mark.parametrize('extra, reason', [
        ({'output_transform': 'train_moments'}, 'output_transform must be None'),
        ({'collocation_density': 'flat'}, 'collocation_density must be None')])
    @pytest.mark.parametrize('config, name', MODES[:2], ids=MODE_IDS[:2])
    def test_the_hard_hand_off_refuses_s1s_output_map_and_point_placement(
            self, config, name, extra, reason):
        from epde.integrate.deepxde_integration import DeepXDEAdapter
        with pytest.raises(DeepXDEConfigError, match=_refusal(name, reason)):
            DeepXDEAdapter(**dict(config, **extra))

    def test_de_refuses_a_point_placement_its_earlier_stages_would_not_share(self):
        from epde.integrate.deepxde_integration import DeepXDEAdapter
        with pytest.raises(DeepXDEConfigError,
                           match=_refusal(MODES[2][1], 'collocation_density must be None')):
            DeepXDEAdapter(**dict(DE, collocation_density='flat'))

    def test_de_keeps_s1s_own_output_transform(self):
        """One network, no hard transform: S1's output map stays S1's."""
        from epde.integrate.deepxde_integration import DeepXDEAdapter
        assert tm.active(DeepXDEAdapter(**dict(DE, output_transform='train_moments')))

    def test_two_mechanisms_at_once_are_refused(self):
        from epde.integrate.deepxde_integration import DeepXDEAdapter
        with pytest.raises(DeepXDEConfigError, match='are two mechanisms; set one'):
            DeepXDEAdapter(**dict(CW, time_marching='full'))

    @pytest.mark.parametrize('extra', [{'output_transform': 'train_moments'},
                                       {'collocation_density': 'flat'}])
    def test_cw_keeps_s1s_own_problem_options(self, extra):
        """CW weighs S1's problem as it is: S1's output map and point placement
        stay S1's."""
        from epde.integrate.deepxde_integration import DeepXDEAdapter
        assert tm.active(DeepXDEAdapter(**dict(CW, **extra)))

    @pytest.mark.parametrize('cfg', [{'time_marching': 'sliding'},
                                     {'causal_weights': 'dynamic'}])
    def test_an_unknown_value_is_refused(self, cfg):
        from epde.integrate.deepxde_integration import DeepXDEAdapter
        with pytest.raises(DeepXDEConfigError):
            DeepXDEAdapter(**dict(S1, **cfg))

    def test_the_marching_configs_are_accepted(self):
        from epde.integrate.deepxde_integration import DeepXDEAdapter
        for cfg in (TM, TM_OBS, DE, CW):
            assert tm.active(DeepXDEAdapter(**cfg))
        assert not tm.active(DeepXDEAdapter(**S1))


def _gate():
    spec = importlib.util.spec_from_file_location(
        'pinn_gate_under_test', os.path.join(REPO, 'projects', 'pinn', 'gate.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class TestTheGateArms:

    def test_the_arms_are_s1_plus_one_mechanism(self):
        gate = _gate()
        s1 = gate.ARMS['S1']['config']
        for arm, key, value in (('TM', 'time_marching', 'full'),
                                ('TM-obs', 'time_marching', 'tail_obs'),
                                ('DE', 'time_marching', 'expand'),
                                ('CW', 'causal_weights', 'frozen')):
            cfg = gate.ARMS[arm]['config']
            assert {k: v for k, v in cfg.items() if k not in (key, 'lr')} == s1
            assert cfg[key] == value and cfg['lr'] is None
        assert 'epde/integrate/time_marching.py' in gate.PROVENANCE_FILES

    def test_their_full_configs_construct_and_every_key_is_read(self):
        from epde.integrate.deepxde_integration import DeepXDEAdapter
        gate = _gate()
        for arm in ('TM', 'TM-obs', 'DE', 'CW', '_smoke-TM', '_smoke-TMo', '_smoke-DE',
                    '_smoke-CW'):
            cfg = gate.arm_config(arm)
            adapter = DeepXDEAdapter(**cfg)
            assert [k for k in cfg if not hasattr(adapter, k)] == []

    @pytest.mark.parametrize('arm', ['TM', 'TM-obs', 'DE', 'CW'])
    def test_trace_mode_is_refused(self, arm):
        gate = _gate()
        assert gate.untraceable(arm)
        with pytest.raises(SystemExit, match='trace'):
            gate.run_solve('lorenz', arm, 'true', 0, mode='trace')
        with pytest.raises(SystemExit, match='trace refused'):
            gate.drive('lorenz', ['S1', arm], [0], ['true'], 'trace', 'never_written')
        assert gate.untraceable('S1') is None

    def test_an_lm_trace_batch_is_driven_as_before(self):
        """The up-front refusal covers the marching arms only: an LM arm keeps
        its per-solve refusal in run_solve, so a batch holding one still runs
        its other arms (the LM solve counted as failed), as before Stage 3."""
        gate = _gate()
        assert gate.untraceable('S5-32') == 'has no L-BFGS phase to trace'
        assert gate.trace_refusals(['S1', 'S5-32']) == {}
        assert list(gate.trace_refusals(['S1', 'S5-32', 'CW'])) == ['CW']
        with pytest.raises(SystemExit, match='no L-BFGS phase'):
            gate.run_solve('lorenz', 'S5-32', 'true', 0, mode='trace')
