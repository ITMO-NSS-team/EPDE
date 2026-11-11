"""Time-marching solves for the DeepXDE adapter -- opt-in, learning-rate free.

On Lorenz the S1 PINN (Glorot init, L-BFGS from it, an observation term on the
train window, the residual on collocation points that also cover the held-out
tail) collapses onto an equilibrium of the TRUE equation: the net starts at ~0,
which is the origin equilibrium, and arms that do fit the train window still
slide onto an equilibrium in the no-data tail. Short time domains remove those
zero-residual optima (Rohrhofer et al., TMLR 2023; Wang, Sankaran &
Perdikaris 2022). Each mode here is ONE mechanism added to S1:

* ``time_marching='full'`` (TM): the solver's time extent is cut into windows;
  window 0 is S1's own network (same Glorot draw, same pairing digest) on
  ``[t_lo, e_1]`` with the train observations inside it. Every later window
  warm-starts from a copy of its predecessor and inherits the predecessor's
  state at the edge through a HARD output transform -- continuity holds by
  construction, not by a penalty term (``hard_ic_transform``).
* ``time_marching='tail_obs'`` (TM-obs, a DIAGNOSTIC upper bound): no PINN on
  the train window; only TM's tail windows, started from the OBSERVED state at
  the last train level -- what TM's tail can do when the hand-off is perfect.
* ``time_marching='expand'`` (DE): ONE network, S1's, no hard transform,
  trained on growing domains ``[t_lo, e_s]`` over TM's edges; the last stage
  is S1's own problem (its data object), so its objective IS S1's.
* ``causal_weights='frozen'`` (CW): S1's own problem -- its network, points and
  observations -- with causal weights (Wang, Sankaran & Perdikaris 2022): the
  extent is cut into ``M`` equal chunks, every point's residual and
  observation error is scaled by ``sqrt(w)`` of its chunk, ``w_i =
  exp(-eps * sum_{j<i} L_j)`` from the per-chunk losses, so a chunk weighs in
  only once every earlier one is fitted. The weights are re-read every
  ``CAUSAL_RESTART`` L-BFGS iterations (each restart a fresh L-BFGS) and held
  constant in between; once every chunk's is ``>= CAUSAL_DONE`` they are
  dropped (``w == 1``: S1's own objective) for the rest of the budget.

Rules every mode follows (pinned in ``tests/unit/test_time_marching.py``):

* the loss weights are S1's: ``_solve_trajectory`` computes them on S1's full
  problem at the Glorot init, exactly as S1 does, and hands them here frozen;
  no window re-probes them;
* the window length is tau_min = min_v std(x_v) / rms(dx_v/dt) (``tau_min``),
  measured on the TRAIN levels only and snapped to whole grid levels (TM-obs
  marches TM's own tail windows: same edges, caps and points, so it spends the
  tail's share of the budget); the edges cover S1's time extent and
  include ``t_train``, so no window mixes observed and held-out levels
  (``window_edges``); CW's chunks are tau_min long on average (``M`` = the
  extent over the RAW tau_min, rounded: equal chunks need no snapping);
* ``lbfgs_maxiter`` is the TOTAL budget, split over the windows in proportion to
  their length (``window_caps``; a DE stage gets the share of the window it
  adds; CW spends it in restarts of ``CAUSAL_RESTART``); each window, stage or
  restart runs the adapter's own L-BFGS machinery (budget, compile, stall
  guard, train), so its exit is recorded the same way S1's is;
* held-out observations are never read.

``adapter.last_march_stats`` records the plan and, per window (per stage for
DE, per restart for CW), the iterations, the exit, the seconds and the final
loss.

This module imports numpy only; torch and DeepXDE are imported by the solve,
which runs inside the adapter (and so after DeepXDE's import-time device
handling).
"""
import copy
import math
import time as _time
from typing import List, Sequence, Tuple

import numpy as np

from epde.integrate.heldout import DeepXDEConfigError

#: The values each key accepts; None is today's path. One mechanism at a time.
CHOICES = {'time_marching': (None, 'full', 'tail_obs', 'expand'),
           'causal_weights': (None, 'frozen')}
#: The fewest physics points a window gets, whatever its share of the extent.
MIN_WINDOW_DOMAIN = 32
#: CW's restart cadence: L-BFGS iterations between two readings of the weights.
CAUSAL_RESTART = 200
#: CW drops its weights (w == 1, S1's own objective) once every chunk's is this high.
CAUSAL_DONE = 0.99
#: L-BFGS exits after which CW stops: the phase diverged or could not step.
_CAUSAL_STOPS = ('nonfinite', 'no_step', 'stop_training')
#: How each windowed mode splits ``lbfgs_maxiter`` (N) over the extent T,
#: recorded with its stats so a row says which budget it was screened at.
_CAP_RULES = {'full': 'window k: ceil(N * len_k / T)',
              'tail_obs': "TM's window caps, tail windows only (the tail's share of N)",
              'expand': 'stage s < S: ceil(N * (e_s - e_{s-1}) / T), the time it adds; '
                        "the last stage (S1's problem): S1's own N"}


def active(adapter) -> bool:
    """Whether a marching key is set (the solve then dispatches here)."""
    return any(getattr(adapter, key, None) is not None for key in CHOICES)


def mode_of(adapter) -> str:
    """The mechanism's name in the stats and the dispatch: the ``time_marching``
    value, or ``'causal_<value>'`` for ``causal_weights``."""
    if adapter.time_marching is not None:
        return adapter.time_marching
    return f'causal_{adapter.causal_weights}'


def validate_config(adapter):
    """Refuse a value this module does not know, two mechanisms at once, and
    every option a marching mode cannot honour. Called at the end of the
    adapter's ``__init__``, so a misconfiguration fails before any work rather
    than as an all-NaN solve."""
    for key, allowed in CHOICES.items():
        value = getattr(adapter, key)
        if value not in allowed:
            raise DeepXDEConfigError(f'deepxde_config {key} must be one of {allowed}, '
                                     f'got {value!r}')
    keys = [key for key in CHOICES if getattr(adapter, key) is not None]
    if not keys:
        return
    if len(keys) > 1:
        raise DeepXDEConfigError(f'deepxde_config {" and ".join(keys)} are two mechanisms; '
                                 'set one')
    name = f'{keys[0]}={getattr(adapter, keys[0])!r}'
    hard_ic = adapter.time_marching in ('full', 'tail_obs')
    # DE's last stage is S1's warped problem but its earlier stages are not
    own_points = adapter.time_marching is not None
    refusals = (
        (adapter.epochs != 0, 'is L-BFGS only: set epochs=0'),
        (adapter.lr is not None, 'is learning-rate free: set lr=None'),
        (adapter.second_order != 'lbfgs', "splits the L-BFGS budget: second_order must be "
                                          "'lbfgs'"),
        (adapter.init is not None, 'starts from the Glorot init: init must be None'),
        (adapter.lbfgs_maxiter < 1, 'needs lbfgs_maxiter >= 1 (the TOTAL L-BFGS budget)'),
        (adapter.val_frac > 0.0, 'does not feed the validation checkpoint: use val_frac=0'),
        (adapter.lbfgs_gtol_rel is not None, "keeps DeepXDE's absolute gtol: "
                                             'lbfgs_gtol_rel must be None'),
        (hard_ic and adapter.output_transform is not None,
         'owns the output transform (the hard continuity): output_transform must be None'),
        (own_points and adapter.collocation_density is not None,
         'places its own points per window / stage: collocation_density must be None'),
    )
    for refused, why in refusals:
        if refused:
            raise DeepXDEConfigError(f'deepxde_config {name} {why}')


# ============================================================ the plan (numpy)
def inner_box(grids, mask) -> Tuple[List[np.ndarray], Tuple[int, ...]]:
    """The inner domain's coordinates per EPDE axis (time first) and its shape.
    The observed fields are flat in C order over this box."""
    m = np.asarray(mask).astype(bool)
    axes = [np.unique(np.asarray(g, dtype=np.float64)[m]) for g in grids]
    shape = tuple(len(a) for a in axes)
    if int(m.sum()) != int(np.prod(shape)):
        raise DeepXDEConfigError('time marching needs a box-shaped inner domain')
    return axes, shape


def tau_min(t_levels, fields: Sequence[np.ndarray], t_train: float):
    """``min_v std(x_v) / rms(dx_v/dt)``: the shortest time scale the observed
    fields show, and the per-variable values.

    TRAIN levels only (``t_levels <= t_train``): the fields are cut to them
    BEFORE anything is computed, so the held-out levels cannot enter -- not
    even through ``np.gradient``'s one-sided edge. ``fields`` are time-first
    arrays on ``t_levels``; for a PDE the statistics run over every space-time
    point of the train window. A sinusoid gives ``1 / omega``. A variable that
    never changes (std or rms 0; the rms of a constant's ``np.gradient`` on a
    coordinate array is rounding noise, not 0) imposes no scale.
    """
    levels = np.asarray(t_levels, dtype=np.float64)
    n_train = int(np.sum(levels <= t_train))
    if n_train < 2:
        raise DeepXDEConfigError(f'time marching: {n_train} train level(s); tau_min needs 2')
    taus = []
    for field in fields:
        x = np.asarray(field, dtype=np.float64)[:n_train]
        rate = np.gradient(x, levels[:n_train], axis=0)
        rms = float(np.sqrt(np.mean(rate ** 2)))
        sd = float(np.std(x))
        taus.append(sd / rms if sd > 0.0 and rms > 0.0 else math.inf)
    tau = min(taus)
    if not (math.isfinite(tau) and tau > 0.0):
        raise DeepXDEConfigError(f'time marching: no time scale on the train window '
                                 f'(tau per variable {taus})')
    return tau, taus


def snap(tau: float, levels) -> int:
    """``tau`` in whole grid levels (>= 1), on the grid's mean time step."""
    levels = np.asarray(levels, dtype=np.float64)
    step = (levels[-1] - levels[0]) / (len(levels) - 1)
    return max(1, int(round(tau / step)))


def window_edges(levels, t_train: float, n_delta: int, t_first_obs: float = None) -> List[int]:
    """Window edges as indices into the solver's time ``levels``.

    Anchored on ``t_train``: edges every ``n_delta`` levels from it, backwards
    and forwards, so it is always an edge and no window mixes observed and
    held-out levels. The first and last edge are the extent's own ends (S1's
    collocation geometry), and a partial window at either end joins its
    neighbour, so every window has at least ``n_delta`` levels except where
    ``t_train`` or the end of the extent leaves less room on its side.

    At the front the count starts at ``t_first_obs`` (the first observed level),
    not at the extent's end: the boundary strip before it holds no observation,
    so it joins window 0 instead of becoming a window of its own -- one with
    neither data nor an inherited state, which is exactly the fixed-point trap.
    """
    levels = np.asarray(levels, dtype=np.float64)
    last = len(levels) - 1
    hits = np.flatnonzero(levels == t_train)
    if len(hits) != 1:
        raise DeepXDEConfigError(f'time marching: t_train={t_train!r} is not a level of '
                                 'the time grid')
    split = int(hits[0])
    if split >= last:
        raise DeepXDEConfigError('time marching: no held-out level after t_train')
    start = 0
    if t_first_obs is not None:
        start = int(np.searchsorted(levels, t_first_obs, side='left'))
    n = max(1, int(n_delta))
    left = [split] + list(range(split - n, start, -n))  # split, split - n, ... > start
    if len(left) > 1 and left[-1] - start < n:
        left.pop()                                  # a leading remainder joins window 0
    right = list(range(split + n, last, n))         # ... < last
    if right and last - right[-1] < n:
        right.pop()                                 # a trailing remainder joins the last
    inner = [i for i in sorted(left) if i > 0] + right
    return [0] + inner + [last]


def window_caps(lengths: Sequence[float], total: float, budget: int) -> List[int]:
    """``max(1, ceil(budget * len_k / total))`` per window. The guard keeps a
    product that is an integer in exact arithmetic from rounding up by one."""
    caps = []
    for length in lengths:
        share = budget * float(length) / float(total)
        caps.append(max(1, int(math.ceil(share - 1e-9 * max(1.0, share)))))
    return caps


def window_points(length: float, total: float, count: int, floor: int) -> int:
    """``max(floor, round(count * len_k / total))``."""
    return max(int(floor), int(round(count * float(length) / float(total))))


def chunk_count(extent: float, tau: float) -> int:
    """CW's ``M = max(1, round(T / tau_min))``: the extent over the RAW
    tau_min. The chunks are equal slices of the extent, not grid-aligned, so
    there is nothing to snap (the windows' Delta is snapped; CW's is not)."""
    return max(1, int(round(float(extent) / float(tau))))


def causal_weights(chunk_loss, eps: float) -> np.ndarray:
    """``w_i = exp(-eps * sum_{j<i} L_j)``: 1 for the first chunk and, the
    losses being non-negative, non-increasing in time. A chunk weighs in only
    once the chunks before it are fitted."""
    L = np.asarray(chunk_loss, dtype=np.float64)
    before = np.concatenate([[0.0], np.cumsum(L)[:-1]])
    return np.exp(-float(eps) * before)


def time_orders(eq_list, var_names: Sequence[str], ndim: int) -> List[int]:
    """``m_v = max(1, highest time-derivative order of v)`` over every term and
    target of the CANDIDATE's equations, read from the factors' derivative
    codes (EPDE axis 0 is time) the way the residual reads them -- never from
    the true equation."""
    from epde.integrate.residual_terms import equation_spec
    index = {name: i for i, name in enumerate(var_names)}
    orders = [1] * len(var_names)
    for eq in eq_list:
        spec = equation_spec(eq, index, ndim)
        for factors in [term.factors for term in spec.terms] + [spec.target]:
            for factor in factors:
                if factor.kind == 'var':
                    orders[factor.var] = max(orders[factor.var], int(factor.orders[0]))
    return orders


def fd_weights(z: float, nodes, order: int) -> np.ndarray:
    """Fornberg's weights: ``w[k] @ f(nodes)`` is the k-th derivative at ``z``,
    ``k <= order`` (non-uniform nodes allowed)."""
    x = np.asarray(nodes, dtype=np.float64)
    c = np.zeros((order + 1, len(x)))
    c[0, 0] = 1.0
    c1, c4 = 1.0, x[0] - z
    for i in range(1, len(x)):
        mn = min(i, order)
        c2, c5, c4 = 1.0, c4, x[i] - z
        for j in range(i):
            c3 = x[i] - x[j]
            c2 *= c3
            if j == i - 1:
                for k in range(mn, 0, -1):
                    c[k, i] = c1 * (k * c[k - 1, i - 1] - c5 * c[k, i - 1]) / c2
                c[0, i] = -c1 * c5 * c[0, i - 1] / c2
            for k in range(mn, 0, -1):
                c[k, j] = (c4 * c[k, j] - k * c[k - 1, j]) / c3
            c[0, j] = c4 * c[0, j] / c3
        c1 = c2
    return c


def observed_edge_state(t_levels, fields: Sequence[np.ndarray], t_train: float,
                        n_orders: int) -> List[List[np.ndarray]]:
    """``[[d_t^j x_v(t_train)]_v]_{j < n_orders}`` from the TRAIN levels: the
    observed value itself for j = 0, a backward difference over the last
    ``j + 2`` train levels (second order) above it."""
    levels = np.asarray(t_levels, dtype=np.float64)
    n_train = int(np.sum(levels <= t_train))
    out = []
    for j in range(n_orders):
        q = 1 if j == 0 else j + 2
        if n_train < q:
            raise DeepXDEConfigError(f'time marching: d^{j}/dt^{j} at t_train needs {q} '
                                     f'train levels, there are {n_train}')
        w = fd_weights(levels[n_train - 1], levels[n_train - q:n_train], j)[j]
        out.append([np.tensordot(w, np.asarray(f, dtype=np.float64)[n_train - q:n_train],
                                 axes=(0, 0)) for f in fields])
    return out


def natural_spline_operator(nodes) -> np.ndarray:
    """``S`` with ``M = S @ y``: the second derivatives of the natural cubic
    spline through ``(nodes, y)`` (``M = 0`` at both ends)."""
    x = np.asarray(nodes, dtype=np.float64)
    n = len(x)
    S = np.zeros((n, n))
    if n < 3:
        return S
    h = np.diff(x)
    A = np.zeros((n - 2, n - 2))
    R = np.zeros((n - 2, n))
    for i in range(1, n - 1):
        r = i - 1
        if r > 0:
            A[r, r - 1] = h[i - 1]
        A[r, r] = 2.0 * (h[i - 1] + h[i])
        if r < n - 3:
            A[r, r + 1] = h[i]
        R[r, i - 1] = 6.0 / h[i - 1]
        R[r, i] = -6.0 / h[i - 1] - 6.0 / h[i]
        R[r, i + 1] = 6.0 / h[i]
    S[1:-1] = np.linalg.solve(A, R)
    return S


# ============================================================ torch pieces
def _spline_basis(x, nodes, S, torch):
    """Row p: the weights of the node values in the natural cubic spline at
    ``x[p]`` -- differentiable in ``x``, C2 inside, LINEAR past the ends (the
    natural spline's own extension: its second derivative is 0 there)."""
    n = nodes.numel()
    if n == 1:
        return torch.ones((x.shape[0], 1), dtype=x.dtype, device=x.device)
    xc = x.clamp(float(nodes[0]), float(nodes[-1]))
    i = (torch.bucketize(xc.detach().contiguous(), nodes, right=True) - 1).clamp(0, n - 2)
    h = (nodes[i + 1] - nodes[i])[:, None]
    b = (xc - nodes[i])[:, None] / h
    a = 1.0 - b
    eye = torch.eye(n, dtype=x.dtype, device=x.device)
    value = (a * eye[i] + b * eye[i + 1]
             + (a ** 3 - a) * h ** 2 / 6.0 * S[i] + (b ** 3 - b) * h ** 2 / 6.0 * S[i + 1])
    slope = ((eye[i + 1] - eye[i]) / h
             - (3.0 * a ** 2 - 1.0) * h / 6.0 * S[i] + (3.0 * b ** 2 - 1.0) * h / 6.0 * S[i + 1])
    return value + (x - xc)[:, None] * slope


class _SplineEdgeState:
    """The observed edge state of a PDE as a function of space: the tensor
    product of natural cubic splines through the inner spatial grid, one per
    (time-derivative order, variable). Called with the spatial input columns,
    returns ``[(n, n_vars)]`` per order -- what ``hard_ic_transform`` reads."""

    def __init__(self, space_axes, values, dtype, device):
        import torch
        self._torch = torch
        self.nodes = [torch.as_tensor(a, dtype=dtype, device=device) for a in space_axes]
        self.S = [torch.as_tensor(natural_spline_operator(a), dtype=dtype, device=device)
                  for a in space_axes]
        self.values = [[torch.as_tensor(np.asarray(v), dtype=dtype, device=device)
                        for v in per_var] for per_var in values]

    def __call__(self, xs):
        torch = self._torch
        n = xs.shape[0]
        bases = [_spline_basis(xs[:, a], self.nodes[a], self.S[a], torch)
                 for a in range(len(self.nodes))]
        out = []
        for per_var in self.values:
            cols = []
            for field in per_var:
                acc = bases[0] @ field.reshape(field.shape[0], -1)
                for a in range(1, len(bases)):
                    acc = (acc.reshape(n, bases[a].shape[1], -1) * bases[a][:, :, None]).sum(1)
                cols.append(acc.reshape(n, 1))
            out.append(torch.cat(cols, dim=1))
        return out


def _edge_values(net, xs, e: float, n_orders: int):
    """``[d_t^j net(xs, e)]_{j < n_orders}``, each ``(n, n_out)``. The time
    derivatives by autograd in a fresh ``t`` leaf, with ``create_graph`` so the
    result stays differentiable in ``xs`` (the residual's spatial derivatives
    pass through it)."""
    import torch
    p0 = next(net.parameters())
    t = torch.full((xs.shape[0], 1), float(e), dtype=p0.dtype, device=p0.device)
    with torch.enable_grad():
        if n_orders > 1:
            t.requires_grad_(True)
        out = [net(torch.cat([xs, t], dim=1))]
        for _ in range(1, n_orders):
            cols = []
            for c in range(out[-1].shape[1]):
                g = torch.autograd.grad(out[-1][:, c].sum(), t, create_graph=True,
                                        allow_unused=True)[0]
                cols.append(torch.zeros_like(t) if g is None else g)
            out.append(torch.cat(cols, dim=1))
    return out


def net_edge_state(P, e: float, n_orders: int, n_space: int):
    """The edge state ``[d_t^j P(x, e)]_j`` of a frozen window as the
    ``ic(xs)`` callable ``hard_ic_transform`` reads. With no spatial axis it is
    a constant, evaluated once here and detached; otherwise ``P`` is evaluated
    at every call (it must follow ``x``)."""
    import torch
    if n_space == 0:
        p0 = next(P.parameters())
        values = [v.detach() for v in
                  _edge_values(P, torch.zeros((1, 0), dtype=p0.dtype, device=p0.device),
                               e, n_orders)]
        return lambda xs: values
    return lambda xs: _edge_values(P, xs, e, n_orders)


def constant_edge_state(values, dtype, device):
    """``values[j][v]`` (numbers) as the ``ic(xs)`` callable of an ODE."""
    import torch
    rows = [torch.as_tensor(np.asarray(per_var, dtype=np.float64).reshape(1, -1),
                            dtype=dtype, device=device) for per_var in values]
    return lambda xs: rows


def hard_ic_transform(ic, e: float, length: float, orders: Sequence[int], n_space: int):
    """DeepXDE output transform of a window ``[e, e + length]``::

        y_v(x, t) = sum_{j < m_v} (t - e)^j / j! * ic(x)[j][:, v]
                  + ((t - e) / length)^{m_v} * N_v(x, t)

    At ``t = e`` every term but the first vanishes EXACTLY (``t - e`` is an
    exact zero in float32 when ``t`` is the edge's own float32 value), so the
    window starts from ``ic`` and its first ``m_v - 1`` time derivatives there
    are ``ic``'s: continuity by construction. ``m_v`` is the highest time
    derivative of ``v`` in the candidate, so the residual sees no kink."""
    import torch
    facts = [float(math.factorial(j)) for j in range(max(orders))]

    def transform(x, y):
        s = x[:, n_space:n_space + 1] - e
        edge = ic(x[:, :n_space])
        cols = []
        for v, m_v in enumerate(orders):
            col = edge[0][:, v:v + 1]
            for j in range(1, m_v):
                col = col + s ** j / facts[j] * edge[j][:, v:v + 1]
            cols.append(col + (s / length) ** m_v * y[:, v:v + 1])
        return torch.cat(cols, dim=1)

    return transform


def _piecewise_net(nets, inner_edges):
    """The windows as ONE DeepXDE network: a point goes to the window that owns
    its time (left-closed ``[e_k, e_{k+1})``, the last window closed). Used to
    evaluate S1's own objective at the marched field."""
    import deepxde as dde
    import torch

    class _Piecewise(dde.nn.NN):
        def __init__(self):
            super().__init__()
            self.windows = torch.nn.ModuleList(nets)
            self.inner_edges = [float(e) for e in inner_edges]

        def forward(self, inputs):
            t = inputs[:, -1].detach().contiguous()
            edges = torch.as_tensor(self.inner_edges, dtype=t.dtype, device=t.device)
            owner = torch.bucketize(t, edges, right=True)
            order, parts = [], []
            for k, net in enumerate(self.windows):
                rows = torch.nonzero(owner == k, as_tuple=True)[0]
                if rows.numel():
                    parts.append(net(inputs[rows]))
                    order.append(rows)
            order = torch.cat(order)
            inverse = torch.empty_like(order)
            inverse[order] = torch.arange(order.numel(), device=order.device)
            return torch.cat(parts)[inverse]

    return _Piecewise()


def owners(t, inner_edges) -> np.ndarray:
    """The window that owns each time: left-closed ``[e_k, e_{k+1})``, the last
    window closed (a time past every inner edge belongs to the last one)."""
    return np.searchsorted(np.asarray(inner_edges, dtype=np.float64),
                           np.asarray(t, dtype=np.float64), side='right')


def _frozen(net):
    P = copy.deepcopy(net)
    for p in P.parameters():
        p.requires_grad_(False)
    return P


class _CausalScale:
    """CW's ``sqrt(w)`` of the chunk a time falls in: a detached constant for a
    whole restart (``set`` is called between restarts only), read by the
    weighted residual and observation terms. Chunks are left-closed
    ``[c_j, c_{j+1})``, the last one closed (as ``owners``); the edges are cast
    to the network's dtype once, so every point is binned the same way
    wherever it is read."""

    def __init__(self, edges, dtype, device):
        import torch
        self._torch = torch
        self.n_chunks = len(edges) - 1
        self.inner = torch.as_tensor(np.asarray(edges, dtype=np.float64)[1:-1],
                                     dtype=dtype, device=device)
        self.root = None

    def set(self, w):
        self.root = self._torch.as_tensor(np.sqrt(np.asarray(w, dtype=np.float64)),
                                          dtype=self.inner.dtype, device=self.inner.device)

    def chunk(self, t):
        return self._torch.bucketize(t.detach().reshape(-1).contiguous(), self.inner,
                                     right=True)

    def __call__(self, t):
        return self.root[self.chunk(t)].reshape(-1, 1)


def _causal_pde(pde, scale):
    """S1's residual with each point's error times ``sqrt(w)`` of its chunk."""
    def weighted(x, y):
        s = scale(x[:, -1])
        return [r * s for r in pde(x, y)]
    return weighted


def _causal_bc(bc, scale):
    """S1's observation term of one variable with each point's error times
    ``sqrt(w)`` of its chunk. The error is ``bc``'s own -- plain or
    scale-invariant (``_ScaleInvariantPointSetBC``) -- so the weighting adds to
    whatever observation loss S1 uses; points, values and component are
    ``bc``'s."""
    import deepxde as dde

    class _CausalPointSetBC(dde.icbc.PointSetBC):
        def error(self, X, inputs, outputs, beg, end, aux_var=None):
            return (bc.error(X, inputs, outputs, beg, end, aux_var)
                    * scale(inputs[beg:end, -1]))

    return _CausalPointSetBC(bc.points, bc.values, component=bc.component)


def causal_problem(data, scale):
    """S1's problem ``data`` with causal weights: a SHALLOW copy -- the very
    same points, observations and test set, not one resampled -- whose residual
    and observation terms scale every point's error by ``sqrt(w)`` of its
    chunk (``_causal_pde`` / ``_causal_bc``)."""
    weighted = copy.copy(data)
    weighted.pde = _causal_pde(data.pde, scale)
    weighted.bcs = [_causal_bc(bc, scale) for bc in data.bcs]
    return weighted


def chunk_losses(net, data, weights, scale) -> np.ndarray:
    """CW's ``L_j = sum_i w_i * mean_{term i's points in chunk j} err_i^2`` at the
    network's CURRENT parameters, on S1's own (unweighted) problem ``data``:
    its training points, read the way DeepXDE's ``PDE.losses`` reads them --
    the residual on the rows after the observation points, each observation
    term on its own rows -- and S1's frozen ``weights``. A chunk a term has no
    point in adds nothing for that term (the observations stop at t_train)."""
    import deepxde as dde
    import torch
    p0 = next(net.parameters())
    n = scale.n_chunks
    x = torch.as_tensor(np.asarray(data.train_x), dtype=p0.dtype, device=p0.device)
    x.requires_grad_()
    starts = [int(s) for s in np.cumsum([0] + list(data.num_bcs))]
    total = torch.zeros(n, dtype=torch.float64, device=p0.device)
    try:
        with torch.enable_grad():
            y = net(x)
            f = data.pde(x, y)
            f = list(f) if isinstance(f, (list, tuple)) else [f]
            terms = [(r[starts[-1]:], x[starts[-1]:, -1]) for r in f]
            terms += [(bc.error(data.train_x, x, y, starts[i], starts[i + 1]),
                       x[starts[i]:starts[i + 1], -1]) for i, bc in enumerate(data.bcs)]
            for w, (err, t) in zip(weights, terms, strict=True):
                chunk = scale.chunk(t)
                sq = err.detach().reshape(-1).to(torch.float64) ** 2
                sums = torch.zeros_like(total).index_add_(0, chunk, sq)
                counts = torch.bincount(chunk, minlength=n).to(torch.float64)
                total += float(w) * torch.where(counts > 0, sums / counts.clamp(min=1.0),
                                                torch.zeros_like(sums))
    finally:
        dde.grad.clear()
    return total.cpu().numpy()


# ============================================================ the solve
class _March:
    """What every mode shares: the plan (windows, budgets, points; CW's chunks),
    the frozen weights, and one window's (stage's, restart's) training through
    the adapter's own L-BFGS."""

    def __init__(self, adapter, eq_list, var_names, grids, mask, data_list, split,
                 inner_coords, pde_func, net, data_obj, weights):
        self.adapter = adapter
        self.var_names = list(var_names)
        self.grids = [np.asarray(g) for g in grids]
        # the held-out values blanked once, here: nothing below can read them
        # (every consumer selects train points anyway; this makes it structural)
        self.data_list = [np.where(split.train, np.asarray(d, dtype=np.float64).reshape(-1),
                                   np.nan) for d in data_list]
        self.split = split
        self.inner_coords = inner_coords
        self.pde_func = pde_func
        self.net = net
        self.data_obj = data_obj
        self.weights = [float(w) for w in weights]
        self.n_eqs = len(eq_list)
        self.n_space = len(self.grids) - 1
        self.mask = np.asarray(mask).astype(bool).reshape(-1)
        p0 = next(net.parameters())
        self.dtype, self.device = p0.dtype, p0.device

        self.axes, shape = inner_box(self.grids, mask)
        # the observed fields on the TRAIN levels of the inner box (time first)
        self.train_levels = self.axes[0][self.axes[0] <= split.t_train]
        self.fields = [d.reshape(shape)[:len(self.train_levels)] for d in self.data_list]
        self.levels = np.unique(np.asarray(self.grids[0], dtype=np.float64))
        self.tau, taus = tau_min(self.train_levels, self.fields, split.t_train)
        self.n_delta = snap(self.tau, self.levels)
        self.windows = []
        self.stats = {'mode': mode_of(adapter), 'tau_min': self.tau,
                      'tau': dict(zip(self.var_names, taus)), 'n_delta_levels': self.n_delta}
        if adapter.time_marching is not None:
            self.edge_levels = window_edges(self.levels, split.t_train, self.n_delta,
                                            t_first_obs=float(self.axes[0][0]))
            self.edges = [float(self.levels[i]) for i in self.edge_levels]
            self.total = self.edges[-1] - self.edges[0]
            self.lengths = [b - a for a, b in zip(self.edges[:-1], self.edges[1:])]
            self.caps = window_caps(self.lengths, self.total, adapter.lbfgs_maxiter)
            if adapter.time_marching == 'expand':
                # PI, Oct 6: the last stage IS S1's problem and gets S1's whole
                # budget, so DE differs from S1 only by the curriculum it starts from
                self.caps[-1] = int(adapter.lbfgs_maxiter)
            self.orders = time_orders(eq_list, var_names, len(self.grids))
            runs = self.caps
            if adapter.time_marching == 'tail_obs':
                self.split_k = self.edges.index(float(split.t_train))   # the first tail window
                runs = self.caps[self.split_k:]
            self.stats.update({
                'n_windows': len(self.lengths), 'edges': list(self.edges),
                'edge_levels': [int(i) for i in self.edge_levels],
                't_train': float(split.t_train),
                'time_orders': dict(zip(self.var_names, self.orders)),
                'budget': int(adapter.lbfgs_maxiter), 'caps': list(self.caps),
                # what the mode can spend of ``budget`` and how it is split
                # (TM-obs and DE's last stage: conventions awaiting the PI)
                'cap_rule': _CAP_RULES[adapter.time_marching],
                'budget_planned': int(sum(runs))})
        else:
            # CW: no windows; equal chunks of S1's whole extent, tau_min long
            # on average, and no hand-off (so no time orders)
            extent = float(self.levels[-1] - self.levels[0])
            self.n_chunks = chunk_count(extent, self.tau)
            self.chunk_edges = np.linspace(self.levels[0], self.levels[-1], self.n_chunks + 1)
            self.stats.update({
                'chunk_length': extent / self.n_chunks, 'n_chunks': self.n_chunks,
                'chunk_edges': [float(c) for c in self.chunk_edges],
                't_train': float(split.t_train), 'budget': int(adapter.lbfgs_maxiter),
                'restart_every': int(CAUSAL_RESTART), 'plain_at_min_w': float(CAUSAL_DONE)})
        self.stats.update({'loss_weights': list(self.weights), 'windows': self.windows})
        # filled as the solve goes, so a solve that fails part-way still says
        # how far it got
        adapter.last_march_stats = self.stats

    # ---- one window / stage
    def geometry(self, t0, t1):
        import deepxde as dde
        import epde.integrate.deepxde_integration as dxi
        time = dde.geometry.TimeDomain(t0, t1)
        if self.n_space == 0:
            return time
        return dde.geometry.GeometryXTime(dxi._geometry(self.grids).geometry, time)

    def observations(self, t0, t1, left_closed):
        """The train observations inside ``[t0, t1]`` for a span with no
        inherited state, ``(t0, t1]`` for a window whose state at ``t0`` is
        fixed by the hard transform (a point there would be a constant loss).
        Same loss type as S1's (``_observation_bc``)."""
        import deepxde as dde
        t = self.inner_coords[:, -1]
        inside = self.split.train & (t <= t1) & ((t >= t0) if left_closed else (t > t0))
        if not inside.any():
            return [], 0
        adapter = self.adapter
        # S1's record of its own observation terms. ``_observation_bc`` appends
        # to the record IN PLACE, so the window's terms get a throwaway one.
        kept, adapter.last_obs_stats = adapter.last_obs_stats, None
        try:
            bcs = [adapter._observation_bc(dde, self.inner_coords[inside],
                                           np.asarray(observed, dtype=np.float64)
                                           .reshape(-1)[inside].reshape(-1, 1), var_idx)
                   for var_idx, observed in enumerate(self.data_list)]
        finally:
            adapter.last_obs_stats = kept
        return bcs, int(inside.sum())

    def input_transform(self, net, k):
        """S1's input map with time remapped to the window (space unchanged)."""
        window = [np.array([self.edges[k], self.edges[k + 1]])] + self.grids[1:]
        self.adapter._apply_transforms(net, window, self.data_list, self.split)

    def loss_weights(self, bcs):
        """S1's frozen weights for a problem with or without its observation
        terms (DeepXDE orders the residuals before the observations)."""
        return self.weights[:self.n_eqs] + (self.weights[self.n_eqs:] if bcs else [])

    def problem(self, t0, t1, n_domain, n_test, bcs):
        """S1's kind of DeepXDE problem (same residual, no initial or boundary
        points) on ``[t0, t1]``."""
        import deepxde as dde
        geom = self.geometry(t0, t1)
        if self.n_space == 0:
            return dde.data.PDE(geom, self.pde_func, bcs, num_domain=n_domain,
                                num_boundary=0, num_test=n_test)
        return dde.data.TimePDE(geom, self.pde_func, bcs, num_domain=n_domain,
                                num_boundary=0, num_initial=0, num_test=n_test)

    def fit(self, model, weights, cap, record):
        """One L-BFGS phase through the adapter's own machinery (budget,
        compile, stall guard, train); ``record`` gets the iterations, the exit,
        the seconds and the final loss, and joins the per-window list."""
        import epde.integrate.deepxde_integration as dxi
        self.adapter._set_lbfgs_budget(cap)
        model.compile('L-BFGS', loss_weights=weights)
        dxi._install_lbfgs_stall_guard(model)
        start = _time.perf_counter()
        model.train()
        seconds = _time.perf_counter() - start
        exit_ = getattr(model, 'lbfgs_exit', None) or {}
        record.update({'iters': exit_.get('iterations'), 'exit': exit_.get('reason'),
                       'seconds': seconds,
                       'loss': self.adapter._reported_loss(model, weights)})
        self.windows.append(record)
        return model

    def train(self, k, net, observed):
        """Window ``k`` of a march: its own span, its share of the points and
        of the budget."""
        import deepxde as dde
        adapter = self.adapter
        t0, t1 = self.edges[k], self.edges[k + 1]
        bcs, n_obs = self.observations(t0, t1, left_closed=(k == 0)) if observed else ([], 0)
        n_domain = window_points(self.lengths[k], self.total, adapter.num_domain,
                                 MIN_WINDOW_DOMAIN)
        n_test = window_points(self.lengths[k], self.total, adapter.num_test, 1)
        model = dde.Model(self.problem(t0, t1, n_domain, n_test, bcs), net)
        return self.fit(model, self.loss_weights(bcs), self.caps[k], {
            'window': int(k), 't0': t0, 't1': t1,
            'levels': int(self.edge_levels[k + 1] - self.edge_levels[k]),
            'observed_points': n_obs, 'num_domain': n_domain, 'cap': int(self.caps[k])})

    def hand_off(self, nets, k):
        """Window ``k``'s network (a copy of window ``k-1``'s, warm start) with
        the hard transform holding it to a frozen copy of window ``k-1`` at
        ``e_k``."""
        net = copy.deepcopy(nets[-1])
        ic = net_edge_state(_frozen(nets[-1]), self.edges[k], max(self.orders), self.n_space)
        self.input_transform(net, k)
        net.apply_output_transform(hard_ic_transform(ic, self.edges[k], self.lengths[k],
                                                     self.orders, self.n_space))
        return net

    # ---- the field
    def predict(self, nets, first_window):
        """The full-grid prediction, flat in C order: every time level from the
        window that owns it (``owners``), boundary strips included. Levels
        before ``first_window`` belong to no window and stay NaN."""
        import torch
        X = _input_columns(self.grids)
        owner = owners(X[:, -1], self.edges[first_window + 1:-1]) + first_window
        owner[X[:, -1] < self.edges[first_window]] = -1
        out = np.full((X.shape[0], len(self.var_names)), np.nan)
        for k, net in enumerate(nets, start=first_window):
            rows = np.flatnonzero(owner == k)
            if rows.size:
                with torch.enable_grad():
                    y = net(torch.as_tensor(X[rows], dtype=self.dtype, device=self.device))
                out[rows] = y.detach().cpu().numpy()
        return out, owner

    def windows_loss(self):
        """The windows' own final losses, averaged with their lengths."""
        done = [(w['t1'] - w['t0'], w['loss']) for w in self.windows]
        return float(sum(a * b for a, b in done) / sum(a for a, _ in done))

    def finish(self, solutions, loss, windows_loss=True):
        """``windows_loss=False`` where the entries are not a partition of the
        extent (DE's nested stages, CW's restarts), so their length-weighted
        mean means nothing."""
        reasons = sorted({str(w['exit']) for w in self.windows})
        self.adapter.last_lbfgs_exit = {
            'reason': '+'.join(reasons),
            'iterations': int(sum(w['iters'] or 0 for w in self.windows))}
        if windows_loss:
            self.stats['loss_windows'] = self.windows_loss()
        self.stats['seconds'] = float(sum(w['seconds'] for w in self.windows))
        return [solutions[:, i].reshape(-1) for i in range(solutions.shape[1])], loss


def _input_columns(grids):
    import epde.integrate.deepxde_integration as dxi
    return dxi._input_columns(grids)


def _full(march: _March):
    """TM: window 0 is S1's network on the first window with its observations;
    each later window is handed off from its predecessor. Returns S1's own
    objective at the marched field as the solve's loss."""
    import deepxde as dde
    nets = []
    for k in range(len(march.lengths)):
        if k == 0:
            net = march.net                 # S1's network: its Glorot draw, its digest
            march.input_transform(net, 0)
        else:
            net = march.hand_off(nets, k)
        march.train(k, net, observed=True)
        nets.append(net)
    solutions, _ = march.predict(nets, 0)
    composite = _piecewise_net(nets, march.edges[1:-1])
    model = dde.Model(march.data_obj, composite)
    model.compile('L-BFGS')
    loss = march.adapter._reported_loss(model, march.weights)
    march.stats['loss_s1'] = loss
    return march.finish(solutions, loss)


def _tail_obs(march: _March):
    """TM-obs: TM's tail windows only, the first one started from the OBSERVED
    state at ``t_train``. Train-level points take the observed data (they are
    never scored; points with no observation stay NaN). Returns the windows'
    length-weighted loss (no network covers the train window, so S1's
    objective is not defined here)."""
    split_k = march.split_k
    state = observed_edge_state(march.train_levels, march.fields, march.split.t_train,
                                max(march.orders))
    if march.n_space == 0:
        ic = constant_edge_state(state, march.dtype, march.device)
    else:
        ic = _SplineEdgeState(march.axes[1:], state, march.dtype, march.device)
    nets = []
    for k in range(split_k, len(march.lengths)):
        if k == split_k:
            net = march.net                 # S1's Glorot network, now the first tail window
            march.input_transform(net, k)
            net.apply_output_transform(hard_ic_transform(ic, march.edges[k], march.lengths[k],
                                                         march.orders, march.n_space))
        else:
            net = march.hand_off(nets, k)
        march.train(k, net, observed=False)
        nets.append(net)
    solutions, _ = march.predict(nets, split_k)
    # before t_train: the observed data where there is any (train levels only;
    # march.data_list holds no held-out value)
    rows = np.flatnonzero(march.mask)
    before = march.grids[0].reshape(-1)[rows] < march.split.t_train
    observed = np.stack(march.data_list, axis=1)
    solutions[rows[before]] = observed[before]
    return march.finish(solutions, march.windows_loss())


def _expand(march: _March):
    """DE: ONE network -- S1's, with its Glorot draw and its input map, no hard
    transform -- trained on the growing domains ``[t_lo, e_s]``, s = 1..S, each
    stage a warm start of the last. Stage s holds the train observations with
    ``t <= e_s`` and ``round(num_domain * (e_s - t_lo) / T)`` physics points
    (S1's density); its L-BFGS budget is the share of the time it ADDS,
    ``e_s - e_{s-1}``, so the caps split the total as TM's do. The last stage
    IS S1's problem -- its data object, so its geometry, points and
    observations -- and the solve's loss is S1's own objective at the final
    network, predicted the way S1 predicts."""
    import deepxde as dde
    adapter, net = march.adapter, march.net
    t_lo, last = march.edges[0], len(march.lengths)
    model = None
    for s in range(1, last + 1):
        t1 = march.edges[s]
        if s == last:
            data, n_domain = march.data_obj, int(adapter.num_domain)
            n_obs = int(np.sum(march.split.train))
            weights = list(march.weights)
        else:
            bcs, n_obs = march.observations(t_lo, t1, left_closed=True)
            n_domain = window_points(t1 - t_lo, march.total, adapter.num_domain, 1)
            n_test = window_points(t1 - t_lo, march.total, adapter.num_test, 1)
            data = march.problem(t_lo, t1, n_domain, n_test, bcs)
            weights = march.loss_weights(bcs)
        model = march.fit(dde.Model(data, net), weights, march.caps[s - 1], {
            'stage': int(s), 't0': t_lo, 't1': t1,
            'levels': int(march.edge_levels[s] - march.edge_levels[0]),
            'observed_points': n_obs, 'num_domain': n_domain,
            'cap': int(march.caps[s - 1]), 's1_problem': s == last})
    loss = march.windows[-1]['loss']        # S1's objective, S1's weights
    march.stats['loss_s1'] = loss
    solutions = model.predict(_input_columns(march.grids))
    return march.finish(solutions, loss, windows_loss=False)


def _chunk_summary(losses):
    L = np.asarray(losses, dtype=np.float64)
    return {'sum': float(L.sum()), 'max': float(L.max()), 'argmax': int(np.argmax(L)),
            'first': float(L[0]), 'last': float(L[-1])}


def _causal(march: _March):
    """CW: S1's network, points and observations, with the causal weights
    folded into every point's error (``causal_problem``: a shallow copy of
    S1's data object, so not one point moves).

    At each restart the per-chunk losses ``L_j`` (``chunk_losses``) give
    ``w = causal_weights(L, eps)``, ``eps = M / sum_j L_j`` read ONCE at the
    Glorot init; the weights are constant for the restart's
    ``min(CAUSAL_RESTART, what is left)`` iterations, a fresh L-BFGS through
    the adapter's own machinery. Once ``min w >= CAUSAL_DONE`` the weights are
    dropped: the rest of the budget is one L-BFGS phase on S1's own problem.
    The restarts stop early when a phase diverged (``_CAUSAL_STOPS``) or moved
    no parameter (the next restart would repeat it exactly). The last iterate
    is scored; the solve's loss is S1's own objective there."""
    import deepxde as dde
    adapter, net, data_obj = march.adapter, march.net, march.data_obj
    budget = int(adapter.lbfgs_maxiter)
    scale = _CausalScale(march.chunk_edges, march.dtype, march.device)
    weighted = causal_problem(data_obj, scale)

    start = _time.perf_counter()
    losses = chunk_losses(net, data_obj, march.weights, scale)
    total = float(np.sum(losses))
    if not math.isfinite(total):
        raise FloatingPointError(f'causal weights: the chunk losses at the init are not '
                                 f'finite ({total})')
    eps = march.n_chunks / total if total > 0.0 else 0.0
    march.stats.update({'eps': eps, 'chunk_loss0': [float(v) for v in losses]})
    used, restart, model = 0, 0, None
    while used < budget:
        w = causal_weights(losses, eps)
        plain = float(w.min()) >= CAUSAL_DONE
        record = {'restart': restart, 'min_w': float(w.min()),
                  'front': int(np.argmax(w < 0.5)) if (w < 0.5).any() else len(w),
                  'plain': plain, 'chunk_loss': _chunk_summary(losses),
                  'reweigh_seconds': _time.perf_counter() - start}
        if plain:                             # w == 1: S1's own objective from here on
            cap = budget - used
            model = dde.Model(data_obj, net)
        else:
            cap = min(int(CAUSAL_RESTART), budget - used)
            scale.set(w)
            model = dde.Model(weighted, net)
        record['cap'] = int(cap)
        before = [p.detach().clone() for p in net.parameters()]
        march.fit(model, list(march.weights), cap, record)
        used += int(record['iters'] or 0)
        moved = any(not bool((p.detach() == q).all()) for p, q in zip(net.parameters(), before))
        if plain or record['exit'] in _CAUSAL_STOPS or not moved:
            break
        start = _time.perf_counter()
        losses = chunk_losses(net, data_obj, march.weights, scale)
        restart += 1
    if model.data is not data_obj:
        model = dde.Model(data_obj, net)      # read S1's objective, not the weighted one
        model.compile('L-BFGS')
    loss = adapter._reported_loss(model, march.weights)
    march.stats.update({
        'loss_s1': loss, 'plain_from': next((w['restart'] for w in march.windows
                                             if w['plain']), None),
        'chunk_loss_final': [float(v) for v in
                             chunk_losses(net, data_obj, march.weights, scale)],
        'reweigh_seconds': float(sum(w['reweigh_seconds'] for w in march.windows))})
    solutions = model.predict(_input_columns(march.grids))
    return march.finish(solutions, loss, windows_loss=False)


_MODES = {'full': _full, 'tail_obs': _tail_obs, 'expand': _expand, 'causal_frozen': _causal}


def solve(adapter, eq_list, var_names, grids, mask, data_list, split, inner_coords,
          pde_func, net, data_obj, weights):
    """The marching solve on S1's prepared problem (see ``_solve_trajectory``):
    ``net`` is S1's network at its Glorot init, ``weights`` S1's frozen loss
    weights, ``data_obj`` S1's full problem. Returns ``(solutions, loss)`` like
    the adapter's own solve: the full-grid field per variable, flat."""
    march = _March(adapter, eq_list, var_names, grids, mask, data_list, split,
                   inner_coords, pde_func, net, data_obj, weights)
    return _MODES[mode_of(adapter)](march)
