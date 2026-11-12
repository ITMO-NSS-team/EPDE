"""Weak-form rows for the solver-free search (``DomainConfig.evaluation_form='weak'``).

A term ``T`` of an equation slot becomes ONE weak column. Each entry is the average of ``T`` against a compactly
supported test function on one window of a lattice:

    w_i(T) = int T psi_i dx / int psi_i dx,    psi_i = prod_d (1 - s_d^2)^{q_d},    s_d = (x_d - c_{i,d}) / (m_d h_d)

Derivatives are moved onto ``psi`` by integration by parts (IBP), one (factor, axis) step at a time. A step is
taken only where it lowers the highest derivative order a data factor carries (``reduce``). If a step reproduces
the original monomial, it is solved for: ``u u_x -> -1/2 u^2 psi_x``. Coordinate factors fold in:
``x u_xx -> u (2 psi_x + x psi_xx)``. What cannot be moved stays as a pointwise factor, differentiated with the
fourth-order central stencil on the RAW fields (``_central4``).

Every term of a slot uses the slot's window (half-widths ``m``, row stride ``s``, smoothness ``q``). Every
candidate equation of the slot is therefore fitted and scored on the same rows. The rows form a lattice of shape
``WindowPlan.shape``, flattened in C order, so the grid-shaped instability metrics (``dyad``, ``tile``, ...)
partition it like a grid.

The window is chosen once per slot from the data (:func:`choose_window`, the subset-free ``pcr`` rule of the
Nov 2026 window study):
- per candidate window, Grams of the window's stride-m rows (centres one half-width apart). The Grams are
  taken on the data, on one AR(1) noise draw added to the data (noise model per field from its differences,
  :func:`noise_model`), and on the same physical windows of the every-other-sample grid at stride m/2. A window
  is admissible only if its stride-m rows determine the library fit (rows > p + 1). The fit rows of the chosen
  window may be denser (stride ``s``, set by the row floor);
- the score is a principal-component regression of a target on the library: keep the generalized
  eigen-directions of (G_X, Gn_X) whose signal exceeds the noise (rho > 2, i.e. SNR > 1). The relative shift of
  its coefficients under the noise draw and under step doubling, plus the squared noise-to-energy ratios of the
  feature columns, is the window's score;
- the median over the slot's candidate targets, then the minimum over windows.

Kernels are exact-moment quadrature weights (:func:`exact_kernel`). Each window's weights are divided by the
window's mass ``int psi dx``, so a constant's weak column is exactly 1 and the intercept keeps its meaning. The
factor is common to every column of a window, so coefficients are unchanged.

This module is pure numpy / scipy, with no EPDE state. The search-side glue (EPDE terms to ``WeakTerm``, the
per-slot plan store) lives in ``epde.structure``.

Fixed numerical structure (not tuned):
- the (1 - s^2)^q test-function family and the powers-of-2 half-width lattice;
- quadrature exact on the test function's own polynomial degree;
- smoothness starts at j = 0, 1, 2 above the minimum and is extended by one while the best window sits at the
  largest j tried;
- a noise draw seeded by the caller.

The only compute guard is ``J_GUARD``. It is never reached in the study, and reaching it raises.
"""
from itertools import product
from math import comb
from typing import Callable, Dict, Hashable, List, Optional, Sequence, Tuple

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from numpy.polynomial import polynomial as Pl
from scipy.optimize import brentq
from scipy.signal import lfilter

from epde.preprocessing.deriv_calculators import _central4

__all__ = ['WeakTerm', 'WindowPlan', 'Fields', 'reduce', 'ibp_options', 'max_order', 'exact_kernel',
           'window_kernels', 'wint_s', 'noise_model', 'ar_noise', 'weak_matrix', 'choose_window', 'pcr_score',
           'FD_REACH', 'J_START', 'J_GUARD']

#: Samples the fourth-order central stencil reaches per derivative order.
FD_REACH = 2
#: Initial smoothness offsets j (q = r + j) tried by :func:`choose_window`.
J_START = (0, 1, 2)
#: Compute guard on the smoothness extension: never reached in the window study, and reaching it raises.
J_GUARD = 16

EPS = np.finfo(float).eps

DataFactor = Tuple[Hashable, Tuple[int, ...]]
Key = Tuple[Tuple[DataFactor, ...], Tuple[int, ...], Tuple[int, ...]]


# ----------------------------------------------------------------------------------------------- symbolic terms
def max_order(key: Key) -> int:
    """Highest total derivative order a data factor of ``key`` carries (0 without data factors)."""
    return max((sum(o) for _, o in key[0]), default=0)


def _add(d: dict, k, c: float) -> None:
    d[k] = d.get(k, 0.0) + c


def ibp_options(key: Key) -> List[Dict[Key, float]]:
    """Every single IBP step of ``key``: take one derivative off one data factor along one axis and spread it,
    with a minus sign, over the other data factors, the coordinate powers and psi. A step that reproduces
    ``key`` itself (coefficient kappa) is solved for it: ``key = R / (1 - kappa)``."""
    data, coords, psi = key
    D = len(coords)
    out, seen = [], set()
    for j, (wv, al) in enumerate(data):
        if (wv, al) in seen:
            continue
        seen.add((wv, al))
        for e in range(D):
            if al[e] == 0:
                continue
            lo = list(al)
            lo[e] -= 1
            lowered = (wv, tuple(lo))
            rest = list(data)
            rest.pop(j)
            R = {}
            for i, (v2, a2) in enumerate(rest):
                up = list(a2)
                up[e] += 1
                nd = rest[:i] + [(v2, tuple(up))] + rest[i + 1:] + [lowered]
                _add(R, (tuple(sorted(nd)), coords, psi), -1.0)
            if coords[e] > 0:
                nc = list(coords)
                nc[e] -= 1
                _add(R, (tuple(sorted(rest + [lowered])), tuple(nc), psi), -float(coords[e]))
            npsi = list(psi)
            npsi[e] += 1
            _add(R, (tuple(sorted(rest + [lowered])), coords, tuple(npsi)), -1.0)
            kappa = R.pop(key, 0.0)
            if kappa == 1.0:
                continue
            out.append({k: c / (1.0 - kappa) for k, c in R.items() if c != 0.0})
    return out


def reduce(key: Key, coef: float = 1.0, psi_cap: Optional[Sequence[int]] = None) -> Dict[Key, float]:
    """The monomials of ``coef * key`` after IBP: steps that lower the highest data-derivative order, chosen by
    (new highest order, number of monomials), applied until none lowers it. ``psi_cap`` (per axis) drops the
    steps that would put more derivatives on psi than the test function's smoothness allows (IBP needs
    ``g <= q``); the term then keeps more of its derivatives pointwise. Every result is an exact identity."""
    m0 = max_order(key)
    if m0 == 0:
        return {key: coef}
    best = None
    for R in ibp_options(key):
        if not R:
            continue
        if psi_cap is not None and any(k[2][d] > psi_cap[d] for k in R for d in range(len(psi_cap))):
            continue
        m1 = max(max_order(k) for k in R)
        if m1 < m0 and (best is None or (m1, len(R)) < best[0]):
            best = ((m1, len(R)), R)
    if best is None:
        return {key: coef}
    out = {}
    for k, r in best[1].items():
        for k2, c2 in reduce(k, coef * r, psi_cap).items():
            _add(out, k2, c2)
    return {k: c for k, c in out.items() if c != 0.0}


class WeakTerm:
    """A term's symbolic form.

    The parts:
    - ``data``: data factors ``(variable, per-axis derivative orders)``, one entry per unit of power;
    - ``coords``: per-axis powers of the coordinate;
    - ``opaque``: ids of pointwise factors whose values the caller supplies on the full grid (trigonometric,
      constant, user tokens).

    A term with an opaque factor gets no IBP: it cannot be differentiated symbolically, so its data factors keep
    their derivatives pointwise."""
    __slots__ = ('data', 'coords', 'opaque')

    def __init__(self, data: Sequence[DataFactor], coords: Sequence[int], opaque: Sequence[Hashable] = ()):
        self.data = tuple(sorted((v, tuple(int(x) for x in o)) for v, o in data))
        self.coords = tuple(int(c) for c in coords)
        self.opaque = tuple(sorted(opaque, key=repr))
        if any(len(o) != len(self.coords) for _, o in self.data):
            raise ValueError(f'derivative orders {self.data} do not match {len(self.coords)} axes.')

    @property
    def dims(self) -> int:
        return len(self.coords)

    def monomials(self, psi_cap: Optional[Sequence[int]] = None) -> Dict[Key, float]:
        key = (self.data, self.coords, (0,) * self.dims)
        return {key: 1.0} if self.opaque else reduce(key, psi_cap=psi_cap)

    def __eq__(self, other):
        return isinstance(other, WeakTerm) and (self.data, self.coords, self.opaque) == \
            (other.data, other.coords, other.opaque)

    def __hash__(self):
        return hash((self.data, self.coords, self.opaque))

    def __repr__(self):
        return f'WeakTerm(data={self.data}, coords={self.coords}, opaque={self.opaque})'


def leftover_orders(monos: Dict[Key, float], dims: int) -> List[int]:
    """Per axis, the highest derivative order a data factor keeps pointwise across ``monos``."""
    out = [0] * dims
    for data, _, _ in monos:
        for _, o in data:
            out = [max(out[d], o[d]) for d in range(dims)]
    return out


def psi_orders(monos: Dict[Key, float], dims: int) -> List[int]:
    out = [0] * dims
    for _, _, psi in monos:
        out = [max(out[d], psi[d]) for d in range(dims)]
    return out


# ------------------------------------------------------------------------------------------------ fields
class Fields:
    """Field values on the full grid, their nested fourth-order derivatives (cached), broadcast coordinates and
    the opaque factors' values."""

    def __init__(self, values: Dict[Hashable, np.ndarray], axes: Sequence[np.ndarray],
                 opaque: Optional[Dict[Hashable, np.ndarray]] = None):
        self.values = {k: np.asarray(v, dtype=float) for k, v in values.items()}
        self.axes = [np.asarray(a, dtype=float).reshape(-1) for a in axes]
        self.D = len(self.axes)
        self.shape = tuple(a.size for a in self.axes)
        for k, v in self.values.items():
            if v.shape != self.shape:
                raise ValueError(f'field {k!r} has shape {v.shape}, the grid {self.shape}.')
        self.opaque = {k: np.broadcast_to(np.asarray(v, dtype=float), self.shape) for k, v in (opaque or {}).items()}
        self.steps = [float(a[1] - a[0]) if a.size > 1 else 1.0 for a in self.axes]
        self.coords = [a.reshape([-1 if b == d else 1 for b in range(self.D)]) for d, a in enumerate(self.axes)]
        self._cache = {}

    def get(self, var: Hashable, orders: Tuple[int, ...]) -> np.ndarray:
        key = (var, orders)
        if key not in self._cache:
            if sum(orders) == 0:
                self._cache[key] = self.values[var]
            else:
                a = max(i for i, o in enumerate(orders) if o > 0)
                lo = list(orders)
                lo[a] -= 1
                self._cache[key] = _central4(self.get(var, tuple(lo)), self.axes[a], a)
        return self._cache[key]

    def point(self, data, coords, opaque=()) -> np.ndarray:
        """Pointwise product of data factors, coordinate powers and opaque factors on the full grid."""
        out = np.ones(self.shape)
        for var, orders in data:
            out = out * self.get(var, orders)
        for d, p in enumerate(coords):
            if p:
                out = out * self.coords[d] ** p
        for o in opaque:
            out = out * self.opaque[o]
        return out

    def every_other(self) -> 'Fields':
        """The same fields on the every-other-sample grid of every axis."""
        sl = tuple(slice(None, None, 2) for _ in range(self.D))
        return Fields({k: v[sl] for k, v in self.values.items()}, [a[::2] for a in self.axes],
                      {k: v[sl] for k, v in self.opaque.items()})

    def with_values(self, values: Dict[Hashable, np.ndarray]) -> 'Fields':
        return Fields(values, self.axes, self.opaque)


# ----------------------------------------------------------------------------------------------- kernels
_KCACHE: Dict[Tuple[int, int, int], np.ndarray] = {}


def exact_kernel(m: int, q: int, g: int) -> np.ndarray:
    """Weights (in s, ds = 1/m) of int_{-1}^{1} f(s) phi^(g)(s) ds on the 2m + 1 samples, phi = (1 - s^2)^q.

    The sampled analytic derivative (trapezoid) gets the minimum-norm change that makes the weights reproduce
    the exact moments int phi^(g) P_j for Legendre P_j, j <= min(2m, 2q). Constants and low-order trends in the
    data then cancel exactly, as under an FD stencil."""
    key = (m, q, g)
    if key not in _KCACHE:
        s = np.arange(-m, m + 1) / m
        c = Pl.polyder(Pl.polypow([1.0, 0.0, -1.0], q), g) if g else Pl.polypow([1.0, 0.0, -1.0], q)
        a = Pl.polyval(s, c) / m
        P = min(2 * m, 2 * q)
        V = np.polynomial.legendre.legvander(s, P)
        mu = np.empty(P + 1)
        for j in range(P + 1):
            prod_ = Pl.polyint(Pl.polymul(c, np.polynomial.legendre.leg2poly(np.eye(P + 1)[j])))
            mu[j] = Pl.polyval(1.0, prod_) - Pl.polyval(-1.0, prod_)
        a = a + V @ np.linalg.lstsq(V.T @ V, mu - V.T @ a, rcond=None)[0]
        _KCACHE[key] = a
    return _KCACHE[key]


def window_kernels(ms: Sequence[int], steps: Sequence[float], q: Sequence[int], g_max: Sequence[int]):
    """``K[d][g]``, g <= g_max[d]: the weights of int f D^g psi dx / int psi dx along axis d."""
    K = []
    for d in range(len(ms)):
        m, h = int(ms[d]), float(steps[d])
        mass = float(exact_kernel(m, q[d], 0).sum()) * m * h       # int psi dx (the P_0 moment is exact)
        K.append([exact_kernel(m, q[d], g) * (m * h) ** (1 - g) / mass for g in range(int(g_max[d]) + 1)])
    return K


def wint_s(f: np.ndarray, gam: Sequence[int], ms: Sequence[int], ss: Sequence[int], K) -> np.ndarray:
    """Separable window integrals of ``f`` against ``D^gam psi`` with half-widths ``ms`` and row strides ``ss``:
    an array of the lattice shape."""
    for a in range(f.ndim):
        w = K[a][gam[a]]
        v = sliding_window_view(f, w.size, axis=a)
        sl = [slice(None)] * v.ndim
        sl[a] = slice(None, None, ss[a])
        f = v[tuple(sl)] @ w
    return f


def lattice_shape(n_eff: Sequence[int], ms: Sequence[int], ss: Sequence[int]) -> Tuple[int, ...]:
    return tuple((int(n) - 2 * int(m) - 1) // int(s) + 1 for n, m, s in zip(n_eff, ms, ss))


# ------------------------------------------------------------------------------------------ noise model
def _g(m: int, rho: float, j: int) -> float:
    c = np.array([(-1) ** (m - i) * comb(m, i) for i in range(m + 1)], float)
    lag = j + np.arange(m + 1)[:, None] - np.arange(m + 1)[None, :]
    return float(np.sum(np.outer(c, c) * rho ** np.abs(lag)))


def axis_fit(y: np.ndarray, a: int):
    """``(sigma, rho, m)`` along axis ``a``, or ``None`` when no difference order is noise-dominated.

    The scan runs over orders m = 1, 2, ... (iterated differences). The lag-1 / lag-0 ratio of the m-th
    differences, which for AR(1) noise lies in (-1, -(m-1)/m) and is monotone in rho, is inverted for rho. An
    order whose ratio falls outside that range is signal-dominated and skipped. The fit is kept while its sigma
    falls (first local minimum).

    A fit whose sigma^2 exceeds the data's mean variance along the axis is impossible (the noise is part of the
    data), so that order is skipped like a signal-dominated one. It happens where the low orders are smoother
    than any AR(1) noise and the ratio first enters the range at its rho -> 1 edge; there
    sigma = sqrt(r0 / g(m, rho, 0)) diverges. Example: a robot heading sin(phi), 8250 samples, gave sigma 431 at
    order 196 and choose_window reached J_GUARD.

    The scan also ends:
    - at the first local minimum of the white-equivalent sigma, once a fit exists;
    - at the data limits: 2m + 2 >= n, roundoff of the samples, or float overflow."""
    n = y.shape[a]
    d = np.moveaxis(y, a, 0)
    var0 = float(np.mean(np.var(d, axis=0)))
    floor = EPS * float(np.sqrt(np.mean(y * y)))
    best, prev, m = None, None, 0
    while n > 2 * (m + 1) + 2:
        m += 1
        d = np.diff(d, axis=0)
        r0 = float(np.mean(d * d))
        cm = comb(2 * m, m)
        if not np.isfinite(r0) or cm > 1e300:
            break
        sw = np.sqrt(r0 / cm)
        if sw <= floor:
            break
        stop = prev is not None and sw >= prev
        prev = sw
        ratio = float(np.mean(d[1:] * d[:-1])) / r0
        f = lambda rho: _g(m, rho, 1) / _g(m, rho, 0) - ratio
        lo, hi = -1.0 + 1e-9, 1.0 - 1e-9
        if f(lo) * f(hi) < 0:
            rho = brentq(f, lo, hi)
            sig = float(np.sqrt(r0 / _g(m, rho, 0)))
            if sig * sig <= var0:
                if best is not None and sig >= best[0]:
                    break
                best = (sig, rho, m)
        if stop and best is not None:
            break
    return best


def noise_model(y: np.ndarray):
    """Separable AR(1) noise of a field: ``(sigma, [rho per axis], [difference order per axis])``. Sigma is the
    smallest per-axis estimate (least signal leakage); 0 when no axis has a noise-dominated order. Sigma never
    exceeds the data's variance along an axis."""
    fits = [axis_fit(y, a) for a in range(y.ndim)]
    ok = [f for f in fits if f is not None]
    sig = min(f[0] for f in ok) if ok else 0.0
    rhos = [f[1] if f is not None else 0.0 for f in fits]
    return sig, rhos, [f[2] if f is not None else 0 for f in fits]


def ar_noise(rng: np.random.Generator, shape, sig: float, rhos: Sequence[float]) -> np.ndarray:
    """A stationary separable AR(1) draw of standard deviation ``sig``."""
    e = rng.standard_normal(shape)
    for a, rho in enumerate(rhos):
        if rho != 0.0:
            s = np.sqrt(1.0 - rho ** 2)
            x0 = np.take(e, [0], axis=a)
            e = lfilter([s], [1.0, -rho], e, axis=a, zi=(1.0 - s) * x0)[0]
    return sig * e


# ----------------------------------------------------------------------------------------------- matrix
def _crop(arr: np.ndarray, cut: Sequence[int]) -> np.ndarray:
    return arr[tuple(slice(c, arr.shape[d] - c) for d, c in enumerate(cut))]


def check_cut(monos: Dict[Key, float], dims: int, cut: Sequence[int]) -> None:
    """Every pointwise derivative must stay inside the cut, where the interior stencil applies."""
    left = leftover_orders(monos, dims)
    if any(FD_REACH * left[d] > cut[d] for d in range(dims)):
        raise ValueError(f'a weak column keeps derivatives of orders {left} pointwise; the slot\'s edge cut '
                         f'{list(cut)} covers only {[c // FD_REACH for c in cut]}.')


def weak_matrix(monos_list: Sequence[Dict[Key, float]], opaque_list: Sequence[Tuple], fields: Fields,
                cut: Sequence[int], ms: Sequence[int], ss: Sequence[int], q: Sequence[int],
                cache: Optional[dict] = None) -> Tuple[np.ndarray, Tuple[int, ...]]:
    """Weak columns of terms given by their monomials (``monos_list[k]``) and opaque ids (``opaque_list[k]``) on
    the window ``(ms, ss, q)``, after cropping ``cut`` samples off each end of every axis: ``(rows, lattice
    shape)``. ``cache`` (a dict owned by the caller) holds the cropped pointwise products for reuse."""
    D = fields.D
    g_max = [max([k[2][d] for monos in monos_list for k in monos] or [0]) for d in range(D)]
    if any(g_max[d] > q[d] for d in range(D)):
        raise ValueError(f'psi orders {g_max} exceed the test function\'s smoothness {list(q)}.')
    K = window_kernels(ms, fields.steps, q, g_max)
    cache = {} if cache is None else cache
    ints = {}
    cols, shape = [], None
    for monos, opq in zip(monos_list, opaque_list):
        acc = 0.0
        for (data, coords, psi), c in monos.items():
            pk = (data, coords, tuple(opq))
            if pk not in cache:
                cache[pk] = _crop(fields.point(data, coords, opq), cut)
            ik = (pk, psi)
            if ik not in ints:
                ints[ik] = wint_s(cache[pk], psi, ms, ss, K)
            shape = ints[ik].shape
            acc = acc + c * ints[ik].reshape(-1)
        cols.append(acc)
    return np.column_stack(cols), shape


# ------------------------------------------------------------------------------------------- selection
def _solve(A: np.ndarray, b: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.solve(A, b)
    except np.linalg.LinAlgError:
        return np.linalg.lstsq(A, b, rcond=None)[0]


def signal_directions(X: np.ndarray, N: np.ndarray) -> np.ndarray:
    """The generalized eigen-directions of the signal Gram ``X`` and the noise Gram ``N`` whose signal exceeds
    the noise, ``rho = v'Xv / v'Nv > 2`` (SNR > 1). They are returned in unit-diagonal coordinates
    (``X / sqrt(diag X diag X')``) and X-orthonormal there.

    The problem is solved as a standard eigenproblem of the noise in X-whitened coordinates. ``X`` is truncated to
    its numerical rank: eigenvalues above ``max * p * eps``, the matrix-rank convention. A noise-free column (a
    coordinate token) then gives an exact zero noise eigenvalue instead of the ill-conditioned ``eigh(X, N)``.
    Where both Grams are positive definite, the kept subspace is the one ``eigh(X, N)`` gives."""
    s = 1.0 / np.sqrt(np.diag(X))
    Xs = s[:, None] * X * s[None, :]
    Ns = s[:, None] * N * s[None, :]
    w, U = np.linalg.eigh(Xs)
    keep = w > w.max() * len(w) * EPS
    B = U[:, keep] / np.sqrt(w[keep])                        # B' Xs B = I
    lam, Z = np.linalg.eigh(B.T @ Ns @ B)                    # noise / signal per direction
    return (B @ Z)[:, lam < 0.5]


def pcr_score(G: np.ndarray, Gd: np.ndarray, Gc: np.ndarray, Gn: np.ndarray, t: int) -> float:
    """The ``pcr`` window score of target column ``t`` regressed on the other columns.

    Steps:
    - Keep the generalized eigen-directions of (G_X, Gn_X) with rho > 2: signal energy above the noise energy
      (SNR > 1).
    - Regress on those directions with the rows (c), the noise-draw rows (c_d) and the coarse rows (c_c).
    - Score = (||c_d - c||^2 + ||c - c_c||^2) / ||c||^2 + sum_k a_k^2, where a_k = noise / energy of feature
      column k.

    ``inf`` when the coarse Gram is missing or no direction is kept."""
    if not np.all(np.isfinite(Gc)) or not np.all(np.isfinite(Gd)) or not np.all(np.isfinite(G)):
        return np.inf
    p = G.shape[0]
    rest = np.array([k for k in range(p) if k != t])
    if rest.size == 0:
        return np.inf
    X, Xd, Xc, N = (A[np.ix_(rest, rest)] for A in (G, Gd, Gc, Gn))
    b, bd, bc = G[rest, t], Gd[rest, t], Gc[rest, t]
    dx = np.diag(X)
    if not np.all(dx > 0):
        return np.inf
    a = np.diag(N) / dx
    Vk = signal_directions(X, N)
    if Vk.shape[1] == 0:
        return np.inf
    s = 1.0 / np.sqrt(dx)                                    # the directions live in unit-diagonal coordinates

    def pc(Gx, bx):
        A = Vk.T @ (s[:, None] * Gx * s[None, :]) @ Vk
        return s * (Vk @ _solve(A, Vk.T @ (s * bx)))

    c, cd, cc = pc(X, b), pc(Xd, bd), pc(Xc, bc)
    den = float(np.sum(c ** 2))
    if not den > 0:
        return np.inf
    s = (float(np.sum((cd - c) ** 2)) + float(np.sum((c - cc) ** 2))) / den + float(np.sum(a ** 2))
    return s if np.isfinite(s) else np.inf


class WindowPlan:
    """A slot's window.

    Fields:
    - ``q``: per-axis test-function smoothness, ``q = r + j``;
    - ``ms``: half-widths;
    - ``ss``: row strides;
    - ``cut``: samples cropped off each end of every axis;
    - ``shape``: the row lattice;
    - ``score``: the pcr median;
    - ``table``: one ``(j, ms, ss, shape, score)`` per candidate window, for diagnostics."""
    __slots__ = ('j', 'r', 'q', 'ms', 'ss', 'cut', 'shape', 'score', 'table')

    def __init__(self, j, r, ms, ss, cut, shape, score, table=()):
        self.j, self.r = int(j), tuple(int(x) for x in r)
        self.q = tuple(x + self.j for x in self.r)
        self.ms, self.ss = tuple(int(x) for x in ms), tuple(int(x) for x in ss)
        self.cut, self.shape = tuple(int(x) for x in cut), tuple(int(x) for x in shape)
        self.score, self.table = float(score), list(table)

    @property
    def rows(self) -> int:
        return int(np.prod(self.shape))

    def __repr__(self):
        return (f'WindowPlan(j={self.j}, q={self.q}, m={self.ms}, s={self.ss}, cut={self.cut}, '
                f'rows={self.shape}, score={self.score:.3g})')


def _stride(n_eff, ms, p, rows_ok):
    """The row stride: start at the half-width (50% overlap) and halve until the rows determine a fit of the
    whole library (rows > p + 1) and pass ``rows_ok`` (the instability metric's floor); ``None`` if even
    stride 1 does not."""
    ss = list(ms)
    while True:
        shape = lattice_shape(n_eff, ms, ss)
        if min(shape) >= 1 and int(np.prod(shape)) > p + 1 and rows_ok(shape):
            return tuple(ss), shape
        if max(ss) == 1:
            return None
        ss = [max(1, s // 2) for s in ss]


def choose_window(fields: Fields, library: Sequence[WeakTerm], targets: Sequence[int],
                  rng: np.random.Generator, cut: Optional[Sequence[int]] = None,
                  rows_ok: Callable[[Tuple[int, ...]], bool] = lambda shape: True,
                  noise_vars: Optional[Sequence[Hashable]] = None) -> WindowPlan:
    """The slot's window by the pcr rule (module docstring).

    Arguments:
    - ``library``: the selection library, one ``WeakTerm`` per column;
    - ``targets``: indices of the slot's candidate targets in ``library``;
    - ``cut``: per-axis edge crop on the full grid. The default is the library's own pointwise derivatives,
      ``FD_REACH * order`` doubled for the every-other-sample grid;
    - ``rows_ok``: the instability metric's row floor on a lattice shape;
    - ``noise_vars``: the fields that receive the noise draw (default: all).

    Raises ``ValueError`` when no window is feasible."""
    D = fields.D
    if not targets:
        raise ValueError('the slot has no candidate target in its selection library.')
    monos = [t.monomials() for t in library]
    opq = [t.opaque for t in library]
    r = [0] * D
    left = [0] * D
    for mo in monos:
        r = [max(a, b) for a, b in zip(r, psi_orders(mo, D))]
        left = [max(a, b) for a, b in zip(left, leftover_orders(mo, D))]
    if cut is None:
        cut = [2 * FD_REACH * x for x in left]
    cut = [int(c) for c in cut]
    if any(c % 2 for c in cut):
        raise ValueError(f'the edge cut {cut} must be even (the every-other-sample grid keeps half of it).')
    for mo in monos:
        check_cut(mo, D, [c // 2 for c in cut])               # the coarse grid's reach, in its own samples
    cut_s = [c // 2 for c in cut]
    coarse = fields.every_other()
    names = list(fields.values) if noise_vars is None else list(noise_vars)
    drawn = dict(fields.values)
    for k in names:
        sig, rhos, _ = noise_model(fields.values[k])
        drawn[k] = fields.values[k] + ar_noise(rng, fields.values[k].shape, sig, rhos)
    draw = fields.with_values(drawn)
    n_eff = [fields.shape[d] - 2 * cut[d] for d in range(D)]
    n_eff_s = [coarse.shape[d] - 2 * cut_s[d] for d in range(D)]
    per_axis = []
    for d in range(D):
        ms_d, m = [], 2
        while n_eff[d] >= 2 * m + 1:
            ms_d.append(m)
            m *= 2
        per_axis.append(ms_d)
    p = len(library)
    caches = ({}, {}, {})
    table = []
    js = list(J_START)
    done = set()
    while True:
        for j in js:
            if j in done:
                continue
            done.add(j)
            q = [x + j for x in r]
            for ms in product(*per_axis):
                st = _stride(n_eff, ms, p, rows_ok)
                if st is None:
                    continue
                ss, shape = st
                msc = [max(1, m // 2) for m in ms]
                if any(n_eff_s[d] < 2 * msc[d] + 1 for d in range(D)):
                    continue
                # the score reads the window's own (non-overlapping-centre) rows, stride m, and the coarse grid at
                # stride m/2; the window is admissible only if those rows determine the library fit
                if int(np.prod(lattice_shape(n_eff, ms, ms))) <= p + 1:
                    continue
                M, _ = weak_matrix(monos, opq, fields, cut, ms, ms, q, caches[0])
                Md, _ = weak_matrix(monos, opq, draw, cut, ms, ms, q, caches[1])
                Ms, _ = weak_matrix(monos, opq, coarse, cut_s, msc, msc, q, caches[2])
                G, Gd = M.T @ M, Md.T @ Md
                Gc = Ms.T @ Ms if Ms.shape[0] > Ms.shape[1] else np.full(G.shape, np.nan)
                dM = Md - M
                Gn = dM.T @ dM
                score = float(np.median([pcr_score(G, Gd, Gc, Gn, t) for t in targets]))
                table.append((j, tuple(ms), ss, shape, score))
        if not table:
            raise ValueError(f'no feasible weak-form window: grid {fields.shape}, edge cut {cut}, '
                             f'{p} library columns.')
        best = min(table, key=lambda x: (x[4], x[0]))
        if not np.isfinite(best[4]):
            raise ValueError('no weak-form window has a finite score (no direction with signal above noise).')
        top = max(js)
        if best[0] < top or not any(x[0] == top and np.isfinite(x[4]) for x in table):
            break
        if top + 1 > J_GUARD:
            raise RuntimeError(f'weak-form smoothness extension reached the compute guard J_GUARD={J_GUARD}.')
        js.append(top + 1)
    j, ms, ss, shape, score = best
    return WindowPlan(j, r, ms, ss, cut, shape, score, table)
