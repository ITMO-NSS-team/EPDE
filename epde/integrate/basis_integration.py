"""Fixed-basis backend for ``SolverBasedFitness``: no network, no learning rate.

The candidate's field is a tensor-product spectral expansion,

    u_v(t, x) = sum_ij C_v[i, j] T_i(t) psi_j(x),

with Chebyshev polynomials in time (and on non-periodic space axes) and a
Fourier series on space axes the DATA show to be periodic
(``data_profile.build_profile``, train window only). The coefficients are
optimised by the SAME torch L-BFGS DeepXDE runs (lr=1, strong-Wolfe line
search, history 100, gtol 1e-8, ftol 0), with DeepXDE's chunked loop and its
exit record, on the loss the DeepXDE adapter uses:

    loss = sum_e w_e * mean(residual_e^2 on collocation nodes over the FULL domain)
         + sum_v w_v * mean((u_v - u_obs_v)^2 on the inner TRAIN points)

The candidate is then scored by the host on the held-out tail, exactly as for
DeepXDE (same ``solve()`` contract, same ``time_split``).

What the basis buys: the only length scale L-BFGS needs is curvature, which
the line search supplies. There is no seed (the start is either zero or the
data-only least-squares fit), everything is float64 on the CPU, and it is
bit-reproducible across processes when the thread count is pinned. deepxde is
never imported.

Choices and why:

* Time is never periodic. Its size is ``time_factor`` x the profile's
  Chebyshev count on the train window, which is already stretched to the full
  extent.
* A periodic axis starts at the data grid's Nyquist wavenumber. The data do
  NOT bound the resolution the candidate's solution needs: Allen-Cahn's fronts
  are ~0.014 wide, finer than its grid spacing (0.016). So the solve checks its
  own resolution (``refine='spectral'``). If the top third of an axis's modes
  holds more than ``1 - energy`` of the solution's energy on that axis, the
  axis is doubled and the solve is re-run from the padded coefficients. The
  threshold is the same ``energy`` that sized the basis from the data, so this
  adds no constant, and it reads only the solution, never the held-out data.
  Measured before the rule (fixed Fourier sizes, AC): 1x Nyquist gate +0.20
  (under-resolved), 2x +4.16 (the no-diffusion form rings on its
  near-discontinuities), 3x +0.845, 4x +0.826 (converged; the network reaches
  +0.829).
* Collocation uses Chebyshev-Gauss-Lobatto nodes (time, non-periodic space)
  and a uniform grid (periodic space), ``collocation_factor`` x basis size per
  axis. The factor 2 de-aliases cubic products.
* Weights are ``'variance'``, the data yardsticks of the DeepXDE adapter. The
  gradient-norm weights DeepXDE ships are undefined at both natural starts
  here: the observation gradient vanishes at the data fit, and the physics
  gradient vanishes at C = 0 for a homogeneous PDE.
* ``pde_loss``: ``'mse'`` (the residual as is); ``'sinv_live'`` divides it
  pointwise by the term mass of the CURRENT field (differentiated);
  ``'sinv_data'`` divides by the term mass of the DATA (EPDE's
  finite-difference terms on the train window, frozen; the tail copies the
  last train level). The term mass is the denominator of
  ``Discrepancy._compute_scale_invariant``. The scale-invariant losses are
  dimensionless, so their weight is 1.
"""
import math
import time
from typing import List, Optional

import numpy as np
import torch

import epde.globals as global_var
from epde.integrate.data_profile import axis_coordinates, build_profile
from epde.integrate.heldout import DeepXDEConfigError, time_split, variance_weight
from epde.integrate.residual_terms import (cancellation_ratio, inverse_mass,
                                           mass_diagnostics, required_fields, residual,
                                           system_specs)

DTYPE = torch.float64
DEVICE = torch.device("cpu")
PDE_LOSSES = ("mse", "sinv_live", "sinv_data")
STARTS = ("data_lstsq", "zero")


def default_basis_config() -> dict:
    """The shipped defaults, declared in ``search_config`` (the one place
    settings are declared)."""
    from epde.interface.search_config import _default_basis_config
    return _default_basis_config()


# ============================================================ 1-D bases
class Chebyshev:
    """T_0..T_{size-1} on [lo, hi]."""
    kind = "chebyshev"

    def __init__(self, lo: float, hi: float, size: int):
        if size < 1:
            raise ValueError("a basis needs at least one function")
        self.lo, self.hi, self.size = float(lo), float(hi), int(size)
        self._deriv = {}

    def _derivative_coefficients(self, order: int) -> np.ndarray:
        if order not in self._deriv:
            from numpy.polynomial import chebyshev as C
            D = np.zeros((self.size, self.size))
            for j in range(self.size):
                e = np.zeros(self.size)
                e[j] = 1.0
                d = C.chebder(e, m=order) if order else e
                D[:len(d), j] = d
            self._deriv[order] = D
        return self._deriv[order]

    def matrix(self, points, order: int = 0) -> np.ndarray:
        from numpy.polynomial import chebyshev as C
        p = np.asarray(points, dtype=np.float64)
        xm = 2.0 * (p - self.lo) / (self.hi - self.lo) - 1.0
        V = C.chebvander(xm, self.size - 1)
        if order == 0:
            return V
        scale = (2.0 / (self.hi - self.lo)) ** order
        return (V @ self._derivative_coefficients(order)) * scale

    def refined(self):
        return Chebyshev(self.lo, self.hi, 2 * self.size)

    def modal_energy(self, C: np.ndarray, axis: int) -> np.ndarray:
        """Energy per degree along ``axis`` of a coefficient array."""
        other = tuple(a for a in range(C.ndim) if a != axis)
        return (C ** 2).sum(axis=other)

    def nodes(self, count: int) -> np.ndarray:
        """Chebyshev-Gauss-Lobatto nodes on [lo, hi], ascending."""
        count = max(int(count), 2)
        k = np.arange(count)
        return 0.5 * (self.lo + self.hi) - 0.5 * (self.hi - self.lo) * np.cos(np.pi * k / (count - 1))


class Fourier:
    """1, cos(w_k x), sin(w_k x) for k = 1..K, w_k = 2 pi k / period."""
    kind = "fourier"

    def __init__(self, lo: float, period: float, K: int):
        self.lo, self.period, self.K = float(lo), float(period), int(K)
        self.size = 2 * self.K + 1

    def matrix(self, points, order: int = 0) -> np.ndarray:
        p = np.asarray(points, dtype=np.float64) - self.lo
        out = np.empty((len(p), self.size))
        out[:, 0] = 1.0 if order == 0 else 0.0
        shift = order * np.pi / 2.0
        for k in range(1, self.K + 1):
            w = 2.0 * np.pi * k / self.period
            out[:, 2 * k - 1] = w ** order * np.cos(w * p + shift)
            out[:, 2 * k] = w ** order * np.sin(w * p + shift)
        return out

    def refined(self):
        return Fourier(self.lo, self.period, 2 * self.K)

    def modal_energy(self, C: np.ndarray, axis: int) -> np.ndarray:
        """Energy per wavenumber 0..K along ``axis`` (cos and sin summed)."""
        other = tuple(a for a in range(C.ndim) if a != axis)
        col = (C ** 2).sum(axis=other)
        return np.concatenate([col[:1], col[1::2] + col[2::2]])

    def nodes(self, count: int) -> np.ndarray:
        count = max(int(count), 1)
        return self.lo + self.period * np.arange(count) / count


class Constant:
    """The trivial basis of a 1-D (ODE) problem's missing space axis."""
    kind = "constant"
    size = 1

    def matrix(self, points, order: int = 0) -> np.ndarray:
        return np.full((len(np.atleast_1d(points)), 1), 1.0 if order == 0 else 0.0)

    def nodes(self, count: int) -> np.ndarray:
        return np.zeros(1)


# ============================================================ adapter
class BasisAdapter:
    def __init__(self, pretrained_net=None, **config):
        unknown = sorted(set(config) - set(default_basis_config()))
        if unknown:
            raise DeepXDEConfigError(f"basis_config: unknown keys {unknown}")
        cfg = dict(default_basis_config(), **config)
        self.config = cfg
        self.train_frac = float(cfg["train_frac"])
        if not 0.0 < self.train_frac < 1.0:
            raise ValueError(f"basis_config train_frac must lie strictly between 0 and 1, "
                             f"got {self.train_frac!r}")
        #: the host reads it; the basis backend has no validation block
        self.val_frac = 0.0
        if cfg["pde_loss"] not in PDE_LOSSES:
            raise DeepXDEConfigError(f"basis_config pde_loss must be one of {PDE_LOSSES}, "
                                     f"got {cfg['pde_loss']!r}")
        if cfg["start"] not in STARTS:
            raise DeepXDEConfigError(f"basis_config start must be one of {STARTS}, "
                                     f"got {cfg['start']!r}")
        if cfg["refine"] not in ("spectral", "prediction", "off"):
            raise DeepXDEConfigError("basis_config refine must be 'spectral', 'prediction' "
                                     "or 'off'")
        if not float(cfg["refine_min_gain"]) > 1.0:
            raise DeepXDEConfigError("basis_config refine_min_gain must be > 1 (a refinement "
                                     "has to lower the loss by that factor to be kept going)")
        if cfg["precondition"] not in ("jacobi", "none"):
            raise DeepXDEConfigError("basis_config precondition must be 'jacobi' or 'none'")
        if cfg["periodic"] not in ("auto", "off"):
            raise DeepXDEConfigError("basis_config periodic must be 'auto' or 'off'")
        if cfg["periodic_modes"] not in ("nyquist", "profile"):
            raise DeepXDEConfigError("basis_config periodic_modes must be 'nyquist' or 'profile'")
        for key in ("lbfgs_maxiter", "history_size", "collocation_factor"):
            if int(cfg[key]) < 1:
                raise DeepXDEConfigError(f"basis_config {key} must be >= 1")
        self.domain_key = None
        self.last_lbfgs_exit = None
        self.last_solve_stats = None
        #: live during a solve only: the optimiser, and a function returning
        #: the current full-grid prediction of variable 0 (trace hooks read it)
        self.live_optimizer = None
        self.live_predict = None

    # ---------------------------------------------------------------- setup
    @staticmethod
    def _raw_fields(key, var_names):
        """The observed fields on the FULL grid, as handed to createTrajectory."""
        samples = global_var.samples_manager
        if hasattr(samples, "raw_fields"):
            fields = samples.raw_fields(key)
            return {v: np.asarray(fields[v]) for v in var_names}
        traj = samples[key]
        by_name = {e.var_name: e for e in traj._entries}
        return {v: np.asarray(by_name[v].data_tensor) for v in var_names}

    def _bases(self, prof, grids, coords):
        cfg = self.config
        t_lo, t_hi = float(coords[0].min()), float(coords[0].max())
        size_t = int(math.ceil(float(cfg["time_factor"]) * max(prof.axes[0].modes, 1)))
        bases = [Chebyshev(t_lo, t_hi, size_t)]
        if len(grids) == 1:
            bases.append(Constant())
            return bases
        ax = prof.axes[1]
        if ax.periodic and cfg["periodic"] == "auto":
            n_unique = ax.n if ax.endpoint == "excluded" else ax.n - 1
            K = (int(math.ceil(float(cfg["nyquist_factor"]) * (n_unique // 2)))
                 if cfg["periodic_modes"] == "nyquist"
                 else int(math.ceil(float(cfg["space_factor"]) * max(ax.modes, 1))))
            bases.append(Fourier(ax.lo, ax.period, K))
        else:
            size_x = int(math.ceil(float(cfg["space_factor"]) * max(ax.modes, 1)))
            bases.append(Chebyshev(ax.lo, ax.hi, size_x))
        return bases

    def _data_term_mass(self, eq, key):
        """|target| + sum_k |c_k phi_k| + |b| from EPDE's own term values on the
        inner domain: the denominator of ``Discrepancy._compute_scale_invariant``
        with its in-place term set (``active_only=False``)."""
        from epde.operators.common.objectives import cancellation_parts
        parts = cancellation_parts(eq, active_only=False)
        if parts is None:
            targets, _ = eq.evaluate(active_only=False)
            return np.abs(np.asarray(targets[key], dtype=np.float64).reshape(-1))
        return np.asarray(parts[1][key], dtype=np.float64).reshape(-1)

    # ---------------------------------------------------------------- solve
    def solve(self, equation_or_system, grids: list, data, domain_key: int = None):
        """Same contract as ``DeepXDEAdapter.solve``: the full-grid prediction
        per variable (flattened, C order) and the final loss."""
        threads = self.config["threads"]
        before = torch.get_num_threads()
        if threads:
            torch.set_num_threads(int(threads))
        try:
            return self._solve(equation_or_system, grids, data, domain_key)
        finally:
            if threads:
                torch.set_num_threads(before)
            self.live_optimizer = None
            self.live_predict = None

    def _solve(self, equation_or_system, grids, data, domain_key):
        t_start = time.perf_counter()
        cfg = self.config
        samples = global_var.samples_manager
        self.domain_key = key = (domain_key if domain_key is not None
                                 else samples.trajecatoryIDs[0])
        self.last_lbfgs_exit = None
        self.last_solve_stats = None
        grids = [np.asarray(g, dtype=np.float64) for g in grids]
        ndim = len(grids)
        if ndim not in (1, 2):
            raise NotImplementedError(f"basis backend covers 1-D and 2-D domains; got {ndim}")
        eq_list, var_names, specs = system_specs(equation_or_system, ndim)
        data_list = [data] if isinstance(data, np.ndarray) else list(data)
        if len(data_list) != len(var_names):
            raise ValueError(f"{len(data_list)} observed fields for {len(var_names)} variables")

        # ---- where the observation term looks, and the train window
        samples.require_unstrided('the basis backend')
        mask = np.asarray(samples.gFunc("m")[key]).reshape(grids[0].shape).astype(bool)
        t_inner = grids[0][mask]
        split = time_split(t_inner, self.train_frac, 0.0)
        coords = [axis_coordinates(grids[a], a) for a in range(ndim)]
        raw = self._raw_fields(key, var_names)
        prof = build_profile(grids, raw, split.t_train, energy=float(cfg["energy"]))
        inner_t = np.unique(t_inner)
        train_t = inner_t[inner_t <= split.t_train]
        if ndim == 2:
            box = mask.any(axis=1)[:, None] & mask.any(axis=0)[None, :]
            if not np.array_equal(box, mask):
                raise NotImplementedError("basis backend needs a box-shaped inner domain")
            inner_x = coords[1][mask.any(axis=0)]
        else:
            inner_x = np.zeros(1)
        n_inner_t = len(inner_t)
        obs_np = [np.asarray(o, dtype=np.float64).reshape(n_inner_t, len(inner_x))[:len(train_t)]
                  for o in data_list]

        # ---- weights: the DeepXDE adapter's data yardsticks
        pde_loss = cfg["pde_loss"]
        w_obs = [variance_weight(np.asarray(o, dtype=np.float64).reshape(-1)[split.train])
                 for o in data_list]
        if pde_loss == "mse":
            w_pde = [variance_weight(np.asarray(eq.evaluate(active_only=True)[0][key])
                                     .reshape(-1)[split.train]) for eq in eq_list]
        else:
            w_pde = [1.0] * len(eq_list)
        w_pde = [w * float(cfg["phys_weight_scale"]) for w in w_pde]

        # ---- scale-invariant denominators from the data (train window)
        floors, mass_windows, sinv_diag = [], [], []
        if pde_loss != "mse":
            for eq in eq_list:
                mass = self._data_term_mass(eq, key).reshape(n_inner_t, len(inner_x))
                window = mass[:len(train_t)]
                floors.append(float(cfg["sinv_floor_rel"]) * float(window.mean()))
                mass_windows.append(window + floors[-1])
                sinv_diag.append(mass_diagnostics(mass_windows[-1]))

        # ---- typical magnitude of every field, for the column scaling
        # From finite differences of the OBSERVED field on the train window. The
        # column norms are taken at these values rather than at the start point:
        # at C = 0 a nonlinear term (u^3, u*u_x) has a zero Jacobian, and a
        # coefficient that acts only through one (Burgers' Nyquist sine,
        # invisible on the data grid) got a scale of ~1e13, i.e. a direction of
        # enormous curvature that stalled the line search at t = 0.
        need = required_fields(specs)
        level_mask = coords[0] <= split.t_train
        typical = {}
        for var, orders in need:
            f = np.asarray(raw[var_names[var]], dtype=np.float64)[level_mask]
            for a, n in enumerate(orders):
                c = coords[a][level_mask] if a == 0 else coords[a]
                for _ in range(n):
                    f = np.gradient(f, c, axis=a)
            typical[(var, orders)] = float(np.sqrt(np.mean(f ** 2)))

        problem = dict(ndim=ndim, specs=specs, need=need, coords=coords, train_t=train_t,
                       inner_x=inner_x, obs_np=obs_np, w_obs=w_obs, w_pde=w_pde,
                       pde_loss=pde_loss, floors=floors, mass_windows=mass_windows,
                       sinv_diag=sinv_diag,
                       typical=typical)

        # ---- solve, then refine any axis the solution shows is under-resolved
        bases = self._bases(prof, grids, coords)
        C_init, levels = None, []
        refinable = [a for a, b in enumerate(bases) if hasattr(b, "refined")]
        tol = float(cfg["refine_tol"])
        min_gain = float(cfg["refine_min_gain"])
        stop_reason, previous_loss, previous_solution = "resolved", None, None
        for level in range(int(cfg["max_refinements"]) + 1):
            out = self._solve_level(problem, bases, C_init)
            # 'prediction': refinement has converged once the solution moves,
            # between consecutive levels and over the whole domain, by no more
            # than it misses its own training observations. Below that, a
            # resolution difference cannot be told apart from the data misfit
            # every candidate carries anyway. It reads the solver's own
            # prediction and the TRAIN window only, never the scored tail.
            change = misfit = None
            if previous_solution is not None:
                change = max(float(np.sqrt(np.mean((a - b) ** 2)))
                             for a, b in zip(out["solutions"], previous_solution))
                misfit = max(out["stats"]["obs_rmse"])
            out["stats"]["prediction_change"] = change
            previous_solution = out["solutions"]
            tails = {a: self._tail_fraction(out["C"], bases[a], a) for a in refinable}
            out["stats"]["tail_fraction"] = {str(a): tails[a] for a in refinable}
            levels.append(out["stats"])
            under = [a for a in refinable if tails[a] > tol]
            # A refinement must PAY: the solve's own loss has to fall by
            # ``refine_min_gain``. When it does not, the candidate is limited by
            # its equation, not by the basis -- Burgers without viscosity forms a
            # shock whose spectrum never converges, and three doublings moved its
            # loss 2.359e-2 -> 2.270e-2 (-4%) for 60x the cost (47 s -> 3418 s).
            # Resolution-limited solves fell 30-100x per doubling (AC), so the
            # rule separates the two by orders of magnitude.
            gained = (previous_loss is None
                      or out["loss"] <= previous_loss / min_gain)
            if cfg["refine"] == "off":
                stop_reason = "off"
            elif not np.isfinite(out["loss"]):
                stop_reason = "nonfinite"
            elif not under:
                stop_reason = "resolved"
            elif cfg["refine"] == "prediction":
                stop_reason = ("converged" if change is not None and change <= misfit
                               else None)
                if stop_reason is None and level == int(cfg["max_refinements"]):
                    stop_reason = "max_refinements"
            elif not gained:
                stop_reason = "no_gain"
            elif level == int(cfg["max_refinements"]):
                stop_reason = "max_refinements"
            else:
                stop_reason = None
            if stop_reason is not None:
                break
            previous_loss = out["loss"]
            new = list(bases)
            for a in under:
                new[a] = bases[a].refined()
            C_init = [self._embed(C, new) for C in out["C"]]
            bases = new
        final = levels[-1]
        self.last_lbfgs_exit = {"reason": final["exit"], "iterations": final["iterations"]}
        stats = dict(final)
        stats.update({"levels": levels, "n_levels": len(levels),
                      "resolved": bool(all(v <= tol for v in final["tail_fraction"].values())),
                      "refine_stop": stop_reason,
                      "iterations_total": int(sum(l["iterations"] for l in levels)),
                      "seconds": time.perf_counter() - t_start,
                      "periodic_axes": list(prof.periodic_axes),
                      "profile_modes": [a.modes for a in prof.axes],
                      "threads": torch.get_num_threads()})
        self.last_solve_stats = stats
        if not np.isfinite(out["loss"]):
            return [np.full(grids[0].size, np.nan) for _ in var_names], np.nan
        return out["solutions"], out["loss"]

    @staticmethod
    def _tail_fraction(Cs, basis, axis) -> float:
        """Share of the solution's modal energy on ``axis`` held by the top third
        of the modes (a Fourier axis's mean mode left out). The largest over
        the system's variables."""
        worst = 0.0
        for C in Cs:
            e = basis.modal_energy(np.asarray(C), axis)
            if basis.kind == "fourier":
                e = e[1:]
            total = float(e.sum())
            if total <= 0.0 or len(e) < 3:
                continue
            cut = len(e) - len(e) // 3
            worst = max(worst, float(e[cut:].sum()) / total)
        return worst

    @staticmethod
    def _embed(C, new) -> np.ndarray:
        """Coefficients of a coarse solution in the refined bases, zero-padded.
        Both families are prefix-compatible when doubled: Chebyshev degrees and
        the Fourier layout [1, cos_1, sin_1, ...] keep their indices."""
        C = np.asarray(C)
        out = np.zeros((new[0].size, new[1].size))
        out[:C.shape[0], :C.shape[1]] = C
        return out

    def _solve_level(self, P, bases, C_init):
        cfg = self.config
        ndim, specs, need = P["ndim"], P["specs"], P["need"]
        coords, train_t, inner_x = P["coords"], P["train_t"], P["inner_x"]
        pde_loss, floors, w_pde, w_obs = P["pde_loss"], P["floors"], P["w_pde"], P["w_obs"]
        St, Sx = bases[0].size, bases[1].size
        t_level = time.perf_counter()

        def as_t(a):
            return torch.as_tensor(np.ascontiguousarray(a), dtype=DTYPE, device=DEVICE)

        def pad(o):
            return (o[0], o[1] if ndim == 2 else 0)

        # ---- collocation nodes over the FULL domain (tensor grid)
        cf = int(cfg["collocation_factor"])
        nodes_t = bases[0].nodes(cf * St)
        nodes_x = bases[1].nodes(cf * Sx) if ndim == 2 else np.zeros(1)
        orders_needed = sorted({o for _, o in need} | {(0,) * ndim})
        Pt = {o[0]: as_t(bases[0].matrix(nodes_t, o[0])) for o in orders_needed}
        Px = {pad(o)[1]: as_t(bases[1].matrix(nodes_x, pad(o)[1])) for o in orders_needed}
        TT, XX = np.meshgrid(nodes_t, nodes_x, indexing="ij")
        col_coords = [as_t(TT.reshape(-1))] + ([as_t(XX.reshape(-1))] if ndim == 2 else [])
        Ot = as_t(bases[0].matrix(train_t, 0))
        Ox = as_t(bases[1].matrix(inner_x, 0))
        obs_grids = [as_t(o) for o in P["obs_np"]]
        full_t = as_t(bases[0].matrix(coords[0], 0))
        full_x = as_t(bases[1].matrix(coords[1], 0)) if ndim == 2 else as_t(np.ones((1, 1)))

        data_weights = []
        if pde_loss == "sinv_data":
            def nearest(grid, points):
                i = np.clip(np.searchsorted(grid, points), 0, len(grid) - 1)
                lo = np.clip(i - 1, 0, len(grid) - 1)
                return np.where(np.abs(grid[lo] - points) < np.abs(grid[i] - points), lo, i)
            it = nearest(train_t, nodes_t)          # past the last train level: copy it
            for e, window in enumerate(P["mass_windows"]):
                m = (window[np.ix_(it, nearest(inner_x, nodes_x))].reshape(-1) if ndim == 2
                     else window[it, 0])
                data_weights.append(as_t(inverse_mass(m)))

        def fields_of(Cs):
            out = {}
            for var, orders in need:
                ot, ox = pad(orders)
                out[(var, orders)] = (Pt[ot] @ Cs[var] @ Px[ox].T).reshape(-1)
            return out

        def loss_of(Cs):
            fields = fields_of(Cs)
            total = 0.0
            for e, spec in enumerate(specs):
                if pde_loss == "mse":
                    r = residual(spec, fields, col_coords, torch)
                    term = torch.mean(r ** 2)
                else:
                    r, mass = residual(spec, fields, col_coords, torch, with_mass=True)
                    if pde_loss == "sinv_live":
                        term = torch.mean(cancellation_ratio(r, mass + floors[e], torch) ** 2)
                    else:
                        term = torch.mean((r * data_weights[e]) ** 2)
                total = total + w_pde[e] * term
            for v, U in enumerate(obs_grids):
                pred = Ot @ Cs[v] @ Ox.T
                total = total + w_obs[v] * torch.mean((pred - U) ** 2)
            return total

        # ---- start
        if C_init is not None:
            C0s = [as_t(C) for C in C_init]
        elif cfg["start"] == "data_lstsq":
            # min-norm least squares on the tensor grid: pinv(A (x) B) = pinv(A) (x) pinv(B)
            C0s = [as_t(np.linalg.pinv(Ot.numpy()) @ U.numpy() @ np.linalg.pinv(Ox.numpy()).T)
                   for U in obs_grids]
        else:
            C0s = [torch.zeros((St, Sx), dtype=DTYPE, device=DEVICE) for _ in obs_grids]

        # ---- preconditioning: C = S * Chat, S = 1 / (Jacobian column norm).
        # The loss is a sum of squares. Its Jacobian column for coefficient (i, j)
        # is exact and separable: sum over the fields o the residual reads of
        # dR/d(field_o) * Pt_o[:, i] Px_o[:, j], plus the observation column.
        # Column scaling (scipy's x_scale='jac') removes the ~N^4 spread that
        # high-order Chebyshev derivatives put between coefficients. It is a
        # change of variables, fixed for the whole level, not a step size.
        n_col = len(nodes_t) * len(nodes_x)
        n_obs = len(train_t) * len(inner_x)

        def column_scales():
            leaves = {k: torch.full((n_col,), P["typical"][k], dtype=DTYPE,
                                    device=DEVICE).requires_grad_(True) for k in need}
            sq = [torch.zeros((St, Sx), dtype=DTYPE, device=DEVICE) for _ in C0s]
            keys = list(need)
            for e, spec in enumerate(specs):
                if pde_loss == "mse":
                    r, row = residual(spec, leaves, col_coords, torch), 1.0
                else:
                    r, mass = residual(spec, leaves, col_coords, torch, with_mass=True)
                    row = (cancellation_ratio(torch.ones_like(mass), mass.detach() + floors[e],
                                              torch)
                           if pde_loss == "sinv_live" else data_weights[e])
                grads = torch.autograd.grad(r.sum(), [leaves[k] for k in keys],
                                            allow_unused=True)
                G = {k: (g * row).reshape(len(nodes_t), len(nodes_x))
                     for k, g in zip(keys, grads) if g is not None}
                for k1, g1 in G.items():
                    for k2, g2 in G.items():
                        if k1[0] != k2[0]:
                            continue
                        (a1, b1), (a2, b2) = pad(k1[1]), pad(k2[1])
                        sq[k1[0]] += (w_pde[e] / n_col) * (
                            (Pt[a1] * Pt[a2]).T @ (g1 * g2) @ (Px[b1] * Px[b2]))
            for v in range(len(C0s)):
                sq[v] += (w_obs[v] / n_obs) * ((Ot ** 2).sum(0)[:, None] * (Ox ** 2).sum(0)[None, :])
            return [torch.where(q > 0.0, 1.0 / torch.sqrt(q), torch.ones_like(q)).detach()
                    for q in sq]

        if cfg["precondition"] == "jacobi":
            S = column_scales()
        else:
            S = [torch.ones((St, Sx), dtype=DTYPE, device=DEVICE) for _ in C0s]
        Chat = [(C0 / s).detach().clone().requires_grad_(True) for C0, s in zip(C0s, S)]

        def coeffs():
            return [c * s for c, s in zip(Chat, S)]

        loss_start = float(loss_of(coeffs()).detach())

        # ---- L-BFGS with DeepXDE's settings and chunked loop
        maxiter = int(cfg["lbfgs_maxiter"])
        per_step = min(maxiter, 1000)
        opt = torch.optim.LBFGS(Chat, lr=1, max_iter=per_step, max_eval=int(1.25 * per_step),
                                tolerance_grad=float(cfg["gtol"]),
                                tolerance_change=float(cfg["ftol"]),
                                history_size=int(cfg["history_size"]),
                                line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss = loss_of(coeffs())
            loss.backward()
            return loss

        def predict_full(var=0):
            with torch.no_grad():
                return (full_t @ coeffs()[var] @ full_x.T).reshape(-1).numpy().copy()

        self.live_optimizer = opt
        self.live_predict = predict_full
        t_opt = time.perf_counter()
        reason, prev = "maxiter", 0
        while prev < maxiter:
            opt.param_groups[0]["max_iter"] = min(per_step, maxiter - prev)
            state = opt.state.get(opt._params[0], {})
            evals_before = state.get("func_evals", 0)
            opt.step(closure)
            state = opt.state.get(opt._params[0])
            if state is None or "n_iter" not in state:
                reason = "no_step"
                break
            n_iter = state["n_iter"]
            if n_iter == prev:
                reason = "gtol"
                break
            if prev == n_iter - 1:                   # DeepXDE's own convergence test
                prev = n_iter
                reason = self._stop_reason(lambda: loss_of(coeffs()), Chat, opt, maxiter,
                                           n_iter, state["func_evals"] - evals_before)
                break
            prev = n_iter
            with torch.no_grad():
                if not bool(torch.isfinite(loss_of(coeffs()))):
                    reason = "nonfinite"
                    break
        seconds_opt = time.perf_counter() - t_opt
        final_loss = float(loss_of(coeffs()).detach())
        C_final = [c.detach().numpy().copy() for c in coeffs()]
        with torch.no_grad():
            obs_rmse = [float(torch.sqrt(torch.mean((Ot @ C @ Ox.T - U) ** 2)))
                        for C, U in zip(coeffs(), obs_grids)]
        stats = {"exit": reason, "iterations": int(prev), "loss_start": loss_start,
                 "loss": final_loss, "seconds_level": time.perf_counter() - t_level,
                 "seconds_lbfgs": seconds_opt,
                 "bases": [(b.kind, b.size) for b in bases],
                 "n_params": int(St * Sx * len(C0s)),
                 "collocation": [len(nodes_t), len(nodes_x)], "n_obs": int(n_obs),
                 "weights_pde": list(w_pde), "weights_obs": list(w_obs),
                 "pde_loss": pde_loss, "start": cfg["start"] if C_init is None else "refined",
                 "precondition": cfg["precondition"],
                 "scale_spread": [float(x.max() / x.min()) for x in S],
                 "obs_rmse": obs_rmse}
        if pde_loss != "mse":
            stats["floors"] = list(floors)
            stats["data_mass"] = list(P["sinv_diag"])
        return {"C": C_final, "loss": final_loss, "stats": stats,
                "solutions": [predict_full(v) for v in range(len(C0s))]}

    @staticmethod
    def _stop_reason(loss_fn, params, opt, maxiter, n_iter, evals_in_step):
        """DeepXDE's one-iteration step, resolved as ``_install_lbfgs_stall_guard``
        resolves it: nonfinite, maxiter, gtol, maxeval, converged."""
        loss = loss_fn()
        if not bool(torch.isfinite(loss)):
            return "nonfinite"
        grads = torch.autograd.grad(loss, params)
        flat = torch.cat([g.reshape(-1) for g in grads])
        if not bool(torch.isfinite(flat).all()):
            return "nonfinite"
        if n_iter >= maxiter:
            return "maxiter"
        group = opt.param_groups[0]
        if bool(flat.abs().max() <= group["tolerance_grad"]):
            return "gtol"
        if evals_in_step >= group["max_eval"]:
            return "maxeval"
        return "converged"
