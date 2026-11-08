#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Allen-Cahn PINN with the weight-free loss -- the wave recipe moved
across unchanged, plus one library switch.

    u_t = D u_xx + a u + b u^3,     truth D = 1e-4, a = 5, b = -5
    x in [-1, 1) periodic, t in [0, 1]; AC.mat, 512 x 201

DIFFUSION selects the library: True -> [u_xx, u, u^3] (the true form),
False -> [u, u^3]. Everything else is the family's shipped default::

    loss = l_phys + l_ic + l_bc + l_stat + l_anch + l_data   (no weights)
           LOSS_FORM='log': the sum of their logs instead (combine_loss)

    l_phys  mean((u_t - A @ c_hat)^2) / Var(u_t_FD),  c_hat = global_ols
            on the net's own field (COEF_SOURCE='ols')
    l_data  mean((u - U)^2) / Var(U) on the train block
    l_ic    IC row / Var(U)                     } IC_MODE='scaled' only;
    l_bc    periodic u and u_x at x = -1 vs 1,  } 'none' (default) drops
            each over its own channel variance  } both, l_data covers them
    l_stat  het / per-axis chi, raw unless STAT_BOUND   (STAT='none')
    l_anch  sum_j (c_hat_j - c_FD_j)^2 mean(A_j^2)/Var(y) (ANCHOR='none')
            -- signal units, A and y from the FD design

TRAIN_FRAC=0.8 splits everything the loss touches -- collocation, BC
samples, l_data, the FD design and the yardsticks all stop at t_split.

The grid is periodic (AC.mat's x stops one step short of 1), so the FD
design takes x-derivatives spectrally (FD_X='spectral'; 'stencil' = periodic
centred differences) and the test integration uses a spectral Laplacian.
t-derivatives are centred stencils of FD_ORDER. Measured offline, THETA_FD's
rel-L1 error: 6.1e-3 with the original 2nd-order path, 7.0e-4 with 4th-order
stencils on both axes, 1.3e-7 with 4th-order t + spectral x -- the x
derivative at the sharp fronts is where the error was.

Offline reference, no network, spectral u_xx on the full grid: the true
form leaves residual RMSE 1.3e-7 against 2.6e-2 without diffusion, and
integrating the true equation from the IC reproduces the data to 4.6e-11 --
so the data is exactly this equation, and any gap below is the PINN's.

Run cells by editing the mode block; artifacts are suffixed per mode.
"""

import sys
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy.integrate import solve_ivp
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pinn_common import (                    # shared PINN model + driver
    Checkpoint,
    MLP,
    assert_train_only,
    as_float as _f,
    grad_share,
    rel_l1,
    train,
)
from stat_common import (                    # shared, from dp/cv_metric.py
    bounded,
    chi2_per_term,
    global_ols,
    het_per_window,
    max_corr,
    observation_loss,
    tls_fit,
    tls_scales,
    anchor_penalty,
    anchor_scales,
    central_diff,
    combine_loss,
    fd_core,
    spectral_diff,
)

torch.set_default_dtype(torch.float32)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================ settings
DATA_PATH = "./AC.mat"
# diagnostics only -- never in the loss unless COEF_SOURCE='true'
TRUE_COEFS = {"u_xx": 1e-4, "u": 5.0, "u^3": -5.0}
X_MIN, X_MAX = -1.0, 1.0          # periodic

HIDDEN     = [64] * 5
N_COLL     = 8000
N_BC       = 200
ADAM_ITERS = 20000
ADAM_LR    = 1e-3
LBFGS_MAX  = 20000
SEED       = 0

N_WIN_PER_DIM = 30
WIN_FRAC      = 0.5
EPS = 1e-12

# ============================================================ mode block
DIFFUSION     = True        # library [u_xx, u, u^3] (True) | [u, u^3]
PHYS_AGG      = "mean"      # structural here, as on wave
PHYS_NORM     = "var"       # 'var' | 'max'
COEF_SOURCE = "ols"   # 'ols' | 'fd' | 'true' | 'tls'
# 'tls' = TARGET-FREE physics term: l_phys is the smallest eigenvalue of the
# column-scaled [target | library] Gram (tls_fit), i.e. total least squares
# on the net's own field instead of OLS against the target. Columns are
# scaled by their RMS on the FD design (TLS_SCALE), computed once.
IC_MODE = "none"   # 'scaled' | 'none': 'none' drops l_ic AND l_bc
STAT          = "none"      # 'none' | 'het' | 'chi'
STAT_BOUND    = False       # False = raw chi / tau2/theta_bar^2; see dp
STAT_FLOAT64  = True
STAT_FLOOR    = False
ANCHOR        = "none"      # 'none' | 'fd'
ANCHOR_NORM   = "signal"    # 'signal' | 'relative' (collapsed on D=1e-4)
FD_STRIDE     = 1
FD_ORDER      = 4           # centred-FD accuracy in t (and x if stencil); 2 = old path
FD_X          = "spectral"  # 'spectral' (periodic grid) | 'stencil'
ANCHOR_WEIGHT = "none"      # 'none' | 'stat'
BEST_KEY      = "loss"      # truth-free
LOSS_FORM     = "sum"       # 'sum' | 'log' (sum of log-terms); see combine_loss
PHYS_TERM     = True        # False = data-only baseline: l_phys is still computed
                            # and logged, but left out of the loss
DATA_TERM = True   # pointwise supervision on the observed u
TRAIN_FRAC    = 0.8         # the loss sees ONLY t <= t_split (all x)

_SUFFIX = ("" if DIFFUSION else "_nodiff") \
    + ({"ols": "", "fd": f"_cfd{FD_STRIDE}", "true": "_true", "tls": "_tls"}[COEF_SOURCE]) \
    + ("" if STAT == "none" else f"_{STAT}") \
    + ("_raw" if (STAT != "none" and not STAT_BOUND) else "") \
    + ("" if ANCHOR == "none" else f"_afd{FD_STRIDE}") \
    + ("_asig" if (ANCHOR != "none" and ANCHOR_NORM == "signal") else "") \
    + ("_wstat" if ANCHOR_WEIGHT != "none" else "") \
    + ("" if PHYS_NORM == "var" else "_pmax") \
    + ("_data" if DATA_TERM else "") \
    + ("_noicbc" if IC_MODE == "none" else "") \
    + ("" if FD_ORDER == 2 else f"_fdo{FD_ORDER}") \
    + ("_xspec" if FD_X == "spectral" else "") \
    + ("_llog" if LOSS_FORM == "log" else "") \
    + ("" if PHYS_TERM else "_nophys") \
    + (f"_s{SEED}" if SEED != 0 else "")   # seed 0 keeps the original names

if FD_X not in ("spectral", "stencil"):
    raise ValueError(f"bad FD_X {FD_X!r}")
if IC_MODE not in ("scaled", "none"):
    raise ValueError(f"bad IC_MODE {IC_MODE!r}")
if not PHYS_TERM and not DATA_TERM:
    raise ValueError("PHYS_TERM=False and DATA_TERM=False leave nothing to train")
if IC_MODE == "none" and not DATA_TERM:
    raise ValueError("IC_MODE='none' removes the only anchor on the solution "
                     "family unless DATA_TERM supervises the field")
if ANCHOR_WEIGHT != "none" and (ANCHOR == "none" or STAT == "none"):
    raise ValueError("ANCHOR_WEIGHT='stat' requires ANCHOR='fd' and a STAT")
if not 0.0 < TRAIN_FRAC <= 1.0:
    raise ValueError(f"bad TRAIN_FRAC {TRAIN_FRAC!r}")

FEATURE_NAMES = ("u_xx", "u", "u^3") if DIFFUSION else ("u", "u^3")
THETA_TRUE_NP = np.array([TRUE_COEFS[n] for n in FEATURE_NAMES])


def build_features(u, u_xx):
    """The library, the only thing DIFFUSION changes."""
    cols = ([u_xx] if DIFFUSION else []) + [u, u ** 3]
    return torch.stack(cols, dim=1)


# ============================================================ data
_mat = sio.loadmat(DATA_PATH)
U = _mat["uu"].astype(np.float32)                 # (Nx, Nt) = (512, 201)
x_grid = _mat["x"].ravel().astype(np.float32)
t_grid = _mat["tt"].ravel().astype(np.float32)
Nx, Nt = U.shape
T_MIN = float(t_grid[0])
DX = float(x_grid[1] - x_grid[0])

n_train = int(np.ceil(TRAIN_FRAC * Nt))
t_split = float(t_grid[n_train - 1])
XX, TT = np.meshgrid(x_grid, t_grid, indexing="ij")
X_obs = torch.tensor(np.stack([XX.ravel(), TT.ravel()], axis=1), device=device)

X_IC = torch.tensor(np.stack([x_grid, np.full_like(x_grid, T_MIN)], axis=1),
                    device=device)
U_IC = torch.tensor(U[:, 0:1], device=device)


def make_windows(lo, hi, n, frac):
    w = (hi - lo) * frac
    h = w / 2
    c = np.linspace(lo + h, hi - h, n, dtype=np.float32)
    return c - h, c + h


t_lo, t_hi = make_windows(T_MIN, t_split, N_WIN_PER_DIM, WIN_FRAC)
x_lo, x_hi = make_windows(X_MIN, X_MAX, N_WIN_PER_DIM, WIN_FRAC)
T_LO = torch.tensor(t_lo, device=device).unsqueeze(1)
T_HI = torch.tensor(t_hi, device=device).unsqueeze(1)
X_LO = torch.tensor(x_lo, device=device).unsqueeze(1)
X_HI = torch.tensor(x_hi, device=device).unsqueeze(1)


# ============================================================ yardsticks
# Population variance (ddof=0) at every call site, as in the family.
def _fd_design(stride=1):
    """u_t and the library from the OBSERVED field.

    x (periodic grid): FD_X='spectral' Fourier derivatives, or 'stencil'
    periodic centred differences of FD_ORDER (order 2 = the original
    roll formulas). t: FD_ORDER=2 keeps the original np.gradient path with
    one row dropped per end; higher orders use centred stencils, which
    drop order//2 rows per end."""
    Us = U[::stride, :n_train:stride].astype(np.float64)
    dx = DX * stride
    ts = t_grid[:n_train:stride].astype(np.float64)
    if FD_X == "spectral":
        u_x = spectral_diff(Us, dx, 1, axis=0)
        u_xx = spectral_diff(Us, dx, 2, axis=0)
    elif FD_ORDER == 2:
        u_x = (np.roll(Us, -1, axis=0) - np.roll(Us, 1, axis=0)) / (2.0 * dx)
        u_xx = (np.roll(Us, -1, axis=0) - 2.0 * Us + np.roll(Us, 1, axis=0)) / dx ** 2
    else:
        u_x = central_diff(Us, dx, 1, FD_ORDER, axis=0, periodic=True)
        u_xx = central_diff(Us, dx, 2, FD_ORDER, axis=0, periodic=True)
    if FD_ORDER == 2:
        tcore = slice(1, -1)
        u_t = np.gradient(Us, ts, axis=1)[:, tcore]
    else:
        tcore = fd_core(FD_ORDER)
        u_t = central_diff(Us, float(ts[1] - ts[0]), 1, FD_ORDER, axis=1)
    y = torch.tensor(u_t.ravel())
    A = build_features(torch.tensor(Us[:, tcore].ravel()),
                       torch.tensor(u_xx[:, tcore].ravel()))
    return y, A, torch.tensor(u_x[:, tcore].ravel())


_y_fd, _A_fd, _ux_fd = _fd_design()
PHYS_SCALE = float(_y_fd.abs().max()) + EPS
PHYS_VAR = float(_y_fd.var(unbiased=False)) + EPS
PHYS_DIV = (PHYS_SCALE if PHYS_NORM == "max" else float(np.sqrt(PHYS_VAR)))
IC_NORM_U = float(U[:, :n_train].var()) + EPS
IC_NORM_UX = float(_ux_fd.var(unbiased=False)) + EPS
# TLS column yardsticks: RMS of [u_t | library] on the SAME FD design.
# Always built (cheap) so BOTH readouts are reported in every arm.
TLS_SCALE = tls_scales(_y_fd, _A_fd).to(device=device, dtype=torch.float32)

X_train = torch.tensor(
    np.stack([XX[:, :n_train].ravel(), TT[:, :n_train].ravel()], axis=1),
    device=device)
U_train = torch.tensor(U[:, :n_train].reshape(-1, 1), device=device)
assert_train_only("l_data grid", X_train[:, 1], t_split)
assert_train_only("FD design", torch.tensor(t_grid[:n_train]), t_split)
assert_train_only("IC row", X_IC[:, 1], t_split)


def _fd_anchor(stride):
    y, A, _ = _fd_design(stride)
    c, _ = global_ols(y, A, ridge=EPS)
    return c.to(device=device, dtype=torch.float32)


THETA_FD = (_fd_anchor(FD_STRIDE)
            if (ANCHOR == "fd" or COEF_SOURCE == "fd") else None)
# The anchor's yardsticks, from the SAME design that built THETA_FD.
ANCHOR_COL_MS, ANCHOR_Y_VAR = (
    (lambda s: (s[0].to(device=device, dtype=torch.float32), s[1]))(
        anchor_scales(*_fd_design(FD_STRIDE)[:2]))
    if ANCHOR == "fd" else (None, None))
THETA_TRUE = torch.tensor(THETA_TRUE_NP, dtype=torch.float32, device=device)


# ============================================================ network
def field_derivs(net, xt):
    xt = xt.clone().requires_grad_(True)
    u = net(xt)
    g = torch.autograd.grad(u, xt, torch.ones_like(u), create_graph=True)[0]
    u_x, u_t = g[:, 0:1], g[:, 1]
    u_xx = torch.autograd.grad(u_x, xt, torch.ones_like(u_x),
                               create_graph=True)[0][:, 0]
    return u[:, 0], u_x[:, 0], u_t, u_xx


# ============================================================ loss
def _chi_per_axis(y, A, x_c, t_c):
    """chi summed over ONE score path PER GRID AXIS, as on wave."""
    yy, AA = (y.double(), A.double()) if STAT_FLOAT64 else (y, A)
    total = None
    theta = None
    for key in (t_c, x_c):
        order = torch.argsort(key)
        out = chi2_per_term(yy[order], AA[order], ridge=EPS,
                            use_floor=STAT_FLOOR)
        s = bounded(out["score"]) if STAT_BOUND else out["score"]
        total = s if total is None else total + s
        theta = out["theta"]
    return total, theta


def _masks(x_c, t_c, dtype):
    in_t = (t_c.unsqueeze(0) >= T_LO) & (t_c.unsqueeze(0) < T_HI)
    in_x = (x_c.unsqueeze(0) >= X_LO) & (x_c.unsqueeze(0) < X_HI)
    return torch.cat([in_t, in_x], dim=0).to(dtype)     # (2*N_WIN, N)


def physics_and_stat_loss(net, X_coll):
    x_c, t_c = X_coll[:, 0], X_coll[:, 1]
    u, _u_x, u_t, u_xx = field_derivs(net, X_coll)
    y = u_t
    A = build_features(u, u_xx)

    if STAT == "chi":
        score, c_hat = _chi_per_axis(y, A, x_c, t_c)
    elif STAT == "het":
        h = het_per_window(y, A, _masks(x_c, t_c, A.dtype), ridge=EPS)
        score = h["score"] if STAT_BOUND else h["score_raw"]
        c_hat, _ = global_ols(y, A, ridge=EPS)
    else:
        score = None
        c_hat, _ = global_ols(y, A, ridge=EPS)

    l_stat = (score.sum().to(y.dtype) if score is not None
              else torch.zeros((), device=device))

    l_anch = torch.zeros((), device=device)
    # Gated on ANCHOR, NOT on THETA_FD (COEF_SOURCE='fd' builds it too).
    if ANCHOR == "fd":
        per = anchor_penalty(c_hat, THETA_FD.to(c_hat.dtype), ANCHOR_NORM,
                             ANCHOR_COL_MS, ANCHOR_Y_VAR)
        if ANCHOR_WEIGHT != "none":
            per = per * score.detach().to(per.dtype)
        l_anch = per.sum().to(y.dtype)

    w0 = float("nan")
    if COEF_SOURCE == "tls":
        # Target-free: the smallest eigenvalue of the scaled design, no
        # residual against the target. The logged recovery is the TLS one.
        l_phys, c_hat, w0 = tls_fit(y, A, TLS_SCALE)
    else:
        if COEF_SOURCE == "true":
            c_phys = THETA_TRUE
        elif COEF_SOURCE == "fd":
            c_phys = THETA_FD
        else:
            c_phys = c_hat.to(y.dtype)
        r = (y - A @ c_phys) / PHYS_DIV
        l_phys = (r ** 2).mean()

    return dict(phys=l_phys, stat=l_stat, anch=l_anch,
                c_hat=c_hat.detach(), w0=w0)


def ic_loss(net):
    """u on the t=0 row over Var(U). IC_MODE='none' removes it: it is one
    row of what l_data already supervises."""
    if IC_MODE == "none":
        return torch.zeros((), device=device)
    return ((net(X_IC) - U_IC) ** 2).mean() / IC_NORM_U


def bc_loss(net, X_bc):
    """Periodic: u and u_x equal at x=-1 and x=1, each over its own
    channel variance."""
    if IC_MODE == "none":
        return torch.zeros((), device=device)
    n = X_bc.shape[0] // 2
    u_b, u_x, _ut, _uxx = field_derivs(net, X_bc)
    return ((u_b[:n] - u_b[n:]) ** 2).mean() / IC_NORM_U \
        + ((u_x[:n] - u_x[n:]) ** 2).mean() / IC_NORM_UX


def data_loss(net):
    """Observation term over the field's own variance -- dimensionless,
    weight 1. Zero and gradient-free when DATA_TERM is False."""
    if not DATA_TERM:
        return torch.zeros((), device=device)
    return observation_loss(net(X_train), U_train, IC_NORM_U)


def total_loss(ml, l_ic, l_bc, l_dat):
    """THE loss. No coefficients: their absence is the design."""
    return combine_loss([ml["phys"] if PHYS_TERM else 0.0, l_ic, l_bc, ml["stat"],
                         ml["anch"],
                         l_dat], LOSS_FORM)


# ============================================================ train
torch.manual_seed(SEED); np.random.seed(SEED)
g = torch.Generator(device=device).manual_seed(SEED)

net = MLP(2, 1, HIDDEN).to(device)


def sample_collocation(gen):
    x = torch.rand(N_COLL, generator=gen, device=device) * (X_MAX - X_MIN) + X_MIN
    t = torch.rand(N_COLL, generator=gen, device=device) * (t_split - T_MIN) + T_MIN
    out = torch.stack([x, t], dim=1)
    assert_train_only("collocation", out[:, 1], t_split)
    return out


def sample_bc(gen):
    t = torch.rand(N_BC, generator=gen, device=device) * (t_split - T_MIN) + T_MIN
    left = torch.stack([torch.full_like(t, X_MIN), t], dim=1)
    right = torch.stack([torch.full_like(t, X_MAX), t], dim=1)
    out = torch.cat([left, right], dim=0)
    assert_train_only("bc", out[:, 1], t_split)
    return out


ANCHOR_ERR = (rel_l1(THETA_FD.cpu().numpy(), THETA_TRUE_NP)
              if THETA_FD is not None else None)
if ANCHOR_ERR is not None:
    print(f"FD anchor (data-driven, no truth consulted): "
          f"{THETA_FD.cpu().numpy().round(6).tolist()}  rel-L1 err {ANCHOR_ERR:.4e}")
print(f"library {FEATURE_NAMES}   yardsticks: PHYS_NORM={PHYS_NORM} "
      f"div={PHYS_DIV:.4g} (max={PHYS_SCALE:.4g} sqrtVar={np.sqrt(PHYS_VAR):.4g})  "
      f"Var(U)={IC_NORM_U:.4g}  Var(u_x)={IC_NORM_UX:.4g}   (grid {Nx}x{Nt}, "
      f"device {device})", flush=True)

hist = {"iter": [], "tot": [], "phys": [], "stat": [], "ic": [], "bc": [],
        "anch": [], "dat": [], "gphys": [], "gdat": []}


def sample_batch(gen):
    """Collocation THEN boundary -- the generator is shared, so the order
    is part of the seed's meaning."""
    return sample_collocation(gen), sample_bc(gen)


def _loss_fn(batch):
    X_coll, X_bc = batch
    ml = physics_and_stat_loss(net, X_coll)
    l_ic = ic_loss(net)
    l_bc = bc_loss(net, X_bc)
    l_dat = data_loss(net)
    loss = total_loss(ml, l_ic, l_bc, l_dat)
    terms = {"phys": ml["phys"], "stat": ml["stat"], "anch": ml["anch"],
             "ic": l_ic, "bc": l_bc, "dat": l_dat}
    return loss, terms, ml


def _log(it, cur, terms, ml, gsh):
    hist["iter"].append(it); hist["tot"].append(cur)
    for k in ("phys", "stat", "ic", "bc", "anch", "dat"):
        hist[k].append(_f(terms[k]))
    hist["gphys"].append(gsh["phys"]); hist["gdat"].append(gsh["dat"])
    print(f"[adam {it:5d}] tot={cur:.3e}  phys={_f(terms['phys']):.3e}  "
          f"stat={_f(terms['stat']):.3e}  anch={_f(terms['anch']):.3e}  "
          f"ic={_f(terms['ic']):.3e}  bc={_f(terms['bc']):.3e}  "
          f"dat={_f(terms['dat']):.3e}  "
          f"|g| phys={gsh['phys']:.2e} dat={gsh['dat']:.2e} "
          f"(phys share {grad_share(gsh):.0%})  "
          f"c_hat={ml['c_hat'].cpu().numpy().round(6).tolist()}"
          + (f"  w0={ml['w0']:.3f}" if COEF_SOURCE == "tls" else ""),
          flush=True)


best = Checkpoint(
    net, key=BEST_KEY,
    err_fn=lambda ml: rel_l1(ml["c_hat"].cpu().numpy(), THETA_TRUE_NP),
    extra_fn=lambda ml: {"c": ml["c_hat"].cpu().numpy().tolist()})
final, elapsed = train(
    net, _loss_fn, adam_iters=ADAM_ITERS, adam_lr=ADAM_LR,
    lbfgs_max=LBFGS_MAX, checkpoint=best, sampler=sample_batch, gen=g,
    log_every=2000, log_fn=_log, grad_terms=("phys", "ic", "bc", "dat"))

with torch.no_grad():
    U_final = net(X_obs).cpu().numpy().reshape(Nx, Nt)
rel_final = float(np.linalg.norm(U_final - U) / np.linalg.norm(U))

best.restore()

# ============================================================ evaluate
with torch.no_grad():
    U_pred = net(X_obs).cpu().numpy().reshape(Nx, Nt)
rel_l2 = float(np.linalg.norm(U_pred - U) / np.linalg.norm(U))
_tr, _te = slice(0, n_train), slice(n_train, Nt)
rel_l2_train = float(np.linalg.norm(U_pred[:, _tr] - U[:, _tr])
                     / np.linalg.norm(U[:, _tr]))
rmse_train = float(np.sqrt(np.mean((U_pred[:, _tr] - U[:, _tr]) ** 2)))
rel_l2_test = (float(np.linalg.norm(U_pred[:, _te] - U[:, _te])
                     / np.linalg.norm(U[:, _te])) if n_train < Nt else float("nan"))

X_eval = sample_collocation(torch.Generator(device=device).manual_seed(SEED + 1))
u_e, _ux_e, ut_e, uxx_e = field_derivs(net, X_eval)
y_e = ut_e.detach()
A_e = build_features(u_e.detach(), uxx_e.detach())
c_hat_t, _res = global_ols(y_e, A_e, ridge=EPS)
c_hat_ols = c_hat_t.cpu().numpy().astype(np.float64)
# The TLS readout on the same field, reported in EVERY arm so the
# estimator and the loss can be separated.
with torch.no_grad():
    _lam_e, _c_tls_t, tls_w0_eval = tls_fit(y_e, A_e, TLS_SCALE)
tls_lam_eval = float(_lam_e)
c_hat_tls = _c_tls_t.cpu().numpy().astype(np.float64)
# The ARM's own readout -- the estimator its loss trained. Drives the
# headline error and the integration test.
c_hat = c_hat_tls if COEF_SOURCE == "tls" else c_hat_ols
res_rmse_eval = float(torch.sqrt(((y_e - A_e @ c_hat_t) ** 2).mean()))
res_rmse_eval_true = float(torch.sqrt(((y_e - A_e @ THETA_TRUE) ** 2).mean()))

x_e, t_e = X_eval[:, 0], X_eval[:, 1]
with torch.no_grad():
    het_e = {k: v.cpu().numpy() for k, v in
             het_per_window(y_e, A_e, _masks(x_e, t_e, A_e.dtype),
                            ridge=EPS).items()}
    chi_t = chi2_per_term(y_e[torch.argsort(t_e)].double(),
                          A_e[torch.argsort(t_e)].double(),
                          ridge=EPS, use_floor=STAT_FLOOR)["score"].cpu().numpy()
    chi_x = chi2_per_term(y_e[torch.argsort(x_e)].double(),
                          A_e[torch.argsort(x_e)].double(),
                          ridge=EPS, use_floor=STAT_FLOOR)["score"].cpu().numpy()
    anchor_scale = float(max_corr(y_e, A_e))

net_err = rel_l1(c_hat, THETA_TRUE_NP)
net_err_ols = rel_l1(c_hat_ols, THETA_TRUE_NP)
net_err_tls = rel_l1(c_hat_tls, THETA_TRUE_NP)

# ---------------------------------------------------- test by INTEGRATION
# The net past t_split measures nothing (never trained there). Integrate
# the RECOVERED equation from the state the net holds AT t_split, beside
# two controls from the SAME handover: this library's true coefficients,
# and the full true equation. Spectral periodic Laplacian, DOP853.
_t_test = t_grid[n_train:].astype(np.float64)
_k = 2.0 * np.pi * np.fft.fftfreq(Nx, d=DX)
with torch.no_grad():
    _Xs = torch.tensor(np.stack([x_grid, np.full_like(x_grid, t_split)], axis=1),
                       device=device)
    u0_hand = net(_Xs).cpu().numpy().ravel().astype(np.float64)


def _integrate_ac(coefs, u0):
    D = float(coefs.get("u_xx", 0.0))
    a, b = float(coefs["u"]), float(coefs["u^3"])
    if len(_t_test) == 0:
        return np.zeros((Nx, 0))

    def rhs(_t, u):
        lap = np.real(np.fft.ifft(-(_k ** 2) * np.fft.fft(u))) if D != 0.0 else 0.0
        return D * lap + a * u + b * u ** 3

    sol = solve_ivp(rhs, (t_split, float(_t_test[-1])), u0, t_eval=_t_test,
                    method="DOP853", rtol=1e-10, atol=1e-12)
    if not sol.success:
        return np.full((Nx, len(_t_test)), np.nan)
    return sol.y


_U_te = U[:, n_train:].astype(np.float64)
_test_rows = [("recovered", dict(zip(FEATURE_NAMES, c_hat))),
              ("true, this library", dict(zip(FEATURE_NAMES, THETA_TRUE_NP))),
              ("true, full equation", TRUE_COEFS)]
if THETA_FD is not None:
    # What COEF_SOURCE='fd' actually held the net to, and what ANCHOR='fd'
    # pulled toward -- same handover state as every other row.
    _test_rows.append(("FD coefs (theta_fd)",
                       dict(zip(FEATURE_NAMES, THETA_FD.cpu().numpy()))))
TEST = {}
for name, coefs in _test_rows:
    S = _integrate_ac(coefs, u0_hand)
    TEST[name] = dict(
        U=S,
        rel_l2=float(np.linalg.norm(S - _U_te) / np.linalg.norm(_U_te)),
        rmse=float(np.sqrt(np.mean((S - _U_te) ** 2))),
        rmse_t1=float(np.sqrt(np.mean((S[:, -1] - _U_te[:, -1]) ** 2))))


print(f"\n========== ALLEN-CAHN AUTOSCALE PINN (weight-free loss) ==========")
print(f"library: {FEATURE_NAMES}   (DIFFUSION={DIFFUSION})")
print(f"mode: coefs={COEF_SOURCE}  ic={IC_MODE}  stat={STAT}"
      f"{'' if STAT == 'none' else ('/bounded' if STAT_BOUND else '/RAW')}  "
      f"anchor={ANCHOR}/{ANCHOR_WEIGHT}/{ANCHOR_NORM}  fd_stride={FD_STRIDE}  fd_order={FD_ORDER}  fd_x={FD_X}  "
      f"data_term={DATA_TERM}  select={BEST_KEY}")
print(f"loss = {LOSS_FORM} of [l_phys, l_ic, l_bc, l_stat, l_anch, l_data]   (no weights)")
print(f"train/test split        = {TRAIN_FRAC:.2f} of t  "
      f"(t_split={t_split:.3f}, n_train={n_train}/{Nt})  loss sees train only")
print(f"selected checkpoint     = iter {best.iter}  ({BEST_KEY}={best.best:.3e})")
print(f"  coef err there        = {best.err:.3e}   (truth-based, DIAGNOSTIC)")
print(f"last loss               = {final.item():.3e}")
print(f"training time           = {elapsed:.1f}s")
print(f"rel L2 (best / final)   = {rel_l2:.4e} / {rel_final:.4e}")
print(f"rel L2 train (net)      = {rel_l2_train:.4e}   RMSE train = {rmse_train:.4e}")
print(f"rel L2 TEST  (net)      = {rel_l2_test:.4e}   "
      f"(NOT a result: the net was never trained past t_split)")
for name, r in TEST.items():
    print(f"TEST by INTEGRATION, {name:<20} rel L2 {r['rel_l2']:.4e}   "
          f"RMSE {r['rmse']:.4e}   RMSE(t=1) {r['rmse_t1']:.4e}")
print(f"  c_hat = {np.round(c_hat, 6).tolist()}   "
      f"true = {THETA_TRUE_NP.tolist()}   rel-L1 err = {net_err:.4e}")
print(f"  readouts: OLS {np.round(c_hat_ols, 6).tolist()} (rel-L1 {net_err_ols:.4e})"
      f"   TLS {np.round(c_hat_tls, 6).tolist()} (rel-L1 {net_err_tls:.4e})"
      f"   TLS lam={tls_lam_eval:.3e} w0={tls_w0_eval:.4f}")
print(f"  residual RMSE on eval collocation: OLS {res_rmse_eval:.4e}   "
      f"true coefs {res_rmse_eval_true:.4e}")
if ANCHOR_ERR is not None:
    print(f"  anchor rel-L1 err {ANCHOR_ERR:.4e}   "
          f"({'IMPROVED' if net_err < ANCHOR_ERR else 'no gain'})")
print(f"  chi(t-path) = {np.round(chi_t, 6).tolist()}   "
      f"chi(x-path) = {np.round(chi_x, 6).tolist()}")
print(f"  het = {np.round(het_e['score'], 6).tolist()}   "
      f"het raw = {np.round(het_e['score_raw'], 6).tolist()}")
print(f"  max_corr = {anchor_scale:.4e}   (sparsity.py anchor)")

out_path = f"ac_pinn_auto{_SUFFIX}.npz"
np.savez(out_path,
         U_data=U, U_pred=U_pred, U_final=U_final, x=x_grid, t=t_grid,
         diffusion=np.bool_(DIFFUSION), feature_names=np.array(FEATURE_NAMES),
         c_hat=c_hat, c_true=THETA_TRUE_NP, net_err=np.float64(net_err),
         c_hat_ols=c_hat_ols, c_hat_tls=c_hat_tls,
         net_err_ols=np.float64(net_err_ols), net_err_tls=np.float64(net_err_tls),
         tls_lam=np.float64(tls_lam_eval), tls_w0=np.float64(tls_w0_eval),
         tls_scale=TLS_SCALE.cpu().numpy(),
         c_fd=(THETA_FD.cpu().numpy() if THETA_FD is not None
               else np.full(len(FEATURE_NAMES), np.nan)),
         rel_l2=np.float32(rel_l2), rel_l2_final=np.float32(rel_final),
         rel_l2_train=np.float32(rel_l2_train), rmse_train=np.float32(rmse_train),
         rel_l2_test=np.float32(rel_l2_test),
         rel_l2_test_int=np.float32(TEST["recovered"]["rel_l2"]),   # THE test metric
         rmse_test_int=np.float32(TEST["recovered"]["rmse"]),
         rel_l2_test_int_true_lib=np.float32(TEST["true, this library"]["rel_l2"]),
         rmse_test_int_true_lib=np.float32(TEST["true, this library"]["rmse"]),
         rel_l2_test_int_true=np.float32(TEST["true, full equation"]["rel_l2"]),
         rmse_test_int_true=np.float32(TEST["true, full equation"]["rmse"]),
         rmse_test_int_fd=np.float32(TEST["FD coefs (theta_fd)"]["rmse"]
                                     if THETA_FD is not None else np.nan),
         U_test_int=TEST["recovered"]["U"].astype(np.float32),
         res_rmse_eval=np.float32(res_rmse_eval),
         res_rmse_eval_true=np.float32(res_rmse_eval_true),
         train_frac=np.float32(TRAIN_FRAC), n_train=np.int32(n_train),
         t_split=np.float32(t_split),
         phys_scale=np.float32(PHYS_SCALE), phys_var=np.float32(PHYS_VAR),
         phys_div=np.float32(PHYS_DIV), phys_norm=np.str_(PHYS_NORM),
         phys_agg=np.str_(PHYS_AGG), ic_norm_u=np.float32(IC_NORM_U),
         ic_norm_ux=np.float32(IC_NORM_UX), max_corr=np.float32(anchor_scale),
         chi_t=chi_t, chi_x=chi_x,
         het_score=het_e["score"], het_score_raw=het_e["score_raw"],
         het_se_rel=het_e["se_rel"],
         training_time=np.float32(elapsed), best_iter=np.int32(best.iter),
         loss_history_iter=np.array(hist["iter"], dtype=np.int32),
         loss_history_total=np.array(hist["tot"], dtype=np.float32),
         loss_history_phys=np.array(hist["phys"], dtype=np.float32),
         loss_history_stat=np.array(hist["stat"], dtype=np.float32),
         loss_history_ic=np.array(hist["ic"], dtype=np.float32),
         loss_history_bc=np.array(hist["bc"], dtype=np.float32),
         loss_history_anch=np.array(hist["anch"], dtype=np.float32),
         loss_history_dat=np.array(hist["dat"], dtype=np.float32),
         grad_phys=np.array(hist["gphys"], dtype=np.float32),
         grad_data=np.array(hist["gdat"], dtype=np.float32),
         coef_source=np.str_(COEF_SOURCE), ic_mode=np.str_(IC_MODE),
         stat=np.str_(STAT), stat_bound=np.bool_(STAT_BOUND),
         anchor=np.str_(ANCHOR), anchor_norm=np.str_(ANCHOR_NORM),
         fd_stride=np.int32(FD_STRIDE), fd_order=np.int32(FD_ORDER),
         fd_x=np.str_(FD_X), loss_form=np.str_(LOSS_FORM),
         phys_term=np.bool_(PHYS_TERM),
         anchor_weight=np.str_(ANCHOR_WEIGHT), best_key=np.str_(BEST_KEY))
print(f"saved -> {out_path}")
