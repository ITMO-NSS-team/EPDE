#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""1-D wave PINN with the weight-free loss -- the DP recipe moved across
UNCHANGED. This is the sharpest transfer test in the set.

`pinn_test_cv.py` here carries the only hand-set weights left in the
family::

    W_PDE, W_CV, W_BC, W_IC = 1.0, 1.0, 100.0, 100.0

The two 100s are exactly what the yardstick construction is supposed to
make unnecessary: the BC and IC terms are measured in u^2 while the PDE
residual is measured in u_tt^2, so they need a hand-set factor to
compete. Dividing each by a scale taken from the data removes the
comparison problem instead of compensating for it, and every weight
becomes 1::

    loss = l_phys + l_ic + l_bc + l_stat + l_anch      (no coefficients)

    l_phys  mean( ((u_tt - c * u_xx) / PHYS_SCALE)^2 ),
            PHYS_SCALE = max|u_tt| from FD on the observed grid
    l_ic    mean((u - U_IC)^2)/Var(U) + mean((u_t - UT_IC)^2)/Var(u_t_fd)
    l_bc    mean(u(boundary)^2) / Var(U)
    l_anch  ((c_hat - c_FD)/c_FD)^2, c_FD = OLS of u_tt on u_xx over the
            FD design matrix built from the OBSERVED field -- data-driven,
            no truth, no pointwise supervision on u
    l_stat  het (per-window) or chi (per-axis score paths), weight 1

Nothing is re-tuned for this system. What differs is only what the
system supplies: its own library (u_xx), its own PHYS_SCALE, its own
Var(U).

chi ON A 2-D GRID: the torch chi2_per_term walks ONE cumulative score
path in row order, so on a 2-D grid it must be run once per axis and
summed -- a t-ordered path alone is blind to a defect that is separable
in x (the EPDE panel measured exactly this on wave: t-only scored 0.54x
where a per-axis score gave 390x). Both orderings share the same global
OLS, since the fit is order-invariant and only the path is not.

Harness is wave's own, untouched: HIDDEN [64]*5, N_COLL 8000, N_BC 200,
ADAM 20k + one LBFGS, SEED 0, 30 strips per dim, WIN_FRAC 0.5.

OBSERVATIONS IN THE LOSS (August 2026 sweep). The default is now
COEF_SOURCE='ols' + DATA_TERM=True: the physics residual carries the
net's OWN OLS recovery and an observation term pins the field. That
configuration was previously a COLLAPSE here -- under-determined, with
the trivial state scoring zero -- and supervision is exactly what
removes the degeneracy. Measured on coefficient error, the
pre-registered ranking metric:
    c2 rel err 3.493e-2 -> 8.261e-3 (4.2x), c2_hat 0.038603 ->
    0.039670 against a true 0.04; rel-L2 4.115e-3 -> 1.929e-3.
It also beats THETA_FD itself (3.494e-2), so it REPLACES the FD coefficient
scaffolding rather than supplementing it.

The same sweep tested whether supervision rescues the statistic terms
(chi, het, the FD anchor), since every one of them collapses by the net
driving its own design matrix to zero. It does NOT: all three collapsed
again with the data term on, max_corr falling from 276 to 1.36e-3 /
1.30e-2 / 9.15. Field ownership is not fixed by pinning the field.

Cost of the default, stated plainly: with pointwise supervision the net
fits u, so "coefficients from equation structure alone" weakens to
"coefficients from a smoothed fit". This reverses the July 2026 removal
of the u data term; artifacts before and after are not comparable.

l_data is weight-free by the same rule as every other term -- it divides
by the observed field's own variance (``observation_loss`` in
dp/cv_metric.py, shared by all three systems) -- and TRAIN_FRAC=0.8
splits EVERYTHING the loss touches: physics collocation, BC sampling,
l_data, the FD design and the yardsticks all stop at t_split, exactly as
dp does. The late-time block is genuine extrapolation.
"""

import sys
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pinn_common import (                    # shared PINN model + driver
    Checkpoint,
    MLP,
    assert_train_only,
    as_float as _f,
    grad_share,
    train,
)
from stat_common import (                    # shared, from dp/cv_metric.py
    bounded,
    chi2_per_term,
    global_ols,
    het_per_window,
    max_corr,
    observation_loss,
)

torch.set_default_dtype(torch.float32)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================ settings
DATA_PATH = "./wave_sln_80.csv"
C2_TRUE   = 0.04                  # diagnostics only -- never in the loss
X_MIN, X_MAX = 0.0, 1.0
T_MIN, T_MAX = 0.0, 1.0

HIDDEN     = [64] * 5
N_COLL     = 8000
N_BC       = 200
ADAM_ITERS = 20000
ADAM_LR    = 1e-3
LBFGS_MAX  = 50000
SEED       = 0

N_WIN_PER_DIM = 30
WIN_FRAC      = 0.5
EPS = 1e-12

# ============================================================ mode block
# Physics yardstick: which statistic of the FD target normalizes the
# residual. 'var' -> l_phys = mean(r^2)/Var(y) ~ 1 - R^2, the same
# second-moment rule the IC (and BC) already use, so every term is a
# fraction of its own channel's signal. 'max' = max|target|^2, an
# EXTREMUM: one bad FD point sets the scale for the whole term.
# wave has ALWAYS used the plain mean here (no windowed variant
# exists in this script) -- recorded so artifacts carry the same key
# as dp/duffing, where it is a real switch.
PHYS_AGG      = "mean"     # structural on wave
PHYS_NORM     = "var"       # 'var' | 'max'
COEF_SOURCE = "ols"   # 'ols' | 'fd' | 'true'
IC_MODE = "none"   # 'scaled' | 'none': 'none' drops l_ic AND l_bc,
                        # because l_data already supervises u on the whole
                        # train block -- the boundary columns and the t=0
                        # row ARE part of that block.
STAT          = "none"      # 'none' | 'het' | 'chi'
STAT_BOUND    = True
STAT_FLOAT64  = True
STAT_FLOOR    = False
ANCHOR        = "none"        # 'none' | 'fd'
FD_STRIDE     = 1
ANCHOR_WEIGHT = "none"      # 'none' | 'stat'
BEST_KEY      = "loss"      # truth-free
DATA_TERM = True   # pointwise supervision on the observed u
TRAIN_FRAC    = 0.8         # temporal split, mirroring dp: the loss --
                            # physics, data, BC, the FD design and the
                            # yardsticks -- sees ONLY t < t_split (all
                            # x). The late-time block is never shown to
                            # anything, so error there is genuine
                            # extrapolation from the learned equation.

_SUFFIX = ({"ols": "", "fd": f"_cfd{FD_STRIDE}", "true": "_true"}[COEF_SOURCE]) \
    + ("" if STAT == "none" else f"_{STAT}") \
    + ("_raw" if (STAT == "chi" and not STAT_BOUND) else "") \
    + ("" if ANCHOR == "none" else f"_afd{FD_STRIDE}") \
    + ("_wstat" if ANCHOR_WEIGHT != "none" else "") \
    + ("" if PHYS_NORM == "var" else "_pmax") \
    + ("_data" if DATA_TERM else "") \
    + ("_noicbc" if IC_MODE == "none" else "")

if IC_MODE == "none" and not DATA_TERM:
    raise ValueError("IC_MODE='none' removes the only anchor on the solution "
                     "family unless DATA_TERM supervises the field")

if ANCHOR_WEIGHT != "none" and (ANCHOR == "none" or STAT == "none"):
    raise ValueError("ANCHOR_WEIGHT='stat' requires ANCHOR='fd' and a STAT")
if not 0.0 < TRAIN_FRAC <= 1.0:
    raise ValueError(f"bad TRAIN_FRAC {TRAIN_FRAC!r}")


# ============================================================ data
U = np.loadtxt(DATA_PATH, delimiter=",").astype(np.float32)
Nx, Nt = U.shape

# The split, before anything that reads the field: FD design,
# yardsticks, collocation, BC and the t-windows must all stop at
# t_split, or the late-time block is not held out.
n_train = int(np.ceil(TRAIN_FRAC * Nt))
x_grid = np.linspace(X_MIN, X_MAX, Nx, dtype=np.float32)
t_grid = np.linspace(T_MIN, T_MAX, Nt, dtype=np.float32)
t_split = float(t_grid[n_train - 1])
XX, TT = np.meshgrid(x_grid, t_grid, indexing="ij")
X_obs = torch.tensor(np.stack([XX.ravel(), TT.ravel()], axis=1), device=device)
u_obs = torch.tensor(U.reshape(-1, 1), device=device)

DT = float(t_grid[1] - t_grid[0])
X_IC  = torch.tensor(np.stack([x_grid, np.full_like(x_grid, T_MIN)], axis=1),
                     device=device)
U_IC  = torch.tensor(U[:, 0:1], device=device)
UT_IC = torch.tensor(
    ((-3.0 * U[:, 0] + 4.0 * U[:, 1] - U[:, 2]) / (2.0 * DT)).reshape(-1, 1),
    device=device)


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
# ALL yardsticks use the POPULATION variance (ddof=0): we want the
# spread the observed signal actually has, not an estimate of a
# generative parameter. torch defaults to ddof=1 and numpy to ddof=0,
# so the convention is pinned explicitly at every call site rather
# than inherited from whichever library holds the array.

def _fd_design(stride=1):
    """u_tt and u_xx from the OBSERVED field by central differences --
    equation-level information, never a pointwise fit of u."""
    Us = U[::stride, :n_train:stride].astype(np.float64)
    xs = x_grid[::stride].astype(np.float64)
    ts = t_grid[:n_train:stride].astype(np.float64)
    u_x = np.gradient(Us, xs, axis=0)
    u_xx = np.gradient(u_x, xs, axis=0)
    u_t = np.gradient(Us, ts, axis=1)
    u_tt = np.gradient(u_t, ts, axis=1)
    # Drop the one-sided edges: second differences there are much worse
    # and would bias the anchor. Data property, not a tuned window.
    core = (slice(1, -1), slice(1, -1))
    y = torch.tensor(u_tt[core].ravel())
    A = torch.tensor(u_xx[core].ravel()).unsqueeze(1)
    return y, A, torch.tensor(u_t[core].ravel())


_y_fd, _A_fd, _ut_fd = _fd_design()
PHYS_SCALE = float(_y_fd.abs().max()) + EPS
PHYS_VAR = float(_y_fd.var(unbiased=False)) + EPS
PHYS_DIV = (PHYS_SCALE if PHYS_NORM == "max"
            else float(np.sqrt(PHYS_VAR)))
IC_NORM_U = float(U[:, :n_train].var()) + EPS
IC_NORM_UT = float(_ut_fd.var(unbiased=False)) + EPS

# Train sub-grid for l_data: every x, t < t_split. Physics uses the
# samplers, which are bounded by t_split as well.
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
C2_T = torch.tensor([C2_TRUE], device=device)


# ============================================================ network
def second_derivs(net, xt):
    xt = xt.clone().requires_grad_(True)
    u = net(xt)
    g = torch.autograd.grad(u, xt, torch.ones_like(u), create_graph=True)[0]
    u_x, u_t = g[:, 0:1], g[:, 1:2]
    u_xx = torch.autograd.grad(u_x, xt, torch.ones_like(u_x),
                               create_graph=True)[0][:, 0]
    u_tt = torch.autograd.grad(u_t, xt, torch.ones_like(u_t),
                               create_graph=True)[0][:, 1]
    return u_xx, u_tt


# ============================================================ loss
def _chi_per_axis(y, A, x_c, t_c):
    """chi summed over ONE score path PER GRID AXIS. The OLS is
    order-invariant so both orderings share a fit; only the cumulative
    path differs, and a single-axis path is blind to defects separable
    in the other coordinate."""
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


def physics_and_stat_loss(net, X_coll):
    x_c, t_c = X_coll[:, 0], X_coll[:, 1]
    u_xx, u_tt = second_derivs(net, X_coll)
    y = u_tt
    A = u_xx.unsqueeze(1)

    in_t = (t_c.unsqueeze(0) >= T_LO) & (t_c.unsqueeze(0) < T_HI)
    in_x = (x_c.unsqueeze(0) >= X_LO) & (x_c.unsqueeze(0) < X_HI)
    mask = torch.cat([in_t, in_x], dim=0).to(A.dtype)   # (2*N_WIN, N_COLL)

    if STAT == "chi":
        score, c_hat = _chi_per_axis(y, A, x_c, t_c)
    elif STAT == "het":
        score = het_per_window(y, A, mask, ridge=EPS)["score"]
        c_hat, _ = global_ols(y, A, ridge=EPS)
    else:
        score = None
        c_hat, _ = global_ols(y, A, ridge=EPS)

    l_stat = (score.sum().to(y.dtype) if score is not None
              else torch.zeros((), device=device))

    l_anch = torch.zeros((), device=device)
    # Gated on ANCHOR, NOT on THETA_FD: COEF_SOURCE='fd' also builds
    # THETA_FD, and keying off it would silently apply the anchor in a
    # cell meant to isolate the residual.
    if ANCHOR == "fd":
        a = THETA_FD.to(c_hat.dtype)
        per = ((c_hat - a) / a) ** 2
        if ANCHOR_WEIGHT != "none":
            per = per * score.detach().to(per.dtype)
        l_anch = per.sum().to(y.dtype)

    if COEF_SOURCE == "true":
        c_phys = C2_T
    elif COEF_SOURCE == "fd":
        c_phys = THETA_FD
    else:
        c_phys = c_hat.to(y.dtype)
    r = (y - A @ c_phys) / PHYS_DIV
    l_phys = (r ** 2).mean()

    return dict(phys=l_phys, stat=l_stat, anch=l_anch,
                c_hat=c_hat.detach())


def ic_loss(net):
    """u and u_t on the t=0 slice, each divided by the variance that
    channel explores -- this is what replaces W_IC = 100. IC_MODE='none'
    removes it: it is a single row of what l_data already supervises."""
    if IC_MODE == "none":
        return torch.zeros((), device=device)
    xt = X_IC.clone().requires_grad_(True)
    u = net(xt)
    u_t = torch.autograd.grad(u, xt, torch.ones_like(u),
                              create_graph=True)[0][:, 1:2]
    return ((u - U_IC) ** 2).mean() / IC_NORM_U \
        + ((u_t - UT_IC) ** 2).mean() / IC_NORM_UT


def bc_loss(net, X_bc):
    """Homogeneous Dirichlet, divided by the field's own variance --
    what replaces W_BC = 100."""
    if IC_MODE == "none":
        return torch.zeros((), device=device)
    return (net(X_bc) ** 2).mean() / IC_NORM_U


def data_loss(net):
    """Observation term, divided by the SAME field variance l_ic and
    l_bc already use -- dimensionless, weight 1, no constant added.
    Needs its own forward pass (physics runs on random collocation
    points), but no derivatives, so it is cheaper than one physics
    pass. Zero and gradient-free when DATA_TERM is False."""
    if not DATA_TERM:
        return torch.zeros((), device=device)
    return observation_loss(net(X_train), U_train, IC_NORM_U)


def total_loss(ml, l_ic, l_bc, l_dat):
    """THE loss. No coefficients: their absence is the design."""
    return ml["phys"] + l_ic + l_bc + ml["stat"] + ml["anch"] + l_dat


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
    left  = torch.stack([torch.full_like(t, X_MIN), t], dim=1)
    right = torch.stack([torch.full_like(t, X_MAX), t], dim=1)
    out = torch.cat([left, right], dim=0)
    assert_train_only("bc", out[:, 1], t_split)
    return out


ANCHOR_ERR = (abs(float(THETA_FD) - C2_TRUE) / C2_TRUE
              if THETA_FD is not None else None)
if ANCHOR_ERR is not None:
    print(f"FD anchor (data-driven, no truth consulted): "
          f"c2_fd={float(THETA_FD):.6f}  true={C2_TRUE}  "
          f"rel err {ANCHOR_ERR:.4e}")
print(f"yardsticks: PHYS_NORM={PHYS_NORM} div={PHYS_DIV:.4g} (max={PHYS_SCALE:.4g} sqrtVar={np.sqrt(PHYS_VAR):.4g})  Var(U)={IC_NORM_U:.4g}  "
      f"Var(u_t)={IC_NORM_UT:.4g}   (grid {Nx}x{Nt})")

hist = {"iter": [], "tot": [], "phys": [], "stat": [], "ic": [], "bc": [],
        "anch": [], "dat": [], "gphys": [], "gdat": []}


def sample_batch(gen):
    """Collocation THEN boundary, in that order -- the generator is
    shared, so the order is part of the seed's meaning."""
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
          f"c2_hat={float(ml['c_hat']):.5f}", flush=True)


best = Checkpoint(
    net, key=BEST_KEY,
    err_fn=lambda ml: abs(float(ml["c_hat"]) - C2_TRUE) / C2_TRUE,
    extra_fn=lambda ml: {"c": float(ml["c_hat"])})
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
# The late-time block the LOSS never saw -- no data, no physics, no BC,
# no FD. Genuine extrapolation, the same thing dp reports as test.
_rl2 = lambda s: float(np.linalg.norm(U_pred[:, s] - U[:, s])
                       / np.linalg.norm(U[:, s]))
rel_l2_train = _rl2(slice(0, n_train))
rel_l2_test = (_rl2(slice(n_train, Nt)) if n_train < Nt else float("nan"))

X_eval = sample_collocation(torch.Generator(device=device).manual_seed(SEED + 1))
u_xx_e, u_tt_e = second_derivs(net, X_eval)
y_e, A_e = u_tt_e.detach(), u_xx_e.detach().unsqueeze(1)
c_hat, _ = global_ols(y_e, A_e, ridge=EPS)
c_hat = float(c_hat)

x_e, t_e = X_eval[:, 0], X_eval[:, 1]
in_t = (t_e.unsqueeze(0) >= T_LO) & (t_e.unsqueeze(0) < T_HI)
in_x = (x_e.unsqueeze(0) >= X_LO) & (x_e.unsqueeze(0) < X_HI)
mask_e = torch.cat([in_t, in_x], dim=0).to(A_e.dtype)
with torch.no_grad():
    het_e = {k: v.cpu().numpy()
             for k, v in het_per_window(y_e, A_e, mask_e, ridge=EPS).items()}
    chi_t = chi2_per_term(y_e[torch.argsort(t_e)].double(),
                          A_e[torch.argsort(t_e)].double(),
                          ridge=EPS, use_floor=STAT_FLOOR)["score"]
    chi_x = chi2_per_term(y_e[torch.argsort(x_e)].double(),
                          A_e[torch.argsort(x_e)].double(),
                          ridge=EPS, use_floor=STAT_FLOOR)["score"]
    anchor_scale = float(max_corr(y_e, A_e))

net_err = abs(c_hat - C2_TRUE) / C2_TRUE

# ---------------------------------------------------- test by INTEGRATION
# Evaluating the NET past t_split measures nothing: it was trained only
# on t <= t_split, so its output there is unconstrained by construction.
# rel_l2_test above is kept, but only as the diagnostic that shows
# exactly that. The question the experiment is actually asking is whether
# the RECOVERED equation extrapolates, so integrate u_tt = c2 u_xx
# forward from the state the net holds AT t_split. The ends are held at
# zero because the TRAIN block exhibits |u| <= 1.2e-16 on both boundary
# columns -- read off the observed data, not assumed from the truth.
_t_test = t_grid[n_train:].astype(np.float64)
_dx = float(x_grid[1] - x_grid[0])

_Xs = torch.tensor(
    np.stack([x_grid, np.full_like(x_grid, t_split)], axis=1),
    device=device).requires_grad_(True)
_us = net(_Xs)
_uts = torch.autograd.grad(_us, _Xs, torch.ones_like(_us))[0][:, 1]
u0_hand = _us.detach().cpu().numpy().ravel().astype(np.float64)
ut0_hand = _uts.detach().cpu().numpy().ravel().astype(np.float64)


def _integrate_wave(c2, u0, ut0):
    """Method of lines on the INTERIOR nodes, u = 0 at both ends.

    Second-order central u_xx, so this integrator has a discretization
    error of its own -- which is precisely why the TRUE-coefficient run
    below is reported next to it: that control carries the identical
    scheme error, so the difference between the two rows is coefficient
    error and nothing else.
    """
    if len(_t_test) == 0:
        return np.zeros((Nx, 0))
    n_i = Nx - 2
    pad = np.zeros(Nx)

    def rhs(t, y):
        pad[1:-1] = y[:n_i]
        uxx = (pad[2:] - 2.0 * pad[1:-1] + pad[:-2]) / _dx ** 2
        return np.concatenate([y[n_i:], c2 * uxx])

    sol = solve_ivp(rhs, (t_split, float(_t_test[-1])),
                    np.concatenate([u0[1:-1], ut0[1:-1]]),
                    t_eval=_t_test, method="RK45", rtol=1e-10, atol=1e-12)
    if not sol.success:
        return np.full((Nx, len(_t_test)), np.nan)
    out = np.zeros((Nx, len(_t_test)))
    out[1:-1] = sol.y[:n_i]
    return out


U_int = _integrate_wave(c_hat, u0_hand, ut0_hand)
# Control: the SAME handover state integrated with the TRUE c2. It
# separates 'the recovered coefficient is wrong' from 'the state the net
# hands over at t_split is slightly off', which no single number can do.
U_int_true = _integrate_wave(C2_TRUE, u0_hand, ut0_hand)
_U_te = U[:, n_train:].astype(np.float64)


def _rel_int(a_):
    return (float(np.linalg.norm(a_ - _U_te) / np.linalg.norm(_U_te))
            if _U_te.size else float("nan"))


rel_l2_test_int = _rel_int(U_int)
rel_l2_test_int_true = _rel_int(U_int_true)


print(f"\n========== WAVE AUTOSCALE PINN (weight-free loss) ==========")
print(f"mode: coefs={COEF_SOURCE}  stat={STAT}  anchor={ANCHOR}/"
      f"{ANCHOR_WEIGHT}  fd_stride={FD_STRIDE}  data_term={DATA_TERM}  "
      f"select={BEST_KEY}")
print(f"loss = l_phys + l_ic + l_bc + l_stat + l_anch + l_data   "
      f"(no weights; pinn_test_cv.py uses 1/1/100/100)")
print(f"train/test split        = {TRAIN_FRAC:.2f} of t  "
      f"(t_split={t_split:.3f}, n_train={n_train}/{Nt})  "
      f"loss sees train only")
print(f"selected checkpoint     = iter {best.iter}  "
      f"({BEST_KEY}={best.best:.3e})")
print(f"last loss               = {final.item():.3e}")
print(f"training time           = {elapsed:.1f}s")
print(f"rel L2 (best / final)   = {rel_l2:.4e} / {rel_final:.4e}")
print(f"rel L2 train (net)      = {rel_l2_train:.4e}")
print(f"rel L2 TEST  (net)      = {rel_l2_test:.4e}   "
      f"(NOT a result: the net was never trained past t_split)")
print(f"rel L2 TEST  (INTEGRATED recovered eq) = {rel_l2_test_int:.4e}")
print(f"   same handover, TRUE c2              = {rel_l2_test_int_true:.4e}"
      f"   <- the floor set by the t_split state + scheme alone")
print(f"  c2_hat = {c_hat:.6f}   true = {C2_TRUE}   rel err = {net_err:.4e}")
if ANCHOR_ERR is not None:
    print(f"  anchor = {float(THETA_FD):.6f}  rel err {ANCHOR_ERR:.4e}   "
          f"({'IMPROVED' if net_err < ANCHOR_ERR else 'no gain'})")
print(f"  chi(t-path) = {float(chi_t):.6f}   chi(x-path) = {float(chi_x):.6f}"
      f"   het = {float(het_e['score'][0]):.6f}")
print(f"  max_corr = {anchor_scale:.4e}   (sparsity.py anchor)")

out_path = f"wave_pinn_auto{_SUFFIX}.npz"
np.savez(out_path,
         U_data=U, U_pred=U_pred, U_final=U_final,
         x=x_grid, t=t_grid,
         c2_hat=np.float32(c_hat), c2_true=np.float32(C2_TRUE),
         c2_fd=np.float32(float(THETA_FD) if THETA_FD is not None else np.nan),
         anchor_err=np.float32(ANCHOR_ERR if ANCHOR_ERR is not None else np.nan),
         net_err=np.float32(net_err),
         rel_l2=np.float32(rel_l2), rel_l2_final=np.float32(rel_final),
         rel_l2_train=np.float32(rel_l2_train),
         rel_l2_test=np.float32(rel_l2_test),
         rel_l2_test_int=np.float32(rel_l2_test_int),          # THE test metric
         rel_l2_test_int_true=np.float32(rel_l2_test_int_true),
         U_test_int=U_int.astype(np.float32),
         data_term=np.bool_(DATA_TERM), train_frac=np.float32(TRAIN_FRAC),
         n_train=np.int32(n_train), t_split=np.float32(t_split),
         phys_scale=np.float32(PHYS_SCALE),
         phys_var=np.float32(PHYS_VAR),
         phys_div=np.float32(PHYS_DIV),
         phys_norm=np.str_(PHYS_NORM),
         phys_agg=np.str_(PHYS_AGG),
         ic_norm_u=np.float32(IC_NORM_U), ic_norm_ut=np.float32(IC_NORM_UT),
         max_corr=np.float32(anchor_scale),
         chi_t=np.float32(float(chi_t)), chi_x=np.float32(float(chi_x)),
         het_score=het_e["score"], het_se_rel=het_e["se_rel"],
         training_time=np.float32(elapsed),
         best_iter=np.int32(best.iter),
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
         coef_source=np.str_(COEF_SOURCE), stat=np.str_(STAT),
         stat_bound=np.bool_(STAT_BOUND), anchor=np.str_(ANCHOR),
         fd_stride=np.int32(FD_STRIDE),
         anchor_weight=np.str_(ANCHOR_WEIGHT), best_key=np.str_(BEST_KEY))
print(f"saved -> {out_path}")
