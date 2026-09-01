#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Forced-Duffing PINN with the weight-free loss -- the DP recipe moved
across UNCHANGED.

This is the transfer test. `projects/pic/data/dp/pinn_test_autoscale.py`
established a loss with no tuned constants:

    loss = l_phys + l_ic + l_stat + l_anch          (no coefficients)

where every term is divided by a yardstick computed once from the
observed data, and every coefficient the loss reads comes from OLS --
either on the net's own field (`global_ols`) or on the finite-difference
design matrix built from the observations (`THETA_FD`). No truth, and no
pointwise supervision on x.

NOTHING about the recipe is re-tuned here. What changes is only what the
system itself supplies: its own library (`cv_metric.build_features`), its
own PHYS_SCALE (max|x_tt| from FD), its own Var(x)/Var(v). The claim
under test is that this transfers; if a per-system constant were needed,
the loss would not be hyperparameter-free.

The rest of the harness is duffing's own, so the only difference from
`pinn_test_cv.py` is the loss: HIDDEN [64]*5, N_COLL 8000, ADAM 10k +
one LBFGS, SEED 0, N_WIN 30, WIN_FRAC 0.5. One harness detail changed
for correctness, not for performance: collocation times are SORTED,
because chi's cumulative score path is order-sensitive (the OLS itself
is not). pinn_test_cv.py's weights were W_PDE, W_CV, W_IC = 1, 1, 1.

Measured FD anchor quality on duffing.npz (rel-L1 error sum vs truth):
stride 1 = 1.6e-3, 2 = 6.5e-3, 4 = 2.6e-2, 8 = 1.0e-1; with x alone
observed (v also by FD) stride 1 = 8.0e-3. Compare DP's 7.4e-3 / 5.0e-3.

OBSERVATIONS IN THE LOSS (August 2026 sweep). The default is now
COEF_SOURCE='ols' + DATA_TERM=True: the physics residual carries the
net's OWN OLS recovery and an observation term pins the field. That
configuration was previously a COLLAPSE here -- under-determined, with
the trivial state scoring zero -- and supervision is exactly what
removes the degeneracy. Measured on coefficient error, the
pre-registered ranking metric:
    rel-L1 1.956e-3 -> 4.26e-5 (seed 0) / 4.47e-4 (seed 1),
    against a seed band of [1.38e-3, 1.96e-3] for the old default,
    and rel-L2 3.53e-4 -> ~3e-5 on both seeds.
It also beats THETA_FD itself (1.641e-3), so it REPLACES the FD coefficient
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
splits EVERYTHING the loss touches: physics collocation, l_data, the FD
design and the yardsticks all stop at t_split, exactly as dp does. The
tail is genuine extrapolation from the learned equation.
"""

import os
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.integrate import solve_ivp

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cv_metric import (                      # duffing's own harness
    build_features,
    cv_forms,
    error_stats,
    evaluate_derivatives_on_grid,
    first_second_derivs_t,
    make_windows,
)
from pinn_common import (                    # shared PINN model + driver
    Checkpoint,
    MLP,
    assert_train_only,
    as_float as _f,
    grad_share,
    rel_l1 as _rel_l1,
    stat_scores,
    train,
)
from stat_common import (                    # shared, from dp/cv_metric.py
    HardICWrapper,
    chi2_per_term,
    global_ols,
    het_per_window,
    max_corr,
    observation_loss,
)

torch.set_default_dtype(torch.float32)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================ settings
DATA_PATH  = "./duffing.npz"

HIDDEN     = [64] * 5
N_COLL     = 8000
ADAM_ITERS = 10000
ADAM_LR    = 1e-3
LBFGS_MAX  = 20000
SEED       = 0

N_WIN      = 30
WIN_FRAC   = 0.5
EPS = 1e-12

# ============================================================ mode block
# IDENTICAL defaults to the DP script -- that is the point.
# Physics yardstick: which statistic of the FD target normalizes the
# residual. 'var' -> l_phys = mean(r^2)/Var(y) ~ 1 - R^2, the same
# second-moment rule the IC (and BC) already use, so every term is a
# fraction of its own channel's signal. 'max' = max|target|^2, an
# EXTREMUM: one bad FD point sets the scale for the whole term.
# How the physics residual is aggregated over collocation points.
#   'mean'   = mean(r^2) over all points -- uniform, and what wave has
#              been doing all along.
#   'window' = mean over the N_WIN overlapping windows of each window's
#              own mean. This is INHERITED from the CV/het machinery
#              (those statistics ARE sliding-window OLS); nothing in the
#              default loss needs it, and it is NOT neutral: overlapping
#              windows cover points unevenly, so it is a weighted mean
#              with a 32.5x spread that gives t=0 only 0.053x uniform
#              weight -- starving exactly the neighbourhood an IVP
#              propagates from. N_WIN / WIN_FRAC then touch the loss;
#              under 'mean' they are het-diagnostic-only.
PHYS_AGG      = "mean"     # 'mean' | 'window'
PHYS_NORM     = "var"       # 'var' | 'max'
COEF_SOURCE = "ols"   # 'ols' | 'fd' | 'true'
IC_MODE = "none"   # 'scaled' | 'hard' | 'none' (l_data supervises)
STAT          = "none"      # 'none' | 'het' | 'chi'
STAT_BOUND    = True
STAT_FLOAT64  = True
STAT_FLOOR    = False
ANCHOR        = "none"        # 'none' | 'fd'
FD_STRIDE     = 1
ANCHOR_WEIGHT = "none"      # 'none' | 'stat'
BEST_KEY      = "loss"      # truth-free
DATA_TERM = True   # pointwise supervision on the observed x
TRAIN_FRAC    = 0.8         # temporal split, mirroring dp: the loss --
                            # physics, data, BC, the FD design and the
                            # yardsticks -- sees ONLY the first
                            # TRAIN_FRAC. The tail is never shown to
                            # anything, so test error there is genuine
                            # extrapolation from the learned equation.

_SUFFIX = ({"ols": "", "fd": f"_cfd{FD_STRIDE}", "true": "_true"}[COEF_SOURCE]) \
    + ("_hardic" if IC_MODE == "hard" else "") \
    + ("_noicbc" if IC_MODE == "none" else "") \
    + ("" if STAT == "none" else f"_{STAT}") \
    + ("_raw" if (STAT == "chi" and not STAT_BOUND) else "") \
    + ("" if ANCHOR == "none" else f"_afd{FD_STRIDE}") \
    + ("_wstat" if ANCHOR_WEIGHT != "none" else "") \
    + ("" if PHYS_NORM == "var" else "_pmax") \
    + ("_data" if DATA_TERM else "")

_SUFFIX += "_pm" if PHYS_AGG == "mean" else ""

if IC_MODE == "none" and not DATA_TERM:
    raise ValueError("IC_MODE='none' removes the only anchor on the solution "
                     "family unless DATA_TERM supervises the field")

if COEF_SOURCE not in ("fd", "ols", "true"):
    raise ValueError(f"bad COEF_SOURCE {COEF_SOURCE!r}")
if ANCHOR_WEIGHT != "none" and (ANCHOR == "none" or STAT == "none"):
    raise ValueError("ANCHOR_WEIGHT='stat' requires ANCHOR='fd' and a STAT")
if not 0.0 < TRAIN_FRAC <= 1.0:
    raise ValueError(f"bad TRAIN_FRAC {TRAIN_FRAC!r}")


# ============================================================ load data
_data = np.load(DATA_PATH)
t_grid = _data["t"].astype(np.float32)
x_data = _data["x"].astype(np.float32)
v_data = _data["v"].astype(np.float32)
ALPHA_TRUE = float(_data["alpha"])
BETA_TRUE  = float(_data["beta"])
DELTA_TRUE = float(_data["delta"])
GAMMA_TRUE = float(_data["gamma"])
OMEGA      = float(_data["omega"])
X0_TRUE    = float(_data["x0"])
V0_TRUE    = float(_data["v0"])
T_MIN, T_MAX = float(t_grid[0]), float(t_grid[-1])
N_GRID = len(t_grid)

# The split, before anything that reads the data: the FD design, the
# yardsticks, the collocation sampler and the windows must all stop
# at t_split, or the "held-out" tail is not held out.
n_train = int(np.ceil(TRAIN_FRAC * N_GRID))
t_split = float(t_grid[n_train - 1])

THETA_TRUE = torch.tensor([ALPHA_TRUE, BETA_TRUE, DELTA_TRUE, GAMMA_TRUE],
                          device=device)
COEF_NAMES = ("alpha", "beta", "delta", "gamma")

T_obs = torch.tensor(t_grid.reshape(-1, 1), device=device)
x_obs = torch.tensor(x_data.reshape(-1, 1), device=device)
T_IC  = torch.tensor([[T_MIN]], device=device)

t_lo, t_hi = make_windows(T_MIN, t_split, N_WIN, WIN_FRAC)
T_LO = torch.tensor(t_lo, device=device).unsqueeze(1)
T_HI = torch.tensor(t_hi, device=device).unsqueeze(1)


# ============================================================ yardsticks
# ALL yardsticks use the POPULATION variance (ddof=0): we want the
# spread the observed signal actually has, not an estimate of a
# generative parameter. torch defaults to ddof=1 and numpy to ddof=0,
# so the convention is pinned explicitly at every call site rather
# than inherited from whichever library holds the array.

def _fd_field(stride=1):
    """Library inputs from the OBSERVED field alone: acceleration by
    central differences (x and v are both measured here)."""
    sl = slice(0, n_train, stride)
    tt = t_grid[sl].astype(np.float64)
    xx = x_data[sl].astype(np.float64)
    vv = v_data[sl].astype(np.float64)
    aa = np.gradient(vv, tt)
    return (torch.tensor(tt), torch.tensor(xx), torch.tensor(vv),
            torch.tensor(aa))


def _fd_design(stride=1):
    tt, xx, vv, aa = _fd_field(stride)
    return aa, build_features(tt, xx, vv, OMEGA)


# max|target| from the FD field -- the same rule as DP's PHYS_SCALE.
_y_fd, _A_fd = _fd_design()
PHYS_SCALE = float(_y_fd.abs().max()) + EPS
PHYS_VAR = float(_y_fd.var(unbiased=False)) + EPS
PHYS_DIV = (PHYS_SCALE if PHYS_NORM == "max"
            else float(np.sqrt(PHYS_VAR)))

# IC yardsticks: the variance each channel actually explores.
IC_NORM_X = float(np.var(x_data[:n_train])) + EPS
IC_NORM_V = float(np.var(v_data[:n_train])) + EPS
T_SPAN = t_split - T_MIN

# Train tensors for l_data (physics uses the sampler, which is bounded
# by t_split as well).
T_train = T_obs[:n_train]
x_train = x_obs[:n_train]
assert_train_only("l_data grid", T_train[:, 0], t_split)
assert_train_only("FD design", torch.tensor(t_grid[:n_train]), t_split)
assert_train_only("IC point", T_IC[:, 0], t_split)


def _fd_anchor(stride):
    """THETA_FD: coefficients OLS-recovered from the observed field. No
    truth consulted -- the same global_ols the net's own recovery uses."""
    y, A = _fd_design(stride)
    c, _ = global_ols(y, A, ridge=EPS)
    return c.to(device=device, dtype=torch.float32)


THETA_FD = (_fd_anchor(FD_STRIDE)
            if (ANCHOR == "fd" or COEF_SOURCE == "fd") else None)


# ============================================================ network
def physics_and_stat_loss(net, T_coll):
    t_flat = T_coll[:, 0]
    x, dx_dt, d2x_dt2 = first_second_derivs_t(net, T_coll)
    A = build_features(t_flat, x, dx_dt, OMEGA)
    y = d2x_dt2

    in_t = (t_flat.unsqueeze(0) >= T_LO) & (t_flat.unsqueeze(0) < T_HI)
    mask = in_t.to(A.dtype)

    score, c_hat = stat_scores(y, A, mask, STAT, bound=STAT_BOUND,
                               float64=STAT_FLOAT64, floor=STAT_FLOOR,
                               ridge=EPS)
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
        c_phys = THETA_TRUE
    elif COEF_SOURCE == "fd":
        c_phys = THETA_FD
    else:
        c_phys = c_hat.to(y.dtype)
    r = (y - A @ c_phys) / PHYS_DIV
    l_phys = ((r ** 2).mean() if PHYS_AGG == "mean"
              else ((mask @ (r ** 2))
                    / mask.sum(dim=1).clamp(min=1.0)).mean())

    return dict(phys=l_phys, stat=l_stat, anch=l_anch,
                theta_hat=c_hat.detach(), x=x.detach(),
                gram=(A.t() @ A).detach())


def ic_loss(net):
    """x(0), x_t(0) each divided by the variance that channel explores --
    dimensionless, weight 1. IC_MODE='hard' removes the term entirely."""
    if IC_MODE in ("hard", "none"):
        return torch.zeros((), device=device)
    x0, v0, _ = first_second_derivs_t(net, T_IC)
    return ((x0 - X0_TRUE) ** 2).mean() / IC_NORM_X \
        + ((v0 - V0_TRUE) ** 2).mean() / IC_NORM_V


def data_loss(net):
    """Observation term, divided by the SAME variance yardstick l_ic
    already uses -- dimensionless, weight 1, no constant introduced.
    Needs its own forward pass: physics runs on random collocation
    points, not on the observation grid. Zero and gradient-free when
    DATA_TERM is False."""
    if not DATA_TERM:
        return torch.zeros((), device=device)
    return observation_loss(net(T_train), x_train, IC_NORM_X)


def total_loss(ml, l_ic, l_dat):
    """THE loss. No coefficients: their absence is the design."""
    return ml["phys"] + l_ic + ml["stat"] + ml["anch"] + l_dat


# ============================================================ train
torch.manual_seed(SEED); np.random.seed(SEED)
g = torch.Generator(device=device).manual_seed(SEED)

net = MLP(1, 1, HIDDEN).to(device)
if IC_MODE == "hard":
    net = HardICWrapper(net, T_MIN, T_SPAN,
                        torch.tensor([[X0_TRUE]], device=device),
                        torch.tensor([[V0_TRUE]], device=device)).to(device)

def sample_collocation(gen):
    """duffing's own sampler, SORTED: chi's cumulative score path is
    order-sensitive (the OLS is not)."""
    t = torch.rand(N_COLL, generator=gen, device=device) * (t_split - T_MIN) + T_MIN
    return assert_train_only("collocation",
                             torch.sort(t).values.unsqueeze(1), t_split)


ANCHOR_ERR = (_rel_l1(THETA_FD.cpu().numpy(), THETA_TRUE.cpu().numpy())
              if THETA_FD is not None else None)
if ANCHOR_ERR is not None:
    print(f"FD anchor (data-driven, no truth consulted): "
          f"{THETA_FD.cpu().numpy().round(4).tolist()}  "
          f"(rel-L1 err {ANCHOR_ERR:.4e})")
print(f"yardsticks: PHYS_NORM={PHYS_NORM} div={PHYS_DIV:.4g} (max={PHYS_SCALE:.4g} sqrtVar={np.sqrt(PHYS_VAR):.4g})  Var(x)={IC_NORM_X:.4g}  "
      f"Var(v)={IC_NORM_V:.4g}")

hist = {"iter": [], "tot": [], "phys": [], "stat": [], "ic": [], "anch": [],
        "dat": [], "gphys": [], "gdat": []}


def _loss_fn(T_coll):
    """One Adam/LBFGS step's worth of loss, handed to pinn_common.train."""
    ml = physics_and_stat_loss(net, T_coll)
    l_ic = ic_loss(net)
    l_dat = data_loss(net)
    loss = total_loss(ml, l_ic, l_dat)
    terms = {"phys": ml["phys"], "stat": ml["stat"], "anch": ml["anch"],
             "ic": l_ic, "dat": l_dat}
    return loss, terms, ml


def _log(it, cur, terms, ml, gsh):
    hist["iter"].append(it); hist["tot"].append(cur)
    for k in ("phys", "stat", "ic", "anch", "dat"):
        hist[k].append(_f(terms[k]))
    hist["gphys"].append(gsh["phys"]); hist["gdat"].append(gsh["dat"])
    print(f"[adam {it:5d}] tot={cur:.3e}  phys={_f(terms['phys']):.3e}  "
          f"stat={_f(terms['stat']):.3e}  anch={_f(terms['anch']):.3e}  "
          f"ic={_f(terms['ic']):.3e}  dat={_f(terms['dat']):.3e}  | "
          f"|g| phys={gsh['phys']:.2e} dat={gsh['dat']:.2e} "
          f"(phys share {grad_share(gsh):.0%})  | theta_hat="
          f"{ml['theta_hat'].cpu().numpy().round(4).tolist()}  "
          f"cond={float(torch.linalg.cond(ml['gram'].double())):.1e}",
          flush=True)


best = Checkpoint(net, key=BEST_KEY,
                  err_fn=lambda ml: _rel_l1(ml["theta_hat"].cpu().numpy(),
                                            THETA_TRUE.cpu().numpy()))
final, elapsed = train(
    net, _loss_fn, adam_iters=ADAM_ITERS, adam_lr=ADAM_LR,
    lbfgs_max=LBFGS_MAX, checkpoint=best, sampler=sample_collocation, gen=g,
    log_every=500, log_fn=_log, grad_terms=("phys", "ic", "dat"))

with torch.no_grad():
    x_pred_final = net(T_obs).cpu().numpy().ravel()
rel_final = float(np.linalg.norm(x_pred_final - x_data)
                  / np.linalg.norm(x_data))

best.restore()

# ============================================================ evaluate
with torch.no_grad():
    x_pred = net(T_obs).cpu().numpy().ravel()
stats = error_stats(x_pred, x_data)
rel_l2 = float(np.linalg.norm(x_pred - x_data) / np.linalg.norm(x_data))
# The tail the LOSS never saw -- no data, no physics, no BC, no FD.
# Genuine extrapolation from the learned equation, the same thing dp
# reports as its test columns.
_sl_tr, _sl_te = slice(0, n_train), slice(n_train, N_GRID)
_rl2 = lambda s: float(np.linalg.norm(x_pred[s] - x_data[s])
                       / np.linalg.norm(x_data[s]))
rel_l2_train = _rl2(_sl_tr)
rel_l2_test = _rl2(_sl_te) if n_train < N_GRID else float("nan")

x_g, v_g, a_g = evaluate_derivatives_on_grid(net, T_obs)
# OLS on the TRAIN WINDOW ONLY. Over the full grid this reads the
# net's untrained tail and the recovery is destroyed -- the same run
# scores 3.99e-5 here and 5.3e-1 full-grid. A statistic the loss
# never trained on must not be measured where the model was never
# fitted.
_tr = slice(0, n_train)
A_eval = build_features(T_train[:, 0], torch.tensor(x_g[_tr], device=device),
                        torch.tensor(v_g[_tr], device=device), OMEGA)
y_eval = torch.tensor(a_g[_tr], device=device)
theta_hat, res_eval = global_ols(y_eval, A_eval, ridge=EPS)
theta_hat = theta_hat.cpu().numpy()

in_t = (T_train[:, 0].unsqueeze(0) >= T_LO) & (T_train[:, 0].unsqueeze(0) < T_HI)
mask_eval = in_t.to(A_eval.dtype)
with torch.no_grad():
    het_out = {k: v.cpu().numpy() for k, v in
               het_per_window(y_eval, A_eval, mask_eval, ridge=EPS).items()}
    chi_out = {k: v.cpu().numpy() for k, v in
               chi2_per_term(y_eval.double(), A_eval.double(), ridge=EPS,
                             use_floor=STAT_FLOOR).items()}
    anchor_scale = float(max_corr(y_eval, A_eval))

net_err = _rel_l1(theta_hat, THETA_TRUE.cpu().numpy())


# ---------------------------------------------- test by INTEGRATION
# Evaluating the NET past t_split measures nothing: it was trained
# only on [T_MIN, t_split], so its output there is unconstrained by
# construction. The extrapolation question is whether the RECOVERED
# EQUATION reproduces the held-out tail, so integrate it forward from
# the state the net holds AT t_split (the last train point).
_t_test = t_grid[n_train:].astype(np.float64)


def _integrate(theta, x0, v0):
    """duffing.py's own rhs, with theta in place of the true coefs."""
    a, b, d, gm = (float(v) for v in theta)

    def rhs(t, y):
        x, v = y
        return [v, -a * x - b * x ** 3 - d * v + gm * np.cos(OMEGA * t)]

    if len(_t_test) == 0:
        return np.zeros(0)
    sol = solve_ivp(rhs, (t_split, float(_t_test[-1])), [x0, v0],
                    t_eval=_t_test, method='RK45', rtol=1e-10, atol=1e-12)
    return sol.y[0] if sol.success else np.full(len(_t_test), np.nan)


_x0, _v0 = float(x_g[n_train - 1]), float(v_g[n_train - 1])
x_int = _integrate(theta_hat, _x0, _v0)
# Control: the SAME initial state integrated with the TRUE coefficients.
# Separates 'the coefficients are wrong' from 'the handover state at
# t_split is slightly off', which no single number can do.
x_int_true = _integrate(THETA_TRUE.cpu().numpy(), _x0, _v0)


def _rel(a_, b_):
    return (float(np.linalg.norm(a_ - b_) / np.linalg.norm(b_))
            if len(b_) else float('nan'))


_x_te = x_data[n_train:]
rel_l2_test_int = _rel(x_int, _x_te)
rel_l2_test_int_true = _rel(x_int_true, _x_te)


print(f"\n========== DUFFING AUTOSCALE PINN (weight-free loss) ==========")
print(f"mode: coefs={COEF_SOURCE}  ic={IC_MODE}  stat={STAT}"
      f"{'' if STAT != 'chi' else ('/bounded' if STAT_BOUND else '/RAW')}  "
      f"anchor={ANCHOR}/{ANCHOR_WEIGHT}  fd_stride={FD_STRIDE}  "
      f"data_term={DATA_TERM}  select={BEST_KEY}")
print(f"loss = l_phys + l_ic + l_stat + l_anch + l_data   (no weights)")
print(f"train/test split        = {TRAIN_FRAC:.2f}  "
      f"(t_split={t_split:.3f}, n_train={n_train}/{N_GRID})  "
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
print(f"   same handover, TRUE coefs           = {rel_l2_test_int_true:.4e}"
      f"   <- the floor set by the t_split state alone")
print(f"  max|err| = {stats['max_abs']:.4e}   "
      f"rmse = {stats['rmse']:.4e}")
print(f"  theta_hat = {theta_hat.round(4).tolist()}   "
      f"({', '.join(COEF_NAMES)})")
print(f"  true      = {THETA_TRUE.cpu().numpy().round(4).tolist()}")
if ANCHOR_ERR is not None:
    print(f"  anchor    = {THETA_FD.cpu().numpy().round(4).tolist()}   "
          f"rel-L1 err: net {net_err:.4e}  vs anchor {ANCHOR_ERR:.4e}   "
          f"({'IMPROVED' if net_err < ANCHOR_ERR else 'no gain'})")
else:
    print(f"  rel-L1 err (net vs truth) = {net_err:.4e}")
print(f"  chi = {chi_out['score'].round(6).tolist()}   "
      f"het = {het_out['score'].round(6).tolist()}")
print(f"  max_corr = {anchor_scale:.4e}   (sparsity.py anchor)")

cvf = cv_forms(np.tile(theta_hat, (N_WIN, 1)),
               THETA_TRUE.cpu().numpy())
out_path = f"duffing_pinn_auto{_SUFFIX}.npz"
np.savez(out_path,
         t=t_grid, x_data=x_data, x_pred=x_pred, x_pred_final=x_pred_final,
         v_pred=v_g, a_pred=a_g,
         theta_hat=theta_hat, theta_true=THETA_TRUE.cpu().numpy(),
         theta_fd=(THETA_FD.cpu().numpy() if THETA_FD is not None
                   else np.zeros(0, dtype=np.float32)),
         anchor_err=np.float32(ANCHOR_ERR if ANCHOR_ERR is not None else np.nan),
         net_err=np.float32(net_err),
         rel_l2=np.float32(rel_l2), rel_l2_final=np.float32(rel_final),
         rel_l2_train=np.float32(rel_l2_train),
         rel_l2_test=np.float32(rel_l2_test),          # net, diagnostic only
         rel_l2_test_int=np.float32(rel_l2_test_int),  # THE test metric
         rel_l2_test_int_true=np.float32(rel_l2_test_int_true),
         x_test_int=x_int.astype(np.float32),
         data_term=np.bool_(DATA_TERM), train_frac=np.float32(TRAIN_FRAC),
         n_train=np.int32(n_train), t_split=np.float32(t_split),
         max_abs_err=np.float32(stats["max_abs"]),
         rmse=np.float32(stats["rmse"]),
         phys_scale=np.float32(PHYS_SCALE),
         phys_var=np.float32(PHYS_VAR),
         phys_div=np.float32(PHYS_DIV),
         phys_norm=np.str_(PHYS_NORM),
         phys_agg=np.str_(PHYS_AGG),
         ic_norm_x=np.float32(IC_NORM_X), ic_norm_v=np.float32(IC_NORM_V),
         max_corr=np.float32(anchor_scale),
         chi_score=chi_out["score"], het_score=het_out["score"],
         het_se_rel=het_out["se_rel"],
         anchored_mse_sum=np.float32(cvf["anchored_mse_sum"]),
         training_time=np.float32(elapsed),
         best_iter=np.int32(best.iter), best_err=np.float32(best.err),
         loss_history_dat=np.array(hist["dat"], dtype=np.float32),
         grad_phys=np.array(hist["gphys"], dtype=np.float32),
         grad_data=np.array(hist["gdat"], dtype=np.float32),
         loss_history_iter=np.array(hist["iter"], dtype=np.int32),
         loss_history_total=np.array(hist["tot"], dtype=np.float32),
         loss_history_phys=np.array(hist["phys"], dtype=np.float32),
         loss_history_stat=np.array(hist["stat"], dtype=np.float32),
         loss_history_ic=np.array(hist["ic"], dtype=np.float32),
         loss_history_anch=np.array(hist["anch"], dtype=np.float32),
         coef_source=np.str_(COEF_SOURCE), ic_mode=np.str_(IC_MODE),
         stat=np.str_(STAT), stat_bound=np.bool_(STAT_BOUND),
         anchor=np.str_(ANCHOR), fd_stride=np.int32(FD_STRIDE),
         anchor_weight=np.str_(ANCHOR_WEIGHT), best_key=np.str_(BEST_KEY))
print(f"saved -> {out_path}")
