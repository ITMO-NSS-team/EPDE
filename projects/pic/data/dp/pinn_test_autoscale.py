#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Double-pendulum PINN with a loss that has NO TUNED CONSTANTS.

Every previous variant in this directory sums weighted terms
(``W_PHYS * l_phys + W_CV * l_cv + W_MCV * l_mcv + W_IC * l_ic``), and
the sweep showed the CONCLUSIONS move with those weights: W_IC 1 -> 100
turned an unidentifiable quasi-static drift into a recoverable one, and
W_CV 0 -> 1 turned coefficient error 0.000 into 2.542 with gravity
deleted. Sibling scripts do not even agree on the values
(pinn_test_cv.py 100/100/0/100 vs pinn_test_chi.py 1/1/0/100).

The construction here is the one ``epde.operators.common.sparsity``
already uses in production. ``PhysicsInformedLasso.fit`` never tunes an
alpha::

    active_thresholds = active_cv * max_corr

-- a DIMENSIONLESS data-driven score (each term's own raw Nyblom-Hansen
constancy, computed on coefficients recovered by OLS *from the data*)
times a DATA-DRIVEN scale anchor (``max|X^T W y|``). Transplanted:

  * every loss term is divided by a yardstick computed ONCE from the
    observed data, so it is dimensionless,
  * every coefficient the loss reads comes from ``global_ols`` on the
    net's own field (``COEF_SOURCE = 'ols'``) -- never truth, never a
    trainable parameter,
  * the statistics are already scale-free; the unbounded one is mapped
    through ``bounded`` into [0, 1) so its RANGE needs no weight either,
  * the loss line is therefore literally

        loss = l_phys + l_ic + l_stat + l_data

    with no coefficients in front. Their absence is the design: there is
    nothing here to tune, and the same construction is meant to move to
    duffing / wave untouched.

Terms and their yardsticks::

    l_phys  mean_w mean_{k in w} ((y_eq - A_eq @ c_eq) / PHYS_SCALE_eq)^2
            PHYS_SCALE = max|target| from FD derivatives of the OBSERVED
            data, computed once (stationary across iterations).
    l_ic    ((theta(t0) - theta_obs(t0))^2 / Var_train(theta)
             + (omega(t0) - omega_obs(t0))^2 / Var_train(omega))
            IC_MODE='hard' deletes this term outright (HardICWrapper).
    l_stat  sum_eq sum_j  bounded(chi_j)  or  het_j     -- both in [0, 1)
    l_anch  sum_eq sum_j ((theta_hat_j - theta_FD_j) / theta_FD_j)^2
            theta_FD = OLS on the FD design matrix built from the
            OBSERVED field -- data-driven, no oracle. Relative, so
            dimensionless. ANCHOR_WEIGHT='stat' scales each term's pull
            by that term's own instability score, which is literally
            sparsity.py's ``active_cv * max_corr``.
    l_data  mean_t (theta_model - theta_obs)^2 / Var_train(theta)
            (pointwise supervision on raw u -- reference cell only)

WELL-POSEDNESS (measured, not assumed -- see the identifiability probe).
With COEF_SOURCE='ols' and DATA_TERM=False the loss contains no truth
and no observations beyond the IC, and it is then UNDER-DETERMINED: for
any coefficient triple c, the trajectory integrated from the true
initial state gives OLS recovery exactly c and residual ~0, so every
member of the family scores the same. Worse, the force-free collapse
(theta = theta0 + omega0*t) scores EXACTLY zero on physics, chi and het
-- strictly better than truth. Three configurations restore
well-posedness, and each is a separate cell here, never a blend:

    DATA_TERM=True                pointwise supervision on raw u.
                                  THE DEFAULT since the August 2026
                                  observation sweep, together with
                                  COEF_SOURCE='ols'. Measured, coefficient
                                  error (the pre-registered ranking
                                  metric -- test trajectory swings 3x on
                                  seed alone and cannot rank):
                                      dp       1.85e-3 -> 4.78e-4  (3.9x)
                                      duffing  1.96e-3 -> 4.3e-5..4.5e-4
                                      wave     3.49e-2 -> 8.26e-3  (4.2x)
                                  against a dp baseline band of
                                  [1.267e-2, 1.287e-2] on best_err, so
                                  the gain is outside seed noise on all
                                  three. It also beats THETA_FD ITSELF
                                  everywhere (dp 7.37e-3, duffing
                                  1.64e-3, wave 3.49e-2), which is why it
                                  REPLACES the FD scaffolding rather than
                                  supplementing it: the observation term
                                  removes the degeneracy, and consistency
                                  ("the field satisfies SOME constant-
                                  coefficient member") then pins the
                                  unique one.
    ANCHOR='fd'                   the coefficients recovered from the
                                  OBSERVED field by finite differences
                                  identify the member. Equation-level
                                  information only -- raw u is never
                                  fitted pointwise. No truth, no tuned
                                  constant. (Or COEF_SOURCE='fd', which
                                  puts the same THETA_FD in the residual
                                  instead of in a pull.) Was the default
                                  until the sweep above.
    COEF_SOURCE='true'            physics knows the coefficients
                                  (oracle probe; unique solution given IC)

WHAT THE DEFAULT COSTS, stated because it is a real weakening. With
pointwise supervision the net fits u, so FD-on-net approaches
FD-on-data and "coefficients from equation structure alone" becomes
"coefficients from a smoothed fit". The numbers above show it is not
ONLY that -- every system beats its own FD estimate -- but the earlier,
stronger claim no longer describes the shipped default. This reverses
the July 2026 removal of the u data term, so artifacts from before and
after are not comparable.

AND THE NEGATIVE. On dp the held-out 20% (which sees neither data nor
physics) got WORSE, not better: test rel-L2 2.33/2.54 against a
three-run baseline band of [0.61, 1.86]. Both supervised cells sit above
all three baselines. dp is chaotic and rel-L2 >= 1 already means the
trajectory has decorrelated, so this is a comparison between "useless"
and "useless", on a metric whose own seed spread is 3x -- but it is
consistent across P0 and P4 and is NOT explained away here.

The anchor cell is only interesting when the anchor is imperfect: at
FD_STRIDE=1 the FD recovery is already within 0.5% of truth, so
reproducing it proves little. At stride 8 it carries ~32% error, and the
question becomes whether physics + consistency pull the net's recovery
CLOSER to truth than the estimate it was handed. Both errors are printed
side by side, per equation, with an explicit IMPROVED / no gain verdict.

Run cells by editing the mode block; artifacts are suffixed per mode.
"""

import sys
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
import torch

# pinn_common lives one level up, beside stat_common -- same reason
# duffing and wave add it: every system directory has its own
# cv_metric.py, so the shared pieces cannot live in one of them.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pinn_common import (                    # shared PINN model + driver
    Checkpoint,
    MLP,
    assert_train_only,
    grad_share,
    stat_scores,
    train,
)
from cv_metric import (
    EQUATIONS,
    EQ_NAMES,
    HardICWrapper,
    LearningGifRecorder,
    chi2_per_term,
    compute_equation_thetas,
    evaluate_on_grid,
    global_ols,
    grid_equation_residuals,
    het_per_window,
    make_windows,
    max_corr,
    observation_loss,
    network_derivatives_dp,
    set_gravity,
)

torch.set_default_dtype(torch.float32)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================ settings
DATA_PATH  = "./dp.npz"

# Architecture / optimizer -- IDENTICAL to pinn_test_cv.py and
# pinn_test_chi.py so the loss is the only thing that changed. These are
# not loss weights; they stay frozen across systems too.
HIDDEN     = [64] * 5
ADAM_ITERS = 20000
ADAM_LR    = 1e-3
LBFGS_MAX  = 20000
SEED       = 0

N_WIN      = 30
WIN_FRAC   = 0.5
TRAIN_FRAC = 0.8       # temporal split: train on first TRAIN_FRAC of trajectory
EPS = 1e-12

MAKE_GIF = False       # off by default: these cells run as a sweep


# ============================================================ mode block
# Each entry selects a MECHANISM. None of them is a magnitude, and none
# of them multiplies a loss term.

# Where the physics residual's coefficients come from.
#   'fd'   = THETA_FD, the coefficients recovered from the OBSERVED
#            field by finite differences. FIXED during training, so the
#            residual is a real force on the trajectory -- the
#            data-driven twin of 'true', and the deployable default.
#   'ols'  = global_ols on the net's own derivatives -- data-driven, the
#            same solve the statistic uses (never two thetas in one
#            loss). ON ITS OWN this term is vacuous: it sits at ~5e-6
#            because the OLS residual is what OLS already minimizes, so
#            nothing shapes the trajectory and the loss collapses (cell
#            A: loss 8.15e-10, coefficient error 289). PAIRED WITH
#            ANCHOR='fd' it is the deployable configuration -- the
#            anchor pins which member theta_hat may be, and the residual
#            stops being free (measured: it rises to ~1.8e-2 as the
#            anchor bites).
#   'true' = the oracle THETA_TRUE (reference cell only)
# ANCHOR='fd' is the OTHER way to use the same THETA_FD -- a pull on the
# net's own recovery rather than the residual's coefficients. It moves
# theta_hat only by moving the whole trajectory, so it is SLOW: at iter
# 5000 the anchor was still stuck at 1.97 (= two dead coefficients) and
# gravity absent; by iter 15000 it had fallen to 3.7e-4 with |c| = 9.65
# / 9.83. Do not judge an anchor cell before it has run out.
#
# DEFAULT = 'fd', because it is the variant that TRANSFERS. Measured:
#   DP        'ols'+anchor coef err 1.14e-2  |  'fd' 1.596
#   duffing   'ols'+anchor coef err 9.80 (collapsed to x==0, rel-L2
#             1.0016)  |  'fd' 1.88e-3 with rel-L2 1.73e-4
#   wave      both work; 'fd' rel-L2 4.08e-3
# The anchor wins on DP and is CATASTROPHIC on duffing, whose rest start
# (x0=v0=0) makes the trivial state satisfy the IC and the OLS residual
# exactly; the gradient then runs through an OLS whose design matrix has
# collapsed (cond 4e6) -- the degenerate basin duffing/pinn_test_cv.py's
# own docstring documents and detaches to avoid.
COEF_SOURCE = "ols"

# How the initial condition is imposed.
#   'scaled' = a loss term divided by the data's own variance
#   'hard'   = HardICWrapper: theta0 + omega0*dt + (dt/span)^2 * N(t),
#              exact by construction, so the term disappears entirely
#              (the strongest form of "no weight": no term to weight).
#              The earlier hard-IC attempt used a bare dt^2 multiplier,
#              which scales the net output by span^2 (~64) and wrecked
#              the run; span normalizes it to [0, 1] and is data.
IC_MODE = "none"

# Physics yardstick: which statistic of the FD target normalizes the
# residual.
#   'var' = Var(target)  -> l_phys = mean(r^2)/Var(y) ~ 1 - R^2, the same
#           second-moment rule the IC (and BC) already use, so every term
#           is a fraction of its own channel's signal.
#   'max' = max|target|^2, the original. An EXTREMUM: one bad FD point
#           sets the scale for the whole term.
# On DP these differ by ~4x (max^2 = 406 vs Var = 103 for th1), which
# changes the phys:ic balance, so it is a real one-variable comparison.
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
PHYS_AGG = "mean"      # 'mean' | 'window'

PHYS_NORM = "var"

# Truth-free constancy statistic in the loss, at weight 1.
#   'none' | 'het' (already in [0,1)) | 'chi'
STAT = "none"

# Whether chi is squashed through ``bounded`` into [0, 1).
# BOTH forms are dimensionless -- chi is scale-free outright (invariant
# under y, A -> a*y, a*A; pinned by test_dp_autoscale_loss), so it never
# needed a yardstick and neither form carries a tuned constant. The map
# addresses only its RANGE: raw chi reached 259 and 45.8 in these cells,
# so unbounded it can dominate a loss whose other terms are O(1), while
# bounding compresses a wildly inconsistent state and a mildly
# inconsistent one toward the same value. False = raw. het is bounded by
# construction, so this flag does not apply to it.
STAT_BOUND = True

# Data-driven coefficient anchor -- the identification route that uses
# EQUATION-level information without ever fitting raw u pointwise.
#   'none' = no anchor
#   'fd'   = pull the net's OLS recovery toward the coefficients
#            recovered from the OBSERVED field by finite differences.
#            No oracle: THETA_FD is what sparsity.py would fit, on the
#            data it would fit it on. Measured on this dataset it lands
#            within 0.5% of truth at stride 1 (rel-err sum 7.4e-3 /
#            5.0e-3) and degrades to ~32% at stride 8 -- which is the
#            interesting cell, since a coarse anchor leaves room for the
#            physics + consistency terms to IMPROVE on what they were
#            handed rather than merely reproduce it.
# The anchor is relative ((c - a)/a)^2, so it is dimensionless and needs
# no weight, and it identifies WHICH member of the coefficient family
# the trajectory is -- the thing the supervision-free loss provably
# cannot do (identifiability probe: every member scores the same, and
# the force-free collapse scores exactly zero).
ANCHOR = "none"
# Grid stride used to build THETA_FD -- shared by COEF_SOURCE='fd' and
# ANCHOR='fd'. A property of the data you have, not a knob to tune:
# stride 1 = every observed sample. Stride 8 is the deliberately COARSE
# anchor (~32% error) that leaves room to improve on it.
FD_STRIDE = 1
# Per-term anchor weighting -- the literal shape of sparsity.py's
# ``active_thresholds = active_cv * max_corr``: each term's own
# instability score scales its own pull. 'none' = uniform.
# NOTE a perfectly constant term scores 0 and is then NOT anchored at
# all, which under the family degeneracy leaves it free; that is why
# uniform is the primary and this is the variant.
ANCHOR_WEIGHT = "none"      # 'none' | 'stat'

# Observation term. False = the supervision-free setting the July 2026
# scripts adopted; True = pointwise supervision on raw u. Kept for the
# reference cell only -- ANCHOR='fd' is the route that identifies the
# member from equation-level information instead.
DATA_TERM = True

# The statistic's own precision. Not a tuning knob: the float32 exact-fit
# floor was measured as a zero-loss ATTRACTOR (the net drove both
# equations to near-exact fits of a WRONG gravity-free system and the
# floored chi read exactly [0,0,0]).
STAT_FLOAT64 = True
STAT_FLOOR   = False

# Checkpoint selection. 'loss' is TRUTH-FREE and the default: a
# no-tuning claim cannot rest on an oracle-selected state, which is what
# 'global'/'windowed' (relative L1 coefficient error against THETA_TRUE)
# do. Both errors are logged either way, and the FINAL state's metrics
# are always reported next to the selected one -- if the loss's minimum
# is the degenerate state, truth-free selection returns it, and that is
# the finding rather than a bug to paper over.
BEST_KEY = "loss"

_SUFFIX = ({"ols": "", "fd": f"_cfd{FD_STRIDE}", "true": "_true"}[COEF_SOURCE]) \
    + ("_hardic" if IC_MODE == "hard" else "") \
    + ("_noicbc" if IC_MODE == "none" else "") \
    + ("" if STAT == "none" else f"_{STAT}") \
    + ("_raw" if (STAT == "chi" and not STAT_BOUND) else "") \
    + ("" if ANCHOR == "none" else f"_afd{FD_STRIDE}") \
    + ("_wstat" if ANCHOR_WEIGHT != "none" else "") \
    + ("_data" if DATA_TERM else "") \
    + ("" if PHYS_NORM == "var" else "_pmax")

_SUFFIX += "_pm" if PHYS_AGG == "mean" else ""

if IC_MODE == "none" and not DATA_TERM:
    raise ValueError("IC_MODE='none' removes the only anchor on the solution "
                     "family unless DATA_TERM supervises the field")

if COEF_SOURCE not in ("fd", "ols", "true"):
    raise ValueError("COEF_SOURCE must be 'fd', 'ols' or 'true', got "
                     f"{COEF_SOURCE!r}")
if IC_MODE not in ("scaled", "hard", "none"):
    raise ValueError(f"IC_MODE must be 'scaled', 'hard' or 'none', got {IC_MODE!r}")
if STAT not in ("none", "het", "chi"):
    raise ValueError(f"STAT must be 'none', 'het' or 'chi', got {STAT!r}")
if ANCHOR not in ("none", "fd"):
    raise ValueError(f"ANCHOR must be 'none' or 'fd', got {ANCHOR!r}")
if ANCHOR_WEIGHT != "none" and (ANCHOR == "none" or STAT == "none"):
    # The weight IS the statistic; without both there is nothing to
    # scale and the config would silently degrade to a uniform anchor.
    raise ValueError("ANCHOR_WEIGHT='stat' requires ANCHOR='fd' and a "
                     f"STAT (got ANCHOR={ANCHOR!r}, STAT={STAT!r})")


# ============================================================ load data
_data = np.load(DATA_PATH)
t_grid = _data["t"].astype(np.float32)
th1_data = _data["theta1"].astype(np.float32)
th2_data = _data["theta2"].astype(np.float32)
w1_data  = _data["omega1"].astype(np.float32)
w2_data  = _data["omega2"].astype(np.float32)
G_TRUE   = float(_data["g"])

set_gravity(G_TRUE)

T_MIN, T_MAX = float(t_grid[0]), float(t_grid[-1])
N_GRID = len(t_grid)

THETA_TRUE = {
    eq["name"]: torch.tensor(eq["theta_true"], device=device, dtype=torch.float32)
    for eq in EQUATIONS
}

T_obs  = torch.tensor(t_grid.reshape(-1, 1), device=device)
th_obs = torch.tensor(np.stack([th1_data, th2_data], axis=1), device=device)

T_IC  = T_obs[:1]
TH_IC = th_obs[:1]
OM_IC = torch.tensor([[w1_data[0], w2_data[0]]], device=device)

n_train = int(np.ceil(TRAIN_FRAC * N_GRID))
t_split = float(t_grid[n_train - 1])
T_train  = T_obs[:n_train]
th_train = th_obs[:n_train]
assert_train_only("physics+l_data grid", T_train[:, 0], t_split)
assert_train_only("FD design", torch.tensor(t_grid[:n_train]), t_split)
T_test   = T_obs[n_train:]
th_test  = th_obs[n_train:]


# ============================================================ yardsticks
# ALL yardsticks use the POPULATION variance (ddof=0): we want the
# spread the observed signal actually has, not an estimate of a
# generative parameter. torch defaults to ddof=1 and numpy to ddof=0,
# so the convention is pinned explicitly at every call site rather
# than inherited from whichever library holds the array.

# Every constant below is computed ONCE from the observed train slice.
# They are the only reason unit weights are meaningful, and they are the
# transplanted half of sparsity.py's `active_cv * max_corr`: a
# dimensionless score needs a data-driven scale to live on.
def _fd_field(stride=1):
    """The library's inputs from the OBSERVED field alone: angular
    accelerations by central differences. This is the only place raw
    observations enter the yardsticks and the anchor, and it is
    equation-level information -- the same finite-difference design
    matrix EPDE's own operators are fitted on."""
    sl = slice(0, n_train, stride)
    tt = t_grid[sl].astype(np.float64)
    d_np = dict(
        theta1=th1_data[sl].astype(np.float64),
        theta2=th2_data[sl].astype(np.float64),
        omega1=w1_data[sl].astype(np.float64),
        omega2=w2_data[sl].astype(np.float64),
    )
    d_np["alpha1"] = np.gradient(d_np["omega1"], tt)
    d_np["alpha2"] = np.gradient(d_np["omega2"], tt)
    return {k: torch.tensor(v) for k, v in d_np.items()}


def _fd_target_scales():
    """PHYS_SCALE: max|target| from the FD field, fed through each
    equation's own target_and_features -- nothing about the target's
    magnitude is hardcoded, so the recipe moves to any other system
    unchanged."""
    d = _fd_field()
    return {eq["name"]: float(eq["target_and_features"](d)[0].abs().max()) + EPS
            for eq in EQUATIONS}


def _fd_anchor(stride):
    """THETA_FD: coefficients OLS-recovered from the observed field.

    No truth is consulted -- this is exactly what
    ``sparsity.PhysicsInformedLasso`` fits, on the data it fits it on,
    through the SAME ``global_ols`` the net's own recovery uses."""
    d = _fd_field(stride)
    out = {}
    for eq in EQUATIONS:
        y, A = eq["target_and_features"](d)
        c, _ = global_ols(y, A, ridge=EPS)
        out[eq["name"]] = c.to(device=device, dtype=torch.float32)
    return out


def _fd_target_vars():
    """Var(target) from the same FD field -- the SECOND-MOMENT yardstick,
    the one the IC and the data term already use.

    With it, l_phys = mean(r^2)/Var(y) is the fraction of the target's
    own variance left unexplained (essentially 1 - R^2), so every term in
    the loss becomes a fraction of its channel's signal and they are on
    one interpretable scale. max|target| is an EXTREMUM: a single bad FD
    point sets the scale for the whole term, and it is not the statistic
    any other term uses."""
    d = _fd_field()
    return {eq["name"]: float(eq["target_and_features"](d)[0]
                          .var(unbiased=False)) + EPS
            for eq in EQUATIONS}


PHYS_SCALE = _fd_target_scales()               # {eq_name: max|target|}
PHYS_VAR = _fd_target_vars()                   # {eq_name: Var(target)}
# The divisor actually used (target units, squared by the residual).
PHYS_DIV = ({nm: PHYS_SCALE[nm] for nm in EQ_NAMES} if PHYS_NORM == "max"
            else {nm: float(np.sqrt(PHYS_VAR[nm])) for nm in EQ_NAMES})
THETA_FD = (_fd_anchor(FD_STRIDE)
            if (ANCHOR == "fd" or COEF_SOURCE == "fd") else None)

# IC / data yardsticks: the observed variance of each channel. Dividing
# by these is what replaces the hand-set W_IC = 100 -- the IC term stops
# being measured in rad^2 and starts being measured in fractions of the
# signal the system actually explores.
IC_NORM_TH = th_train.var(dim=0, unbiased=False) + EPS                      # (2,)
_om_train = torch.tensor(np.stack([w1_data[:n_train], w2_data[:n_train]],
                                  axis=1), device=device)
IC_NORM_OM = _om_train.var(dim=0, unbiased=False) + EPS                     # (2,)
T_SPAN = float(t_grid[n_train - 1] - t_grid[0])

# Windows over the TRAIN domain: they are masked against T_train, so
# spanning to T_MAX left the late windows EMPTY. No test data entered
# (no leak), but empty windows are a degenerate input to het/mcv.
t_lo, t_hi = make_windows(T_MIN, t_split, N_WIN, WIN_FRAC)
PERIOD = None
T_LO = torch.tensor(t_lo, device=device).unsqueeze(1)
T_HI = torch.tensor(t_hi, device=device).unsqueeze(1)


# ============================================================ windowed-moment CV
# DIAGNOSTIC ONLY -- never in the loss (it would inject observations
# through the back door in the DATA_TERM=False cells). It is the
# watchdog for the collapse mode: a loss built only from the net's own
# field cannot see a parked or drifting trajectory, and the probe
# measured the force-free collapse scoring exactly zero everywhere.
_tm = T_train[:, 0]
_mcv_in = (_tm.unsqueeze(0) >= T_LO) & (_tm.unsqueeze(0) < T_HI)
MCV_MASK = _mcv_in.to(torch.float32)                   # (N_WIN, n_train)
MCV_VALID = MCV_MASK.sum(dim=1) >= 8


def _win_moments(vals):
    cnt = MCV_MASK.sum(dim=1, keepdim=True).clamp(min=1.0)
    mu = (MCV_MASK @ vals) / cnt
    var = (MCV_MASK @ vals ** 2) / cnt - mu ** 2
    return mu, (var.clamp(min=0.0) + 1e-12).sqrt()


MCV_MU_D, MCV_SD_D = _win_moments(th_train)
MCV_NORM = th_train.var(dim=0, unbiased=False) + EPS


def moment_cv_loss(th_model):
    mu_m, sd_m = _win_moments(th_model)
    per = ((mu_m - MCV_MU_D) ** 2 + (sd_m - MCV_SD_D) ** 2) / MCV_NORM.unsqueeze(0)
    return per[MCV_VALID].mean()


# ============================================================ network
def physics_and_stat_loss(net, T_coll):
    """One autograd pass -> every term of the loss.

    The physics residual's coefficients are the net's OWN OLS recovery
    (COEF_SOURCE='ols'), which makes this a CONSISTENCY objective rather
    than a fit: it asks whether the field satisfies SOME constant-
    coefficient member of the library, not whether it satisfies the true
    one. theta_hat stays attached -- by the envelope theorem the path
    through the argmin contributes nothing at the optimum, so this is a
    convention, not a bias.
    """
    results, derivs, mask = compute_equation_thetas(
        net, T_coll, T_LO, T_HI, ridge=EPS, period=PERIOD)
    th_model = torch.stack([derivs["theta1"], derivs["theta2"]], dim=1)
    cnt = mask.sum(dim=1).clamp(min=1.0)

    l_phys = 0.0
    l_stat = 0.0
    l_anch = 0.0
    per_eq_residual = {}
    theta_means = {}
    theta_hats = {}
    conds = {}
    for eq in EQUATIONS:
        nm = eq["name"]
        _theta_pw, theta_mean = results[nm]
        y, A = eq["target_and_features"](derivs)

        score, c_hat = stat_scores(y, A, mask, STAT, bound=STAT_BOUND,
                                   float64=STAT_FLOAT64, floor=STAT_FLOOR,
                                   ridge=EPS)
        if score is not None:
            l_stat = l_stat + score.sum().to(th_model.dtype)

        if ANCHOR == "fd":
            # Gated on ANCHOR, NOT on THETA_FD: COEF_SOURCE='fd' also
            # builds THETA_FD, and keying off it would silently apply
            # the anchor in a cell meant to isolate the residual.
            # Relative pull toward the DATA's own recovery -> already
            # dimensionless, so weight 1. Weighted per term by that
            # term's own instability score under ANCHOR_WEIGHT='stat',
            # which is sparsity.py's score x scale shape.
            a = THETA_FD[nm].to(c_hat.dtype)
            per = ((c_hat - a) / a) ** 2
            if ANCHOR_WEIGHT != "none":
                per = per * score.detach().to(per.dtype)
            l_anch = l_anch + per.sum().to(th_model.dtype)

        if COEF_SOURCE == "true":
            c_phys = THETA_TRUE[nm]
        elif COEF_SOURCE == "fd":
            c_phys = THETA_FD[nm]           # fixed -> a real force
        else:
            c_phys = c_hat.to(th_model.dtype)   # vacuous, see the warning
        r = (y - A @ c_phys) / PHYS_DIV[nm]
        eq_mse = ((r ** 2).mean() if PHYS_AGG == "mean"
                  else ((mask @ (r ** 2)) / cnt).mean())
        l_phys = l_phys + eq_mse

        per_eq_residual[nm] = eq_mse
        theta_means[nm] = theta_mean
        theta_hats[nm] = c_hat.detach()
        # Kept as the Gram, not as cond(): the SVD is only worth paying
        # for on print iterations.
        conds[nm] = (A.t() @ A).detach()

    return dict(phys=l_phys, stat=l_stat, anch=l_anch, per_eq=per_eq_residual,
                theta_means=theta_means, theta_hats=theta_hats,
                grams=conds, th_model=th_model)


def ic_loss(net):
    """Initial condition, each channel divided by the variance the
    observed signal actually explores -- dimensionless, weight 1. Under
    IC_MODE='hard' the wrapper satisfies it exactly and this returns a
    constant zero (kept callable so the loss line never branches)."""
    if IC_MODE in ("hard", "none"):
        return torch.zeros((), device=device)
    d0 = network_derivatives_dp(net, T_IC)
    th0 = torch.stack([d0["theta1"], d0["theta2"]], dim=1)     # (1, 2)
    om0 = torch.stack([d0["omega1"], d0["omega2"]], dim=1)     # (1, 2)
    return (((th0 - TH_IC) ** 2 / IC_NORM_TH).mean()
            + ((om0 - OM_IC) ** 2 / IC_NORM_OM).mean())


def data_loss(th_model):
    """Observation term, normalized by the same variance yardstick.
    Zero (and gradient-free) when DATA_TERM is False."""
    if not DATA_TERM:
        return torch.zeros((), device=device)
    return observation_loss(th_model, th_train, IC_NORM_TH)


def total_loss(ml, l_ic):
    """THE loss. No coefficients: their absence is the design."""
    return (ml["phys"] + l_ic + ml["stat"] + ml["anch"]
            + data_loss(ml["th_model"]))


# ============================================================ train
torch.manual_seed(SEED); np.random.seed(SEED)

net = MLP(1, 2, HIDDEN).to(device)
if IC_MODE == "hard":
    net = HardICWrapper(net, T_MIN, T_SPAN, TH_IC, OM_IC).to(device)


gif_rec = None
if MAKE_GIF:
    PLOTS_DIR = Path(__file__).resolve().parent / "plots"
    PLOTS_DIR.mkdir(exist_ok=True)
    gif_rec = LearningGifRecorder(
        t_grid, th1_data, th2_data, t_split,
        label=f"autoscale{_SUFFIX}",
        out_path=PLOTS_DIR / f"learning_auto{_SUFFIX}.gif",
        total_iters=ADAM_ITERS)


def _gif_predict():
    with torch.no_grad():
        p = net(T_obs).detach().cpu().numpy()
    return p[:, 0], p[:, 1]


def _coef_err_sum(ml, key):
    """Relative L1 coefficient error against truth. DIAGNOSTIC unless
    BEST_KEY selects on it -- reported always so the truth-free
    selection can be compared with the oracle one."""
    d = ml["theta_hats"] if key == "global" else ml["theta_means"]
    return sum(
        ((d[nm].detach().to(THETA_TRUE[nm].dtype) - THETA_TRUE[nm]).abs()
         / THETA_TRUE[nm].abs()).sum().item()
        for nm in EQ_NAMES
    )


best = Checkpoint(net, key=BEST_KEY,
                  err_fn=lambda ml: _coef_err_sum(ml, "global"),
                  select_fn=lambda ml: _coef_err_sum(ml, BEST_KEY))


hist = {"iter": [], "tot": [], "phys": [], "stat": [], "mcv": [], "ic": [],
        "data": [], "anch": [], "gphys": [], "gdat": []}


def _f(x):
    """Scalar value of a loss term that may be a live tensor or a plain
    0.0 (terms are literal zeros when their mode is off)."""
    return float(x.detach()) if torch.is_tensor(x) else float(x)


def _rel_l1(c, ref):
    """Relative L1 coefficient error, the comparison currency here."""
    c = np.asarray(c, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    return float((np.abs(c - ref) / np.abs(ref)).sum())


# What the anchor itself is worth: the net's recovery is only
# interesting if it lands CLOSER to truth than the estimate it was
# handed (that is the whole point of the coarse-anchor cell).
ANCHOR_ERR = ({nm: _rel_l1(THETA_FD[nm].cpu().numpy(),
                           THETA_TRUE[nm].cpu().numpy())
               for nm in EQ_NAMES} if THETA_FD is not None else None)
if ANCHOR_ERR is not None:
    print("FD anchor (data-driven, no truth consulted): "
          + "  ".join(f"{nm}={THETA_FD[nm].cpu().numpy().round(4).tolist()} "
                      f"(rel-L1 err {ANCHOR_ERR[nm]:.4e})" for nm in EQ_NAMES))

def _loss_fn(_batch):
    """dp evaluates physics on the fixed train grid, so there is no
    sampler -- pinn_common.train passes batch=None."""
    ml = physics_and_stat_loss(net, T_train)
    l_ic = ic_loss(net)
    loss = total_loss(ml, l_ic)
    terms = {"phys": ml["phys"], "stat": ml["stat"], "anch": ml["anch"],
             "ic": l_ic, "dat": data_loss(ml["th_model"])}
    return loss, terms, ml


def _log(it, cur, terms, ml, gsh):
    l_mcv = moment_cv_loss(ml["th_model"]).item()
    l_dat = float(terms["dat"].detach())
    hist["gphys"].append(gsh["phys"]); hist["gdat"].append(gsh["dat"])
    hist["iter"].append(it)
    hist["tot"].append(cur)
    hist["phys"].append(ml["phys"].item())
    hist["stat"].append(_f(ml["stat"]))
    hist["mcv"].append(l_mcv)
    hist["ic"].append(terms["ic"].item())
    hist["data"].append(l_dat)
    hist["anch"].append(_f(ml["anch"]))
    phys_msgs = "  ".join(f"{nm}={ml['per_eq'][nm].item():.2e}"
                          for nm in EQ_NAMES)
    # Collapse watchdogs: |theta_hat| and cond(A^T A) are the only
    # way to see a degenerate field from inside a truth-free loss.
    diag = "  ".join(
        f"{nm}: |c|={ml['theta_hats'][nm].abs().sum().item():.2e} "
        f"cond={float(torch.linalg.cond(ml['grams'][nm].double())):.1e}"
        for nm in EQ_NAMES)
    print(f"[adam {it:5d}] tot={cur:.3e}  phys={ml['phys'].item():.3e}  "
          f"stat={_f(ml['stat']):.3e}  anch={_f(ml['anch']):.3e}  "
          f"ic={terms['ic'].item():.3e}  data={l_dat:.3e}  mcv={l_mcv:.3e}  "
          f"|g| phys={gsh['phys']:.2e} dat={gsh['dat']:.2e} "
          f"(phys share {grad_share(gsh):.0%})  "
          f"| {phys_msgs}  | {diag}", flush=True)


final, elapsed = train(
    net, _loss_fn, adam_iters=ADAM_ITERS, adam_lr=ADAM_LR,
    lbfgs_max=LBFGS_MAX, checkpoint=best, sampler=None, gen=None,
    log_every=1000, log_fn=_log, grad_terms=("phys", "ic", "dat"),
    on_iter=(lambda it: gif_rec.maybe_capture(it, _gif_predict))
    if gif_rec is not None else None)


def _predict():
    with torch.no_grad():
        p = net(T_obs).cpu().numpy()
    return p[:, 0], p[:, 1]


def _rel_l2(pred, data, sl):
    p = pred[sl]; d = data[sl]
    return float(np.linalg.norm(p - d) / np.linalg.norm(d))


sl_train = slice(0, n_train)
sl_test  = slice(n_train, None)

# The FINAL state's accuracy, measured before the checkpoint is restored:
# if truth-free selection is picking a worse state than simply stopping,
# that must be visible rather than hidden by the restore.
_f1, _f2 = _predict()
final_rel = (_rel_l2(_f1, th1_data, sl_train), _rel_l2(_f2, th2_data, sl_train),
             _rel_l2(_f1, th1_data, sl_test),  _rel_l2(_f2, th2_data, sl_test))

best.restore()
if gif_rec is not None:
    gif_rec.render()

# ============================================================ evaluate
th1_pred, th2_pred = _predict()
rel_l2_th1_train = _rel_l2(th1_pred, th1_data, sl_train)
rel_l2_th2_train = _rel_l2(th2_pred, th2_data, sl_train)
rel_l2_th1_test  = _rel_l2(th1_pred, th1_data, sl_test)
rel_l2_th2_test  = _rel_l2(th2_pred, th2_data, sl_test)

ols_results, _eval_derivs, _eval_mask = compute_equation_thetas(
    net, T_train, T_LO, T_HI, ridge=EPS, period=PERIOD)
theta_per_window = {nm: ols_results[nm][0].detach().cpu().numpy() for nm in EQ_NAMES}
theta_means_final = {nm: ols_results[nm][1].detach().cpu().numpy() for nm in EQ_NAMES}

het_by_eq = {}
chi_by_eq = {}
anchor_by_eq = {}
theta_hat_final = {}
with torch.no_grad():
    for eq in EQUATIONS:
        nm = eq["name"]
        y_e, A_e = eq["target_and_features"](_eval_derivs)
        het_by_eq[nm] = {k: v.cpu().numpy() for k, v in
                         het_per_window(y_e, A_e, _eval_mask, ridge=EPS).items()}
        out_e = chi2_per_term(y_e.double(), A_e.double(), ridge=EPS,
                              use_floor=STAT_FLOOR)
        chi_by_eq[nm] = {k: v.cpu().numpy() for k, v in out_e.items()}
        theta_hat_final[nm] = out_e["theta"].cpu().numpy()
        # The sparsity.py scale anchor, carried as a diagnostic so the
        # score x anchor product stays comparable with the production rule.
        anchor_by_eq[nm] = float(max_corr(y_e, A_e))

grid_d = evaluate_on_grid(net, T_obs)
res_by_eq = grid_equation_residuals(grid_d, theta_means_final)

coef_err_final = sum(
    float(np.abs(theta_hat_final[nm] - THETA_TRUE[nm].cpu().numpy())
          .sum() / np.abs(THETA_TRUE[nm].cpu().numpy()).sum())
    for nm in EQ_NAMES)

# ---------------------------------------------------- test by INTEGRATION
# rel L2 test above evaluates the NET past t_split, where it was never
# trained -- it measures the network's behaviour off its domain, not the
# recovered equation's. Integrate the recovered pair forward from the
# state the net holds AT t_split instead.
#
# The double pendulum is CHAOTIC, so both rows below grow with the test
# span no matter how good the coefficients are. That is why the
# true-coefficient control is mandatory here and not merely useful: it
# is the Lyapunov floor set by the handover state alone, and the
# recovered row can only be read relative to it.
_t_test = t_grid[n_train:].astype(np.float64)


def _integrate_dp(th_by_eq, y0):
    """The recovered pair is IMPLICIT in the accelerations --

        a1 = c11 a2 cos(dth) + c12 w2^2 sin(dth) + c13 sin(th1)
        a2 = c21 a1 cos(dth) + c22 w1^2 sin(dth) + c23 sin(th2)

    each appears in the other's feature list -- so a step is a 2x2 solve,
    not an explicit right-hand side. It is singular exactly when
    c11 c21 cos^2(dth) = 1; a recovered pair that hits that is reported
    as a failed integration rather than clipped into looking fine.
    """
    c1 = np.asarray(th_by_eq["th1"], dtype=np.float64)
    c2 = np.asarray(th_by_eq["th2"], dtype=np.float64)

    def rhs(t, y):
        th1, th2, w1, w2 = y
        dth = th1 - th2
        cd = np.cos(dth)
        sd = np.sin(dth)
        b1 = c1[1] * w2 * w2 * sd + c1[2] * np.sin(th1)
        b2 = c2[1] * w1 * w1 * sd + c2[2] * np.sin(th2)
        det = 1.0 - c1[0] * c2[0] * cd * cd
        if abs(det) < 1e-12:
            raise ValueError("singular acceleration coupling")
        return [w1, w2, (b1 + c1[0] * cd * b2) / det,
                (b2 + c2[0] * cd * b1) / det]

    if len(_t_test) == 0:
        return np.full((2, 0), np.nan)
    try:
        sol = solve_ivp(rhs, (t_split, float(_t_test[-1])), y0,
                        t_eval=_t_test, method="RK45", rtol=1e-10,
                        atol=1e-12)
    except ValueError:
        return np.full((2, len(_t_test)), np.nan)
    return sol.y[:2] if sol.success else np.full((2, len(_t_test)), np.nan)


_y0 = [float(grid_d[k][n_train - 1])
       for k in ("theta1", "theta2", "omega1", "omega2")]
th_int = _integrate_dp(theta_hat_final, _y0)
th_int_true = _integrate_dp(
    {nm: THETA_TRUE[nm].cpu().numpy() for nm in EQ_NAMES}, _y0)


def _rel_int(pred, truth):
    return (float(np.linalg.norm(pred - truth) / np.linalg.norm(truth))
            if len(truth) else float("nan"))


_th1_te, _th2_te = th1_data[n_train:], th2_data[n_train:]
rel_l2_test_int = (_rel_int(th_int[0], _th1_te), _rel_int(th_int[1], _th2_te))
rel_l2_test_int_true = (_rel_int(th_int_true[0], _th1_te),
                        _rel_int(th_int_true[1], _th2_te))


print(f"\n========== DP AUTOSCALE PINN (weight-free loss) ==========")
print(f"mode: coefs={COEF_SOURCE}  ic={IC_MODE}  stat={STAT}  "
      f"anchor={ANCHOR}/{ANCHOR_WEIGHT}  fd_stride={FD_STRIDE}  "
      f"data_term={DATA_TERM}  select={BEST_KEY}")
print(f"loss = l_phys + l_ic + l_stat + l_anch + l_data   (no weights)")
print(f"yardsticks: PHYS_NORM={PHYS_NORM} -> div="
      + ", ".join(f"{k}={v:.4g}" for k, v in PHYS_DIV.items())
      + "  (max=" + ", ".join(f"{v:.4g}" for v in PHYS_SCALE.values())
      + "  sqrtVar=" + ", ".join(f"{np.sqrt(v):.4g}" for v in PHYS_VAR.values()) + ")"
      + f"  Var(theta)={IC_NORM_TH.cpu().numpy().round(4).tolist()}"
      + f"  Var(omega)={IC_NORM_OM.cpu().numpy().round(4).tolist()}")
print(f"selected checkpoint     = iter {best.iter}  "
      f"({BEST_KEY}={best.best:.3e})")
print(f"  loss at that point    = {best.loss:.3e}")
print(f"  coef err there        = {best.err:.3e}   (truth-based, DIAGNOSTIC)")
print(f"last loss               = {final.item():.3e}")
print(f"training time           = {elapsed:.1f}s")
print(f"train/test split        = {TRAIN_FRAC:.2f}  "
      f"(t_split={t_split:.3f}, n_train={n_train}/{N_GRID})")
print(f"rel L2 train (th1, th2) = {rel_l2_th1_train:.4e}, {rel_l2_th2_train:.4e}")
print(f"rel L2 test  (th1, th2) = {rel_l2_th1_test:.4e}, {rel_l2_th2_test:.4e}"
      f"   (net past t_split: NOT a result)")
print(f"rel L2 TEST  by INTEGRATION of the recovered pair = "
      f"{rel_l2_test_int[0]:.4e}, {rel_l2_test_int[1]:.4e}")
print(f"   same handover, TRUE coefs (chaotic floor)      = "
      f"{rel_l2_test_int_true[0]:.4e}, {rel_l2_test_int_true[1]:.4e}")
print(f"  final state instead   = train {final_rel[0]:.4e}, {final_rel[1]:.4e}"
      f"   test {final_rel[2]:.4e}, {final_rel[3]:.4e}")
for eq in EQUATIONS:
    nm = eq["name"]
    print(f"  {nm:>4}: theta_hat = {theta_hat_final[nm].round(4).tolist()}   "
          f"true = {eq['theta_true'].tolist()}")
    if ANCHOR_ERR is not None:
        # The cell's headline: did physics + consistency IMPROVE on the
        # estimate they were handed, or merely reproduce it?
        net_err = _rel_l1(theta_hat_final[nm], THETA_TRUE[nm].cpu().numpy())
        print(f"        anchor    = {THETA_FD[nm].cpu().numpy().round(4).tolist()}"
              f"   rel-L1 err: net {net_err:.4e}  vs anchor "
              f"{ANCHOR_ERR[nm]:.4e}   "
              f"({'IMPROVED' if net_err < ANCHOR_ERR[nm] else 'no gain'})")
    print(f"        OLS mean  = {theta_means_final[nm].round(4).tolist()}   "
          f"mean|res| = {float(np.abs(res_by_eq[nm]).mean()):.3e}")
    c = chi_by_eq[nm]
    print(f"        chi score = {c['score'].round(6).tolist()}  "
          f"bounded = {(c['score'] / (1 + c['score'])).round(6).tolist()}  "
          f"rss/yy = {float(c['rss'] / c['yy']):.3e}")
    h = het_by_eq[nm]
    print(f"        het score = {h['score'].round(6).tolist()}  "
          f"se_rel = {h['se_rel'].round(4).tolist()}")
    print(f"        max_corr  = {anchor_by_eq[nm]:.4e}   (sparsity.py anchor)")

save_kwargs = dict(
    t=t_grid,
    theta1_data=th1_data, theta2_data=th2_data,
    omega1_data=w1_data,  omega2_data=w2_data,
    theta1_pred=th1_pred, theta2_pred=th2_pred,
    omega1_pred=grid_d["omega1"], omega2_pred=grid_d["omega2"],
    alpha1_pred=grid_d["alpha1"], alpha2_pred=grid_d["alpha2"],
    g_true=G_TRUE,
    training_time=elapsed,
    train_frac=np.float32(TRAIN_FRAC),
    n_train=np.int32(n_train),
    t_split=np.float32(t_split),
    rel_l2_th1_train=np.float32(rel_l2_th1_train),
    rel_l2_th2_train=np.float32(rel_l2_th2_train),
    rel_l2_th1_test=np.float32(rel_l2_th1_test),
    rel_l2_th2_test=np.float32(rel_l2_th2_test),
    rel_l2_final=np.array(final_rel, dtype=np.float32),
    coef_err_final=np.float32(coef_err_final),
    rel_l2_test_int=np.array(rel_l2_test_int, dtype=np.float32),
    rel_l2_test_int_true=np.array(rel_l2_test_int_true, dtype=np.float32),
    th_test_int=th_int.astype(np.float32),
    best_iter=np.int32(best.iter),
    best_err=np.float32(best.err),
    loss_history_iter=np.array(hist["iter"], dtype=np.int32),
    loss_history_total=np.array(hist["tot"], dtype=np.float32),
    loss_history_phys=np.array(hist["phys"], dtype=np.float32),
    loss_history_phys_raw=np.array(hist["phys"], dtype=np.float32),
    loss_history_mcv=np.array(hist["mcv"], dtype=np.float32),
    loss_history_ic=np.array(hist["ic"], dtype=np.float32),
    loss_history_data=np.array(hist["data"], dtype=np.float32),
    loss_history_stat=np.array(hist["stat"], dtype=np.float32),
    loss_history_anch=np.array(hist["anch"], dtype=np.float32),
    grad_phys=np.array(hist["gphys"], dtype=np.float32),
    grad_data=np.array(hist["gdat"], dtype=np.float32),
    # Alias: compare_pinns' shared loss panel hard-keys on the cv slot.
    loss_history_cv=np.array(hist["stat"], dtype=np.float32),
)
for nm in EQ_NAMES:
    save_kwargs[f"theta_{nm}_per_window"] = theta_per_window[nm]
    save_kwargs[f"theta_hat_{nm}"]        = theta_hat_final[nm]
    save_kwargs[f"residual_{nm}_grid"]    = res_by_eq[nm]
    save_kwargs[f"phys_scale_{nm}"]       = np.float32(PHYS_SCALE[nm])
    save_kwargs[f"phys_var_{nm}"]         = np.float32(PHYS_VAR[nm])
    save_kwargs[f"phys_div_{nm}"]         = np.float32(PHYS_DIV[nm])
    save_kwargs[f"max_corr_{nm}"]         = np.float32(anchor_by_eq[nm])
    save_kwargs[f"het_{nm}_score"]        = het_by_eq[nm]["score"]
    save_kwargs[f"het_{nm}_se_rel"]       = het_by_eq[nm]["se_rel"]
    save_kwargs[f"het_{nm}_n_valid"]      = het_by_eq[nm]["n_valid"]
    save_kwargs[f"chi_{nm}_score"]        = chi_by_eq[nm]["score"]
    save_kwargs[f"chi_{nm}_theta_global"] = chi_by_eq[nm]["theta"]
    save_kwargs[f"chi_{nm}_rss"]          = np.float64(chi_by_eq[nm]["rss"])
    save_kwargs[f"chi_{nm}_yy"]           = np.float64(chi_by_eq[nm]["yy"])
if THETA_FD is not None:
    for nm in EQ_NAMES:
        save_kwargs[f"theta_fd_{nm}"]     = THETA_FD[nm].cpu().numpy()
        save_kwargs[f"anchor_err_{nm}"]   = np.float32(ANCHOR_ERR[nm])
save_kwargs["anchor"]        = np.str_(ANCHOR)
save_kwargs["fd_stride"]     = np.int32(FD_STRIDE)
save_kwargs["anchor_weight"] = np.str_(ANCHOR_WEIGHT)
save_kwargs["phys_norm"]   = np.str_(PHYS_NORM)
save_kwargs["phys_agg"]    = np.str_(PHYS_AGG)
save_kwargs["coef_source"] = np.str_(COEF_SOURCE)
save_kwargs["ic_mode"]     = np.str_(IC_MODE)
save_kwargs["stat"]        = np.str_(STAT)
save_kwargs["stat_bound"]  = np.bool_(STAT_BOUND)
save_kwargs["data_term"]   = np.bool_(DATA_TERM)
save_kwargs["best_key"]    = np.str_(BEST_KEY)
out_path = f"dp_pinn_auto{_SUFFIX}.npz"
np.savez(out_path, **save_kwargs)
print(f"saved -> {out_path}")
