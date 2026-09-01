"""
chi2-of-OLS DP PINN -- two candidate equations.

Same physics term as the baseline (TRUE coefs in the residual), plus a
RAW Nyblom-Hansen constancy penalty on the GLOBAL OLS recovery of each
equation's coefficients -- the WINDOWLESS counterpart of pinn_test_cv's
30-overlapping-window het score (see cv_metric.chi2_per_term; it mirrors
epde's survival.chi2_scores(..., rescale=False), the production keep-rule
/ Instability formula):

    L_phys = sum_eq mean_w mean_{k in w} ( (target_eq(t_k) - features_eq(t_k) @ theta_true_eq) / PHYS_SCALE_eq )^2
    L_chi  = sum_eq sum_j chi2_j    with, per equation over T_train,
             c = global OLS,  r = y - A c,  S_j(t) = cumsum_t A_tj r_t,
             chi2_j = (1/N) sum_t S_j(t)^2 / D_j   (D_j = own fitted-
             signal energy, detached -- the calibration boundary)
    L_mcv  = windowed-moment CV: per-window (mean, std) of the MODEL angles
             vs the DATA's (DIAGNOSTIC ONLY by default: W_MCV = 0)
    L_anch = sum_eq sum_j w_j * (theta_glob_j - theta*_j)^2 [/ theta*_j^2]
             (truth anchor on the SAME global OLS recovery -- the
             anchored half of pinn_test_cv's cv2+anchored; the
             normalization follows ANCH_NORM and the per-term weight
             w_j follows ANCH_WEIGHT -- 'chi_norm' makes each term's own
             INCONSISTENCY its anchor weight; W_ANCH = 0 gives the
             pure-chi cells)
    L_ic   = mean( (theta(0) - theta_data(0))^2 ) + mean( (omega(0) - omega_data(0))^2 )
    L      = W_PHYS * L_phys + W_CV * L_chi + W_ANCH * L_anch
             + W_MCV * L_mcv + W_IC * L_ic

The chi term is TRUTH-FREE and window-free: one global differentiable
OLS on the train slice supplies theta (the evidence rule -- never a
trainable parameter), and each term is scored by how much its cumulative
score path bulges between its pinned endpoints. Unlike het's [0,1)
score, raw chi2 is unbounded -- read the early [adam ...] prints to see
its magnitude against phys/ic before trusting a full run. An exact fit
floors the score to ZERO, so chi -- like het -- rewards degeneracy and
cannot forbid it: physics + IC are the collapse blockers, and L_mcv is
the logged watchdog.

The physics residual is normalized by the constant per-equation
PHYS_SCALE (computed once from train-slice FD derivatives -- stationary
objective, nothing hardcoded) and decomposed into per-window MSEs over
the same overlapping windows pinn_test_cv uses, so the two scripts train
the same physics estimand and their loss histories compare head-to-head.
The 30-window OLS recovery still runs each iteration: it feeds the
windowed physics decomposition, the best-checkpoint coefficient error,
and the saved theta_*_per_window -- only the LOSS's stability statistic
is the windowless chi.

NO observations of the solution in the loss: the observed angles enter
ONLY through the initial condition (theta(0), omega(0)). L_mcv is
computed and logged every iteration as a parked-state/branch diagnostic
(coefficient constancy is branch-blind: every solution of the true ODE,
wound or parked at k*pi, recovers the true theta -- L_mcv is the
statistic that would detect the wrong branch), but it is OFF in the loss
because it requires solution observations, which this experiment avoids.
Set W_MCV > 0 only for an explicitly observation-using ablation.

OLS gradient through `torch.linalg.solve` can be ill-conditioned at small
network amplitudes -- this variant shares that trait with pinn_test_cv
(one global solve instead of 30 windowed ones).

Physics + chi losses are computed on the TRAIN slice of the data
grid (t in [0, TRAIN_FRAC * T_MAX]) each iteration -- the held-out tail
never enters the loss, and the chi path's cumsum runs over the
time-ordered train points. Post-hoc OLS at evaluation also runs on the
TRAIN slice only, so the network's unconstrained extrapolation past
t_split cannot pollute the reported per-window coefficients.
"""

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from cv_metric import (
    EQUATIONS,
    EQ_NAMES,
    LearningGifRecorder,
    chi2_per_term,
    compute_equation_thetas,
    evaluate_on_grid,
    grid_equation_residuals,
    het_per_window,
    make_windows,
    network_derivatives_dp,
    set_gravity,
)

torch.set_default_dtype(torch.float32)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================ settings
DATA_PATH  = "./dp.npz"

HIDDEN     = [64] * 5
ADAM_ITERS = 20000
ADAM_LR    = 1e-3
LBFGS_MAX  = 20000
# W_MCV = 0: the moment-CV term needs observations of the solution, which
# this experiment avoids -- it is computed/logged as a diagnostic only.
# The W_CV slot carries the chi term (the coef-match precedent of reusing
# the cv slot). The IC is the ONLY branch-selecting evidence in the loss
# (1 data row vs 801 collocation points).
# W_PHYS = W_IC = 0 is the PURE-STATISTICS cell: loss = chi + anchor
# only. The anchor pulls the global OLS recovery toward theta* (and
# blocks collapse: a parked trajectory OLS-recovers theta ~ 0, costing
# the full relative error); chi stabilizes the fit along the path.
# BRANCH-FREE by construction: with no IC and no physics, ANY
# trajectory of the true ODE (any initial condition) is a global
# optimum, so coefficient recovery is the only meaningful metric --
# rel-L2 vs the observed trajectory is expected to be garbage. Physics
# is still computed and logged (phys/phys_raw) as a diagnostic. The
# physics-bearing cells all ran 100/100/0/100 (mirroring
# pinn_test_cv.py) with W_ANCH per their suffix; the anch+IC nophys
# cell ran 0/1/0/100 with W_ANCH=100.
W_PHYS, W_CV, W_MCV, W_IC =1.0, 1.0, 0.0, 100.0
SEED       = 0

N_WIN      = 30
WIN_FRAC   = 0.5
TRAIN_FRAC = 0.8       # temporal split: train on first TRAIN_FRAC of trajectory
EPS = 1e-12

# Emit a learning-process GIF (model theta1/theta2 over the domain across
# Adam iters) as part of this run -- see LearningGifRecorder in cv_metric.py.
MAKE_GIF = True

# chi statistic precision/floor controls. The float32+floor cell
# (dp_pinn_chi.npz) ended ON the exact-fit floor with a WRONG recovery
# (both equations at rss/yy ~5-6e-5, below the ~9.5e-5 float32 floor,
# so chi read exactly [0,0,0]): the floor acted as a zero-loss
# attractor, not a benign self-limit. CHI_FLOAT64 computes the chi term
# in float64 (floor would sit at ~1.8e-13) and CHI_NO_FLOOR removes the
# cliff entirely -- chi stays a live (if tiny) functional at any fit
# quality. Artifacts are suffixed so the two cells coexist in
# compare_pinns.py.
CHI_FLOAT64  = True
CHI_NO_FLOOR = True

# Truth-anchored coefficient term (the "anchored" half of pinn_test_cv's
# cv2+anchored form, applied to the GLOBAL OLS recovery chi already
# computes -- theta comes out of chi2_per_term gradient-bearing, so the
# anchor is free):
#   L_anch = sum_eq sum_j (theta_glob_j - theta*_j)^2 / theta*_j^2
# The f64 no-floor cell showed chi's blind spot is a wrong-but-CONSISTENT
# basin (sign-flipped, gravity-free): the anchor forbids that theta the
# way cv2+anchored forbids the trivial one -- chi supplies constancy
# pressure, the anchor supplies identity. ORACLE term (theta_true enters
# the loss, like the physics residual in this oracle-physics probe
# family; het remains the only non-oracle stability form). 0 disables
# (the pure-chi cells). The artifact suffix records it so cells coexist
# in compare_pinns.py.
W_ANCH = 1.0

# Anchor normalization. 'rel' = the anchored_mse convention,
# (theta - theta*)^2 / theta*^2: scale-fair for REPORTING, but as a
# LOSS every DELETED coefficient costs exactly 1 -- the largest physics
# (gravity, theta* = -9.81) is the cheapest to drop per unit of basin
# relief it buys, and gravity measured dead in every rel-anchored cell.
# 'abs' = (theta - theta*)^2: deletion costs theta*^2 (gravity ~96 vs
# coupling 0.25-1) -- the anchor pulls hardest on the largest-magnitude
# physics. The suffix records the form so cells coexist.
ANCH_NORM = "abs"

# PER-TERM anchor weighting by the term's own INCONSISTENCY:
#   L_anch = sum_eq sum_j w_j * (theta_glob_j - theta*_j)^2 [/ theta*_j^2]
# 'none' = uniform (w_j = 1). 'chi' (the DEFAULT) = w_j is that term's
# chi score, RAW: no floor, no rescale -- the inconsistency itself is
# the price, so the anchor's total magnitude moves with the fit and a
# term with chi = 0 gets NO pull at all (it is, by the statistic's own
# reading, perfectly consistent). 'chi_norm' = floored at ANCH_W_FLOOR *
# max(chi) and rescaled to mean 1 across the equation's terms, which
# keeps only the RELATIVE emphasis and a stable W_ANCH -- available, but
# the rescale is exactly what the raw form is meant to avoid.
# Why this aims where it should: chi measures each term against its OWN
# fitted-signal energy D_j = sum_i (w_i X_ij (c_j X_ij))^2, so a
# coefficient the fit drives toward 0 COLLAPSES its D_j and explodes its
# own score (survival.py's degenerate-form guard). Measured in the
# abs-anchored pure cell: chi th1 [0, 0, 0.232], th2 [1e-6, 8.1e-5,
# 0.502] -- the DELETED gravity term carries ~all the inconsistency, so
# chi weighting concentrates the anchor pull on exactly the term that is
# missing. Trade-off: terms already recovered well get their pull cut to
# ~0 and may drift.
# The weight is DETACHED (a weight, not a term -- the causal gate and
# het's calibration follow the same rule): with gradient through w_j the
# net could cut the anchor by making a WRONG term look consistent, which
# is the incentive backwards. Attaching it is a one-line change.
ANCH_WEIGHT = "none"
# Scale-free floor for 'chi_norm', as a fraction of the equation's max
# chi: keeps a consistent term's pull small-but-alive and makes the
# mean-1 rescale well-defined. All-zero chi (no evidence) -> uniform.
ANCH_W_FLOOR = 1e-3

# w_j = ANCH_W_BASE + chi_j. REJECTED as a design (kept only so the
# measured cell stays reproducible; 0 = off, the pure form).
#
# The problem it was meant to solve is real: raw chi weights make the
# anchor a PRODUCT of wrongness x inconsistency, so the optimizer zeroes
# the EASY factor -- measured in the pure raw cell, loss fell to 0.205
# with gravity still ~96 units wrong, and the run ENDED at loss 4.2 with
# worse coefficients than the loss-5.9e5 state the checkpoint kept.
#
# But a constant baseline is the wrong cure. It has no cross-system
# meaning (it declares "chi below BASE is irrelevant" at a scale chi
# never agreed to), and at chi ~ 1e-6 for consistent terms the weight is
# just BASE everywhere except one spike -- the cell degenerates into the
# uniform anchor plus a bump, testing neither approach purely.
#
# The structural read: the zero-weight case is NOT the deleted term
# (a coefficient driven to 0 collapses its own D_j and EXPLODES chi).
# It is a term that is PRESENT, STABLE and WRONG -- gravity's couplings
# sat at chi ~ 0.13 while being 2x off. Only the anchor knows wrongness;
# chi is wrongness-BLIND by construction. So no multiplicative weight
# built from chi can price a stable-but-wrong term, with or without a
# floor. The constant-free composition is ADDITIVE: ANCH_WEIGHT='none'
# with ANCH_COUPLING='add' keeps chi and the anchor as separate terms,
# each with its own gradient path, and neither can switch the other off.
ANCH_W_BASE = 0.0

# ---------------------------------------------------------- trainable theta
# Port of pinn_test_cv_trainable_theta.py's approach. THETA_TRAINABLE
# replaces the ORACLE coefficients in the physics residual with an
# nn.Parameter per equation -- shape (K,), GLOBAL, because chi's estimand
# is the single 801-point fit (the sibling script uses (N_WIN, K) to match
# its per-window OLS estimand). The net and theta then co-adapt:
#   r = (y - A @ theta_train) / PHYS_SCALE
# THE EVIDENCE RULE (survival.py:41-46) IS NOT NEGOTIABLE HERE: chi and
# the anchor keep reading the net's OLS RECOVERY, never theta_train.
# A constancy statistic computed over optimizer-owned coefficients is a
# tying prior the optimizer trivially zeroes -- it would measure nothing
# about the trajectory. Same split the sibling script uses for W_HET.
# theta_train only enters L_phys, so W_PHYS must be > 0 (enforced below).
THETA_TRAINABLE   = True
THETA_LR          = 1e-2
# Truth-init + warm-up freeze, mirroring the sibling script: coefficient
# error is 0 at iter 0 by construction and the run tests whether the
# landscape KEEPS theta at truth while the net learns. Safe here because
# the best key reads the NET's OLS recovery, not theta_train, so a frozen
# theta cannot lock the checkpoint. False = zeros-init discovery run.
THETA_INIT_TRUE   = True
THETA_FREEZE_ITERS = 2000

# How the chi term and the anchor COMBINE into the total loss.
# 'add' = W_CV * chi + W_ANCH * anch. With ANCH_WEIGHT = 'chi_norm' the
#         inconsistency weighting is already applied PER TERM inside
#         physics_and_cv_loss, so the two totals simply add.
# 'mul' = chi * anch, ONE GLOBAL product: the summed inconsistency
#         scales the whole anchor. Coarser than per-term weighting (it
#         cannot tell WHICH term is inconsistent, and it couples the two
#         equations through a single scalar), and it carries a perverse
#         incentive -- shrinking total chi discounts the entire anchor,
#         including the terms that are still wrong. Kept selectable.
# Both optimizers (Adam and the L-BFGS closure) must use the same one.
ANCH_COUPLING = "add"

_ANCH_TAG = ("_anch"
             + ("abs" if ANCH_NORM == "abs" else "")
             + {"none": "", "chi": "chiw", "chi_log": "chiwl",
                "chi_norm": "chiwn"}[ANCH_WEIGHT]
             + ("b" if ANCH_W_BASE > 0
                and ANCH_WEIGHT in ("chi", "chi_log") else "")
             + ("mul" if ANCH_COUPLING == "mul" else ""))

CHI_SUFFIX = ("_f64" if (CHI_FLOAT64 or CHI_NO_FLOOR) else "") \
    + (_ANCH_TAG if W_ANCH > 0 else "") \
    + ("_tth" if THETA_TRAINABLE else "") \
    + ("_nophys" if W_PHYS == 0 else "") \
    + ("_noic" if W_IC == 0 else "")

if THETA_TRAINABLE and W_PHYS == 0:
    # theta_train enters ONLY the physics residual -- with W_PHYS = 0 it
    # is a dangling parameter and the run silently measures the
    # non-trainable cell. Fail loud (the Instability.compute convention).
    raise ValueError(
        "THETA_TRAINABLE requires W_PHYS > 0: the trainable coefficients "
        "only enter L_phys, so at W_PHYS = 0 they receive no gradient and "
        "the cell is identical to THETA_TRAINABLE = False.")

# Causal physics weighting (Wang/Sankaran/Perdikaris, "Respecting
# causality is all you need"): per-window weights w_i =
# exp(-CAUSAL_EPS * sum_{j<i} L_j), detached, over the SUM of both
# equations' per-window MSEs (one shared trajectory -> one causal gate).
# Later windows only count once earlier ones fit -- the observation-free
# pressure toward the IC-implied branch. Granularity is the window
# decomposition itself (coarse at WIN_FRAC = 0.5: window 0 already spans
# half the train slice). w_last (logged) -> 1 = gate fully open.
# 0 disables.
CAUSAL_EPS = 0.0

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

# IC supervision: angles + angular velocities at t = t_grid[0].
T_IC  = T_obs[:1]
TH_IC = th_obs[:1]
OM_IC = torch.tensor([[w1_data[0], w2_data[0]]], device=device)

# Temporal train/test split: only T_train enters the physics + chi
# loss. Windows still span [0, T] (see make_windows below); the OLS mask
# is built from T_train so the per-window fits only see the train
# portion, and the chi path accumulates over the (time-ordered) train
# points only. Post-hoc OLS sweeps the train slice for diagnostics.
n_train = int(np.ceil(TRAIN_FRAC * N_GRID))
t_split = float(t_grid[n_train - 1])
T_train  = T_obs[:n_train]
th_train = th_obs[:n_train]
T_test   = T_obs[n_train:]
th_test  = th_obs[n_train:]


# Per-equation physics normalization scale, computed ONCE at startup from
# the train slice: finite-difference derivatives of the observed state feed
# each equation's own target_and_features, so nothing about the target's
# magnitude is hardcoded and the mechanism works for arbitrary equations.
# A CONSTANT scale keeps the objective stationary across iterations (the
# previous max|y_hat| normalization moved with the net's own prediction:
# loss values incomparable between iterations, and the untrained net's
# tiny |alpha| max inflates early physics). The scale enters the loss only
# as a fixed per-equation weight: no gradient toward observations, and
# the residual's zero set is unchanged.
def _fd_target_scales():
    tt = t_grid[:n_train].astype(np.float64)
    d_np = dict(
        theta1=th1_data[:n_train].astype(np.float64),
        theta2=th2_data[:n_train].astype(np.float64),
        omega1=w1_data[:n_train].astype(np.float64),
        omega2=w2_data[:n_train].astype(np.float64),
    )
    d_np["alpha1"] = np.gradient(d_np["omega1"], tt)
    d_np["alpha2"] = np.gradient(d_np["omega2"], tt)
    d = {k: torch.tensor(v) for k, v in d_np.items()}
    return {eq["name"]: float(eq["target_and_features"](d)[0].abs().max()) + EPS
            for eq in EQUATIONS}


PHYS_SCALE = _fd_target_scales()               # {eq_name: constant scale}

t_lo, t_hi = make_windows(T_MIN, T_MAX, N_WIN, WIN_FRAC)
PERIOD = None
T_LO = torch.tensor(t_lo, device=device).unsqueeze(1)
T_HI = torch.tensor(t_hi, device=device).unsqueeze(1)


# ============================================================ windowed-moment CV
# DIAGNOSTIC (W_MCV = 0 by default): per-window (mean, std) of the MODEL
# angles vs the DATA's, normalized by the data variance. Coefficient
# constancy is branch-blind (every solution of the true ODE recovers the
# true theta), so this is the statistic that DETECTS a wrong branch or a
# parked k*pi state in the loss history -- but weighting it into the
# loss would inject solution observations, which this experiment avoids.
# Computed on the TRAIN slice only.
_tm = T_train[:, 0]
if PERIOD is None:
    _mcv_in = (_tm.unsqueeze(0) >= T_LO) & (_tm.unsqueeze(0) < T_HI)
else:
    _c = (T_LO + T_HI) / 2.0
    _h = (T_HI - T_LO) / 2.0
    _dd = (_tm.unsqueeze(0) - _c).abs()
    _mcv_in = torch.minimum(_dd, PERIOD - _dd) < _h
MCV_MASK = _mcv_in.to(torch.float32)                   # (N_WIN, n_train)
MCV_VALID = MCV_MASK.sum(dim=1) >= 8                   # enough points to matter


def _win_moments(vals):
    """Per-window weighted mean/std of (n_train, 2) values -> two (N_WIN, 2)."""
    cnt = MCV_MASK.sum(dim=1, keepdim=True).clamp(min=1.0)
    mu = (MCV_MASK @ vals) / cnt
    var = (MCV_MASK @ vals ** 2) / cnt - mu ** 2
    return mu, (var.clamp(min=0.0) + 1e-12).sqrt()


MCV_MU_D, MCV_SD_D = _win_moments(th_train)
MCV_NORM = th_train.var(dim=0) + EPS                   # (2,) data variance


def moment_cv_loss(th_model):
    """th_model: (n_train, 2) model angles on T_train, reused from
    physics_and_cv_loss so the diagnostic costs no extra forward pass."""
    mu_m, sd_m = _win_moments(th_model)
    per = ((mu_m - MCV_MU_D) ** 2 + (sd_m - MCV_SD_D) ** 2) / MCV_NORM.unsqueeze(0)
    return per[MCV_VALID].mean()


# ============================================================ network
class MLP(nn.Module):
    def __init__(self, hidden=HIDDEN):
        super().__init__()
        layers = [1] + list(hidden) + [2]
        mods = []
        for i in range(len(layers) - 1):
            mods.append(nn.Linear(layers[i], layers[i + 1]))
            if i < len(layers) - 2:
                mods.append(nn.Tanh())
        self.net = nn.Sequential(*mods)
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, t):
        return self.net(t)


# ============================================================ physics + chi
def _chi_term(y, A):
    """chi2_per_term under the module's precision/floor flags. The
    float64 cast keeps the whole chi graph (solve, path, calibration)
    in double precision; gradients flow back to the float32 net."""
    if CHI_FLOAT64:
        y, A = y.double(), A.double()
    return chi2_per_term(y, A, ridge=EPS, use_floor=not CHI_NO_FLOOR)


def _anchor_weights(chi_j):
    """Per-term anchor weights from the term's own chi score (DETACHED --
    see ANCH_WEIGHT). 'chi' passes the raw score; 'chi_norm' floors it at
    ANCH_W_FLOOR * max(chi) (scale-free) and rescales to mean 1, falling
    back to uniform when there is no inconsistency evidence at all."""
    w = chi_j.detach()
    if ANCH_WEIGHT == "chi":
        return ANCH_W_BASE + w
    if ANCH_WEIGHT == "chi_log":
        # Same baseline, but log1p compresses chi's ~1e-5..5e3 spread so
        # one exploding term cannot swamp the rest of the anchor.
        return ANCH_W_BASE + torch.log1p(w)
    w = w + ANCH_W_FLOOR * w.max()
    m = w.mean()
    return w / m if m > 0 else torch.ones_like(w)


def _stat_loss(l_cv, l_anch):
    """Combine the chi and anchor totals per ANCH_COUPLING. One helper so
    Adam and the L-BFGS closure can never drift onto different objectives."""
    if ANCH_COUPLING == "mul":
        return l_cv * l_anch
    return W_CV * l_cv + W_ANCH * l_anch


def physics_and_cv_loss(net, T_coll, theta_params=None):
    """One autograd pass -> per-equation physics + per-equation chi penalty.

    Physics uses the TRUE coefs, normalized by the constant per-equation
    PHYS_SCALE and decomposed into per-window MSEs over the same
    overlapping windows pinn_test_cv uses (both scripts train the same
    physics estimand). The chi penalty is WINDOWLESS: cv_metric.
    chi2_per_term on the full time-ordered train slice (T_coll rows are
    ascending t), gradient through the global linalg.solve, calibration
    denominator detached. With CAUSAL_EPS > 0 the window MSEs get
    detached causal weights shared between the equations; phys is the
    weighted (trained) value, phys_raw the unweighted one (comparable
    across iterations while the gate drifts open).
    Returns a dict: phys, phys_raw, cv (loss tensors; cv = the chi term),
    w_last (float, causal gate state), per_eq (trained residual per
    equation), theta_means, th_model ((n_train, 2) model angles, reused
    by the moment-CV diagnostic).
    """
    results, derivs, mask = compute_equation_thetas(net, T_coll, T_LO, T_HI, ridge=EPS, period=PERIOD)
    th_model = torch.stack([derivs["theta1"], derivs["theta2"]], dim=1)
    cnt = mask.sum(dim=1).clamp(min=1.0)

    l_cv_total = 0.0
    l_anch_total = 0.0
    per_eq_mse = {}    # (N_WIN,) per-window MSE of the true-coef residual
    theta_means = {}   # 30-window OLS mean, per equation
    theta_globs = {}   # the GLOBAL OLS recovery chi/anchor actually train
    for eq in EQUATIONS:
        nm = eq["name"]
        theta_pw, theta_mean = results[nm]
        y, A = eq["target_and_features"](derivs)
        # Physics coefficients: the ORACLE truth, or the trainable
        # parameter when THETA_TRAINABLE (the net and theta co-adapt).
        # chi and the anchor below are UNAFFECTED -- they read the net's
        # OLS recovery, never theta_train (the evidence rule).
        c_phys = (THETA_TRUE[nm] if theta_params is None
                  else theta_params[nm])
        r = (y - A @ c_phys) / PHYS_SCALE[nm]
        per_eq_mse[nm] = (mask @ (r ** 2)) / cnt
        # chi penalty on the GLOBAL OLS recovery (gradient through
        # linalg.solve; T_coll rows ascend in t, as the cumsum requires),
        # plus the truth anchor on the SAME recovery (always computed --
        # W_ANCH decides whether it trains; logged either way).
        out_chi = _chi_term(y, A)
        chi_j = out_chi["score"]
        l_cv_total = l_cv_total + chi_j.sum()
        tt = THETA_TRUE[nm].to(out_chi["theta"].dtype)
        anch_pc = (out_chi["theta"] - tt) ** 2
        if ANCH_NORM == "rel":
            anch_pc = anch_pc / tt ** 2
        if ANCH_WEIGHT != "none":
            anch_pc = anch_pc * _anchor_weights(chi_j)
        l_anch_total = l_anch_total + anch_pc.sum()
        theta_means[nm] = theta_mean
        theta_globs[nm] = out_chi["theta"].detach()

    # One shared causal gate from the summed per-window MSEs (the two
    # equations describe the same trajectory). Weights are detached:
    # gradient flows through the MSEs only.
    w_causal = None
    if CAUSAL_EPS > 0:
        mse_tot = sum(per_eq_mse[nm] for nm in EQ_NAMES)    # (N_WIN,)
        with torch.no_grad():
            cum_before = torch.cumsum(mse_tot, dim=0) - mse_tot
            w_causal = torch.exp(-CAUSAL_EPS * cum_before)

    l_phys_total = 0.0
    l_phys_raw_total = 0.0
    per_eq_residual = {}
    for nm in EQ_NAMES:
        m = per_eq_mse[nm]
        raw = m.mean()
        l_phys_eq = (w_causal * m).mean() if w_causal is not None else raw
        l_phys_total = l_phys_total + l_phys_eq
        l_phys_raw_total = l_phys_raw_total + raw
        per_eq_residual[nm] = l_phys_eq
    w_last = float(w_causal[-1]) if w_causal is not None else 1.0

    return dict(phys=l_phys_total, phys_raw=l_phys_raw_total, cv=l_cv_total,
                anch=l_anch_total, w_last=w_last, per_eq=per_eq_residual,
                theta_means=theta_means, theta_globs=theta_globs,
                th_model=th_model)


def ic_loss(net):
    """Soft initial condition: theta(0) and omega(0) from the data."""
    d0 = network_derivatives_dp(net, T_IC)
    th0 = torch.stack([d0["theta1"], d0["theta2"]], dim=1)     # (1, 2)
    om0 = torch.stack([d0["omega1"], d0["omega2"]], dim=1)     # (1, 2)
    return ((th0 - TH_IC) ** 2).mean() + ((om0 - OM_IC) ** 2).mean()


# ============================================================ train
torch.manual_seed(SEED); np.random.seed(SEED)

net = MLP().to(device)

# Trainable physics coefficients (one GLOBAL (K,) vector per equation --
# chi's estimand is the single 801-point fit). None when disabled, which
# routes physics_and_cv_loss back to the oracle THETA_TRUE.
theta_params = None
if THETA_TRAINABLE:
    theta_params = {}
    for eq in EQUATIONS:
        init = (torch.tensor(eq["theta_true"].astype(np.float32),
                             device=device)
                if THETA_INIT_TRUE else torch.zeros(eq["k"], device=device))
        theta_params[eq["name"]] = nn.Parameter(init)
    # Warm-up freeze: theta holds its init while the net fits at that
    # theta (Adam skips params whose grad is None); thawed in the loop.
    if THETA_FREEZE_ITERS > 0:
        for p in theta_params.values():
            p.requires_grad_(False)

opt = (torch.optim.Adam(net.parameters(), lr=ADAM_LR) if theta_params is None
       else torch.optim.Adam([
           {"params": net.parameters(), "lr": ADAM_LR},
           {"params": list(theta_params.values()), "lr": THETA_LR}]))

# Learning-process GIF recorder (captures theta1/theta2 over the full grid
# at scheduled Adam iters; renders after training).
gif_rec = None
if MAKE_GIF:
    PLOTS_DIR = Path(__file__).resolve().parent / "plots"
    PLOTS_DIR.mkdir(exist_ok=True)
    gif_rec = LearningGifRecorder(
        t_grid, th1_data, th2_data, t_split,
        label=f"chi2{CHI_SUFFIX}",
        out_path=PLOTS_DIR / f"learning_chi{CHI_SUFFIX}.gif",
        total_iters=ADAM_ITERS)


def _gif_predict():
    with torch.no_grad():
        p = net(T_obs).detach().cpu().numpy()
    return p[:, 0], p[:, 1]


# Track best checkpoint by RELATIVE L1 coefficient error across all
# equations: sum_eq sum_j |theta_j - theta_true_j| / |theta_true_j|.
# NOTE this criterion is BRANCH-BLIND (every solution of the true ODE
# OLS-recovers truth) and ignores the IC: the restored net is "the most
# truth-identifiable trajectory", not "closest to the data". Kept by
# design -- this script is the oracle-physics probe. There is no iter-0
# lock (OLS on the untrained net gives large err, unlike trainable
# truth-init). The snapshot is taken BEFORE opt.step() so the saved
# state is the one whose err was measured.
#
# BEST_KEY picks WHICH recovery the criterion reads:
#   'global'   = the single 801-point OLS that chi and the anchor
#                actually train (chi2_per_term's theta).
#   'windowed' = the 30-window OLS mean (the original criterion, kept
#                for comparability with the cv/baseline scripts).
# 'global' is the default because the windowed mean is NOT robust here:
# a half-domain window landing where a feature has almost no energy has
# a near-singular Gram and throws its theta_w anywhere (window means of
# 1e2-1e4 observed while the global fit sat near truth), so the
# criterion was REJECTING states the loss considered excellent --
# measured: the chi-weighted-anchor run passed through anchor 0.72 at
# iter 15000 and saved a state whose anchor was ~500.
BEST_KEY = "global"

best = {"err": float("inf"), "loss": float("inf"), "state": None}


def _coef_err_sum(ml):
    d = ml["theta_globs"] if BEST_KEY == "global" else ml["theta_means"]
    return sum(
        ((d[nm].detach().to(THETA_TRUE[nm].dtype) - THETA_TRUE[nm]).abs()
         / THETA_TRUE[nm].abs()).sum().item()
        for nm in EQ_NAMES
    )


# Adam loss history (sampled at the print frequency).
hist = {"iter": [], "tot": [], "phys": [], "phys_raw": [], "cv": [],
        "anch": [], "mcv": [], "ic": [], "wlast": []}

t0 = time.time()
for it in range(ADAM_ITERS):
    if gif_rec is not None:
        gif_rec.maybe_capture(it, _gif_predict)
    if theta_params is not None and it == THETA_FREEZE_ITERS:
        for p in theta_params.values():
            p.requires_grad_(True)
    opt.zero_grad()
    ml = physics_and_cv_loss(net, T_train, theta_params)
    l_phys, l_cv = ml["phys"], ml["cv"]
    l_mcv = moment_cv_loss(ml["th_model"])
    l_ic = ic_loss(net)
    loss = (W_PHYS * l_phys + _stat_loss(l_cv, ml["anch"])
            + W_MCV * l_mcv + W_IC * l_ic)
    loss.backward()
    cur = loss.item()
    err = _coef_err_sum(ml)
    if err < best["err"]:
        best["err"] = err
        best["loss"] = cur
        best["state"] = {k: v.detach().clone() for k, v in net.state_dict().items()}
        if theta_params is not None:
            best["theta"] = {nm: theta_params[nm].detach().clone()
                             for nm in EQ_NAMES}
    opt.step()
    if it % 1000 == 0:
        hist["iter"].append(it)
        hist["tot"].append(cur)
        hist["phys"].append(l_phys.item())
        hist["phys_raw"].append(ml["phys_raw"].item())
        hist["cv"].append(l_cv.item())
        hist["anch"].append(ml["anch"].item())
        hist["mcv"].append(l_mcv.item())
        hist["ic"].append(l_ic.item())
        hist["wlast"].append(ml["w_last"])
        phys_msgs = "  ".join(f"{nm}={ml['per_eq'][nm].item():.2e}" for nm in EQ_NAMES)
        err_msgs = "  ".join(
            f"{nm}={(ml['theta_means'][nm] - THETA_TRUE[nm]).detach().cpu().numpy().round(4).tolist()}"
            for nm in EQ_NAMES)
        print(f"[adam {it:5d}] tot={loss.item():.3e}  chi={l_cv.item():.3e}  "
              f"anch={ml['anch'].item():.3e}  "
              f"mcv={l_mcv.item():.3e}  ic={l_ic.item():.3e}  "
              f"w_last={ml['w_last']:.2f}  | phys: {phys_msgs}  | err: {err_msgs}")

# Final Adam-state frame (the GIF animates the Adam learning process; the
# subsequent L-BFGS refinement is a single .step() and is not animated).
if gif_rec is not None:
    gif_rec.maybe_capture(ADAM_ITERS, _gif_predict)

# L-BFGS jointly over net (+ trainable theta). Thaw theta even if
# ADAM_ITERS <= THETA_FREEZE_ITERS left the warm-up freeze on.
_lbfgs_params = list(net.parameters())
if theta_params is not None:
    for p in theta_params.values():
        p.requires_grad_(True)
    _lbfgs_params += list(theta_params.values())
lbfgs = torch.optim.LBFGS(_lbfgs_params,
    max_iter=LBFGS_MAX, max_eval=LBFGS_MAX,
    tolerance_grad=1e-8, tolerance_change=1e-10,
    history_size=50, line_search_fn="strong_wolfe")


def closure():
    lbfgs.zero_grad()
    ml = physics_and_cv_loss(net, T_train, theta_params)
    l_mcv = moment_cv_loss(ml["th_model"])
    l_ic = ic_loss(net)
    loss = (W_PHYS * ml["phys"] + _stat_loss(ml["cv"], ml["anch"])
            + W_MCV * l_mcv + W_IC * l_ic)
    loss.backward()
    err = _coef_err_sum(ml)
    if err < best["err"]:
        best["err"] = err
        best["loss"] = loss.item()
        best["state"] = {k: v.detach().clone() for k, v in net.state_dict().items()}
        if theta_params is not None:
            best["theta"] = {nm: theta_params[nm].detach().clone()
                             for nm in EQ_NAMES}
    return loss


final = lbfgs.step(closure)
elapsed = time.time() - t0

# Restore the lowest-loss checkpoint we saw during training.
net.load_state_dict(best["state"])
if theta_params is not None and best.get("theta") is not None:
    for nm in EQ_NAMES:
        theta_params[nm].data.copy_(best["theta"][nm])

if gif_rec is not None:
    gif_rec.render()


# ============================================================ evaluate
with torch.no_grad():
    th_pred = net(T_obs).cpu().numpy()
th1_pred = th_pred[:, 0]
th2_pred = th_pred[:, 1]

def _rel_l2(pred, data, sl):
    p = pred[sl]; d = data[sl]
    return float(np.linalg.norm(p - d) / np.linalg.norm(d))


sl_train = slice(0, n_train)
sl_test  = slice(n_train, None)
rel_l2_th1_train = _rel_l2(th1_pred, th1_data, sl_train)
rel_l2_th2_train = _rel_l2(th2_pred, th2_data, sl_train)
rel_l2_th1_test  = _rel_l2(th1_pred, th1_data, sl_test)
rel_l2_th2_test  = _rel_l2(th2_pred, th2_data, sl_test)

# Post-hoc per-window OLS on the TRAIN slice only -- using T_obs would let
# the network's unconstrained extrapolation in (t_split, T_MAX] pollute the
# OLS design matrix for windows whose span overlaps the test region.
ols_results, _eval_derivs, _eval_mask = compute_equation_thetas(
    net, T_train, T_LO, T_HI, ridge=EPS, period=PERIOD)
theta_per_window = {nm: ols_results[nm][0].detach().cpu().numpy() for nm in EQ_NAMES}
theta_means_final = {nm: ols_results[nm][1].detach().cpu().numpy() for nm in EQ_NAMES}

# het diagnostic on the same recovery: score ~ 0 with SMALL se_rel =
# genuinely consistent windows; score ~ 0 with HUGE se_rel / low n_valid =
# no evidence (parked/degenerate trajectory). Kept post-hoc so the
# chi-trained net's low chi can be cross-checked as evidence-backed.
het_by_eq = {}
chi_by_eq = {}
with torch.no_grad():
    for eq in EQUATIONS:
        y_e, A_e = eq["target_and_features"](_eval_derivs)
        het_by_eq[eq["name"]] = {k: v.cpu().numpy() for k, v in
                                 het_per_window(y_e, A_e, _eval_mask,
                                                ridge=EPS).items()}
        out_e = _chi_term(y_e, A_e)
        chi_by_eq[eq["name"]] = {k: v.cpu().numpy()
                                 for k, v in out_e.items()}
        chi_by_eq[eq["name"]]["anch_w"] = (
            _anchor_weights(out_e["score"]).cpu().numpy()
            if ANCH_WEIGHT != "none"
            else np.ones_like(chi_by_eq[eq["name"]]["score"]))

# Grid residuals -- use the discovered (mean OLS) coefs.
grid_d = evaluate_on_grid(net, T_obs)
res_by_eq = grid_equation_residuals(grid_d, theta_means_final)

print(f"\n========== DP CHI2 PINN "
      f"(float64={CHI_FLOAT64}, floor={not CHI_NO_FLOOR}, "
      f"anchor={ANCH_NORM}/{ANCH_WEIGHT}+{ANCH_W_BASE:g}@{W_ANCH:g}, "
      f"theta={'trainable' if THETA_TRAINABLE else 'oracle'}) ==========")
print(f"best coef err (used)    = {best['err']:.3e}")
print(f"  loss at that point    = {best['loss']:.3e}")
print(f"last loss               = {final.item():.3e}")
print(f"training time           = {elapsed:.1f}s")
print(f"train/test split        = {TRAIN_FRAC:.2f}  (t_split={t_split:.3f}, n_train={n_train}/{N_GRID})")
print(f"rel L2 train (th1, th2) = {rel_l2_th1_train:.4e}, {rel_l2_th2_train:.4e}")
print(f"rel L2 test  (th1, th2) = {rel_l2_th1_test:.4e}, {rel_l2_th2_test:.4e}")
for eq in EQUATIONS:
    nm = eq["name"]
    print(f"  {nm:>4}: OLS mean  = {theta_means_final[nm].round(4).tolist()}   "
          f"true = {eq['theta_true'].tolist()}")
    print(f"        mean|res| = {float(np.abs(res_by_eq[nm]).mean()):.3e}")
    c = chi_by_eq[nm]
    rss_ratio = float(c["rss"] / c["yy"])
    print(f"        chi score = {c['score'].round(6).tolist()}  "
          f"theta_glob = {c['theta'].round(4).tolist()}  "
          f"rss/yy = {rss_ratio:.3e}  max|s_end| = {float(c['s_end'].max()):.3e}")
    print(f"        anchor w  = {c['anch_w'].round(4).tolist()}")
    if theta_params is not None:
        print(f"        theta_TRN = "
              f"{theta_params[nm].detach().cpu().numpy().round(4).tolist()}"
              f"   (trainable physics coefs)")
    h = het_by_eq[nm]
    print(f"        het score = {h['score'].round(6).tolist()}  "
          f"se_rel = {h['se_rel'].round(4).tolist()}  "
          f"n_valid = {h['n_valid'].astype(int).tolist()}")

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
    loss_history_iter=np.array(hist["iter"], dtype=np.int32),
    loss_history_total=np.array(hist["tot"], dtype=np.float32),
    loss_history_phys=np.array(hist["phys"], dtype=np.float32),
    loss_history_phys_raw=np.array(hist["phys_raw"], dtype=np.float32),
    loss_history_wlast=np.array(hist["wlast"], dtype=np.float32),
    loss_history_cv=np.array(hist["cv"], dtype=np.float32),
    loss_history_anch=np.array(hist["anch"], dtype=np.float32),
    loss_history_mcv=np.array(hist["mcv"], dtype=np.float32),
    loss_history_ic=np.array(hist["ic"], dtype=np.float32),
    # Alias: the cv slot above keeps compare_pinns' shared loss panel
    # working; this key documents what the slot actually carried.
    loss_history_chi=np.array(hist["cv"], dtype=np.float32),
)
for nm in EQ_NAMES:
    save_kwargs[f"theta_{nm}_per_window"] = theta_per_window[nm]
    save_kwargs[f"residual_{nm}_grid"]    = res_by_eq[nm]
    save_kwargs[f"phys_scale_{nm}"]       = np.float32(PHYS_SCALE[nm])
    save_kwargs[f"het_{nm}_score"]        = het_by_eq[nm]["score"]
    save_kwargs[f"het_{nm}_se_rel"]       = het_by_eq[nm]["se_rel"]
    save_kwargs[f"het_{nm}_n_valid"]      = het_by_eq[nm]["n_valid"]
    save_kwargs[f"chi_{nm}_score"]        = chi_by_eq[nm]["score"]
    save_kwargs[f"chi_{nm}_theta_global"] = chi_by_eq[nm]["theta"]
    save_kwargs[f"chi_{nm}_D"]            = chi_by_eq[nm]["D"]
    save_kwargs[f"chi_{nm}_rss"]          = np.float64(chi_by_eq[nm]["rss"])
    save_kwargs[f"chi_{nm}_yy"]           = np.float64(chi_by_eq[nm]["yy"])
    save_kwargs[f"chi_{nm}_s_end"]        = chi_by_eq[nm]["s_end"]
    save_kwargs[f"chi_{nm}_anch_w"]       = chi_by_eq[nm]["anch_w"]
    save_kwargs[f"chi_{nm}_path"]         = chi_by_eq[nm]["path"].astype(np.float32)
save_kwargs["chi_float64"]  = np.bool_(CHI_FLOAT64)
save_kwargs["chi_no_floor"] = np.bool_(CHI_NO_FLOOR)
save_kwargs["w_anch"]       = np.float32(W_ANCH)
save_kwargs["anch_norm"]    = np.str_(ANCH_NORM)
save_kwargs["anch_weight"]  = np.str_(ANCH_WEIGHT)
save_kwargs["anch_coupling"] = np.str_(ANCH_COUPLING)
save_kwargs["anch_w_base"]  = np.float32(ANCH_W_BASE)
save_kwargs["theta_trainable"] = np.bool_(THETA_TRAINABLE)
if theta_params is not None:
    for nm in EQ_NAMES:
        save_kwargs[f"theta_{nm}_trained"] = \
            theta_params[nm].detach().cpu().numpy()
save_kwargs["w_phys"]       = np.float32(W_PHYS)
save_kwargs["w_cv"]         = np.float32(W_CV)
save_kwargs["w_ic"]         = np.float32(W_IC)
out_path = f"dp_pinn_chi{CHI_SUFFIX}.npz"
np.savez(out_path, **save_kwargs)
print(f"saved -> {out_path}")
