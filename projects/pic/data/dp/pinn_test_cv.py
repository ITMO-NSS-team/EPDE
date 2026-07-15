"""
CV-of-OLS DP PINN -- two candidate equations.

Same physics term as the baseline (TRUE coefs in the residual), plus a
CV-anchored consistency penalty on the per-window OLS recovery of each
equation's coefficients:

    L_phys = sum_eq mean( (target_eq - features_eq @ theta_true_eq)^2 )
    L_cv   = sum_eq CV-form( OLS_recovery(theta_per_window_eq), theta_true_eq )
    L_data = mean( (theta1_pred - theta1_data)^2 + (theta2_pred - theta2_data)^2 )
    L      = W_PHYS * L_phys + W_CV * L_cv + W_DATA * L_data

OLS gradient through `torch.linalg.solve` can be ill-conditioned at small
network amplitudes -- this is the variant that exhibits the degenerate
basin we discussed for the Duffing case. Compare against cv-trainable
(which sidesteps the solve gradient).

Physics + CV + data losses are computed on the TRAIN slice of the data
grid (t in [0, TRAIN_FRAC * T_MAX]) each iteration. Windows still span
the full [0, T], but the OLS mask in the CV penalty is built from
TRAIN-ONLY points -- the held-out tail never enters the loss. Post-hoc
OLS at evaluation runs on the full grid so per-window coefficients can
be inspected across both train and test regions.
"""

import time
import numpy as np
import torch
import torch.nn as nn

from cv_metric import (
    EQUATIONS,
    EQ_NAMES,
    compute_equation_thetas,
    evaluate_on_grid,
    grid_equation_residuals,
    make_windows,
    make_windows_circular,
    network_derivatives_dp,
    set_gravity,
)

torch.set_default_dtype(torch.float32)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================ settings
DATA_PATH  = "./dp.npz"

HIDDEN     = [64] * 5
ADAM_ITERS = 200000
ADAM_LR    = 1e-3
LBFGS_MAX  = 20000
W_PHYS, W_CV, W_DATA = 0.0, 1.0, 1.0
SEED       = 0

N_WIN      = 30
WIN_FRAC   = 0.5
TRAIN_FRAC = 0.8       # temporal split: train on first TRAIN_FRAC of trajectory
CIRCULAR_WINDOWS = False  # True -> treat t-axis as a loop (windows wrap at T_MAX)
EPS = 1e-12

# CV regularizer form applied per equation:
#   "cv2"          : sum_j var_i(theta_ji) / mean_i(theta_ji)^2
#   "anchored_mse" : sum_j mean_i((theta_ji - theta_j_true)^2) / theta_j_true^2
CV_FORM = "anchored_mse"


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

# Temporal train/test split: only T_train enters the physics + CV + data
# loss. Windows still span [0, T] (see make_windows below); the CV mask
# is built from T_train so the OLS within each window only sees the train
# portion. Post-hoc OLS sweeps the full grid for diagnostics.
n_train = int(np.ceil(TRAIN_FRAC * N_GRID))
t_split = float(t_grid[n_train - 1])
T_train  = T_obs[:n_train]
th_train = th_obs[:n_train]
T_test   = T_obs[n_train:]
th_test  = th_obs[n_train:]

if CIRCULAR_WINDOWS:
    t_lo, t_hi = make_windows_circular(T_MIN, T_MAX, N_WIN, WIN_FRAC)
    PERIOD = float(T_MAX - T_MIN)
else:
    t_lo, t_hi = make_windows(T_MIN, T_MAX, N_WIN, WIN_FRAC)
    PERIOD = None
T_LO = torch.tensor(t_lo, device=device).unsqueeze(1)
T_HI = torch.tensor(t_hi, device=device).unsqueeze(1)


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


# ============================================================ physics + CV
def cv_form_value(theta_per_window, theta_true):
    """Return scalar CV penalty over one equation's per-window theta."""
    if CV_FORM == "cv2":
        var = theta_per_window.var(dim=0, unbiased=True)
        mu = theta_per_window.mean(dim=0)
        per_coef = var / (mu ** 2 + EPS)
    elif CV_FORM == "anchored_mse":
        sq = (theta_per_window - theta_true.unsqueeze(0)) ** 2
        per_coef = sq.mean(dim=0) / (theta_true ** 2)
    else:
        raise ValueError(f"unknown CV_FORM: {CV_FORM}")
    return per_coef.sum()


def physics_and_cv_loss(net, T_coll):
    """One autograd pass -> per-equation residual + per-equation CV penalty.

    Physics residual is normalized by max(|y|) per equation so the gravity-
    dominated absolute scale doesn't drown out the smaller coupling-coef
    contributions across the two equations.
    """
    results, derivs, _mask = compute_equation_thetas(net, T_coll, T_LO, T_HI, ridge=EPS, period=PERIOD)
    l_phys_total = 0.0
    l_cv_total = 0.0
    per_eq_residual = {}
    theta_means = {}
    for eq in EQUATIONS:
        nm = eq["name"]
        theta_pw, theta_mean = results[nm]
        # Scale-normalized residual using TRUE coefs.
        y, A = eq["target_and_features"](derivs)
        scale_eq = y.detach().abs().max() + EPS
        r = (y - A @ THETA_TRUE[nm]) / scale_eq
        l_phys_eq = (r ** 2).mean()
        l_phys_total = l_phys_total + l_phys_eq
        per_eq_residual[nm] = l_phys_eq
        # CV penalty on the OLS recovery (gradient through linalg.solve).
        l_cv_total = l_cv_total + cv_form_value(theta_pw, THETA_TRUE[nm])
        theta_means[nm] = theta_mean
    return l_phys_total, l_cv_total, per_eq_residual, theta_means


def data_loss(net):
    th_pred = net(T_train)
    return ((th_pred - th_train) ** 2).mean()


# ============================================================ train
torch.manual_seed(SEED); np.random.seed(SEED)

net = MLP().to(device)
opt = torch.optim.Adam(net.parameters(), lr=ADAM_LR)

# Track best checkpoint by RELATIVE L1 coefficient error across all
# equations: sum_eq sum_j |theta_mean_j - theta_true_j| / |theta_true_j|.
best = {"err": float("inf"), "loss": float("inf"), "state": None}


def _coef_err_sum(theta_means_dict):
    return sum(
        ((theta_means_dict[nm].detach() - THETA_TRUE[nm]).abs()
         / THETA_TRUE[nm].abs()).sum().item()
        for nm in EQ_NAMES
    )


# Adam loss history (sampled at the print frequency).
hist = {"iter": [], "tot": [], "phys": [], "cv": [], "data": []}

t0 = time.time()
for it in range(ADAM_ITERS):
    opt.zero_grad()
    l_phys, l_cv, per_eq, theta_means = physics_and_cv_loss(net, T_train)
    l_data = data_loss(net)
    loss = W_PHYS * l_phys + W_CV * l_cv + W_DATA * l_data
    loss.backward()
    opt.step()
    cur = loss.item()
    err = _coef_err_sum(theta_means)
    if err < best["err"]:
        best["err"] = err
        best["loss"] = cur
        best["state"] = {k: v.detach().clone() for k, v in net.state_dict().items()}
    if it % 1000 == 0:
        hist["iter"].append(it)
        hist["tot"].append(cur)
        hist["phys"].append(l_phys.item())
        hist["cv"].append(l_cv.item())
        hist["data"].append(l_data.item())
        phys_msgs = "  ".join(f"{nm}={per_eq[nm].item():.2e}" for nm in EQ_NAMES)
        err_msgs = "  ".join(
            f"{nm}={(theta_means[nm] - THETA_TRUE[nm]).detach().cpu().numpy().round(4).tolist()}"
            for nm in EQ_NAMES)
        print(f"[adam {it:5d}] tot={loss.item():.3e}  cv={l_cv.item():.3e}  "
              f"data={l_data.item():.3e}  | phys: {phys_msgs}  | err: {err_msgs}")

# L-BFGS on the same full-grid pool
lbfgs = torch.optim.LBFGS(net.parameters(),
    max_iter=LBFGS_MAX, max_eval=LBFGS_MAX,
    tolerance_grad=1e-8, tolerance_change=1e-10,
    history_size=50, line_search_fn="strong_wolfe")


def closure():
    lbfgs.zero_grad()
    l_phys, l_cv, _, theta_means = physics_and_cv_loss(net, T_train)
    l_data = data_loss(net)
    loss = W_PHYS * l_phys + W_CV * l_cv + W_DATA * l_data
    loss.backward()
    err = _coef_err_sum(theta_means)
    if err < best["err"]:
        best["err"] = err
        best["loss"] = loss.item()
        best["state"] = {k: v.detach().clone() for k, v in net.state_dict().items()}
    return loss


final = lbfgs.step(closure)
elapsed = time.time() - t0

# Restore the lowest-loss checkpoint we saw during training.
net.load_state_dict(best["state"])


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
ols_results, _, _ = compute_equation_thetas(net, T_train, T_LO, T_HI, ridge=EPS, period=PERIOD)
theta_per_window = {nm: ols_results[nm][0].detach().cpu().numpy() for nm in EQ_NAMES}
theta_means_final = {nm: ols_results[nm][1].detach().cpu().numpy() for nm in EQ_NAMES}

# Grid residuals -- use the discovered (mean OLS) coefs.
grid_d = evaluate_on_grid(net, T_obs)
res_by_eq = grid_equation_residuals(grid_d, theta_means_final)

print(f"\n========== DP CV-OLS PINN ==========")
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
    loss_history_cv=np.array(hist["cv"], dtype=np.float32),
    loss_history_data=np.array(hist["data"], dtype=np.float32),
)
for nm in EQ_NAMES:
    save_kwargs[f"theta_{nm}_per_window"] = theta_per_window[nm]
    save_kwargs[f"residual_{nm}_grid"]    = res_by_eq[nm]
out_path = f"dp_pinn_cv_ols{'_circular' if CIRCULAR_WINDOWS else ''}.npz"
np.savez(out_path, **save_kwargs)
print(f"saved -> {out_path}")
