"""
Baseline double pendulum PINN -- two candidate equations with TRUE coefficients.

The candidate equation form (`EQUATIONS` in cv_metric.py) is the hypothesis
to verify; its coefficient VALUES are known and used directly here:

    L_phys_eq = mean( (target_eq - features_eq @ theta_true_eq)^2 )
    L_phys    = sum_eq L_phys_eq
    L_data    = mean( (theta1_pred - theta1_data)^2 + (theta2_pred - theta2_data)^2 )
    L         = W_PHYS * L_phys + W_DATA * L_data

No trainable coefficients -- this is the standard PINN reference against
which the CV variants are compared. The post-hoc per-window OLS recovery
on the trained network is saved as a diagnostic ("does OLS on the
network's output reproduce what we know is right?"), but it does not
affect training.

Saves `dp_pinn_baseline.npz` with:
- field predictions on the full grid
- per-equation true coefs broadcast to (N_WIN, K_eq) as the "headline" theta
- post-hoc per-window OLS theta on the full data grid
- per-equation grid residuals using TRUE coefs

Physics + data losses are computed on the TRAIN slice of the data grid
(t in [0, TRAIN_FRAC * T_MAX]) each iteration -- no random collocation
sampling. The held-out tail is a pure extrapolation test for the network
-- it never enters the loss. Windows still span the full [0, T]; post-hoc
OLS runs on the full grid so per-window coefficients can be inspected
across both train and test regions.
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
W_PHYS, W_DATA = 1.0, 1.0
SEED       = 0

N_WIN      = 20
WIN_FRAC   = 0.5
TRAIN_FRAC = 0.8       # temporal split: train on first TRAIN_FRAC of trajectory
CIRCULAR_WINDOWS = False  # True -> treat t-axis as a loop (windows wrap at T_MAX)
EPS = 1e-8


# ============================================================ load data
_data = np.load(DATA_PATH)
t_grid = _data["t"].astype(np.float32)
th1_data = _data["theta1"].astype(np.float32)
th2_data = _data["theta2"].astype(np.float32)
w1_data  = _data["omega1"].astype(np.float32)
w2_data  = _data["omega2"].astype(np.float32)
G_TRUE   = float(_data["g"])

set_gravity(G_TRUE)   # refresh theta_true vectors with the .npz's g

T_MIN, T_MAX = float(t_grid[0]), float(t_grid[-1])
N_GRID = len(t_grid)

THETA_TRUE = {
    eq["name"]: torch.tensor(eq["theta_true"], device=device, dtype=torch.float32)
    for eq in EQUATIONS
}

T_obs   = torch.tensor(t_grid.reshape(-1, 1), device=device)
th_obs  = torch.tensor(np.stack([th1_data, th2_data], axis=1), device=device)

# Temporal train/test split: train on [0, TRAIN_FRAC*T_MAX], test on the
# held-out tail. Windows still span [0, T] (see make_windows below);
# post-hoc OLS sweeps the full grid for diagnostics.
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


# ============================================================ losses
def physics_loss(net, T_pts):
    """Sum of per-equation residual losses using TRUE coefficients.

    Each equation's residual is normalized by max(|y|) so the gravity-
    dominated target scale (~9.81) doesn't drown out the smaller coupling-
    coef contributions: loss becomes relative error, equation-agnostic.
    """
    derivs = network_derivatives_dp(net, T_pts)
    total = 0.0
    per_eq = {}
    for eq in EQUATIONS:
        y, A = eq["target_and_features"](derivs)
        scale_eq = y.detach().abs().max() + EPS
        r = (y - A @ THETA_TRUE[eq["name"]]) / scale_eq
        l = (r ** 2).mean()
        per_eq[eq["name"]] = l
        total = total + l
    return total, per_eq


def data_loss(net):
    th_pred = net(T_train)
    return ((th_pred - th_train) ** 2).mean()


# ============================================================ train
torch.manual_seed(SEED); np.random.seed(SEED)

net = MLP().to(device)
opt = torch.optim.Adam(net.parameters(), lr=ADAM_LR)

# Track best checkpoint by RELATIVE L1 coefficient error across all
# equations: sum_eq sum_j |theta_mean_j - theta_true_j| / |theta_true_j|.
# Baseline doesn't compute OLS in the training loop, so we sample at the
# log frequency (every 1000 iter) -- cheap enough.
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
    l_phys, per_eq = physics_loss(net, T_train)
    l_data = data_loss(net)
    loss = W_PHYS * l_phys + W_DATA * l_data
    loss.backward()
    opt.step()
    cur = loss.item()
    if it % 1000 == 0:
        # Side OLS solve to measure relative coef error -- baseline doesn't
        # do this in the loss, so we add it here only at log steps.
        ols_results, _, _ = compute_equation_thetas(
            net, T_train, T_LO, T_HI, ridge=EPS, period=PERIOD)
        theta_means_ols = {nm: ols_results[nm][1] for nm in EQ_NAMES}
        err = _coef_err_sum(theta_means_ols)
        if err < best["err"]:
            best["err"] = err
            best["loss"] = cur
            best["state"] = {k: v.detach().clone() for k, v in net.state_dict().items()}
        hist["iter"].append(it)
        hist["tot"].append(cur)
        hist["phys"].append(l_phys.item())
        hist["cv"].append(0.0)              # baseline has no CV term
        hist["data"].append(l_data.item())
        eq_msgs = "  ".join(f"{nm}={per_eq[nm].item():.2e}" for nm in EQ_NAMES)
        print(f"[adam {it:5d}] tot={loss.item():.3e}  data={l_data.item():.3e}  "
              f"err={err:.3e}  "
              + eq_msgs)

# L-BFGS on the same full-grid pool
lbfgs = torch.optim.LBFGS(net.parameters(),
    max_iter=LBFGS_MAX, max_eval=LBFGS_MAX,
    tolerance_grad=1e-8, tolerance_change=1e-10,
    history_size=50, line_search_fn="strong_wolfe")


def closure():
    lbfgs.zero_grad()
    l_phys, _ = physics_loss(net, T_train)
    l_data = data_loss(net)
    loss = W_PHYS * l_phys + W_DATA * l_data
    loss.backward()
    ols_results, _, _ = compute_equation_thetas(
        net, T_train, T_LO, T_HI, ridge=EPS, period=PERIOD)
    theta_means_ols = {nm: ols_results[nm][1] for nm in EQ_NAMES}
    err = _coef_err_sum(theta_means_ols)
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

# Headline theta per equation = TRUE coefs broadcast to (N_WIN, K_eq).
theta_true_broadcast = {
    eq["name"]: np.tile(eq["theta_true"].astype(np.float32)[None, :], (N_WIN, 1))
    for eq in EQUATIONS
}

# Diagnostic: post-hoc OLS recovery on the TRAIN slice only -- using the
# full grid here would let the network's unconstrained extrapolation in
# (t_split, T_MAX] pollute the right-edge windows' design matrices.
ols_results, _, _ = compute_equation_thetas(net, T_train, T_LO, T_HI, ridge=EPS, period=PERIOD)
theta_per_window_ols = {nm: ols_results[nm][0].detach().cpu().numpy()
                        for nm in EQ_NAMES}

# Grid residuals using TRUE coefs (matches the loss).
grid_d = evaluate_on_grid(net, T_obs)
theta_true_np = {eq["name"]: eq["theta_true"] for eq in EQUATIONS}
res_by_eq = grid_equation_residuals(grid_d, theta_true_np)

print(f"\n========== DP BASELINE PINN (TRUE coefs in residual) ==========")
print(f"best coef err (used)    = {best['err']:.3e}")
print(f"  loss at that point    = {best['loss']:.3e}")
print(f"last loss               = {final.item():.3e}")
print(f"training time           = {elapsed:.1f}s")
print(f"train/test split        = {TRAIN_FRAC:.2f}  (t_split={t_split:.3f}, n_train={n_train}/{N_GRID})")
print(f"rel L2 train (th1, th2) = {rel_l2_th1_train:.4e}, {rel_l2_th2_train:.4e}")
print(f"rel L2 test  (th1, th2) = {rel_l2_th1_test:.4e}, {rel_l2_th2_test:.4e}")
for eq in EQUATIONS:
    nm = eq["name"]
    ols_mean = theta_per_window_ols[nm].mean(axis=0)
    print(f"  {nm:>4}: true     = {eq['theta_true'].tolist()}")
    print(f"        OLS mean  = {ols_mean.round(4).tolist()}")
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
    save_kwargs[f"theta_{nm}_per_window"]     = theta_true_broadcast[nm]
    save_kwargs[f"theta_{nm}_per_window_ols"] = theta_per_window_ols[nm]
    save_kwargs[f"residual_{nm}_grid"]        = res_by_eq[nm]
out_path = f"dp_pinn_baseline{'_circular' if CIRCULAR_WINDOWS else ''}.npz"
np.savez(out_path, **save_kwargs)
print(f"saved -> {out_path}")
