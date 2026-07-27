"""
CV DP PINN with TRAINABLE per-window theta -- two candidate equations.

For each equation eq with K_eq=3 features, theta_eq_per_window is an
nn.Parameter of shape (N_WIN, K_eq). At each collocation point we use
the theta of the nearest-center time window:

    L_phys_eq = mean_k ( (target_eq(t_k) - features_eq(t_k) @ theta_{w(k)})^2 )
    L_cv_eq   = anchored_mse( theta_per_window_eq, theta_true_eq )
    L_phys    = sum_eq L_phys_eq
    L_cv      = sum_eq L_cv_eq
    L_data    = mean( (theta1_pred - theta1_data)^2 + (theta2_pred - theta2_data)^2 )
    L         = W_PHYS * L_phys + W_CV * L_cv + W_DATA * L_data

No linalg.solve in the training loop. Per-point physics gradient on the
net (standard PINN signal) and analytic gradient on theta_per_window
(no inverse). The anchored CV term pulls theta toward truth.

Physics + CV + data losses are computed on the TRAIN slice of the data
grid (t in [0, TRAIN_FRAC * T_MAX]) each iteration. Window assignment
for the pointwise residual is precomputed once on the train slice
(WIN_IDX_TRAIN). Each train point's residual updates the theta of its
nearest-center window. The held-out tail is a pure extrapolation test
-- it never enters the loss. Windows still span the full [0, T]; the
test-region windows' thetas only receive gradient from any train points
that happen to be mapped to them by nearest-center.

`theta_per_window` is initialized at the KNOWN TRUE coefficients (broadcast
across windows), so this run is testing whether the joint physics + CV
loss landscape keeps theta at truth -- not whether anchored CV can pull
it there from zeros. Useful for validating the loss before testing
discovery from a neutral start.
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
THETA_LR   = 1e-2
LBFGS_MAX  = 20000
W_PHYS, W_CV, W_DATA = 0.0, 1.0, 1.0
SEED       = 0

N_WIN      = 30
WIN_FRAC   = 0.5
TRAIN_FRAC = 0.8       # temporal split: train on first TRAIN_FRAC of trajectory
CIRCULAR_WINDOWS = True  # True -> treat t-axis as a loop (windows + nearest-center wrap at T_MAX)
EPS = 1e-8

CV_FORM = "anchored_mse"   # "cv2" | "anchored_mse"

# Trainable theta initialization per equation. We start at the KNOWN true
# coefficients (populated after `set_gravity(G_TRUE)` below) -- this tests
# whether physics + CV hold theta at truth rather than whether anchored
# CV can pull it there from a neutral start.


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

# Temporal train/test split: physics + CV + data only on T_train. Windows
# still span [0, T] (see make_windows below); post-hoc OLS sweeps the
# full grid for diagnostics.
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
WIN_CENTERS = torch.tensor(0.5 * (t_lo + t_hi), device=device)


def window_index_for(t_flat):
    """Nearest-center window index for each time. (N,) -> (N,) ints.

    In CIRCULAR_WINDOWS mode the distance wraps at PERIOD, so a point
    near t=0 can be assigned to a window centered near t=T_MAX.
    """
    dists = (t_flat.unsqueeze(1) - WIN_CENTERS.unsqueeze(0)).abs()
    if CIRCULAR_WINDOWS:
        dists = torch.minimum(dists, PERIOD - dists)
    return dists.argmin(dim=1)


# Precompute window index for every observation point (deterministic).
WIN_IDX_OBS = window_index_for(T_obs[:, 0])
WIN_IDX_TRAIN = WIN_IDX_OBS[:n_train]          # only train points enter the loss


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
def cv_form_value(theta_per_window, theta_true):
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


def merged_loss(net, theta_params):
    """Per-point physics residual on the TRAIN slice + CV.

    Uses T_train (first n_train points) and the precomputed WIN_IDX_TRAIN.
    Physics residual normalized by max(|y|) per equation so absolute target
    scale (gravity-dominated) doesn't dominate the loss landscape.
    Returns (l_phys_total, l_cv_total, per_eq_residual_dict, theta_means_dict).
    """
    derivs = network_derivatives_dp(net, T_train)
    win_idx = WIN_IDX_TRAIN                        # (n_train,)

    l_phys_total = 0.0
    l_cv_total = 0.0
    per_eq_residual = {}
    theta_means = {}
    for eq in EQUATIONS:
        nm = eq["name"]
        theta_pw = theta_params[nm]                # (N_WIN, K_eq)
        theta_at_pt = theta_pw[win_idx]            # (N_GRID, K_eq)
        y, A = eq["target_and_features"](derivs)
        # Scale-normalized pointwise residual.
        scale_eq = y.detach().abs().max() + EPS
        r = (y - (A * theta_at_pt).sum(dim=1)) / scale_eq
        l_phys_eq = (r ** 2).mean()
        l_phys_total = l_phys_total + l_phys_eq
        per_eq_residual[nm] = l_phys_eq
        l_cv_total = l_cv_total + cv_form_value(theta_pw, THETA_TRUE[nm])
        theta_means[nm] = theta_pw.mean(dim=0)
    return l_phys_total, l_cv_total, per_eq_residual, theta_means


def data_loss(net):
    th_pred = net(T_train)
    return ((th_pred - th_train) ** 2).mean()


# ============================================================ train
torch.manual_seed(SEED); np.random.seed(SEED)

net = MLP().to(device)

theta_params = {}
for eq in EQUATIONS:
    init = np.tile(eq["theta_true"].astype(np.float32)[None, :], (N_WIN, 1))
    p = nn.Parameter(torch.tensor(init, device=device))
    theta_params[eq["name"]] = p

opt = torch.optim.Adam([
    {"params": net.parameters(), "lr": ADAM_LR},
    {"params": list(theta_params.values()), "lr": THETA_LR},
])

# Track best (net, theta_params) by RELATIVE L1 coefficient error across
# all equations: sum_eq sum_j |theta_mean_j - theta_true_j| / |theta_true_j|.
best = {"err": float("inf"), "loss": float("inf"), "state": None, "theta": None}


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
    l_phys, l_cv, per_eq, theta_means = merged_loss(net, theta_params)
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
        best["theta"] = {nm: theta_params[nm].detach().clone() for nm in EQ_NAMES}
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

# L-BFGS jointly over net + all theta_per_window
lbfgs = torch.optim.LBFGS(
    list(net.parameters()) + list(theta_params.values()),
    max_iter=LBFGS_MAX, max_eval=LBFGS_MAX,
    tolerance_grad=1e-8, tolerance_change=1e-10,
    history_size=50, line_search_fn="strong_wolfe",
)


def closure():
    lbfgs.zero_grad()
    l_phys, l_cv, _, theta_means = merged_loss(net, theta_params)
    l_data = data_loss(net)
    loss = W_PHYS * l_phys + W_CV * l_cv + W_DATA * l_data
    loss.backward()
    err = _coef_err_sum(theta_means)
    if err < best["err"]:
        best["err"] = err
        best["loss"] = loss.item()
        best["state"] = {k: v.detach().clone() for k, v in net.state_dict().items()}
        best["theta"] = {nm: theta_params[nm].detach().clone() for nm in EQ_NAMES}
    return loss


final = lbfgs.step(closure)
elapsed = time.time() - t0

# Restore the lowest-loss (net, theta) we saw during training.
net.load_state_dict(best["state"])
for nm in EQ_NAMES:
    theta_params[nm].data.copy_(best["theta"][nm])


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

# Trained theta_per_window (the headline).
theta_per_window_trained = {nm: theta_params[nm].detach().cpu().numpy()
                            for nm in EQ_NAMES}
theta_mean_trained = {nm: theta_per_window_trained[nm].mean(axis=0)
                      for nm in EQ_NAMES}

# Diagnostic: post-hoc OLS theta on the TRAIN slice only -- using T_obs would
# let the network's unconstrained extrapolation in (t_split, T_MAX] pollute
# the OLS design matrix for windows whose span overlaps the test region.
ols_results, _, _ = compute_equation_thetas(net, T_train, T_LO, T_HI, ridge=EPS, period=PERIOD)
theta_per_window_ols = {nm: ols_results[nm][0].detach().cpu().numpy()
                        for nm in EQ_NAMES}

# Grid residuals using TRAINED theta mean.
grid_d = evaluate_on_grid(net, T_obs)
res_by_eq = grid_equation_residuals(grid_d, theta_mean_trained)

print(f"\n========== DP CV-PINN (trainable theta, 2 equations) ==========")
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
    print(f"  {nm:>4}: trained mean = {theta_mean_trained[nm].round(4).tolist()}")
    print(f"        OLS mean     = {ols_mean.round(4).tolist()}")
    print(f"        true         = {eq['theta_true'].tolist()}")
    print(f"        mean|res|    = {float(np.abs(res_by_eq[nm]).mean()):.3e}")

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
    save_kwargs[f"theta_{nm}_per_window"]     = theta_per_window_trained[nm]
    save_kwargs[f"theta_{nm}_per_window_ols"] = theta_per_window_ols[nm]
    save_kwargs[f"residual_{nm}_grid"]        = res_by_eq[nm]
out_path = f"dp_pinn_cv_trainable{'_circular' if CIRCULAR_WINDOWS else ''}.npz"
np.savez(out_path, **save_kwargs)
print(f"saved -> {out_path}")
