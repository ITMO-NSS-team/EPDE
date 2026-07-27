"""
CV-of-OLS NS PINN -- three candidate equations.

Same physics term as the baseline (TRUE coefs in the residual), plus a
CV-anchored consistency penalty on the per-window OLS recovery of each
equation's coefficients:

    L_phys = sum_eq mean( (target_eq - features_eq @ theta_true_eq)^2 )
    L_cv   = sum_eq CV-form( OLS_recovery(theta_per_window_eq), theta_true_eq )
    L_data = mean( (u_pred - u_data)^2 + (v_pred - v_data)^2 )
    L      = W_PHYS * L_phys + W_CV * L_cv + W_DATA * L_data

OLS gradient through `torch.linalg.solve` can be ill-conditioned at small
network amplitudes -- this is the variant that exhibits the degenerate
basin we discussed for the Duffing case. Compare against cv-trainable
(which sidesteps the solve gradient).
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
    network_derivatives_ns,
)

torch.set_default_dtype(torch.float32)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================ settings
DATA_PATH  = "./ns_data.npz"

HIDDEN     = [40] * 8
N_COLL     = 10000
ADAM_ITERS = 10000
ADAM_LR    = 1e-3
LBFGS_MAX  = 20000
W_PHYS, W_CV, W_DATA = 1.0, 1.0, 1.0
SEED       = 0

N_WIN      = 10
WIN_FRAC   = 0.5
EPS = 1e-8

# CV regularizer form applied per equation:
#   "cv2"          : sum_j var_i(theta_ji) / mean_i(theta_ji)^2
#   "anchored_mse" : sum_j mean_i((theta_ji - theta_j_true)^2) / theta_j_true^2
CV_FORM = "anchored_mse"


# ============================================================ load data
_data = np.load(DATA_PATH)
t_grid = _data["t"].astype(np.float32)
x_grid = _data["x"].astype(np.float32)
y_grid = _data["y"].astype(np.float32)
u_data = _data["u"].astype(np.float32)
v_data = _data["v"].astype(np.float32)
p_data = _data["p"].astype(np.float32)
LAMBDA_1_TRUE = float(_data["lambda_1"])
LAMBDA_2_TRUE = float(_data["lambda_2"])

Nt, Ny, Nx = u_data.shape
T_MIN, T_MAX = float(t_grid[0]), float(t_grid[-1])
X_MIN, X_MAX = float(x_grid[0]), float(x_grid[-1])
Y_MIN, Y_MAX = float(y_grid[0]), float(y_grid[-1])

THETA_TRUE = {
    eq["name"]: torch.tensor(eq["theta_true"], device=device, dtype=torch.float32)
    for eq in EQUATIONS
}

TT, YY, XX = np.meshgrid(t_grid, y_grid, x_grid, indexing="ij")
X_obs_np = np.stack([XX.ravel(), YY.ravel(), TT.ravel()], axis=1)
X_obs = torch.tensor(X_obs_np, device=device)
uv_obs = torch.tensor(np.stack([u_data.ravel(), v_data.ravel()], axis=1),
                     device=device)

t_lo, t_hi = make_windows(T_MIN, T_MAX, N_WIN, WIN_FRAC)
T_LO = torch.tensor(t_lo, device=device).unsqueeze(1)
T_HI = torch.tensor(t_hi, device=device).unsqueeze(1)


# ============================================================ network
class NormalizedMLP(nn.Module):
    def __init__(self, hidden=HIDDEN):
        super().__init__()
        lo = torch.tensor([X_MIN, Y_MIN, T_MIN], dtype=torch.float32)
        hi = torch.tensor([X_MAX, Y_MAX, T_MAX], dtype=torch.float32)
        self.register_buffer("lo", lo)
        self.register_buffer("hi", hi)
        layers = [3] + list(hidden) + [3]
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

    def forward(self, X):
        Xn = 2.0 * (X - self.lo) / (self.hi - self.lo) - 1.0
        return self.net(Xn)


# ============================================================ samplers
def sample_collocation(g):
    x = torch.rand(N_COLL, generator=g, device=device) * (X_MAX - X_MIN) + X_MIN
    y = torch.rand(N_COLL, generator=g, device=device) * (Y_MAX - Y_MIN) + Y_MIN
    t = torch.rand(N_COLL, generator=g, device=device) * (T_MAX - T_MIN) + T_MIN
    return torch.stack([x, y, t], dim=1)


# ============================================================ physics + CV
def cv_form_value(theta_per_window, theta_true):
    """Return scalar CV penalty over one equation's per-window theta."""
    if CV_FORM == "cv2":
        var = theta_per_window.var(dim=0, unbiased=True)
        mu = theta_per_window.mean(dim=0)
        per_coef = var / (mu ** 2 + EPS)
    elif CV_FORM == "anchored_mse":
        sq = (theta_per_window - theta_true.unsqueeze(0)) ** 2
        per_coef = sq.mean(dim=0) / (theta_true ** 2 + EPS)
    else:
        raise ValueError(f"unknown CV_FORM: {CV_FORM}")
    return per_coef.sum()


def physics_and_cv_loss(net, X_coll):
    """One autograd pass -> per-equation residual + per-equation CV penalty."""
    results, derivs, _mask = compute_equation_thetas(net, X_coll, T_LO, T_HI, ridge=EPS)
    l_phys_total = 0.0
    l_cv_total = 0.0
    per_eq_residual = {}
    theta_means = {}
    for eq in EQUATIONS:
        nm = eq["name"]
        theta_pw, theta_mean = results[nm]
        # Standard residual using TRUE coefs (gradient-bearing, well-conditioned).
        y, A = eq["target_and_features"](derivs)
        r = y - A @ THETA_TRUE[nm]
        l_phys_eq = (r ** 2).mean()
        l_phys_total = l_phys_total + l_phys_eq
        per_eq_residual[nm] = l_phys_eq
        # CV penalty on the OLS recovery (gradient through linalg.solve).
        l_cv_total = l_cv_total + cv_form_value(theta_pw, THETA_TRUE[nm])
        theta_means[nm] = theta_mean
    return l_phys_total, l_cv_total, per_eq_residual, theta_means


def data_loss(net):
    uv_pred = net(X_obs)[:, :2]
    return ((uv_pred - uv_obs) ** 2).mean()


# ============================================================ train
torch.manual_seed(SEED); np.random.seed(SEED)
g = torch.Generator(device=device).manual_seed(SEED)

net = NormalizedMLP().to(device)
opt = torch.optim.Adam(net.parameters(), lr=ADAM_LR)

t0 = time.time()
for it in range(ADAM_ITERS):
    opt.zero_grad()
    X_coll = sample_collocation(g)
    l_phys, l_cv, per_eq, theta_means = physics_and_cv_loss(net, X_coll)
    l_data = data_loss(net)
    loss = W_PHYS * l_phys + W_CV * l_cv + W_DATA * l_data
    loss.backward()
    opt.step()
    if it % 1000 == 0:
        eq_msgs = "  ".join(f"{nm}={per_eq[nm].item():.2e}" for nm in EQ_NAMES)
        print(f"[adam {it:5d}] tot={loss.item():.3e}  "
              f"cv={l_cv.item():.3e}  data={l_data.item():.3e}  | "
              + eq_msgs)

# L-BFGS with fixed collocation
X_coll_fix = sample_collocation(g).detach()
lbfgs = torch.optim.LBFGS(net.parameters(),
    max_iter=LBFGS_MAX, max_eval=LBFGS_MAX,
    tolerance_grad=1e-8, tolerance_change=1e-10,
    history_size=50, line_search_fn="strong_wolfe")


def closure():
    lbfgs.zero_grad()
    l_phys, l_cv, _, _ = physics_and_cv_loss(net, X_coll_fix)
    l_data = data_loss(net)
    loss = W_PHYS * l_phys + W_CV * l_cv + W_DATA * l_data
    loss.backward()
    return loss


final = lbfgs.step(closure)
elapsed = time.time() - t0


# ============================================================ evaluate
with torch.no_grad():
    uvp_pred = net(X_obs).cpu().numpy()
u_pred = uvp_pred[:, 0].reshape(Nt, Ny, Nx)
v_pred = uvp_pred[:, 1].reshape(Nt, Ny, Nx)
p_pred = uvp_pred[:, 2].reshape(Nt, Ny, Nx)

rel_l2_u = float(np.linalg.norm(u_pred - u_data) / np.linalg.norm(u_data))
rel_l2_v = float(np.linalg.norm(v_pred - v_data) / np.linalg.norm(v_data))

# Post-hoc per-window OLS on the SAME eval pool the other scripts use.
g_eval = torch.Generator(device=device).manual_seed(SEED + 999)
X_eval = sample_collocation(g_eval)
ols_results, _, _ = compute_equation_thetas(net, X_eval, T_LO, T_HI, ridge=EPS)
theta_per_window = {nm: ols_results[nm][0].detach().cpu().numpy() for nm in EQ_NAMES}
theta_means_final = {nm: ols_results[nm][1].detach().cpu().numpy() for nm in EQ_NAMES}

# Grid residuals -- use the discovered (mean OLS) coefs.
grid_d = evaluate_on_grid(net, X_obs)
res_by_eq = grid_equation_residuals(grid_d, theta_means_final)
res_by_eq_grid = {nm: res_by_eq[nm].reshape(Nt, Ny, Nx) for nm in EQ_NAMES}

print(f"\n========== NS CV-OLS PINN ==========")
print(f"final loss        = {final.item():.3e}")
print(f"training time     = {elapsed:.1f}s")
print(f"rel L2 (u, v)     = {rel_l2_u:.4e}, {rel_l2_v:.4e}")
for eq in EQUATIONS:
    nm = eq["name"]
    print(f"  {nm:>4}: OLS mean  = {theta_means_final[nm].round(4).tolist()}   "
          f"true = {eq['theta_true'].tolist()}")
    print(f"        mean|res| = {float(np.abs(res_by_eq_grid[nm]).mean()):.3e}")

save_kwargs = dict(
    t=t_grid, x=x_grid, y=y_grid,
    u_data=u_data, v_data=v_data, p_data=p_data,
    u_pred=u_pred, v_pred=v_pred, p_pred=p_pred,
    lambda_1_true=LAMBDA_1_TRUE, lambda_2_true=LAMBDA_2_TRUE,
    training_time=elapsed,
)
for nm in EQ_NAMES:
    save_kwargs[f"theta_{nm}_per_window"] = theta_per_window[nm]
    save_kwargs[f"residual_{nm}_grid"]    = res_by_eq_grid[nm]
np.savez("ns_pinn_cv_ols.npz", **save_kwargs)
