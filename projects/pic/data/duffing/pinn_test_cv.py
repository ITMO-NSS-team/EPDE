"""
CV-of-OLS PINN for the forced Duffing oscillator -- CV ADDED to the
standard ODE residual (not a replacement).

Loss = W_PDE * L_pde + W_CV * L_cv + W_DATA * L_data

where

    L_pde = mean((x_tt + DELTA*x_t + ALPHA*x + BETA*x^3 - GAMMA*cos(OMEGA*t))^2)
    L_cv  = CV-form over sliding-window OLS estimates of (alpha, beta, delta, gamma)

For each time-window W_i, OLS over collocation points in W_i recovers a
4-vector (alpha_i, beta_i, delta_i, gamma_i) such that

    x_tt = -alpha*x - beta*x^3 - delta*x_t + gamma*cos(omega*t)

across that window. The CV-OLS loss aggregates per-coefficient
coefficient-of-variation^2 (or anchored MSE) across windows:

    "cv2":           sum_j  var_i(theta_ji) / mean_i(theta_ji)^2
    "anchored_mse":  sum_j  mean_i((theta_ji - theta_j_true)^2) / theta_j_true^2

Both are scale-invariant per coefficient, so summing across the four
coefficients yields a unitless loss. Choose at the top via CV_FORM.

The PDE residual term uses TRUE coefficients (this is the baseline term);
the CV term acts as an additional consistency regularizer. The discovered
(alpha, beta, delta, gamma) fall out as the mean of the per-window OLS
estimates at convergence -- printed as diagnostics and saved.

To avoid doubling the autograd cost, `cv_ols_loss` computes BOTH terms
from a single forward+backward graph (one call to `first_second_derivs_t`).
"""

import time
import numpy as np
import torch
import torch.nn as nn

from cv_metric import (
    coefs_from_derivs,
    compute_coefs_per_window,
    evaluate_derivatives_on_grid,
    first_second_derivs_t,
    ode_residual_from_grid,
    make_windows,
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
W_PDE, W_CV, W_DATA = 1.0, 1.00, 1.0
SEED       = 0

N_WIN      = 30
WIN_FRAC   = 0.5
EPS = 1e-8

# CV regularizer form:
#   "cv2"          : sum_j  var_i(theta_ji) / mean_i(theta_ji)^2  (no truth needed)
#   "anchored_mse" : sum_j  mean_i((theta_ji - theta_j_true)^2) / theta_j_true^2
CV_FORM = "anchored_mse"


# ============================================================ load data
_data = np.load(DATA_PATH)
t_grid = _data["t"].astype(np.float32)
x_data = _data["x"].astype(np.float32)
ALPHA_TRUE = float(_data["alpha"])
BETA_TRUE  = float(_data["beta"])
DELTA_TRUE = float(_data["delta"])
GAMMA_TRUE = float(_data["gamma"])
OMEGA      = float(_data["omega"])
T_MIN, T_MAX = float(t_grid[0]), float(t_grid[-1])
N_GRID = len(t_grid)

THETA_TRUE = torch.tensor([ALPHA_TRUE, BETA_TRUE, DELTA_TRUE, GAMMA_TRUE],
                          device=device)

T_obs = torch.tensor(t_grid.reshape(-1, 1), device=device)
x_obs = torch.tensor(x_data.reshape(-1, 1), device=device)

t_lo, t_hi = make_windows(T_MIN, T_MAX, N_WIN, WIN_FRAC)
T_LO = torch.tensor(t_lo, device=device).unsqueeze(1)
T_HI = torch.tensor(t_hi, device=device).unsqueeze(1)


# ============================================================ network
class MLP(nn.Module):
    def __init__(self, hidden=HIDDEN):
        super().__init__()
        layers = [1] + list(hidden) + [1]
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


# ============================================================ CV-OLS + PDE loss
def cv_ols_loss(net, T_coll):
    """Compute both physics losses from a single autograd pass.

    Returns (cv_loss, pde_res, theta_per_window, theta_mean) where:
      pde_res = mean((x_tt + DELTA*x_t + ALPHA*x + BETA*x^3
                      - GAMMA*cos(OMEGA*t))^2)              -- baseline residual
      cv_loss = aggregated CV form across the 4 per-window coefficients
                -- DETACHED, diagnostic only, no gradient flows back.

    Gradients through `torch.linalg.solve(A^T A + eps*I, A^T y)` at small
    x_pred amplitude are ill-conditioned (three of the four design-matrix
    columns collapse), creating a degenerate basin where the CV-anchored
    loss is satisfied by a vanishing network output. We therefore detach
    x, dx_dt, d2x_dt2 before the OLS solve, so only the PDE residual +
    data terms drive parameter updates. The CV value is still computed
    on every step (and reflects the current network state) -- it just
    doesn't backprop.
    """
    t_flat = T_coll[:, 0]
    x, dx_dt, d2x_dt2 = first_second_derivs_t(net, T_coll)

    # Standard ODE residual term (uses TRUE coefficients) -- gradient-bearing.
    pde_res_pointwise = (d2x_dt2
                         + DELTA_TRUE * dx_dt
                         + ALPHA_TRUE * x
                         + BETA_TRUE * x ** 3
                         - GAMMA_TRUE * torch.cos(OMEGA * t_flat))
    pde_res = (pde_res_pointwise ** 2).mean()

    # Per-window OLS -- detach so no gradient flows through the linalg.solve.
    theta_per_window, theta_mean = coefs_from_derivs(
        t_flat, x.detach(), dx_dt.detach(), d2x_dt2.detach(),
        T_LO, T_HI, OMEGA, ridge=EPS)

    if CV_FORM == "cv2":
        var = theta_per_window.var(dim=0, unbiased=True)
        mean = theta_per_window.mean(dim=0)
        per_coef = var / (mean ** 2 + EPS)
    elif CV_FORM == "anchored_mse":
        sq = (theta_per_window - THETA_TRUE.unsqueeze(0)) ** 2
        per_coef = sq.mean(dim=0) / (THETA_TRUE ** 2)
    else:
        raise ValueError(f"unknown CV_FORM: {CV_FORM}")

    return per_coef.sum(), pde_res, theta_per_window, theta_mean


# ============================================================ samplers
def sample_collocation(g):
    t = torch.rand(N_COLL, generator=g, device=device) * (T_MAX - T_MIN) + T_MIN
    return t.unsqueeze(1)


# ============================================================ train
torch.manual_seed(SEED); np.random.seed(SEED)
g = torch.Generator(device=device).manual_seed(SEED)

net = MLP().to(device)
opt = torch.optim.Adam(net.parameters(), lr=ADAM_LR)

t0 = time.time()
for it in range(ADAM_ITERS):
    opt.zero_grad()
    T_coll = sample_collocation(g)
    cv_loss, l_pde, theta_per_window, theta_mean = cv_ols_loss(net, T_coll)
    l_data = ((net(T_obs) - x_obs) ** 2).mean()
    loss = W_PDE * l_pde + W_CV * cv_loss + W_DATA * l_data
    loss.backward()
    opt.step()
    if it % 1000 == 0:
        tm = theta_mean.detach().cpu().numpy()
        print(f"[adam {it:5d}] tot={loss.item():.3e}  "
              f"pde={l_pde.item():.3e}  cv={cv_loss.item():.3e}  "
              f"data={l_data.item():.3e}  "
              f"(a,b,d,g)=({tm[0]:.3f},{tm[1]:.3f},{tm[2]:.3f},{tm[3]:.3f})")

# L-BFGS with fixed collocation
T_coll_fix = sample_collocation(g).detach()
lbfgs = torch.optim.LBFGS(net.parameters(),
    max_iter=LBFGS_MAX, max_eval=LBFGS_MAX,
    tolerance_grad=1e-8, tolerance_change=1e-10,
    history_size=50, line_search_fn="strong_wolfe")


def closure():
    lbfgs.zero_grad()
    cv_loss, l_pde, _, _ = cv_ols_loss(net, T_coll_fix)
    l_data = ((net(T_obs) - x_obs) ** 2).mean()
    loss = W_PDE * l_pde + W_CV * cv_loss + W_DATA * l_data
    loss.backward()
    return loss


final = lbfgs.step(closure)
elapsed = time.time() - t0


# ============================================================ evaluate
with torch.no_grad():
    x_pred = net(T_obs).cpu().numpy().reshape(-1)
rel_l2 = float(np.linalg.norm(x_pred - x_data) / np.linalg.norm(x_data))

# Same eval pool the baseline uses -- deterministic via seed.
g_eval = torch.Generator(device=device).manual_seed(SEED + 999)
T_eval = sample_collocation(g_eval)
theta_per_window_t, theta_mean_t = compute_coefs_per_window(
    net, T_eval, T_LO, T_HI, OMEGA, ridge=EPS)
theta_per_window = theta_per_window_t.detach().cpu().numpy()
theta_mean = theta_mean_t.detach().cpu().numpy()

# Full-grid derivatives + residual using the DISCOVERED coefficients.
x_grid_pred, v_grid_pred, a_grid_pred = evaluate_derivatives_on_grid(net, T_obs)
residual_grid = ode_residual_from_grid(
    t_grid, x_grid_pred, v_grid_pred, a_grid_pred,
    theta_mean[0], theta_mean[1], theta_mean[2], theta_mean[3], OMEGA)

# Also compute the final CV loss (post-hoc) for diagnostics.
cv_loss_final, _, _, _ = cv_ols_loss(net, T_eval)
cv_loss_final = float(cv_loss_final.detach())

print(f"\n========== CV-OLS PINN ==========")
print(f"final loss        = {final.item():.3e}")
print(f"training time     = {elapsed:.1f}s")
print(f"rel L2 vs data    = {rel_l2:.4e}")
print(f"discovered theta  = (alpha={theta_mean[0]:.4f}, beta={theta_mean[1]:.4f}, "
      f"delta={theta_mean[2]:.4f}, gamma={theta_mean[3]:.4f})")
print(f"true theta        = (alpha={ALPHA_TRUE}, beta={BETA_TRUE}, "
      f"delta={DELTA_TRUE}, gamma={GAMMA_TRUE})")
print(f"final CV loss     = {cv_loss_final:.3e}")
print(f"mean|ODE residual|= {float(np.abs(residual_grid).mean()):.3e}   "
      f"(using DISCOVERED coefs)")

np.savez("duffing_pinn_cv_ols.npz",
         t=t_grid, x_data=x_data, x_pred=x_pred,
         theta_per_window=theta_per_window,
         theta_mean=theta_mean,
         x_grid_pred=x_grid_pred, v_grid_pred=v_grid_pred, a_grid_pred=a_grid_pred,
         residual_grid=residual_grid,
         alpha_true=ALPHA_TRUE, beta_true=BETA_TRUE, delta_true=DELTA_TRUE,
         gamma_true=GAMMA_TRUE, omega_true=OMEGA,
         cv_loss_final=cv_loss_final,
         training_time=elapsed)
