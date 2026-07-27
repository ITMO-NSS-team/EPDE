"""
CV PINN with TRAINABLE per-window theta -- forced Duffing oscillator.

Replaces the per-window OLS solve with `nn.Parameter` per-window
coefficients (alpha_i, beta_i, delta_i, gamma_i). The PDE residual is
pointwise, using each collocation point's nearest-center window theta:

    r(t_k) = x_tt(t_k) + delta_{w(k)}*x_t(t_k) + alpha_{w(k)}*x(t_k)
             + beta_{w(k)}*x^3(t_k) - gamma_{w(k)}*cos(omega*t_k)

    L_phys = mean_k r(t_k)^2

The CV regularizer is applied DIRECTLY to `theta_per_window` (no OLS
solve, no linalg.solve in the training loop):

    "cv2":          sum_j  var_i(theta_ji) / mean_i(theta_ji)^2
    "anchored_mse": sum_j  mean_i((theta_ji - theta_j_true)^2) / theta_j_true^2

Both L_phys and L_cv have well-conditioned gradients regardless of
network amplitude:
  * L_phys -> per-point gradient on the net (standard PINN residual)
  * L_phys -> per-window gradient on theta (one term per point in window)
  * L_cv   -> per-coefficient gradient on theta (analytic, no inverse)

The merge prevents the x_pred ~ 0 degenerate basin that broke the
OLS-based CV PINN: anchored-CV pulls theta away from zero, which then
makes L_phys non-zero at x=0, which drives the net out.

Joint Adam (and L-BFGS) over `net.parameters() + [theta_per_window]`.

Post-hoc, we also compute the OLS-recovered theta for diagnostic
comparison -- saved alongside the trained theta in
`duffing_pinn_cv_trainable.npz`.
"""

import time
import numpy as np
import torch
import torch.nn as nn

from cv_metric import (
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
THETA_LR   = 1e-2          # higher LR for theta (only 120 scalars vs net's thousands)
LBFGS_MAX  = 20000
W_PHYS, W_CV, W_DATA = 1.0, 1.0, 1.0
SEED       = 0

N_WIN      = 30
WIN_FRAC   = 0.5
EPS = 1e-8

# CV regularizer form (regularizes the TRAINED theta_per_window directly):
#   "cv2"          : sum_j  var_i(theta_ji) / mean_i(theta_ji)^2  (no truth needed)
#   "anchored_mse" : sum_j  mean_i((theta_ji - theta_j_true)^2) / theta_j_true^2
CV_FORM = "anchored_mse"

# theta_per_window initial values (alpha, beta, delta, gamma).
# Zero is the principled neutral start -- anchored CV will pull toward truth.
THETA_INIT = (0.0, 0.0, 0.0, 0.0)


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
WIN_CENTERS = torch.tensor(0.5 * (t_lo + t_hi), device=device)   # (N_WIN,)


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


def window_index_for(t_flat):
    """Nearest-center window index for each collocation time. (N,) -> (N,) ints."""
    dists = (t_flat.unsqueeze(1) - WIN_CENTERS.unsqueeze(0)).abs()
    return dists.argmin(dim=1)


# ============================================================ merged physics + CV
def merged_loss(net, theta_per_window, T_coll):
    """Pointwise physics residual using trainable per-window theta + CV regularizer.

    Returns (l_phys, l_cv, theta_mean).
    """
    t_flat = T_coll[:, 0]
    x, dx_dt, d2x_dt2 = first_second_derivs_t(net, T_coll)

    win_idx = window_index_for(t_flat)                  # (N_COLL,) ints
    theta_at_pt = theta_per_window[win_idx]             # (N_COLL, 4)
    alpha = theta_at_pt[:, 0]
    beta  = theta_at_pt[:, 1]
    delta = theta_at_pt[:, 2]
    gamma = theta_at_pt[:, 3]

    # Pointwise residual -- each point gives a clean gradient on both
    # the net (through x, dx_dt, d2x_dt2) AND on theta_per_window
    # (through alpha/beta/delta/gamma).
    residual = (d2x_dt2
                + delta * dx_dt
                + alpha * x
                + beta * x ** 3
                - gamma * torch.cos(OMEGA * t_flat))
    l_phys = (residual ** 2).mean()

    # CV regularizer applied DIRECTLY to theta_per_window -- no OLS solve,
    # no linalg.solve, no inverse. Gradient is analytic and well-conditioned.
    if CV_FORM == "cv2":
        var = theta_per_window.var(dim=0, unbiased=True)
        mu = theta_per_window.mean(dim=0)
        per_coef = var / (mu ** 2 + EPS)
    elif CV_FORM == "anchored_mse":
        sq = (theta_per_window - THETA_TRUE.unsqueeze(0)) ** 2
        per_coef = sq.mean(dim=0) / (THETA_TRUE ** 2)
    else:
        raise ValueError(f"unknown CV_FORM: {CV_FORM}")
    l_cv = per_coef.sum()

    return l_phys, l_cv, theta_per_window.mean(dim=0)


# ============================================================ samplers
def sample_collocation(g):
    t = torch.rand(N_COLL, generator=g, device=device) * (T_MAX - T_MIN) + T_MIN
    return t.unsqueeze(1)


# ============================================================ train
torch.manual_seed(SEED); np.random.seed(SEED)
g = torch.Generator(device=device).manual_seed(SEED)

net = MLP().to(device)

theta_per_window = nn.Parameter(
    torch.tensor([list(THETA_INIT)] * N_WIN, device=device, dtype=torch.float32)
)

opt = torch.optim.Adam([
    {"params": net.parameters(), "lr": ADAM_LR},
    {"params": [theta_per_window], "lr": THETA_LR},
])

t0 = time.time()
for it in range(ADAM_ITERS):
    opt.zero_grad()
    T_coll = sample_collocation(g)
    l_phys, l_cv, theta_mean = merged_loss(net, theta_per_window, T_coll)
    l_data = ((net(T_obs) - x_obs) ** 2).mean()
    loss = W_PHYS * l_phys + W_CV * l_cv + W_DATA * l_data
    loss.backward()
    opt.step()
    if it % 1000 == 0:
        tm = theta_mean.detach().cpu().numpy()
        print(f"[adam {it:5d}] tot={loss.item():.3e}  "
              f"phys={l_phys.item():.3e}  cv={l_cv.item():.3e}  "
              f"data={l_data.item():.3e}  "
              f"(a,b,d,g)=({tm[0]:.3f},{tm[1]:.3f},{tm[2]:.3f},{tm[3]:.3f})")

# L-BFGS with fixed collocation
T_coll_fix = sample_collocation(g).detach()
lbfgs = torch.optim.LBFGS(
    list(net.parameters()) + [theta_per_window],
    max_iter=LBFGS_MAX, max_eval=LBFGS_MAX,
    tolerance_grad=1e-8, tolerance_change=1e-10,
    history_size=50, line_search_fn="strong_wolfe",
)


def closure():
    lbfgs.zero_grad()
    l_phys, l_cv, _ = merged_loss(net, theta_per_window, T_coll_fix)
    l_data = ((net(T_obs) - x_obs) ** 2).mean()
    loss = W_PHYS * l_phys + W_CV * l_cv + W_DATA * l_data
    loss.backward()
    return loss


final = lbfgs.step(closure)
elapsed = time.time() - t0


# ============================================================ evaluate
with torch.no_grad():
    x_pred = net(T_obs).cpu().numpy().reshape(-1)
rel_l2 = float(np.linalg.norm(x_pred - x_data) / np.linalg.norm(x_data))

# Trained per-window theta (what the optimizer learned).
theta_per_window_trained = theta_per_window.detach().cpu().numpy()
theta_mean_trained = theta_per_window_trained.mean(axis=0)

# Diagnostic: post-hoc OLS theta from the trained network, same eval pool
# the baseline uses. If the merged approach worked, these should agree
# with `theta_per_window_trained` once both are at the true coefficients.
g_eval = torch.Generator(device=device).manual_seed(SEED + 999)
T_eval = sample_collocation(g_eval)
theta_ols_t, theta_ols_mean_t = compute_coefs_per_window(
    net, T_eval, T_LO, T_HI, OMEGA, ridge=EPS)
theta_per_window_ols = theta_ols_t.detach().cpu().numpy()
theta_mean_ols = theta_ols_mean_t.detach().cpu().numpy()

# Full-grid derivatives + residual using the TRAINED theta mean.
x_grid_pred, v_grid_pred, a_grid_pred = evaluate_derivatives_on_grid(net, T_obs)
residual_grid = ode_residual_from_grid(
    t_grid, x_grid_pred, v_grid_pred, a_grid_pred,
    theta_mean_trained[0], theta_mean_trained[1],
    theta_mean_trained[2], theta_mean_trained[3], OMEGA)

print(f"\n========== CV-OLS PINN (trainable theta) ==========")
print(f"final loss        = {final.item():.3e}")
print(f"training time     = {elapsed:.1f}s")
print(f"rel L2 vs data    = {rel_l2:.4e}")
print(f"trained theta     = (alpha={theta_mean_trained[0]:.4f}, "
      f"beta={theta_mean_trained[1]:.4f}, "
      f"delta={theta_mean_trained[2]:.4f}, "
      f"gamma={theta_mean_trained[3]:.4f})")
print(f"post-hoc OLS theta= (alpha={theta_mean_ols[0]:.4f}, "
      f"beta={theta_mean_ols[1]:.4f}, "
      f"delta={theta_mean_ols[2]:.4f}, "
      f"gamma={theta_mean_ols[3]:.4f})")
print(f"true theta        = (alpha={ALPHA_TRUE}, beta={BETA_TRUE}, "
      f"delta={DELTA_TRUE}, gamma={GAMMA_TRUE})")
print(f"mean|ODE residual|= {float(np.abs(residual_grid).mean()):.3e}   "
      f"(using TRAINED theta mean)")

np.savez("duffing_pinn_cv_trainable.npz",
         t=t_grid, x_data=x_data, x_pred=x_pred,
         theta_per_window=theta_per_window_trained,
         theta_per_window_ols=theta_per_window_ols,
         theta_mean=theta_mean_trained,
         theta_mean_ols=theta_mean_ols,
         x_grid_pred=x_grid_pred, v_grid_pred=v_grid_pred, a_grid_pred=a_grid_pred,
         residual_grid=residual_grid,
         alpha_true=ALPHA_TRUE, beta_true=BETA_TRUE, delta_true=DELTA_TRUE,
         gamma_true=GAMMA_TRUE, omega_true=OMEGA,
         training_time=elapsed)
