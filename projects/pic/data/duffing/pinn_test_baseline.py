"""
Baseline PINN for the forced Duffing oscillator (torch).

Loss = standard ODE residual + data:

    L_pde  = mean((x_tt + DELTA*x_t + ALPHA*x + BETA*x^3
                   - GAMMA*cos(OMEGA*t))^2)
    L_data = mean((x_pred - x_data)^2)
    L      = W_PDE * L_pde + W_DATA * L_data

All coefficients are hard-coded (read from duffing.npz). This is the
baseline against which `pinn_test_cv.py` (which discovers the coefficients
from a CV regularizer on per-window OLS estimates) is compared.

Saves `duffing_pinn_baseline.npz` with prediction, derivatives, post-hoc
per-window OLS coefficients, ODE residual, and training time -- matches
the schema `pinn_test_cv.py` saves so `compare_pinns.py` can load both
and produce side-by-side stats + plots.
"""

import time
import numpy as np
import torch
import torch.nn as nn

from cv_metric import (
    compute_coefs_per_window,
    evaluate_derivatives_on_grid,
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
W_PDE, W_DATA = 1.0, 1.0
SEED       = 0

N_WIN      = 30
WIN_FRAC   = 0.5
EPS = 1e-8


# ============================================================ load data
_data = np.load(DATA_PATH)
t_grid = _data["t"].astype(np.float32)
x_data = _data["x"].astype(np.float32)
ALPHA  = float(_data["alpha"])
BETA   = float(_data["beta"])
DELTA  = float(_data["delta"])
GAMMA  = float(_data["gamma"])
OMEGA  = float(_data["omega"])
T_MIN, T_MAX = float(t_grid[0]), float(t_grid[-1])
N_GRID = len(t_grid)

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


def first_second_derivs_t(net, t):
    """Return (x, dx_dt, d2x_dt2) each (N,)."""
    t = t.clone().requires_grad_(True)
    x = net(t)
    dx_dt = torch.autograd.grad(x, t, torch.ones_like(x), create_graph=True)[0]
    d2x_dt2 = torch.autograd.grad(dx_dt, t, torch.ones_like(dx_dt),
                                  create_graph=True)[0]
    return x.squeeze(-1), dx_dt.squeeze(-1), d2x_dt2.squeeze(-1)


# ============================================================ ODE residual loss
def ode_residual_loss(net, T_coll):
    """mean((x_tt + delta*x_t + alpha*x + beta*x^3 - gamma*cos(omega*t))^2)."""
    t_flat = T_coll[:, 0]
    x, dx_dt, d2x_dt2 = first_second_derivs_t(net, T_coll)
    residual = (d2x_dt2
                + DELTA * dx_dt
                + ALPHA * x
                + BETA * x ** 3
                - GAMMA * torch.cos(OMEGA * t_flat))
    return (residual ** 2).mean()


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
    l_pde = ode_residual_loss(net, T_coll)
    l_data = ((net(T_obs) - x_obs) ** 2).mean()
    loss = W_PDE * l_pde + W_DATA * l_data
    loss.backward()
    opt.step()
    if it % 1000 == 0:
        print(f"[adam {it:5d}] tot={loss.item():.3e}  "
              f"pde={l_pde.item():.3e}  data={l_data.item():.3e}")

# L-BFGS with fixed collocation
T_coll_fix = sample_collocation(g).detach()
lbfgs = torch.optim.LBFGS(net.parameters(),
    max_iter=LBFGS_MAX, max_eval=LBFGS_MAX,
    tolerance_grad=1e-8, tolerance_change=1e-10,
    history_size=50, line_search_fn="strong_wolfe")


def closure():
    lbfgs.zero_grad()
    l_pde = ode_residual_loss(net, T_coll_fix)
    l_data = ((net(T_obs) - x_obs) ** 2).mean()
    loss = W_PDE * l_pde + W_DATA * l_data
    loss.backward()
    return loss


final = lbfgs.step(closure)
elapsed = time.time() - t0


# ============================================================ evaluate
with torch.no_grad():
    x_pred = net(T_obs).cpu().numpy().reshape(-1)
rel_l2 = float(np.linalg.norm(x_pred - x_data) / np.linalg.norm(x_data))

# Post-hoc per-window OLS on the same eval pool the CV PINN uses
# (deterministic via seed -> bit-identical pool across scripts).
g_eval = torch.Generator(device=device).manual_seed(SEED + 999)
T_eval = sample_collocation(g_eval)
theta_per_window_t, theta_mean_t = compute_coefs_per_window(
    net, T_eval, T_LO, T_HI, OMEGA, ridge=EPS)
theta_per_window = theta_per_window_t.detach().cpu().numpy()
theta_mean = theta_mean_t.detach().cpu().numpy()

# Full-grid derivatives + residual using the TRUE coefficients
# (this is the baseline -- it knows the coefficients).
x_grid_pred, v_grid_pred, a_grid_pred = evaluate_derivatives_on_grid(net, T_obs)
residual_grid = ode_residual_from_grid(
    t_grid, x_grid_pred, v_grid_pred, a_grid_pred,
    ALPHA, BETA, DELTA, GAMMA, OMEGA)

print(f"\n========== BASELINE PINN (ODE residual) ==========")
print(f"final loss        = {final.item():.3e}")
print(f"training time     = {elapsed:.1f}s")
print(f"rel L2 vs data    = {rel_l2:.4e}")
print(f"post-hoc theta    = (alpha={theta_mean[0]:.4f}, beta={theta_mean[1]:.4f}, "
      f"delta={theta_mean[2]:.4f}, gamma={theta_mean[3]:.4f})")
print(f"true theta        = (alpha={ALPHA}, beta={BETA}, "
      f"delta={DELTA}, gamma={GAMMA})")
print(f"mean|ODE residual|= {float(np.abs(residual_grid).mean()):.3e}")

np.savez("duffing_pinn_baseline.npz",
         t=t_grid, x_data=x_data, x_pred=x_pred,
         theta_per_window=theta_per_window,
         theta_mean=theta_mean,
         x_grid_pred=x_grid_pred, v_grid_pred=v_grid_pred, a_grid_pred=a_grid_pred,
         residual_grid=residual_grid,
         alpha_true=ALPHA, beta_true=BETA, delta_true=DELTA,
         gamma_true=GAMMA, omega_true=OMEGA,
         training_time=elapsed)
