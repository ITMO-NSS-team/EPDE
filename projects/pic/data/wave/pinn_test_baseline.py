"""
Torch baseline PINN for the 1D wave equation -- pure PDE residual loss.

Structural mirror of `pinn_test_cv.py`. The architecture, optimizers,
samplers, seeds, and BC/data losses are byte-identical. The only
substantive difference is the physics loss:

    L_pde = mean( (u_tt - C2_TRUE * u_xx)^2 )

with `C2_TRUE = 0.04` hard-coded (this is what makes it the baseline:
it gets the answer handed to it, whereas the CV variant has to discover
the constant).

Saves `wave_pinn_baseline.npz` with the same fields the CV script saves
(post-hoc `c2_per_window` over the same eval pool, full-grid derivatives,
PDE residual map) so the compare script can rank both models on the same
metrics.
"""

import time
import numpy as np
import torch
import torch.nn as nn

from cv_metric import compute_c2_per_window, evaluate_derivatives_on_grid

torch.set_default_dtype(torch.float32)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================ settings
DATA_PATH = "./wave_sln_80.csv"
C2_TRUE   = 0.04
X_MIN, X_MAX = 0.0, 1.0
T_MIN, T_MAX = 0.0, 1.0

HIDDEN     = [64] * 5
N_COLL     = 8000
N_BC       = 200
ADAM_ITERS = 20000
ADAM_LR    = 1e-3
LBFGS_MAX  = 50000
W_PDE, W_BC, W_DATA = 1.0, 100.0, 100.0
SEED       = 0

N_WIN_PER_DIM = 30
WIN_FRAC      = 0.5
EPS = 1e-8


# ============================================================ data + windows
U = np.loadtxt(DATA_PATH, delimiter=",").astype(np.float32)
Nx, Nt = U.shape
x_grid = np.linspace(X_MIN, X_MAX, Nx, dtype=np.float32)
t_grid = np.linspace(T_MIN, T_MAX, Nt, dtype=np.float32)
XX, TT = np.meshgrid(x_grid, t_grid, indexing="ij")
X_obs = torch.tensor(np.stack([XX.ravel(), TT.ravel()], axis=1), device=device)
u_obs = torch.tensor(U.reshape(-1, 1), device=device)


def make_windows(lo, hi, n, frac):
    w = (hi - lo) * frac
    h = w / 2
    c = np.linspace(lo + h, hi - h, n, dtype=np.float32)
    return c - h, c + h

t_lo, t_hi = make_windows(T_MIN, T_MAX, N_WIN_PER_DIM, WIN_FRAC)
x_lo, x_hi = make_windows(X_MIN, X_MAX, N_WIN_PER_DIM, WIN_FRAC)
T_LO = torch.tensor(t_lo, device=device).unsqueeze(1)
T_HI = torch.tensor(t_hi, device=device).unsqueeze(1)
X_LO = torch.tensor(x_lo, device=device).unsqueeze(1)
X_HI = torch.tensor(x_hi, device=device).unsqueeze(1)


# ============================================================ network
class MLP(nn.Module):
    def __init__(self, hidden=HIDDEN):
        super().__init__()
        layers = [2] + list(hidden) + [1]
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
    def forward(self, x):
        return self.net(x)


def second_derivs(net, xt):
    """Return u_xx and u_tt at xt via autograd."""
    xt = xt.clone().requires_grad_(True)
    u  = net(xt)
    g  = torch.autograd.grad(u, xt, torch.ones_like(u), create_graph=True)[0]
    u_x, u_t = g[:, 0:1], g[:, 1:2]
    u_xx = torch.autograd.grad(u_x, xt, torch.ones_like(u_x),
                               create_graph=True)[0][:, 0]
    u_tt = torch.autograd.grad(u_t, xt, torch.ones_like(u_t),
                               create_graph=True)[0][:, 1]
    return u_xx, u_tt


# ============================================================ PDE residual loss
def pde_residual_loss(net, X_coll, c2=C2_TRUE):
    """Standard PINN residual: mean((u_tt - c^2 * u_xx)^2) at X_coll."""
    u_xx, u_tt = second_derivs(net, X_coll)
    return ((u_tt - c2 * u_xx) ** 2).mean()


# ============================================================ samplers
def sample_collocation(g):
    x = torch.rand(N_COLL, generator=g, device=device) * (X_MAX - X_MIN) + X_MIN
    t = torch.rand(N_COLL, generator=g, device=device) * (T_MAX - T_MIN) + T_MIN
    return torch.stack([x, t], dim=1)

def sample_bc(g):
    t = torch.rand(N_BC, generator=g, device=device) * (T_MAX - T_MIN) + T_MIN
    left  = torch.stack([torch.full_like(t, X_MIN), t], dim=1)
    right = torch.stack([torch.full_like(t, X_MAX), t], dim=1)
    return torch.cat([left, right], dim=0)


# ============================================================ train
torch.manual_seed(SEED); np.random.seed(SEED)
g = torch.Generator(device=device).manual_seed(SEED)

net = MLP().to(device)
opt = torch.optim.Adam(net.parameters(), lr=ADAM_LR)

t0 = time.time()
for it in range(ADAM_ITERS):
    opt.zero_grad()
    X_coll = sample_collocation(g)
    X_bc   = sample_bc(g)
    l_pde  = pde_residual_loss(net, X_coll)
    l_bc   = (net(X_bc) ** 2).mean()
    l_data = ((net(X_obs) - u_obs) ** 2).mean()
    loss   = W_PDE * l_pde + W_BC * l_bc + W_DATA * l_data
    loss.backward()
    opt.step()

    if it % 2000 == 0:
        print(f"[adam {it:5d}] tot={loss.item():.3e}  pde={l_pde.item():.3e}  "
              f"bc={l_bc.item():.3e}  data={l_data.item():.3e}")

# L-BFGS with fixed collocation
X_coll_fix = sample_collocation(g).detach()
X_bc_fix   = sample_bc(g).detach()
lbfgs = torch.optim.LBFGS(net.parameters(),
    max_iter=LBFGS_MAX, max_eval=LBFGS_MAX,
    tolerance_grad=1e-8, tolerance_change=1e-10,
    history_size=50, line_search_fn="strong_wolfe")

def closure():
    lbfgs.zero_grad()
    l_pde  = pde_residual_loss(net, X_coll_fix)
    l_bc   = (net(X_bc_fix) ** 2).mean()
    l_data = ((net(X_obs) - u_obs) ** 2).mean()
    loss = W_PDE * l_pde + W_BC * l_bc + W_DATA * l_data
    loss.backward()
    return loss

final = lbfgs.step(closure)
elapsed = time.time() - t0


# ============================================================ evaluate
with torch.no_grad():
    U_pred = net(X_obs).cpu().numpy().reshape(Nx, Nt)
rel_l2 = float(np.linalg.norm(U_pred - U) / np.linalg.norm(U))

# Post-hoc c^2 over the same eval pool the CV PINN uses -- bit-identical
# pool across scripts because torch.Generator is deterministic on a given
# device for the same seed.
g_eval = torch.Generator(device=device).manual_seed(SEED + 999)
X_eval = sample_collocation(g_eval)
c2_per_window, c2_discovered = compute_c2_per_window(
    net, X_eval, X_LO, X_HI, T_LO, T_HI, eps=EPS)

# Full-grid u_xx, u_tt, and PDE residual for the comparison plots.
u_xx_grid, u_tt_grid = evaluate_derivatives_on_grid(net, X_obs, Nx, Nt)
pde_residual_grid = u_tt_grid - C2_TRUE * u_xx_grid

print(f"\n========== BASELINE PINN (PDE residual) ==========")
print(f"final loss        = {final.item():.3e}")
print(f"training time     = {elapsed:.1f}s")
print(f"rel L2 vs data    = {rel_l2:.4e}")
print(f"post-hoc mean c^2 = {c2_discovered:.5f}   (true = {C2_TRUE})")
print(f"c^2_i range       = [{float(c2_per_window.min()):.5f}, "
      f"{float(c2_per_window.max()):.5f}]")
print(f"mean|PDE residual|= {float(np.abs(pde_residual_grid).mean()):.3e}")

np.savez("wave_pinn_baseline.npz",
         x=x_grid, t=t_grid, u_data=U, u_pred=U_pred,
         c2_discovered=c2_discovered,
         c2_per_window=c2_per_window,
         u_xx_grid=u_xx_grid, u_tt_grid=u_tt_grid,
         pde_residual_grid=pde_residual_grid,
         training_time=elapsed)
