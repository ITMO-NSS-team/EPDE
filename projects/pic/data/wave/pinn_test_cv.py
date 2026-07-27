"""
PINN with CV^2-of-OLS regularizer ADDED to the standard PDE residual.

Loss = W_PDE * L_pde + W_CV * L_cv + W_BC * L_bc + W_DATA * L_data

where

    L_pde = mean( (u_tt - C2_TRUE * u_xx)^2 )                  -- baseline term
    L_cv  = CV-form over sliding-window OLS estimates of c^2   -- consistency reg.

For each moving-horizon strip W_i (30 in t, 30 in x, each half-domain
wide), an OLS fit regresses u_tt on u_xx using autograd-computed
derivatives at collocation points inside the strip:

    c^2_i = sum_{p in W_i} u_tt(p) * u_xx(p)  /  sum_{p in W_i} u_xx(p)^2

The CV term penalizes inconsistency of these per-window estimates so the
solution implies a stable wave speed across windows. Mean_i(c^2_i) is
printed as a diagnostic; the PDE term still uses C2_TRUE for the residual.
"""

import time
import numpy as np
import torch
import torch.nn as nn

torch.set_default_dtype(torch.float32)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================ settings
DATA_PATH = "./wave_sln_80.csv"
C2_TRUE   = 0.04                  # only for the post-hoc diagnostic
X_MIN, X_MAX = 0.0, 1.0
T_MIN, T_MAX = 0.0, 1.0

HIDDEN     = [64] * 5
N_COLL     = 8000
N_BC       = 200
ADAM_ITERS = 20000
ADAM_LR    = 1e-3
LBFGS_MAX  = 50000
W_PDE, W_CV, W_BC, W_DATA = 1.0, 1.0, 100.0, 100.0
SEED       = 0

N_WIN_PER_DIM = 30
WIN_FRAC      = 0.5
EPS = 1e-8

# CV regularizer form:
#   "cv2"           : var_i(c2_i) / mean_i(c2_i)^2          -- coefficient unknown
#   "var_anchored"  : var_i(c2_i) / C2_TRUE^2               -- consistency only, fixed scale
#   "anchored_mse"  : mean_i((c2_i - C2_TRUE)^2) / C2_TRUE^2 -- variance + bias to truth
CV_FORM = "anchored_mse"


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
    return u_xx, u_tt        # both (N,)


# ============================================================ CV-OLS + PDE loss
def cv_ols_loss(net, X_coll):
    """Compute both physics losses from a single autograd pass.

    Returns (cv2, pde_res, c2_i, mu) where:
      pde_res = mean((u_tt - C2_TRUE * u_xx)^2)   -- standard PINN residual
      cv2     = CV form over per-window OLS c^2_i (consistency regularizer)
      c2_i    = (2*N_WIN,) per-window estimates
      mu      = mean(c2_i)
    """
    x_c, t_c = X_coll[:, 0], X_coll[:, 1]
    u_xx, u_tt = second_derivs(net, X_coll)

    # Standard PDE residual term.
    pde_res = ((u_tt - C2_TRUE * u_xx) ** 2).mean()

    # which collocation points fall in which strip: (N_WIN, N_COLL)
    in_t = (t_c.unsqueeze(0) >= T_LO) & (t_c.unsqueeze(0) < T_HI)
    in_x = (x_c.unsqueeze(0) >= X_LO) & (x_c.unsqueeze(0) < X_HI)
    masks = torch.cat([in_t, in_x], dim=0).to(u_xx.dtype)   # (2*N_WIN, N_COLL)

    # per-window OLS:   c2_i = sum(u_tt * u_xx) / sum(u_xx^2)
    ut_ux  = (u_tt * u_xx).unsqueeze(0)        # (1, N_COLL)
    uxx_sq = (u_xx ** 2 ).unsqueeze(0)
    num    = (masks * ut_ux ).sum(dim=1)       # (2*N_WIN,)
    den    = (masks * uxx_sq).sum(dim=1)
    c2_i   = num / (den + EPS)

    mu  = c2_i.mean()
    if CV_FORM == "cv2":
        cv2 = c2_i.var(unbiased=True) / (mu ** 2 + EPS)
    elif CV_FORM == "var_anchored":
        cv2 = c2_i.var(unbiased=True) / (C2_TRUE ** 2)
    elif CV_FORM == "anchored_mse":
        cv2 = ((c2_i - C2_TRUE) ** 2).mean() / (C2_TRUE ** 2)
    else:
        raise ValueError(f"unknown CV_FORM: {CV_FORM}")
    return cv2, pde_res, c2_i, mu


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
    cv2, l_pde, c2_i, c2_est = cv_ols_loss(net, X_coll)
    l_bc   = (net(X_bc) ** 2).mean()
    l_data = ((net(X_obs) - u_obs) ** 2).mean()
    loss   = W_PDE * l_pde + W_CV * cv2 + W_BC * l_bc + W_DATA * l_data
    loss.backward()
    opt.step()

    if it % 2000 == 0:
        print(f"[adam {it:5d}] tot={loss.item():.3e}  "
              f"pde={l_pde.item():.3e}  cv2={cv2.item():.3e}  "
              f"bc={l_bc.item():.3e}  data={l_data.item():.3e}  "
              f"c2_est={c2_est.item():.4f}  "
              f"c2_i in [{c2_i.min().item():.4f},{c2_i.max().item():.4f}]")

# L-BFGS with fixed collocation
X_coll_fix = sample_collocation(g).detach()
X_bc_fix   = sample_bc(g).detach()
lbfgs = torch.optim.LBFGS(net.parameters(),
    max_iter=LBFGS_MAX, max_eval=LBFGS_MAX,
    tolerance_grad=1e-8, tolerance_change=1e-10,
    history_size=50, line_search_fn="strong_wolfe")

def closure():
    lbfgs.zero_grad()
    cv2, l_pde, _, _ = cv_ols_loss(net, X_coll_fix)
    l_bc   = (net(X_bc_fix) ** 2).mean()
    l_data = ((net(X_obs) - u_obs) ** 2).mean()
    loss = W_PDE * l_pde + W_CV * cv2 + W_BC * l_bc + W_DATA * l_data
    loss.backward()
    return loss

final = lbfgs.step(closure)
elapsed = time.time() - t0


# ============================================================ evaluate
with torch.no_grad():
    U_pred = net(X_obs).cpu().numpy().reshape(Nx, Nt)
rel_l2 = float(np.linalg.norm(U_pred - U) / np.linalg.norm(U))

# discovered c^2 -- recompute over a dense pool
with torch.no_grad():
    pass
g_eval = torch.Generator(device=device).manual_seed(SEED + 999)
X_eval = sample_collocation(g_eval)
cv2_final, _, c2_i_final, c2_est_final = cv_ols_loss(net, X_eval)
c2_est_final = float(c2_est_final.detach())
cv2_final    = float(cv2_final.detach())

# Full-grid u_xx, u_tt, and PDE residual for the comparison plots.
u_xx_eval, u_tt_eval = second_derivs(net, X_obs)
u_xx_grid = u_xx_eval.detach().cpu().numpy().reshape(Nx, Nt)
u_tt_grid = u_tt_eval.detach().cpu().numpy().reshape(Nx, Nt)
pde_residual_grid = u_tt_grid - C2_TRUE * u_xx_grid

print(f"\n========== CV-OLS PINN ==========")
print(f"final loss        = {final.item():.3e}")
print(f"training time     = {elapsed:.1f}s")
print(f"rel L2 vs data    = {rel_l2:.4e}")
print(f"discovered c^2    = {c2_est_final:.5f}   (true = {C2_TRUE})")
print(f"final CV^2(c^2_i) = {cv2_final:.3e}")
print(f"c^2_i range       = [{c2_i_final.min().item():.5f}, "
      f"{c2_i_final.max().item():.5f}]")
print(f"mean|PDE residual|= {float(np.abs(pde_residual_grid).mean()):.3e}")

np.savez("wave_pinn_cv_ols.npz",
         x=x_grid, t=t_grid, u_data=U, u_pred=U_pred,
         c2_discovered=c2_est_final, cv2_final=cv2_final,
         c2_per_window=c2_i_final.detach().cpu().numpy(),
         u_xx_grid=u_xx_grid, u_tt_grid=u_tt_grid,
         pde_residual_grid=pde_residual_grid,
         training_time=elapsed)