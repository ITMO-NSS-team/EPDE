"""
Baseline NS PINN -- three candidate equations with TRUE coefficients.

The candidate equation form (`EQUATIONS` in cv_metric.py) is the hypothesis
to verify; its coefficient VALUES are known and used directly here:

    L_phys_eq = mean( (target_eq - features_eq @ theta_true_eq)^2 )
    L_phys    = sum_eq L_phys_eq
    L_data    = mean( (u_pred - u_data)^2 + (v_pred - v_data)^2 )
    L         = W_PHYS * L_phys + W_DATA * L_data

No trainable coefficients -- this is the standard PINN reference against
which the CV variants are compared. The post-hoc per-window OLS recovery
on the trained network is saved as a diagnostic ("does OLS on the
network's output reproduce what we know is right?"), but it does not
affect training.

Saves `ns_pinn_baseline.npz` with:
- field predictions on the full grid
- per-equation true coefs broadcast to (N_WIN, K_eq) as the "headline" theta
- post-hoc per-window OLS theta on the same eval pool the CV scripts use
- per-equation grid residuals using TRUE coefs
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
W_PHYS, W_DATA = 1.0, 1.0
SEED       = 0

N_WIN      = 10
WIN_FRAC   = 0.5
EPS = 1e-8


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

# Truth tensors, one per equation, used in the residual term directly.
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


# ============================================================ losses
def physics_loss(net, X_coll):
    """Sum of per-equation residual losses using TRUE coefficients."""
    derivs = network_derivatives_ns(net, X_coll)
    total = 0.0
    per_eq = {}
    for eq in EQUATIONS:
        y, A = eq["target_and_features"](derivs)
        r = y - A @ THETA_TRUE[eq["name"]]
        l = (r ** 2).mean()
        per_eq[eq["name"]] = l
        total = total + l
    return total, per_eq


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
    l_phys, per_eq = physics_loss(net, X_coll)
    l_data = data_loss(net)
    loss = W_PHYS * l_phys + W_DATA * l_data
    loss.backward()
    opt.step()
    if it % 1000 == 0:
        eq_msgs = "  ".join(f"{nm}={per_eq[nm].item():.2e}" for nm in EQ_NAMES)
        print(f"[adam {it:5d}] tot={loss.item():.3e}  data={l_data.item():.3e}  "
              + eq_msgs)

# L-BFGS with fixed collocation
X_coll_fix = sample_collocation(g).detach()
lbfgs = torch.optim.LBFGS(net.parameters(),
    max_iter=LBFGS_MAX, max_eval=LBFGS_MAX,
    tolerance_grad=1e-8, tolerance_change=1e-10,
    history_size=50, line_search_fn="strong_wolfe")


def closure():
    lbfgs.zero_grad()
    l_phys, _ = physics_loss(net, X_coll_fix)
    l_data = data_loss(net)
    loss = W_PHYS * l_phys + W_DATA * l_data
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

# Headline theta per equation = the TRUE coefs broadcast to (N_WIN, K_eq).
# A scalar IS trivially consistent across windows, so cv2 over this is 0
# -- which is the correct interpretation (the baseline asserts truth).
theta_true_broadcast = {
    eq["name"]: np.tile(eq["theta_true"].astype(np.float32)[None, :], (N_WIN, 1))
    for eq in EQUATIONS
}

# Diagnostic: post-hoc OLS recovery on the SAME eval pool the CV scripts use.
g_eval = torch.Generator(device=device).manual_seed(SEED + 999)
X_eval = sample_collocation(g_eval)
ols_results, _, _ = compute_equation_thetas(net, X_eval, T_LO, T_HI, ridge=EPS)
theta_per_window_ols = {nm: ols_results[nm][0].detach().cpu().numpy()
                        for nm in EQ_NAMES}

# Grid residuals using TRUE coefs (matches the loss).
grid_d = evaluate_on_grid(net, X_obs)
theta_true_np = {eq["name"]: eq["theta_true"] for eq in EQUATIONS}
res_by_eq = grid_equation_residuals(grid_d, theta_true_np)
res_by_eq_grid = {nm: res_by_eq[nm].reshape(Nt, Ny, Nx) for nm in EQ_NAMES}

print(f"\n========== NS BASELINE PINN (TRUE coefs in residual) ==========")
print(f"final loss        = {final.item():.3e}")
print(f"training time     = {elapsed:.1f}s")
print(f"rel L2 (u, v)     = {rel_l2_u:.4e}, {rel_l2_v:.4e}")
for eq in EQUATIONS:
    nm = eq["name"]
    ols_mean = theta_per_window_ols[nm].mean(axis=0)
    print(f"  {nm:>4}: true     = {eq['theta_true'].tolist()}")
    print(f"        OLS mean  = {ols_mean.round(4).tolist()}")
    print(f"        mean|res| = {float(np.abs(res_by_eq_grid[nm]).mean()):.3e}")

save_kwargs = dict(
    t=t_grid, x=x_grid, y=y_grid,
    u_data=u_data, v_data=v_data, p_data=p_data,
    u_pred=u_pred, v_pred=v_pred, p_pred=p_pred,
    lambda_1_true=LAMBDA_1_TRUE, lambda_2_true=LAMBDA_2_TRUE,
    training_time=elapsed,
)
for nm in EQ_NAMES:
    save_kwargs[f"theta_{nm}_per_window"]     = theta_true_broadcast[nm]
    save_kwargs[f"theta_{nm}_per_window_ols"] = theta_per_window_ols[nm]
    save_kwargs[f"residual_{nm}_grid"]        = res_by_eq_grid[nm]
np.savez("ns_pinn_baseline.npz", **save_kwargs)
