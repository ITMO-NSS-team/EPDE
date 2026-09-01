"""
1D Fourier Neural Operator (FNO) flow-map surrogate for the double
pendulum -- the purely DATA-DRIVEN reference in the PINN comparison.

The operator learns the fixed-horizon flow map of the (autonomous) system
on angle windows:

    G: [theta1, theta2] on (t_k, ..., t_{k+M-1})
         -> [theta1, theta2] on (t_{k+M}, ..., t_{k+2M-1})

trained by MSE over all overlapping window pairs drawn from the TRAIN
slice only (t in [0, TRAIN_FRAC * T_MAX]):

    L = mean( (G(u_win) - u_next_win)^2 )        (normalized channels)

This is the opposite supervision regime from the PINN scripts: the FNO
sees the TRAIN trajectory densely and knows NO physics, while the PINNs
see physics (+ CV) and only the initial condition. Both are judged on the
same held-out tail.

Inference is an autoregressive rollout: the first M data points seed the
model (they are ground truth by construction -- `seed_len` is saved in
the artifact), then G is applied repeatedly on its own output to cover
the full grid, so everything past the seed (including the whole test
region) is model prediction with compounding error.

Post-hoc diagnostics mirror the PINN scripts, with finite-difference
derivatives standing in for autograd (the rollout lives on the uniform
data grid): omega/alpha via np.gradient, then the same per-window OLS
(`ols_per_window`, TRAIN-slice mask, windows spanning the full [0, T])
and per-equation grid residuals.

Saves `dp_fno.npz` matching the schema `compare_pinns.py` expects. The
training loss history is stored as `loss_history_data` (it IS a data
loss); phys/cv histories are all-zero so those panels skip this model.
"""

import time
import numpy as np
import torch
import torch.nn as nn

from cv_metric import (
    EQUATIONS,
    EQ_NAMES,
    grid_equation_residuals,
    make_windows,
    ols_per_window,
    set_gravity,
)

torch.set_default_dtype(torch.float32)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================ settings
DATA_PATH  = "./dp.npz"

M          = 50        # window length (points); G maps M points -> next M
MODES      = 16        # retained Fourier modes (< M//2 + 1)
WIDTH      = 64        # channel width of the Fourier layers
N_LAYERS   = 4
ADAM_ITERS = 20000
ADAM_LR    = 1e-3
LR_STEP    = 5000      # halve the lr every LR_STEP iters
SEED       = 0

TRAIN_FRAC = 0.8       # temporal split: train on first TRAIN_FRAC of trajectory
N_WIN      = 30        # post-hoc OLS windows (matches the CV scripts)
WIN_FRAC   = 0.5
EPS = 1e-8


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
DT = float(t_grid[1] - t_grid[0])

n_train = int(np.ceil(TRAIN_FRAC * N_GRID))
t_split = float(t_grid[n_train - 1])

TH = np.stack([th1_data, th2_data], axis=0)          # (2, N_GRID)

# Channel normalization from the train slice only.
TH_MEAN = TH[:, :n_train].mean(axis=1, keepdims=True)
TH_STD  = TH[:, :n_train].std(axis=1, keepdims=True) + EPS

# Overlapping window pairs, both windows fully inside the train slice.
starts = np.arange(0, n_train - 2 * M + 1)
X_np = np.stack([(TH[:, i:i + M]         - TH_MEAN) / TH_STD for i in starts])
Y_np = np.stack([(TH[:, i + M:i + 2 * M] - TH_MEAN) / TH_STD for i in starts])
X_pairs = torch.tensor(X_np, device=device)          # (P, 2, M)
Y_pairs = torch.tensor(Y_np, device=device)          # (P, 2, M)
print(f"train pairs: {len(starts)}  (window M={M} pts = {M * DT:.2f}s, "
      f"n_train={n_train}/{N_GRID})")

# Local (window-relative) coordinate channel -- translation-invariant, so
# it does not break the autonomy of the learned flow map.
GRID_CH = torch.linspace(0.0, 1.0, M, device=device).view(1, 1, M)


# ============================================================ FNO
class SpectralConv1d(nn.Module):
    def __init__(self, in_ch, out_ch, modes):
        super().__init__()
        self.modes = modes
        scale = 1.0 / (in_ch * out_ch)
        self.weight = nn.Parameter(
            scale * torch.randn(in_ch, out_ch, modes, dtype=torch.cfloat))

    def forward(self, x):                               # (B, C, N)
        x_ft = torch.fft.rfft(x, dim=-1)                # (B, C, N//2+1)
        out_ft = torch.zeros(x.shape[0], self.weight.shape[1], x_ft.shape[-1],
                             dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes] = torch.einsum(
            "bim,iom->bom", x_ft[:, :, :self.modes], self.weight)
        return torch.fft.irfft(out_ft, n=x.shape[-1], dim=-1)


class FNO1d(nn.Module):
    def __init__(self, in_ch=3, out_ch=2, width=WIDTH, modes=MODES,
                 n_layers=N_LAYERS):
        super().__init__()
        self.lift = nn.Conv1d(in_ch, width, 1)
        self.spectral = nn.ModuleList(
            [SpectralConv1d(width, width, modes) for _ in range(n_layers)])
        self.pointwise = nn.ModuleList(
            [nn.Conv1d(width, width, 1) for _ in range(n_layers)])
        self.proj1 = nn.Conv1d(width, 128, 1)
        self.proj2 = nn.Conv1d(128, out_ch, 1)

    def forward(self, u):                               # (B, 2, M) normalized
        grid = GRID_CH.expand(u.shape[0], 1, u.shape[-1])
        x = self.lift(torch.cat([u, grid], dim=1))
        for k, (spec, pw) in enumerate(zip(self.spectral, self.pointwise)):
            y = spec(x) + pw(x)
            x = torch.nn.functional.gelu(y) if k < len(self.spectral) - 1 else y
        return self.proj2(torch.nn.functional.gelu(self.proj1(x)))


# ============================================================ train
torch.manual_seed(SEED); np.random.seed(SEED)

model = FNO1d().to(device)
opt = torch.optim.Adam(model.parameters(), lr=ADAM_LR)
sched = torch.optim.lr_scheduler.StepLR(opt, step_size=LR_STEP, gamma=0.5)

# Adam loss history (sampled at the print frequency) -- phys/cv are zero
# (the FNO has no physics or CV term); the data MSE doubles as the total.
hist = {"iter": [], "tot": [], "phys": [], "cv": [], "data": []}

t0 = time.time()
for it in range(ADAM_ITERS):
    opt.zero_grad()
    loss = ((model(X_pairs) - Y_pairs) ** 2).mean()
    loss.backward()
    opt.step()
    sched.step()
    if it % 1000 == 0:
        hist["iter"].append(it)
        hist["tot"].append(loss.item())
        hist["phys"].append(0.0)
        hist["cv"].append(0.0)
        hist["data"].append(loss.item())
        print(f"[adam {it:5d}] data={loss.item():.3e}  "
              f"lr={sched.get_last_lr()[0]:.2e}")
elapsed = time.time() - t0


# ============================================================ rollout
model.eval()
with torch.no_grad():
    win = torch.tensor((TH[:, :M] - TH_MEAN) / TH_STD,
                       device=device).unsqueeze(0)     # (1, 2, M) data seed
    chunks = [win]
    n_have = M
    while n_have < N_GRID:
        win = model(win)
        chunks.append(win)
        n_have += M
    pred_n = torch.cat(chunks, dim=-1)[0].cpu().numpy()  # (2, >= N_GRID)
TH_pred = (pred_n * TH_STD + TH_MEAN)[:, :N_GRID].astype(np.float32)
th1_pred, th2_pred = TH_pred[0], TH_pred[1]

def _rel_l2(pred, data, sl):
    p = pred[sl]; d = data[sl]
    return float(np.linalg.norm(p - d) / np.linalg.norm(d))


sl_train = slice(0, n_train)
sl_test  = slice(n_train, None)
rel_l2_th1_train = _rel_l2(th1_pred, th1_data, sl_train)
rel_l2_th2_train = _rel_l2(th2_pred, th2_data, sl_train)
rel_l2_th1_test  = _rel_l2(th1_pred, th1_data, sl_test)
rel_l2_th2_test  = _rel_l2(th2_pred, th2_data, sl_test)


# ============================================================ FD derivatives
# The rollout lives on the uniform data grid, so omega/alpha come from
# second-order finite differences instead of autograd.
w1_pred = np.gradient(th1_pred, DT).astype(np.float32)
w2_pred = np.gradient(th2_pred, DT).astype(np.float32)
a1_pred = np.gradient(w1_pred, DT).astype(np.float32)
a2_pred = np.gradient(w2_pred, DT).astype(np.float32)

grid_d = dict(theta1=th1_pred, theta2=th2_pred,
              omega1=w1_pred,  omega2=w2_pred,
              alpha1=a1_pred,  alpha2=a2_pred)


# ============================================================ post-hoc OLS
# Mirror the PINN scripts: windows span the full [0, T], the mask sees
# TRAIN-slice points only (float64 for the 3x3 solves).
t_lo, t_hi = make_windows(T_MIN, T_MAX, N_WIN, WIN_FRAC)
T_LO = torch.tensor(t_lo, dtype=torch.float64).unsqueeze(1)
T_HI = torch.tensor(t_hi, dtype=torch.float64).unsqueeze(1)
t_train_t = torch.tensor(t_grid[:n_train], dtype=torch.float64)
mask = ((t_train_t.unsqueeze(0) >= T_LO)
        & (t_train_t.unsqueeze(0) < T_HI)).to(torch.float64)

derivs_train = {k: torch.tensor(v[:n_train], dtype=torch.float64)
                for k, v in grid_d.items()}
theta_per_window = {}
theta_means_final = {}
for eq in EQUATIONS:
    y, A = eq["target_and_features"](derivs_train)
    theta_pw, theta_mean = ols_per_window(y, A, mask, ridge=EPS)
    theta_per_window[eq["name"]] = theta_pw.numpy()
    theta_means_final[eq["name"]] = theta_mean.numpy()

res_by_eq = grid_equation_residuals(grid_d, theta_means_final)

print(f"\n========== DP FNO (data-driven flow map, no physics) ==========")
print(f"final data loss         = {hist['tot'][-1]:.3e}")
print(f"training time           = {elapsed:.1f}s")
print(f"train/test split        = {TRAIN_FRAC:.2f}  (t_split={t_split:.3f}, n_train={n_train}/{N_GRID})")
print(f"rollout                 = {len(chunks)} windows of {M} pts "
      f"(first window = data seed)")
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
    omega1_pred=w1_pred,  omega2_pred=w2_pred,
    alpha1_pred=a1_pred,  alpha2_pred=a2_pred,
    g_true=G_TRUE,
    training_time=elapsed,
    train_frac=np.float32(TRAIN_FRAC),
    n_train=np.int32(n_train),
    t_split=np.float32(t_split),
    seed_len=np.int32(M),
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
np.savez("dp_fno.npz", **save_kwargs)
print("saved -> dp_fno.npz")
