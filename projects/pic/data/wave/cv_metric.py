"""
Shared helpers for the PINN CV-vs-baseline comparison harness.

Both `pinn_test_cv.py` and `pinn_test_baseline.py` train a torch MLP on the
1D wave equation and need to dump (a) the 60 per-window c^2 estimates from
sliding-window OLS at a fixed eval pool and (b) full-grid u_xx / u_tt / PDE
residual maps. This module centralizes those primitives so the two training
scripts produce bit-identical evaluation artifacts (modulo network weights).

The window definition and OLS formula are anchored to `pinn_test_cv.py`'s
training-time loss (lines 66-127 of the original file) so the post-hoc
metric matches the training-time metric exactly when the same eval pool is
used.

The compare script (`compare_pinns.py`) reads the resulting .npz files and
only needs `cv_forms` / `error_stats` -- it does not redo any derivatives.

Run as a script to execute an analytical smoke test:

    python cv_metric.py
"""

import numpy as np
import torch


EPS = 1e-8


# ============================================================ windows
def make_windows(lo, hi, n, frac):
    """Return (lo, hi) arrays for `n` half-width-`frac*(hi-lo)/2` windows
    centered evenly across [lo+h, hi-h]. Verbatim from pinn_test_cv.py:66."""
    w = (hi - lo) * frac
    h = w / 2
    c = np.linspace(lo + h, hi - h, n, dtype=np.float32)
    return c - h, c + h


# ============================================================ autograd
def second_derivs(net, xt):
    """Return u_xx and u_tt at xt via autograd. xt: (N, 2). Output: each (N,)."""
    xt = xt.clone().requires_grad_(True)
    u = net(xt)
    g = torch.autograd.grad(u, xt, torch.ones_like(u), create_graph=True)[0]
    u_x, u_t = g[:, 0:1], g[:, 1:2]
    u_xx = torch.autograd.grad(u_x, xt, torch.ones_like(u_x),
                               create_graph=True)[0][:, 0]
    u_tt = torch.autograd.grad(u_t, xt, torch.ones_like(u_t),
                               create_graph=True)[0][:, 1]
    return u_xx, u_tt


# ============================================================ sliding-window OLS
def compute_c2_per_window(net, X_coll, X_LO, X_HI, T_LO, T_HI, eps=EPS):
    """Per-window OLS estimate c^2_i = sum(u_tt * u_xx) / sum(u_xx^2).

    Mirrors `cv_ols_loss` in pinn_test_cv.py:113-127 exactly. Returns
    (c2_i, mu) as detached cpu numpy arrays.
    """
    x_c, t_c = X_coll[:, 0], X_coll[:, 1]
    u_xx, u_tt = second_derivs(net, X_coll)

    in_t = (t_c.unsqueeze(0) >= T_LO) & (t_c.unsqueeze(0) < T_HI)
    in_x = (x_c.unsqueeze(0) >= X_LO) & (x_c.unsqueeze(0) < X_HI)
    masks = torch.cat([in_t, in_x], dim=0).to(u_xx.dtype)

    ut_ux = (u_tt * u_xx).unsqueeze(0)
    uxx_sq = (u_xx ** 2).unsqueeze(0)
    num = (masks * ut_ux).sum(dim=1)
    den = (masks * uxx_sq).sum(dim=1)
    c2_i = num / (den + eps)
    mu = c2_i.mean()

    return c2_i.detach().cpu().numpy(), float(mu.detach())


# ============================================================ full-grid derivatives
def evaluate_derivatives_on_grid(net, X_obs, Nx, Nt, chunk=20000):
    """Autograd u_xx and u_tt at every grid point in X_obs.

    X_obs: torch tensor (Nx*Nt, 2). Chunked to bound peak memory on dense
    grids; the wave problem (81x81 = 6561 points) fits in a single chunk.
    Returns (u_xx_grid, u_tt_grid) as numpy (Nx, Nt).
    """
    n = X_obs.shape[0]
    out_xx = np.empty(n, dtype=np.float32)
    out_tt = np.empty(n, dtype=np.float32)
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        u_xx, u_tt = second_derivs(net, X_obs[start:stop])
        out_xx[start:stop] = u_xx.detach().cpu().numpy()
        out_tt[start:stop] = u_tt.detach().cpu().numpy()
    return out_xx.reshape(Nx, Nt), out_tt.reshape(Nx, Nt)


# ============================================================ pure-numpy aggregations
def cv_forms(c2_i, c2_true=0.04, eps=EPS):
    """All CV-style metrics on a length-60 array of per-window c^2 estimates."""
    c2 = np.asarray(c2_i, dtype=np.float64)
    mean = float(c2.mean())
    median = float(np.median(c2))
    std = float(c2.std(ddof=1))
    cv2 = float(c2.var(ddof=1) / (mean ** 2 + eps))
    anchored_mse = float(((c2 - c2_true) ** 2).mean() / (c2_true ** 2))
    rel_bias = float((mean - c2_true) / c2_true)
    return dict(
        mean=mean, median=median, std=std, cv2=cv2,
        anchored_mse=anchored_mse, rel_bias=rel_bias,
        c2_min=float(c2.min()), c2_max=float(c2.max()),
    )


def error_stats(u_pred, u_data):
    """rel-L2, RMSE, and max |error| for a predicted field vs the data grid."""
    diff = (np.asarray(u_pred) - np.asarray(u_data)).ravel()
    rel_l2 = float(np.linalg.norm(diff) / (np.linalg.norm(u_data) + EPS))
    rmse = float(np.sqrt((diff ** 2).mean()))
    max_abs = float(np.abs(diff).max())
    return dict(rel_l2=rel_l2, rmse=rmse, max_abs=max_abs)


# ============================================================ smoke test
if __name__ == "__main__":
    # Analytical net: u(x, t) = sin(pi*x) * cos(2*pi*t).
    # Then u_xx = -pi^2 * u and u_tt = -4*pi^2 * u, so c^2 = u_tt / u_xx = 4
    # everywhere -- the sliding-window OLS should recover c^2 = 4.0 with
    # near-zero spread. This tests the helpers without training anything.
    PI = float(np.pi)

    class AnalyticalNet(torch.nn.Module):
        def forward(self, xt):
            x, t = xt[:, 0:1], xt[:, 1:2]
            return torch.sin(PI * x) * torch.cos(2.0 * PI * t)

    net = AnalyticalNet()

    # Same window grid the wave problem uses.
    t_lo, t_hi = make_windows(0.0, 1.0, 30, 0.5)
    x_lo, x_hi = make_windows(0.0, 1.0, 30, 0.5)
    T_LO = torch.tensor(t_lo).unsqueeze(1)
    T_HI = torch.tensor(t_hi).unsqueeze(1)
    X_LO = torch.tensor(x_lo).unsqueeze(1)
    X_HI = torch.tensor(x_hi).unsqueeze(1)

    g = torch.Generator().manual_seed(0)
    X_coll = torch.stack([torch.rand(8000, generator=g),
                          torch.rand(8000, generator=g)], dim=1)

    c2_i, mu = compute_c2_per_window(net, X_coll, X_LO, X_HI, T_LO, T_HI)
    forms = cv_forms(c2_i, c2_true=4.0)

    # Grid-derivative check at 81x81 (matches wave problem size).
    Nx = Nt = 81
    xg = np.linspace(0, 1, Nx, dtype=np.float32)
    tg = np.linspace(0, 1, Nt, dtype=np.float32)
    XX, TT = np.meshgrid(xg, tg, indexing="ij")
    X_obs = torch.tensor(np.stack([XX.ravel(), TT.ravel()], axis=1))
    u_xx_grid, u_tt_grid = evaluate_derivatives_on_grid(net, X_obs, Nx, Nt)
    u_grid = np.sin(PI * XX) * np.cos(2.0 * PI * TT)
    u_xx_expected = -(PI ** 2) * u_grid
    u_tt_expected = -(4.0 * PI ** 2) * u_grid
    err_xx = float(np.max(np.abs(u_xx_grid - u_xx_expected)))
    err_tt = float(np.max(np.abs(u_tt_grid - u_tt_expected)))

    print("=== cv_metric.py smoke test (u = sin(pi x) cos(2 pi t)) ===")
    print(f"expected c^2          = 4.0")
    print(f"mean(c^2_i)           = {forms['mean']:.6f}")
    print(f"median(c^2_i)         = {forms['median']:.6f}")
    print(f"cv2 = var/mean^2      = {forms['cv2']:.3e}")
    print(f"rel_bias              = {forms['rel_bias']:.3e}")
    print(f"c^2_i range           = [{forms['c2_min']:.6f}, {forms['c2_max']:.6f}]")
    print(f"max|u_xx_grid - true| = {err_xx:.3e}")
    print(f"max|u_tt_grid - true| = {err_tt:.3e}")

    ok = (abs(forms['mean'] - 4.0) / 4.0 < 0.01
          and forms['cv2'] < 0.05
          and err_xx < 1e-3 and err_tt < 1e-3)
    print("PASS" if ok else "FAIL")
