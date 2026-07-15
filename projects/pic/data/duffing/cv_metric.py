"""
Shared helpers for the Duffing CV-vs-baseline PINN comparison.

Forced Duffing in regression form:

    x_tt = -alpha*x - beta*x^3 - delta*x_t + gamma*cos(omega*t)

For each time-window W_i (N_WIN strips along t, each WIN_FRAC*(T_max - T_min)
wide), the per-window OLS over collocation points inside W_i recovers a
4-vector (alpha_i, beta_i, delta_i, gamma_i) by solving the normal
equations with a small ridge:

    A_i^T A_i  theta_i  =  A_i^T y_i  +  eps * I   (4x4 system)

where columns of A are [-x, -x^3, -x_t, cos(omega*t)] and y is x_tt.

Used by both `pinn_test_baseline.py` (post-training diagnostic) and
`pinn_test_cv.py` (training loss + post-training diagnostic) so both
runs produce bit-identical artifacts when fed the same eval pool.
"""

import numpy as np
import torch


EPS = 1e-8


# ============================================================ windows
def make_windows(lo, hi, n, frac):
    """N windows of half-width frac*(hi-lo)/2 evenly centered in [lo+h, hi-h]."""
    w = (hi - lo) * frac
    h = w / 2
    c = np.linspace(lo + h, hi - h, n, dtype=np.float32)
    return c - h, c + h


# ============================================================ autograd
def first_second_derivs_t(net, t):
    """For input `t` shape (N, 1), return (x, dx_dt, d2x_dt2) each (N,)."""
    t = t.clone().requires_grad_(True)
    x = net(t)
    dx_dt = torch.autograd.grad(x, t, torch.ones_like(x), create_graph=True)[0]
    d2x_dt2 = torch.autograd.grad(dx_dt, t, torch.ones_like(dx_dt),
                                  create_graph=True)[0]
    return x.squeeze(-1), dx_dt.squeeze(-1), d2x_dt2.squeeze(-1)


# ============================================================ multi-coefficient OLS
def build_features(t_coll, x, dx_dt, omega):
    """Design matrix shape (N, 4) with cols [-x, -x^3, -dx_dt, cos(omega*t)].

    With target `d2x_dt2`, OLS recovers (alpha, beta, delta, gamma) directly
    -- no sign flips needed.
    """
    return torch.stack([
        -x,
        -(x ** 3),
        -dx_dt,
        torch.cos(omega * t_coll),
    ], dim=1)


def coefs_from_derivs(t_flat, x, dx_dt, d2x_dt2, T_LO, T_HI, omega, ridge=EPS):
    """Per-window OLS given pre-computed derivatives. No autograd here.

    Lets callers share a single autograd pass between the OLS and other
    loss terms (e.g., the standard PDE residual). Returns the same shape
    as `compute_coefs_per_window`.
    """
    A = build_features(t_flat, x, dx_dt, omega)            # (N_COLL, 4)
    y = d2x_dt2                                            # (N_COLL,)

    in_t = (t_flat.unsqueeze(0) >= T_LO) & (t_flat.unsqueeze(0) < T_HI)
    mask = in_t.to(A.dtype)                                # (N_WIN, N_COLL)

    # Per-window normal equations.
    AtA = torch.einsum('wk,kj,kl->wjl', mask, A, A)        # (N_WIN, 4, 4)
    Aty = torch.einsum('wk,kj,k->wj',   mask, A, y)        # (N_WIN, 4)

    I = torch.eye(4, device=A.device, dtype=A.dtype).unsqueeze(0)
    theta = torch.linalg.solve(AtA + ridge * I,
                               Aty.unsqueeze(-1)).squeeze(-1)  # (N_WIN, 4)
    return theta, theta.mean(dim=0)


def compute_coefs_per_window(net, T_coll, T_LO, T_HI, omega, ridge=EPS):
    """Per-window OLS (alpha, beta, delta, gamma) -- standalone path.

    T_coll: (N_COLL, 1) collocation times.
    T_LO, T_HI: (N_WIN, 1) window bounds.

    Returns (theta_per_window, theta_mean):
      theta_per_window: (N_WIN, 4) torch tensor, columns (alpha, beta, delta, gamma).
      theta_mean:       (4,)       torch tensor, mean across windows.
    """
    t_flat = T_coll[:, 0]                                  # (N_COLL,)
    x, dx_dt, d2x_dt2 = first_second_derivs_t(net, T_coll)
    return coefs_from_derivs(t_flat, x, dx_dt, d2x_dt2, T_LO, T_HI, omega, ridge)


# ============================================================ pure-numpy aggregations
COEF_NAMES = ('alpha', 'beta', 'delta', 'gamma')


def cv_forms(theta_per_window, theta_true, eps=EPS):
    """Per-coefficient stats + summed aggregates.

    theta_per_window: (N_WIN, 4) array.
    theta_true:       (4,) array of ground-truth coefficients.

    Returns a dict {alpha: {...}, beta: {...}, ..., cv2_sum: float, anchored_mse_sum: float}.
    """
    tw = np.asarray(theta_per_window, dtype=np.float64)
    tt = np.asarray(theta_true, dtype=np.float64)
    out = {}
    for j, name in enumerate(COEF_NAMES):
        c = tw[:, j]
        mean = float(c.mean())
        cv2 = float(c.var(ddof=1) / (mean ** 2 + eps))
        anchored = float(((c - tt[j]) ** 2).mean() / (tt[j] ** 2))
        out[name] = dict(
            mean=mean, median=float(np.median(c)), std=float(c.std(ddof=1)),
            cv2=cv2, anchored_mse=anchored,
            rel_bias=float((mean - tt[j]) / tt[j]),
            min=float(c.min()), max=float(c.max()),
        )
    out['cv2_sum'] = sum(out[n]['cv2'] for n in COEF_NAMES)
    out['anchored_mse_sum'] = sum(out[n]['anchored_mse'] for n in COEF_NAMES)
    return out


def error_stats(x_pred, x_data):
    """rel-L2, RMSE, and max |error| for the predicted trajectory."""
    diff = (np.asarray(x_pred) - np.asarray(x_data)).ravel()
    rel_l2 = float(np.linalg.norm(diff) / (np.linalg.norm(x_data) + EPS))
    rmse = float(np.sqrt((diff ** 2).mean()))
    max_abs = float(np.abs(diff).max())
    return dict(rel_l2=rel_l2, rmse=rmse, max_abs=max_abs)


# ============================================================ full-grid derivatives + residual
def evaluate_derivatives_on_grid(net, T_obs, chunk=20000):
    """Autograd x, dx_dt, d2x_dt2 at every grid point in T_obs.

    T_obs: torch (N, 1). Returns three numpy arrays of length N.
    """
    n = T_obs.shape[0]
    out_x = np.empty(n, dtype=np.float32)
    out_v = np.empty(n, dtype=np.float32)
    out_a = np.empty(n, dtype=np.float32)
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        x, dx_dt, d2x_dt2 = first_second_derivs_t(net, T_obs[start:stop])
        out_x[start:stop] = x.detach().cpu().numpy()
        out_v[start:stop] = dx_dt.detach().cpu().numpy()
        out_a[start:stop] = d2x_dt2.detach().cpu().numpy()
    return out_x, out_v, out_a


def ode_residual_from_grid(t_grid, x_grid, v_grid, a_grid,
                           alpha, beta, delta, gamma, omega):
    """ODE residual r(t) = x_tt + delta*x_t + alpha*x + beta*x^3 - gamma*cos(omega t).

    All inputs numpy arrays of length N. Returns r of length N.
    """
    return (a_grid + delta * v_grid + alpha * x_grid + beta * x_grid ** 3
            - gamma * np.cos(omega * t_grid))


# ============================================================ smoke test
if __name__ == "__main__":
    # Synthetic OLS recovery test: random features, known coefficients, verify
    # the batched per-window solver recovers them across 5 disjoint windows.
    torch.manual_seed(0)
    theta_true = torch.tensor([1.3, -0.7, 0.2, 0.5])  # arbitrary
    N_COLL = 300
    N_WIN = 5
    points_per_window = N_COLL // N_WIN

    A_synth = torch.randn(N_COLL, 4)
    y_synth = A_synth @ theta_true
    # No noise -- ridge eps is tiny so recovery should be exact to float32.

    mask = torch.zeros(N_WIN, N_COLL)
    for i in range(N_WIN):
        mask[i, i * points_per_window:(i + 1) * points_per_window] = 1.0

    AtA = torch.einsum('wk,kj,kl->wjl', mask, A_synth, A_synth)
    Aty = torch.einsum('wk,kj,k->wj',   mask, A_synth, y_synth)
    I = EPS * torch.eye(4).unsqueeze(0)
    theta = torch.linalg.solve(AtA + I, Aty.unsqueeze(-1)).squeeze(-1)

    err = (theta - theta_true.unsqueeze(0)).abs().max().item()
    print("=== cv_metric.py smoke test (synthetic 4-feature OLS) ===")
    print(f"theta_true   = {theta_true.tolist()}")
    print(f"theta[0]     = {theta[0].tolist()}")
    print(f"theta[-1]    = {theta[-1].tolist()}")
    print(f"max |theta_i - theta_true| = {err:.3e}")
    print("PASS" if err < 1e-3 else "FAIL")

    # Sanity check on `compute_coefs_per_window` using an analytical x(t).
    # Use a function for which the regression coefficients are known.
    # Choose x(t) = A*sin(omega*t), then:
    #   x_t  = A*omega*cos(omega*t)
    #   x_tt = -A*omega^2*sin(omega*t)
    # Plug into x_tt + delta*x_t + alpha*x + beta*x^3 = gamma*cos(omega*t):
    #   -A*omega^2*sin + delta*A*omega*cos + alpha*A*sin + beta*A^3*sin^3
    #     = gamma*cos
    # Using sin^3 = (3 sin - sin(3w))/4, the 3w-harmonic prevents exact closure
    # unless beta = 0. So this analytical check only works for beta = 0; we use
    # it to verify the linear part of the regression -- not all 4 coefficients.
    # Skipping a full analytical check; the synthetic-OLS test above is the
    # primary correctness gate.
