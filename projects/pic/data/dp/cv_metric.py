"""
Shared helpers for the double pendulum CV-vs-baseline PINN comparison,
generic-form version.

We treat the system as a HYPOTHESIS to verify: two candidate equations in
(theta1, theta2), each rearranged so a specific second derivative is the
target and the rest are RHS terms with unknown coefficients. The discovery
question is whether the per-window OLS-fitted coefficients are (a)
consistent across windows and (b) close to the values we secretly know
the right form would give.

Candidate equations (encoded in `EQUATIONS` below):

    Eq 1 (target = theta1_tt, K = 3):
        theta1_tt = c1*(theta2_tt*cos(theta1-theta2))
                  + c2*(theta2_t^2  *sin(theta1-theta2))
                  + c3*sin(theta1)
        true coefs  ~ [-0.5, -0.5, -g]

    Eq 2 (target = theta2_tt, K = 3):
        theta2_tt = c1*(theta1_tt*cos(theta1-theta2))
                  + c2*(theta1_t^2  *sin(theta1-theta2))
                  + c3*sin(theta2)
        true coefs  ~ [-1.0, 1.0, -g]

(Equal-mass, equal-length double pendulum: m1=m2=L1=L2=1. The factor of
-g lives inside the gravity-feature coefficient -- features are kept as
raw observables, mirroring how NS keeps `u_xx` raw and lets `theta_true`
carry the viscosity 0.01.)

Per equation, OLS over collocation points in each time window gives a
K-vector theta_i. CV^2 across windows is the consistency metric. The
true coefficients are not used in the OLS itself -- only in the
anchored-MSE flavour of the CV penalty (and for diagnostics).

The cross-coupling features (Eq 1 uses theta2_tt as a feature, Eq 2
uses theta1_tt) are fine: at every collocation point the OLS uses the
network's autograd `theta_tt`, which is just another column of the
design matrix. The two residuals must be satisfied jointly for the
network to be self-consistent.
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


def make_windows_circular(lo, hi, n, frac):
    """N windows of half-width frac*(hi-lo)/2 placed in [lo, hi) with wrap.

    Centers are at `lo + i*(hi-lo)/n` for i in [0, n). Returned (t_lo, t_hi)
    may extend past [lo, hi]; pair with `period = hi - lo` in
    `compute_equation_thetas` so the mask uses circular distance.

    Use this when treating the t-axis as periodic (only physically meaningful
    if the data actually loops). For chaotic DP trajectories it mixes
    physically unrelated states -- diagnostic only.
    """
    w = (hi - lo) * frac
    h = w / 2
    c = np.linspace(lo, hi, n, endpoint=False, dtype=np.float32)
    return c - h, c + h


# ============================================================ autograd
def network_derivatives_dp(net, T):
    """Compute (theta1, theta2, omega1, omega2, alpha1, alpha2) on T.

    T: (N, 1) collocation times. Returns dict of (N,) tensors with the
    field values and their first/second time derivatives (omega = theta_t,
    alpha = theta_tt).
    """
    T = T.clone().requires_grad_(True)
    out = net(T)                                    # (N, 2): [theta1, theta2]
    th1 = out[:, 0]
    th2 = out[:, 1]
    ones = torch.ones_like(th1)
    w1 = torch.autograd.grad(th1, T, ones, create_graph=True)[0][:, 0]
    w2 = torch.autograd.grad(th2, T, ones, create_graph=True)[0][:, 0]
    a1 = torch.autograd.grad(w1,  T, ones, create_graph=True)[0][:, 0]
    a2 = torch.autograd.grad(w2,  T, ones, create_graph=True)[0][:, 0]
    return dict(
        theta1=th1, theta2=th2,
        omega1=w1,  omega2=w2,
        alpha1=a1,  alpha2=a2,
    )


# ============================================================ candidate equation specs
def _eq1_tf(d):
    """target = alpha1; features = [alpha2*cos(dth), omega2^2*sin(dth), sin(theta1)]."""
    dth = d["theta1"] - d["theta2"]
    return d["alpha1"], torch.stack([
        d["alpha2"] * torch.cos(dth),
        d["omega2"] ** 2 * torch.sin(dth),
        torch.sin(d["theta1"]),
    ], dim=1)


def _eq2_tf(d):
    """target = alpha2; features = [alpha1*cos(dth), omega1^2*sin(dth), sin(theta2)]."""
    dth = d["theta1"] - d["theta2"]
    return d["alpha2"], torch.stack([
        d["alpha1"] * torch.cos(dth),
        d["omega1"] ** 2 * torch.sin(dth),
        torch.sin(d["theta2"]),
    ], dim=1)


# `theta_true` is populated at module import time after the .npz reader at the
# bottom of the file (so callers can `import EQUATIONS` and find truth baked in).
# The values below are placeholders for the equal-mass, equal-length case with
# g = 9.81 -- override via the helper if you change `dp.npz` parameters.
_G_DEFAULT = 9.81

EQUATIONS = [
    {
        "name": "th1",
        "k": 3,
        "feature_names": ("alpha2*cos(dth)", "omega2^2*sin(dth)", "sin(theta1)"),
        "theta_true": np.array([-0.5, -0.5, -_G_DEFAULT], dtype=np.float64),
        "target_and_features": _eq1_tf,
    },
    {
        "name": "th2",
        "k": 3,
        "feature_names": ("alpha1*cos(dth)", "omega1^2*sin(dth)", "sin(theta2)"),
        "theta_true": np.array([-1.0,  1.0, -_G_DEFAULT], dtype=np.float64),
        "target_and_features": _eq2_tf,
    },
]

EQ_NAMES = tuple(e["name"] for e in EQUATIONS)


def set_gravity(g):
    """Refresh `theta_true` in `EQUATIONS` to use the supplied g (called from PINNs)."""
    EQUATIONS[0]["theta_true"] = np.array([-0.5, -0.5, -float(g)], dtype=np.float64)
    EQUATIONS[1]["theta_true"] = np.array([-1.0,  1.0, -float(g)], dtype=np.float64)


# ============================================================ per-window OLS
def ols_per_window(target, features, mask, ridge=EPS):
    """Weighted normal equations with adaptive (AtA-relative) ridge.

    The ridge term is scaled by the per-window mean diagonal of AtA, so
    a fixed small `ridge` value gives Tikhonov regularization that's
    negligible at any amplitude (relative to AtA itself). This avoids
    the small-amplitude "ridge dominates → theta collapses to zero"
    failure mode that a fixed absolute ridge has.

    target:    (N,) torch tensor
    features:  (N, K) torch tensor
    mask:      (N_WIN, N) torch tensor (boolean -> float)
    Returns (theta: (N_WIN, K), theta_mean: (K,)).
    """
    A = features
    y = target
    AtA = torch.einsum('wk,kj,kl->wjl', mask, A, A)
    Aty = torch.einsum('wk,kj,k->wj',   mask, A, y)
    K = A.shape[1]
    I = torch.eye(K, device=A.device, dtype=A.dtype).unsqueeze(0)
    # Adaptive ridge: scale with each window's mean diagonal of AtA.
    scale = AtA.diagonal(dim1=-2, dim2=-1).mean(dim=-1).unsqueeze(-1).unsqueeze(-1)
    theta = torch.linalg.solve(AtA + ridge * scale * I,
                               Aty.unsqueeze(-1)).squeeze(-1)
    return theta, theta.mean(dim=0)


def compute_equation_thetas(net, T_coll, T_LO, T_HI, ridge=EPS, period=None):
    """Run autograd once, then OLS for both candidate equations.

    Mask logic:
    - `period=None` (default): rectangular mask `t in [T_LO, T_HI)`.
    - `period=float`: circular mask. Center is `(T_LO + T_HI) / 2`, half-
      width is `(T_HI - T_LO) / 2`. A point `t` is in the window iff
      `min(|t - center|, period - |t - center|) < half_w`. Use
      `period = T_MAX - T_MIN` to treat the t-axis as a loop.

    Returns dict mapping equation name -> (theta_per_window, theta_mean),
    plus the derivs dict so callers can reuse them for residuals,
    plus the mask (N_WIN, N_COLL) so the residual loss can reuse it.
    """
    derivs = network_derivatives_dp(net, T_coll)
    t_flat = T_coll[:, 0]
    if period is None:
        in_t = (t_flat.unsqueeze(0) >= T_LO) & (t_flat.unsqueeze(0) < T_HI)
    else:
        centers = (T_LO + T_HI) / 2.0
        half_w  = (T_HI - T_LO) / 2.0
        d = (t_flat.unsqueeze(0) - centers).abs()
        d_circ = torch.minimum(d, period - d)
        in_t = d_circ < half_w
    mask = in_t.to(derivs["theta1"].dtype)

    results = {}
    for eq in EQUATIONS:
        y, A = eq["target_and_features"](derivs)
        theta, theta_mean = ols_per_window(y, A, mask, ridge=ridge)
        results[eq["name"]] = (theta, theta_mean)
    return results, derivs, mask


def per_point_residual_variance(theta_pw, y, A, mask, eps=EPS):
    """Variance across covering windows of per-window residuals, averaged over points.

    For each collocation point k, gather the residuals r_ik = y_k - A_k @ theta_i
    from every window i covering k (mask[i, k] = 1) and take the population
    variance across those windows. Return the mean over points.

    Inputs:
        theta_pw: (N_WIN, K)   torch tensor, gradient-bearing OLS theta per window.
        y:        (N,)         torch tensor target values.
        A:        (N, K)       torch tensor features.
        mask:     (N_WIN, N)   torch float (1 if point n in window i).
    Returns:
        scalar torch tensor (mean over collocation points of the per-point variance).
    """
    pred = A @ theta_pw.t()                                  # (N, N_WIN)
    r = y.unsqueeze(1) - pred                                # (N, N_WIN)
    mT = mask.t()                                            # (N, N_WIN)
    n_k = mT.sum(dim=1)                                      # (N,)
    mean_k = (mT * r).sum(dim=1) / (n_k)               # (N,)
    var_k = (mT * (r - mean_k.unsqueeze(1)) ** 2).sum(dim=1) / (n_k)
    return var_k.mean()


# ============================================================ residuals
def equation_residual(target, features, theta):
    """target: (N,), features: (N, K), theta: (K,) or (N, K)."""
    if theta.ndim == 1:
        pred = features @ theta
    else:
        pred = (features * theta).sum(dim=1)
    return target - pred


# ============================================================ pure-numpy aggregations
def cv_forms(theta_per_window, theta_true, eps=EPS):
    """Per-coefficient stats + sums (per equation).

    theta_per_window: (N_WIN, K) numpy
    theta_true:       (K,)       numpy
    Returns dict with per-feature stats + cv2_sum + anchored_mse_sum.
    """
    tw = np.asarray(theta_per_window, dtype=np.float64)
    tt = np.asarray(theta_true, dtype=np.float64)
    K = tw.shape[1]
    out = {}
    for j in range(K):
        c = tw[:, j]
        mean = float(c.mean())
        if c.size > 1:
            std = float(c.std(ddof=1))
            cv2 = float(c.var(ddof=1) / (mean ** 2 + eps))
        else:
            std = 0.0
            cv2 = 0.0
        anch = float(((c - tt[j]) ** 2).mean() / (tt[j] ** 2 + eps))
        out[j] = dict(
            mean=mean, median=float(np.median(c)), std=std,
            cv2=cv2, anchored_mse=anch,
            rel_bias=float((mean - tt[j]) / (tt[j] + eps)),
            min=float(c.min()), max=float(c.max()),
        )
    out["cv2_sum"]          = sum(out[j]["cv2"]          for j in range(K))
    out["anchored_mse_sum"] = sum(out[j]["anchored_mse"] for j in range(K))
    return out


def error_stats(pred, data):
    """rel-L2 / RMSE / max-abs error on a predicted field."""
    diff = (np.asarray(pred) - np.asarray(data)).ravel()
    rel_l2 = float(np.linalg.norm(diff) / (np.linalg.norm(data) + EPS))
    rmse = float(np.sqrt((diff ** 2).mean()))
    max_abs = float(np.abs(diff).max())
    return dict(rel_l2=rel_l2, rmse=rmse, max_abs=max_abs)


# ============================================================ full-grid evaluation
def evaluate_on_grid(net, T_grid, chunk=20000):
    """Run network + autograd derivatives over T_grid in chunks. (N, 1) -> dict."""
    n = T_grid.shape[0]
    keys = ['theta1', 'theta2', 'omega1', 'omega2', 'alpha1', 'alpha2']
    out = {k: np.empty(n, dtype=np.float32) for k in keys}
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        d = network_derivatives_dp(net, T_grid[start:stop])
        for k in keys:
            out[k][start:stop] = d[k].detach().cpu().numpy()
    return out


def grid_equation_residuals(grid_derivs, thetas_by_eq):
    """Compute equation residuals on the grid (numpy via torch).

    grid_derivs: dict of numpy arrays of length N.
    thetas_by_eq: dict mapping equation name -> (K,) numpy coef vector.

    Returns dict mapping equation name -> (N,) residual array.
    """
    g = {k: torch.tensor(v) for k, v in grid_derivs.items()}
    res = {}
    for eq in EQUATIONS:
        y, A = eq["target_and_features"](g)
        c = torch.tensor(thetas_by_eq[eq["name"]], dtype=A.dtype)
        r = (y - A @ c).numpy()
        res[eq["name"]] = r
    return res


# ============================================================ smoke test
if __name__ == "__main__":
    # Synthetic test: random features + known coefficients per equation.
    # Verify the batched per-window solver recovers both equations.
    torch.manual_seed(0)
    N = 500
    N_WIN = 5
    ppw = N // N_WIN
    mask = torch.zeros(N_WIN, N)
    for i in range(N_WIN):
        mask[i, i * ppw:(i + 1) * ppw] = 1.0

    for eq in EQUATIONS:
        K = eq["k"]
        theta_true = torch.tensor(eq["theta_true"], dtype=torch.float32)
        A_synth = torch.randn(N, K)
        y_synth = A_synth @ theta_true
        theta, theta_mean = ols_per_window(y_synth, A_synth, mask, ridge=EPS)
        err = (theta - theta_true.unsqueeze(0)).abs().max().item()
        print(f"[{eq['name']:>4}] K={K:>2}  max|theta_i - true| = {err:.3e}   "
              f"theta_mean = {theta_mean.numpy().round(4).tolist()}   "
              f"true = {eq['theta_true'].tolist()}")
