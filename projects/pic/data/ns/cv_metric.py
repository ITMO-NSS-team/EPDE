"""
Shared helpers for the NS-PINN CV-vs-baseline comparison, generic-form
version.

We treat the system as a HYPOTHESIS to verify: three candidate equations
in (u, v, p), each rearranged so a specific derivative is the target and
the rest are RHS terms with unknown coefficients. The discovery question
is whether the per-window OLS-fitted coefficients are (a) consistent
across windows and (b) close to the values we secretly know the right
form would give.

Candidate equations (encoded in `EQUATIONS` below):

    Eq 1 (target = u_t, K = 5):
        u_t = c1*(u*u_x) + c2*(v*u_y) + c3*P_x + c4*u_xx + c5*u_yy
        true coefs ~ [-1, -1, -1, 0.01, 0.01]

    Eq 2 (target = v_y, K = 1):    [continuity, u_x + v_y = 0]
        v_y = c1*u_x
        true coef  ~ [-1]

    Eq 3 (target = P_y, K = 5):    [v-momentum solved for P_y]
        P_y = c1*v_t + c2*(u*v_x) + c3*(v*v_y) + c4*v_xx + c5*v_yy
        true coefs ~ [-1, -1, -1, 0.01, 0.01]

Per equation, OLS over collocation points in each time window gives a
K-vector theta_i. CV^2 across windows is the consistency metric. We
deliberately do NOT use the true coefficients in any training term --
they're only printed for diagnostics in the comparison script.
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
def network_derivatives_ns(net, X):
    """Compute u, v, p and the eleven derivatives. X: (N, 3) = (x, y, t)."""
    X = X.clone().requires_grad_(True)
    uvp = net(X)
    u, v, p = uvp[:, 0], uvp[:, 1], uvp[:, 2]
    ones = torch.ones_like(u)
    grad_u = torch.autograd.grad(u, X, ones, create_graph=True)[0]
    grad_v = torch.autograd.grad(v, X, ones, create_graph=True)[0]
    grad_p = torch.autograd.grad(p, X, ones, create_graph=True)[0]
    u_x, u_y, u_t = grad_u[:, 0], grad_u[:, 1], grad_u[:, 2]
    v_x, v_y, v_t = grad_v[:, 0], grad_v[:, 1], grad_v[:, 2]
    p_x, p_y     = grad_p[:, 0], grad_p[:, 1]
    grad_ux = torch.autograd.grad(u_x, X, ones, create_graph=True)[0]
    grad_uy = torch.autograd.grad(u_y, X, ones, create_graph=True)[0]
    grad_vx = torch.autograd.grad(v_x, X, ones, create_graph=True)[0]
    grad_vy = torch.autograd.grad(v_y, X, ones, create_graph=True)[0]
    u_xx, u_yy = grad_ux[:, 0], grad_uy[:, 1]
    v_xx, v_yy = grad_vx[:, 0], grad_vy[:, 1]
    return dict(u=u, v=v, p=p,
                u_t=u_t, u_x=u_x, u_y=u_y, u_xx=u_xx, u_yy=u_yy,
                v_t=v_t, v_x=v_x, v_y=v_y, v_xx=v_xx, v_yy=v_yy,
                p_x=p_x, p_y=p_y)


# ============================================================ candidate equation specs
def _eq1_tf(d):
    """target = u_t; features = [u*u_x, v*u_y, p_x, u_xx, u_yy]."""
    return d["u_t"], torch.stack([d["u"] * d["u_x"],
                                   d["v"] * d["u_y"],
                                   d["p_x"],
                                   d["u_xx"],
                                   d["u_yy"]], dim=1)


def _eq2_tf(d):
    """target = v_y; features = [u_x]."""
    return d["v_y"], d["u_x"].unsqueeze(1)


def _eq3_tf(d):
    """target = p_y; features = [v_t, u*v_x, v*v_y, v_xx, v_yy]."""
    return d["p_y"], torch.stack([d["v_t"],
                                   d["u"] * d["v_x"],
                                   d["v"] * d["v_y"],
                                   d["v_xx"],
                                   d["v_yy"]], dim=1)


EQUATIONS = [
    {
        "name": "u_t",
        "k": 5,
        "feature_names": ("u*u_x", "v*u_y", "p_x", "u_xx", "u_yy"),
        "theta_true": np.array([-1.0, -1.0, -1.0, 0.01, 0.01], dtype=np.float64),
        "target_and_features": _eq1_tf,
    },
    {
        "name": "v_y",
        "k": 1,
        "feature_names": ("u_x",),
        "theta_true": np.array([-1.0], dtype=np.float64),
        "target_and_features": _eq2_tf,
    },
    {
        "name": "p_y",
        "k": 5,
        "feature_names": ("v_t", "u*v_x", "v*v_y", "v_xx", "v_yy"),
        "theta_true": np.array([-1.0, -1.0, -1.0, 0.01, 0.01], dtype=np.float64),
        "target_and_features": _eq3_tf,
    },
]

EQ_NAMES = tuple(e["name"] for e in EQUATIONS)


# ============================================================ per-window OLS
def ols_per_window(target, features, mask, ridge=EPS):
    """Weighted normal equations -> theta_per_window, theta_mean.

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
    theta = torch.linalg.solve(AtA + ridge * I,
                               Aty.unsqueeze(-1)).squeeze(-1)
    return theta, theta.mean(dim=0)


def compute_equation_thetas(net, X_coll, T_LO, T_HI, ridge=EPS):
    """Run autograd once, then OLS for all three candidate equations.

    Returns dict mapping equation name -> (theta_per_window, theta_mean).
    Also returns the derivs dict so callers can reuse them for residuals.
    """
    derivs = network_derivatives_ns(net, X_coll)
    t_flat = X_coll[:, 2]
    in_t = (t_flat.unsqueeze(0) >= T_LO) & (t_flat.unsqueeze(0) < T_HI)
    mask = in_t.to(derivs["u"].dtype)

    results = {}
    for eq in EQUATIONS:
        y, A = eq["target_and_features"](derivs)
        theta, theta_mean = ols_per_window(y, A, mask, ridge=ridge)
        results[eq["name"]] = (theta, theta_mean)
    return results, derivs, mask


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
        # std with ddof=1 needs N_WIN > 1; baseline broadcasts a scalar so
        # std is 0 and cv2 is 0 -- well-defined and fine.
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
def evaluate_on_grid(net, X_grid, chunk=20000):
    """Run network + autograd derivatives over X_grid in chunks. (N, 3) -> dict."""
    n = X_grid.shape[0]
    keys = ['u', 'v', 'p', 'u_t', 'u_x', 'u_y', 'u_xx', 'u_yy',
            'v_t', 'v_x', 'v_y', 'v_xx', 'v_yy', 'p_x', 'p_y']
    out = {k: np.empty(n, dtype=np.float32) for k in keys}
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        d = network_derivatives_ns(net, X_grid[start:stop])
        for k in keys:
            out[k][start:stop] = d[k].detach().cpu().numpy()
    return out


def grid_equation_residuals(grid_derivs, thetas_by_eq):
    """Compute equation residuals on the grid (numpy, no torch).

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
    # Synthetic test: build u,v,p that exactly satisfy the 3 equations with
    # known coefficients, evaluate ols_per_window over random masks, verify
    # recovery.
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
