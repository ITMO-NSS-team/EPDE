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


def het_per_window(target, features, mask, ridge=EPS,
                   detach_calibration=True):
    """Calibrated heterogeneity of the per-window OLS recovery (torch,
    differentiable) -- the DerSimonian-Laird excess-variance score::

        s2_wj   = sigma2_w * [(A_w^T A_w)^{-1}]_jj     (sampling variance)
        Q_j     = sum_w (theta_wj - theta_bar_j)^2 / s2_wj
        tau2_j  = max(0, (Q_j - df_j) / C_j)
        score_j = tau2_j / (tau2_j + theta_bar_j^2)     in [0, 1)
        score_raw_j = tau2_j / theta_bar_j^2            in [0, inf)

    This is raw cv2 with the part of the window-to-window variance the
    window fits' own standard errors already predict SUBTRACTED, and a
    bounded singularity-free denominator. It separates the two cases raw
    cv2 conflates: score ~ 0 with SMALL ``se_rel`` = genuinely consistent
    recovery; score ~ 0 with HUGE ``se_rel`` = no evidence (degenerate /
    parked trajectory -- ridge-collapsed thetas agreeing about nothing).

    TRUTH-FREE: uses only (target, features) from the net's trajectory.
    As a LOSS it must be paired with a collapse blocker (physics + IC):
    a fully degenerate state has zero evidence and therefore ZERO het --
    the score rewards degeneracy, it cannot forbid it. Windows whose
    Gram is non-invertible or whose diag of the inverse is non-positive
    contribute no evidence (weight 0). With heavily OVERLAPPING windows
    the theta_w are correlated, so the chi-square calibration (df = usable
    windows - 1) is conservative: correlated sampling deviations shrink Q,
    biasing tau2 toward 0 -- genuine drift still dominates.

    With ``detach_calibration=True`` (default) the evidence weights
    (s2-derived w, S1, S2, C, df) are DETACHED from the graph: the
    gradient of the score flows only through the per-window thetas and
    their weighted mean. Without it, a het LOSS has a second descent
    direction -- inflate your own standard errors until the evidence
    (and the loss) vanishes -- i.e. the degeneracy-reward hole becomes an
    active gradient incentive. Diagnostics (no_grad) are unaffected.

    ``score_raw`` is the UNBOUNDED form, the ``var / mu^2`` shape of
    ``epde.operators.common.survival.heterogeneity_raw_scores``: the same
    tau2 over the bare squared level. It keeps ordering two states that
    both saturate ``score`` near 1, and its gradient does not fade as the
    inconsistency grows (``score``'s does, as ``theta_bar^2/denom^2``).
    A level at float zero is clamped to machine eps rather than sent to
    inf: a loss term must stay finite, and eps is the same floor this
    function already uses for rss. Unlike the package form there is no
    unresolved-level guard -- the bounded ``score`` here has none either.

    Inputs: target (N,), features (N, K), mask (N_WIN, N) float.
    Returns dict of tensors: ``score`` (K,), ``score_raw`` (K,), ``tau2`` (K,),
    ``theta_bar`` (K,), ``se_rel`` (K,) relative standard error
    sqrt(mean s2)/|theta_bar| (inf = no signal), ``n_valid`` (K,).
    """
    A = features
    y = target
    K = A.shape[1]
    tiny = torch.finfo(A.dtype).eps
    AtA = torch.einsum('wk,kj,kl->wjl', mask, A, A)
    Aty = torch.einsum('wk,kj,k->wj', mask, A, y)
    yy = mask @ (y * y)                                     # (W,)
    I = torch.eye(K, device=A.device, dtype=A.dtype).unsqueeze(0)
    scale = AtA.diagonal(dim1=-2, dim2=-1).mean(dim=-1).unsqueeze(-1).unsqueeze(-1)
    Ainv = torch.linalg.inv(AtA + (ridge * scale + tiny) * I)
    theta = torch.einsum('wjl,wl->wj', Ainv, Aty)           # (W, K)
    cnt = mask.sum(dim=1)                                   # (W,)
    dof = (cnt - K).clamp(min=1.0)
    rss = (yy - 2.0 * (theta * Aty).sum(dim=1)
           + torch.einsum('wj,wjl,wl->w', theta, AtA, theta)).clamp(min=0.0)
    # Exact-fit floor (units of y^2): a window may not claim infinite
    # precision when rss cancels to float zero.
    sigma2 = torch.maximum(rss / dof, tiny * yy / dof)
    d = Ainv.diagonal(dim1=-2, dim2=-1)                     # (W, K)
    s2 = sigma2.unsqueeze(1) * d
    s2_cal = s2.detach() if detach_calibration else s2
    valid = (s2_cal > 0) & torch.isfinite(s2_cal) \
        & (cnt.unsqueeze(1) >= K + 1)
    w = torch.where(valid,
                    1.0 / torch.where(valid, s2_cal, torch.ones_like(s2_cal)),
                    torch.zeros_like(s2_cal))
    S1 = w.sum(dim=0)
    S2 = (w ** 2).sum(dim=0)
    n_valid = valid.sum(dim=0).to(A.dtype)
    theta_bar = (w * theta).sum(dim=0) / S1.clamp(min=tiny)
    Q = (w * (theta - theta_bar.unsqueeze(0)) ** 2).sum(dim=0)
    df_j = (n_valid - 1.0).clamp(min=1.0)
    C = (S1 - S2 / S1.clamp(min=tiny)).clamp(min=tiny)
    tau2 = ((Q - df_j) / C).clamp(min=0.0)
    denom = tau2 + theta_bar ** 2
    score = torch.where(denom > 0, tau2 / denom, torch.zeros_like(denom))
    score_raw = tau2 / (theta_bar ** 2).clamp(min=tiny)
    mean_s2 = torch.where(valid, s2, torch.zeros_like(s2)).sum(dim=0) \
        / n_valid.clamp(min=1.0)
    se_rel = torch.where(n_valid > 0, mean_s2.sqrt() / theta_bar.abs(),
                         torch.full_like(theta_bar, float('inf')))
    return dict(score=score, score_raw=score_raw, tau2=tau2,
                theta_bar=theta_bar, se_rel=se_rel, n_valid=n_valid)


def global_ols(target, features, weights=None, ridge=EPS):
    """ONE guarded global weighted OLS -> (theta, residual).

    THE single solve shared by every consumer of a data-driven
    coefficient: ``chi2_per_term``'s statistic and any loss term that
    wants the net's own recovery. Two independent solves inside one loss
    is a correctness trap -- the physics residual and the statistic
    would then be talking about different thetas.

    Guard convention (het's, not ``ols_per_window``'s): an AtA-relative
    ridge PLUS an absolute ``tiny``, so a parked net (AtA -> 0) yields a
    guarded solve instead of a LinAlgError, with a minimum-norm lstsq
    fallback if the guard is defeated outright.

    target (N,), features (N, K), optional weights (N,). Gradient-bearing.
    """
    A = features
    y = target
    K = A.shape[1]
    tiny = torch.finfo(A.dtype).eps
    w = weights if weights is not None else torch.ones_like(y)

    wA = w.unsqueeze(1) * A
    AtA = A.t() @ wA
    Aty = wA.t() @ y
    I = torch.eye(K, device=A.device, dtype=A.dtype)
    scale = AtA.diagonal().mean()
    guarded = AtA + (ridge * scale + tiny) * I
    try:
        c = torch.linalg.solve(guarded, Aty)
    except torch.linalg.LinAlgError:
        # A fully degenerate trajectory (e.g. an LBFGS line-search probe
        # saturating the net: derivatives -> 0, AtA -> 0) can defeat the
        # ridge guard outright. Fall back to the minimum-norm solution --
        # the survival._solve_gram lstsq convention. Measured live: the
        # pure-statistics cell (no coercive physics/IC term) crashed
        # here mid-line-search.
        c = torch.linalg.lstsq(guarded, Aty.unsqueeze(-1)).solution.squeeze(-1)
    return c, y - A @ c


def tls_scales(target, features):
    """Column yardsticks for ``tls_fit``: the RMS of every column of
    ``[target | features]``, computed ONCE from the OBSERVED (finite-
    difference) design -- never from the net's field, for the reason
    ``anchor_scales`` gives (a net-derived yardstick is a free variable the
    net can game). Uncentred, because the TLS relation has no intercept.
    Floored at the dtype's smallest positive value, not at EPS, so the
    scale-invariance that licenses weight 1 survives a rescaled equation.
    """
    Z = torch.cat([target.unsqueeze(1), features], dim=1)
    rms = (Z ** 2).mean(dim=0).sqrt()
    return rms.clamp(min=float(torch.finfo(Z.dtype).tiny))


def tls_fit(target, features, col_scale):
    """Total least squares on ``[target | features] / col_scale``.

    The target-free counterpart of ``global_ols``: OLS fixes the target's
    coefficient at 1 and charges the whole residual to the target column;
    TLS treats every column alike and returns the direction the scaled
    design is closest to annihilating. Returns ``(lam, theta, w0)``:

    lam    mean((Z v)^2) with Z the scaled design and v the unit right-
           singular vector of its smallest singular value, i.e. the
           smallest eigenvalue of ``Z^T Z / N``. Gradient-bearing through
           Z ONLY. v is solved detached: it minimises the Rayleigh quotient
           over the unit sphere, so the path through v contributes exactly
           zero (envelope theorem -- the same argument that lets
           ``global_ols``'s theta stay attached as a mere convention), and
           detaching drops a term that is analytically zero but
           numerically ill-conditioned when two eigenvalues approach.
           Computed as a residual, not as an eigenvalue, so its precision
           is the residual's (no condition-number squaring).
           Each scaled column has mean square 1 on the data, so the trace
           of ``Z^T Z / N`` is K+1 there and lam <= 1: dimensionless,
           weight 1, like every other term.
    theta  target-gauge coefficients (detached, float64): with
           w = v / col_scale, ``y ~= A @ theta`` for ``theta = -w[1:] /
           w[0]``. Defined only while the target takes part in the
           relation.
    w0     |v[0]|, the target's share of the null direction. Near 0 the
           relation the design is closest to satisfying does not involve
           the target at all (a library identity), and theta blows up --
           the TLS failure mode OLS cannot have, logged as a diagnostic.

    target (N,), features (N, K), col_scale (K+1,) from ``tls_scales``.
    """
    Z = torch.cat([target.unsqueeze(1), features], dim=1) / col_scale
    with torch.no_grad():
        Zd = Z.double()
        # (K+1)x(K+1) Gram in float64, eigh on the CPU: K <= 4 here, so the
        # small solve is cheaper than a device SVD of the (N, K+1) design.
        evals, evecs = torch.linalg.eigh((Zd.t() @ Zd).cpu())
        v = evecs[:, 0].to(Z.device)
        w = v / col_scale.double()
        theta = -w[1:] / w[0]
    r = Z @ v.to(Z.dtype)
    return (r ** 2).mean(), theta, float(v[0].abs())


class HardICWrapper(torch.nn.Module):
    """Impose the initial condition EXACTLY, so no IC loss term exists.

    ``theta(t) = theta0 + omega0*(t - t0) + m(t) * inner(t)`` with
    ``m(t) = ((t - t0)/span)^2``. Both ``m(t0) = 0`` and ``m'(t0) = 0``,
    so the wrapped field reproduces ``theta(t0) = theta0`` and
    ``omega(t0) = omega0`` to machine precision for ANY inner network --
    a constraint that cannot be traded off, and therefore carries no
    weight to tune.

    ``span`` normalizes the multiplier to [0, 1] over the train slice.
    The earlier hard-IC attempt used a bare ``(t - t0)^2``, which
    multiplies the inner net by span^2 (~64 here) and wrecks the output
    scale; ``span`` is data (the train duration), not a knob.

    ``inner`` stays a submodule, so ``parameters()`` / ``state_dict()``
    behave as the training loop expects.
    """

    def __init__(self, inner, t0, span, th0, om0):
        super().__init__()
        self.inner = inner
        self.register_buffer("t0", torch.as_tensor(t0).reshape(1, 1))
        self.register_buffer("span", torch.as_tensor(span).reshape(1, 1))
        self.register_buffer("th0", torch.as_tensor(th0).reshape(1, -1))
        self.register_buffer("om0", torch.as_tensor(om0).reshape(1, -1))

    def forward(self, t):
        dt = t - self.t0
        m = (dt / self.span) ** 2
        return self.th0 + self.om0 * dt + m * self.inner(t)


def bounded(score):
    """``score / (1 + score)``: [0, inf) -> [0, 1), strictly monotone.

    A loss term needs a weight only when its scale is arbitrary. het
    already lands in [0, 1); raw chi does not (panel values up to ~1e4),
    and that unbounded scale is exactly what forced a ``W_CV`` to exist.
    This map has no constant in it and preserves order, so it changes
    nothing about WHICH term is judged least constant -- the same
    property ``sparsity._keep_rule_scores`` relies on when it calls the
    ``rescale`` factor order-preserving.

    OPT-IN (``STAT_BOUND=True``); the autoscale scripts default to the raw
    statistics. Order is preserved but gradient is not: d/dx = 1/(1+x)^2,
    so at chi ~ 259 the pull is ~1.5e-5 of its value near zero -- the map
    silences the statistic exactly on the states it judges worst.
    """
    return score / (1.0 + score)


def grad_norms(terms, params):
    """||d term / d params|| for each loss term SEPARATELY.

    Answers the question the loss print cannot: a term can be tiny in
    VALUE and still own the gradient, so "l_data is the same order as
    l_phys at convergence" does not establish that physics is doing any
    work. Call before backward() -- every grad is taken with
    retain_graph=True so the real backward pass still has its graph.

    Terms that are constant zeros (a disabled flag) report 0.0 rather
    than raising, which is what makes this safe to call unconditionally.
    """
    import torch
    out = {}
    for key, value in terms.items():
        if not torch.is_tensor(value) or not value.requires_grad:
            out[key] = 0.0
            continue
        gs = torch.autograd.grad(value, params, retain_graph=True,
                                 allow_unused=True)
        tot = sum(float((g ** 2).sum()) for g in gs if g is not None)
        out[key] = tot ** 0.5
    return out


# Centred finite-difference weights, keyed [derivative][order of accuracy];
# the stencil half-width is order // 2 in every case.
_CENTRAL_WEIGHTS = {
    1: {2: (-1 / 2, 0.0, 1 / 2),
        4: (1 / 12, -2 / 3, 0.0, 2 / 3, -1 / 12),
        6: (-1 / 60, 3 / 20, -3 / 4, 0.0, 3 / 4, -3 / 20, 1 / 60),
        8: (1 / 280, -4 / 105, 1 / 5, -4 / 5, 0.0, 4 / 5, -1 / 5, 4 / 105,
            -1 / 280)},
    2: {2: (1.0, -2.0, 1.0),
        4: (-1 / 12, 4 / 3, -5 / 2, 4 / 3, -1 / 12),
        6: (1 / 90, -3 / 20, 3 / 2, -49 / 18, 3 / 2, -3 / 20, 1 / 90),
        8: (-1 / 560, 8 / 315, -1 / 5, 8 / 5, -205 / 72, 8 / 5, -1 / 5,
            8 / 315, -1 / 560)},
}


def central_diff(f, h, deriv=1, order=4, axis=-1, periodic=False):
    """Centred finite difference of the given ORDER OF ACCURACY on a
    uniform grid of spacing ``h`` (numpy).

    The FD-derived pieces of the weight-free loss (THETA_FD and the
    yardsticks) are only as good as this derivative. Measured offline on
    the train windows: DP's THETA_FD coefficient error is 1.8e-3 with
    ``np.gradient`` (2nd order) and 3.0e-5 at order 4, 1.6e-6 at order 6.
    Clean simulated data only -- the noise gain grows with the order.

    Non-periodic: the ``order // 2`` points at each end along ``axis`` have
    no centred stencil and are DROPPED, so the result is shorter by
    ``2 * (order // 2)``; slice companion channels with ``fd_core(order)``.
    Periodic: wraps, same length.
    """
    try:
        w = _CENTRAL_WEIGHTS[deriv][order]
    except KeyError:
        raise ValueError(f"no centred stencil for deriv={deriv}, "
                         f"order={order}") from None
    f = np.moveaxis(np.asarray(f, dtype=np.float64), axis, -1)
    k = len(w) // 2
    out = np.zeros_like(f)
    for i, c in enumerate(w):
        if c:
            out += c * np.roll(f, k - i, axis=-1)
    out /= h ** deriv
    if not periodic:
        out = out[..., k:-k]
    return np.moveaxis(out, -1, axis)


def fd_core(order):
    """Slice that aligns a channel with ``central_diff(..., order)``'s
    trimmed output along the differentiated axis."""
    k = order // 2
    return slice(k, -k)


def spectral_diff(f, h, deriv=1, axis=-1):
    """Fourier derivative on a PERIODIC uniform grid (numpy): exact for
    band-limited fields. Allen-Cahn's u_xx at sharp fronts is where the
    stencil error concentrates -- spectral takes THETA_FD's rel-L1 error
    from 7.0e-4 (4th-order stencils) to 1.3e-7."""
    f = np.asarray(f, dtype=np.float64)
    n = f.shape[axis]
    k = 2.0 * np.pi * np.fft.fftfreq(n, d=h)
    shape = [1] * f.ndim
    shape[axis] = n
    mult = ((1j * k) ** deriv).reshape(shape)
    return np.real(np.fft.ifft(mult * np.fft.fft(f, axis=axis), axis=axis))


def anchor_scales(target, features):
    """Signal-unit yardsticks for ``anchor_penalty``, computed ONCE from
    the design that built the anchor: ``(mean(A_j^2) per column,
    Var(y))``. Taken from the OBSERVED (finite-difference) design, never
    from the net's field -- the net could shrink its own columns and
    switch the pull off, which is how every net-derived score in this
    project collapsed."""
    # Floor at the dtype's smallest positive value, NOT at EPS: an additive
    # EPS=1e-8 is 1% of Var(y) once the equation is scaled by 1e-3, which
    # breaks the scale-invariance that licenses weight 1. The floor only
    # matters for a constant target, where the penalty is meaningless.
    return ((features ** 2).mean(dim=0),
            max(float(target.var(unbiased=False)),
                float(torch.finfo(target.dtype).tiny)))


def anchor_penalty(c_hat, anchor, norm="signal", col_ms=None, y_var=None):
    """Per-term pull of the net's coefficient recovery toward a
    data-driven anchor, weight-free.

    ``norm='signal'`` (default)::

        per_j = (c_j - a_j)^2 * mean(A_j^2) / Var(y)

    the fraction of the target's variance that this coefficient's miss
    would leave unexplained on the observed design -- the same 1 - R^2
    form ``l_phys`` has, so weight 1 is commensurate with it. Invariant
    under rescaling a column (c_j, a_j pick up 1/s, mean(A_j^2) s^2) and
    under a common rescaling of the equation.

    ``norm='relative'`` is ``((c_j - a_j) / a_j)^2``. Also dimensionless,
    but it equals the signal form divided by the term's own energy share
    ``a_j^2 mean(A_j^2) / Var(y)``, so a WEAK term is pulled as hard as
    its weakness: Allen-Cahn's diffusion has share 7e-5, the relative
    anchor opened at 1.3e8 against O(1) terms, and the net destroyed its
    field to pin D. Kept opt-in so the earlier ``_afd`` runs reproduce.
    Diagonal by design (per term, so ANCHOR_WEIGHT='stat' can scale it);
    it does not credit misses that cancel between collinear columns.
    """
    if norm == "relative":
        return ((c_hat - anchor) / anchor) ** 2
    if norm != "signal":
        raise ValueError(f"unknown anchor norm {norm!r}")
    if col_ms is None or y_var is None:
        raise ValueError("norm='signal' needs col_ms and y_var "
                         "(anchor_scales on the anchor's own design)")
    return (c_hat - anchor) ** 2 * col_ms.to(c_hat.dtype) / y_var


def combine_loss(terms, form="sum"):
    """THE total loss from its terms, with no coefficients in either form.

    ``form='sum'``: ``t1 + t2 + ...`` in the given order -- the original
    loss line, bit-for-bit (adding onto 0.0 is exact).

    ``form='log'``: ``sum log(t_i)`` over the ACTIVE terms. Gradient is
    ``sum grad(t_i) / t_i``: each term's RELATIVE improvement counts
    equally, so a term that has already shrunk keeps its pull instead of
    being drowned by a larger one. Measured imbalance under 'sum' with
    fixed FD: physics owned ~90% of the gradient on Allen-Cahn, the data
    term 80-97% on DP. A constant factor on any term becomes an additive
    constant, so the variance yardsticks drop out of the gradient
    entirely. It is the Gaussian profile likelihood with one unknown noise
    level per term, at equal counts (the count-weighted N/2 log form would
    make the balance depend on N_COLL, a harness choice).

    MEASURED TO COLLAPSE (Sep 2026, fixed FD 4th order, 3 seeds each): the
    net goes to an EQUILIBRIUM that satisfies the fixed equation exactly --
    Allen-Cahn u == 0, the double pendulum hanging at rest -- so log(l_phys)
    runs toward -inf (best loss -26..-29) while log(l_data) pays only a
    finite ~+11. AC: train rel-L2 1.0000 on all seeds, test RMSE 0.91 vs
    3.6e-4 under 'sum'; DP: coef err 1.87-1.92 vs 5.4e-5, max_corr 1e-12..
    1e-16. Nearly every physical equation has such equilibria, and ANY
    relative-progress balancing (log, 1/term weights, uncertainty weights)
    has this gradient, so it is not an Allen-Cahn quirk. Kept opt-in only
    so the result reproduces; do not use.

    Terms that carry no gradient (disabled modes return a constant zero)
    are skipped in 'log' -- log(0) would be -inf and they cannot move the
    optimum anyway; the reported value excludes them.
    """
    if form == "sum":
        total = 0.0
        for t in terms:
            total = total + t
        return total
    if form == "log":
        active = [t for t in terms if torch.is_tensor(t) and t.requires_grad]
        if not active:
            raise ValueError("form='log' needs at least one gradient-bearing term")
        total = 0.0
        for t in active:
            total = total + torch.log(t)
        return total
    raise ValueError(f"unknown loss form {form!r}")


def observation_loss(pred, obs, norm):
    """Pointwise supervision on the observed field, weight-free.

    ``mean((pred - obs)**2 / norm)``, with ``norm`` the OBSERVED
    field's own variance -- the same yardstick l_ic and l_bc already
    divide by. That division is the whole licence for weight 1: the
    ratio is dimensionless and invariant under a common rescaling of
    the field, so the term is commensurate with l_phys (itself
    mean(r^2)/Var(y)) without a constant in front of either.

    ``norm`` may be a scalar (duffing, wave: one field) or a
    per-channel vector (dp: theta1, theta2 explore different
    ranges); dividing inside the mean covers both, and for a scalar
    it is identical to dividing the mean.

    Lives here, not in the three experiment scripts, so dp / duffing /
    wave cannot drift into three different observation terms.
    """
    return ((pred - obs) ** 2 / norm).mean()


def max_corr(target, features, weights=None):
    """``max |A^T W y|`` -- the scale anchor of
    ``epde.operators.common.sparsity.PhysicsInformedLasso.fit``
    (``active_thresholds = active_cv * max_corr``).

    There the dimensionless per-term score is multiplied by this to reach
    coefficient units. The PINN loss anchors each term by its own
    yardstick instead, so this is carried as a DIAGNOSTIC: saving it per
    run keeps the score x anchor product inspectable against the
    production rule.
    """
    w = weights if weights is not None else torch.ones_like(target)
    return (features.t() @ (w * target)).abs().max()


def chi2_per_term(target, features, weights=None, ridge=EPS,
                  detach_calibration=True, use_floor=True, denom="signal"):
    """Raw Nyblom-Hansen cumulative-score-path constancy per term (torch,
    differentiable) -- the WINDOWLESS third member of the truth-free
    statistic family, mirroring
    ``epde.operators.common.survival.chi2_scores(..., rescale=False)``
    on a 1-D grid with ``fit_intercept=False``::

        c      = (A^T W A + lam I)^{-1} A^T W y  (ONE global weighted OLS)
        r      = y - A @ c
        s[i,j] = w_i A_ij r_i        (col sums ~ 0: path pinned both ends)
        S_j(t) = cumsum_{i<=t} s[i,j]            (one path, in time order)
        D_j    = sum_i (w_i A_ij (c_j A_ij))^2   (own fitted-signal energy)
        L_j    = (1/N) sum_t S_j(t)^2 / D_j      (Cramer-von Mises)

    PRECONDITION: rows of target/features are in grid (time) order --
    the OLS is order-invariant but the cumsum path is not.

    TRUTH-FREE: theta is the global OLS recovery of the net's own
    trajectory (the evidence rule -- never a trainable parameter). As a
    LOSS it must be paired with a collapse blocker (physics + IC): an
    exact fit floors to ZERO score, so chi -- like het -- rewards
    degeneracy, it cannot forbid it; the mcv diagnostic stays the
    watchdog.

    With ``detach_calibration=True`` (default) the calibration
    denominator ``D_j`` (including its dependence on ``c``) is DETACHED:
    the score's gradient flows through the path ``S`` -- through ``r``,
    the columns ``A``, and ``c`` via ``linalg.solve`` -- but not through
    ``D``. Otherwise the loss acquires a second descent direction --
    inflate your own signal energy (grow ``|c_j|`` or your own
    derivative amplitudes) and the score falls without the path getting
    any more constant -- the exact analog of het's
    inflate-your-own-standard-errors hole. ``c`` in the NUMERATOR stays
    attached: it is evidence, not calibration.

    The solve is ``global_ols`` -- het's guard convention (AtA-relative
    ridge PLUS an absolute ``tiny``), not ``ols_per_window``'s
    relative-only ridge: a parked net (AtA -> 0) must yield a guarded
    solve, not a LinAlgError. The ridge breaks the exact zero-sum pinning by
    ``S_N = lam*c`` -- negligible at these scales; ``s_end`` is
    returned so runs can verify it.

    ``use_floor=False`` skips the exact-fit floor: the score stays the
    raw path functional even at machine-precision fits (tiny, but never
    an exact zero cliff). The float32 DP run measured the floor as an
    ATTRACTOR, not a benign self-limit -- the net drove both equations
    to near-exact fits of a WRONG (gravity-free) system where the
    floored chi read exactly [0,0,0] -- so the no-floor variant (paired
    with float64 inputs, which alone would push the floor down to
    ~2e-13) removes that zero-loss escape hatch.

    Inputs: target (N,), features (N, K), optional weights (N,).
    Returns dict: ``score`` (K,) live loss term, ``theta`` (K,) live
    global-OLS recovery, and detached diagnostics ``D`` (K,), ``rss``,
    ``yy`` (exact-fit floor margin), ``s_end`` (K,) = |S_N| zero-sum
    violation, ``path`` (N, K) cumulative score paths.
    """
    A = features
    y = target
    N, K = A.shape
    tiny = torch.finfo(A.dtype).eps
    w = weights if weights is not None else torch.ones_like(y)

    # THE shared solve (see ``global_ols``): the statistic and any loss
    # term reading the net's own recovery must use one and the same theta.
    c, r = global_ols(y, A, weights=w, ridge=ridge)
    wA = w.unsqueeze(1) * A
    rss = (w * r * r).sum()
    yy = (w * y * y).sum()

    # Score path: GRADIENT-BEARING (through r, A, and c via the solve).
    s = wA * r.unsqueeze(1)                                   # (N, K)
    S = torch.cumsum(s, dim=0)                                # (N, K)

    D = ((wA * (A * c.unsqueeze(0))) ** 2).sum(dim=0)         # (K,)
    if denom == "none":
        # Mechanism probe: the raw path energy, L_j = (1/N) sum_t S_j^2,
        # with NO normalization. D_j is proportional to c_j^2, so as a
        # LOSS the standard form can be reduced by shrinking a
        # coefficient -- suspected cause of the gravity deletion the
        # PINN cells kept showing (het, whose denominator tau^2 +
        # theta_bar^2 is bounded, does not delete it). Dropping D_j
        # removes that route; the score is then NOT scale-free across
        # terms, which is the price of the test.
        L = torch.nan_to_num((S ** 2).sum(dim=0) / N)
        score = (torch.where(rss > tiny * N * yy, L, torch.zeros_like(L))
                 if use_floor else L)
        return dict(score=score, theta=c, D=D.detach(),
                    rss=rss.detach(), yy=yy.detach(),
                    s_end=S[-1].detach().abs(), path=S.detach())
    D_cal = D.detach() if detach_calibration else D
    # Double-where so the untaken 0/0 branch cannot backprop NaN (the
    # het valid-mask idiom); a dead column is 0/0 -> 0 (no evidence).
    ok = D_cal > 0
    L = torch.where(
        ok,
        (S ** 2).sum(dim=0)
        / (N * torch.where(ok, D_cal, torch.ones_like(D_cal))),
        torch.zeros_like(D_cal))

    # Exact-fit floor (survival.chi2_scores convention): below it the
    # path is reading float round-off, not physics.
    if use_floor:
        score = torch.where(rss > tiny * N * yy, L, torch.zeros_like(L))
    else:
        score = L
    score = torch.nan_to_num(score)

    return dict(score=score, theta=c, D=D.detach(),
                rss=rss.detach(), yy=yy.detach(),
                s_end=S[-1].detach().abs(), path=S.detach())


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


# ============================================================ learning-process GIF
def _learning_capture_iters(total, n_frames=90):
    """Front-loaded capture schedule: dense early (the field moves fastest
    then), regular cadence later, always including 0 and `total`."""
    early = [0, 25, 50, 100, 200, 400, 700, 1000, 1500, 2500, 4000, 6000]
    step = max(1, total // max(1, n_frames))
    reg = list(range(0, total + 1, step))
    return sorted({e for e in early + reg if 0 <= e <= total} | {0, total})


class LearningGifRecorder:
    """Snapshot a model's field prediction over the domain during training
    and render the learning process as a GIF.

    Used IN-PLACE by the dp pinn_test_* scripts (guarded by their MAKE_GIF
    flag): each real training run emits its own learning animation, so there
    is no separate driver script. `predict_fn()` must return (theta1, theta2)
    as arrays over the full time grid.

    matplotlib is imported here -- only when a recorder is actually built,
    i.e. MAKE_GIF is on -- so importing cv_metric stays matplotlib-free. If
    the import fails the recorder disables itself and warns rather than
    crashing a long training run at the very end.
    """

    def __init__(self, t_grid, th1_data, th2_data, t_split, label, out_path,
                 total_iters, fps=12, hold_frames=18, n_frames=90,
                 color="magenta"):
        self.t = np.asarray(t_grid).reshape(-1)
        self.th1_data = np.asarray(th1_data).reshape(-1)
        self.th2_data = np.asarray(th2_data).reshape(-1)
        self.t_split = float(t_split)
        self.label = label
        self.out_path = out_path
        self.fps = fps
        self.hold_frames = hold_frames
        self.color = color
        self._cap = set(_learning_capture_iters(total_iters, n_frames))
        self._done = set()
        self.frames = []
        self._n_train = int(np.searchsorted(self.t, self.t_split, side="right"))
        self.enabled = True
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib.animation import FuncAnimation, PillowWriter
            self._plt = plt
            self._FuncAnimation = FuncAnimation
            self._PillowWriter = PillowWriter
        except Exception as exc:   # matplotlib missing -> no-op, warn once
            self.enabled = False
            print(f"[gif] disabled ({exc!r}); training continues without a GIF")

    def maybe_capture(self, it, predict_fn):
        """Store a frame if `it` is on the schedule (idempotent per iter)."""
        if not self.enabled or it not in self._cap or it in self._done:
            return
        self._done.add(it)
        th1p, th2p = predict_fn()
        th1p = np.asarray(th1p).reshape(-1)
        th2p = np.asarray(th2p).reshape(-1)

        def _rl(p, d, sl):
            return float(np.linalg.norm(p[sl] - d[sl])
                         / (np.linalg.norm(d[sl]) + EPS))

        sl_tr = slice(0, self._n_train)
        sl_te = slice(self._n_train, None)
        self.frames.append((
            int(it), th1p.copy(), th2p.copy(),
            _rl(th1p, self.th1_data, sl_tr), _rl(th1p, self.th1_data, sl_te),
            _rl(th2p, self.th2_data, sl_tr), _rl(th2p, self.th2_data, sl_te),
        ))

    def render(self):
        """Assemble captured frames into the GIF at `out_path`."""
        if not self.enabled or not self.frames:
            return
        plt = self._plt

        def ylim_for(data, preds):
            allv = np.concatenate([data] + preds)
            lo, hi = np.percentile(allv, [1, 99])
            pad = 0.15 * (hi - lo + EPS)
            return lo - pad, hi + pad

        y1 = ylim_for(self.th1_data, [f[1] for f in self.frames])
        y2 = ylim_for(self.th2_data, [f[2] for f in self.frames])

        fig = plt.figure(figsize=(10, 8.6))
        gs = fig.add_gridspec(3, 1, height_ratios=(2.0, 2.0, 1.3), hspace=0.3)
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1], sharex=ax1)
        axm = fig.add_subplot(gs[2])
        for ax, data, ylab, ylim in (
            (ax1, self.th1_data, r"$\theta_1(t)$", y1),
            (ax2, self.th2_data, r"$\theta_2(t)$", y2),
        ):
            ax.plot(self.t, data, "k-", lw=1.6, label="data (truth)", zorder=1)
            ax.axvline(self.t_split, color="gray", ls="--", alpha=0.6,
                       label=f"train/test split (t={self.t_split:.2f})")
            ax.set_ylim(*ylim)
            ax.set_ylabel(ylab)
            ax.grid(alpha=0.3)
        ax2.set_xlabel("t")

        (line1,) = ax1.plot([], [], color=self.color, lw=1.4, label="model", zorder=2)
        (line2,) = ax2.plot([], [], color=self.color, lw=1.4, label="model", zorder=2)
        ax1.legend(fontsize=8, loc="upper right")
        ax2.legend(fontsize=8, loc="upper right")

        # Metric panel: rel-L2 train/test curves growing with the animation.
        its = np.array([f[0] for f in self.frames], dtype=float)
        mets = np.array([f[3:7] for f in self.frames])       # (F, 4)
        mspecs = [("th1 train", "C0", "-", 0), ("th1 test", "C0", "--", 1),
                  ("th2 train", "C2", "-", 2), ("th2 test", "C2", "--", 3)]
        mlines = []
        for name, c, ls, _j in mspecs:
            (ln,) = axm.plot([], [], ls, color=c, lw=1.2, label=name)
            mlines.append(ln)
        cursor = axm.axvline(0, color="gray", lw=0.8, alpha=0.6)
        axm.axhline(1.0, color="gray", ls=":", lw=0.8, alpha=0.7)
        axm.set_xscale("symlog", linthresh=100)
        axm.set_yscale("log")
        axm.set_xlim(0, max(its.max(), 1) * 1.05)
        axm.set_ylim(max(mets.min() * 0.5, 1e-6), mets.max() * 2)
        axm.set_xlabel("Adam iter")
        axm.set_ylabel("rel-L2 vs data")
        axm.legend(fontsize=7, ncol=4, loc="lower left")
        axm.grid(alpha=0.3, which="both")
        suptitle = fig.suptitle("", fontsize=10)

        order = list(range(len(self.frames))) + \
            [len(self.frames) - 1] * self.hold_frames

        def init():
            line1.set_data([], [])
            line2.set_data([], [])
            for ln in mlines:
                ln.set_data([], [])
            suptitle.set_text("")
            return (line1, line2, *mlines, cursor, suptitle)

        def update(k):
            it, th1p, th2p, rl1_tr, rl1_te, rl2_tr, rl2_te = self.frames[k]
            line1.set_data(self.t, th1p)
            line2.set_data(self.t, th2p)
            for ln, (_nm, _c, _ls, j) in zip(mlines, mspecs):
                ln.set_data(its[:k + 1], mets[:k + 1, j])
            cursor.set_xdata([it, it])
            suptitle.set_text(
                f"DP learning ({self.label}) -- Adam iter {it:>6d}   "
                f"rel-L2 train: th1={rl1_tr:.2e}, th2={rl2_tr:.2e}   "
                f"test: th1={rl1_te:.2e}, th2={rl2_te:.2e}")
            return (line1, line2, *mlines, cursor, suptitle)

        anim = self._FuncAnimation(fig, update, frames=order, init_func=init,
                                   blit=False, interval=1000 / self.fps)
        anim.save(str(self.out_path), writer=self._PillowWriter(fps=self.fps))
        plt.close(fig)
        print(f"[gif] saved -> {self.out_path}  ({len(self.frames)} frames)")


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

    # chi2_per_term convention check: a term whose coefficient DRIFTS
    # over the sample must outscore the same term with a constant one.
    t = torch.linspace(0.0, 1.0, 600)
    f1 = torch.sin(20 * t)
    f2 = torch.cos(13 * t)
    A_chi = torch.stack([f1, f2], dim=1)
    bump = 0.1 * torch.cos(57 * t)           # keeps rss off the exact-fit floor
    out_const = chi2_per_term(2.0 * f1 - 3.0 * f2 + bump, A_chi)
    out_drift = chi2_per_term(2.0 * f1 - 3.0 * (1.0 + t) * f2 + bump, A_chi)
    sc_const, sc_drift = out_const["score"], out_drift["score"]
    tiny32 = torch.finfo(torch.float32).eps
    assert out_const["rss"] > tiny32 * 600 * out_const["yy"], "on the floor"
    assert sc_drift[1] > sc_const[1] > 0.0, (sc_drift, sc_const)
    print(f"[chi2] drifting-coef term scores {sc_drift[1]:.3e} > "
          f"constant-coef {sc_const[1]:.3e}")
