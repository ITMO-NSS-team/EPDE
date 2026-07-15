"""Candidate Hadamard-stability complementary objectives (clean-data regime).

Each objective is a functional  J(f, t, w, gshape, var) -> float  (lower = better
for the TRUE equation / a valid identity), computed solver-free from the seeded
candidate's design matrix. They are the survivors of the Jun-2026 research panel
(see projects/thesis/objective_experiment_results.md and the memory
``project_hadamard_objective_research``):

  * bic_objective       -- parameter-count description length (Occam). The only
                           axis ORTHOGONAL to every Gram/coefficient statistic;
                           breaks discrepancy's overfit tie via +k*ln(Neff).
  * biw_objective       -- Block-Invariance Wald ratio: between-block coefficient
                           scatter normalized by the per-block Cramer-Rao bound
                           (chi^2_{L-1} per term). Cramer-Rao normalization is the
                           C1 mechanism (collinearity inflates scatter AND
                           variance -> cancels for a collinear-but-exact identity).
  * gengap_block_objective -- contiguous TIME-split extrapolation residual + a
                           precision-normalized coefficient drift (the L=2-time
                           special case of BIW; a cheap control).

All three only catch APPROXIMATE / overfit wrong forms; on a single clean
trajectory an EXACT accidental shadow is information-theoretically inseparable
from a valid identity (the documented honest limit) -- bic is the only one that
breaks the exact-overfit tie, and it does so by parsimony, not stability.
"""
from __future__ import annotations

import os
import sys

import numpy as np

_THIS = os.path.dirname(os.path.abspath(__file__))
_THESIS = os.path.abspath(os.path.join(_THIS, '..'))
_ROOT = os.path.abspath(os.path.join(_THIS, '..', '..', '..'))
for _p in (_ROOT, _THESIS, _THIS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from inverse_stability import wls_metrics  # noqa: E402


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _weighted(f, t, w):
    f = np.asarray(f, dtype=float)
    if f.ndim == 1:
        f = f[:, None]
    t = np.asarray(t, dtype=float).reshape(-1)
    sw = np.sqrt(np.clip(np.asarray(w, dtype=float).reshape(-1), 0, None))
    return f * sw[:, None], t * sw, sw


def _block_fit(A, b, p):
    """WLS fit + diagonal coefficient variance on one (already-weighted) block."""
    th, *_ = np.linalg.lstsq(A, b, rcond=None)
    resid = A @ th - b
    dof = max(A.shape[0] - p, 1)
    s2 = float(resid @ resid) / dof
    Ginv = np.linalg.pinv(A.T @ A)
    var = s2 * np.clip(np.diag(Ginv), 0.0, None)
    return th, np.clip(var, 1e-30, None)


def _axis_blocks(gshape, axis, n_blocks):
    """Contiguous index blocks along ``axis`` of the flattened grid."""
    idx = np.arange(int(np.prod(gshape))).reshape(gshape)
    out = []
    for s in np.array_split(np.arange(gshape[axis]), n_blocks):
        if s.size == 0:
            continue
        sl = [slice(None)] * len(gshape)
        sl[axis] = s
        out.append(idx[tuple(sl)].ravel())
    return out


# --------------------------------------------------------------------------
# 1. BIC  (parameter-count description length; lower = better)
# --------------------------------------------------------------------------
def bic_objective(f, t, w, gshape=None, var=None, floor=1e-12, per_term=False):
    """BIC = Neff*ln(nRSS + floor) + k*ln(Neff).

    nRSS = ||A theta - b||^2 / ||b||^2 is the squared discrepancy (so the data
    term is shared scale across candidates of a system); k = #terms; Neff = the
    number of active samples. The data term ties any two exact fits at
    Neff*ln(floor); the k*ln(Neff) term then ranks the parsimonious one below the
    overfit -- breaking the machine-precision tie that makes plain discrepancy
    0/14. ``floor`` sets how exact a fit must be before parsimony decides.

    per_term=True returns the per-term BIC *increase* contribution (for the
    regularizer): term j's marginal description-length cost
    Neff*ln(nRSS_with / nRSS_without_j) + ln(Neff); positive => term not worth
    keeping.
    """
    A, b, sw = _weighted(f, t, w)
    bb = float(b @ b) + 1e-30
    k = A.shape[1]
    Neff = float(np.count_nonzero(sw > 0))
    Neff = max(Neff, 2.0)
    th, *_ = np.linalg.lstsq(A, b, rcond=None)
    rss = float(np.sum((A @ th - b) ** 2))
    nrss = rss / bb
    if not per_term:
        return float(Neff * np.log(nrss + floor) + k * np.log(Neff))
    # per-term marginal cost: refit without each column.
    out = np.zeros(k)
    for j in range(k):
        keep = [i for i in range(k) if i != j]
        if not keep:
            out[j] = np.inf
            continue
        thj, *_ = np.linalg.lstsq(A[:, keep], b, rcond=None)
        rss_wo = float(np.sum((A[:, keep] @ thj - b) ** 2))
        nrss_wo = rss_wo / bb
        out[j] = Neff * np.log((nrss + floor) / (nrss_wo + floor)) + np.log(Neff)
    return out


# --------------------------------------------------------------------------
# 2. BIW  (Block-Invariance Wald ratio; lower = better)
# --------------------------------------------------------------------------
def biw_objective(f, t, w, gshape, var=None, n_blocks=3, min_per_factor=3,
                  per_term=False):
    """Between-block coefficient heterogeneity, Cramer-Rao normalized.

    Partition the grid into contiguous blocks along EACH axis; per axis fit WLS
    coefficients per block with their diagonal covariance, and form the per-term
    Wald statistic  Q_k = sum_b (theta_{b,k} - theta_bar_k)^2 / Var_{b,k}  with
    theta_bar the inverse-variance-weighted mean -- distributed chi^2_{L-1} under
    the null "coefficient k is constant across blocks" (a true law / valid
    identity). Q_k is normalized by (L-1) and the WORST axis is taken (the most
    heterogeneous direction, so space-modulation and time-nonstationarity are
    both caught). The objective sums the per-term Q over terms.

    Cramer-Rao normalization: a collinear-but-EXACT identity has large per-block
    variance AND large raw scatter that cancel -> Q ~ 1 (not penalized); an
    accidental fit whose coefficients genuinely drift has scatter exceeding its
    variance -> Q >> 1.
    """
    A, b, sw = _weighted(f, t, w)
    p = A.shape[1]
    n = A.shape[0]
    gshape = tuple(int(x) for x in gshape)
    min_per = max(min_per_factor * p, p + 2)
    perterm = np.zeros(p)
    used_any = False
    for axis in range(len(gshape)):
        L = max(2, min(n_blocks, n // min_per))
        if L < 2 or gshape[axis] < 2:
            continue
        blocks = _axis_blocks(gshape, axis, L)
        thetas, varis = [], []
        ok = True
        for bl in blocks:
            if bl.size < min_per:
                ok = False
                break
            th, vv = _block_fit(A[bl], b[bl], p)
            thetas.append(th)
            varis.append(vv)
        if not ok or len(thetas) < 2:
            continue
        TH = np.asarray(thetas)            # (L, p)
        VA = np.asarray(varis)             # (L, p)
        prec = 1.0 / VA
        thbar = np.sum(prec * TH, axis=0) / np.sum(prec, axis=0)
        Qk = np.sum((TH - thbar) ** 2 / VA, axis=0) / max(len(thetas) - 1, 1)
        perterm = np.maximum(perterm, Qk)
        used_any = True
    if not used_any:
        return (np.full(p, np.nan) if per_term else float('nan'))
    if per_term:
        return perterm
    return float(np.sum(perterm))


# --------------------------------------------------------------------------
# 3. gengap_block  (contiguous-time extrapolation gap + coef drift; lower=better)
# --------------------------------------------------------------------------
def gengap_block_objective(f, t, w, gshape, var=None, drift_weight=1.0):
    """Fit on the early TIME half, evaluate on the disjoint later half.

    A well-posed law learned on one time window predicts the next; an overfit /
    non-stationary accidental fit does not. Returns the extrapolation residual
    ratio plus a precision-normalized coefficient drift between the two halves
    (axis 0 = time, EPDE convention).
    """
    A, b, sw = _weighted(f, t, w)
    p = A.shape[1]
    gshape = tuple(int(x) for x in gshape)
    idx = np.arange(A.shape[0]).reshape(gshape)
    h = gshape[0] // 2
    if h < max(p + 1, 2):
        return float('nan')
    tr = idx[:h].ravel()
    te = idx[h:].ravel()
    th_tr, _ = _block_fit(A[tr], b[tr], p)
    gap = float(np.linalg.norm(A[te] @ th_tr - b[te])
                / (np.linalg.norm(b[te]) + 1e-30))
    th_te, var_te = _block_fit(A[te], b[te], p)
    _, var_tr = _block_fit(A[tr], b[tr], p)
    se = np.sqrt(var_tr + var_te)
    drift = float(np.mean(np.abs(th_tr - th_te) / np.maximum(se, 1e-12)))
    return gap + drift_weight * drift


# --------------------------------------------------------------------------
# 4. BIW-CV  (robust block-invariance: coefficient-of-variation across blocks)
# --------------------------------------------------------------------------
def biw_cv_objective(f, t, w, gshape, var=None, n_blocks=3, min_per_factor=3,
                     per_term=False):
    """Robust block-invariance: per term, max over axes of the across-block
    coefficient std / |mean| (a scale-free coefficient-of-variation), NOT the
    CRLB-normalized Wald ratio.

    The CRLB Wald ratio (biw_objective) divides by the per-block sampling
    variance, which -> 0 for a near-exact fit, so deterministic FD-derivative
    drift explodes it on the TRUE PDE. The coefficient-of-variation normalizes by
    the coefficient MAGNITUDE instead (the same scale-free normalization vcoef's
    region-variation uses, via Var/<c>^2), so an exact true law with a slightly
    FD-drifting but O(1) coefficient stays small, while a genuinely
    coordinate-modulated / accidental coefficient (drifts O(1) relative to its
    own mean) is flagged. This is the explicit-block analog of vcoef NC.
    """
    A, b, sw = _weighted(f, t, w)
    p = A.shape[1]
    n = A.shape[0]
    gshape = tuple(int(x) for x in gshape)
    min_per = max(min_per_factor * p, p + 2)
    perterm = np.zeros(p)
    used = False
    scale = None
    for axis in range(len(gshape)):
        L = max(2, min(n_blocks, n // min_per))
        if L < 2 or gshape[axis] < 2:
            continue
        blocks = _axis_blocks(gshape, axis, L)
        thetas = []
        ok = True
        for bl in blocks:
            if bl.size < min_per:
                ok = False
                break
            th, _ = _block_fit(A[bl], b[bl], p)
            thetas.append(th)
        if not ok or len(thetas) < 2:
            continue
        TH = np.asarray(thetas)            # (L, p)
        mean = TH.mean(axis=0)
        std = TH.std(axis=0, ddof=1)
        if scale is None:
            scale = float(np.sqrt(np.mean(mean ** 2))) + 1e-30
        denom = np.abs(mean) + 1e-3 * scale
        cv = std / denom
        perterm = np.maximum(perterm, cv)
        used = True
    if not used:
        return (np.full(p, np.nan) if per_term else float('nan'))
    return perterm if per_term else float(np.sum(perterm))


# --------------------------------------------------------------------------
# 5. gengap_gap  (extrapolation residual only, no exploding drift term)
# --------------------------------------------------------------------------
def gengap_gap_objective(f, t, w, gshape, var=None):
    """Contiguous-time extrapolation residual ratio ONLY (drop the
    precision-normalized drift, which explodes like biw on exact fits)."""
    A, b, sw = _weighted(f, t, w)
    p = A.shape[1]
    gshape = tuple(int(x) for x in gshape)
    idx = np.arange(A.shape[0]).reshape(gshape)
    h = gshape[0] // 2
    if h < max(p + 1, 2):
        return float('nan')
    tr = idx[:h].ravel()
    te = idx[h:].ravel()
    th_tr, _ = _block_fit(A[tr], b[tr], p)
    return float(np.linalg.norm(A[te] @ th_tr - b[te])
                 / (np.linalg.norm(b[te]) + 1e-30))


OBJECTIVES = {
    'bic': bic_objective,
    'biw': biw_objective,
    'biw_cv': biw_cv_objective,
    'gengap_block': gengap_block_objective,
    'gengap_gap': gengap_gap_objective,
}
