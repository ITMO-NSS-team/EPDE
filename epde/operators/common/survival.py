#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Block-based instability estimators for the ``Instability`` objective.

Four alternatives to the varying-coefficient (``vcoef``) score, selectable
via ``epde.globals.instability_metric`` (see ``set_instability_metric``):

* ``survival_scores`` -- STATISTICAL axis: B moving-block bootstrap
  refits of the fixed structure; a term is unstable when its coefficient
  flips sign or spreads widely across resamples.
* ``tile_scores`` -- basis-free SPATIAL axis (Hadamard): one refit per
  disjoint domain tile; a term is unstable when its coefficient differs
  between tiles. Unlike the additive per-axis cosine basis of ``vcoef``
  this makes no separability or smoothness commitment, so it can flag
  multiplicative x-t coefficient variation.
* ``heterogeneity_scores`` -- CALIBRATED heterogeneity (Cochran's Q /
  I^2): per-block refits like ``tile``, but the between-block dispersion
  is measured against the SAMPLING variance the within-block fits
  themselves predict, not against the coefficient's own magnitude. A
  term is unstable only when its coefficients drift across blocks by
  more than the block fits' standard errors explain.
* ``chi2_scores`` -- per-term coefficient constancy from the CUMULATIVE
  SCORE PATH (Nyblom-Hansen). Theta is fitted once on all the data and
  never re-estimated; the running score ``S_j = cumsum(w_i X_ij r_i)``
  propagates once along EACH grid axis, pinned to zero at both ends by
  the normal equations. A constant coefficient leaves an aimless
  bridge; a drifting one leaves a large one-sided bulge. The bulge is
  measured against the term's own fitted-signal energy, never against
  the residual (three residual-scale yardsticks failed; see the
  docstring). No refits at all, so no minimum block size.

The first three run entirely in Gram space on per-block Gram/Gy stacks (the
``PhysicsInformedLasso`` convention: intercept column folded block-wise,
weighted by ``sample_weights``), so the data-sized work is one pass and
every refit is a small (p+1) solve. Aggregation is robust
(median / MAD) or chi-square-calibrated (``het``), and per-term scores
are dimensionless -- invariant to per-column and global rescaling, like
the vcoef and CV scores.

Usage rules the estimators cannot enforce (documented here because the
PINN testbed proved them the hard way): the statistic carries evidence
ONLY over coefficients RE-ESTIMATED from data segments -- computing it
over optimizer-owned coefficients degenerates it into a tying prior the
optimizer zeroes by equalizing blocks; and it certifies EQUATION
identity only, never solution/branch identity (every solution of the
same equation re-estimates the same coefficients).
"""

from __future__ import annotations

import numpy as np

from epde import _loop_stats

# Estimator defaults -- fixed algorithm policy, not per-run tuning knobs
# (no in-tree caller overrides them; ``Instability.compute`` calls both
# estimators with defaults). The signature parameters exist for offline
# experiments (e.g. the instability panel), not for the live search.
_DEFAULT_BOOTSTRAP_DRAWS = 32     # B: block-bootstrap refits per equation
_DEFAULT_N_BLOCKS = 16            # survival: contiguous time-slab count
_DEFAULT_N_TILES = 8              # tile: disjoint domain tiles
_DEFAULT_N_BLOCKS_HET = 16        # het: refit blocks (clamped by sample count)
# chi2 needs no size knob: one score path per grid axis, each accumulated
# at that axis's own level resolution, so there is nothing to tune.


def _slab_edges(n_samples: int, grid_shape, n_blocks: int):
    """Contiguous slab boundaries over the flat sample axis.

    When the flat sample count matches ``prod(grid_shape)`` (the standard
    case: C-ordered flattening of the grid), boundaries are aligned to
    whole steps of the FIRST grid axis -- time -- so each slab is a
    physically contiguous space-time chunk; otherwise plain contiguous
    index slabs are used. ``n_blocks`` is clamped to what the axis can
    support. Returns the ``(n_blocks + 1,)`` edge array.
    """
    grid_shape = tuple(int(s) for s in (() if grid_shape is None else grid_shape))
    if grid_shape and int(np.prod(grid_shape)) == n_samples and grid_shape[0] > 1:
        n_t = grid_shape[0]
        stride = n_samples // n_t
        n_blocks = max(2, min(int(n_blocks), n_t))
        return np.linspace(0, n_t, n_blocks + 1, dtype=int) * stride
    n_blocks = max(2, min(int(n_blocks), n_samples // 2))
    return np.linspace(0, n_samples, n_blocks + 1, dtype=int)


def block_gram_partition(features, target, sample_weights, grid_shape,
                         n_blocks: int, return_yy: bool = False):
    """Per-block augmented Gram stacks over contiguous slabs of axis 0.

    Splits the sample axis into ``n_blocks`` contiguous slabs. When the
    flat sample count matches ``prod(grid_shape)`` (the standard case:
    C-ordered flattening of the grid), slab boundaries are aligned to
    whole steps of the first grid axis (time slabs) so each block is a
    physically contiguous space-time chunk; otherwise plain contiguous
    index slabs are used.

    Returns ``(G_blocks, Gy_blocks)`` with shapes ``(B, p+1, p+1)`` and
    ``(B, p+1)``; the last row/column is the folded intercept (the
    ``sparsity.py`` Gram convention), so callers slice ``[:p, :p]`` for a
    no-intercept solve. With ``return_yy=True`` a third array ``(B,)`` of
    per-block weighted target energies ``y^T W y`` is appended (needed by
    the residual-variance computation in ``heterogeneity_scores``).
    """
    X = np.asarray(features, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    y = np.asarray(target, dtype=float).reshape(-1)
    n_samples, p = X.shape
    sw = (np.asarray(sample_weights, dtype=float).reshape(-1)
          if sample_weights is not None else np.ones(n_samples))

    # grid_shape may arrive as a tuple, list, or ndarray (ctx.data_shape in
    # the live fitness path) -- avoid ambiguous ndarray truth tests.
    edges = _slab_edges(n_samples, grid_shape, n_blocks)
    n_blocks = edges.size - 1

    G_blocks = np.zeros((n_blocks, p + 1, p + 1))
    Gy_blocks = np.zeros((n_blocks, p + 1))
    yy_blocks = np.zeros(n_blocks)
    for b in range(n_blocks):
        lo, hi = edges[b], edges[b + 1]
        Xb, yb, swb = X[lo:hi], y[lo:hi], sw[lo:hi]
        wXb = swb[:, None] * Xb
        G_blocks[b, :p, :p] = Xb.T @ wXb
        ws = wXb.sum(axis=0)
        G_blocks[b, :p, -1] = ws
        G_blocks[b, -1, :p] = ws
        G_blocks[b, -1, -1] = float(swb.sum())
        Gy_blocks[b, :p] = Xb.T @ (swb * yb)
        Gy_blocks[b, -1] = float((swb * yb).sum())
        yy_blocks[b] = float(np.dot(yb, swb * yb))
    if return_yy:
        return G_blocks, Gy_blocks, yy_blocks
    return G_blocks, Gy_blocks


def _solve_gram(G, Gy):
    """Solve ``G c = Gy`` with an lstsq fallback for singular blocks.

    The fallback is recorded via ``_loop_stats`` (not silent): a singular
    per-block Gram means an under-determined refit whose coefficients can
    inflate the instability score, so the run summary should surface how
    often that happened."""
    try:
        return np.linalg.solve(G, Gy)
    except np.linalg.LinAlgError:
        _loop_stats.record('survival.lstsq_fallback', 1, 1)
        return np.linalg.lstsq(G, Gy, rcond=None)[0]


def _robust_dispersion(coefs):
    """Per-column ``MAD / |median|`` of a ``(draws, p)`` coefficient stack.

    Dimensionless and outlier-robust; a column whose median coefficient is
    ~0 maps to a huge (nan_to_num-capped) score, mirroring the vcoef
    degenerate-form guard.
    """
    med = np.median(coefs, axis=0)
    mad = np.median(np.abs(coefs - med[None, :]), axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        rel = mad / np.abs(med)
    return np.nan_to_num(rel), med


def survival_scores(features, target, sample_weights, grid_shape,
                    fit_intercept: bool = True, B: int = _DEFAULT_BOOTSTRAP_DRAWS,
                    n_blocks: int = _DEFAULT_N_BLOCKS, seed: int = 0):
    """Per-term coefficient-survival instability (STATISTICAL axis).

    B moving-block bootstrap draws: multinomial block counts reweight the
    per-block Grams (``G* = sum_b c_b G_b``) and the fixed structure is
    refit per draw -- no re-selection, so the score is independent of the
    sparsity pass. Per-term score::

        score_j = sign_flip_rate_j + MAD_j / |median_j|

    where a flip is a draw whose coefficient sign differs from the median
    sign. Deterministic: the RNG is a local ``default_rng`` seeded from
    ``(seed, p, B)`` -- the global NumPy stream is never touched.
    """
    G_blocks, Gy_blocks = block_gram_partition(
        features, target, sample_weights, grid_shape, n_blocks)
    n_blocks = G_blocks.shape[0]
    p = G_blocks.shape[1] - 1
    sl = slice(None) if fit_intercept else slice(0, p)

    rng = np.random.default_rng(
        np.uint64(0x5EED) + np.uint64(seed) * 7919 + np.uint64(p) * 104729 + np.uint64(B))
    coefs = np.empty((B, p))
    for b in range(B):
        counts = rng.multinomial(n_blocks, np.full(n_blocks, 1.0 / n_blocks))
        G = np.tensordot(counts, G_blocks, axes=1)[sl, sl]
        Gy = counts @ Gy_blocks
        coefs[b] = _solve_gram(G, Gy[sl])[:p]

    rel_spread, med = _robust_dispersion(coefs)
    med_sign = np.sign(med)
    flips = (np.sign(coefs) != med_sign[None, :]).mean(axis=0)
    return np.nan_to_num(flips + rel_spread)


def tile_scores(features, target, sample_weights, grid_shape,
                fit_intercept: bool = True, n_tiles: int = _DEFAULT_N_TILES):
    """Per-term between-tile instability (basis-free SPATIAL axis).

    One refit of the fixed structure per disjoint domain tile; per-term
    score = robust between-tile dispersion ``MAD_j / |median_j|`` of the
    tile coefficients. A true constant-coefficient term agrees across
    tiles -> ~0; a spurious term needs different coefficients in
    different regions -> large.

    ``n_tiles`` is clamped so each tile keeps >= 4 samples per unknown
    (else the per-tile solves are meaninglessly under-determined).
    """
    X = np.asarray(features, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    n_samples, p = X.shape
    max_tiles = max(2, n_samples // (4 * (p + 1)))
    n_tiles = max(2, min(int(n_tiles), max_tiles))

    G_blocks, Gy_blocks = block_gram_partition(
        features, target, sample_weights, grid_shape, n_tiles)
    n_tiles = G_blocks.shape[0]
    sl = slice(None) if fit_intercept else slice(0, p)

    coefs = np.empty((n_tiles, p))
    for b in range(n_tiles):
        coefs[b] = _solve_gram(G_blocks[b][sl, sl], Gy_blocks[b][sl])[:p]

    rel_spread, _ = _robust_dispersion(coefs)
    return rel_spread


def heterogeneity_scores(features, target, sample_weights, grid_shape,
                         fit_intercept: bool = True,
                         n_blocks: int = _DEFAULT_N_BLOCKS_HET):
    """Per-term calibrated heterogeneity: Q-calibrated excess variance.

    One refit of the fixed structure per contiguous block, PLUS the
    per-block coefficient sampling variance from the same Gram solve::

        s2_bj   = sigma2_b * [(X_b^T W X_b)^{-1}]_jj
        Q_j     = sum_b (theta_bj - theta_bar_j)^2 / s2_bj
        tau2_j  = max(0, (Q_j - df) / C_j)          (DerSimonian-Laird)
        score_j = tau2_j / (tau2_j + theta_bar_j^2)  in [0, 1)

    with ``theta_bar`` the inverse-variance-weighted mean, ``sigma2_b``
    the block's residual variance and ``C = S1 - S2/S1`` the DL
    denominator (``S1 = sum w``, ``S2 = sum w^2``, ``w = 1/s2``). This is
    the noise-calibrated analogue of raw cv2: ``Var/mean^2`` with the
    part of ``Var`` the block fits' own standard errors already predict
    SUBTRACTED, and a bounded singularity-free denominator. (Bounded
    ``I^2 = (Q-df)/Q`` was rejected: it measures the SIGNIFICANCE of
    heterogeneity and saturates at 1 on near-clean data where even a tiny
    deterministic drift is "significant" -- the true/wrong separation
    lives in the excess-variance MAGNITUDE relative to the coefficient.)
    Unlike the raw-dispersion scores this measures dispersion in units of
    the dispersion the block fits themselves predict, which closes the
    known leaks in one move:

    * a poorly fit block inflates ``sigma2_b`` and stops masquerading as
      "instability of a true term" (the misfit/significance confound);
    * an unexcited or near-degenerate block inflates ``(X^T X)^{-1}`` and
      contributes ~0 evidence instead of ridge-collapsed fake consistency
      (soft excitation gate -- also why valid-but-degenerate analytical
      identities are NOT flagged: their inflated standard errors shrink
      the statistic rather than growing it);
    * no division by a coefficient mean -- bounded, singularity-free,
      and invariant to per-column and global rescaling.

    A pure-noise column is NOT flagged (its scatter matches its standard
    errors -- insignificance is the t-stat anchor's job, not this one's).
    The block solves use a tiny AtA-relative ridge so a degenerate
    direction yields a huge standard error (zero evidence) instead of a
    LinAlgError. Fails loudly when the data cannot support >= 3 usable
    blocks -- a silently under-calibrated score would corrupt the Pareto
    front invisibly.
    """
    X = np.asarray(features, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    n_samples, p = X.shape
    k = p + 1 if fit_intercept else p
    max_blocks = n_samples // (4 * (p + 1))
    if max_blocks < 3:
        raise ValueError(
            'heterogeneity_scores needs >= 3 blocks of >= 4*(p+1) samples '
            f'to calibrate; got n_samples={n_samples} with p={p}')
    n_blocks = max(3, min(int(n_blocks), max_blocks))

    G_blocks, Gy_blocks, yy_blocks = block_gram_partition(
        features, target, sample_weights, grid_shape, n_blocks,
        return_yy=True)
    n_blocks = G_blocks.shape[0]
    sl = slice(None) if fit_intercept else slice(0, p)
    eye = np.eye(k)
    tiny = np.finfo(float).eps

    theta = np.zeros((n_blocks, p))
    s2 = np.full((n_blocks, p), np.inf)   # inf variance == zero evidence
    for b in range(n_blocks):
        G = G_blocks[b][sl, sl]
        Gy = Gy_blocks[b][sl]
        # AtA-relative ridge (the ols/vcoef convention): a degenerate
        # direction inverts to a huge diagonal -> huge s2 -> ~0 weight,
        # with no exception path.
        scale = max(float(np.trace(G)) / k, tiny)
        Ginv = np.linalg.inv(G + (1e-12 * scale) * eye)
        c = Ginv @ Gy
        theta[b] = c[:p]
        n_eff = float(G_blocks[b][-1, -1])
        dof = max(n_eff - k, 1.0)
        rss = max(float(yy_blocks[b] - 2.0 * (c @ Gy) + c @ (G @ c)), 0.0)
        # Exact-fit floor: on clean data rss can cancel to float zero; a
        # block may not claim infinite precision (units of y^2 kept).
        sigma2 = max(rss / dof, tiny * yy_blocks[b] / dof)
        if not sigma2 > 0.0:
            continue                      # zero-energy block: no evidence
        s2_b = sigma2 * np.diag(Ginv)[:p]
        s2_b[~(s2_b > 0.0)] = np.inf      # non-positive/NaN -> no evidence
        s2[b] = s2_b

    usable = np.isfinite(s2)
    n_use = usable.sum(axis=0)
    if np.any(n_use < 3):
        bad = np.where(n_use < 3)[0].tolist()
        raise ValueError(
            'heterogeneity_scores: fewer than 3 usable (non-degenerate) '
            f'blocks for coefficient(s) {bad} of p={p}; the block refits '
            'are too degenerate to calibrate a heterogeneity score')
    w = np.where(usable, 1.0 / np.where(usable, s2, 1.0), 0.0)
    S1 = w.sum(axis=0)
    S2 = (w ** 2).sum(axis=0)
    theta_bar = (w * theta).sum(axis=0) / S1
    Q = (w * (theta - theta_bar[None, :]) ** 2).sum(axis=0)
    df = (n_use - 1).astype(float)
    C = S1 - S2 / S1                      # > 0 whenever >= 2 usable blocks
    with np.errstate(divide='ignore', invalid='ignore'):
        tau2 = (Q - df) / C
    tau2 = np.clip(np.nan_to_num(tau2), 0.0, None)
    denom = tau2 + theta_bar ** 2
    return np.where(denom > 0.0, tau2 / denom, 0.0)


def chi2_scores(features, target, sample_weights, grid_shape = None,
                fit_intercept: bool = True):
    """Per-term coefficient-constancy from the CUMULATIVE SCORE PATH.

    Theta comes from OLS on ALL the data and is never re-estimated; what
    propagates is the running score, once along EACH grid axis. With
    ``s[i,j] = w_i X_ij r_i`` the per-observation contribution to term
    ``j``'s normal equation::

        c        = (Xa^T W Xa)^{-1} Xa^T W y      (global weighted OLS)
        r        = y - Xa . c
        S_j^d(t) = sum of s[:,j] over the first t levels of grid axis d
        D_j      = sum_i (w_i X_ij (c_j X_ij))^2  (own SIGNAL energy)
        L_j^d    = (1 / n_d) sum_t S_j^d(t)^2 / D_j
        L_j      = mean_d L_j^d                   (axes with >= 3 levels)

    The global normal equations force every ``S_j^d`` to end at zero
    exactly, so each term owns one path per axis, pinned at both ends: it
    can only score by BULGING in between. Under a constant coefficient the
    bulge is the aimless wander of a Brownian bridge; a coefficient that
    really drifts leaves the term
    systematically under-fitted over one stretch of t and over-fitted over
    another, which is a large one-sided excursion. ``L_j`` is the
    Cramer-von Mises functional of that path -- the Nyblom-Hansen
    parameter-constancy statistic.

    Why this shape, given what has already failed in this tree:

    * **Per-term, so an extra term costs something.** A spurious column
      whose coefficient the fit drives to ~0 still owns a score path with
      a nonzero bulge, so it adds a strictly positive ``L_j``. The
      residual-distribution revision could not do this: a zero-coefficient
      term leaves the residual field bit-identical (``kdv`` and
      ``burgers_inviscid`` scored the truth and its overfit at exactly
      330.5 and 2.301e4), so ANY statistic reading only the residual ties,
      loses on discrepancy, and is dominated. It scored 5/14 with 8 of 9
      failures to ``wrong:overfit``
      (``projects/thesis/instability_panel_chi2.md``).
    * **The yardstick is the term's own SIGNAL energy -- nothing
      residual-derived.** Three residual-scale denominators failed here in
      sequence: ``biw_objective``'s sampling variance (sigma^2 -> 0 on the
      near-exact clean fit, ranking inverted), the contingency-table
      marginals (5/14 on the panel), and the Nyblom score energy
      ``V_j = sum s^2`` (10/14, but the truth scored WORST in 13/14
      systems). Self-normalising by the residual cancels its magnitude
      exactly -- verified L identical for residual scales 1e-1..1e-6 --
      leaving a pure COHERENCE detector, and on clean solver-free data the
      truth's deterministic FD residual is the most coherent in any
      candidate set. ``D_j`` is the same functional form with ``r_i``
      replaced by the term's own fitted signal ``c_j X_ij``: the residual
      magnitude enters the statistic instead of cancelling (the vcoef
      convention -- variation measured against LEVEL), and a junk term
      whose coefficient the fit drives to ~0 collapses its own ``D_j`` and
      explodes, the ``gamma_0^2`` degenerate-form guard shape.
    * **One path per axis, not only t.** A t-only path is blind to
      time-stationary spatial wrongness: on the standing-wave system the
      x-modulated ``sin(2x)*u_xx`` form has separable misfit
      ``s = g(x) h(t)`` with ``h >= 0``, and the normal equations pin
      ``sum_x g ~ 0`` -- every time-level increment vanishes and the
      t-path lies flat, 0.54x BELOW the truth (the panel's one non-strict
      cell for the t-only revision). The same misfit's x-path is the
      running sum of ``g`` and bulges. Averaging the per-axis functionals
      closes exactly that hole; single-axis (ODE) grids are unchanged.
    * **No refits.** Unlike ``tile``/``survival``/``het`` nothing is
      re-estimated on a subset, so there is no minimum block size, no
      per-block conditioning, and no lstsq fallback -- the statistic is one
      pass over the data plus one cumsum per axis.

    Scale-invariant: rescaling column ``j`` by ``a`` sends ``c_j -> c_j/a``,
    so ``S_j^2`` and ``D_j`` both pick up ``a^2``; rescaling the target or
    the weights scales both alike. NOT pivotal though -- ``L_j`` grows with
    the sample count under misfit, so values compare within a system only
    (like the discrepancy axis; all the domination lens needs). Returns one
    score per feature column (the intercept, when fitted, is dropped like
    the other estimators). Deterministic; no RNG. Fails loudly when no
    grid axis supplies >= 3 levels.
    """
    X = np.asarray(features, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    y = np.asarray(target, dtype=float).reshape(-1)
    n_samples, p = X.shape
    w = (np.asarray(sample_weights, dtype=float).reshape(-1)
         if sample_weights is not None else np.ones(n_samples))
    Xa = np.hstack([X, np.ones((n_samples, 1))]) if fit_intercept else X
    k = Xa.shape[1]

    # One path per grid axis when the grid is recognisable (C-ordered
    # flattening, the ctx.data_shape convention); a flat sample-axis path
    # otherwise. grid_shape may arrive as tuple/list/ndarray.
    gs = tuple(int(v) for v in (() if grid_shape is None else grid_shape))
    grid_ok = bool(gs) and int(np.prod(gs)) == n_samples
    axes_used = [d for d, n_d in enumerate(gs) if n_d >= 3] if grid_ok else []
    if grid_ok and not axes_used:
        raise ValueError(
            'chi2_scores needs >= 3 levels along at least one grid axis to '
            f'accumulate a score path; got grid_shape={gs}')
    if not grid_ok and n_samples < 3:
        raise ValueError(
            'chi2_scores needs >= 3 samples to accumulate a score path; '
            f'got n_samples={n_samples}')

    # THETA from OLS on ALL the data -- one fit, never re-estimated.
    wXa = w[:, None] * Xa
    c = _solve_gram(Xa.T @ wXa, Xa.T @ (w * y))
    r = y - Xa @ c

    # Exact-fit floor (the ``heterogeneity_scores`` convention): below it the
    # residual is accumulated float rounding and the score path is reading
    # round-off, not physics. An equation that reproduces the target to
    # machine precision has nothing left to test.
    rss = float(np.dot(r, w * r))
    yy = float(np.dot(y, w * y))
    if not rss > np.finfo(float).eps * max(n_samples, 1) * yy:
        return np.zeros(p)

    # Per-observation score contributions s[i, j] = w_i * X_ij * r_i. Their
    # column sums are the weighted normal equations, hence EXACTLY zero --
    # every per-axis path is pinned at both ends.
    s = wXa * r[:, None]
    # Signal-scale yardstick: the same functional form as the score energy,
    # with r_i replaced by the term's own fitted signal c_j * X_ij. A junk
    # term whose coefficient the fit drives to ~0 collapses its own D_j and
    # explodes (the vcoef gamma_0^2 degenerate-form guard); a dead column is
    # 0/0 -> 0 (no evidence). Both routed through nan_to_num, the vcoef
    # convention.
    D = ((wXa * (Xa * c[None, :])) ** 2).sum(axis=0)

    def _path_functional(increments, n_levels):
        S = np.cumsum(increments, axis=0)
        with np.errstate(divide='ignore', invalid='ignore'):
            return (S ** 2).sum(axis=0) / (n_levels * D)

    if not grid_ok:
        L = _path_functional(s, n_samples)
    else:
        sg = s.reshape(gs + (k,))
        per_axis = []
        for d in axes_used:
            other = tuple(a for a in range(len(gs)) if a != d)
            # Collapse the orthogonal axes: one increment per level of axis
            # d, i.e. the axis-d marginal of the score field.
            inc = sg.sum(axis=other) if other else sg
            per_axis.append(_path_functional(inc, gs[d]))
        L = np.mean(per_axis, axis=0)
    return np.nan_to_num(L)[:p]
