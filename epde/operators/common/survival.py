#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Block-based instability estimators for the ``Instability`` objective.

The six basis-free estimators (``_BASIS_FREE_METRICS`` below), selectable
via ``active_config().objectives.instability_metric`` beside the Gram-backed
``vcoef`` and ``cv``:

* ``survival_scores`` -- STATISTICAL axis: B bootstrap refits of the fixed
  structure, each resampling whole FIXED contiguous slabs of storage axis 0
  iid (the slab boundaries never move); a term is unstable when its
  coefficient flips sign or spreads widely across resamples.
* ``tile_scores`` -- one refit per contiguous slab of storage axis 0 (the
  same ``block_gram_partition`` slabs ``survival`` and ``het`` cut -- time
  slabs in the production (t, x) layout, not spatial tiles); a term is
  unstable when its coefficient differs between slabs. It sees variation
  along axis >= 1 only through the axis-0 drift it induces in the per-slab
  fit, so a purely spatial coefficient modulation on a separable field can
  be invisible to it.
* ``heterogeneity_scores`` / ``heterogeneity_raw_scores`` -- CALIBRATED
  heterogeneity (Cochran's Q / DerSimonian-Laird): per-block refits like
  ``tile``, but the between-block dispersion is measured against the
  SAMPLING variance the within-block fits themselves predict, not against
  the coefficient's own magnitude. A term is unstable only when its
  coefficients drift across blocks by more than the block fits' standard
  errors explain. ``het`` is bounded in [0, 1]; ``het_raw`` is not.
* ``chi2_scores`` -- per-term coefficient constancy from the CUMULATIVE
  SCORE PATH (Nyblom-Hansen). Theta is fitted once on all the data and
  never re-estimated; the running score ``S_j = cumsum(w_i X_ij r_i)``
  propagates once along EACH grid axis, pinned to zero at both ends by
  the normal equations. A constant coefficient leaves an aimless
  bridge; a drifting one leaves a large one-sided bulge. The bulge is
  measured against the term's own fitted-signal energy, never against
  the residual (three residual-scale yardsticks failed; see the
  docstring). No refits at all, so no minimum block size.
  ``chi2_centered_scores`` is the same statistic on weighted-centered
  columns and target.

The block estimators run entirely in Gram space on per-block Gram/Gy stacks
(the ``PhysicsInformedLasso`` convention: intercept column folded block-wise,
weighted by ``sample_weights``), so the data-sized work is one pass and
every refit is a small (p+1) solve. Aggregation is robust
(median / MAD) or chi-square-calibrated (``het``). Per-term scores are
dimensionless and invariant to rescaling the target. ``survival``, ``tile``,
``chi2`` and ``chi2_centered`` are also invariant to per-column and weight
rescaling (to round-off); ``het`` / ``het_raw`` are NOT -- their block ridge
sits on the raw, non-equilibrated Gram, so one column's scale moves the
ridge every other column sees, and the weight scale enters their residual
degrees of freedom through ``sum(w)``.

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
_DEFAULT_N_TILES = 8              # tile: time slabs along storage axis 0
_DEFAULT_N_BLOCKS_HET = 16        # het: refit blocks (clamped by sample count)


def _reference_width(p: int) -> int:
    """The candidate width the block-count clamp is allowed to depend on.

    ``tile`` and ``het`` size their partition as ``n_samples // (4*(w+1))`` so
    every block keeps >= 4 samples per unknown. Using the CANDIDATE's own ``p``
    for ``w`` made the partition candidate-dependent, and in the one direction
    that flatters complexity: a wider equation got fewer, larger blocks, hence
    lower between-block dispersion, hence a BETTER instability score. That is a
    combinatorial reward for adding terms, paid on the Pareto axis.

    ``survival_scores`` already refuses to let ``p`` reach its randomness for
    the same reason -- see the common-random-numbers note there. This is the
    same principle applied to the partition: resolve the width from the search
    configuration's ``equation_terms_max_number`` (the widest equation the
    search can build), so within one search every candidate is scored on the
    SAME partition and score differences are differences in the equations.

    ``max`` with the candidate's own ``p`` keeps the >= 4-samples-per-unknown
    guarantee intact when a caller hands over a matrix wider than the
    configured cap -- offline analysis tools do.
    """
    try:
        from epde.interface.search_config import active_config
        configured = int(active_config().search_space.equation_terms_max_number)
    except Exception:
        configured = 0
    return max(int(p), configured)
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

    Dimensionless and outlier-robust. A column whose median coefficient is
    ~0 while its draws still spread maps to a huge (nan_to_num-capped) score;
    but a column whose draws are all exactly zero (a dead column) is 0/0 ->
    0.0, the BEST score -- this is not an unresolved-level guard like
    ``het``'s (:func:`_het_unresolved`).
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

    B bootstrap draws over FIXED contiguous slabs of storage axis 0: each draw
    resamples whole slabs iid (multinomial slab counts reweight the per-slab
    Grams, ``G* = sum_b c_b G_b``) -- not a moving-block bootstrap, the slab
    boundaries never move -- and the fixed structure is refit per draw. No
    re-selection, so the score is independent of the sparsity pass. Per-term
    score::

        score_j = sign_flip_rate_j + MAD_j / |median_j|

    where a flip is a draw whose coefficient sign differs from the median
    sign. Deterministic: the RNG is a local ``default_rng`` seeded from
    ``(seed, B)`` -- never from ``p`` (see below), and the global NumPy
    stream is never touched.
    """
    G_blocks, Gy_blocks = block_gram_partition(
        features, target, sample_weights, grid_shape, n_blocks)
    n_blocks = G_blocks.shape[0]
    p = G_blocks.shape[1] - 1
    sl = slice(None) if fit_intercept else slice(0, p)

    # The stream must NOT depend on ``p``. It used to, and that made this the
    # one estimator whose two call shapes disagreed beyond round-off: an
    # explicit ones column with ``fit_intercept=False`` (p = nf + 1) and a
    # folded intercept (p = nf) resampled one design matrix two different
    # ways. (The other block estimators agree across the two shapes only to
    # the float path -- the folded Gram is a different summation; the
    # ``Instability`` objective now uses the keep-rule's explicit shape, so
    # its two consumers make the same call.)
    #
    # Dropping it also buys COMMON RANDOM NUMBERS: candidate structures of
    # different widths now share one block-resampling pattern, so differences
    # between their scores are differences in the equations rather than in the
    # draws. Determinism was never carried by ``p`` -- the fixed seed does it.
    rng = np.random.default_rng(
        np.uint64(0x5EED) + np.uint64(seed) * 7919 + np.uint64(B))
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
    """Per-term between-slab instability.

    One refit of the fixed structure per contiguous slab of storage axis 0
    (:func:`block_gram_partition`, the partition ``survival`` and ``het``
    use; in the production (t, x) layout these are TIME slabs, not spatial
    tiles); per-term score = robust between-slab dispersion
    ``MAD_j / |median_j|`` of the slab coefficients. A true
    constant-coefficient term agrees across slabs -> ~0; a term whose
    coefficient drifts along axis 0 needs different coefficients per slab ->
    large. Variation along axis >= 1 reaches it only through the axis-0 drift
    it induces in the per-slab fit: a purely spatial modulation on a
    separable field can score at round-off, exactly like the truth, and
    transposing the storage flips which axis it sees.

    ``n_tiles`` is clamped so each tile keeps >= 4 samples per unknown
    (else the per-tile solves are meaninglessly under-determined). The width
    that clamp reads is the SEARCH's widest allowed equation, not this
    candidate's -- see :func:`_reference_width`.
    """
    X = np.asarray(features, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    n_samples, p = X.shape
    max_tiles = max(2, n_samples // (4 * (_reference_width(p) + 1)))
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


def _het_components(features, target, sample_weights, grid_shape,
                    fit_intercept: bool = True,
                    n_blocks: int = _DEFAULT_N_BLOCKS_HET):
    """Per-term calibrated heterogeneity: Q-calibrated excess variance.

    One refit of the fixed structure per contiguous block, PLUS the
    per-block coefficient sampling variance from the same Gram solve::

        s2_bj   = sigma2_b * [(X_b^T W X_b)^{-1}]_jj
        Q_j     = sum_b (theta_bj - theta_bar_j)^2 / s2_bj
        tau2_j  = max(0, (Q_j - df) / C_j)          (DerSimonian-Laird)
        score_j = tau2_j / (tau2_j + theta_bar_j^2)  in [0, 1]

    with ``score_j = 1`` reserved for a coefficient whose LEVEL is itself
    unresolved (``theta_bar^2 * S1 <= 1``, i.e. within one standard error of
    zero) -- the degenerate-form guard, see the comment on the return.

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
    * no division by a coefficient mean -- bounded and singularity-free.
      It is NOT invariant to per-column or weight rescaling, though: the
      ridge below is placed on the raw, non-equilibrated block Gram, so one
      column's scale changes the effective ridge the others see, and
      ``sum(w)`` enters the residual degrees of freedom.

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
    max_blocks = n_samples // (4 * (_reference_width(p) + 1))
    if max_blocks < 3:
        raise ValueError(
            'heterogeneity_scores needs >= 3 blocks of >= 4*(w+1) samples '
            f'to calibrate; got n_samples={n_samples} with p={p} and reference '
            f'width w={_reference_width(p)} (equation_terms_max_number)')
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
    return tau2, theta_bar, S1


def _het_unresolved(theta_bar, S1):
    """Mask of coefficients whose LEVEL is indistinguishable from zero.

    DEGENERATE-FORM GUARD. Without it this estimator returns its BEST score
    (0.0) for a coefficient it knows nothing about, because both routes to a
    small score converge: a well-determined constant coefficient gives
    ``tau2 == 0``, and so does a column so poorly excited that every block's
    ``s2`` swamps its scatter, driving ``Q`` below ``df``. The second case is
    absence of evidence, and reporting it as evidence of stability is what
    let a hitchhiking column ride for free -- on Allen-Cahn, appending a
    coordinate-modulated copy of ``u_xx`` (fitted coefficient -9.7e-07)
    scored exactly 0.0 AND shaved 0.8% off the real term's score, so the
    summed objective DROPPED while the discrepancy also improved and the
    true equation was dominated on both axes. Measured 9/13 systems
    non-dominated before this guard, 13/13 after.
    
    ``S1`` is the total Fisher information for the coefficient and ``1/S1``
    the sampling variance of ``theta_bar``, so ``theta_bar**2 * S1`` is that
    coefficient's squared t-statistic against zero -- both already computed
    above, and invariant to per-column rescaling (``theta_bar`` picks up
    ``1/a``, ``S1`` picks up ``a**2``). Below one standard error the level
    is indistinguishable from zero, the ratio's denominator is noise, and no
    claim of homogeneity can be made: the estimator returns its supremum
    instead. Only ``het`` / ``het_raw`` carry such an unresolved-LEVEL guard;
    the rest of the family guards exact zeros at most (``vcoef``'s
    ``C == 0`` maps to the cap; ``chi2``'s collapsing ``D_j`` lifts a ~0 but
    nonzero level, while an exactly dead column is 0/0 -> 0.0, as it is for
    ``survival``, ``tile``, ``cv`` and ``chi2_centered``).
    
    This does NOT reintroduce significance testing, which stays the t-stat
    anchor's job: a column whose scatter merely matches its standard errors
    still scores ~0 as long as its LEVEL is resolved. Only a coefficient
    that is itself unresolved is refused a stability certificate, so the
    valid-but-degenerate analytical identities this estimator deliberately
    does not flag keep their low scores whenever their coefficients are
    determined.
    """
    return ~(theta_bar ** 2 * S1 > 1.0)


def heterogeneity_scores(features, target, sample_weights, grid_shape,
                         fit_intercept: bool = True,
                         n_blocks: int = _DEFAULT_N_BLOCKS_HET):
    """BOUNDED calibrated heterogeneity, ``tau2 / (tau2 + theta_bar^2)``.

    See :func:`_het_components` for the construction. The excess variance is
    divided by ``tau2 + theta_bar^2`` rather than by ``theta_bar^2`` alone, so
    the score is confined to ``[0, 1]`` and a shrinking level saturates it
    instead of sending it to infinity. Unresolved levels return the supremum
    ``1.0`` via :func:`_het_unresolved`. Compare
    :func:`heterogeneity_raw_scores`, which keeps the same numerator over the
    bare level and is unbounded above.
    """
    tau2, theta_bar, S1 = _het_components(features, target, sample_weights,
                                          grid_shape, fit_intercept, n_blocks)
    denom = tau2 + theta_bar ** 2
    score = np.where(denom > 0.0, tau2 / denom, 0.0)
    return np.where(_het_unresolved(theta_bar, S1), 1.0, score)


def heterogeneity_raw_scores(features, target, sample_weights, grid_shape,
                             fit_intercept: bool = True,
                             n_blocks: int = _DEFAULT_N_BLOCKS_HET):
    """UNBOUNDED calibrated heterogeneity, ``tau2 / theta_bar^2``.

    The same DerSimonian-Laird excess variance as
    :func:`heterogeneity_scores`, divided by the bare squared level. This is
    the ``var / mu^2`` shape the rest of the family uses (``vcoef``'s
    ``NC / gamma_0^2``, ``cv``'s ``std^2 / mu^2``), so it ranks a drifting
    coefficient without the compression the bounded form applies near 1: two
    forms that both saturate the bounded score can still be ordered here.

    The cost is the family's usual one -- it is unbounded above, and a
    collapsing level sends it to ``+inf``, which ``np.nan_to_num`` renders as
    the cap ~1.8e308. The ``Instability`` reduction treats the cap as
    absorbing (one or more capped terms make the equation value exactly the
    cap, never ``inf`` or half the cap), so a capped equation is the worst
    on the axis but every capped equation ties there. Note
    the bound was never what caused the hitchhiking failure the guard in
    :func:`_het_unresolved` fixes: that came from ``tau2`` itself clipping to
    zero, which this form shares, so the same guard applies here.
    """
    tau2, theta_bar, S1 = _het_components(features, target, sample_weights,
                                          grid_shape, fit_intercept, n_blocks)
    with np.errstate(divide='ignore', invalid='ignore'):
        score = tau2 / (theta_bar ** 2)
    score = np.where(_het_unresolved(theta_bar, S1), np.inf, score)
    return np.nan_to_num(score)


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
    score per column of ``features``: an intercept the estimator appends
    itself (``fit_intercept=True``) is fitted and its entry dropped. A
    caller that wants the intercept SCORED -- the keep-rule and, under the
    regularized-intercept rule, the ``Instability`` objective -- passes an
    explicit ones column with ``fit_intercept=False`` instead. Deterministic;
    no RNG. Fails loudly when no grid axis supplies >= 3 levels.
    """
    X = np.asarray(features, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    y = np.asarray(target, dtype=float).reshape(-1)
    n_samples, p = X.shape
    w = (np.asarray(sample_weights, dtype=float).reshape(-1)
         if sample_weights is not None else np.ones(n_samples))
    Xa = np.hstack([X, np.ones((n_samples, 1))]) if fit_intercept else X
    path_grid = _score_path_grid(grid_shape, n_samples)
    L, _, _ = _chi2_fit(Xa, y, w, path_grid)
    return np.nan_to_num(L)[:p]


def _score_path_grid(grid_shape, n_samples):
    """``(gs, axes_used)`` for the chi2 score paths -- the fail-loud grid
    contract shared by :func:`chi2_scores` and :func:`chi2_centered_scores`.

    One path per grid axis when the grid is recognisable (C-ordered
    flattening, the ctx.data_shape convention); a flat sample-axis path
    otherwise, signalled by ``gs is None``. grid_shape may arrive as
    tuple/list/ndarray.
    """
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
    return (gs if grid_ok else None), axes_used


def _path_functional(s, D, path_grid):
    """``L_j = mean_d (1/n_d) sum_t S_j^d(t)^2 / D_j`` for per-observation
    score contributions ``s`` (n, k) and yardsticks ``D`` (k,) -- the
    Cramer-von Mises functional of each column's cumulative score path, one
    path per usable grid axis (see :func:`chi2_scores`). Not nan_to_num'd:
    the caller owns the exceptional-value convention."""
    gs, axes_used = path_grid

    def _one(increments, n_levels):
        S = np.cumsum(increments, axis=0)
        with np.errstate(divide='ignore', invalid='ignore'):
            return (S ** 2).sum(axis=0) / (n_levels * D)

    if gs is None:
        return _one(s, s.shape[0])
    sg = s.reshape(gs + (s.shape[1],))
    per_axis = []
    for d in axes_used:
        other = tuple(a for a in range(len(gs)) if a != d)
        # Collapse the orthogonal axes: one increment per level of axis
        # d, i.e. the axis-d marginal of the score field.
        inc = sg.sum(axis=other) if other else sg
        per_axis.append(_one(inc, gs[d]))
    return np.mean(per_axis, axis=0)


def _chi2_fit(Xa, y, w, path_grid):
    """The chi2 statistic on the design ``Xa`` exactly as given (no column
    appended): ``(L, r, c)`` -- the raw per-column functional, the global
    weighted-OLS residual and coefficients. See :func:`chi2_scores`."""
    # THETA from OLS on ALL the data -- one fit, never re-estimated.
    wXa = w[:, None] * Xa
    c = _solve_gram(Xa.T @ wXa, Xa.T @ (w * y))
    r = y - Xa @ c

    # NO exact-fit floor. There used to be one -- ``rss <= eps*n*yy`` returned
    # all-zero scores, on the reading that a machine-precision fit has nothing
    # left to test. It is unnecessary and, since this statistic also scales the
    # sparsity keep-rule's L1 threshold, actively harmful: an all-zero vector
    # is a TIE, the threshold vanishes, coordinate descent degenerates to plain
    # OLS and dead columns ride along at ~1e-13. On the 5-seed ODE benchmark
    # that was the difference between 0/5 and 5/5 recoveries.
    # Unnecessary, because the natural computation already handles the case:
    # with a round-off residual the numerator is round-off SQUARED while D_j
    # stays the term's real signal energy, so a true term scores ~1e-30 -- the
    # same "nothing to test" answer -- while a dead column's collapsing D_j
    # still lifts it clear of the others. The floor destroyed exactly the
    # ordering that survives without it.

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
    return _path_functional(s, D, path_grid), r, c


def chi2_centered_scores(features, target, sample_weights, grid_shape=None,
                         fit_intercept: bool = True):
    """:func:`chi2_scores` on weighted-CENTERED columns and target.

    Identical statistic, computed after subtracting each column's and the
    target's ``sample_weights``-weighted mean. The motivation is what the
    score path is made of: with ``s[i,j] = w_i X_ij r_i``, a constant ``c0``
    left in the residual contributes ``c0 * cumsum(w_i X_ij)`` to term ``j``'s
    path -- the column's own running-sum profile, entering as if it were
    coefficient drift, with nothing in ``D_j`` to offset it. A column with a
    large mean therefore earns a monotone excursion that says nothing about
    the stability of its coefficient.

    That leak is live rather than hypothetical: under the regularized-intercept
    rule (``Instability.compute``, ``sparsity.instability_scores``) a design
    whose intercept the sparsity step drove to zero is scored with no ones
    column at all, so the weighted normal equations no longer force
    ``sum_i w_i r_i = 0``.

    Centering removes the offset before the path is accumulated instead of
    relying on an intercept column to absorb it. Measured against ``chi2`` on
    a mean-carrying manufactured library, it widens the separation between a
    weak TRUE term and a spurious column roughly fourfold (8.1e4 vs 2.1e4),
    and it is 13/13 non-dominated on the system panel. It does NOT address the
    ``1/c_j^2`` weak-term bias -- that bias is inseparable from this family's
    ability to flag a spurious column at all, since both read a collapsing
    fitted level (see the ``D_j`` discussion in :func:`chi2_scores`).

    A CONSTANT column (exactly constant over the rows with ``w > 0``) is left
    out of the centered design -- centering would zero it -- and scored with
    the chi2 INTERCEPT functional instead. Centering is an implicit intercept
    (Frisch-Waugh-Lovell), so the centered residual ``r`` IS the uncentered
    fit-with-intercept residual, and a constant column of value ``k`` gets
    exactly the entry :func:`chi2_scores` returns for it when it carries the
    intercept: increments ``w_i k r_i``, yardstick
    ``D_0 = c_0^2 k^2 sum_i w_i^2`` with ``c_0`` the equivalent uncentered
    intercept (weighted mean of ``y`` minus ``sum_j c_j`` times the weighted
    mean of ``X_j``). ``k`` cancels, so the score is a property of the offset,
    not of the constant's value. This is the column the keep-rule appends
    (``sparsity.instability_scores``, ``fit_intercept=False``) and the
    objective appends whenever the intercept survived, so the intercept gets
    a real L1 threshold and can be pruned like under every other metric; it
    used to score ``0.0``, so a round-off intercept was never pruned.

    Several constant columns are exactly collinear, so how the offset splits
    between them is not identifiable; each is scored as the sole carrier of
    the whole ``c_0``, which makes them tie rather than depend on an arbitrary
    split. A ``c_0`` of exactly zero follows the ``D_j -> 0`` convention of
    :func:`chi2_scores` (``nan_to_num``).

    Scale-invariant and deterministic, exactly as :func:`chi2_scores`; the
    same fail-loud grid contracts apply.
    """
    X = np.asarray(features, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    y = np.asarray(target, dtype=float).reshape(-1)
    n_samples, p = X.shape
    w = (np.asarray(sample_weights, dtype=float).reshape(-1)
         if sample_weights is not None else np.ones(n_samples))

    sw = float(w.sum())
    if not sw > 0.0:
        raise ValueError('chi2_centered_scores: sample weights sum to zero, '
                         'so no weighted mean is defined')
    # EXACT constancy over the rows that carry weight. The old test,
    # spread > eps*max(|mu|, 1) on a cancelling E[X^2] - mu^2, missed constants
    # such as 0.1, 3.7 and 1e-8 (and the ones column itself under non-unit
    # weights), centring them into round-off garbage scored up to 1e30, while
    # declaring any column below ~1e-16 constant -- a free 0.0 for a resolved
    # junk term that chi2 scores 11.7.
    support = w > 0.0
    const = np.ptp(X[support], axis=0) == 0.0
    mu = (w @ X) / sw
    y_mean = (w @ y) / sw

    keep = np.flatnonzero(~const)
    Xc = X[:, keep] - mu[keep][None, :]
    Xca = np.hstack([Xc, np.ones((n_samples, 1))]) if fit_intercept else Xc
    path_grid = _score_path_grid(grid_shape, n_samples)
    L, r, c = _chi2_fit(Xca, y - y_mean, w, path_grid)

    scores = np.zeros(p, dtype=float)
    scores[keep] = np.nan_to_num(L)[:keep.size]
    cidx = np.flatnonzero(const)
    if cidx.size:
        # The uncentered intercept the centered fit implies; an intercept
        # appended to the centered design is W-orthogonal to it and ~0, but
        # it is part of the same offset.
        c0 = (y_mean - float(c[:keep.size] @ mu[keep])
              + (float(c[-1]) if fit_intercept else 0.0))
        k_vals = X[support][0, cidx]
        s0 = (w * r)[:, None] * k_vals[None, :]
        D0 = c0 ** 2 * k_vals ** 2 * float(np.sum(w ** 2))
        scores[cidx] = np.nan_to_num(_path_functional(s0, D0, path_grid))
    return scores


#: THE basis-free estimator table -- the single source of truth for which
#: instability metrics need no Gram setup. ``Instability.compute`` dispatches
#: on it and ``sparsity._KEEP_RULE_ESTIMATORS`` IS it (same object, not a
#: copy), so the objective and the keep-rule cannot drift apart as metrics are
#: added: one statistic, both sides. ``vcoef`` and ``cv`` are absent because
#: they score from their own Gram setup rather than from this module.
#:
#: It lives here rather than in ``objectives`` so that ``sparsity`` can reach
#: it without importing ``objectives`` -- that edge would close a cycle,
#: since ``subset_selection`` imports ``sparsity``.
_BASIS_FREE_METRICS = {
    'survival': survival_scores, 'tile': tile_scores,
    'het': heterogeneity_scores, 'het_raw': heterogeneity_raw_scores,
    'chi2': chi2_scores, 'chi2_centered': chi2_centered_scores,
}
