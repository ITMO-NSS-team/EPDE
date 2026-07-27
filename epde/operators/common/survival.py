#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Block-based instability estimators for the ``Instability`` objective.

Two alternatives to the varying-coefficient (``vcoef``) score, selectable
via ``epde.globals.instability_metric`` (see ``set_instability_metric``):

* ``survival_scores`` -- STATISTICAL axis: B moving-block bootstrap
  refits of the fixed structure; a term is unstable when its coefficient
  flips sign or spreads widely across resamples.
* ``tile_scores`` -- basis-free SPATIAL axis (Hadamard): one refit per
  disjoint domain tile; a term is unstable when its coefficient differs
  between tiles. Unlike the additive per-axis cosine basis of ``vcoef``
  this makes no separability or smoothness commitment, so it can flag
  multiplicative x-t coefficient variation.

Both run entirely in Gram space on per-block Gram/Gy stacks (the
``PhysicsInformedLasso`` convention: intercept column folded block-wise,
weighted by ``sample_weights``), so the data-sized work is one pass and
every refit is a small (p+1) solve. Aggregation is robust
(median / MAD), and per-term scores are dimensionless -- invariant to
per-column and global rescaling, like the vcoef and CV scores.
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


def block_gram_partition(features, target, sample_weights, grid_shape,
                         n_blocks: int):
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
    no-intercept solve.
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
    grid_shape = tuple(int(s) for s in (() if grid_shape is None else grid_shape))
    if grid_shape and int(np.prod(grid_shape)) == n_samples and grid_shape[0] > 1:
        # Align slab boundaries to whole time steps.
        n_t = grid_shape[0]
        stride = n_samples // n_t
        n_blocks = max(2, min(int(n_blocks), n_t))
        t_edges = np.linspace(0, n_t, n_blocks + 1, dtype=int)
        edges = t_edges * stride
    else:
        n_blocks = max(2, min(int(n_blocks), n_samples // 2))
        edges = np.linspace(0, n_samples, n_blocks + 1, dtype=int)

    G_blocks = np.zeros((n_blocks, p + 1, p + 1))
    Gy_blocks = np.zeros((n_blocks, p + 1))
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
