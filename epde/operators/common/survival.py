#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Block-based instability estimators for the ``Instability`` objective.

The seven basis-free estimators (``_BASIS_FREE_METRICS`` below), selectable
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
* ``dyad_scores`` -- ``tile``'s statistic averaged over EVERY dyadic
  partition (2, 4, 8, ... slabs down to the same samples-per-unknown floor)
  of every grid axis in turn, averaged over the axes, so no slab count is
  chosen and a coefficient drifting along space is seen as well as one
  drifting in time. Parameter-free. Its one-axis component,
  ``dyad_t_scores`` (the partitions of storage axis 0), is not a metric of
  its own.
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
``dyad``, ``chi2`` and ``chi2_centered`` are also invariant to per-column and
weight rescaling (to round-off); ``het`` / ``het_raw`` are NOT -- their block ridge
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

import functools

import numpy as np

from epde import _loop_stats

# Estimator defaults -- fixed algorithm policy, not per-run tuning knobs
# (no in-tree caller overrides them; ``Instability.compute`` calls both
# estimators with defaults). The signature parameters exist for offline
# experiments (e.g. the instability panel), not for the live search.
_DEFAULT_BOOTSTRAP_DRAWS = 32     # B: block-bootstrap refits per equation
_DEFAULT_N_BLOCKS = 16            # survival: contiguous time-slab count
_DEFAULT_N_TILES = 8              # tile: time slabs along storage axis 0
_DEFAULT_N_BLOCKS_HET = 16        # het: refit blocks (clamped by the axis-0 levels)


def _block_rows(width) -> int:
    """The rows every block or slab keeps at reference width ``width``:
    ``search_config.SLAB_ROWS_PER_COLUMN * (width + 1)``, read at call time."""
    from epde.interface import search_config
    return search_config.SLAB_ROWS_PER_COLUMN * (int(width) + 1)


def _reference_width(p: int) -> int:
    """The candidate width the block-count clamp is allowed to depend on.

    ``tile`` and ``het`` size their partition so every block keeps
    ``SLAB_ROWS_PER_COLUMN*(w+1)`` rows, that many samples per unknown
    (:func:`_level_slab_capacity`).
    Using the CANDIDATE's own ``p`` for ``w`` made the partition
    candidate-dependent, and in the one direction that flatters complexity: a
    wider equation got fewer, larger blocks, hence lower between-block
    dispersion, hence a BETTER instability score. That is a combinatorial
    reward for adding terms, paid on the Pareto axis.

    ``survival_scores`` already refuses to let ``p`` reach its randomness for
    the same reason -- see the common-random-numbers note there. This is the
    same principle applied to the partition: resolve the width from the search
    configuration's ``equation_terms_max_number`` (the widest equation the
    search can build), so within one search every candidate is scored on the
    SAME partition and score differences are differences in the equations.

    ``max`` with the candidate's own ``p`` keeps the samples-per-unknown
    guarantee intact when a caller hands over a matrix wider than the
    configured cap -- offline analysis tools do.

    Under ``'assignment'`` mode's ``selected_terms_max_number`` no fit is
    wider than that cap (the constant included), so the widest equation is
    the smaller of the two caps: a structure holding more terms to choose from
    leaves the partition of every candidate as it was.
    """
    try:
        from epde.interface.search_config import active_config
        configured = configured_width(active_config().search_space)
    except Exception:
        configured = 0
    return max(int(p), configured)


def configured_width(search_space) -> int:
    """The widest equation a search configuration can fit: its
    ``equation_terms_max_number``, or its mode's ``selected_terms_max_number``
    when that is smaller (see :func:`_reference_width`)."""
    from epde.interface.search_config import rps_param
    configured = int(search_space.equation_terms_max_number)
    selected = rps_param(search_space, 'selected_terms_max_number')
    if selected is not None:
        configured = min(configured, int(selected))
    return configured


def row_floor(metric: str, grid_shape, w_ref: int):
    """Why a grid of ``grid_shape`` is too small for ``metric``'s partition at
    reference width ``w_ref``, or ``None`` when it is not.

    The estimators clamp their partitions rather than refuse (``tile`` and
    ``dyad`` to two slabs), so a short grid degrades them silently; the row
    budget asks first. Every block or slab must keep ``SLAB_ROWS_PER_COLUMN * (w_ref + 1)`` rows --
    measured on the partition the estimator actually cuts, at whole levels of
    axis 0, not on the row total. Each estimator takes its block count from
    the axis-0 levels (:func:`_level_slab_capacity`), so a slab falls below
    the floor only where the estimator's minimum count (two slabs, three
    blocks) overrides it. ``chi2`` needs three levels along some axis. Other
    metrics need no floor here.

    The block floors are monotone along every axis but the first, and along
    the first from two levels on. A single axis-0 level is off that scale:
    the estimators cut it as one row per level (:func:`_axis0_layout`), so it
    gets the verdict of the 1-D grid of the same rows. For ``tile`` and
    ``dyad`` that still fits -- two flat slabs need twice the rows two whole
    levels do -- but ``het`` cuts three flat blocks from one level holding
    three blocks' rows, where two levels make at most two blocks: at width 6
    ``(1, 100)`` passes, ``(2, 100)`` is refused and ``(3, 100)`` passes.

    Two-level grids under ``het`` / ``het_raw`` are the one case the level
    count refuses that the row-total count passed: it measured only the two
    slabs :func:`_slab_edges` clamps two levels to. ``het`` raised on them
    anyway (fewer than three usable blocks), so the row budget now refuses
    them at fit start instead of the search failing at its first ``het``
    score. Under a row budget that takes an interior box two levels long
    along axis 0: the stride resolver keeps three levels per axis otherwise.
    """
    shape = tuple(int(n) for n in grid_shape)
    rows = int(np.prod(shape))
    block = _block_rows(w_ref)
    if metric in ('chi2', 'chi2_centered'):
        if not any(n >= 3 for n in shape):
            return f'{metric} needs 3 levels along some axis, the grid is {shape}'
        return None
    if metric in ('het', 'het_raw'):
        capacity = _level_slab_capacity(rows, shape, block)
        if capacity < 3:
            n_0, stride = _axis0_layout(rows, shape)
            levels = -(-block // stride)
            return (f'{metric} needs 3 blocks of {block} rows at whole levels of axis 0 '
                    f'({3 * levels * stride} rows, {3 * levels} levels of {stride}), '
                    f'the grid has {rows} ({n_0} levels)')
        parts = [_slab_edges(rows, shape, max(3, min(_DEFAULT_N_BLOCKS_HET, capacity)))]
    elif metric == 'tile':
        capacity = _level_slab_capacity(rows, shape, block)
        parts = [_slab_edges(rows, shape, max(2, min(_DEFAULT_N_TILES, capacity)))]
    elif metric == 'dyad':
        # Every axis dyad cuts (_axis_layouts), each as storage axis 0.
        parts = [edges for _axis, _order, moved in _axis_layouts(rows, shape)
                 for edges in _dyadic_slab_edges(rows, moved, block)]
    else:
        return None
    smallest = min(int(np.min(np.diff(edges))) for edges in parts)
    if smallest < block:
        return (f'{metric} cuts the grid {shape} into blocks along axis 0 as small as {smallest} rows, '
                f'below {block} = {block // (int(w_ref) + 1)} * (width {int(w_ref)} + 1)')
    return None
# chi2 needs no size knob: one score path per grid axis, each accumulated
# at that axis's own level resolution, so there is nothing to tune.


def _slab_edges(n_samples: int, grid_shape, n_blocks: int):
    """Contiguous slab boundaries over the flat sample axis.

    When the flat sample count matches ``prod(grid_shape)`` (the standard
    case: C-ordered flattening of the grid), boundaries are aligned to
    whole steps of the FIRST grid axis -- time -- so each slab is a
    physically contiguous space-time chunk; otherwise, or when that axis has
    a single level, plain contiguous index slabs are used. ``n_blocks`` is
    clamped to what the axis can support. Returns the ``(n_blocks + 1,)``
    edge array.
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
    C-ordered flattening of the grid) and the first grid axis has two or
    more levels, slab boundaries are aligned to whole steps of that axis
    (time slabs) so each block is a physically contiguous space-time chunk;
    otherwise -- a single axis-0 level included -- plain contiguous index
    slabs are used (:func:`_slab_edges`).

    Returns ``(G_blocks, Gy_blocks)`` with shapes ``(B, p+1, p+1)`` and
    ``(B, p+1)``; the last row/column is the folded intercept (the
    ``sparsity.py`` Gram convention), so callers slice ``[:p, :p]`` for a
    no-intercept solve. With ``return_yy=True`` a third array ``(B,)`` of
    per-block weighted target energies ``y^T W y`` is appended (needed by
    the residual-variance computation in ``heterogeneity_scores``).
    """
    n_samples = np.shape(features)[0]
    # grid_shape may arrive as a tuple, list, or ndarray (ctx.data_shape in
    # the live fitness path) -- avoid ambiguous ndarray truth tests.
    edges = _slab_edges(n_samples, grid_shape, n_blocks)
    return _edge_gram_blocks(features, target, sample_weights, edges,
                             return_yy=return_yy)


def _edge_gram_blocks(features, target, sample_weights, edges,
                      return_yy: bool = False):
    """:func:`block_gram_partition`'s per-block augmented Gram stacks over the
    flat-index slabs ``[edges[b], edges[b+1])`` given explicitly."""
    X = np.asarray(features, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    y = np.asarray(target, dtype=float).reshape(-1)
    n_samples, p = X.shape
    sw = (np.asarray(sample_weights, dtype=float).reshape(-1)
          if sample_weights is not None else np.ones(n_samples))
    n_blocks = edges.size - 1

    G_blocks = np.zeros((n_blocks, p + 1, p + 1))
    Gy_blocks = np.zeros((n_blocks, p + 1))
    yy_blocks = np.zeros(n_blocks)
    # The weighted products are formed once, outside the slab loop: being
    # elementwise, their slices equal the per-slab products bit for bit, and
    # every slab keeps the same reductions on views of the same layout. The
    # Grams are unchanged to the last bit; the loop does a third fewer numpy
    # calls per slab, which is what dyad_t's one-row-per-level fallback spends
    # its time on (32 slabs per keep-rule call on an ODE).
    wX = sw[:, None] * X
    swy = sw * y
    for b in range(n_blocks):
        lo, hi = edges[b], edges[b + 1]
        Xb, wXb, swyb = X[lo:hi], wX[lo:hi], swy[lo:hi]
        G_blocks[b, :p, :p] = Xb.T @ wXb
        ws = wXb.sum(axis=0)
        G_blocks[b, :p, -1] = ws
        G_blocks[b, -1, :p] = ws
        G_blocks[b, -1, -1] = float(sw[lo:hi].sum())
        Gy_blocks[b, :p] = Xb.T @ swyb
        Gy_blocks[b, -1] = float(swyb.sum())
        if return_yy:
            yy_blocks[b] = float(np.dot(y[lo:hi], swyb))
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

    ``n_tiles`` is clamped so each tile keeps ``SLAB_ROWS_PER_COLUMN`` samples per unknown
    (else the per-tile solves are meaninglessly under-determined), counted
    in whole levels of axis 0 as the tiles are cut
    (:func:`_level_slab_capacity`); below two tiles' worth of levels the
    two-tile floor wins. The width that clamp reads is the SEARCH's widest
    allowed equation, not this candidate's -- see :func:`_reference_width`.
    """
    X = np.asarray(features, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    n_samples, p = X.shape
    max_tiles = max(2, _level_slab_capacity(n_samples, grid_shape,
                                            _block_rows(_reference_width(p))))
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


def _dyadic_slab_edges(n_samples: int, grid_shape, min_rows: int):
    """Flat-index edges of the dyadic partitions of storage axis 0.

    Partition ``s`` (``s = 1, 2, ...``) cuts axis 0 into ``2**s`` equal slabs,
    ``np.linspace(0, n_0, 2**s + 1).round()`` levels, for as long as every slab
    keeps at least ``ceil(min_rows / rows_per_level)`` levels. The level
    edges are nested (each coarser edge is an edge of every finer partition),
    so the finest partition's block Grams sum to every coarser one's. When
    the flat sample count matches ``prod(grid_shape)`` and the first grid
    axis has two or more levels, the slabs are whole steps of that axis
    (time); otherwise -- a single axis-0 level included, read as the 1-D grid
    of its rows (:func:`_axis0_layout`) -- plain contiguous index slabs with
    ``min_rows`` samples each. At least one partition (two slabs) is always
    returned, like ``tile``'s two-tile floor.

    Memoized on the three arguments, which are all the partitions depend on:
    dyad_t asks for them on every keep-rule call and rebuilding them was a
    measurable share of its cost. The arrays are shared between calls, so
    they come back READ-ONLY.
    """
    shape = tuple(int(s) for s in (() if grid_shape is None else grid_shape))
    return list(_dyadic_slab_edges_cached(int(n_samples), shape, int(min_rows)))


@functools.lru_cache(maxsize=256)
def _dyadic_slab_edges_cached(n_samples: int, grid_shape: tuple, min_rows: int):
    n_0, stride = _axis0_layout(n_samples, grid_shape)
    n_max = max(2, _level_slab_capacity(n_samples, grid_shape, min_rows))
    parts = []
    s = 1
    while 2 ** s <= n_max:
        edges = np.linspace(0, n_0, 2 ** s + 1).round().astype(int) * stride
        edges.setflags(write=False)
        parts.append(edges)
        s += 1
    return tuple(parts)


def _axis0_layout(n_samples: int, grid_shape):
    """``(levels along storage axis 0, rows per level)``: the grid's first axis
    when the flat sample count matches ``prod(grid_shape)`` (C-ordered
    flattening) and that axis has two or more levels, else one row per level
    -- so a single axis-0 level is read as the 1-D grid of its rows, as
    :func:`_slab_edges` cuts it."""
    grid_shape = tuple(int(s) for s in (() if grid_shape is None else grid_shape))
    if grid_shape and int(np.prod(grid_shape)) == n_samples and grid_shape[0] > 1:
        return grid_shape[0], n_samples // grid_shape[0]
    return n_samples, 1


def _level_slab_capacity(n_samples: int, grid_shape, min_rows: int) -> int:
    """The most contiguous slabs of storage axis 0 that keep ``min_rows`` rows
    each when cut at whole levels: ``n_0 // ceil(min_rows / rows_per_level)``
    (:func:`_axis0_layout`; one row per level when the flat sample count does
    not match the grid). ``dyad_t``, ``tile`` and ``het`` all size their
    partitions by it. :func:`_slab_edges` cuts at most this many slabs at
    least that many levels each, so every slab keeps ``min_rows``.

    Counting the ROW total instead (``n_samples // min_rows``, ``tile``'s and
    ``het``'s old rule) agrees wherever ``min_rows`` is a whole number of
    levels, but on a coarse axis 0 the level rounding cut slabs below the
    floor -- (13, 3, 3) at width 6 gave four slabs of 27, 27, 27 and 36 rows
    under a 28-row floor -- and made :func:`row_floor` non-monotone (that grid
    failed where (12, 3, 3) passed).
    """
    n_0, stride = _axis0_layout(n_samples, grid_shape)
    return n_0 // max(1, -(-int(min_rows) // stride))


#: Above this many doubles the one-row-per-level outer-product stack of
#: :func:`_level_gram_blocks` is not built and the slab loop is used instead.
_MAX_OUTER_PRODUCT_DOUBLES = 4_000_000
#: The most gathered slab-Gram entries :func:`dyad_t_subset_scores` solves in
#: one batch; more subsets of one size are solved in chunks.
_DYAD_SUBSET_CHUNK_DOUBLES = 4_000_000


#: The 1-D outer-product Grams kept for the paired version (``_level_gram_blocks``):
#: the latest few, from this many rows up (below, the comparisons would cost
#: about what they save).
_PAIRED_GRAMS = []
_PAIRED_GRAMS_KEPT = 2
_PAIRED_MIN_ROWS = 10_000


def _same_bits(a, b) -> bool:
    """``a`` and ``b`` hold the same doubles bit for bit: equal values with equal
    signs, so a -0.0 against a 0.0 differs (a NaN never matches)."""
    return a.shape == b.shape and np.array_equal(a, b) and np.array_equal(np.signbit(a), np.signbit(b))


def _level_gram_blocks(X, y, sample_weights, grid_shape, edges, append_constant: bool = True):
    """:func:`_edge_gram_blocks` for slab edges that are whole axis-0 levels,
    in one batched multiply: the per-level augmented Grams (one per step of
    the first grid axis, all the same size) summed over each slab with
    ``reduceat``. One row per level (a 1-D grid, or no grid) sums the per-row
    outer products the same way, while that stack stays under
    ``_MAX_OUTER_PRODUCT_DOUBLES`` -- a rule on the augmented width ``p + 1``
    whatever ``append_constant`` says, as the two paths round differently;
    larger inputs keep the slab loop.

    ``append_constant=False`` leaves the folded constant out: the Grams are
    ``p`` wide, each entry the same bytes as in the augmented Gram (the same
    products, summed in the same order). On the 1-D outer path from
    ``_PAIRED_MIN_ROWS`` rows, a design without a ones column (plain mode's S)
    is still built with the folded constant and kept, and a design equal to
    it bit for bit, ones column and all (its S+{1}), with the same target,
    weights and slabs, is served that Gram instead of building its own.

    The row-wise path sums the same products in a different order than the
    slab loop, so it agrees with it to round-off, not bit for bit. On the
    exhaustive-pool study (Oct 2026) it gave the same dominance verdicts as
    the loop everywhere, and its errors against an exact SVD reference were
    no larger (median 3.1e-8 vs 3.5e-8 on LV, 2.0e-9 vs 2.2e-9 on Lorenz)."""
    n_samples, p = X.shape
    width = p + 1 if append_constant else p
    n_0, stride = _axis0_layout(n_samples, grid_shape)
    if stride == 1:
        if n_samples * (p + 1) ** 2 > _MAX_OUTER_PRODUCT_DOUBLES:
            G, Gy = _edge_gram_blocks(X, y, sample_weights, edges)
            return (G, Gy) if append_constant else (G[:, :p, :p], Gy[:, :p])
        sw = (np.asarray(sample_weights, dtype=float).reshape(-1)
              if sample_weights is not None else np.ones(n_samples))
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        pos = np.asarray(edges[:-1])
        share = not append_constant and n_samples >= _PAIRED_MIN_ROWS and p > 0
        if share and np.all(X[:, -1] == 1.0):
            for A, y_kept, sw_kept, pos_kept, G, Gy in _PAIRED_GRAMS:
                if (A.shape == X.shape and np.array_equal(pos_kept, pos) and _same_bits(y_kept, y)
                        and _same_bits(sw_kept, sw) and _same_bits(A, X)):
                    return G, Gy
            share = False
        A = np.hstack([X, np.ones((n_samples, 1))]) if append_constant or share else X
        wA = sw[:, None] * A
        G = np.add.reduceat(A[:, :, None] * wA[:, None, :], pos, axis=0)
        Gy = np.add.reduceat(wA * y[:, None], pos, axis=0)
        if share:
            G.setflags(write=False)
            Gy.setflags(write=False)
            _PAIRED_GRAMS.insert(0, (A, y.copy(), sw.copy(), pos.copy(), G, Gy))
            del _PAIRED_GRAMS[_PAIRED_GRAMS_KEPT:]
            return G[:, :p, :p], Gy[:, :p]
        return G, Gy
    sw = (np.asarray(sample_weights, dtype=float).reshape(-1)
          if sample_weights is not None else np.ones(n_samples))
    A = np.hstack([X, np.ones((n_samples, 1))]).reshape(n_0, stride, p + 1)
    wA = sw.reshape(n_0, stride, 1) * A
    G_levels = np.matmul(A.transpose(0, 2, 1), wA)
    Gy_levels = np.einsum('lsk,ls->lk', wA, np.asarray(y, dtype=float).reshape(n_0, stride))
    pos = edges[:-1] // stride
    G, Gy = np.add.reduceat(G_levels, pos, axis=0), np.add.reduceat(Gy_levels, pos, axis=0)
    return (G, Gy) if width == p + 1 else (G[:, :p, :p], Gy[:, :p])


_DYAD_RCOND = 1e-12               # dyad_t: min-norm cut of the unit-diagonal slab Gram
#: dyad_t's batched slab solve. Private: the A/B switch of the benchmark
#: harness, not a user setting. ``'csne'`` is the default (Oct 2026): the
#: Gram solves move 17-25% of lorenz's subset scores by more than 1e-3 under
#: a 1e-14 perturbation of the data, csne none, at less than eigh's cost.
#:   ``'eigh'``      every slab by eigendecomposition of its Gram;
#:   ``'cholesky'``  :func:`_solve_min_norm_cholesky`: the same solution, to
#:                   round-off, where a bound certifies no eigenvalue is cut;
#:                   eigh elsewhere;
#:   ``'hybrid'``    :func:`_solve_min_norm_hybrid`: the Cholesky solution
#:                   where the bound certifies a well-conditioned slab
#:                   (``_HYBRID_RATIO``), elsewhere the same rule evaluated
#:                   from the slab's design (:func:`_solve_min_norm_design`),
#:                   which does not square the slab's conditioning;
#:   ``'csne'``      :func:`_solve_min_norm_csne`: the hybrid's accuracy at the
#:                   Gram solve's cost -- the Cholesky solution down to
#:                   ``_CSNE_RATIO``, corrected by one refinement step from the
#:                   slab design below ``_HYBRID_RATIO``; the design solve only
#:                   under ``_CSNE_RATIO``.
_DYAD_SOLVE = 'csne'
#: The solves that read the slab designs (:func:`dyad_t_design_factors`).
_DESIGN_SOLVES = ('hybrid', 'csne')
#: The certificate's margin over the cut: a slab takes the Cholesky solve only
#: when its smallest eigenvalue is provably above ``_CERT_MARGIN`` times the
#: cut, far from where round-off in the bound could matter.
_CERT_MARGIN = 100.0
#: Under ``'hybrid'``: a slab keeps the Gram (Cholesky) solve only when its
#: unit-diagonal Gram provably has ``lambda_min / lambda_max >=
#: _HYBRID_RATIO`` -- there the Gram's round-off (an absolute ~1e-15) costs at
#: most ~1e-11 relative -- and is solved from its design otherwise.
_HYBRID_RATIO = 1e-4
#: Under ``'csne'``: the certified ratio down to which a slab keeps the
#: Cholesky solution (refined once below ``_HYBRID_RATIO``); slabs it does not
#: certify are solved from their design. A hundred times the cut
#: (``_DYAD_RCOND``), so a certified slab has nothing to cut, and far enough
#: above round-off that one refinement step reaches the design solve's
#: accuracy (the Cholesky solution is good to ~eps / ratio <= 1e-6 there).
_CSNE_RATIO = 1e-10


def _solve_min_norm(G, Gy, design=None):
    """Min-norm least squares of a stack of Gram systems: see
    :func:`_solve_min_norm_eigh`; under ``_DYAD_SOLVE = 'cholesky'`` the
    certified slabs are solved by :func:`_solve_min_norm_cholesky`, under
    ``'hybrid'`` by :func:`_solve_min_norm_hybrid` and under ``'csne'`` by
    :func:`_solve_min_norm_csne`, which need ``design``: a callable giving,
    for stack rows ``rows``, the slab designs' R factors restricted to the
    stack's columns and to its target (:func:`dyad_t_design_factors`)."""
    if _DYAD_SOLVE == 'cholesky':
        return _solve_min_norm_cholesky(G, Gy)
    if _DYAD_SOLVE in _DESIGN_SOLVES:
        if design is None:
            raise ValueError(f"_DYAD_SOLVE={_DYAD_SOLVE!r} needs the slab designs (dyad_t_design_factors).")
        if _DYAD_SOLVE == 'csne':
            return _solve_min_norm_csne(G, Gy, design)
        return _solve_min_norm_hybrid(G, Gy, design)
    return _solve_min_norm_eigh(G, Gy)


def _solve_min_norm_cholesky(G, Gy):
    """:func:`_solve_min_norm_eigh`'s solution, slab by slab, from a
    vectorised Cholesky of the unit-diagonal Gram wherever no eigenvalue can
    be cut.

    Each slab Gram is equilibrated as in the eigh solve (``C = G / d d^T``).
    The Cholesky factor ``L`` and its inverse are formed elementwise across
    the stack (no per-matrix LAPACK call), and ``trace(C^-1) = ||L^-1||_F^2``
    bounds the smallest eigenvalue from below: ``lambda_min >= 1 /
    trace(C^-1)``, while ``lambda_max <= trace(C) <= k``. A slab whose bound
    clears ``_CERT_MARGIN * _DYAD_RCOND * k`` has nothing for the eigh solve to
    cut, so its min-norm solution is ``C^-1 g`` and the Cholesky solve gives
    it (to round-off). Every other slab -- rank-deficient, near the cut, a
    zero column, a non-finite entry -- goes to the eigh solve unchanged."""
    G = np.asarray(G, dtype=float)
    Gy = np.asarray(Gy, dtype=float)
    theta, certified = _cholesky_certified(G, Gy, _CERT_MARGIN * _DYAD_RCOND)
    if not np.all(certified):
        rest = ~certified
        theta[rest] = _solve_min_norm_eigh(G[rest], Gy[rest], total=len(G))
    return theta


def _solve_min_norm_hybrid(G, Gy, design):
    """The min-norm rule of :func:`_solve_min_norm_eigh`, evaluated
    accurately: the Cholesky solve of the Gram where the bound of
    :func:`_cholesky_certified` proves ``lambda_min / lambda_max >=
    _HYBRID_RATIO`` (the Gram's round-off is negligible there), and
    :func:`_solve_min_norm_design` on every other slab -- near-singular,
    rank-deficient, a zero column, a non-finite entry. ``design(rows)`` gives
    those slabs' ``(R[:, S], R[:, y])``."""
    G = np.asarray(G, dtype=float)
    Gy = np.asarray(Gy, dtype=float)
    theta, certified = _cholesky_certified(G, Gy, _HYBRID_RATIO)
    if not np.all(certified):
        rest = np.flatnonzero(~certified)
        M, rhs = design(rest)
        theta[rest] = _solve_min_norm_design(M, rhs, total=len(G))
    return theta


def _solve_min_norm_csne(G, Gy, design):
    """The hybrid's rule and accuracy (:func:`_solve_min_norm_hybrid`) at
    about the Gram solve's cost, by corrected semi-normal equations.

    The bound of :func:`_cholesky_certified` sorts the slabs into three bands
    of the unit-diagonal Gram's ``lambda_min / lambda_max``:

    * ``>= _HYBRID_RATIO``: the Cholesky solution, as in the hybrid;
    * ``>= _CSNE_RATIO``: the Cholesky solution plus ONE refinement step whose
      residual comes from the slab design, not the Gram --
      ``r = R[:, y] - R[:, S] theta``, ``theta += C^-1 (R[:, S]^T r)``
      (equilibrated) -- which shrinks the Gram's round-off error (~eps /
      ratio, at most ~1e-6 relative here) to the design solve's level;
    * below, or uncertified (rank-deficient, a zero column, a non-finite
      entry): :func:`_solve_min_norm_design`, as in the hybrid.

    Every certified slab is far above the cut, so in exact arithmetic this
    is the eigh solve's min-norm rule. ``design(rows)`` gives the refined and
    the uncertified slabs' ``(R[:, S], R[:, y])`` in one call.

    Taking the step's residual from the slab's raw weighted rows instead
    (``r = w y - w X_S theta``, the textbook form, no factor for a refined
    slab) was measured slower (Oct 2026: csne time 2.3 -> 3.0 s on burgers,
    2.2 -> 3.1 s on pde_compound): with :class:`_SlabFactors` only the read
    slabs are factored, and their QR costs less than gathering their rows."""
    G = np.asarray(G, dtype=float)
    Gy = np.asarray(Gy, dtype=float)
    k = G.shape[1]
    Linv, d, ok, trace_inv = _cholesky_factors(G)
    with np.errstate(all='ignore'):        # uncertified slabs are recomputed below
        certified = ok & (trace_inv * (_CSNE_RATIO * k) < 1.0)
        plain = certified & (trace_inv * (_HYBRID_RATIO * k) < 1.0)
        theta = (_apply_unit_inverse(Linv, (Gy / d).T) / d.T).T
    rows = np.flatnonzero(~plain)
    if rows.size:
        M, rhs = design(rows)
        refine = certified[rows]
        if refine.any():
            at, Ms = rows[refine], M[refine]
            r = rhs[refine] - np.einsum('mqk,mk->mq', Ms, theta[at])
            step = _apply_unit_inverse(Linv[:, :, at], (np.einsum('mqk,mq->mk', Ms, r) / d[at]).T)
            theta[at] += (step / d[at].T).T
        if not refine.all():
            theta[rows[~refine]] = _solve_min_norm_design(M[~refine], rhs[~refine], total=len(G))
    return theta


def _solve_min_norm_design(M, rhs, total=None):
    """:func:`_solve_min_norm_eigh`'s rule on the slab DESIGN instead of its
    Gram: for each stack row, the min-norm least-squares solution of ``M x =
    rhs`` (``M`` the ``(q, k)`` R-factor columns of the slab's weighted
    design, ``rhs`` its target column; :func:`dyad_t_design_factors`),
    columns scaled to unit norm, singular values with ``sigma^2 <=
    _DYAD_RCOND * sigma_max^2`` cut -- in exact arithmetic exactly the eigh
    solve's ``lambda <= _DYAD_RCOND * lambda_max`` on the unit-diagonal Gram.
    The singular values carry round-off relative to ``sigma_max`` (not to
    ``lambda_max``), so the cut sits far above it. Non-finite input fails
    loudly (the SVD does not converge)."""
    M = np.asarray(M, dtype=float)
    d = np.linalg.norm(M, axis=1)
    d = np.where(d > 0, d, 1.0)
    U, s, Vt = np.linalg.svd(M / d[:, None, :], full_matrices=False)
    keep = s * s > _DYAD_RCOND * (s[:, :1] * s[:, :1])
    n_deficient = int(np.count_nonzero(~keep.all(axis=1)))
    if n_deficient:
        _loop_stats.record('dyad_t.rank_deficient_slabs', n_deficient,
                           M.shape[0] if total is None else total)
    inv = np.where(keep, 1.0 / np.where(keep, s, 1.0), 0.0)
    proj = np.einsum('mji,mj->mi', U, rhs)
    return np.einsum('mji,mj->mi', Vt, inv * proj) / d


def dyad_t_design_factors(B, sample_weights, parts):
    """The R factor of every slab's weighted design rows ``sqrt(w) * B``, for
    every partition in ``parts`` in their order (the slab order of
    :func:`dyad_t_blocks`' stacks): ``(n_slabs, q, q)``, ``q`` B's columns.
    Since ``Q`` is orthonormal, ``||B_S x - b_y|| = ||R[:, S] x - R[:, y]||``
    for any columns ``S`` and ``y`` of ``B``, so one factor serves every
    subset and target, as one Gram does."""
    B = np.asarray(B, dtype=float)
    n, q = B.shape
    sw = np.ones(n) if sample_weights is None else np.asarray(sample_weights, dtype=float).reshape(-1)
    W = np.sqrt(sw)[:, None] * B
    out = []
    for edges in parts:
        sizes = np.diff(edges)
        rows = max(int(sizes.max()), q)
        P = np.zeros((len(sizes), rows, q))
        for b in range(len(sizes)):
            P[b, :sizes[b]] = W[edges[b]:edges[b + 1]]
        out.append(np.linalg.qr(P, mode='r')[:, :q, :])
    return np.concatenate(out)


class _SlabFactors(object):
    """:func:`dyad_t_design_factors`, slab by slab on first use: ``factors(slabs)``
    gives the R factors of those slabs (stack indices, repeats allowed) and
    keeps them. The design solves read a slab's design only where the Cholesky
    solve is not certified, and most slabs are certified, so the eager build
    factored mostly slabs nobody read. Each slab's factor is the eager one bit
    for bit: the same rows, padded to the same height (its partition's tallest
    slab, at least ``q``), through the same batched LAPACK QR, whose result
    for one matrix does not depend on the others in the batch. ``B`` is the
    design or a callable building it, so a call that reads no slab does not
    assemble it either."""

    def __init__(self, B, sample_weights, parts):
        self._B = B
        self._sample_weights = sample_weights
        self._parts = parts
        self._W = None
        self._done = np.zeros(sum(edges.size - 1 for edges in parts), dtype=bool)

    def _weighted_design(self):
        B = np.asarray(self._B() if callable(self._B) else self._B, dtype=float)
        sw = (np.ones(B.shape[0]) if self._sample_weights is None
              else np.asarray(self._sample_weights, dtype=float).reshape(-1))
        self._W = np.sqrt(sw)[:, None] * B
        q = B.shape[1]
        self._lo = np.concatenate([edges[:-1] for edges in self._parts])
        self._hi = np.concatenate([edges[1:] for edges in self._parts])
        self._height = np.concatenate([np.full(edges.size - 1, max(int(np.diff(edges).max()), q))
                                       for edges in self._parts])
        self._R = np.empty((self._done.size, q, q))

    def __call__(self, slabs):
        slabs = np.asarray(slabs, dtype=int)
        todo = np.unique(slabs[~self._done[slabs]])
        if todo.size:
            if self._W is None:
                self._weighted_design()
            q = self._W.shape[1]
            levels = np.arange(int(self._height[todo].max()))
            for height in np.unique(self._height[todo]):
                group = todo[self._height[todo] == height]
                # Every slab's rows in one gather, padded with zero rows to the
                # height (the rows past a slab's end are clipped, then zeroed):
                # the stack dyad_t_design_factors fills row by row.
                rows = levels[:height]
                P = np.take(self._W, self._lo[group][:, None] + rows[None, :], axis=0, mode='clip')
                P[rows[None, :] >= (self._hi[group] - self._lo[group])[:, None]] = 0.0
                self._R[group] = np.linalg.qr(P, mode='r')[:, :q, :]
            self._done[todo] = True
        return self._R[slabs]


def _cholesky_certified(G, Gy, bound):
    """The vectorised Cholesky solve of the unit-diagonal Gram stack, and
    which slabs it is certified for: ``trace(C^-1) = ||L^-1||_F^2`` bounds
    ``lambda_min >= 1 / trace(C^-1)``, ``lambda_max <= trace(C) <= k``, so a
    slab with ``trace(C^-1) * bound * k < 1`` has ``lambda_min / lambda_max >
    bound``. Uncertified slabs' solutions are not meaningful."""
    k = G.shape[1]
    Linv, d, ok, trace_inv = _cholesky_factors(G)
    with np.errstate(all='ignore'):        # uncertified slabs are recomputed by the caller
        certified = ok & (trace_inv * (bound * k) < 1.0)
        theta = (_apply_unit_inverse(Linv, (Gy / d).T) / d.T).T
    return theta, certified


def _apply_unit_inverse(Linv, g):
    """``C^-1 g = L^-T L^-1 g`` for every slab of the stack: ``Linv`` the
    ``(k, k, m)`` inverse Cholesky factors of :func:`_cholesky_factors`, ``g``
    ``(k, m)``."""
    g = np.ascontiguousarray(g)
    with np.errstate(all='ignore'):
        w = np.einsum('ijm,jm->im', Linv, g)
        return np.einsum('jim,jm->im', Linv, w)


def _cholesky_factors(G):
    """The vectorised Cholesky factorisation of the unit-diagonal Gram stack
    ``C = G / d d^T`` (``d`` the square roots of the diagonal): the inverse
    factors ``L^-1`` ``(k, k, m)``, formed elementwise across the stack (no
    per-matrix LAPACK call), ``d`` ``(m, k)``, whether every pivot stayed
    positive with a finite ``trace(C^-1) = ||L^-1||_F^2``, and that trace --
    what :func:`_cholesky_certified`'s bound reads."""
    m, k, _ = G.shape
    dg = np.diagonal(G, axis1=-2, axis2=-1)
    d = np.sqrt(np.where(dg > 0, dg, 1.0))
    with np.errstate(all='ignore'):        # failed slabs are flagged, never used
        C = np.ascontiguousarray((G / (d[:, :, None] * d[:, None, :])).transpose(1, 2, 0))   # (k, k, m)
        L = np.zeros((k, k, m))
        ok = np.ones(m, dtype=bool)
        for j in range(k):
            s = C[j, j] - np.sum(L[j, :j] * L[j, :j], axis=0)
            ok &= s > 0
            L[j, j] = np.sqrt(np.where(s > 0, s, 1.0))
            for i in range(j + 1, k):
                L[i, j] = (C[i, j] - np.sum(L[i, :j] * L[j, :j], axis=0)) / L[j, j]
        Linv = np.zeros((k, k, m))
        for j in range(k):
            Linv[j, j] = 1.0 / L[j, j]
            for i in range(j + 1, k):
                Linv[i, j] = -np.sum(L[i, j:i] * Linv[j:i, j], axis=0) / L[i, i]
        trace_inv = np.sum(Linv * Linv, axis=(0, 1))
    return Linv, d, ok & np.isfinite(trace_inv), trace_inv


def _solve_min_norm_eigh(G, Gy, total=None):
    """Min-norm least squares of a STACK of Gram systems ``(m, k, k)``,
    ``(m, k)`` in one batched ``eigh``: each Gram made unit-diagonal first (a
    term's units cannot steer the solve), eigenvalues ``<= _DYAD_RCOND *
    lambda_max`` cut -- the solution ``lstsq(G, Gy, rcond=_DYAD_RCOND)`` gives
    on a symmetric PSD Gram, at one call instead of one per slab.

    A thin slab can make the design EXACTLY rank-deficient -- on a one-level
    time slab ``t * f(x)`` is ``f(x)`` times a constant -- and
    ``np.linalg.solve`` does not raise there: round-off leaves a tiny pivot
    and it returns huge, meaningless coefficients. The min-norm solution
    splits the unidentified direction instead; a full-rank slab gets the
    plain solve (to round-off). A call with rank-deficient slabs is recorded
    in ``_loop_stats`` (``'dyad_t.rank_deficient_slabs'``: their count out of
    the slabs solved -- ``total`` of them when this solves the remainder of a
    larger stack), as ``survival`` records its lstsq fallbacks."""
    dg = np.diagonal(G, axis1=-2, axis2=-1)
    d = np.sqrt(np.where(dg > 0, dg, 1.0))
    lam, V = np.linalg.eigh(G / (d[:, :, None] * d[:, None, :]))
    keep = np.abs(lam) > _DYAD_RCOND * np.abs(lam).max(axis=1, keepdims=True)
    n_deficient = int(np.count_nonzero(~keep.all(axis=1)))
    if n_deficient:
        _loop_stats.record('dyad_t.rank_deficient_slabs', n_deficient,
                           G.shape[0] if total is None else total)
    inv = np.where(keep, 1.0 / np.where(keep, lam, 1.0), 0.0)
    proj = np.einsum('mji,mj->mi', V, Gy / d)
    return np.einsum('mij,mj->mi', V, inv * proj) / d


def _median_axis0(a):
    """``np.median(a, axis=0)`` of a small 2-D stack, by one sort.

    The same values: an odd count takes the middle element and an even count
    halves the sum of the two middle ones, which is what np.median's
    partition-then-mean computes. It skips np.median's per-call overhead,
    which dominated dyad_t's ten medians per score on these few-row stacks.
    Non-finite input goes to np.median itself, for its NaN propagation.
    """
    if not np.all(np.isfinite(a)):
        return np.median(a, axis=0)
    s = np.sort(a, axis=0)
    m = s.shape[0] // 2
    if s.shape[0] % 2:
        return s[m]
    return (s[m - 1] + s[m]) / 2.0


def dyad_t_scores(features, target, sample_weights, grid_shape,
                  fit_intercept: bool = True):
    """Per-term instability: ``tile``'s statistic averaged over EVERY dyadic
    partition of storage axis 0.

    For each partition ``s`` (2, 4, 8, ... equal time slabs; see
    :func:`_dyadic_slab_edges`) the fixed structure is refitted on every slab
    and the term scores the robust between-slab dispersion
    ``MAD_b(c_bj) / |median_b c_bj|``; the per-term score is the mean of that
    over the partitions. A constant-coefficient term agrees across slabs at
    every scale -> ~0; a drifting one disagrees at whichever scales resolve
    the drift, so no single slab count has to be guessed (``tile``'s
    ``n_tiles``). The slab floor is ``tile``'s: ``SLAB_ROWS_PER_COLUMN`` samples per unknown of
    the SEARCH's widest equation (:func:`_reference_width`), so every
    candidate in one search is scored on the same partitions.

    Slab coefficients come from the per-slab Gram stacks
    (:func:`block_gram_partition` convention: weighted by ``sample_weights``,
    intercept folded block-wise; built in one pass over the data,
    :func:`_level_gram_blocks`), every slab of every partition solved in one
    batched equilibrated min-norm solve (:func:`_solve_min_norm`). A column whose
    slab coefficients are all zero scores 0.0 (``tile``'s dead-column
    convention); a zero median under nonzero spread scores the cap
    (``np.finfo(float).max``), as does any partition doing so for that term.
    Parameter-free: the partitions follow from the grid and the floor alone.
    Like ``tile`` it sees axis >= 1 only through the axis-0 drift a spatial
    modulation induces in the per-slab fit.
    """
    X = np.asarray(features, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    n_samples, p = X.shape
    min_rows = _block_rows(_reference_width(p))
    parts = _dyadic_slab_edges(n_samples, grid_shape, min_rows)
    fine = parts[-1]
    # As wide as the fit: the folded constant only with ``fit_intercept``.
    G_fine, Gy_fine = _level_gram_blocks(X, target, sample_weights, grid_shape, fine,
                                         append_constant=fit_intercept)

    G_all, Gy_all = [], []
    for edges in parts:
        pos = np.searchsorted(fine, edges[:-1])
        G_all.append(np.add.reduceat(G_fine, pos, axis=0))
        Gy_all.append(np.add.reduceat(Gy_fine, pos, axis=0))
    design = None
    if _DYAD_SOLVE in _DESIGN_SOLVES:
        factors = _SlabFactors(lambda: np.hstack([X, np.ones((n_samples, 1)),
                                                  np.asarray(target, dtype=float).reshape(-1, 1)]),
                               sample_weights, parts)
        cols = np.arange(p + 1 if fit_intercept else p)

        def design(rows):
            R = factors(rows)
            return R[:, :, cols], R[:, :, p + 1]
    theta = _solve_min_norm(np.concatenate(G_all), np.concatenate(Gy_all), design)[:, :p]

    total = np.zeros(p)
    start = 0
    for edges in parts:
        coefs = theta[start:start + edges.size - 1]
        start += edges.size - 1
        med = _median_axis0(coefs)
        mad = _median_axis0(np.abs(coefs - med[None, :]))
        with np.errstate(divide='ignore', invalid='ignore'):
            rel = mad / np.abs(med)
        total += np.where(np.isnan(rel), 0.0, rel)
    with np.errstate(over='ignore'):
        return np.nan_to_num(total / len(parts))


def _median_axis1(a):
    """The median along axis 1 of a ``(m, n, k)`` stack: :func:`_median_axis0`
    for ``m`` stacks at once (same values; non-finite input goes to
    ``np.median``)."""
    if not np.all(np.isfinite(a)):
        return np.median(a, axis=1)
    s = np.sort(a, axis=1)
    m = s.shape[1] // 2
    if s.shape[1] % 2:
        return s[:, m]
    return (s[:, m - 1] + s[:, m]) / 2.0


def dyad_t_subset_scores(features, target, sample_weights, grid_shape, subsets, per_term=False):
    """``sum(dyad_t_scores(features[:, S], target, sample_weights, grid_shape,
    fit_intercept=False))`` for MANY column subsets ``S`` at once: the slabs
    of storage axis 0, one axis of :func:`dyad_subset_scores`.

    The same statistic, built once: a subset's per-slab Gram is a sub-block of
    the Gram of all the columns, so the slab Grams of every partition are
    formed once and sliced per subset (equal to the per-subset build to
    round-off), and all subsets of one size are solved in one batched
    equilibrated min-norm ``eigh`` with vectorised medians. ``features`` must
    already carry any constant column the subsets select (the keep-rule's
    explicit ones column); ``subsets`` is a sequence of column-index lists.
    Returns one summed score per subset (``nan_to_num`` per term, as the
    table does), or under ``per_term`` a list of each subset's per-term
    scores (``None`` for the empty subset).
    """
    X = np.asarray(features, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    y = np.asarray(target, dtype=float).reshape(-1)
    n_samples = X.shape[0]
    out = [None] * len(subsets) if per_term else np.zeros(len(subsets))
    by_size = {}
    for k, S in enumerate(subsets):
        by_size.setdefault(len(S), []).append(k)
    blocks = {}
    for size, idxs in sorted(by_size.items()):
        if size == 0:
            if not per_term:
                out[idxs] = np.inf
            continue
        # dyad_t_scores reads its partition floor off the candidate's width,
        # through the search-wide reference width.
        min_rows = dyad_t_min_rows(size)
        if min_rows not in blocks:
            parts, G_all, Gy_all = dyad_t_blocks(X, y, sample_weights, grid_shape, n_samples, min_rows)
            design = None
            if _DYAD_SOLVE in _DESIGN_SOLVES:
                design = (_SlabFactors(lambda: np.hstack([X, y[:, None]]), sample_weights, parts), X.shape[1])
            blocks[min_rows] = (parts, G_all, Gy_all, design)
        parts, G_all, Gy_all, design = blocks[min_rows]
        scores = dyad_t_scores_from_blocks(parts, G_all, Gy_all, [subsets[k] for k in idxs], design,
                                           per_term=per_term)
        if per_term:
            for k, row in zip(idxs, scores):
                out[k] = row
        else:
            out[idxs] = scores
    return out


def dyad_t_min_rows(size: int) -> int:
    """:func:`dyad_t_subset_scores`' slab floor for a subset of ``size``
    columns: ``tile``'s ``SLAB_ROWS_PER_COLUMN`` rows per unknown of the
    search-wide reference width."""
    return _block_rows(_reference_width(size))


def dyad_t_blocks(X, y, sample_weights, grid_shape, n_samples, min_rows):
    """The slab Grams of every dyadic partition of ``X`` (``(n_slabs, p + 1,
    p + 1)``, the appended constant last) and their cross moments with ``y``,
    with the partitions: what :func:`dyad_t_scores_from_blocks` slices per
    subset. A subset of ``X``'s columns is scored from these sub-blocks."""
    parts = _dyadic_slab_edges(n_samples, grid_shape, min_rows)
    fine = parts[-1]
    G_fine, Gy_fine = _level_gram_blocks(X, y, sample_weights, grid_shape, fine)
    G_all, Gy_all = [], []
    for edges in parts:
        pos = np.searchsorted(fine, edges[:-1])
        G_all.append(np.add.reduceat(G_fine, pos, axis=0))
        Gy_all.append(np.add.reduceat(Gy_fine, pos, axis=0))
    return parts, np.concatenate(G_all), np.concatenate(Gy_all)


def dyad_t_scores_from_blocks(parts, G_all, Gy_all, subsets, design=None, per_term=False):
    """The summed dyad_t score of each subset (column-index lists of ONE
    size) from :func:`dyad_t_blocks`' slab Grams: every slab of every
    partition solved by :func:`_solve_min_norm`, the between-slab medians
    taken per subset. ``design`` -- ``(R, y column)``, R the stack of
    :func:`dyad_t_design_factors` or a :class:`_SlabFactors`, the subsets
    indexing R's columns -- is what the solves of ``_DESIGN_SOLVES`` read
    slabs from. ``per_term``: the ``(n_subsets, size)`` per-term scores
    instead of their sums."""
    size = len(subsets[0])
    n_slabs = G_all.shape[0]
    out = np.zeros((len(subsets), size)) if per_term else np.zeros(len(subsets))
    # The subsets in chunks of at most _DYAD_SUBSET_CHUNK_DOUBLES gathered Gram
    # entries: every slab is solved and every subset's medians are taken on
    # their own, so the scores are the one-batch scores, but memory no longer
    # grows with the number of subsets.
    step = max(1, _DYAD_SUBSET_CHUNK_DOUBLES // (n_slabs * size * size))
    for lo in range(0, len(subsets), step):
        cols = np.array(subsets[lo:lo + step], dtype=int)                 # (m, size)
        G = G_all[:, cols[:, :, None], cols[:, None, :]].transpose(1, 0, 2, 3)
        Gy = Gy_all[:, cols].transpose(1, 0, 2)
        gather = None
        if design is not None:
            R, ycol = design

            def gather(rows, cols=cols, R=R, ycol=ycol):
                sub, slab = np.divmod(rows, n_slabs)                   # stack row = subset * n_slabs + slab
                F = R(slab) if callable(R) else R[slab]
                M = F[np.arange(slab.size)[:, None, None], np.arange(F.shape[1])[None, :, None],
                      cols[sub][:, None, :]]
                return M, F[:, :, ycol]
        theta = _solve_min_norm(G.reshape(-1, size, size), Gy.reshape(-1, size),
                                gather).reshape(len(cols), n_slabs, size)
        total = np.zeros((len(cols), size))
        start = 0
        for edges in parts:
            coefs = theta[:, start:start + edges.size - 1, :]
            start += edges.size - 1
            med = _median_axis1(coefs)
            mad = _median_axis1(np.abs(coefs - med[:, None, :]))
            with np.errstate(divide='ignore', invalid='ignore'):
                rel = mad / np.abs(med)
            total += np.where(np.isnan(rel), 0.0, rel)
        with np.errstate(over='ignore'):
            terms = np.nan_to_num(total / len(parts))
        out[lo:lo + len(cols)] = terms if per_term else np.sum(np.nan_to_num(terms), axis=1)
    return out


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
    # Blocks are cut at whole levels of axis 0, so their count is too
    # (:func:`_level_slab_capacity`): every block keeps SLAB_ROWS_PER_COLUMN*(w+1) samples.
    max_blocks = _level_slab_capacity(n_samples, grid_shape,
                                      _block_rows(_reference_width(p)))
    if max_blocks < 3:
        n_0, stride = _axis0_layout(n_samples, grid_shape)
        raise ValueError(
            f'heterogeneity_scores needs >= 3 blocks of >= {_block_rows(0)}*(w+1) samples, '
            'cut at whole levels of storage axis 0, to calibrate; got '
            f'n_samples={n_samples} ({n_0} levels of {stride}) with p={p} and '
            f'reference width w={_reference_width(p)} (equation_terms_max_number, or '
            "system_rps_params['selected_terms_max_number'] when that is smaller)")
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
def _axis_layouts(n_samples: int, grid_shape) -> list:
    """``[(axis, row order, grid shape)]``: the rows reordered so each grid axis
    with two or more levels becomes storage axis 0 (C-ordered flattening). One
    entry, the rows as they are, when the grid has one such axis or the rows are
    not the flattening of ``grid_shape`` (a strided row budget)."""
    shape = tuple(int(s) for s in (() if grid_shape is None else grid_shape))
    axes = [a for a, levels in enumerate(shape) if levels >= 2]
    if len(axes) < 2 or int(np.prod(shape)) != int(n_samples):
        return [(0, None, grid_shape)]
    index = np.arange(int(n_samples)).reshape(shape)
    layouts = []
    for a in axes:
        moved = (shape[a],) + tuple(s for k, s in enumerate(shape) if k != a)
        layouts.append((a, None if a == 0 else np.moveaxis(index, a, 0).reshape(-1), moved))
    return layouts


def _combine_axes(scores):
    """The per-axis per-term scores ``(n_axes, ...)`` as one: their mean over
    the axes, the cap (``np.finfo(float).max``) for a term capped on any axis."""
    cap = np.finfo(float).max
    capped = np.any(scores >= cap, axis=0)
    return np.where(capped, cap, np.sum(np.where(capped, 0.0, scores) / scores.shape[0], axis=0))


def dyad_scores(features, target, sample_weights, grid_shape, fit_intercept: bool = True):
    """Per-term instability: :func:`dyad_t_scores` (the dyadic slabs of storage
    axis 0) along EVERY grid axis in turn (the rows reordered so the axis is
    storage axis 0, :func:`_axis_layouts`), averaged over the axes; a capped
    score on any axis caps the term.

    Slabs along one axis each span the whole of every other axis, so a
    coefficient that drifts along another axis is averaged out inside every
    slab. A finite-difference error term is the case in point: one term
    standing in for a mix of error terms, whose best coefficient changes with x
    as the field's shape does. On pde_compound under second-order differences
    ``truth + u_x u_xxx`` scored 5.6e-3 on time slabs against the truth's
    7.3e-3, and 0.39 against 1.7e-3 on x slabs. Parameter-free: the partitions
    follow from the grid and the slab floor alone. One axis, or rows that are
    not the grid's flattening, are cut along storage axis 0 alone."""
    X = np.asarray(features, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    layouts = _axis_layouts(X.shape[0], grid_shape)
    if len(layouts) == 1:
        return dyad_t_scores(X, target, sample_weights, grid_shape, fit_intercept=fit_intercept)
    y = np.asarray(target, dtype=float).reshape(-1)
    w = np.asarray(sample_weights, dtype=float).reshape(-1)
    scores = []
    for _axis, order, shape in layouts:
        if order is None:
            scores.append(np.asarray(dyad_t_scores(X, y, w, shape, fit_intercept=fit_intercept), dtype=float))
        else:
            scores.append(np.asarray(dyad_t_scores(X[order], y[order], w[order], shape,
                                                   fit_intercept=fit_intercept), dtype=float))
    return _combine_axes(np.stack(scores))


def dyad_subset_scores(features, target, sample_weights, grid_shape, subsets):
    """``sum(dyad_scores(features[:, S], target, sample_weights, grid_shape,
    fit_intercept=False))`` for MANY column subsets ``S`` at once -- the
    per-subset instability ``subset_selection.instability_table`` tabulates,
    for the exhaustive subset search of ``system_rps_mode='assignment'``:
    :func:`dyad_t_subset_scores` along every grid axis in turn
    (:func:`_axis_layouts`), each subset's per-term scores combined over the
    axes as :func:`dyad_scores` combines them (:func:`_combine_axes`) and
    summed (``nan_to_num`` per term). ``inf`` for the empty subset."""
    X = np.asarray(features, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    y = np.asarray(target, dtype=float).reshape(-1)
    w = None if sample_weights is None else np.asarray(sample_weights, dtype=float).reshape(-1)
    layouts = _axis_layouts(X.shape[0], grid_shape)
    if len(layouts) == 1:
        return dyad_t_subset_scores(X, y, w, grid_shape, subsets)
    per_axis = [dyad_t_subset_scores(X if order is None else X[order], y if order is None else y[order],
                                     w if order is None or w is None else w[order], shape, subsets,
                                     per_term=True)
                for _axis, order, shape in layouts]
    out = np.full(len(subsets), np.inf)
    by_size = {}
    for k, S in enumerate(subsets):
        if len(S):
            by_size.setdefault(len(S), []).append(k)
    for idxs in by_size.values():
        scores = np.stack([np.stack([axis[k] for k in idxs]) for axis in per_axis])      # (axes, m, size)
        with np.errstate(over='ignore'):
            out[idxs] = np.sum(np.nan_to_num(_combine_axes(scores)), axis=1)
    return out


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
    'survival': survival_scores, 'tile': tile_scores, 'dyad': dyad_scores,
    'het': heterogeneity_scores, 'het_raw': heterogeneity_raw_scores,
    'chi2': chi2_scores, 'chi2_centered': chi2_centered_scores,
}
