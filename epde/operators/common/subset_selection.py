#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Exhaustive subset fits in Gram space.

EPDE libraries are small -- an equation carries ``equation_terms_max_number``
terms, so a candidate fit sees ``p <= ~7`` feature columns -- and the
``2^(p+1)`` column subsets of ``[features | 1]`` can be fitted exhaustively in
Gram space (one weighted Gram per library, one k x k solve per subset). This
module holds those primitives for the two system right-part modes that fit
subsets themselves instead of running a sparsity operator:

* ``system_rps_mode='assignment'`` (``right_part_selection._equation_variants``)
  tabulates every subset's Discrepancy and Instability and keeps the
  non-dominated ones;
* ``system_rps_mode='plain'`` (``plain_fit._versions``) fits a candidate's own
  terms, with and without the constant.

:func:`subset_table` and :func:`instability_table` give the Discrepancy and
Instability objectives' own values per subset, :func:`subset_coefficients`
the weighted-OLS refit of one support, and :func:`trajectory_gram`,
:func:`support_scores` and :func:`store_fit` write a fitted support onto an
equation in ``VWSRSparsity``'s conventions, so the coefficient step and the
fitness host read it like any sparsity operator's output.

The knee subset-selection operator these primitives were first written for
(``sparsity_cls='knee'``, with its complexity-budget realization) has been
removed. The retired 19-library study that calibrated it,
``projects/thesis/true_subset_search.py``, keeps local copies of its
selection rules and still reads :func:`subset_table` from here.
"""

import numpy as np

from epde.operators.common.sparsity import cv_scores, instability_scores
from epde.operators.common.stability import GramSetup, VaryingCoefSetup

#: How the always-present ones column is handled by :func:`subset_table` and
#: :func:`subset_coefficients`.
#:   'selectable' -- the intercept is column ``p``, an ordinary member of the
#:                   enumeration, so a subset can leave it out: the constant is
#:                   a selectable term. What the assignment and plain modes
#:                   use: three downstream readers (``Instability.compute``'s
#:                   intercept rule, ``Complexity('terms')``,
#:                   ``amplification_ratio``) treat ``weights[-1] == 0`` as
#:                   "this equation has no constant term", and an intercept
#:                   that is non-zero by construction would lie to all three.
#:   'always'     -- the intercept is forced into every subset and excluded
#:                   from the amplification sum. The convention of the
#:                   19-library study, kept so that study reproduces exactly.
INTERCEPT_MODES = ('selectable', 'always')

#: Discrepancy options a subset can be scored on without solving a PDE. These
#: are computed here from the residual of the subset's own fit, reproducing
#: ``objectives.Discrepancy._compute_<option>`` for a SINGLE trajectory -- the
#: callers already run per trajectory, so the mean-over-trajectories the
#: filler applies is the caller's loop.
SOLVER_FREE_DISCREPANCIES = ('wape', 'l2', 'l2_relative', 'scale_invariant')


def discrepancy_value(metric, resid, t, w, contribs=None, intercept=0.0):
    """One trajectory's Discrepancy objective value for a fitted subset.

    Mirrors ``objectives.Discrepancy`` option for option:

    * ``wape``            ``sum|resid| / sum|t|`` -- UNWEIGHTED, as the filler
                          has it (the g_func weighting appears only in ``l2``);
    * ``l2``              ``||resid * w||_2``. The filler additionally divides
                          by ``penalty_coeff`` for an all-zero-weight equation;
                          that constant is a fitness-host parameter and is not
                          reachable here, but the empty subset is never a
                          candidate (the assignment mode's tables start at
                          bitmask 1, and every plain-mode mask carries the
                          features), so the branch is unreachable rather than
                          approximated;
    * ``l2_relative``     ``||resid|| / ||t||``;
    * ``scale_invariant`` pointwise cancellation ratio
                          ``mean(|resid| / (|t| + sum_k|c_k phi_k| + |b|))``,
                          which needs the per-term contributions.
    """
    if metric == 'wape':
        den = float(np.sum(np.abs(t)))
        return float(np.sum(np.abs(resid)) / den) if den > 0 else np.inf
    if metric == 'l2':
        return float(np.linalg.norm(resid * w, ord=2))
    if metric == 'l2_relative':
        den = float(np.linalg.norm(t))
        return float(np.linalg.norm(resid) / den) if den > 0 else np.inf
    if metric == 'scale_invariant':
        mass = np.abs(t) + abs(intercept)
        if contribs is not None and contribs.size:
            mass = mass + np.abs(contribs).sum(axis=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            return float(np.mean(np.nan_to_num(np.abs(resid) / mass)))
    raise ValueError(
        'Discrepancy option {0!r} cannot score a subset without solving the '
        'system; a subset fit accepts {1}.'.format(
            metric, list(SOLVER_FREE_DISCREPANCIES)))


# --------------------------------------------------------------------------
# Exhaustive subset table (Gram space)
# --------------------------------------------------------------------------
def subset_table(f, t, w, *, intercept: str = 'selectable',
                 discrepancy: str = None, subsets=None, return_coefs: bool = False):
    """Fit curve and amplification for EVERY column subset of ``[f | 1]``
    -- or for the bitmasks in ``subsets`` only, in that order.

    ``discrepancy=None`` gives the weighted RSS, read straight off the Gram
    identity -- the cheap default the 19-library study is calibrated on.
    Naming an option of :data:`SOLVER_FREE_DISCREPANCIES` instead returns the
    ``Discrepancy`` OBJECTIVE's value per subset, which is what the Pareto
    axis reads. That is not a free swap: WAPE and the scale-invariant option
    are L1 quantities with no Gram form, so each subset needs its residual
    vector materialised -- O(n*k) per subset against the O(k^3) solve that was
    the whole cost before. It also loses a guarantee: best-RSS-per-size is
    monotone in the subset size, but a curve the least-squares fit does not
    minimise need not be -- the best subset of size k+1 can score worse than
    the best of size k.

    One weighted Gram, one robust k x k solve per subset; RSS via the OLS
    identity ``yy - c'Gy_S``, floored at the machine level (exact-fit
    libraries would otherwise hand the log-ratios rounding noise).

    Amplification is the production guard ratio
    ``A(S) = sum_j |c_j| * ||col_j||_W / ||y||_W`` -- the same quantity
    ``right_part_selection.amplification_ratio`` computes against
    ``search_space.rps_amplification_cap``, including the intercept's
    ``|c| * sqrt(sum w)`` contribution under ``'selectable'``.

    Returns ``(rss, amp, n_cols)``; both arrays are indexed by the subset's
    bitmask over ``n_cols`` columns, or aligned with ``subsets`` when it is
    given (the same arithmetic per subset, so the values are the dense
    table's at those bitmasks). ``return_coefs`` appends each subset's
    coefficients in :func:`subset_coefficients`' full layout -- the same
    solve, so the same bytes (zeros for the empty subset).
    """
    if intercept not in INTERCEPT_MODES:
        raise ValueError('intercept must be one of {0}, got {1!r}.'.format(
            list(INTERCEPT_MODES), intercept))
    if discrepancy is not None and discrepancy not in SOLVER_FREE_DISCREPANCIES:
        raise ValueError(
            'Discrepancy option {0!r} cannot score a subset without solving '
            'the system; a subset fit accepts {1}.'.format(
                discrepancy, list(SOLVER_FREE_DISCREPANCIES)))
    f = np.asarray(f, dtype=float)
    t = np.asarray(t, dtype=float).reshape(-1)
    w = np.asarray(w, dtype=float).reshape(-1)
    n, p = f.shape
    Xa = np.hstack([f, np.ones((n, 1))])
    G = Xa.T @ (w[:, None] * Xa)
    Gy = Xa.T @ (w * t)
    yy = float(t @ (w * t))
    diag = np.sqrt(np.maximum(np.diag(G), 0.0))
    floor = np.finfo(float).eps * max(n, 1) * yy
    forced = intercept == 'always'
    n_cols = p if forced else p + 1

    masks = range(1 << n_cols) if subsets is None else subsets
    rss = np.empty(len(masks))
    amp = np.empty(len(masks))
    coefs = [np.zeros(p + 1) for _ in masks] if return_coefs else None
    # The empty model's own discrepancy sets the scale the machine floor is
    # taken against -- the exact analogue of ``yy`` for the RSS curve.
    base = (discrepancy_value(discrepancy, t, t, w, None, 0.0)
            if discrepancy is not None else yy)
    dfloor = np.finfo(float).eps * max(n, 1) * base
    for i, S in enumerate(masks):
        cols = [j for j in range(n_cols) if (S >> j) & 1]
        if forced:
            cols.append(p)
        elif not cols:
            # The empty model: intercept-free, so its residual is the target
            # itself. Never a candidate, but it anchors the size-0 end of the
            # curve.
            rss[i] = max(base, dfloor if discrepancy is not None else floor)
            amp[i] = 0.0
            continue
        GS = G[np.ix_(cols, cols)]
        # Cholesky-guarded solve: LU raises only on an EXACTLY zero pivot, so
        # eps-singular subsets returned garbage coefficients -- negative
        # pre-floor RSS reading as fake exact fits, which flipped the
        # selection on 92 of 200 adversarial seeds in the study. Cholesky
        # raises on those, and lstsq's min-norm projection keeps the RSS
        # identity exact (0 of 200 corrupted with the guard).
        try:
            np.linalg.cholesky(GS)
            c = np.linalg.solve(GS, Gy[cols])
        except np.linalg.LinAlgError:
            c = np.linalg.lstsq(GS, Gy[cols], rcond=None)[0]
        if return_coefs:
            coefs[i][cols] = c
        if discrepancy is None:
            rss[i] = max(yy - float(c @ Gy[cols]), floor)
        else:
            # The residual vector, which no Gram identity supplies for an L1
            # objective. ``contribs`` is only built for the one option that
            # needs the per-term split.
            cols_f = [j for j in cols if j < p]
            Xs = Xa[:, cols]
            resid = t - Xs @ c
            contribs = (Xa[:, cols_f] * c[:len(cols_f)][None, :]
                        if discrepancy == 'scale_invariant' else None)
            # ``cols`` is built in increasing column order, so the
            # intercept -- column p -- is last whenever it is in at all.
            b = float(c[-1]) if (forced or cols[-1] == p) else 0.0
            rss[i] = max(discrepancy_value(discrepancy, resid, t, w, contribs,
                                           b), dfloor)
        if forced:
            amp[i] = (float(np.abs(c[:-1]) @ diag[cols[:-1]])
                      / max(np.sqrt(yy), 1e-300))
        else:
            amp[i] = (float(np.abs(c) @ diag[cols])
                      / max(np.sqrt(yy), 1e-300))
    if return_coefs:
        return rss, amp, n_cols, coefs
    return rss, amp, n_cols


def subset_coefficients(f, t, w, support, *, intercept: str = 'selectable'):
    """Weighted-OLS coefficients on ``support``, in FULL layout.

    Returns a length ``p+1`` vector (one slot per feature column, then the
    intercept) with zeros off the support -- the unified coefficient layout
    of ``Equation._validate_weight_layout``. Same Cholesky guard and the same
    arithmetic as the support's entry in :func:`subset_table`: the caller
    picks the support and an unpenalized fit supplies the magnitudes.
    """
    f = np.asarray(f, dtype=float)
    t = np.asarray(t, dtype=float).reshape(-1)
    w = np.asarray(w, dtype=float).reshape(-1)
    n, p = f.shape
    Xa = np.hstack([f, np.ones((n, 1))])
    G = Xa.T @ (w[:, None] * Xa)
    Gy = Xa.T @ (w * t)
    cols = [j for j in range(p + 1) if support[j]]
    if intercept == 'always' and p not in cols:
        cols.append(p)
    full = np.zeros(p + 1)
    if not cols:
        return full
    GS = G[np.ix_(cols, cols)]
    try:
        np.linalg.cholesky(GS)
        c = np.linalg.solve(GS, Gy[cols])
    except np.linalg.LinAlgError:
        c = np.linalg.lstsq(GS, Gy[cols], rcond=None)[0]
    full[cols] = c
    return full


# --------------------------------------------------------------------------
# Exhaustive instability table
# --------------------------------------------------------------------------
def instability_table(metric, X, y, sw, grid_shape, n_cols, *,
                      gram_setup=None, subsets=None):
    """Per-subset instability -- the Instability axis of the assignment
    mode's subset search -- indexed by bitmask, or aligned with the bitmasks
    in ``subsets`` when it is given.

    One score per subset, reduced by SUM over the subset's members -- the
    same reduction ``Instability.compute`` applies, so the number a subset is
    judged by here is the number it would carry on the Pareto front.
    ``inf`` for the empty subset.

    COST WARNING. This is one estimator call per subset, against one Gram
    solve per subset for :func:`subset_table`. Measured on Lotka-Volterra
    (120 points, 7 columns) it is ~3x the RSS table; on the large-grid PDE
    systems of the 19-library study the same comparison was ~240x. Under
    ``instability_metric='dyad'`` the assignment mode tabulates the same
    statistic with ``survival.dyad_subset_scores`` instead, which builds the
    slab Grams once for all the subsets.
    """
    masks = range(1 << n_cols) if subsets is None else subsets
    scores = np.full(len(masks), np.inf)
    n_features = n_cols - 1
    for i, S in enumerate(masks):
        if S == 0:
            continue
        mask = mask_of(S, n_cols)
        weights = (gram_setup.solve(mask) if metric == 'cv' else None)
        per_term = instability_scores(metric, X, y, sw, grid_shape, mask,
                                      n_features, gram_setup=gram_setup,
                                      cv_reducer=cv_scores, weights=weights)
        scores[i] = float(np.sum(np.nan_to_num(np.asarray(per_term,
                                                          dtype=float))))
    return scores


def mask_of(subset, n_cols):
    """Bitmask integer -> boolean column mask."""
    return np.array([(int(subset) >> j) & 1 for j in range(n_cols)],
                    dtype=bool)


# --------------------------------------------------------------------------
# Writing a fitted support onto an equation
# --------------------------------------------------------------------------
def trajectory_gram(gram_setup, X, y, sw, grid_shape, gram_mode, main_var,
                    key):
    """One trajectory's Gram view: ``gram_setup`` when the caller already
    holds one, else built here for the Gram-coupled metrics, else ``None`` (a
    basis-free metric needs none)."""
    if gram_setup is None and gram_mode is not None:
        # ``sample_key`` selects the trajectory whose field the
        # Taylor-microscale basis resolution reads -- see the same
        # note in PhysicsInformedLasso.fit.
        gram_setup = (
            VaryingCoefSetup(X, y, sw, grid_shape, main_var=main_var,
                             sample_key=key)
            if gram_mode == 'vcoef'
            else GramSetup(X, y, sw, grid_shape))
    return gram_setup


def support_scores(gram_setup, active, metric):
    """``(vc_score, sw_weights)`` of one trajectory's fitted support -- the
    two caches the Instability filler reads, in VWSR's conventions."""
    vc_score = (gram_setup.score(active)
                if getattr(gram_setup, 'is_vcoef', False) and active.any()
                else None)
    sw_weights = (gram_setup.solve(active)
                  if metric == 'cv' and gram_setup is not None
                  and active.any()
                  else None)
    return vc_score, sw_weights


def store_fit(objective, weights, sw_weights, vc_scores):
    """Write a fit onto the equation, in VWSRSparsity's conventions.

    ``weights`` is the full-layout vector (one slot per non-target term, then
    the intercept, zeros off the support); ``sw_weights`` and ``vc_scores``
    map each trajectory to its :func:`support_scores` entries. A zero slot is
    a term the fit left out -- the intercept's included -- so an all-zero
    vector is the empty equation the fitness host declines.
    """
    # ``subset_coefficients`` fits on the physical scale, so the final
    # magnitudes ARE the internal ones. ``LinRegBasedCoeffsEquation`` promotes
    # this vector rather than refitting it when the wired sparsity class
    # declares ``fits_physical_scale`` (the default VWSR does; plain mode
    # refuses one that does not), and stays the sole writer of
    # ``weights_final_evald``.
    objective.weights_internal = weights
    objective.weights_internal_evald = True
    # Assigned UNCONDITIONALLY, including the None case: the equation may
    # carry an earlier fit's caches (a deep copy of a fitted equation, or one
    # EqRightPartSelector's term sweep copies both out of), so a skipped
    # assignment would leave that fit's value behind.
    objective._cached_sw_weights = (
        None if not sw_weights or all(v is None for v in sw_weights.values())
        else sw_weights)
    objective._cached_vc_score = (
        None if not vc_scores or all(v is None for v in vc_scores.values())
        else vc_scores)
