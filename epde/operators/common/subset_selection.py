#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Subset-selection sparsity: pick the support by fit GEOMETRY, not by a
shrinkage threshold.

``PhysicsInformedLasso`` (``sparsity.py``) decides the support with a
shrinkage rule -- kill feature *j* when ``|rho_j| <= score_j * max_corr`` --
and then throws the shrinkage away in its relaxed refit. That needs a SCALE,
and ``max_corr`` is a signal-scale anchor doing a noise-scale job: measured on
the true Lotka-Volterra u-equation the resulting threshold spans five orders
of magnitude across the six instability metrics (0.0001% of ``max_corr`` for
'het' to 6.6% for 'tile'), so one estimator's rule is a no-op where another's
strips valid structure.

A SUBSET rule needs neither. EPDE libraries are small -- an equation carries
``equation_terms_max_number`` terms, so a candidate fit sees ``p <= ~7``
feature columns -- and the ``2^(p+1)`` column subsets can be enumerated
exhaustively in Gram space (one weighted Gram per library, one k x k solve per
subset). Greedy-path error disappears and the design question reduces to the
SELECTION OBJECTIVE that ranks the true subset first.

The objective implemented here is the scree KNEE and its extensions, ported
from the 19-library study in ``projects/thesis/true_subset_search.py`` (which
imports these same primitives, so the two can never drift): ``knee_ext2``
recovered the exact true support on 18 of 19 libraries with ZERO false
negatives, both Lotka-Volterra equations included -- where a production-style
necessity-margin rule scored 15/19 with LV among its failures.

SCOPE. Every rule here reads an RSS table, and on clean EPDE data the
residual is finite-difference error rather than noise. The same study put the
noise ceiling for ALL RSS-table rules at ~5/19 at 1% noise. This is a
clean-data instrument.
"""

import numpy as np

import epde.globals as global_var
from epde.interface.search_config import active_config
from epde.operators.utils.template import CompoundOperator
from epde.operators.common.sparsity import (_term_tie_label, cv_scores,
                                            instability_scores)
from epde.operators.common.stability import GramSetup, VaryingCoefSetup
from epde.structure.main_structures import COMPLEXITY_METAPARAMETER
from epde import _loop_stats

#: Selection rules, in increasing order of what they consult. Each is a PURE
#: rule -- one statistic, the same mechanics -- so they can be compared
#: against each other without a hybrid confounding the result.
REALIZATIONS = ('knee', 'knee_ext2', 'knee_ext2_amp', 'ke2_stab',
                'stab_chain')

#: Realizations that select by a COMPLEXITY BUDGET instead of a chain rule:
#: the equation carries an integer budget ``c`` in its own
#: ``('complexity', var)`` metaparameter -- a property separate from the
#: ``('sparsity', var)`` alpha, which nothing here reads -- and the operator
#: returns the best-fitting subset at the largest achievable complexity
#: ``<= c``, leaving the screening to the Pareto front: the candidate
#: generator of a best-subset study. Kept OUT of :data:`REALIZATIONS` on
#: purpose: a budget has no chain, and :func:`select_from_chain` falls through
#: to ``extend2`` for any listed name it does not special-case, so listing it
#: there would silently run ``knee_ext2`` the day someone dispatched it by
#: name.
BUDGET_REALIZATIONS = ('complexity',)

#: How the always-present ones column is handled by :func:`subset_table`.
#:   'selectable' -- the intercept is column ``p``, an ordinary member of the
#:                   enumeration, so the rule can leave it out. What
#:                   :class:`KneeSparsity` uses: three downstream readers
#:                   (``Instability.compute``'s intercept rule,
#:                   ``Complexity('terms')``, ``amplification_ratio``) treat
#:                   ``weights[-1] == 0`` as "this equation has no constant
#:                   term", and an intercept that is non-zero by construction
#:                   would lie to all three.
#:   'always'     -- the intercept is forced into every subset and excluded
#:                   from the amplification sum. The convention of the
#:                   19-library study, kept so that study reproduces exactly.
INTERCEPT_MODES = ('selectable', 'always')

#: Discrepancy options a subset can be scored on without solving a PDE. These
#: are computed here from the residual of the subset's own fit, reproducing
#: ``objectives.Discrepancy._compute_<option>`` for a SINGLE trajectory -- the
#: operator already runs per trajectory, so the mean-over-trajectories the
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
                          reachable here, but the empty subset is never
                          selected (every rule starts at size 1), so the branch
                          is unreachable rather than approximated;
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
        'system; the knee curve accepts {1}.'.format(
            metric, list(SOLVER_FREE_DISCREPANCIES)))


# --------------------------------------------------------------------------
# Exhaustive subset table (Gram space)
# --------------------------------------------------------------------------
def subset_table(f, t, w, *, intercept: str = 'selectable',
                 discrepancy: str = None):
    """Fit curve and amplification for EVERY column subset of ``[f | 1]``.

    ``discrepancy=None`` gives the weighted RSS, read straight off the Gram
    identity -- the cheap default the 19-library study is calibrated on.
    Naming an option of :data:`SOLVER_FREE_DISCREPANCIES` instead returns the
    ``Discrepancy`` OBJECTIVE's value per subset, which is what the Pareto
    axis reads. That is not a free swap: WAPE and the scale-invariant option
    are L1 quantities with no Gram form, so each subset needs its residual
    vector materialised -- O(n*k) per subset against the O(k^3) solve that was
    the whole cost before. It also loses a guarantee: best-RSS-per-size is
    monotone in the subset size, but a curve the least-squares fit does not
    minimise need not be, which is why :func:`build_chain` accumulates the
    minimum for these.

    One weighted Gram, one robust k x k solve per subset; RSS via the OLS
    identity ``yy - c'Gy_S``, floored at the machine level (exact-fit
    libraries would otherwise hand the log-ratios rounding noise).

    Amplification is the production guard ratio
    ``A(S) = sum_j |c_j| * ||col_j||_W / ||y||_W`` -- the same quantity
    ``right_part_selection.amplification_ratio`` computes against
    ``search_space.rps_amplification_cap``, including the intercept's
    ``|c| * sqrt(sum w)`` contribution under ``'selectable'``.

    Returns ``(rss, amp, n_cols)``; both arrays are indexed by the subset's
    bitmask over ``n_cols`` columns.
    """
    if intercept not in INTERCEPT_MODES:
        raise ValueError('intercept must be one of {0}, got {1!r}.'.format(
            list(INTERCEPT_MODES), intercept))
    if discrepancy is not None and discrepancy not in SOLVER_FREE_DISCREPANCIES:
        raise ValueError(
            'Discrepancy option {0!r} cannot score a subset without solving '
            'the system; the knee curve accepts {1}.'.format(
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

    n_subsets = 1 << n_cols
    rss = np.empty(n_subsets)
    amp = np.empty(n_subsets)
    # The empty model's own discrepancy sets the scale the machine floor is
    # taken against -- the exact analogue of ``yy`` for the RSS curve.
    base = (discrepancy_value(discrepancy, t, t, w, None, 0.0)
            if discrepancy is not None else yy)
    dfloor = np.finfo(float).eps * max(n, 1) * base
    for S in range(n_subsets):
        cols = [j for j in range(n_cols) if (S >> j) & 1]
        if forced:
            cols.append(p)
        elif not cols:
            # The empty model: intercept-free, so its residual is the target
            # itself. Never selected (every rule's chain starts at size 1) but
            # it anchors the size-0 end of the curve.
            rss[S] = max(base, dfloor if discrepancy is not None else floor)
            amp[S] = 0.0
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
        if discrepancy is None:
            rss[S] = max(yy - float(c @ Gy[cols]), floor)
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
            rss[S] = max(discrepancy_value(discrepancy, resid, t, w, contribs,
                                           b), dfloor)
        if forced:
            amp[S] = (float(np.abs(c[:-1]) @ diag[cols[:-1]])
                      / max(np.sqrt(yy), 1e-300))
        else:
            amp[S] = (float(np.abs(c) @ diag[cols])
                      / max(np.sqrt(yy), 1e-300))
    return rss, amp, n_cols


def subset_coefficients(f, t, w, support, *, intercept: str = 'selectable'):
    """Weighted-OLS coefficients on ``support``, in FULL layout.

    Returns a length ``p+1`` vector (one slot per feature column, then the
    intercept) with zeros off the support -- the unified coefficient layout
    of ``Equation._validate_weight_layout``. Same Cholesky guard and the same
    arithmetic as the winning subset's entry in :func:`subset_table`: this is
    the relaxed refit, so the RULE picks the support and an unpenalized fit
    supplies the magnitudes.
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
# The knee chain and its extensions
# --------------------------------------------------------------------------
def instability_table(metric, X, y, sw, grid_shape, n_cols, *,
                      gram_setup=None):
    """Per-subset instability, for the realizations that read BOTH axes.

    One score per subset, reduced by SUM over the subset's members -- the
    same reduction ``Instability.compute`` applies, so the number a subset is
    judged by here is the number it would carry on the Pareto front.
    ``inf`` for the empty subset.

    COST WARNING. This is one estimator call per subset, against one Gram
    solve per subset for :func:`subset_table`. Measured on Lotka-Volterra
    (120 points, 7 columns) it is ~3x the RSS table; on the large-grid PDE
    systems of the 19-library study the same comparison was ~240x. Only
    ``'stab_chain'`` pays it -- ``'ke2_stab'`` scores at the <= p extension
    decisions instead of over all 2^n_cols subsets.
    """
    scores = np.full(1 << n_cols, np.inf)
    n_features = n_cols - 1
    for S in range(1, 1 << n_cols):
        mask = mask_of(S, n_cols)
        weights = (gram_setup.solve(mask) if metric == 'cv' else None)
        per_term = instability_scores(metric, X, y, sw, grid_shape, mask,
                                      n_features, gram_setup=gram_setup,
                                      cv_reducer=cv_scores, weights=weights)
        scores[S] = float(np.sum(np.nan_to_num(np.asarray(per_term,
                                                          dtype=float))))
    return scores


def stability_admissible_chain(rss, instab, n_cols, amp=None, amp_cap=None):
    """Best-RSS-per-size chain restricted to instability-admissible subsets.

    The HYBRID reading of "use both objectives": fit geometry still owns the
    chain and the elbow, but each size class is first narrowed to the subsets
    whose summed instability is no worse than the MEDIAN of that class. The
    median is per-equation self-calibration -- the convention every other
    condition in this module follows -- so no cross-system constant enters.
    Falls back to the unrestricted class when the restriction empties it,
    exactly as the amplification restriction does.

    Deliberately NOT a Pareto rule over the two axes. Measured on an inflated
    LV library, the true support is not even non-dominated in
    (rss, instability): four subsets carrying coordinate-modulated identity
    columns beat it on BOTH axes at once, because those columns absorb the
    modulation and so lower the instability while also lowering the residual.
    Any rule monotone in both objectives therefore prefers the junk. Keeping
    RSS in charge and using instability only to narrow the field avoids that,
    because the elbow reads the SHAPE of the RSS curve rather than its level.
    """
    sizes = np.array([bin(S).count('1') for S in range(1 << n_cols)])
    best_rss = np.empty(n_cols + 1)
    best_sub = np.empty(n_cols + 1, dtype=int)
    for k in range(n_cols + 1):
        cand = np.where(sizes == k)[0]
        if amp is not None and amp_cap is not None:
            adm = cand[amp[cand] <= amp_cap]
            if len(adm):
                cand = adm
        finite = cand[np.isfinite(instab[cand])]
        if len(finite) > 1:
            adm = finite[instab[finite] <= np.median(instab[finite])]
            if len(adm):
                cand = adm
        best_sub[k] = int(cand[np.argmin(rss[cand])])
        best_rss[k] = rss[best_sub[k]]
    d = -np.diff(np.log(np.minimum.accumulate(best_rss)))
    return best_rss, best_sub, d


def knee_chain(rss, n_cols, amp=None, amp_cap=None):
    """Best-RSS-per-size chain over the subset table.

    With ``amp``/``amp_cap`` the per-size argmin is restricted to
    amplification-admissible subsets, falling back to the unrestricted best
    when a size has none. Returns ``(best_rss, best_sub, d)`` with ``d`` the
    log-RSS drop from size k-1 to size k.
    """
    sizes = np.array([bin(S).count('1') for S in range(1 << n_cols)])
    best_rss = np.empty(n_cols + 1)
    best_sub = np.empty(n_cols + 1, dtype=int)
    for k in range(n_cols + 1):
        cand = np.where(sizes == k)[0]
        if amp is not None and amp_cap is not None:
            adm = cand[amp[cand] <= amp_cap]
            if len(adm):
                cand = adm
        best_sub[k] = int(cand[np.argmin(rss[cand])])
        best_rss[k] = rss[best_sub[k]]
    d = -np.diff(np.log(best_rss))               # drop from k-1 -> k
    return best_rss, best_sub, d


def knee_size(d):
    """The scree elbow: the size after which the per-size drop falls off
    hardest (``argmax d_k - d_{k+1}``)."""
    d2 = np.append(d, 0.0)
    return int(np.argmax(d2[:-1] - d2[1:])) + 1


def extend2(d, best_rss, best_sub, n_cols, k):
    """Nested-dominant extension with the min-accepted-drop yardstick.

    Grow the size past the elbow while the next size's best subset

    (a) NESTS on the current one -- an extension, not a different model;
    (b) does not land on the machine floor -- an exact fit is not evidence;
    (c) its drop exceeds ALL remaining possible improvement; and
    (d) its drop is at least as large as the weakest already-accepted drop,
        i.e. "a new term must be at least as informative as the weakest term
        already accepted".

    (d) is the gate that separates a weak-but-real term (the Allen-Cahn
    diffusion term, whose drop is small in absolute terms yet no smaller
    than the weakest true term already in) from same-looking junk. Every
    condition is self-calibrated against this equation's own curve; there is
    no constant to tune.
    """
    gmin = best_rss.min()
    while k < n_cols:
        nxt, cur = int(best_sub[k + 1]), int(best_sub[k])
        if (nxt & cur) != cur:
            break
        if best_rss[k + 1] <= gmin:
            break
        if not d[k] > d[k + 1:].sum():
            break
        if not d[k] >= d[:k].min():
            break
        k += 1
    return k


def extend2_stab(d, best_rss, best_sub, n_cols, k, stability):
    """:func:`extend2` plus a STABILITY veto on each extension term.

    Fit geometry FINDS the subset; stability only VETOES extensions. Beyond
    the four geometric conditions, the new column must also be no more
    unstable than the least stable already-accepted member, scored WITHIN the
    extended subset's own fit -- the min-accepted-drop yardstick mirrored
    onto the stability axis, and self-calibrated in the same way.

    The elbow subset (the CORE) is never stability-tested: fit geometry is
    only provably blind at the extension decision, and a mildly incoherent
    but genuinely necessary core term (the viscous Navier-Stokes terms) must
    not be vetoed by a criterion that was never asked whether the term is
    needed.

    ``stability(mask) -> scores`` takes a length ``n_cols`` boolean support
    and returns one score per selected column, aligned to
    ``np.flatnonzero(mask)``. Returns ``(subset, vetoed_column_or_None)``.
    """
    gmin = best_rss.min()
    vetoed = None
    while k < n_cols:
        nxt, cur = int(best_sub[k + 1]), int(best_sub[k])
        if (nxt & cur) != cur:
            break
        if best_rss[k + 1] <= gmin:
            break
        if not d[k] > d[k + 1:].sum():
            break
        if not d[k] >= d[:k].min():
            break
        j_new = (nxt & ~cur).bit_length() - 1
        mask = np.array([(nxt >> j) & 1 for j in range(n_cols)], dtype=bool)
        sc = np.nan_to_num(np.asarray(stability(mask), dtype=float))
        members = np.flatnonzero(mask)
        pos = int(np.where(members == j_new)[0][0])
        s_new = float(sc[pos])
        others = np.delete(sc, pos)
        if s_new > (float(others.max()) if others.size else 0.0):
            vetoed = j_new                 # less stable than every accepted
            break                          # member -> reject the extension
        k += 1
    return int(best_sub[k]), vetoed


def build_chain(realization, rss, amp, n_cols, *, amp_cap=None,
                instab=None, accumulate=False):
    """The ``(best_rss, best_sub, d)`` chain the realization selects from.

    The amplification-restricted realizations take their drops from the
    ACCUMULATED (monotone) chain: an admissible-only size class can be worse
    than the smaller one, and the raw diff would then go negative and corrupt
    the tail-dominance sums in :func:`extend2`.

    ``instab`` is required by ``'stab_chain'`` and ignored by the others.
    ``accumulate`` forces the same monotone treatment on the unrestricted
    chains, which a DISCREPANCY curve needs: the least-squares fit does not
    minimise it, so the best subset of size k+1 can score worse than the best
    of size k, and the raw log-diff would go negative and corrupt the
    tail-dominance sums in :func:`extend2`. It is the identity on an RSS
    curve, which is monotone by construction.
    """
    if realization == 'stab_chain':
        if instab is None:
            raise ValueError("realization 'stab_chain' needs an instability "
                             'table.')
        return stability_admissible_chain(rss, instab, n_cols, amp=amp,
                                          amp_cap=amp_cap)
    if realization in ('knee_ext2_amp', 'ke2_stab'):
        best_rss, best_sub, _ = knee_chain(rss, n_cols, amp=amp,
                                           amp_cap=amp_cap)
        d = -np.diff(np.log(np.minimum.accumulate(best_rss)))
        return best_rss, best_sub, d
    best_rss, best_sub, d = knee_chain(rss, n_cols)
    if accumulate:
        d = -np.diff(np.log(np.minimum.accumulate(best_rss)))
    return best_rss, best_sub, d


def select_from_chain(realization, best_rss, best_sub, d, n_cols, *,
                      stability=None):
    """Apply ``realization``'s rule to a prebuilt chain.

    Split out from :func:`select_support` so the greedy fallback -- which
    builds its chain without a subset table -- runs the SAME rule.
    Returns ``(subset_bitmask, vetoed_column_or_None)``.
    """
    if realization not in REALIZATIONS:
        raise ValueError('Unknown knee realization {0!r}; expected one of '
                         '{1}.'.format(realization, list(REALIZATIONS)))
    k = knee_size(d)
    if realization == 'knee':
        return int(best_sub[k]), None
    if realization == 'ke2_stab':
        if stability is None:
            raise ValueError("realization 'ke2_stab' needs a stability "
                             'callback.')
        return extend2_stab(d, best_rss, best_sub, n_cols, k, stability)
    return int(best_sub[extend2(d, best_rss, best_sub, n_cols, k)]), None


def select_support(realization, rss, amp, n_cols, *, amp_cap=None,
                   stability=None, instab=None, accumulate=False):
    """The realization's chosen subset, as a bitmask over ``n_cols``."""
    best_rss, best_sub, d = build_chain(realization, rss, amp, n_cols,
                                        amp_cap=amp_cap, instab=instab,
                                        accumulate=accumulate)
    return select_from_chain(realization, best_rss, best_sub, d, n_cols,
                             stability=stability)


def mask_of(subset, n_cols):
    """Bitmask integer -> boolean column mask."""
    return np.array([(int(subset) >> j) & 1 for j in range(n_cols)],
                    dtype=bool)


# --------------------------------------------------------------------------
# The complexity budget ('complexity' realization)
# --------------------------------------------------------------------------
#: The intercept column's tie-break label in :func:`select_by_budget`. Any
#: fixed value no term label can equal will do: there is one intercept.
INTERCEPT_TIE_LABEL = ('<intercept>',)

#: Relative tolerance under which :func:`select_by_budget` reads two subsets'
#: ``curve`` -- then ``amp`` -- values as TIED. Not an exact comparison: on
#: clean data every "truth + junk" subset of a level is floored to the same
#: curve, and their amplifications then differ only in the last few ulps --
#: the junk coefficients are rounding noise, and the rounding depends on the
#: column order through the Cholesky solve. Compared exactly, those ulps
#: decided the pick, and the order-free label key was reached only when two
#: amplifications happened to round identically (measured: a budget above an
#: exact truth picked up to five different label sets over 24 column orders).
#: 1e-9 is ~7 orders above that rounding and far below any difference
#: between two genuinely different fits.
BUDGET_TIE_RTOL = 1e-9


def _near_minimum(values, pool, rtol=BUDGET_TIE_RTOL):
    """The members of ``pool`` whose ``values`` entry is within ``rtol`` of
    the pool's minimum.

    A band anchored at the minimum rather than a rounding of every value:
    quantising (say, to 9 significant digits) puts a bin edge between two
    ulp-apart values now and then, which is the order dependence over again;
    the band always holds both.
    """
    vals = values[pool]
    best = vals.min()
    # Written as a difference so a best value at the float maximum (an
    # infinite curve after ``nan_to_num``) cannot overflow the band, with the
    # equality term keeping an all-infinite pool whole (inf - inf is nan).
    with np.errstate(over='ignore', invalid='ignore'):
        keep = (vals == best) | (vals - best <= rtol * abs(best))
    return pool[keep]


def term_atoms(term):
    """A term's complexity: its number of factors AFTER power collapse.

    ``Term.defined`` and ``Term.randomize`` both end in ``filter_powers``, which
    merges equal factors into one factor carrying the summed power -- so
    ``u*u`` is ONE atom ``u^2``, the best-subset study's convention (a power of
    a variable is an atom costing 1, a product of two atoms costs 2). A trig
    token, whose frequency is a parameter, a grid token and a derivative token
    are one factor each. Deliberately not
    ``eq_mo_objectives._complexity_of_equation``: that weighs a derivative by
    its order and multiplies the term by its last factor's power, so it is not
    an atom count.
    """
    return len(term.structure)


def column_costs(equation):
    """Per-column costs, aligned to the feature matrix the operator fits.

    The non-target terms in structure order -- the layout of both the tier-3
    ``Z`` slice and ``Equation.evaluate`` -- then the intercept at cost 1. The
    target is the fixed left-hand side and costs nothing, as in the study.
    """
    atoms = [term_atoms(term) for idx, term in enumerate(equation.structure)
             if idx != equation.target_idx]
    return np.append(np.asarray(atoms, dtype=int), 1)


def subset_costs(col_costs):
    """Cost of every column subset, indexed by bitmask (bit j = column j)."""
    col_costs = np.asarray(col_costs, dtype=int)
    n_cols = col_costs.size
    bits = (np.arange(1 << n_cols)[:, None] >> np.arange(n_cols)) & 1
    return bits @ col_costs


def complexity_budget(equation):
    """The equation's ``('complexity', var)`` metaparameter value, or ``None``.

    The budget is a property of its own: the population constructors seed it
    only when the configured sparsity operator selects by a budget, and the
    ``('sparsity', var)`` alpha beside it is never consulted here. An equation
    without the key -- a translated truth, a standalone ``Equation``, a test
    stub -- has no budget, which :func:`resolve_budget` reads as "every term
    may stay".
    """
    try:
        return equation.metaparameters[
            (COMPLEXITY_METAPARAMETER, equation.main_var_to_explain)]['value']
    except (AttributeError, KeyError, TypeError):
        return None


def resolve_budget(value, total):
    """The integer budget a metaparameter value stands for.

    Rounded half up and clipped to ``[1, total]``. The population seeds and
    the mutation steps integers, but a value set by hand is read by its
    nearest integer, so ``2.6`` is ``3`` and ``2.5`` is ``3`` (``np.round``
    would make it ``2``). ``None``, a non-numeric value, a non-finite one or
    one ``<= 0`` means NO budget, i.e. ``total`` -- never ``1``: refitting an
    equation that was never given a budget (a translated truth, a standalone
    equation) at one atom would collapse a correct equation over a gene it
    does not carry.
    """
    try:
        value = float(value)
    except (TypeError, ValueError):
        return int(total)
    if not np.isfinite(value) or value <= 0:
        return int(total)
    return int(min(max(np.floor(value + 0.5), 1), total))


def select_by_budget(curve, amp, costs, budget, n_cols, *, amp_cap=None,
                     labels=None):
    """The best subset at the largest achievable complexity ``<= budget``.

    Candidates hold at least one FEATURE column: the fitness host declines an
    equation whose term weights are all zero, so the empty and the
    intercept-only subsets would be wasted picks. With ``amp_cap`` they are
    narrowed to the amplification-admissible ones, falling back to all of them
    when none is admissible (:func:`knee_chain`'s convention). The narrowing
    comes BEFORE the budget, so the level chosen is the largest one an
    admissible subset reaches, not an amplified pick the host would decline.

    The level is the highest cost present among the candidates costing
    ``<= budget``. When none fits -- ``budget=1`` with no single-atom term --
    it is the CHEAPEST candidate's cost instead, so the realized complexity
    then EXCEEDS the budget: the alternative is no candidate at all. Within the
    level the lowest ``curve`` wins.

    Ties are common, not an edge case: on clean data with a budget above the
    true complexity many "truth + junk" subsets sit on the same machine floor.
    They break by ``amp``, then by the sorted ``labels`` of the subset's
    columns (one label per column, the intercept's included), which do not
    depend on the column order. An equation's structure order is random, so a
    bitmask tie-break would make the pick depend on it; ``labels=None`` falls
    back to the bitmask for callers that have no labels. "Tied" means within
    :data:`BUDGET_TIE_RTOL` of the best, for ``curve`` and ``amp`` alike: the
    floored subsets' amplifications differ in rounding that itself depends on
    the column order, so an exact comparison would decide by that order.
    """
    if n_cols < 2:
        raise ValueError('A budget needs at least one feature column besides '
                         'the intercept; got n_cols={0}.'.format(n_cols))
    curve = np.nan_to_num(np.asarray(curve, dtype=float), nan=np.inf)
    amp = (np.zeros(curve.size) if amp is None
           else np.nan_to_num(np.asarray(amp, dtype=float), nan=np.inf))
    costs = np.asarray(costs)
    subsets = np.arange(1 << n_cols)
    feature_bits = (1 << (n_cols - 1)) - 1          # the intercept is the top bit
    cand = subsets[(subsets & feature_bits) != 0]
    if amp_cap is not None:
        admissible = cand[amp[cand] <= amp_cap]
        if admissible.size:
            cand = admissible
    within = cand[costs[cand] <= budget]
    if within.size:
        pool = within[costs[within] == costs[within].max()]
    else:
        pool = cand[costs[cand] == costs[cand].min()]

    # Lexicographic with a tolerance band: the near-best curve, then within
    # those the near-best amplification, then the labels.
    pool = _near_minimum(amp, _near_minimum(curve, pool))
    if labels is None:
        return int(pool.min())
    return int(min(pool, key=lambda S: tuple(sorted(
        labels[j] for j in range(n_cols) if (int(S) >> j) & 1))))


def budget_prefix(best_sub, col_costs, budget, single_curve):
    """The greedy fallback's budget pick: the largest nested prefix of a
    forward chain that fits ``budget``.

    The chain is nested, so its costs never decrease and the last prefix that
    fits is the largest. Without the exhaustive table this is a heuristic --
    forward selection need not visit the best subset of a complexity, nor even
    reach the largest achievable one.

    When no feature-carrying prefix fits, the chain's FIRST pick is already
    too dear -- a two-atom term at a budget of 1, say -- but a cheaper column
    may still fit on its own. The pick is then the affordable feature column
    with the lowest ``single_curve`` (one entry per column: the curve of that
    column's one-column fit on the same data, as ``greedy_chain(...,
    singles=True)`` records it), and only when no feature column costs
    ``<= budget`` the best of the CHEAPEST ones -- the cheapest feature
    subset, exceeding the budget for the reason :func:`select_by_budget`
    gives. There is no amplification narrowing at this step: a one-column
    least-squares fit has ``A = |<x, y>_W| / (||x||_W ||y||_W) <= 1``
    (Cauchy-Schwarz), admissible under any cap of at least 1.
    """
    col_costs = np.asarray(col_costs, dtype=int)
    n_cols = col_costs.size
    feature_bits = (1 << (n_cols - 1)) - 1

    def cost(S):
        return int(sum(col_costs[j] for j in range(n_cols) if (S >> j) & 1))

    carrying = [int(S) for S in best_sub[1:] if int(S) & feature_bits]
    fitting = [S for S in carrying if cost(S) <= budget]
    if fitting:
        return fitting[-1]
    single_curve = np.nan_to_num(np.asarray(single_curve, dtype=float),
                                 nan=np.inf)
    features = np.arange(n_cols - 1)              # the intercept is the last
    affordable = features[col_costs[features] <= budget]
    pool = (affordable if affordable.size else
            features[col_costs[features] == col_costs[features].min()])
    return 1 << int(pool[np.argmin(single_curve[pool])])


# --------------------------------------------------------------------------
# Greedy fallback for libraries too large to enumerate
# --------------------------------------------------------------------------
def greedy_chain(f, t, w, *, intercept: str = 'selectable', amp_cap=None,
                 discrepancy: str = None, singles: bool = False):
    """Forward-selection chain, for ``n_cols > max_exhaustive_columns``.

    At each size the column whose addition minimises RSS is taken (preferring
    an amplification-admissible choice when one exists, mirroring
    :func:`knee_chain`). The SAME ``knee_size`` / ``extend2`` rule then runs
    on the resulting chain.

    Weaker than the exhaustive path in a way worth naming: a forward chain is
    nested by construction, so :func:`extend2`'s nesting test can never fire
    and one of its four exit conditions is inert. Live equations carry at
    most ``equation_terms_max_number`` terms, so this is a safety valve
    rather than a path the search normally takes.

    Returns ``(best_rss, best_sub, d, n_cols)`` in :func:`knee_chain` shape.
    ``singles=True`` appends ``(curve, amp)``, two length-``n_cols`` arrays
    holding every column's one-column fit -- the first step scores them all
    anyway, and :func:`budget_prefix` needs them when the chain's first pick
    does not fit a budget.
    """
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

    base = (discrepancy_value(discrepancy, t, t, w, None, 0.0)
            if discrepancy is not None else yy)
    dfloor = np.finfo(float).eps * max(n, 1) * base

    def fit(cols):
        full = list(cols) + ([p] if forced else [])
        GS = G[np.ix_(full, full)]
        try:
            np.linalg.cholesky(GS)
            c = np.linalg.solve(GS, Gy[full])
        except np.linalg.LinAlgError:
            c = np.linalg.lstsq(GS, Gy[full], rcond=None)[0]
        if discrepancy is None:
            r = max(yy - float(c @ Gy[full]), floor)
        else:
            cols_f = [j for j in full if j < p]
            resid = t - Xa[:, full] @ c
            contribs = (Xa[:, cols_f] * c[:len(cols_f)][None, :]
                        if discrepancy == 'scale_invariant' else None)
            b = float(c[-1]) if (forced or (full and full[-1] == p)) else 0.0
            r = max(discrepancy_value(discrepancy, resid, t, w, contribs, b),
                    dfloor)
        keep = c[:-1] if forced else c
        idx = full[:-1] if forced else full
        a = float(np.abs(keep) @ diag[idx]) / max(np.sqrt(yy), 1e-300)
        return r, a

    best_rss = np.empty(n_cols + 1)
    best_sub = np.empty(n_cols + 1, dtype=int)
    best_rss[0] = fit([])[0] if forced else max(
        base, dfloor if discrepancy is not None else floor)
    best_sub[0] = 0
    if singles:
        single_fits = np.full((2, n_cols), np.inf)
    chosen = []
    for k in range(1, n_cols + 1):
        pick = pick_adm = None
        for j in range(n_cols):
            if j in chosen:
                continue
            r, a = fit(chosen + [j])
            if singles and k == 1:
                single_fits[:, j] = r, a
            if pick is None or r < pick[0]:
                pick = (r, j)
            if amp_cap is not None and a <= amp_cap:
                if pick_adm is None or r < pick_adm[0]:
                    pick_adm = (r, j)
        r, j = pick_adm if pick_adm is not None else pick
        chosen.append(j)
        best_rss[k] = r
        best_sub[k] = sum(1 << c for c in chosen)
    d = -np.diff(np.log(best_rss))
    if singles:
        return best_rss, best_sub, d, n_cols, (single_fits[0], single_fits[1])
    return best_rss, best_sub, d, n_cols


class KneeSparsity(CompoundOperator):
    """Sparse-regression operator that selects the support by SUBSET SEARCH.

    Drop-in alternative to :class:`~epde.operators.common.sparsity.VWSRSparsity`
    (``sparsity_cls='knee'``), mirroring it step for step -- same tier-3
    super-Gram fast path, same per-trajectory fit-then-average, same relaxed
    refit semantics, same unified coefficient layout -- and differing in the
    one step that is the point: where VWSR runs an RFE loop around coordinate
    descent and reads the support off which coefficients shrank to zero, this
    enumerates the column subsets and applies one of the knee rules in
    :data:`REALIZATIONS`.

    The consequence worth stating: there is no L1 strength to set. VWSR's
    threshold is ``score_j * max_corr``, so the pruning strength moves with
    whichever instability estimator is configured; a knee rule reads only the
    shape of the RSS-versus-size curve, which is scale-free. The estimator is
    still consulted -- but only by ``'ke2_stab'``, and only to veto an
    extension.

    The opt-in ``'complexity'`` realization (:data:`BUDGET_REALIZATIONS`) is
    a different kind of rule: no elbow at all, but the best-fitting subset at
    the largest complexity within a per-equation integer budget, carried by
    the separate ``('complexity', var)`` metaparameter -- a best-subset
    candidate generator whose output the Pareto front screens. See
    :meth:`_apply_budget` and :meth:`complexity_interval`.
    """
    key = 'KneeBasedSparsity'

    #: This operator derives its support from the RSS curve and never reads
    #: the ``('sparsity', var)`` metaparameter. The degenerate interval says
    #: exactly that -- seeding is a no-op -- exactly as for VWSRSparsity.
    initial_sparsity_interval = (1.0, 1.0)

    #: The relaxed refit behind the chosen support (``subset_coefficients``)
    #: is an unpenalized weighted OLS on the physical scale, so the support
    #: decision and the fitted magnitudes are the same vector.
    fits_physical_scale = True

    #: Above this many columns (features + intercept) the exhaustive
    #: enumeration is replaced by :func:`greedy_chain`. 12 columns is 4096
    #: subsets of at most 12x12; live equations carry at most
    #: ``equation_terms_max_number`` terms, so the fallback is a safety valve.
    max_exhaustive_columns = 12

    _realization = 'knee_ext2_amp'
    _fit_curve = 'discrepancy'

    @property
    def fit_curve(self):
        """What the knee's elbow is read off: ``'discrepancy'`` or ``'rss'``.

        ``'discrepancy'`` (the default) scores every subset with the
        ``Discrepancy`` filler's own option -- the SAME quantity the Pareto
        axis reads, resolved from ``objectives.discrepancy_metric``. The knee
        then measures the curve the search is actually optimising rather than
        a Gram-space by-product of the fit; with the default WAPE these are
        genuinely different rules, one an L1 curve and the other a weighted
        L2 one.

        ``'rss'`` keeps the weighted residual sum of squares. It is cheaper --
        no residual vector is materialised, so the cost stays O(k^3) per
        subset instead of O(n*k) -- and it is what the 19-library study is
        calibrated on, so it is retained for comparison rather than deleted.
        """
        return self._fit_curve

    @fit_curve.setter
    def fit_curve(self, value):
        if value not in ('discrepancy', 'rss'):
            raise ValueError(
                "fit_curve must be 'discrepancy' or 'rss', got "
                '{0!r}.'.format(value))
        self._fit_curve = value

    @property
    def realization(self):
        """Which rule of :data:`REALIZATIONS` or :data:`BUDGET_REALIZATIONS`
        selects the support.

        A property so an unknown value fails LOUD at configuration time.
        ``search_config.validate_sparsity_kwargs`` checks kwarg NAMES against
        the operator's attributes but not their values, so a typo'd
        realization would otherwise sit on the operator silently selecting
        the default -- the failure mode the config layer exists to remove.
        """
        return self._realization

    @realization.setter
    def realization(self, value):
        if value not in REALIZATIONS + BUDGET_REALIZATIONS:
            raise ValueError(
                'Unknown knee realization {0!r}; expected one of {1}.'.format(
                    value, list(REALIZATIONS + BUDGET_REALIZATIONS)))
        self._realization = value

    @classmethod
    def complexity_interval(cls, sparsity_kwargs, terms_max, factors_max):
        """The range of the ``('complexity', var)`` budget, or ``None``.

        ``None`` unless ``sparsity_kwargs`` selects a realization of
        :data:`BUDGET_REALIZATIONS` -- and ``None`` is what keeps a default
        run's metaparameters, genes and random stream exactly as they were:
        the population constructors add the budget only when handed an
        interval. Under a budget it is ``(1, (terms_max - 1) * F + 1)``, the
        largest complexity one equation can carry: ``terms_max`` counts the
        target, which costs nothing, every other term holds at most ``F``
        factors, and the intercept costs 1. ``F`` is ``factors_max`` itself
        or, for the ``{'factors_num': [...], 'probas': [...]}`` form, its
        largest entry -- the cap ``Term.randomize`` draws under.

        Asked by ``EpdeSearch.fit`` with the RESOLVED term/factor maxima, which
        are only known at fit time, and the realization is an INSTANCE setting
        -- so a class attribute could carry neither. Being a classmethod it is
        callable, so ``search_config.sparsity_settings`` does not offer it as a
        setting.
        """
        realization = (sparsity_kwargs or {}).get('realization',
                                                  cls._realization)
        if realization not in BUDGET_REALIZATIONS:
            return None
        if terms_max is None or factors_max is None:
            raise ValueError(
                'The {0!r} realization derives its budget range from the '
                "fit's term and factor maxima; pass terms_max and "
                'factors_max.'.format(realization))
        if isinstance(factors_max, dict):
            factors_max = max(factors_max['factors_num'])
        return (1, (int(terms_max) - 1) * int(factors_max) + 1)

    @_loop_stats.timed('KneeSparsity.apply')
    def apply(self, objective, arguments: dict):
        self_args, subop_args = self.parse_suboperator_args(arguments=arguments)

        objectives_cfg = active_config().objectives
        metric = objectives_cfg.instability_metric
        gram_mode = objectives_cfg.gram_mode
        amp_cap = active_config().search_space.rps_amplification_cap
        realization = self.realization
        # The curve the elbow is read off. Resolved from the search config, so
        # the knee and the discrepancy Pareto axis cannot disagree about what
        # "fits better" means -- the same rule that makes ``ke2_stab`` follow
        # ``instability_metric``. A solver-only option has no subset-level
        # form and ``subset_table`` refuses it by name.
        curve = (objectives_cfg.discrepancy_metric
                 if self.fit_curve == 'discrepancy' else None)

        # Tier 3 fast path, identical to VWSRSparsity.apply: when the upstream
        # EqRPS term-sweep has precomputed a super-Gram (and the cached Z over
        # all terms), derive target / features plus the per-target Gram view by
        # slicing instead of re-evaluating the terms. Everything is keyed by
        # trajectory ID.
        gram_super = getattr(objective, '_gram_super', None)
        if gram_super is not None:
            Z = gram_super['Z']
            tgt = objective.target_idx
            target = {key: Z_key[:, tgt] for key, Z_key in Z.items()}
            features = {key: Z_key[:, [i for i in range(Z_key.shape[1])
                                       if i != tgt]]
                        for key, Z_key in Z.items()}
            super_mode = gram_super.get('mode')
            if super_mode == 'vcoef':
                gram_setups = VaryingCoefSetup.from_full(gram_super, tgt)
            elif super_mode == 'axis':
                gram_setups = GramSetup.from_full(gram_super, tgt)
            else:
                gram_setups = {}      # basis-free metric: Z only
        else:
            target, features = objective.evaluate()
            gram_setups = {}

        n_slots = len(objective.structure)
        if features is None:
            # No non-target term to select from: the same "treat it as empty"
            # verdict LASSOSparsity reaches for degenerate features, which the
            # fitness host turns into a declined candidate.
            self._store(objective, np.zeros(n_slots), None, None)
            return

        assert isinstance(target, dict) and isinstance(features, dict), (
            'Unexpected behavior: target and features, obtained from '
            f'obj.evaluate have to be dicts, instead got: {type(target)} and '
            f'{type(features)}.')

        gfuncs = global_var.samples_manager.gFunc('dm')
        grid_shapes = global_var.samples_manager.inner_shapes

        if realization in BUDGET_REALIZATIONS:
            # No chain to walk: the selection needs every trajectory's table
            # before it can choose one support for all of them, so it cannot
            # ride the per-trajectory loop below.
            self._apply_budget(objective, target, features, gram_setups,
                               gfuncs, grid_shapes, curve=curve,
                               amp_cap=amp_cap, metric=metric,
                               gram_mode=gram_mode)
            return

        sampled_full_coefs = []
        sampled_sw_weights = {}
        sampled_vc_scores = {}

        for key in target.keys():
            X = np.asarray(features[key], dtype=float)
            y = np.asarray(target[key], dtype=float).reshape(-1)
            sw = np.asarray(gfuncs[key], dtype=float).reshape(-1)
            grid_shape = grid_shapes[key]
            n_features = X.shape[1]

            if not (np.all(np.isfinite(X)) and np.all(np.isfinite(y))):
                sampled_full_coefs.append(np.zeros(n_features + 1))
                sampled_sw_weights[key] = None
                sampled_vc_scores[key] = None
                continue

            gram_setup = self._trajectory_gram(
                gram_setups.get(key), X, y, sw, grid_shape, gram_mode,
                objective.main_var_to_explain, key)

            stability = None
            if realization == 'ke2_stab':
                stability = self._stability_callback(
                    metric, X, y, sw, grid_shape, n_features, gram_setup)

            n_cols = n_features + 1
            if n_cols <= self.max_exhaustive_columns:
                rss, amp, n_cols = subset_table(X, y, sw, discrepancy=curve)
                instab = (instability_table(metric, X, y, sw, grid_shape,
                                            n_cols, gram_setup=gram_setup)
                          if realization == 'stab_chain' else None)
                best_rss, best_sub, d = build_chain(
                    realization, rss, amp, n_cols, amp_cap=amp_cap,
                    instab=instab, accumulate=curve is not None)
                _loop_stats.record('KneeSparsity.exhaustive', 1, 1)
            else:
                # The greedy fallback builds no subset table, so there is
                # nothing to narrow by instability: 'stab_chain' degrades to
                # the plain chain above the column cap. Recorded, not silent.
                if realization == 'stab_chain':
                    _loop_stats.record('KneeSparsity.stab_chain_degraded', 1, 1)
                best_rss, best_sub, d, n_cols = greedy_chain(
                    X, y, sw, amp_cap=amp_cap, discrepancy=curve)
                _loop_stats.record('KneeSparsity.greedy', 1, 1)
            subset, vetoed = select_from_chain(realization, best_rss, best_sub,
                                               d, n_cols, stability=stability)
            if vetoed is not None:
                _loop_stats.record('KneeSparsity.stability_veto', 1, 1)

            support = mask_of(subset, n_cols)
            # The relaxed refit: the rule picked the support, an unpenalized
            # weighted OLS supplies the magnitudes -- VWSR's convention, and
            # the reason no shrinkage bias reaches the coefficients.
            full = subset_coefficients(X, y, sw, support)
            sampled_full_coefs.append(full)

            sampled_vc_scores[key], sampled_sw_weights[key] = \
                self._support_scores(gram_setup, full != 0, metric)

        weights = np.mean(np.stack(sampled_full_coefs, axis=1), axis=1)
        self._store(objective, weights, sampled_sw_weights, sampled_vc_scores)

    def _apply_budget(self, objective, target, features, gram_setups, gfuncs,
                      grid_shapes, *, curve, amp_cap, metric, gram_mode):
        """The ``'complexity'`` realization: the best subset within a budget.

        Every subset of the equation's non-target terms plus the selectable
        intercept is priced at its atom count (:func:`column_costs`), the
        equation's ``('complexity', var)`` budget is read through
        :func:`complexity_budget` (no key, or a non-positive / non-finite
        value, means no budget), and :func:`select_by_budget` returns the
        lowest-curve subset at the LARGEST cost the budget reaches -- the
        candidate a best-subset study would enumerate at that level. Whether
        that level was worth its terms is the Pareto front's question, not
        this operator's: no elbow, no instability, one statistic.

        MULTI-TRAJECTORY. The chain realizations select per trajectory and
        average, so their support is the UNION of the per-trajectory ones --
        which can blow straight through a budget. Here one subset is chosen
        for every trajectory, on the MEAN of the per-trajectory curve and
        amplification tables, and each trajectory is then refit on that common
        support before the full vectors are averaged as usual. The mean of
        per-trajectory curves, each scored at that trajectory's own fit, is
        what can be tabulated per subset; the filler scores the AVERAGED
        coefficients, so the two agree exactly for one trajectory and
        approximately for several.

        Above ``max_exhaustive_columns`` the table is replaced by one forward
        chain over the row-stacked finite trajectories (LASSOSparsity's
        multisample convention) and the pick is its largest prefix within the
        budget (:func:`budget_prefix`), or the best affordable single column
        of the same stacked data when even the first prefix is too dear -- a
        safety valve, as for the chains.
        """
        col_costs = column_costs(objective)
        n_cols = col_costs.size
        budget = resolve_budget(complexity_budget(objective),
                                int(col_costs.sum()))

        samples = {}
        for key in target.keys():
            X = np.asarray(features[key], dtype=float)
            y = np.asarray(target[key], dtype=float).reshape(-1)
            sw = np.asarray(gfuncs[key], dtype=float).reshape(-1)
            if X.shape[1] + 1 != n_cols:
                # The budget would be priced on the wrong columns.
                raise ValueError(
                    'Feature matrix of trajectory {0!r} has {1} columns but '
                    'the equation has {2} non-target terms.'.format(
                        key, X.shape[1], n_cols - 1))
            finite = bool(np.all(np.isfinite(X)) and np.all(np.isfinite(y)))
            samples[key] = (X, y, sw, finite)

        finite_keys = [key for key, sample in samples.items() if sample[3]]
        if not finite_keys:
            self._store(objective, np.zeros(n_cols), None, None)
            return

        if n_cols <= self.max_exhaustive_columns:
            tables = [subset_table(*samples[key][:3], discrepancy=curve)
                      for key in finite_keys]
            labels = [_term_tie_label(term)
                      for idx, term in enumerate(objective.structure)
                      if idx != objective.target_idx] + [INTERCEPT_TIE_LABEL]
            subset = select_by_budget(
                np.mean([table[0] for table in tables], axis=0),
                np.mean([table[1] for table in tables], axis=0),
                subset_costs(col_costs), budget, n_cols, amp_cap=amp_cap,
                labels=labels)
            _loop_stats.record('KneeSparsity.budget_exhaustive', 1, 1)
        else:
            # ``singles``: the one-column fits of the first forward step, on
            # the same stacked data, for budget_prefix's fallback.
            _, best_sub, _, _, (single_curve, _) = greedy_chain(
                np.vstack([samples[key][0] for key in finite_keys]),
                np.concatenate([samples[key][1] for key in finite_keys]),
                np.concatenate([samples[key][2] for key in finite_keys]),
                amp_cap=amp_cap, discrepancy=curve, singles=True)
            subset = budget_prefix(best_sub, col_costs, budget, single_curve)
            _loop_stats.record('KneeSparsity.budget_greedy', 1, 1)

        support = mask_of(subset, n_cols)
        if col_costs[support].sum() > budget:
            # Over budget only when nothing affordable was eligible: on the
            # exhaustive path no amplification-admissible feature subset (no
            # feature subset at all, when none is admissible) costs <= budget,
            # on the greedy path no single feature column does. Both then take
            # the cheapest feature subset (select_by_budget, budget_prefix) --
            # recorded, not silent.
            _loop_stats.record('KneeSparsity.budget_exceeded', 1, 1)

        sampled_full_coefs = []
        sampled_sw_weights = {}
        sampled_vc_scores = {}
        for key, (X, y, sw, finite) in samples.items():
            if not finite:
                sampled_full_coefs.append(np.zeros(n_cols))
                sampled_sw_weights[key] = None
                sampled_vc_scores[key] = None
                continue
            gram_setup = self._trajectory_gram(
                gram_setups.get(key), X, y, sw, grid_shapes[key], gram_mode,
                objective.main_var_to_explain, key)
            full = subset_coefficients(X, y, sw, support)
            sampled_full_coefs.append(full)
            sampled_vc_scores[key], sampled_sw_weights[key] = \
                self._support_scores(gram_setup, full != 0, metric)

        weights = np.mean(np.stack(sampled_full_coefs, axis=1), axis=1)
        self._store(objective, weights, sampled_sw_weights, sampled_vc_scores)

    @staticmethod
    def _trajectory_gram(gram_setup, X, y, sw, grid_shape, gram_mode,
                         main_var, key):
        """One trajectory's Gram view: the super-Gram's slice when the tier-3
        path supplied one, else built here for the Gram-coupled metrics, else
        ``None`` (a basis-free metric needs none)."""
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

    @staticmethod
    def _support_scores(gram_setup, active, metric):
        """``(vc_score, sw_weights)`` of one trajectory's converged support --
        the two caches the Instability filler reads, in VWSR's conventions."""
        vc_score = (gram_setup.score(active)
                    if getattr(gram_setup, 'is_vcoef', False) and active.any()
                    else None)
        sw_weights = (gram_setup.solve(active)
                      if metric == 'cv' and gram_setup is not None
                      and active.any()
                      else None)
        return vc_score, sw_weights

    @staticmethod
    def _stability_callback(metric, X, y, sw, grid_shape, n_features,
                            gram_setup):
        """``mask -> per-selected-column instability``, for the extension veto.

        Routed through ``sparsity.instability_scores`` rather than picking an
        estimator here: the veto must not disagree with the Instability Pareto
        axis about what makes a term unstable. That is also why ``'ke2_stab'``
        has no ``_chi2`` / ``_vc`` name variants -- it scores with whatever
        ``objectives.instability_metric`` is set to, so a run at
        ``instability_metric='vcoef'`` gets the vcoef-vetoed rule for free.

        The intercept participates like any other column when the mask
        selects it, which is consistent with the rest of this operator
        treating it as an ordinary selectable column.
        """
        def stability(mask):
            weights = (gram_setup.solve(mask) if metric == 'cv' else None)
            return instability_scores(metric, X, y, sw, grid_shape, mask,
                                      n_features, gram_setup=gram_setup,
                                      cv_reducer=cv_scores, weights=weights)
        return stability

    @staticmethod
    def _store(objective, weights, sw_weights, vc_scores):
        """Write the fit onto the equation, in VWSRSparsity's conventions."""
        # ``subset_coefficients`` already fits on the physical scale, so the
        # final magnitudes ARE the internal ones -- declared once as
        # ``fits_physical_scale = True`` and acted on by
        # ``LinRegBasedCoeffsEquation``, which promotes this vector rather than
        # refitting and stays the sole writer of ``weights_final_evald``.
        objective.weights_internal = weights
        objective.weights_internal_evald = True
        # Assigned UNCONDITIONALLY, including the None case: EqRightPartSelector
        # copies both out of the equation during its term sweep, so a skipped
        # assignment would leave the previous candidate target's value behind.
        objective._cached_sw_weights = (
            None if not sw_weights or all(v is None for v in sw_weights.values())
            else sw_weights)
        objective._cached_vc_score = (
            None if not vc_scores or all(v is None for v in vc_scores.values())
            else vc_scores)

    def use_default_tags(self):
        self._tags = {'sparsity', 'gene level', 'no suboperators', 'inplace'}
