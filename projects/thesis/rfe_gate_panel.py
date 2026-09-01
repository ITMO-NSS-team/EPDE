"""Offline comparison of RFE kill rules for the VWSR sparsity operator.

``PhysicsInformedLasso.fit`` already runs an RFE outer loop; its kill
criterion is a coordinate-descent soft-threshold ``|rho_j| <= score_j *
max_corr`` -- a SIGNAL-scale anchor doing a noise-scale job (weak-term
masking). This panel asks whether a statistics/significance-based gate
recovers the true support better, BEFORE anything is wired into the live
pipeline (`epde/` is imported read-only; zero production changes).

Per system (and per variable), build a truth + junk LIBRARY problem: the
seeded truth equation inflated with up to ``--k-junk`` pool-legal junk terms
(higher powers, extra derivatives, grid/trend tokens, cross-variable bait --
the families the live junk fronts actually produce). Each rule then has to
recover the true support:

    max_corr      today's production rule: PhysicsInformedLasso.fit as-is.
    tstat_vcoef   greedy gate on VaryingCoefSetup.score (NC/gamma_0^2),
                  kill the worst column iff score >= 1/(2 ln p)
                  (Donoho universal threshold; p = active cols incl.
                  intercept; the last column is never killed).
    tstat_axis    same gate on the axis backup statistic get_cv
                  (var/mu^2 across sliding windows).
    tstat_chi2    same gate on the Nyblom-Hansen chi2 score paths --
                  chi2 is NON-pivotal, so the 1/(2 ln p) calibration is
                  heuristic here; reported for comparison only.
    chi2 sweep    the real chi2 result: recovery as a function of an
                  absolute threshold over a log grid -- does a
                  cross-system constant exist?

    Statistical per-term thresholds (the "replace soft thresholding with a
    significance rule" menu), all on the badness = keep-when-small scale:

    tstat_ols     classical weighted-OLS backward elimination, badness =
                  1/t_j^2, universal bar (kill iff |t| < sqrt(2 ln p)).
    tstat_hac     same with a PER-TERM Newey-West inflation F_j from each
                  term's own score series x_j*r (t_hac = t/sqrt(F_j)) --
                  the sandwich-diagonal correction classical t-tests need
                  on autocorrelated FD-error residuals; a residual-only
                  common factor cannot even change the kill order.
    perm_a05      circular-shift permutation null per term: p-value of the
                  observed partial alignment vs shifted copies of the SAME
                  column (dependence-respecting, no noise model);
                  kill iff p > 0.05 (alpha bars at p-grid midpoints).
    boot_a05      non-overlapping block bootstrap sign-instability p-value
                  in Gram space (block resample -> re-solve OLS -> sign
                  flip rate); kill iff p > 0.05.
    energy_s01    ENERGY yardstick instead of std (user directive): kill
                  the smallest coef*term RMS-share while share < 1%
                  (term_contribution.py's L2 share; no sigma anywhere,
                  so immune to the clean-data sigma-hat collapse).

    The panel compares the APPROACHES PURE (user directive: "use
    stability for stability and immateriality for immateriality",
    "compare approaches, not use both"): stability-based elimination =
    tstat_vcoef / tstat_chi2 (kill most unstable, recalc, stop at the
    bar); immateriality-based elimination = energy (kill least
    material, recalc, stop at the floor). No fused rules.

    Optimal-floor identification (user request), three levels:

    energy_gap    truth-free PER-EQUATION floor: the geometric midpoint
                  of the largest log-gap in the round-1 share spectrum
                  (junk cluster vs material cluster); same energy kill
                  order, stop at that floor. Parameter-free.
    energy_jump   truth-free stop on the same kill order: keep the
                  model just BEFORE the most damaging kill (largest
                  jump in log weighted RSS along the elimination path).
                  Order by immateriality, stop by fit damage.
    exact opt     benchmark-level: the kill order is floor-independent,
                  so floor -> outcome is a step function with recorded
                  breakpoints; the md enumerates ALL distinct floors
                  exactly (no grid) and reports the optimal INTERVAL
                  (ground-truth-dependent -- calibration, not deploy).
    + post-hoc bar/alpha sweeps for each (kill order is
      threshold-independent, so sweeps are free).

Key economy: for a fixed statistic the greedy kill SEQUENCE is
threshold-independent (the threshold only decides where to STOP), so each
statistic needs ONE full elimination trajectory per library; every stopping
rule is evaluated post-hoc on the recorded rounds.

Statistic calls use ``fit_intercept=True`` -- mimicking the live keep-rule
context (the intercept is a killable column under vcoef/axis and counts
toward p), unlike the objective panel's ``fit_intercept=False``. chi2 drops
the intercept from its score vector, so under the chi2 gate the intercept is
fitted but never killable (recorded, not judged).

    python projects/thesis/rfe_gate_panel.py                  # all systems
    python projects/thesis/rfe_gate_panel.py --systems ode lv kdv
    python projects/thesis/rfe_gate_panel.py --noise-level 1.0
    python projects/thesis/rfe_gate_panel.py --summarize      # from saved JSON
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

_THIS = os.path.dirname(os.path.abspath(__file__))
_HADAMARD = os.path.join(_THIS, 'hadamard')
_ROOT = os.path.abspath(os.path.join(_THIS, '..', '..'))
for _p in (_ROOT, _THIS, _HADAMARD):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
import yaml  # noqa: E402
import epde.globals as gv  # noqa: E402
from epde.interface.equation_translator import translate_equation  # noqa: E402
import epde.operators.common.sparsity as sparsity_module  # noqa: E402
from epde.operators.common.sparsity import PhysicsInformedLasso  # noqa: E402
from epde.operators.common.stability import (  # noqa: E402
    GramSetup, VaryingCoefSetup,
)
from epde.operators.common.survival import (  # noqa: E402
    chi2_scores, block_gram_partition, _solve_gram, _DEFAULT_N_BLOCKS,
)
from kdv_sindy_test import build_pool_only, _normalize_grid_labels  # noqa: E402
from thesis_runner import load_config, pipeline_settings, _set_seeds  # noqa: E402
from objective_experiment import ALL_SYSTEMS, _as_eqdict  # noqa: E402
from instability_panel import _wrap_noisy  # noqa: E402

RULES = ('max_corr', 'max_corr_raw', 'tstat_vcoef',
         'tstat_axis', 'tstat_chi2', 'tstat_ols', 'tstat_hac', 'perm_a05',
         'boot_a05', 'energy_s01', 'energy_gap', 'energy_jump')
TAUS = np.logspace(-4, 8, 49)
ALPHAS = (0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5)
T_BARS = (1.0, 1.5, 1.96, 2.5, 3.0, 4.0, 6.0)
SHARE_MINS = (0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2)
OUT = os.path.join(_THIS, 'rfe_gate_panel.json')
MD_OUT = os.path.join(_THIS, 'rfe_gate_panel.md')


def _tau_universal(p: int) -> float:
    """Donoho universal threshold on the inverse-squared-t scale:
    tau(p) = 1/(2 ln p), kill iff score >= tau. DECREASES with p (among more
    candidates the max of p null statistics grows like sqrt(2 ln p), so a
    term needs a smaller score to be distinguishable). tau(1) = +inf: the
    last surviving column is never gate-killed."""
    if p <= 1:
        return np.inf
    return 1.0 / (2.0 * np.log(p))


# --------------------------------------------------------------------------
# Library assembly: truth + junk terms translated onto the system's pool
# --------------------------------------------------------------------------
def _junk_candidates(var, variable_names, n_axes):
    """Ordered, deterministic junk-term menu for ``var``'s equation.
    Families mirror what live junk fronts actually contain: extra
    derivatives (incl. cross-variable -- the lv parasite bait), grid/trend
    tokens, higher powers, cross products. Pool-illegal or duplicate
    candidates are skipped downstream, so over-generation is harmless."""
    others = [v for v in variable_names if v != var]
    cands = [f'{var}{{power: 2.0}}']
    for d in range(n_axes):
        for u2 in variable_names:
            cands.append(f'd{u2}/dx{d}{{power: 1.0}}')
    cands.append('x_0{power: 1.0, dim: 0.0}')
    for u2 in others:
        cands.append(f'{u2}{{power: 1.0}}')
    for d in range(n_axes):
        for u2 in variable_names:
            cands.append(f'd^2{u2}/dx{d}^2{{power: 1.0}}')
    cands.append(f'{var}{{power: 3.0}}')
    cands.append('x_0{power: 2.0, dim: 0.0}')
    cands.append(f'{var}{{power: 1.0}} * d{var}/dx{n_axes - 1}{{power: 1.0}}')
    for u2 in others:
        cands.append(f'{var}{{power: 1.0}} * {u2}{{power: 1.0}}')
    return cands


def _with_eq(truth, var, new_eq, variable_names):
    if len(variable_names) == 1:
        return new_eq
    d = dict(truth)
    d[var] = new_eq
    return d


def _translate(eqd, variable_names, pool):
    return translate_equation(_normalize_grid_labels(eqd), pool,
                              all_vars=list(variable_names))


def _term_labels(soeq, var):
    eq = soeq.vals[var]
    return [term.name for i, term in enumerate(eq.structure)
            if i != eq.target_idx]


def _target_label(soeq, var):
    eq = soeq.vals[var]
    return eq.structure[eq.target_idx].name


def _build_eq_string(base_str, junk):
    """Truth LHS + '1.0 * junk' terms (before any trailing '+ 0.0')."""
    lhs, rhs = base_str.split(' = ', 1)
    tail = ''
    if lhs.endswith(' + 0.0'):
        lhs, tail = lhs[:-len(' + 0.0')], ' + 0.0'
    return (lhs + ''.join(f' + 1.0 * {c}' for c in junk) + tail
            + ' = ' + rhs)


def _inflate_truth(truth, var, variable_names, pool, n_axes, k_junk):
    """Translate truth + up to ``k_junk`` junk terms onto the pool.

    Candidates are accepted one at a time: a candidate is kept iff the
    inflated system still translates, term names stay unique, all truth
    terms survive by name, and the target term is unmoved. Returns
    ``(soeq, labels, is_true, junk_accepted)`` for ``var``'s equation.
    """
    base_str = truth if len(variable_names) == 1 else truth[var]
    soeq_truth = _translate(truth, variable_names, pool)
    true_names = set(_term_labels(soeq_truth, var))
    target_name = _target_label(soeq_truth, var)

    accepted = []
    for cand in _junk_candidates(var, variable_names, n_axes):
        if len(accepted) >= k_junk:
            break
        trial_str = _build_eq_string(base_str, accepted + [cand])
        try:
            soeq = _translate(_with_eq(truth, var, trial_str, variable_names),
                              variable_names, pool)
        except Exception:
            continue
        names = _term_labels(soeq, var)
        if target_name in names:
            # The candidate IS this equation's target: OLS would fit
            # target = 1.0 * copy exactly (residual -> 0, every statistic
            # degenerates). The live pipeline bans a target reappearing as
            # a standalone term (RPS target-term uniqueness), so such a
            # library is unrealistic bait. Cross-EQUATION targets (e.g.
            # dv/dx0 in lv's u-equation) remain legit junk.
            continue
        if len(set(names)) != len(names):
            continue                      # duplicate term label
        if not true_names.issubset(set(names)):
            continue                      # a truth term got clobbered
        if len(names) != len(true_names) + len(accepted) + 1:
            continue                      # candidate collapsed into another
        if _target_label(soeq, var) != target_name:
            continue                      # translator moved the target
        accepted.append(cand)

    final_str = _build_eq_string(base_str, accepted)
    soeq = _translate(_with_eq(truth, var, final_str, variable_names),
                      variable_names, pool)
    labels = _term_labels(soeq, var)
    is_true = [lbl in true_names for lbl in labels]
    return soeq, labels, is_true, accepted


def _features(soeq, var):
    """(features, target) for ``var``'s equation over the WIDE column set --
    the live keep-rule context (sparsity takes ``evaluate()``, every
    non-target term), unlike the objective panel's raw columns."""
    eq = soeq.vals[var]
    eq.main_var_to_explain = var
    eq.weights_internal = np.append(np.ones(len(eq.structure) - 1), 0.0)
    eq.weights_internal_evald = True
    eq.weights_final_evald = True
    t, f = eq.evaluate()
    f = np.asarray(f, dtype=float)
    if f.ndim == 1:
        f = f[:, None]
    return f, np.asarray(t, dtype=float).reshape(-1)


# --------------------------------------------------------------------------
# Greedy elimination trajectories + post-hoc stopping rules
# --------------------------------------------------------------------------
def _greedy_trajectory(n_cols, score_fn):
    """Kill-worst-first all the way down to one column, recording each
    round. The kill sequence is threshold-independent; stopping rules are
    applied post-hoc by ``_apply_stop``. ``score_fn(mask)`` returns scores
    aligned to the active columns of the boolean ``mask``."""
    mask = np.ones(n_cols, dtype=bool)
    rounds = []
    while mask.sum() > 1:
        scores = np.asarray(score_fn(mask), dtype=float)
        active = np.flatnonzero(mask)
        worst = int(np.argmax(scores))
        rounds.append({'active': [int(i) for i in active],
                       'scores': [float(s) for s in scores],
                       'worst_col': int(active[worst]),
                       'max_score': float(scores[worst])})
        mask = mask.copy()
        mask[active[worst]] = False
    return rounds


def _apply_stop(rounds, n_cols, tau_fn):
    """Walk the recorded rounds; stop at the first round whose worst score
    is under the bar. Returns the kept column set."""
    if not rounds:
        return set(range(n_cols))
    for r in rounds:
        if r['max_score'] < tau_fn(len(r['active'])):
            return set(r['active'])
    last = rounds[-1]
    return set(last['active']) - {last['worst_col']}


def _verdict_cols(kept_feature_cols, is_true):
    """FN/FP over FEATURE columns (intercept judged separately)."""
    fn = [i for i, tr in enumerate(is_true)
          if tr and i not in kept_feature_cols]
    fp = [i for i, tr in enumerate(is_true)
          if not tr and i in kept_feature_cols]
    return {'fn': len(fn), 'fp': len(fp),
            'fn_cols': fn, 'fp_cols': fp,
            'exact': not fn and not fp,
            'kept': sorted(int(i) for i in kept_feature_cols)}


def _round1_margin(rounds, is_true, feature_cols_only=True):
    """min(junk score) / max(true score) at round 1 over feature columns;
    > 1 means every junk term already scores worse than every true term."""
    if not rounds:
        return None
    r0 = rounds[0]
    true_s, junk_s = [], []
    for col, s in zip(r0['active'], r0['scores']):
        if feature_cols_only and col >= len(is_true):
            continue                       # intercept column
        (true_s if is_true[col] else junk_s).append(s)
    if not true_s or not junk_s:
        return None
    mx_true = max(true_s)
    mn_junk = min(junk_s)
    if mx_true <= 0:
        return float('inf') if mn_junk > 0 else None
    return float(mn_junk / mx_true)


# --------------------------------------------------------------------------
# Statistical per-term significance statistics (badness = keep-when-small).
# Every one is expressed on the "badness" scale so the SAME trajectory /
# post-hoc-threshold machinery applies: t-tests as 1/t^2 (the universal bar
# 1/(2 ln p) is then literally the Bonferroni-flavor |t| >= sqrt(2 ln p)),
# permutation and bootstrap rules as p-values (bar = alpha).
# --------------------------------------------------------------------------
def _ols_badness(Xa, y, w):
    """Classical weighted-OLS inverse-squared t-statistics, 1/t_j^2.

    sigma^2 from the active fit's own residual with a dof correction --
    the textbook backward-elimination statistic. Invalid as a calibrated
    test here (FD-error residuals are autocorrelated, not iid), but the
    honest baseline every correction is judged against."""
    wy = w * y

    def fn(mask):
        X = Xa[:, mask]
        G = X.T @ (w[:, None] * X)
        gy = X.T @ wy
        c = _solve_gram(G, gy)
        r = y - X @ c
        dof = max(float(w.sum()) - X.shape[1], 1.0)
        s2 = float(r @ (w * r)) / dof
        null_mask = None
        try:
            Ginv = np.linalg.inv(G)
        except np.linalg.LinAlgError:
            # Exactly singular Gram: pinv variances are SMALL in the
            # identified subspace, which would grade exactly-dependent
            # columns as maximally significant (adversarial-review
            # finding). Classical inference says the opposite -- an
            # unidentifiable coefficient is untestable -> maximally bad.
            _, s, Vt = np.linalg.svd(G)
            tol = s.max() * G.shape[0] * np.finfo(float).eps
            dead = s < tol
            null_mask = ((Vt[dead] ** 2).sum(axis=0) > 1e-8
                         if np.any(dead) else None)
            Ginv = np.linalg.pinv(G)
        var = np.maximum(s2 * np.diag(Ginv), 1e-300)
        t2 = c ** 2 / var
        bad = 1.0 / np.maximum(t2, 1e-300)
        if null_mask is not None:
            bad[null_mask] = 1e300
        return bad
    return fn


def _hac_factor(r, gshape):
    """Newey-West-style variance inflation from the residual field's own
    autocorrelation: per axis, Bartlett-weighted F_d = 1 + 2 sum w_l rho_l
    (clipped >= 1 -- inflation only), F = prod F_d. The crude standard
    shortcut: one common factor scales every term's variance."""
    field = np.asarray(r, dtype=float).reshape(gshape)
    F = 1.0
    for d, n_d in enumerate(gshape):
        if n_d < 8:
            continue
        x = np.moveaxis(field, d, 0).reshape(n_d, -1)
        x = x - x.mean(axis=0, keepdims=True)
        denom = float((x * x).sum())
        if denom <= 0:
            continue
        L = min(n_d // 4, 50)
        F_d = 1.0
        for lag in range(1, L + 1):
            rho = float((x[:-lag] * x[lag:]).sum()) / denom
            F_d += 2.0 * (1.0 - lag / (L + 1.0)) * rho
        F *= max(F_d, 1.0)
    return F


def _hac_badness(Xa, y, w, gshape):
    """PER-TERM HAC correction: badness_j = F_j / t_j^2 with F_j the
    Bartlett inflation of term j's own SCORE series u_ij = x_ij * r_i
    (the Newey-West sandwich diagonal, term by term).

    NOT the residual's ACF alone: a residual-only factor is a common
    scalar per round, so it provably cannot change the kill order (it
    is tstat_ols stopped deeper), and it OVER-corrects -- a white
    regressor against an AR residual needs F ~ 1, not the residual's
    F >> 1 (adversarial-review Monte Carlo: residual-only F = 14.7
    where the true factor was 1.0). The score ACF is the product
    rho_x * rho_r structure that the sandwich actually integrates."""
    base = _ols_badness(Xa, y, w)
    wy = w * y

    def fn(mask):
        bad = base(mask)
        X = Xa[:, mask]
        c = _solve_gram(X.T @ (w[:, None] * X), X.T @ wy)
        r = y - X @ c
        F = np.array([_hac_factor(X[:, j] * r, gshape)
                      for j in range(X.shape[1])])
        return bad * F
    return fn


def _perm_badness(f, y, w, gshape, n_shifts=99, seed=0, chunk=16):
    """Per-term circular-shift permutation p-values (feature cols only).

    For term j: fit the OTHER active columns (+ intercept), keep their
    residual r_O (W-orthogonal to them), and compare the observed partial
    alignment T = |x_j' W r_O| / ||x_j perp||_W against the same statistic
    for circularly shifted copies of x_j (random shift per grid axis,
    wrap). Shifting preserves the column's marginal distribution and
    autocorrelation while breaking its alignment with the data -- a
    dependence-respecting null with NO noise model, which is exactly what
    classical t-tests lack here. p_j = (1 + #{T_shift >= T_obs}) / (K+1).
    Deterministic per mask (rng reseeded each call; shifts are only
    partially common across rounds -- position-indexed, not
    column-indexed). CAVEAT (adversarial-review Monte Carlo): the
    circular null is exact under circular stationarity but
    ANTICONSERVATIVE (2-4x level inflation) on non-periodic,
    envelope-sharing smooth fields -- i.e. exactly PDE library columns --
    so these p-values are dependence-aware RANKING scores; only the
    empirical alpha sweep calibrates them."""
    n = f.shape[0]
    ones = np.ones((n, 1))
    K = n_shifts if n <= 200_000 else 49

    def fn(mask):
        rng = np.random.default_rng(seed)
        act = np.flatnonzero(mask)
        out = np.empty(act.size)
        for pos, j in enumerate(act):
            oth = [c for c in act if c != j]
            XO = np.hstack([f[:, oth], ones]) if oth else ones
            GO = XO.T @ (w[:, None] * XO)
            cO = _solve_gram(GO, XO.T @ (w * y))
            rO = w * (y - XO @ cO)          # weighted residual, W-orth to XO

            def stats(cols):
                b = XO.T @ (w[:, None] * cols)
                a = np.linalg.lstsq(GO, b, rcond=None)[0]
                perp2 = (cols * (w[:, None] * cols)).sum(axis=0) \
                    - (b * a).sum(axis=0)
                num = np.abs(cols.T @ rO)
                return num / np.sqrt(np.maximum(perp2, 1e-300))

            T0 = float(stats(f[:, j:j + 1])[0])
            field = f[:, j].reshape(gshape)
            n_ge = 0
            done = 0
            while done < K:
                m = min(chunk, K - done)
                cols = np.empty((n, m))
                for k in range(m):
                    sh = tuple(int(rng.integers(1, s)) if s > 1 else 0
                               for s in gshape)
                    cols[:, k] = np.roll(field, sh,
                                         axis=tuple(range(len(gshape)))
                                         ).ravel()
                n_ge += int(np.sum(stats(cols) >= T0))
                done += m
            out[pos] = (1.0 + n_ge) / (K + 1.0)
        return out
    return fn


def _boot_badness(f, y, w, gshape, draws=199, seed=0):
    """Non-overlapping (Carlstein) block-bootstrap sign-instability
    p-values (augmented cols).

    Blocks come from ``block_gram_partition`` (fixed disjoint grid slabs,
    so the resample respects spatial dependence); each draw resamples
    blocks with replacement IN GRAM SPACE (a weighted sum of precomputed
    block Grams -- no data touched) and re-solves the active OLS.
    badness_j = 2 * min(P(c_j > 0), P(c_j < 0)): ~1 when the
    coefficient's sign is a coin flip across resamples, ~1/draws when it
    never flips. The same seeded draw indices are reused every round
    (common random numbers). CAVEAT (adversarial review): this is a
    regime-consistency score, not a calibrated bootstrap p -- junk with a
    small but sign-CONSISTENT projection (trend/hitchhiker family) pins
    at the floor and no alpha can kill it; power is against regionally
    sign-inconsistent junk. Near-singular draws (a block resample that
    omits a localized term's support) are not guarded."""
    G_blocks, Gy_blocks = block_gram_partition(f, y, w, gshape,
                                               n_blocks=_DEFAULT_N_BLOCKS)
    B = G_blocks.shape[0]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, B, size=(draws, B))
    G_draws = G_blocks[idx].sum(axis=1)       # (draws, p+1, p+1)
    Gy_draws = Gy_blocks[idx].sum(axis=1)     # (draws, p+1)

    def fn(mask):
        act = np.flatnonzero(mask)
        coefs = np.empty((draws, act.size))
        for b in range(draws):
            coefs[b] = _solve_gram(G_draws[b][np.ix_(act, act)],
                                   Gy_draws[b][act])
        gt = (coefs > 0).mean(axis=0)
        lt = (coefs < 0).mean(axis=0)
        eq = 1.0 - gt - lt          # lstsq min-norm zeros: split evenly,
        pos = gt + 0.5 * eq         # else they deflate p (keep-bias).
        neg = lt + 0.5 * eq
        return np.maximum(2.0 * np.minimum(pos, neg), 1.0 / (draws + 1.0))
    return fn


def _energy_badness(Xa, y, w):
    """ENERGY-share materiality (user directive: divide not by std but by
    energy). badness_j = 1 / share_j with

        share_j = RMS_W(c_j * x_j) / sum_k RMS_W(c_k * x_k)

    -- the coef*term RMS-share of ``projects/thesis/term_contribution.py``
    (L2 magnitude-weighted, sums to 1 across active columns incl. the
    intercept), the statistic that separated true/spurious 94-100% on
    clean data and stayed robust under collinearity where dR^2 broke.
    No sigma anywhere: the yardstick is the model's own signal energy
    (the same denominator family as chi2's D_j), so the rule asks "does
    this term carry a material fraction of the dynamics?" instead of
    "is its coefficient larger than the noise?" -- immune to the
    clean-data sigma-hat collapse that blinds classical t-tests. Refit
    after each kill redistributes the shares (backward elimination on
    materiality). No principled bar: the share floor is swept."""
    wy = w * y
    w_sum = float(w.sum())

    def fn(mask):
        X = Xa[:, mask]
        c = _solve_gram(X.T @ (w[:, None] * X), X.T @ wy)
        contrib = X * c[None, :]
        rms = np.sqrt((w[:, None] * contrib ** 2).sum(axis=0) / w_sum)
        tot = float(rms.sum())
        share = rms / tot if tot > 0 else np.zeros_like(rms)
        return 1.0 / np.maximum(share, 1e-12)
    return fn


# --------------------------------------------------------------------------
# Per-variable evaluation
# --------------------------------------------------------------------------
def _eval_variable(f, t, w, gshape, var, is_true, taus):
    nfeat = f.shape[1]
    out = {}

    # --- R0: today's production rule --------------------------------------
    t0 = time.perf_counter()
    pil = PhysicsInformedLasso(grid_shape=gshape, main_var=var)
    pil.fit(f, t, w)
    kept0 = {int(i) for i in np.flatnonzero(pil.coef_ != 0)}
    v0 = _verdict_cols(kept0, is_true)
    v0.update({'intercept_kept': bool(pil.intercept_ != 0),
               'secs': time.perf_counter() - t0})
    out['max_corr'] = v0

    # --- R0b: production CD rule, RAW (unrescaled) chi2 kill scores -------
    # Production ('max_corr' above) now rescales the kill scores by the
    # active fit's yy/rss inside sparsity._keep_rule_scores; this control
    # column monkeypatches that wrapper back to the plain Nyblom-Hansen
    # scores, measuring what the rescale changes in support recovery.
    t0 = time.perf_counter()
    _real_keep = sparsity_module._keep_rule_scores

    def _raw_keep_scores(cols, y, sw, gs):
        return sparsity_module.chi2_scores(cols, y, sw, gs,
                                           fit_intercept=False)

    sparsity_module._keep_rule_scores = _raw_keep_scores
    try:
        pil_r = PhysicsInformedLasso(grid_shape=gshape, main_var=var)
        pil_r.fit(f, t, w)
    finally:
        sparsity_module._keep_rule_scores = _real_keep
    kept0b = {int(i) for i in np.flatnonzero(pil_r.coef_ != 0)}
    v0b = _verdict_cols(kept0b, is_true)
    v0b.update({'intercept_kept': bool(pil_r.intercept_ != 0),
                'secs': time.perf_counter() - t0})
    out['max_corr_raw'] = v0b

    # --- trajectories (one per statistic; thresholds applied post-hoc) ----
    t0 = time.perf_counter()
    vc = VaryingCoefSetup(f, t, w, gshape, main_var=var)   # fit_intercept=True
    traj_vc = _greedy_trajectory(nfeat + 1, vc.score)
    secs_vc = time.perf_counter() - t0

    t0 = time.perf_counter()
    ax = GramSetup(f, t, w, gshape)
    get_cv = PhysicsInformedLasso().get_cv
    traj_ax = _greedy_trajectory(nfeat + 1,
                                 lambda m: get_cv(ax.solve(m)))
    secs_ax = time.perf_counter() - t0

    t0 = time.perf_counter()
    traj_chi = _greedy_trajectory(
        nfeat, lambda m: chi2_scores(f[:, m], t, w, gshape,
                                     fit_intercept=True))
    secs_chi = time.perf_counter() - t0

    # --- statistical per-term significance trajectories -------------------
    Xa = np.hstack([f, np.ones((f.shape[0], 1))])

    t0 = time.perf_counter()
    traj_ols = _greedy_trajectory(nfeat + 1, _ols_badness(Xa, t, w))
    secs_ols = time.perf_counter() - t0
    t0 = time.perf_counter()
    traj_hac = _greedy_trajectory(nfeat + 1, _hac_badness(Xa, t, w, gshape))
    secs_hac = time.perf_counter() - t0
    t0 = time.perf_counter()
    traj_perm = _greedy_trajectory(nfeat, _perm_badness(f, t, w, gshape))
    secs_perm = time.perf_counter() - t0
    t0 = time.perf_counter()
    traj_boot = _greedy_trajectory(nfeat + 1, _boot_badness(f, t, w, gshape))
    secs_boot = time.perf_counter() - t0
    t0 = time.perf_counter()
    traj_en = _greedy_trajectory(nfeat + 1, _energy_badness(Xa, t, w))
    secs_en = time.perf_counter() - t0

    # Permutation p-values are discrete multiples of 1/(K_perm+1); alpha
    # bars are evaluated at grid MIDPOINTS so "kill iff p > alpha" holds
    # exactly and no bar sits on the attainable p-floor (which would
    # degenerate into eliminate-to-one; adversarial-review finding).
    K_perm = 99 if f.shape[0] <= 200_000 else 49
    half_step = 0.5 / (K_perm + 1.0)
    alpha05 = (lambda p: 0.05)
    perm05 = (lambda p: 0.05 + half_step)
    for name, traj, n_cols, secs, bar in (
            ('tstat_vcoef', traj_vc, nfeat + 1, secs_vc, _tau_universal),
            ('tstat_axis', traj_ax, nfeat + 1, secs_ax, _tau_universal),
            ('tstat_chi2', traj_chi, nfeat, secs_chi, _tau_universal),
            ('tstat_ols', traj_ols, nfeat + 1, secs_ols, _tau_universal),
            ('tstat_hac', traj_hac, nfeat + 1, secs_hac, _tau_universal),
            ('perm_a05', traj_perm, nfeat, secs_perm, perm05),
            ('boot_a05', traj_boot, nfeat + 1, secs_boot, alpha05),
            ('energy_s01', traj_en, nfeat + 1, secs_en,
             lambda p: 1.0 / 0.01)):
        kept = _apply_stop(traj, n_cols, bar)
        v = _verdict_cols({c for c in kept if c < nfeat}, is_true)
        v.update({'intercept_kept': (nfeat in kept) if n_cols > nfeat
                  else None,
                  'secs': secs})
        out[name] = v

    # --- truth-free optimal-floor identification on the energy order -----
    # (a) gap floor: the largest log-gap of the ROUND-1 share spectrum
    # separates the junk cluster from the material cluster; the floor is
    # the gap's geometric midpoint. Parameter-free, per equation.
    t0 = time.perf_counter()
    gap_floor = 0.0
    if traj_en and len(traj_en[0]['scores']) >= 2:
        shares0 = np.sort(1.0 / np.asarray(traj_en[0]['scores']))
        logs = np.log(np.maximum(shares0, 1e-300))
        gi = int(np.argmax(np.diff(logs)))
        gap_floor = float(np.exp(0.5 * (logs[gi] + logs[gi + 1])))
    kept = _apply_stop(traj_en, nfeat + 1,
                       lambda p, _b=1.0 / max(gap_floor, 1e-300): _b)
    v = _verdict_cols({c for c in kept if c < nfeat}, is_true)
    v.update({'intercept_kept': nfeat in kept, 'gap_floor': gap_floor,
              'secs': time.perf_counter() - t0})
    out['energy_gap'] = v

    # (b) residual-jump stop: walk the same kill order; keep the model
    # just BEFORE the kill that causes the largest jump in log weighted
    # RSS (junk kills barely move the fit, the first true-term kill
    # spikes it). RSS floored at the machine level so exact-fit
    # libraries degrade to keep-all (flat jumps), like chi2's floor.
    t0 = time.perf_counter()
    yy = float(t @ (w * t))

    def _rss(cols):
        X = Xa[:, sorted(cols)]
        c = _solve_gram(X.T @ (w[:, None] * X), X.T @ (w * t))
        r = t - X @ c
        return max(float(r @ (w * r)), np.finfo(float).eps * yy)

    if traj_en:
        states = [set(rd['active']) for rd in traj_en]
        states.append(set(traj_en[-1]['active'])
                      - {traj_en[-1]['worst_col']})
        jumps = np.diff(np.log([_rss(s) for s in states]))
        kept = states[int(np.argmax(jumps))] if len(jumps) else states[0]
    else:
        kept = set(range(nfeat + 1))
    v = _verdict_cols({c for c in kept if c < nfeat}, is_true)
    v.update({'intercept_kept': nfeat in kept,
              'secs': time.perf_counter() - t0})
    out['energy_jump'] = v

    # --- post-hoc threshold sweeps (kill order is threshold-independent) --
    def _sweep(traj, n_cols, grid):
        sw = {'grid': [float(x) for x in grid], 'fn': [], 'fp': [],
              'exact': []}
        for bar in grid:
            kept = _apply_stop(traj, n_cols, lambda p, _b=bar: _b)
            v = _verdict_cols({c for c in kept if c < nfeat}, is_true)
            sw['fn'].append(v['fn'])
            sw['fp'].append(v['fp'])
            sw['exact'].append(v['exact'])
        return sw

    out['chi2_sweep'] = _sweep(traj_chi, nfeat, taus)
    # taus for chi2 back-compat key names
    out['chi2_sweep']['taus'] = out['chi2_sweep']['grid']
    # t-bars enter on the badness scale 1/t^2 (bar decreasing in t).
    out['ols_sweep'] = _sweep(traj_ols, nfeat + 1,
                              [1.0 / b ** 2 for b in T_BARS])
    out['ols_sweep']['t_bars'] = list(T_BARS)
    out['hac_sweep'] = _sweep(traj_hac, nfeat + 1,
                              [1.0 / b ** 2 for b in T_BARS])
    out['hac_sweep']['t_bars'] = list(T_BARS)
    out['perm_sweep'] = _sweep(traj_perm, nfeat,
                               [a + half_step for a in ALPHAS])
    out['perm_sweep']['alphas'] = list(ALPHAS)
    out['perm_sweep']['k_shifts'] = K_perm
    out['boot_sweep'] = _sweep(traj_boot, nfeat + 1, ALPHAS)
    # badness = 1/share, so the bar for a share floor s is 1/s.
    out['energy_sweep'] = _sweep(traj_en, nfeat + 1,
                                 [1.0 / s for s in SHARE_MINS])
    out['energy_sweep']['share_mins'] = list(SHARE_MINS)

    out['margins'] = {
        'vcoef': _round1_margin(traj_vc, is_true),
        'axis': _round1_margin(traj_ax, is_true),
        'chi2': _round1_margin(traj_chi, is_true),
        'ols_t': _round1_margin(traj_ols, is_true),
        'perm': _round1_margin(traj_perm, is_true),
        'boot': _round1_margin(traj_boot, is_true),
        'energy': _round1_margin(traj_en, is_true),
    }
    out['trajectories'] = {'vcoef': traj_vc, 'axis': traj_ax,
                           'chi2': traj_chi, 'ols': traj_ols,
                           'hac': traj_hac, 'perm': traj_perm,
                           'boot': traj_boot, 'energy': traj_en}
    return out


# --------------------------------------------------------------------------
# Per-system experiment
# --------------------------------------------------------------------------
def run_system(system, noise_level=0.0, noise_seed=0, k_junk=6, taus=TAUS):
    cfg = load_config(system)
    _, _, variable_names, _ = cfg.load_data()
    variable_names = list(variable_names)

    y = yaml.safe_load(open(os.path.join(_THIS, 'configs', f'{system}.yaml'),
                            encoding='utf-8'))
    teqs = y.get('truth_equations') or []
    if not teqs:
        return {'system': system, 'error': 'no truth_equations (unknown-'
                'truth config); the panel needs a seeded truth'}
    truth = _as_eqdict(teqs, variable_names)

    cfg = load_config(system)
    if noise_level > 0:
        _wrap_noisy(cfg, noise_level, noise_seed)
    _set_seeds(0)
    # (Gram mode is derived from the instability metric now: the default
    # 'chi2' already resolves to the varying-coefficient Gram this asked for.)
    search = build_pool_only(cfg, pipeline_settings('new'))
    w = np.asarray(gv.grid_cache.g_func[gv.grid_cache.g_func_mask]).reshape(-1)
    gshape = tuple(int(n) for n in gv.grid_cache.inner_shape)
    n_axes = len(gshape)

    per_var = {}
    for var in variable_names:
        try:
            soeq, labels, is_true, junk = _inflate_truth(
                truth, var, variable_names, search.pool, n_axes, k_junk)
            f, t = _features(soeq, var)

            rec = _eval_variable(f, t, w, gshape, var, is_true, taus)
            rec.update({'labels': labels, 'is_true': is_true,
                        'junk_accepted': junk,
                        'n_true': int(sum(is_true)),
                        'n_junk': int(len(is_true) - sum(is_true))})

            # Sanity pin: R0 on the PURE-truth library must keep the truth.
            soeq_t = _translate(truth, variable_names, search.pool)
            ft, tt = _features(soeq_t, var)
            pil = PhysicsInformedLasso(grid_shape=gshape, main_var=var)
            pil.fit(ft, tt, w)
            rec['truth_only_max_corr_fn'] = int(np.sum(pil.coef_ == 0))
            per_var[var] = rec
        except Exception as exc:
            per_var[var] = {'error':
                            f'{type(exc).__name__}: {str(exc)[:120]}'}
    return {'system': system, 'variables': variable_names,
            'noise_level': noise_level, 'k_junk': k_junk,
            'per_var': per_var}


# --------------------------------------------------------------------------
# Markdown rendering
# --------------------------------------------------------------------------
def _agg_rule(rec, rule):
    """Aggregate a rule's verdict over the system's variables."""
    fn = fp = 0
    ok = True
    for var, r in rec['per_var'].items():
        if 'error' in r or rule not in r:
            return None
        fn += r[rule]['fn']
        fp += r[rule]['fp']
        ok = ok and r[rule]['exact']
    return {'fn': fn, 'fp': fp, 'exact': ok}


def render_markdown(recs):
    recs_ok = [r for r in recs if 'per_var' in r]
    nls = sorted({float(r.get('noise_level', 0.0)) for r in recs_ok})
    scen = ('clean data' if nls == [0.0] else
            'noise ' + '/'.join(f'{n:g}%' for n in nls)
            + ' of std, run.py convention')
    lines = [
        f'# RFE kill-rule panel ({scen})',
        '',
        'Support recovery on truth + junk libraries. EXACT = every true '
        'term kept AND every junk term killed (feature columns; the '
        'intercept is judged separately in the JSON). tstat_* gates kill '
        'the worst-scoring column iff score >= 1/(2 ln p) (universal '
        'threshold; heuristic calibration for the non-pivotal chi2). '
        'max_corr is the production CD rule.',
        '',
        '| system | ' + ' | '.join(RULES) + ' |',
        '|---' * (len(RULES) + 1) + '|',
    ]
    exact_counts = {r: 0 for r in RULES}
    fn_tot = {r: 0 for r in RULES}
    fp_tot = {r: 0 for r in RULES}
    for rec in recs_ok:
        cells = []
        for rule in RULES:
            a = _agg_rule(rec, rule)
            if a is None:
                cells.append('n/a')
                continue
            exact_counts[rule] += a['exact']
            fn_tot[rule] += a['fn']
            fp_tot[rule] += a['fp']
            cells.append('EXACT' if a['exact']
                         else f"FN={a['fn']},FP={a['fp']}")
        lines.append(f"| {rec['system']} | " + ' | '.join(cells) + ' |')
    n = len(recs_ok)
    lines.append('| **N exact** | ' + ' | '.join(
        f"**{exact_counts[r]}/{n}**" for r in RULES) + ' |')
    lines.append('| total FN (true killed) | ' + ' | '.join(
        str(fn_tot[r]) for r in RULES) + ' |')
    lines.append('| total FP (junk kept) | ' + ' | '.join(
        str(fp_tot[r]) for r in RULES) + ' |')

    # Truth-only sanity pin
    bad = [rec['system'] for rec in recs_ok
           if any(isinstance(r, dict) and r.get('truth_only_max_corr_fn')
                  for r in rec['per_var'].values())]
    lines += ['', f'Truth-only sanity (max_corr on the pure-truth library '
              f'kills a true term): {len(bad)}/{n} systems'
              + (f' ({", ".join(bad)})' if bad else '')]

    # Threshold-calibration curves (all post-hoc from the trajectories)
    def _sweep_section(key, title, grid_key='grid', fmt='{:.3g}',
                       label='bar', stride=1):
        pairs = []
        for rec in recs_ok:
            for var, r in rec['per_var'].items():
                if isinstance(r, dict) and key in r:
                    pairs.append(r[key])
        if not pairs:
            return []
        grid = (pairs[0][grid_key] if grid_key in pairs[0]
                else pairs[0]['grid'])
        exact = [sum(p['exact'][i] for p in pairs) for i in range(len(grid))]
        best_i = int(np.argmax(exact))
        sec = ['', f'## {title}', '',
               f'{len(pairs)} (system, variable) libraries. Best: '
               f'{label} = {fmt.format(grid[best_i])} -> '
               f'{exact[best_i]}/{len(pairs)} exact.',
               '', f'| {label} | exact | total FN | total FP |',
               '|---|---|---|---|']
        for i in range(0, len(grid), stride):
            tfn = sum(p['fn'][i] for p in pairs)
            tfp = sum(p['fp'][i] for p in pairs)
            sec.append(f'| {fmt.format(grid[i])} | {exact[i]}/{len(pairs)} '
                       f'| {tfn} | {tfp} |')
        return sec

    lines += _sweep_section('chi2_sweep', 'chi2 absolute-threshold '
                            'calibration', grid_key='taus', label='tau',
                            stride=4)
    lines += _sweep_section('ols_sweep', 'OLS t-test bar calibration '
                            '(kill iff |t| <= bar)', grid_key='t_bars',
                            fmt='{:.2f}', label='t bar')
    lines += _sweep_section('hac_sweep', 'Per-term HAC t-test bar '
                            'calibration', grid_key='t_bars',
                            fmt='{:.2f}', label='t bar')
    lines += _sweep_section('perm_sweep', 'Permutation-null alpha '
                            'calibration (kill iff p > alpha; midpoint '
                            'bars)', grid_key='alphas',
                            fmt='{:.2f}', label='alpha')
    lines += _sweep_section('boot_sweep', 'Bootstrap sign-stability alpha '
                            'calibration (kill iff p > alpha)',
                            fmt='{:.2f}', label='alpha')
    lines += _sweep_section('energy_sweep', 'Energy-share floor calibration '
                            '(kill iff share < s_min)',
                            grid_key='share_mins', fmt='{:.3f}',
                            label='s_min')

    # EXACT floor optimization: the energy kill order is floor-independent,
    # so per library floor -> outcome is a step function whose breakpoints
    # are the recorded per-round worst shares. Enumerating the geometric
    # midpoints between ALL breakpoints evaluates every distinct outcome --
    # the true optimum (an interval), no grid resolution involved.
    libs = []
    for rec in recs_ok:
        for var, r in rec['per_var'].items():
            if isinstance(r, dict) and 'trajectories' in r:
                libs.append((r['trajectories']['energy'], r['is_true']))
    if libs:
        bps = sorted({1.0 / rd['max_score'] for traj, _ in libs
                      for rd in traj if rd['max_score'] > 0})
        cands = ([bps[0] / 2.0]
                 + [float(np.sqrt(a * b)) for a, b in zip(bps, bps[1:])]
                 + [bps[-1] * 2.0]) if bps else []
        rows = []
        for fl in cands:
            tot_e = tot_fn = tot_fp = 0
            for traj, is_true in libs:
                nf = len(is_true)
                kept = _apply_stop(traj, nf + 1,
                                   lambda p, _b=1.0 / fl: _b)
                v = _verdict_cols({c for c in kept if c < nf}, is_true)
                tot_e += v['exact']
                tot_fn += v['fn']
                tot_fp += v['fp']
            rows.append((fl, tot_e, tot_fn, tot_fp))
        if rows:
            best_e = max(r[1] for r in rows)
            win = [r for r in rows if r[1] == best_e]
            lines += [
                '',
                '## Exact floor optimization (breakpoint enumeration)',
                '',
                f'{len(cands)} distinct outcomes enumerated from '
                f'{len(bps)} breakpoints across {len(libs)} libraries '
                f'(no grid). Optimum: {best_e}/{len(libs)} exact, achieved '
                f'for floors in [{win[0][0]:.4g}, {win[-1][0]:.4g}] '
                f'(FN={win[0][2]}, FP={win[0][3]} at the low end).',
            ]

    # Round-1 separability margins
    margin_keys = ('vcoef', 'axis', 'chi2', 'ols_t', 'perm', 'boot',
                   'energy')
    lines += [
        '',
        '## Round-1 separability (min junk badness / max true badness)',
        '',
        '> 1 = every junk term already scores worse than every true term '
        'before any elimination; the gate then only needs a threshold '
        'INSIDE the gap.',
        '',
        '| system | var | ' + ' | '.join(margin_keys) + ' |',
        '|---' * (len(margin_keys) + 2) + '|',
    ]
    for rec in recs_ok:
        for var, r in rec['per_var'].items():
            if not isinstance(r, dict) or 'margins' not in r:
                continue
            cells = []
            for k in margin_keys:
                m = r['margins'].get(k)
                cells.append('n/a' if m is None else f'{m:.3g}')
            lines.append(f"| {rec['system']} | {var} | "
                         + ' | '.join(cells) + ' |')

    errs = [r for r in recs if 'error' in r]
    if errs:
        lines += ['', '## Skipped systems', '']
        lines += [f"- {r['system']}: {r['error']}" for r in errs]
    return '\n'.join(lines) + '\n'


# --------------------------------------------------------------------------
def main(argv=None):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
    p = argparse.ArgumentParser()
    p.add_argument('--systems', nargs='+', default=ALL_SYSTEMS)
    p.add_argument('--out', default=OUT)
    p.add_argument('--md-out', default=MD_OUT)
    p.add_argument('--k-junk', type=int, default=6,
                   help='Max junk terms added to each truth equation.')
    p.add_argument('--noise-level', type=float, default=0.0,
                   help='Gaussian noise as %% of each field std (run.py '
                        'convention), injected before preprocessing. '
                        '0 = clean (default).')
    p.add_argument('--noise-seed', type=int, default=0)
    p.add_argument('--summarize', action='store_true',
                   help='Render markdown from an existing --out JSON.')
    args = p.parse_args(argv)

    if args.summarize:
        recs = json.load(open(args.out, encoding='utf-8'))
        md = render_markdown(recs)
        print(md)
        with open(args.md_out, 'w', encoding='utf-8') as fh:
            fh.write(md)
        return 0

    results = []
    for system in args.systems:
        t0 = time.perf_counter()
        try:
            rec = run_system(system, noise_level=args.noise_level,
                             noise_seed=args.noise_seed, k_junk=args.k_junk)
        except Exception as exc:
            rec = {'system': system,
                   'error': f'{type(exc).__name__}: {str(exc)[:160]}'}
        rec['secs'] = time.perf_counter() - t0
        results.append(rec)
        status = rec.get('error', 'ok')
        print(f'[{system}] {status} ({rec["secs"]:.1f}s)', flush=True)

    with open(args.out, 'w', encoding='utf-8') as fh:
        json.dump(results, fh, indent=1)
    md = render_markdown(results)
    with open(args.md_out, 'w', encoding='utf-8') as fh:
        fh.write(md)
    print(md)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
