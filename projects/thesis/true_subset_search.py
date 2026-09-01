"""Find the TRUE subset within a truth+junk library (per-equation, no
cross-system constants).

User goal: "each system has its own floor, it can't be constant. I want
to design an algorithm which will find the true subset within the
library." The gate-panel finding that motivates this: the optimal
energy-share floor is an interval that moves ~20x between clean and 1%
noise -- no fixed threshold is regime-robust, so the decision must come
from each equation's own structure.

The reframe: EPDE libraries are small (p <= ~11 features after
inflation; <= 7 in live equations), so the 2^p subsets can be
enumerated EXHAUSTIVELY in Gram space (one weighted Gram per library,
one k x k solve per subset -- seconds per system). Greedy-path errors
disappear entirely; the design question reduces to the SELECTION
OBJECTIVE that ranks the true subset first. Objectives compared (each
evaluated post-hoc from the same recorded subset table):

    margin      necessity-margin, parameter-free:
                M(S) = min_{j in S} log(RSS(S\\j)/RSS(S))
                     - max_{j not in S} log(RSS(S)/RSS(S+j))
                The true subset is where every member is NECESSARY
                (dropping it damages the fit) and every non-member is
                REDUNDANT (adding it barely helps). Ties -> smaller S.
    margin_amp  the same, restricted to amplification-admissible
                subsets: A(S) = sum |c_j|*||col_j||_W / ||y||_W <= cap
                (the production rps_amplification_cap = 100, the one
                pre-existing constant). Cuts the lv/lorenz
                identity-compensator subsets that fool ANY pure
                fit-based criterion: dropping a true term there is
                compensable, but only at parasite-scale coefficients.
    bic         n log(RSS/n) + k log n (argmin) -- the textbook
                baseline, expected to fail on clean data (RSS is FD
                error, not noise; prior repo finding).
    knee        best-RSS-per-size curve, k* at the drop-then-flat
                elbow (argmax d_k - d_{k+1}, d_k the log-RSS drop from
                size k-1 to k) -- the classic scree heuristic.
    knee_ext2   the workflow-designed winner: start at the knee size,
                then EXTEND while the next size's best subset nests on
                the current one, stays off the machine floor, its drop
                dominates the whole remaining tail, and is at least as
                large as the weakest accepted drop (min-accepted-drop
                yardstick -- accepts ac's weak-but-real diffusion,
                rejects same-looking junk). Deterministic, constant-free.
    knee_ext2_amp  the live form: the per-size chain is restricted to
                amplification-admissible subsets (production cap 100,
                per-size fallback when a size has none).
    stab_*      STABILITY-ONLY selection (the pure counterpart, user
                question "what about stability only?"): argmin over
                subsets of the members' aggregated instability on the
                subset's own fit -- vcoef / chi2, sum (production
                Instability convention) / max (worst member). No RSS
                quantity in the decision; ties -> smaller subset.
    ke2_stab_*  the DESIGNED ALGORITHM (user: "find true subset within
                library but test stability"): the knee_ext2_amp chain
                finds the subset; each EXTENSION term beyond the elbow
                must additionally pass a stability test within the
                extended fit -- no more unstable than the least stable
                accepted member (self-calibrated, no constants). The
                core is never stability-tested. chi2 / vcoef variants.

Scoring vs ground truth: exact support recovery / FN / FP per library,
plus the true subset's RANK under each objective (how close the
objective is even when argmax misses). Subset tables (RSS, amplitude)
are saved to JSON for further objective design.

    python projects/thesis/true_subset_search.py --systems ode lv kdv
    python projects/thesis/true_subset_search.py                # all
    python projects/thesis/true_subset_search.py --noise-level 1.0
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
from epde.operators.common.stability import VaryingCoefSetup  # noqa: E402
from epde.operators.common.survival import (  # noqa: E402
    _solve_gram, chi2_scores,
)
# The knee family now LIVES in epde -- the same primitives KneeSparsity
# (sparsity_cls='knee') selects with, imported here rather than duplicated, so
# the study and the operator can never drift apart. ``intercept='always'``
# below keeps this script's original convention (the intercept is fitted in
# every subset and excluded from the amplification sum); the operator uses
# 'selectable' instead, because three downstream readers treat a zero
# intercept slot as "no constant term".
from epde.operators.common.subset_selection import (  # noqa: E402
    select_support, subset_table,
)
from kdv_sindy_test import build_pool_only  # noqa: E402
from thesis_runner import load_config, pipeline_settings, _set_seeds  # noqa: E402
from objective_experiment import ALL_SYSTEMS, _as_eqdict  # noqa: E402
from instability_panel import _wrap_noisy  # noqa: E402
from rfe_gate_panel import _inflate_truth, _features  # noqa: E402

OBJECTIVES = ('margin', 'margin_amp', 'bic', 'knee', 'knee_ext2',
              'knee_ext2_amp', 'stab_vc_sum', 'stab_vc_max',
              'stab_chi2_sum', 'stab_chi2_max', 'ke2_stab_chi2',
              'ke2_stab_vc')
STAB_OBJECTIVES = ('stab_vc_sum', 'stab_vc_max', 'stab_chi2_sum',
                   'stab_chi2_max')
KE2_STAB_OBJECTIVES = ('ke2_stab_chi2', 'ke2_stab_vc')
AMP_CAP = 100.0          # the production rps_amplification_cap
OUT = os.path.join(_THIS, 'true_subset_search.json')
MD_OUT = os.path.join(_THIS, 'true_subset_search.md')


# --------------------------------------------------------------------------
# Selection objectives (post-hoc on the table)
# --------------------------------------------------------------------------
def margin_scores(rss, p):
    """Necessity-margin per non-empty subset; -inf where undefined."""
    log_rss = np.log(rss)
    scores = np.full(1 << p, -np.inf)
    for S in range(1, 1 << p):
        in_dam = np.inf
        out_gain = 0.0
        for j in range(p):
            bit = 1 << j
            if S & bit:
                in_dam = min(in_dam, log_rss[S ^ bit] - log_rss[S])
            else:
                out_gain = max(out_gain, log_rss[S] - log_rss[S | bit])
        scores[S] = in_dam - out_gain
    return scores


def _argmax_smallest(scores, p, admissible=None):
    """argmax score; ties -> smaller subset, then lower index
    (deterministic). ``admissible``: optional boolean mask."""
    best_S, best_key = None, None
    for S in range(1, 1 << p):
        if admissible is not None and not admissible[S]:
            continue
        key = (scores[S], -bin(S).count('1'))
        if best_key is None or key > best_key:
            best_key, best_S = key, S
    return best_S if best_S is not None else (1 << p) - 1


def select_subset(objective, rss, amp, p, w_sum):
    if objective in ('margin', 'margin_amp'):
        scores = margin_scores(rss, p)
        adm = (amp <= AMP_CAP) if objective == 'margin_amp' else None
        if adm is not None and not adm[1:].any():
            adm = None               # nothing admissible: fall back
        return _argmax_smallest(scores, p, adm), scores
    if objective == 'bic':
        n_eff = max(w_sum, 2.0)
        sizes = np.array([bin(S).count('1') for S in range(1 << p)])
        bic = n_eff * np.log(rss / n_eff) + (sizes + 1) * np.log(n_eff)
        bic[0] = np.inf
        return int(np.argmin(bic)), -bic
    if objective in ('knee', 'knee_ext2', 'knee_ext2_amp'):
        # epde.operators.common.subset_selection owns the chain, the elbow and
        # the nested-dominant extension (including the accumulated-chain drops
        # the amp-restricted variant needs).
        return select_support(objective, rss, amp, p, amp_cap=AMP_CAP)[0], None
    raise ValueError(objective)


def _knee_chain(rss, p, amp=None):
    """Best-RSS-per-size chain; with ``amp`` the per-size argmin is
    restricted to amplification-admissible subsets (production cap),
    falling back to the unrestricted best when a size has none."""
    sizes = np.array([bin(S).count('1') for S in range(1 << p)])
    best_rss = np.empty(p + 1)
    best_sub = np.empty(p + 1, dtype=int)
    for k in range(p + 1):
        cand = np.where(sizes == k)[0]
        if amp is not None:
            adm = cand[amp[cand] <= AMP_CAP]
            if len(adm):
                cand = adm
        best_sub[k] = int(cand[np.argmin(rss[cand])])
        best_rss[k] = rss[best_sub[k]]
    d = -np.diff(np.log(best_rss))               # drop from k-1 -> k
    return best_rss, best_sub, d


def _knee_size(d):
    d2 = np.append(d, 0.0)
    return int(np.argmax(d2[:-1] - d2[1:])) + 1


def _extend2(d, best_rss, best_sub, p, k):
    """Nested-dominant extension with the min-accepted-drop yardstick:
    grow the size while the next best subset (a) NESTS on the current
    one, (b) does not land on the machine floor, (c) its drop exceeds
    ALL remaining possible improvement, and (d) its drop is at least as
    large as the weakest already-accepted drop -- "a new term must be
    at least as informative as the weakest accepted term". This is the
    gate that accepts ac's weak-but-real diffusion term yet rejects
    kdv-style junk whose drop is large in ratio but weaker than every
    accepted true term."""
    gmin = best_rss.min()
    while k < p:
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


def knee_ext2_stab(rss, amp, p, stat, f, t, w, gshape, vc):
    """The user's designed algorithm: FIND the subset by fit geometry, TEST
    stability. Now a thin wrapper: ``subset_selection.extend2_stab`` owns the
    walk, and this supplies the stability statistic it vetoes on -- chi2 score
    paths or the varying-coefficient NC ratio, each scored WITHIN the extended
    subset's own fit. The elbow subset (the CORE) is never stability-tested;
    stability's only job is vetoing extensions, the one decision point where
    fit geometry is provably blind (kdv_cossin's u_tt: fit-informative but
    incoherent). Returns (subset, vetoed_term)."""
    def stability(mask):
        if stat == 'chi2':
            return chi2_scores(f[:, mask], t, w, gshape, fit_intercept=True)
        return np.asarray(vc.score(np.append(mask, True)), dtype=float)[:-1]

    return select_support('ke2_stab', rss, amp, p, amp_cap=AMP_CAP,
                          stability=stability)


def stability_tables(f, t, w, gshape, var, p, vc):
    """STABILITY-ONLY subset scores (the pure counterpart of the RSS
    objectives): per subset, fit and aggregate the MEMBERS' instability
    (vcoef NC/gamma_0^2 and chi2 score paths; sum = the production
    Instability convention, max = the worst-member lens). The
    always-fitted intercept is EXCLUDED from aggregation: its ~0
    coefficient explodes its own score identically for every subset and
    would mask the differences (the VaryingCoefSetup docstring's PDE
    warning). Selection = argmin; ties -> smaller subset. No RSS
    quantity is consulted anywhere in the decision."""
    n_subsets = 1 << p
    out = {k: np.full(n_subsets, np.inf) for k in STAB_OBJECTIVES}
    for S in range(1, n_subsets):
        mask = np.array([(S >> j) & 1 for j in range(p)], dtype=bool)
        try:
            sc = np.nan_to_num(np.asarray(
                vc.score(np.append(mask, True)), dtype=float))[:-1]
            out['stab_vc_sum'][S] = float(np.sum(sc))
            out['stab_vc_max'][S] = float(np.max(sc))
        except Exception:
            pass
        try:
            sc = np.nan_to_num(np.asarray(
                chi2_scores(f[:, mask], t, w, gshape,
                            fit_intercept=True), dtype=float))
            out['stab_chi2_sum'][S] = float(np.sum(sc))
            out['stab_chi2_max'][S] = float(np.max(sc))
        except Exception:
            pass
    return out


def _argmin_smallest(vals, p):
    best_S, best_key = None, None
    for S in range(1, 1 << p):
        v = vals[S]
        if not np.isfinite(v):
            continue
        key = (v, bin(S).count('1'), S)
        if best_key is None or key < best_key:
            best_key, best_S = key, S
    return best_S if best_S is not None else (1 << p) - 1


# --------------------------------------------------------------------------
# Per-system experiment
# --------------------------------------------------------------------------
def _verdict(S, is_true, p):
    true_S = sum(1 << j for j in range(p) if is_true[j])
    fn = bin(true_S & ~S).count('1')
    fp = bin(S & ~true_S).count('1')
    return {'fn': fn, 'fp': fp, 'exact': S == true_S,
            'kept': [j for j in range(p) if (S >> j) & 1]}


def run_system(system, noise_level=0.0, noise_seed=0, k_junk=6):
    cfg = load_config(system)
    _, _, variable_names, _ = cfg.load_data()
    variable_names = list(variable_names)
    y = yaml.safe_load(open(os.path.join(_THIS, 'configs', f'{system}.yaml'),
                            encoding='utf-8'))
    teqs = y.get('truth_equations') or []
    if not teqs:
        return {'system': system, 'error': 'no truth_equations'}
    truth = _as_eqdict(teqs, variable_names)

    cfg = load_config(system)
    if noise_level > 0:
        _wrap_noisy(cfg, noise_level, noise_seed)
    _set_seeds(0)
    # (Gram mode is derived from the instability metric now: the default
    # 'chi2' already resolves to the varying-coefficient Gram this asked for.)
    search = build_pool_only(cfg, pipeline_settings('new'))
    w = np.asarray(gv.grid_cache.g_func[gv.grid_cache.g_func_mask]).reshape(-1)
    gshape = tuple(int(v) for v in gv.grid_cache.inner_shape)

    per_var = {}
    for var in variable_names:
        try:
            soeq, labels, is_true, junk = _inflate_truth(
                truth, var, variable_names, search.pool, len(gshape), k_junk)
            f, t = _features(soeq, var)
            t0 = time.perf_counter()
            rss, amp, p = subset_table(f, t, w, intercept='always')
            secs_table = time.perf_counter() - t0

            vc = VaryingCoefSetup(f, t, w, gshape, main_var=var)
            t0 = time.perf_counter()
            stab = stability_tables(f, t, w, gshape, var, p, vc)
            secs_stab = time.perf_counter() - t0

            true_S = sum(1 << j for j in range(p) if is_true[j])
            rec = {'labels': labels, 'is_true': is_true,
                   'n_true': int(sum(is_true)), 'p': p,
                   'secs_table': secs_table, 'secs_stab': secs_stab,
                   'true_amp': float(amp[true_S]),
                   'objectives': {}}
            for obj in OBJECTIVES:
                if obj in KE2_STAB_OBJECTIVES:
                    stat = 'chi2' if obj.endswith('chi2') else 'vc'
                    S, vetoed = knee_ext2_stab(rss, amp, p, stat,
                                               f, t, w, gshape, vc)
                    v = _verdict(S, is_true, p)
                    if vetoed is not None:
                        v['vetoed'] = labels[vetoed]
                elif obj in STAB_OBJECTIVES:
                    vals = stab[obj]
                    S = _argmin_smallest(vals, p)
                    v = _verdict(S, is_true, p)
                    order = np.argsort(vals, kind='stable')
                    v['true_rank'] = int(np.where(order == true_S)[0][0]) + 1
                else:
                    S, scores = select_subset(obj, rss, amp, p,
                                              float(w.sum()))
                    v = _verdict(S, is_true, p)
                    if scores is not None:
                        order = np.argsort(scores)[::-1]
                        v['true_rank'] = int(
                            np.where(order == true_S)[0][0]) + 1
                rec['objectives'][obj] = v
            # Save the table for further objective design (float32).
            rec['table'] = {'rss': [float(x) for x in rss],
                            'amp': [float(x) for x in amp]}
            per_var[var] = rec
        except Exception as exc:
            per_var[var] = {'error': f'{type(exc).__name__}: '
                                     f'{str(exc)[:120]}'}
    return {'system': system, 'variables': variable_names,
            'noise_level': noise_level, 'per_var': per_var}


# --------------------------------------------------------------------------
def render_markdown(recs):
    recs_ok = [r for r in recs if 'per_var' in r]
    nls = sorted({float(r.get('noise_level', 0.0)) for r in recs_ok})
    scen = ('clean data' if nls == [0.0] else
            'noise ' + '/'.join(f'{n:g}%' for n in nls) + ' of std')
    lines = [
        f'# True-subset search: exhaustive enumeration ({scen})',
        '',
        'Every feature subset fitted (Gram space, intercept always in); '
        'objectives select per equation with NO cross-system constant '
        '(margin_amp uses only the pre-existing production amplification '
        'cap as admissibility). true_rank = rank of the true subset '
        'under the objective (1 = found).',
        '',
        '| system | var | ' + ' | '.join(OBJECTIVES) + ' | true rank '
        '(margin/margin_amp) |',
        '|---' * (len(OBJECTIVES) + 3) + '|',
    ]
    tot = {o: {'exact': 0, 'fn': 0, 'fp': 0} for o in OBJECTIVES}
    n_libs = 0
    for rec in recs_ok:
        for var, r in rec['per_var'].items():
            if not isinstance(r, dict) or 'objectives' not in r:
                continue
            n_libs += 1
            cells = []
            for o in OBJECTIVES:
                v = r['objectives'][o]
                tot[o]['exact'] += v['exact']
                tot[o]['fn'] += v['fn']
                tot[o]['fp'] += v['fp']
                cells.append('EXACT' if v['exact']
                             else f"FN={v['fn']},FP={v['fp']}")
            ranks = (str(r['objectives']['margin'].get('true_rank', '-'))
                     + '/'
                     + str(r['objectives']['margin_amp'].get('true_rank',
                                                             '-')))
            lines.append(f"| {rec['system']} | {var} | "
                         + ' | '.join(cells) + f' | {ranks} |')
    lines.append('| **total exact** | | ' + ' | '.join(
        f"**{tot[o]['exact']}/{n_libs}**" for o in OBJECTIVES) + ' | |')
    lines.append('| total FN | | ' + ' | '.join(
        str(tot[o]['fn']) for o in OBJECTIVES) + ' | |')
    lines.append('| total FP | | ' + ' | '.join(
        str(tot[o]['fp']) for o in OBJECTIVES) + ' | |')
    errs = [r for r in recs if 'error' in r]
    if errs:
        lines += ['', '## Skipped systems', '']
        lines += [f"- {r['system']}: {r['error']}" for r in errs]
    return '\n'.join(lines) + '\n'


def main(argv=None):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument('--systems', nargs='+', default=ALL_SYSTEMS)
    ap.add_argument('--out', default=OUT)
    ap.add_argument('--md-out', default=MD_OUT)
    ap.add_argument('--k-junk', type=int, default=6)
    ap.add_argument('--noise-level', type=float, default=0.0)
    ap.add_argument('--noise-seed', type=int, default=0)
    args = ap.parse_args(argv)

    results = []
    for system in args.systems:
        t0 = time.perf_counter()
        try:
            rec = run_system(system, noise_level=args.noise_level,
                             noise_seed=args.noise_seed,
                             k_junk=args.k_junk)
        except Exception as exc:
            rec = {'system': system,
                   'error': f'{type(exc).__name__}: {str(exc)[:160]}'}
        rec['secs'] = time.perf_counter() - t0
        results.append(rec)
        print(f"[{system}] {rec.get('error', 'ok')} "
              f"({rec['secs']:.1f}s)", flush=True)

    with open(args.out, 'w', encoding='utf-8') as fh:
        json.dump(results, fh)
    md = render_markdown(results)
    with open(args.md_out, 'w', encoding='utf-8') as fh:
        fh.write(md)
    print(md)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
