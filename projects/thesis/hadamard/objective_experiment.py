"""Which complementary objective ranks the TRUE equation below truly-WRONG forms?

The thesis needs a MOEA/D objective, complementary to discrepancy, that is
minimised by the true equation. The target is **truly-wrong** forms (overfits,
wrong structures, under-specified) -- NOT analytical identities, which are
credited valid and must stay in the "good" cluster with the truth.

For each of the 14 systems we assemble candidates:
  * `true`               -- the seeded truth (from configs/<sys>.yaml).
  * `identity`           -- valid analytical alternatives (truth_alternatives).
  * `wrong:missing`      -- a true term dropped (constructed).
  * `wrong:overfit`      -- truth + a spurious extra term (constructed).
  * `wrong:discovered`   -- structurally-wrong equations the pipeline actually
                            produced (results/<sys>/*rep*.json, struct.fail).

and score each, across a noise sweep, by three objectives (lower = better):
  * `disc`   -- normalised weighted residual of the solver-free OLS fit
                (the discrepancy objective).
  * `instab` -- the current complement: sum of VaryingCoefSetup per-term scores
                (vcoef instability).
  * `gengap` -- candidate complement: held-out normalised residual under K-fold
                cross-validation over grid points (generalisation; an exact law
                or identity holds on every fold, an overfit/accidental fit does
                not).

Multi-equation systems (lorenz, lv, ns) aggregate per-equation: `disc`/`gengap`
mean over variables, `instab` sum over variables (as MOEA/D does).

Verdict per (system, noise, objective): is `true` strictly below every
truly-wrong form (separation > 0), and do the valid identities stay below the
wrong forms (not penalised)?

    python projects/thesis/hadamard/objective_experiment.py --systems kdv lv ac
    python projects/thesis/hadamard/objective_experiment.py            # all 14
"""
from __future__ import annotations

import argparse
import json
import glob
import os
import re
import sys

_THIS = os.path.dirname(os.path.abspath(__file__))
_THESIS = os.path.abspath(os.path.join(_THIS, '..'))
_ROOT = os.path.abspath(os.path.join(_THIS, '..', '..', '..'))
for _p in (_ROOT, _THESIS, _THIS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
import yaml  # noqa: E402
import epde.globals as gv  # noqa: E402
from epde.interface.equation_translator import translate_equation  # noqa: E402
from epde.operators.common.stability import VaryingCoefSetup  # noqa: E402
from kdv_sindy_test import build_pool_only, _normalize_grid_labels  # noqa: E402
from thesis_runner import load_config, pipeline_settings, _set_seeds  # noqa: E402
from inverse_stability import _noise_load  # noqa: E402

ALL_SYSTEMS = ['ode', 'vdp', 'lorenz', 'lv', 'wave', 'kdv_cossin', 'kdv', 'ac',
               'burgers_inviscid', 'pde_divide', 'pde_compound',
               'burgers_viscous', 'ks', 'ns']
NOISE = [0.0, 1.0, 4.0, 8.0]
RESULTS = os.path.join(_THESIS, 'results')
OUT = os.path.join(_THESIS, 'objective_experiment.json')


# --------------------------------------------------------------------------
# Truth / identity / wrong-form assembly
# --------------------------------------------------------------------------
def _as_eqdict(eq_list, variable_names):
    """A list of per-equation strings -> {var: string} (or the bare string)."""
    if len(variable_names) == 1:
        return eq_list[0]
    return {v: eq_list[i] for i, v in enumerate(variable_names)}


def _drop_a_term(eq_str):
    """Drop one non-constant LHS term -> a 'missing' wrong form (or None)."""
    if ' = ' not in eq_str:
        return None
    lhs, rhs = eq_str.split(' = ', 1)
    terms = [t.strip() for t in lhs.split(' + ')]
    # keep only "real" terms (not the bare 0.0 intercept) as drop candidates
    droppable = [i for i, t in enumerate(terms)
                 if not re.fullmatch(r'[-+0-9.eE]+', t)]
    if len(droppable) < 2:
        return None
    keep = [t for i, t in enumerate(terms) if i != droppable[-1]]
    return ' + '.join(keep) + ' = ' + rhs


def _add_a_term(eq_str, var):
    """Add a spurious term -> an 'overfit' wrong form (or None)."""
    if ' = ' not in eq_str:
        return None
    lhs, rhs = eq_str.split(' = ', 1)
    for extra in (f'{var}{{power: 2.0}}', f'{var}{{power: 3.0}}',
                  'du/dx0{power: 1.0}'):
        if extra not in eq_str and extra not in rhs:
            return lhs + f' + 1.0 * {extra} = ' + rhs
    return None


def _constructed_wrong(truth, variable_names):
    """Build {label: eqdict} of constructed wrong forms from the truth."""
    out = {}
    if len(variable_names) == 1:
        miss = _drop_a_term(truth)
        if miss:
            out['wrong:missing'] = miss
        over = _add_a_term(truth, variable_names[0])
        if over:
            out['wrong:overfit'] = over
    else:
        # perturb the FIRST variable's equation, keep the rest true
        v0 = variable_names[0]
        miss = _drop_a_term(truth[v0])
        if miss:
            d = dict(truth)
            d[v0] = miss
            out['wrong:missing'] = d
        over = _add_a_term(truth[v0], v0)
        if over:
            d = dict(truth)
            d[v0] = over
            out['wrong:overfit'] = d
    return out


_MARKER_RE = re.compile(r'^\s*[/|\\]\s*')


def _discovered_wrong(system, variable_names, limit=3):
    """Parse structurally-wrong discoveries from results/<system>/*rep*.json."""
    d = os.path.join(RESULTS, system)
    if not os.path.isdir(d):
        return []
    out = []
    for fp in sorted(glob.glob(os.path.join(d, '*rep*.json'))):
        if len(out) >= limit:
            break
        try:
            rec = json.load(open(fp, encoding='utf-8'))
        except Exception:
            continue
        if rec.get('structural_success') is not False:
            continue
        text = rec.get('discovered_text')
        if not text:
            continue
        # discovered_text: list of equation lines + a trailing metaparam dict.
        lines = [ln for ln in text
                 if isinstance(ln, str) and "'terms_number'" not in ln
                 and "'max_terms_number'" not in ln
                 and ' = ' in ln]
        lines = [_MARKER_RE.sub('', ln).strip() for ln in lines]
        if not lines:
            continue
        if len(variable_names) == 1:
            out.append((os.path.basename(fp), lines[0]))
            continue
        # multi-eq: map each line to a variable by the target's variable letter
        eqd = {}
        for ln in lines:
            tgt = ln.split(' = ', 1)[1]
            for v in variable_names:
                if re.search(rf'\b{re.escape(v)}\b', tgt) or \
                   re.search(rf'd[\w^]*{re.escape(v)}/', tgt):
                    eqd.setdefault(v, ln)
                    break
        if set(eqd) == set(variable_names):
            out.append((os.path.basename(fp), eqd))
    return out


# --------------------------------------------------------------------------
# Objective computation
# --------------------------------------------------------------------------
def _features(soeq, var):
    eq = soeq.vals[var]
    eq.main_var_to_explain = var
    eq.weights_internal = np.ones(len(eq.structure) - 1)
    eq.weights_internal_evald = True
    eq.weights_final_evald = True
    _, t, f = eq.evaluate(normalize=False, return_val=False)
    f = np.asarray(f, dtype=float)
    if f.ndim == 1:
        f = f[:, None]
    return f, np.asarray(t, float).reshape(-1)


def _disc(f, t, w):
    sw = np.sqrt(np.clip(w, 0, None))
    A, b = f * sw[:, None], t * sw
    th, *_ = np.linalg.lstsq(A, b, rcond=None)
    return float(np.linalg.norm(A @ th - b) / (np.linalg.norm(b) + 1e-30))


def _gengap(f, t, w, k=3, seed=0):
    n = f.shape[0]
    if n < 2 * k:
        return np.nan
    rng = np.random.default_rng(seed)
    folds = np.array_split(rng.permutation(n), k)
    res = []
    for i in range(k):
        te = folds[i]
        tr = np.concatenate([folds[j] for j in range(k) if j != i])
        swtr = np.sqrt(np.clip(w[tr], 0, None))
        th, *_ = np.linalg.lstsq(f[tr] * swtr[:, None], t[tr] * swtr, rcond=None)
        swte = np.sqrt(np.clip(w[te], 0, None))
        A, b = f[te] * swte[:, None], t[te] * swte
        res.append(np.linalg.norm(A @ th - b) / (np.linalg.norm(b) + 1e-30))
    return float(np.mean(res))


# The vcoef instability score splits into two numerator channels over the same
# gamma_0^2 denominator: 'var' = Var(gamma_0) (significance / structural-misfit)
# and 'nc' = NC_deb (noise-debiased region-variation); 'both' is their sum (the
# production objective). This experiment asks which channel best ranks the TRUE
# equation below truly-wrong forms -- the user's "var vs nc vs both".
INSTAB_COMPONENTS = ('both', 'var', 'nc')


def _instab(f, t, w, gshape, var, component='both'):
    setup = VaryingCoefSetup(f, t, w, gshape, main_var=var, fit_intercept=False)
    return float(np.sum(np.asarray(setup.score(None, component=component),
                                   dtype=float)))


def score_candidate(eqdict, variable_names, pool, w, gshape):
    """Aggregate (disc, gengap, instab[/var/nc/both]) over the candidate's
    equations. ``instab`` is kept as an alias of ``instab_both`` so existing
    saved JSON / ``--summarize`` stay valid."""
    soeq = translate_equation(_normalize_grid_labels(eqdict), pool,
                              all_vars=list(variable_names))
    discs, gaps = [], []
    instabs = {c: [] for c in INSTAB_COMPONENTS}
    for v in variable_names:
        f, t = _features(soeq, v)
        discs.append(_disc(f, t, w))
        gaps.append(_gengap(f, t, w))
        for c in INSTAB_COMPONENTS:
            try:
                instabs[c].append(_instab(f, t, w, gshape, v, component=c))
            except Exception:
                instabs[c].append(np.nan)
    out = {'disc': float(np.mean(discs)),
           'gengap': float(np.nanmean(gaps))}
    for c in INSTAB_COMPONENTS:
        out[f'instab_{c}'] = float(np.nansum(instabs[c]))
    out['instab'] = out['instab_both']  # back-compat alias
    return out


# --------------------------------------------------------------------------
# Per-system experiment
# --------------------------------------------------------------------------
def run_system(system, noise_levels=NOISE):
    cfg = load_config(system)
    _, _, variable_names, _ = cfg.load_data()
    variable_names = list(variable_names)

    y = yaml.safe_load(open(os.path.join(_THESIS, 'configs', f'{system}.yaml'),
                            encoding='utf-8'))
    teqs = y.get('truth_equations') or []
    truth = _as_eqdict(teqs, variable_names)

    candidates = [('true', 'true', truth)]
    for alt in (y.get('truth_alternatives') or []):
        if alt:
            candidates.append((f'identity[{len(candidates)}]', 'identity',
                               _as_eqdict(alt, variable_names)))
    for label, eqd in _constructed_wrong(truth, variable_names).items():
        candidates.append((label, 'wrong', eqd))
    for fname, eqd in _discovered_wrong(system, variable_names):
        candidates.append((f'wrong:disc[{fname}]', 'wrong', eqd))

    per_noise = {}
    for nl in noise_levels:
        cfg = load_config(system)
        if nl > 0:
            cfg.load_data = _noise_load(cfg.load_data, nl, seed=0)
        _set_seeds(0)
        gv.set_gram_config('vcoef')
        search = build_pool_only(cfg, pipeline_settings('new'))
        w = np.asarray(gv.grid_cache.g_func[gv.grid_cache.g_func_mask]).reshape(-1)
        gshape = tuple(int(n) for n in gv.grid_cache.inner_shape)

        rows = []
        for label, cat, eqd in candidates:
            try:
                sc = score_candidate(eqd, variable_names, search.pool, w, gshape)
                sc.update({'label': label, 'category': cat})
                rows.append(sc)
            except Exception as exc:
                rows.append({'label': label, 'category': cat,
                             'error': f'{type(exc).__name__}: {str(exc)[:80]}'})
        per_noise[str(nl)] = rows
    return {'system': system, 'variables': variable_names,
            'n_candidates': len(candidates), 'per_noise': per_noise}


def _verdict(rows, objective):
    """Separation (min wrong - true) and whether identities stay below wrong."""
    def val(r):
        return r.get(objective, np.nan)
    true = next((val(r) for r in rows if r.get('category') == 'true'), np.nan)
    wrong = [val(r) for r in rows if r.get('category') == 'wrong'
             and np.isfinite(val(r))]
    ident = [val(r) for r in rows if r.get('category') == 'identity'
             and np.isfinite(val(r))]
    if not np.isfinite(true) or not wrong:
        return None
    sep = float(min(wrong) - true)             # >0 => true strictly best
    ident_ok = (not ident) or (max(ident) < min(wrong))
    return {'true': float(true), 'min_wrong': float(min(wrong)),
            'separation': sep, 'true_below_all_wrong': sep > 0,
            'identity_below_wrong': bool(ident_ok), 'n_wrong': len(wrong),
            'n_ident': len(ident)}


def _verdict_pareto(rows, instab_key, disc_key='disc'):
    """MOEA/D-relevant lens: is TRUE non-dominated on (disc, instab) vs every
    truly-wrong form? (A wrong form dominates TRUE iff it is <= on BOTH axes
    and strictly < on at least one.) Non-dominated => TRUE survives on the
    returned Pareto front. Unlike ``_verdict`` this credits a TIE on the
    instability axis (the NC-under-noise case), since a tie is not domination.
    """
    def pair(r):
        return r.get(disc_key, np.nan), r.get(instab_key, np.nan)
    true = next((pair(r) for r in rows if r.get('category') == 'true'),
                (np.nan, np.nan))
    if not (np.isfinite(true[0]) and np.isfinite(true[1])):
        return None
    td, ti = true
    dominators = []
    for r in rows:
        if r.get('category') != 'wrong':
            continue
        wd, wi = pair(r)
        if not (np.isfinite(wd) and np.isfinite(wi)):
            continue
        if wd <= td and wi <= ti and (wd < td or wi < ti):
            dominators.append(r.get('label'))
    n_wrong = sum(1 for r in rows if r.get('category') == 'wrong'
                  and np.isfinite(pair(r)[0]) and np.isfinite(pair(r)[1]))
    if n_wrong == 0:
        return None
    return {'true_nondominated': len(dominators) == 0,
            'n_dominators': len(dominators), 'dominators': dominators,
            'n_wrong': n_wrong}


def _scoreboard(recs, noise, objectives, title):
    """Print a # TRUE-below-all-wrong scoreboard for the given objective keys."""
    n = len(recs)
    print(f"\n{title} (/{n}); 't' = TRUE strictly below all wrong, "
          f"'id' = identities kept below wrong")
    print(f"{'noise':>6} | " + ' | '.join(f'{o:^20}' for o in objectives))
    for nl in noise:
        cells = []
        for o in objectives:
            verds = [_verdict(r['per_noise'][nl], o) for r in recs]
            verds = [v for v in verds if v is not None]
            ok = sum(v['true_below_all_wrong'] for v in verds)
            idok = sum(v['identity_below_wrong'] for v in verds)
            cells.append(f'{ok:2d}/{len(verds)} t {idok:2d} id'.center(20))
        print(f"{nl:>6} | " + ' | '.join(cells))


def _per_system_table(recs, noise, objectives, title):
    """Per-system pass matrix (OK/XX per objective) at each noise level."""
    print(f"\n{title}  (OK=TRUE below all wrong, XX=not; '+'=identity-safe)")
    hdr = f"{'system':16s} {'noise':>5} | " + ' | '.join(f'{o:^12}' for o in objectives)
    print(hdr)
    print('-' * len(hdr))
    for r in recs:
        for nl in noise:
            cells = []
            for o in objectives:
                v = _verdict(r['per_noise'][nl], o)
                if v is None:
                    cells.append('   n/a'.center(12))
                    continue
                mark = 'OK' if v['true_below_all_wrong'] else 'XX'
                idn = '+' if v['identity_below_wrong'] else '-'
                cells.append(f"{mark}{idn} {v['separation']:+.1e}".center(12))
            print(f"{r['system']:16s} {nl:>5} | " + ' | '.join(cells))


def summarize(out=OUT):
    """Cross-system scoreboard from a saved experiment JSON."""
    recs = [r for r in json.load(open(out, encoding='utf-8')) if 'per_noise' in r]
    noise = sorted({k for r in recs for k in r['per_noise']},
                   key=lambda s: float(s))
    n = len(recs)
    print(f"Systems: {n}  {[r['system'] for r in recs]}")

    # 1) the original disc/instab/gengap board (instab == instab_both)
    print(f"\nSCOREBOARD - # systems where TRUE is strictly below ALL truly-wrong "
          f"forms (/{n}); 'id' = identities kept below wrong")
    print(f"{'noise':>6} | " + ' | '.join(f'{o:^20}' for o in
                                           ('disc', 'instab', 'gengap',
                                            'instab|gengap')))
    for nl in noise:
        cells = []
        combo_ok = combo_tot = 0
        verds = {o: [] for o in ('disc', 'instab', 'gengap')}
        for r in recs:
            vs = {o: _verdict(r['per_noise'][nl], o)
                  for o in ('disc', 'instab', 'gengap')}
            for o in verds:
                if vs[o] is not None:
                    verds[o].append(vs[o])
            if vs['instab'] is not None or vs['gengap'] is not None:
                combo_tot += 1
                combo_ok += ((vs['instab'] and vs['instab']['true_below_all_wrong'])
                             or (vs['gengap'] and vs['gengap']['true_below_all_wrong']))
        for o in ('disc', 'instab', 'gengap'):
            ok = sum(v['true_below_all_wrong'] for v in verds[o])
            idok = sum(v['identity_below_wrong'] for v in verds[o])
            cells.append(f'{ok:2d}/{len(verds[o])} t {idok:2d} id'.center(20))
        cells.append(f'{combo_ok:2d}/{combo_tot}'.center(20))
        print(f"{nl:>6} | " + ' | '.join(cells))

    # 2) the var-vs-nc-vs-both instability comparison (the focus of this run)
    have_components = all(f'instab_{c}' in (recs[0]['per_noise'][noise[0]][0]
                          if recs and recs[0]['per_noise'].get(noise[0]) else {})
                          for c in INSTAB_COMPONENTS) if recs else False
    comp_objs = ['instab_both', 'instab_var', 'instab_nc']
    if have_components:
        _scoreboard(recs, noise, comp_objs,
                    'INSTABILITY CHANNELS [strict: TRUE below ALL wrong] '
                    '- var vs nc vs both')
        # Pareto lens: TRUE non-dominated on (disc, instab) -- what MOEA/D
        # selection actually returns; credits ties on the instability axis.
        print(f"\nINSTABILITY CHANNELS [Pareto: TRUE non-dominated on "
              f"(disc, instab)] (/{n})")
        print(f"{'noise':>6} | " + ' | '.join(f'{o:^20}' for o in comp_objs))
        for nl in noise:
            cells = []
            for o in comp_objs:
                vs = [_verdict_pareto(r['per_noise'][nl], o) for r in recs]
                vs = [v for v in vs if v is not None]
                ok = sum(v['true_nondominated'] for v in vs)
                cells.append(f'{ok:2d}/{len(vs)} nd'.center(20))
            print(f"{nl:>6} | " + ' | '.join(cells))
        _per_system_table(recs, noise, comp_objs,
                          'Per-system instability-channel pass matrix '
                          '[strict separation]')


def main(argv=None):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
    p = argparse.ArgumentParser()
    p.add_argument('--systems', nargs='+', default=ALL_SYSTEMS)
    p.add_argument('--noise', nargs='+', type=float, default=NOISE)
    p.add_argument('--out', default=OUT)
    p.add_argument('--summarize', action='store_true',
                   help='Print the scoreboard from an existing --out JSON and exit.')
    args = p.parse_args(argv)

    if args.summarize:
        summarize(args.out)
        return 0

    results = []
    for system in args.systems:
        try:
            rec = run_system(system, args.noise)
            results.append(rec)
            # console summary
            console_objs = ('disc', 'instab_both', 'instab_var', 'instab_nc',
                            'gengap')
            print(f"\n=== {system}  ({rec['n_candidates']} candidates, "
                  f"vars={rec['variables']}) ===")
            print(f"{'noise':>6} | "
                  + ' | '.join(f'{o:^22}' for o in console_objs))
            for nl in args.noise:
                cells = []
                for obj in console_objs:
                    v = _verdict(rec['per_noise'][str(nl)], obj)
                    if v is None:
                        cells.append('  n/a'.center(22))
                        continue
                    mark = 'OK' if v['true_below_all_wrong'] else 'XX'
                    idn = 'i+' if v['identity_below_wrong'] else 'i-'
                    cells.append(f"{mark} sep={v['separation']:+.2g} {idn}"
                                 .center(22))
                print(f"{nl:6.1f} | " + ' | '.join(cells))
        except Exception as exc:
            import traceback
            traceback.print_exc()
            results.append({'system': system, 'error': repr(exc)})
        with open(args.out, 'w', encoding='utf-8') as fh:
            json.dump(results, fh, indent=2,
                      default=lambda o: float(o) if isinstance(
                          o, (np.floating, np.integer)) else str(o))
    print('\nDONE ->', args.out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
