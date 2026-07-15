"""Analyze the final Pareto front of a thesis discovery run, decomposing each
discovered equation with the coef x term contribution tooling.

Summary first (objective vector [disc_k | instab_k], hamming, #EXTRA terms per
solution), then a detailed contribution decomposition (L2/per-pt/dom%/tgt%,
TRUE vs EXTRA) of the key front members (lowest hamming, lowest total
discrepancy, highest instability).

Usage: python projects/thesis/analyze_pareto.py [system] [path/to/rep.json]
"""
import os
import re
import sys
import json

_THIS = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.abspath(os.path.join(_THIS, '..', '..')), _THIS):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
from term_contribution import (build_pool, analyze_equation, _norm_factor, _term, _eval,
                               CONFIGS_DIR)
from thesis_runner import _load_yaml
from epde.interface.equation_translator import parse_equation_str, float_convertable


def _canon_term(factor_strs):
    items = []
    for fs in factor_strs:
        fs = _norm_factor(fs)
        label = fs.split('{')[0].strip()
        m = re.search(r'power:\s*([0-9.]+)', fs)
        items.append((label, float(m.group(1)) if m else 1.0))
    return frozenset(items)


def _truth_terms_by_target(truth_eqs):
    out = {}
    for eq in truth_eqs:
        *left, right = parse_equation_str(eq)
        true_set = set()
        for term in left:
            factors = term[1:] if (float_convertable(term[0]) and len(term) > 1) else term
            if len(factors) == 1 and float_convertable(factors[0]):
                continue
            true_set.add(_canon_term(factors))
        out[_canon_term(right)] = true_set
    return out


def _clean_lines(sol_lines):
    """Strip SoEq decoration (/, |, \\) and keep only equation lines."""
    out = []
    for raw in sol_lines:
        line = raw.strip().lstrip('/|\\').strip()
        if ' = ' in line:
            out.append(line)
    return out


def _designX(line, pool, all_vars):
    """Rebuild the (N x k) design matrix of an equation's left terms (for cond)."""
    *left, right = parse_equation_str(line)
    cols = []
    for term in left:
        factors = term[1:] if (float_convertable(term[0]) and len(term) > 1) else term
        if len(factors) == 1 and float_convertable(factors[0]):
            continue
        cols.append(_eval(_term(factors, pool, all_vars)))
    return np.column_stack(cols) if cols else np.zeros((1, 0))


def _agg(decomp):
    """Aggregate per-equation contribution metrics to one solution (worst eq)."""
    return dict(
        n_terms=sum(d[4]['n_terms'] for d in decomp),
        max_tgt=max(d[4]['max_tgt'] for d in decomp),
        sum_tgt=max(d[4]['sum_tgt'] for d in decomp),
        coef_range=max(d[4]['coef_range'] for d in decomp),
        cond=max(d[4]['cond'] for d in decomp),
        l2_eff=max(d[4]['l2_eff'] for d in decomp),
        pp_eff=max(d[4]['pp_eff'] for d in decomp),
        dom_eff=max(d[4]['dom_eff'] for d in decomp),
    )


def _truth_objectives(eqs_by_var, all_vars, search):
    """Per-equation WAPE discrepancy + vcoef instability of a known equation
    system through the same solver-free fitness the 'new' pipeline uses.
    Returns {var: (disc, instab)}."""
    try:
        import epde.globals as global_var
        from epde import globals as gv
        from epde.operators.common.fitness import SolverFreeFitness
        from epde.operators.common.objectives import WAPEDiscrepancy, Instability, FitContext
        from epde.operators.common.sparsity import VWSRSparsity
        from epde.operators.common.coeff_calculation import LinRegBasedCoeffsEquation
        from epde.operators.utils.default_parameter_loader import EvolutionaryParams
        from epde.interface.equation_translator import translate_equation
        gv.set_gram_config('vcoef')
        op = EvolutionaryParams().get_default_params_for_operator('SolverFreeFitness')
        disc = WAPEDiscrepancy()
        fit = SolverFreeFitness(list(op.keys()), objectives=[disc, Instability()], primary=disc)
        fit.params = op
        fit.set_suboperators({'sparsity': VWSRSparsity(), 'coeff_calc': LinRegBasedCoeffsEquation()})
        soeq = translate_equation(eqs_by_var, search.pool, all_vars)
        g = global_var.grid_cache.g_func[global_var.grid_cache.g_func_mask].reshape(-1)
        ctx = FitContext(g_fun_vals=g, data_shape=global_var.grid_cache.inner_shape,
                         penalty_coeff=0.5, for_rps=False)
        res = {}
        for var in all_vars:
            eq = soeq.vals[var]
            eq.main_var_to_explain = var
            eq.metaparameters = {('sparsity', x): {'optimizable': False, 'value': 1e-6} for x in all_vars}
            eq.weights_internal = np.ones(len(eq.structure) - 1)
            eq.weights_internal_evald = True
            eq.weights_final_evald = True
            fit.apply(eq, {}, force_out_of_place=True)
            eq.fitness_calculated = False
            eq.stability_calculated = False
            fit.apply(eq, {}, force_out_of_place=False)
            res[var] = (float(disc.compute(eq, ctx)), float(eq.coefficients_stability))
        return res
    except Exception as exc:
        print(f"  [truth objectives skip] {exc!r}")
        return {v: (float('nan'), float('nan')) for v in all_vars}


def _decompose(line, pool, all_vars, truth_map):
    rows, r2, tgt = analyze_equation(line, [], pool, all_vars)
    true_set = truth_map.get(_canon_term(tgt.split(' * ')), set())
    tagged = [(lbl, _canon_term(lbl.split(' * ')) in true_set, share, pp, dom, expl)
              for (lbl, _t, coef, rms, share, share_tgt, expl, pp, relm, dom, dr2) in rows]
    n_extra = sum(1 for t in tagged if not t[1])
    # Candidate degenerate-overfit metrics for this equation.
    coefs = np.array([r[2] for r in rows], float)
    expl = np.array([r[6] for r in rows], float)         # tgt% (signed)
    nz = np.abs(coefs[coefs != 0.0])
    X = _designX(line, pool, all_vars)
    cond = float(np.linalg.cond(X - X.mean(0))) if X.shape[1] > 1 else 1.0

    # Effective # contributing terms = 1/sum(share^2) (Herfindahl inverse) of a
    # share distribution: ~k_true for a clean eq, larger when contribution is
    # spread across many terms (degenerate overfit). Truth-free.
    def _eff(vals):
        s = np.abs(np.asarray(vals, float)); tot = s.sum()
        if tot <= 0:
            return float(len(s))
        s = s / tot
        h = float(np.sum(s * s))
        return 1.0 / h if h > 0 else float(len(s))
    l2 = [r[4] for r in rows]; pp = [r[7] for r in rows]; dm = [r[9] for r in rows]

    m = dict(n_terms=len(rows),
             max_tgt=float(np.max(np.abs(expl))) if expl.size else 0.0,
             sum_tgt=float(np.sum(np.abs(expl))) if expl.size else 0.0,
             coef_range=float(nz.max() / nz.min()) if nz.size else 1.0,
             cond=cond,
             l2_eff=_eff(l2), pp_eff=_eff(pp), dom_eff=_eff(dm))
    return tagged, r2, tgt, n_extra, m


def main(system='lorenz', repfile=None):
    repfile = repfile or os.path.join(_THIS, 'results', system, 'new_rep00.json')
    rec = json.load(open(repfile, encoding='utf-8'))
    sols_raw = rec.get('discovered_text_per_solution') or []
    objs = rec.get('objectives_per_solution') or []
    hams = rec.get('hamming_per_solution') or []
    truth_eqs = (_load_yaml(os.path.join(CONFIGS_DIR, f'{system}.yaml')).get('truth_equations') or [])
    truth_map = _truth_terms_by_target(truth_eqs)

    cfg, search, all_vars = build_pool(system)

    # Dedupe by cleaned-structure; keep first index of each unique solution.
    uniq = {}
    for i, s in enumerate(sols_raw):
        key = tuple(_clean_lines(s))
        if key and key not in uniq:
            uniq[key] = i
    n_eq = len(truth_eqs)

    def disc_instab(ov):
        if ov and len(ov) == 2 * n_eq:                 # [disc_0..disc_k, instab_0..instab_k]
            return ov[:n_eq], ov[n_eq:]
        return ov, None

    items = []
    for key, i in uniq.items():
        eqs = list(key)
        ov = objs[i] if i < len(objs) else None
        ham = hams[i] if i < len(hams) else None
        decomp = [_decompose(line, search.pool, all_vars, truth_map) for line in eqs]
        n_extra = sum(d[3] for d in decomp)
        disc, inst = disc_instab(ov)
        agg = _agg(decomp)
        agg['disc'] = sum(disc) if disc else float('nan')
        agg['instab'] = sum(inst) if inst else float('nan')
        items.append(dict(i=i, eqs=eqs, ov=ov, ham=ham, decomp=decomp,
                          n_extra=n_extra, agg=agg))

    # ---------- per-EQUATION grouped view ----------
    from collections import defaultdict

    print(f"### {system}: {rec.get('n_pareto_solutions')} Pareto-0 ({len(items)} unique)  "
          f"structural_success={rec.get('structural_success')}  best_hamming={rec.get('hamming')}  "
          f"epochs={rec.get('n_epochs')}  (metrics SPLIT PER EQUATION)")

    # TRUE per-equation objectives + decomposition, and the true term-set per var.
    truth_pe, truth_dec, truth_struct = {}, {}, {}
    if n_eq == len(all_vars):
        truth_pe = _truth_objectives({all_vars[k]: truth_eqs[k] for k in range(n_eq)},
                                     all_vars, search)
        for k in range(n_eq):
            try:
                td = _decompose(truth_eqs[k], search.pool, all_vars, truth_map)
                truth_dec[all_vars[k]] = td
                truth_struct[all_vars[k]] = frozenset(_canon_term(t[0].split(' * ')) for t in td[0])
            except Exception:
                truth_dec[all_vars[k]] = None

    # Group discovered eq-forms per var by structural canon (coeff-independent).
    groups = defaultdict(dict)
    for it in items:
        disc, inst = disc_instab(it['ov'])
        for k in range(min(n_eq, len(it['eqs']))):
            var = all_vars[k]
            tagged, r2, tgt, n_extra, m = it['decomp'][k]
            ckey = frozenset(_canon_term(t[0].split(' * ')) for t in tagged)
            g = groups[var].get(ckey)
            if g is None:
                g = dict(ckey=ckey, disc=(disc[k] if disc else float('nan')),
                         instab=(inst[k] if inst else float('nan')),
                         m=m, n_extra=n_extra, r2=r2, count=0)
                groups[var][ckey] = g
            g['count'] += 1

    cols = [('disc', 10, '.1e'), ('instab', 10, '.1e'), ('n', 4, 'd'), ('extra', 6, 'd'),
            ('L2_eff', 8, '.1f'), ('pp_eff', 8, '.1f'), ('coef_rng', 9, '.0f'),
            ('cond', 10, '.0f'), ('R2', 9, '.4f')]

    def fmt(label, count, disc, instab, m, n_extra, r2, mark=''):
        vals = {'disc': disc, 'instab': instab, 'n': m['n_terms'], 'extra': int(n_extra),
                'L2_eff': m['l2_eff'], 'pp_eff': m['pp_eff'], 'coef_rng': m['coef_range'],
                'cond': m['cond'], 'R2': r2}
        s = f"  {str(label):>8}{str(count):>4}"
        for name, w, f in cols:
            v = vals[name]
            s += (f"{v:>{w}.1e}" if f == '.1e' else f"{v:>{w}d}" if f == 'd'
                  else f"{v:>{w}.1f}" if f == '.1f' else f"{v:>{w}.0f}" if f == '.0f'
                  else f"{v:>{w}.4f}")
        return s + (f"   {mark}" if mark else '')

    hdr = f"  {'form':>8}{'#':>4}" + ''.join(f"{n:>{w}}" for n, w, _ in cols)
    for k in range(n_eq):
        var = all_vars[k]
        forms = sorted(groups[var].values(),
                       key=lambda r: (r['disc'] if r['disc'] == r['disc'] else 1e9))
        n_corr = sum(r['count'] for r in forms if r['ckey'] == truth_struct.get(var))
        print(f"\n=== eq[{k}] {var}  ({len(forms)} distinct discovered forms; "
              f"{n_corr}/{len(items)} solutions structurally correct) ===")
        print(hdr)
        td = truth_dec.get(var)
        if td:
            pe = truth_pe.get(var, (float('nan'), float('nan')))
            print(fmt('TRUE', '-', pe[0], pe[1], td[4], td[3], td[1], mark='<-- truth'))
        for rank, r in enumerate(forms[:8]):
            mark = '== truth struct' if r['ckey'] == truth_struct.get(var) else ''
            print(fmt(f'f{rank}', r['count'], r['disc'], r['instab'],
                      r['m'], r['n_extra'], r['r2'], mark=mark))
        if len(forms) > 8:
            print(f"      ... +{len(forms) - 8} more forms")

        # Decisive test: TRUTH's rank under each candidate objective among all
        # discovered forms (rank 1 = no form beats truth = the metric correctly
        # ranks truth best). Lower-is-better metrics.
        if td:
            cand = {'disc': lambda r: r['disc'], 'instab': lambda r: r['instab'],
                    'n_terms': lambda r: r['m']['n_terms'], 'coef_rng': lambda r: r['m']['coef_range'],
                    'gram_cond': lambda r: r['m']['cond'], 'pp_eff': lambda r: r['m']['pp_eff']}
            tvals = {'disc': pe[0], 'instab': pe[1], 'n_terms': td[4]['n_terms'],
                     'coef_rng': td[4]['coef_range'], 'gram_cond': td[4]['cond'],
                     'pp_eff': td[4]['pp_eff']}
            ranks = {name: 1 + sum(1 for r in forms if f(r) < tvals[name])
                     for name, f in cand.items()}
            print("   TRUTH rank by: " +
                  "  ".join(f"{name}={ranks[name]}" for name in cand))


if __name__ == '__main__':
    args = sys.argv[1:]
    main(args[0] if args else 'lorenz', args[1] if len(args) > 1 else None)
