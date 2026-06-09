"""Analyze the 4 ode Pareto solutions to find a per-term CV that prunes the
tiny-coefficient spurious terms (the systematic hamming=1) while keeping the
true terms.

Each discovered ode equation has the 3 TRUE terms (u, du/dx0*sin, x; coef O(1))
plus 1-2 SPURIOUS terms with coef 1e-4..1e-2. The current CV var0/g0^2 (=1/t^2)
keeps the spurious ones because on clean data they are precisely estimated
(significant). We tabulate candidate per-term statistics, tag true/spurious, and
look for one where spurious >> true so it could drive pruning.

Candidates (per term j; gamma_0 = const coef, var0 = Var(gamma_0), gamma_k =
basis modes, phi_j = feature column, y = target, w = sample weights):
  cv_sig    = var0 / gamma_0^2                 significance 1/t^2 (current)
  ncraw_g2  = sum gamma_k^2 / gamma_0^2        region-variation ratio
  mad_med   = mad_x(beta)/|median_x(beta)|     robust region-variation
  contrib   = ||gamma_0 phi_j||_w              the signal the term adds to y
  frac      = contrib / ||y||_w                fraction of target explained
  inv_frac  = ||y||_w / contrib                small-contribution -> large
  relmag    = max_k(gamma_0_k^2) / gamma_0_j^2 inverse relative magnitude^2
  sig_x_mag = cv_sig * relmag                  significance scaled by smallness
"""
from __future__ import annotations
import os
import sys

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_THIS, '..', '..'))
for _p in (_ROOT, _THIS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import epde.globals as gv
from epde.interface.equation_translator import translate_equation
from epde.operators.common.stability import VaryingCoefSetup
from kdv_sindy_test import build_pool_only, _normalize_grid_labels
from thesis_runner import load_config, pipeline_settings, _set_seeds

_EPS = 1e-12

TRUTH = ("-4.0 * u{power: 1.0} + -1.0 * du/dx0{power: 1.0} * "
         "sin{power: 1.0, freq: 2.0, dim: 0.0} + "
         "1.5 * x{power: 1.0, dim: 0.0} = d^2u/dx0^2{power: 1.0}")

EQS = [
    "-0.9936633679622026 * du/dx0{power: 1.0} * sin{power: 1.0, freq: 2.0, dim: 0.0} + -3.9868949002903613 * u{power: 1.0} + 0.00011940105848764598 * u{power: 3.0} * du/dx0{power: 1.0} + 1.4973740096088297 * x{power: 1.0, dim: 0.0} + 0.0 = d^2u/dx0^2{power: 1.0}",
    "-0.008456117004594182 * du/dx0{power: 1.0} * d^2u/dx0^2{power: 1.0} + -3.999889678771497 * u{power: 1.0} + 1.4981092221855905 * x{power: 1.0, dim: 0.0} + -1.0293913688973846 * du/dx0{power: 1.0} * sin{power: 1.0, freq: 2.0, dim: 0.0} + 0.0 = d^2u/dx0^2{power: 1.0}",
    "-3.999955869355026 * u{power: 1.0} + 1.4980032313263567 * x{power: 1.0, dim: 0.0} + -0.9757548527888852 * du/dx0{power: 1.0} * sin{power: 1.0, freq: 2.0, dim: 0.0} + -0.0036890595533533733 * d^2u/dx0^2{power: 1.0} * du/dx0{power: 1.0} + 0.011783526161822731 * du/dx0{power: 2.0} + 0.0 = d^2u/dx0^2{power: 1.0}",
    "-3.9994631249571406 * u{power: 1.0} + 1.4983573574999929 * x{power: 1.0, dim: 0.0} + -1.02610182727155 * du/dx0{power: 1.0} * sin{power: 1.0, freq: 2.0, dim: 0.0} + -0.00742408614998562 * du/dx0{power: 1.0} * d^2u/dx0^2{power: 1.0} + 0.0003709182161524908 * du/dx0{power: 1.0} * u{power: 2.0} + 0.0 = d^2u/dx0^2{power: 1.0}",
]


def main():
    cfg = load_config('ode')
    _set_seeds(0)
    search = build_pool_only(cfg, pipeline_settings('new'))
    sw = np.asarray(gv.grid_cache.g_func[gv.grid_cache.g_func_mask]).reshape(-1)
    gshape = gv.grid_cache.inner_shape

    tsoeq = translate_equation(_normalize_grid_labels(TRUTH), search.pool,
                               all_vars=['u'])
    teq = tsoeq.vals['u']
    truth_names = {t.name for i, t in enumerate(teq.structure)
                   if i != teq.target_idx}

    for idx, eqs in enumerate(EQS):
        soeq = translate_equation(_normalize_grid_labels(eqs), search.pool,
                                  all_vars=['u'])
        eq = soeq.vals['u']
        eq.main_var_to_explain = 'u'
        eq.weights_internal = np.ones(len(eq.structure) - 1)
        eq.weights_internal_evald = True
        eq.weights_final_evald = True
        _, target, feats = eq.evaluate(normalize=True, return_val=False)
        feats = np.asarray(feats, dtype=float)
        y = np.asarray(target, dtype=float).reshape(-1)
        feat_terms = [t for i, t in enumerate(eq.structure)
                      if i != eq.target_idx]
        n = len(feat_terms)

        setup = VaryingCoefSetup(feats, y, sw, gshape, main_var='u',
                                 fit_intercept=False)
        sol = setup._solve_gammas(None)
        gamma, var, B, mk = sol['gamma'], sol['var'], sol['B'], sol['mk']
        Bvals = setup._Bvals
        is_const = mk == 0
        nonconst = ~is_const

        yw = float(np.sqrt(np.sum(sw * y * y)))
        rows = []
        g0s = np.empty(n)
        for i in range(n):
            sl = slice(i * B, (i + 1) * B)
            g = gamma[sl]
            v = var[sl]
            g0 = float(g[is_const][0])
            g0s[i] = g0
        maxg2 = float(np.max(g0s ** 2))
        for i in range(n):
            sl = slice(i * B, (i + 1) * B)
            g = gamma[sl]
            v = var[sl]
            g0 = float(g[is_const][0])
            var0 = float(v[is_const][0])
            C = g0 * g0
            nc_raw = float(np.sum(g[nonconst] ** 2))
            beta = Bvals @ g
            med = float(np.median(beta))
            mad = float(np.median(np.abs(beta - med)))
            contrib = float(np.sqrt(np.sum(sw * (g0 * feats[:, i]) ** 2)))
            frac = contrib / (yw + _EPS)
            rows.append({
                'name': feat_terms[i].name,
                'true': feat_terms[i].name in truth_names,
                'g0': g0,
                'cv_sig': var0 / (C + 1e-30),
                'ncraw_g2': nc_raw / (C + 1e-30),
                'mad_med': mad / (abs(med) + _EPS),
                'frac': frac,
                'inv_frac': 1.0 / (frac + _EPS),
                'relmag': maxg2 / (C + 1e-30),
                'sig_x_mag': (var0 / (C + 1e-30)) * (maxg2 / (C + 1e-30)),
            })

        print(f"\n=== Solution {idx} ===")
        hdr = (f"{'term':34s} {'T?':3} {'g0':>10} {'cv_sig':>9} "
               f"{'ncraw_g2':>9} {'mad_med':>8} {'frac':>9} {'inv_frac':>9} "
               f"{'relmag':>9} {'sig_x_mag':>10}")
        print(hdr)
        print('-' * len(hdr))
        for r in rows:
            print(f"{r['name'][:34]:34s} {'T' if r['true'] else 'SP':3} "
                  f"{r['g0']:>10.3g} {r['cv_sig']:>9.2g} {r['ncraw_g2']:>9.2g} "
                  f"{r['mad_med']:>8.2g} {r['frac']:>9.3g} {r['inv_frac']:>9.3g} "
                  f"{r['relmag']:>9.3g} {r['sig_x_mag']:>10.2g}")
        # per-formula separation: min over SPUR - max over TRUE (want > 0)
        tv = {k: [r[k] for r in rows if r['true']] for k in
              ('cv_sig', 'ncraw_g2', 'mad_med', 'inv_frac', 'relmag', 'sig_x_mag')}
        sv = {k: [r[k] for r in rows if not r['true']] for k in tv}
        print("  separation (min_spur - max_true, want>0):")
        for k in tv:
            if sv[k] and tv[k]:
                sep = min(sv[k]) - max(tv[k])
                print(f"    {k:12s} {sep:>11.3g}"
                      f"  {'OK' if sep > 0 else 'x'}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
