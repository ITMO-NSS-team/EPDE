"""Find a per-term CV that KEEPS ac's weak diffusion (true, coef 1e-4) and
PRUNES ode's spurious u^3*u' (false, coef 1.2e-4). Both have tiny, highly
significant coefficients, so the only signal is the debiased region-variation
NC_deb -- but the normaliser decides whether the actual Lasso prune fires.

The Lasso keeps term j iff  |rho_j| >= cv_j * max_corr, i.e.  cv_j <= r_j,
where r_j = |rho_j|/max_corr is the term's relative correlation (rho computed
exactly as PhysicsInformedLasso iter-1: rho = X_aug^T (w*y)). So we want, for
each candidate cv:
    ac-diffusion : cv <= r   (KEEP)
    ode-spurious : cv >  r   (PRUNE)

Per-term quantities (gamma_0, var0, NC_raw, NC_deb, contrib=||gamma_0*phi||_w),
candidate cv formulas, and the verdict are tabulated for the two critical terms.
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

# (system, equation string, set of TRUE non-target term names, critical term
#  substring, want_keep)
ODE_SOL0 = ("-0.9936633679622026 * du/dx0{power: 1.0} * sin{power: 1.0, freq: 2.0, dim: 0.0} + -3.9868949002903613 * u{power: 1.0} + 0.00011940105848764598 * u{power: 3.0} * du/dx0{power: 1.0} + 1.4973740096088297 * x{power: 1.0, dim: 0.0} + 0.0 = d^2u/dx0^2{power: 1.0}")
AC_TRUTH = ("0.0001 * d^2u/dx1^2{power: 1.0} + -5.0 * u{power: 3.0} + 5.0 * u{power: 1.0} = du/dx0{power: 1.0}")

CASES = [
    ('ode', ODE_SOL0, 'u{power: 3.0}', False),   # spurious u^3*u' -> PRUNE
    ('ac',  AC_TRUTH, 'd^2u/dx1^2', True),        # weak diffusion  -> KEEP
]

# candidate cv formulas of the per-term quantities
FORMULAS = {
    'var0/g0^2':        lambda q: q['var0'] / (q['C'] + 1e-30),
    'NCraw/g0^2':       lambda q: q['nc_raw'] / (q['C'] + 1e-30),
    'NCdeb/g0^2':       lambda q: q['nc_deb'] / (q['C'] + 1e-30),
    'NCdeb':            lambda q: q['nc_deb'],
    'NCdeb/var0':       lambda q: q['nc_deb'] / (q['var0'] + 1e-30),
    'NCdeb/(g0^2+var0)':lambda q: q['nc_deb'] / (q['C'] + q['var0'] + 1e-30),
    'sqrt(NCdeb)/|g0|': lambda q: np.sqrt(q['nc_deb']) / (abs(q['g0']) + _EPS),
    'NCdeb/contrib^2':  lambda q: q['nc_deb'] / (q['contrib'] ** 2 + 1e-30),
    'NCdeb/contrib':    lambda q: q['nc_deb'] / (q['contrib'] + _EPS),
    'NCdeb*var0/g0^4':  lambda q: q['nc_deb'] * q['var0'] / (q['C'] ** 2 + 1e-30),
    'NCdeb/(g0^2*contrib)': lambda q: q['nc_deb'] / (q['C'] * q['contrib'] + 1e-30),
}


def analyse(system, eq_str, crit_sub):
    cfg = load_config(system)
    _set_seeds(0)
    search = build_pool_only(cfg, pipeline_settings('new'))
    sw = np.asarray(gv.grid_cache.g_func[gv.grid_cache.g_func_mask]).reshape(-1)
    gshape = gv.grid_cache.inner_shape

    soeq = translate_equation(_normalize_grid_labels(eq_str), search.pool,
                              all_vars=['u'])
    eq = soeq.vals['u']
    eq.main_var_to_explain = 'u'
    eq.weights_internal = np.ones(len(eq.structure) - 1)
    eq.weights_internal_evald = True
    eq.weights_final_evald = True
    _, target, feats = eq.evaluate(normalize=True, return_val=False)
    feats = np.asarray(feats, dtype=float)
    y = np.asarray(target, dtype=float).reshape(-1)
    feat_terms = [t for i, t in enumerate(eq.structure) if i != eq.target_idx]
    n = len(feat_terms)

    # rho exactly as PhysicsInformedLasso iter-1: X_aug = [feats, 1], rho = X^T(w y)
    X_aug = np.hstack([feats, np.ones((feats.shape[0], 1))])
    X_T_y = X_aug.T @ (sw * y)
    max_corr = float(np.max(np.abs(X_T_y)))
    r = np.abs(X_T_y) / (max_corr + 1e-30)   # relative correlation per col

    setup = VaryingCoefSetup(feats, y, sw, gshape, main_var='u',
                             fit_intercept=False)
    sol = setup._solve_gammas(None)
    gamma, var, B, mk = sol['gamma'], sol['var'], sol['B'], sol['mk']
    is_const = mk == 0
    nonconst = ~is_const

    out = []
    for i in range(n):
        sl = slice(i * B, (i + 1) * B)
        g = gamma[sl]
        v = var[sl]
        g0 = float(g[is_const][0])
        var0 = float(v[is_const][0])
        nc_g2 = g[nonconst] ** 2
        nc_raw = float(np.sum(nc_g2))
        nc_deb = float(np.sum(np.maximum(nc_g2 - v[nonconst], 0.0)))
        contrib = float(np.sqrt(np.sum(sw * (g0 * feats[:, i]) ** 2)))
        q = {'g0': g0, 'C': g0 * g0, 'var0': var0, 'nc_raw': nc_raw,
             'nc_deb': nc_deb, 'contrib': contrib}
        out.append({'name': feat_terms[i].name, 'r': float(r[i]), 'q': q,
                    'crit': crit_sub in feat_terms[i].name})
    return out


def main():
    crit = {}
    for system, eq_str, crit_sub, want_keep in CASES:
        rows = analyse(system, eq_str, crit_sub)
        print(f"\n##### {system}: r=|rho|/max_corr per term (keep iff cv<=r)")
        for row in rows:
            star = ' *<-- CRITICAL' if row['crit'] else ''
            print(f"  {row['name'][:40]:40s} r={row['r']:.4g} "
                  f"g0={row['q']['g0']:.3g} NCdeb={row['q']['nc_deb']:.3g} "
                  f"contrib={row['q']['contrib']:.3g}{star}")
        for row in rows:
            if row['crit']:
                crit[system] = {'r': row['r'], 'q': row['q'],
                                'want_keep': want_keep}

    # the decisive table: for each formula, cv and verdict on both critical terms
    print(f"\n{'='*78}\nFORMULA SCREEN  (need ac KEEP: cv<=r  AND  ode PRUNE: cv>r)\n{'='*78}")
    hdr = (f"{'formula':22s} | {'ac cv':>10} {'ac r':>9} {'keep?':>6} | "
           f"{'ode cv':>10} {'ode r':>9} {'prune?':>7} | {'BOTH':>5}")
    print(hdr)
    print('-' * len(hdr))
    for fname, f in FORMULAS.items():
        ac = crit['ac']
        ode = crit['ode']
        ac_cv = float(f(ac['q']))
        ode_cv = float(f(ode['q']))
        ac_keep = ac_cv <= ac['r']
        ode_prune = ode_cv > ode['r']
        both = ac_keep and ode_prune
        print(f"{fname:22s} | {ac_cv:>10.3g} {ac['r']:>9.3g} "
              f"{('yes' if ac_keep else 'NO'):>6} | "
              f"{ode_cv:>10.3g} {ode['r']:>9.3g} "
              f"{('yes' if ode_prune else 'NO'):>7} | "
              f"{('YES' if both else '-'):>5}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
