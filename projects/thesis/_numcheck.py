"""Numerical-correctness check of VaryingCoefSetup on the REAL 14-dataset
inputs (the seeded-truth features/grids/g_func weights each system feeds the
estimator). Verifies, per (system, variable):

  A. G symmetric & finite (direct + super-Gram).
  B. super/from_full == direct construction (G, Phiy, yWy, score) -- the
     EqRPS fast path used in the live search.
  C. gamma_0 solves the weighted normal equations (|X^T W (y - X g0)| ~ 0) --
     robust to collinearity, unlike comparing to a separate WLS solve.
  D. Var(gamma_0) matches sigma^2 * diag((X^T W X)^-1) (reported; the
     equilibration floor makes it approximate on near-singular blocks).
  E. score finite, >= 0, no NaN/Inf anywhere.
  F. Parseval (exact, basis property): var(beta(x)) == NC_raw per feature.

Usage: python _numcheck.py [system ...]   (default: all 14)
Run, then delete."""
from __future__ import annotations
import os, sys, traceback

_THIS = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_THIS, '..', '..'))
for _p in (_ROOT, _THIS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import yaml
import epde.globals as gv
from epde.operators.common.stability import VaryingCoefSetup as VC
from kdv_sindy_test import build_pool_only, _normalize_grid_labels
from thesis_runner import load_config, pipeline_settings, _set_seeds
from vcoef_stat_compare import _ALL
from epde.interface.equation_translator import translate_equation


def _inputs(system):
    """[(var, Z (N,n_terms), target_idx, w (N,), grid_shape), ...] for truth."""
    cfg = load_config(system)
    _set_seeds(0)
    gv.set_gram_config('vcoef')
    search = build_pool_only(cfg, pipeline_settings('new'))
    coords, data, variable_names, dim = cfg.load_data()
    all_vars = list(variable_names)
    truth = yaml.safe_load(open(os.path.join(_THIS, 'configs', f'{system}.yaml')))
    teqs = truth.get('truth_equations') or []
    seeded = (teqs[0] if len(all_vars) == 1
              else {v: teqs[i] for i, v in enumerate(all_vars)})
    seeded = _normalize_grid_labels(seeded)
    soeq = translate_equation(seeded, search.pool, all_vars=all_vars)
    out = []
    for v in all_vars:
        eq = soeq.vals[v]
        eq.main_var_to_explain = v
        eq.weights_internal = np.ones(len(eq.structure) - 1)
        eq.weights_internal_evald = True
        eq.weights_final_evald = True
        eq.evaluate(normalize=False, return_val=False)   # populate grid cache
        Z = np.vstack([t.evaluate(False, grids=None)
                       for t in eq.structure]).T.astype(float)
        w = np.asarray(gv.grid_cache.g_func[gv.grid_cache.g_func_mask], float).reshape(-1)
        gshape = tuple(int(n) for n in gv.grid_cache.inner_shape)
        out.append((v, Z, int(eq.target_idx), w, gshape))
    return out


def _check(v, Z, tgt, w, gshape):
    N, n_terms = Z.shape
    feat_idx = [i for i in range(n_terms) if i != tgt]
    Xf = Z[:, feat_idx]
    yt = Z[:, tgt]
    direct = VC(Xf, yt, w, gshape, main_var=v, fit_intercept=True)
    sup = VC.precompute_super(Z, w, gshape, main_var=v)
    ff = VC.from_full(sup, tgt)

    m = {}
    # A. symmetric & finite
    G = direct.G
    m['Gsym'] = float(np.abs(G - G.T).max() / (np.abs(G).max() + 1e-30))
    m['finite'] = bool(np.all(np.isfinite(G)) and np.all(np.isfinite(direct.Phiy))
                       and np.all(np.isfinite(sup['G_super'])))
    # B. super/from_full == direct
    m['dG'] = float(np.abs(ff.G - direct.G).max() / (np.abs(direct.G).max() + 1e-30))
    m['dPhiy'] = float(np.abs(ff.Phiy - direct.Phiy).max() / (np.abs(direct.Phiy).max() + 1e-30))
    m['dyWy'] = float(abs(ff.yWy - direct.yWy) / (abs(direct.yWy) + 1e-30))
    sc_d = direct.score(None)
    sc_f = ff.score(None)
    m['dscore'] = float(np.abs(sc_f - sc_d).max())
    # C. gamma_0 weighted normal equations
    sol = direct._solve_gammas(None)
    B = sol['B']; nf = sol['nf']
    g0 = sol['gamma'][np.arange(nf) * B]            # const per feature (incl intercept)
    Xa = np.column_stack([Xf, np.ones(N)])
    XtWy = Xa.T @ (w * yt)
    resid = Xa.T @ (w * (yt - Xa @ g0))
    m['normeq'] = float(np.abs(resid).max() / (np.abs(XtWy).max() + 1e-30))
    # D. Var(gamma_0) vs sigma^2 diag((X^T W X)^-1)
    A = Xa.T @ (w[:, None] * Xa)
    Neff = float(w.sum())
    rss = max(float(yt @ (w * yt) - g0 @ XtWy), 0.0)
    sigma2 = rss / max(Neff - nf, 1.0)
    try:
        var_ref = sigma2 * np.diag(np.linalg.inv(A))
        var_vc = sol['var'][np.arange(nf) * B]
        rel = np.abs(var_vc - var_ref) / (np.abs(var_ref) + 1e-30)
        m['dVar'] = float(np.nanmax(rel))
    except np.linalg.LinAlgError:
        m['dVar'] = float('nan')
    m['condA'] = float(np.linalg.cond(A))
    # E. score sane
    m['score_ok'] = bool(np.all(np.isfinite(sc_d)) and np.all(sc_d >= -1e-9))
    # F. Parseval exact
    st = direct.beta_field_stats(None)
    g = sol['gamma']
    par = 0.0
    for i in range(nf):
        nc_raw = float(np.sum(g[i * B + 1:(i + 1) * B] ** 2))
        par = max(par, abs(st['std'][i] ** 2 - nc_raw) / (nc_raw + 1e-12))
    m['parseval'] = float(par)
    return m


# tolerances
TOL = dict(Gsym=1e-10, dG=1e-7, dPhiy=1e-7, dyWy=1e-9, dscore=1e-6,
           normeq=1e-6, parseval=1e-6)


def verdict(m):
    bad = []
    if not m['finite']:
        bad.append('NONFINITE')
    if not m['score_ok']:
        bad.append('score')
    for k, t in TOL.items():
        if not (m[k] <= t):
            bad.append(f'{k}={m[k]:.1e}')
    return bad


def main():
    systems = sys.argv[1:] or list(_ALL)
    n_ok = n_tot = 0
    for s in systems:
        try:
            rows = _inputs(s)
        except Exception as e:
            print(f"{s:18s} INPUT-ERROR {type(e).__name__}: {str(e)[:60]}")
            continue
        for (v, Z, tgt, w, gshape) in rows:
            n_tot += 1
            try:
                m = _check(v, Z, tgt, w, gshape)
            except Exception as e:
                print(f"{s:14s}/{v:3s} CHECK-ERROR {type(e).__name__}: {str(e)[:50]}")
                traceback.print_exc()
                continue
            bad = verdict(m)
            tag = f"{s}/{v}" if len(rows) > 1 else s
            if not bad:
                n_ok += 1
                print(f"{tag:18s} OK   superGmax={m['dG']:.0e} normeq={m['normeq']:.0e} "
                      f"parseval={m['parseval']:.0e} dVar={m['dVar']:.0e} condA={m['condA']:.0e}")
            else:
                print(f"{tag:18s} FAIL {', '.join(bad)}  (condA={m['condA']:.0e})")
    print(f"\n==== numerically correct: {n_ok}/{n_tot} (system,var) blocks ====")


if __name__ == '__main__':
    sys.exit(main())
