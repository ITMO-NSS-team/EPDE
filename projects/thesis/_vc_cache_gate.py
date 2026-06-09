"""Equivalence gate for the cv_cache-style vcoef stability reuse.

For each system, seed the truth equation, run the real sparsity pass (which now
caches per-term stability scores on ``eq._cached_vc_score``), then check:

  (a) WIRING / no-super equality:
        sum(eq._cached_vc_score)  ==  vc_stability_total_lr(normalize=True feats)
      Both use the same normalize=True regime + the same score(); must match to
      ~fp. This is exactly what the fitness reuse branch sums.

  (b) SCALE-INVARIANCE (the super-Gram path assumption):
        vc_stability_total_lr(normalize=True)  ~=  vc_stability_total_lr(normalize=False)
      The super-Gram is built from RAW term values while the in-place fitness
      uses normalized features; reuse is lossless only if the score is invariant
      to per-column scaling. This checks it on real equations.

PASS => the cache reuse is lossless under the active vc_score_formula.
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
import yaml
import epde.globals as gv
from epde.interface.equation_translator import translate_equation
from epde.operators.common.fitness import vc_stability_total_lr
from kdv_sindy_test import build_pool_only, make_fit_operator, _normalize_grid_labels
from thesis_runner import load_config, pipeline_settings, _set_seeds

_SYSTEMS = ['lorenz', 'ks', 'ac']
_TOL = 1e-9


def _names(eq):
    return [t.name for i, t in enumerate(eq.structure) if i != eq.target_idx]


def check(system):
    cfg = load_config(system)
    _set_seeds(0)
    gv.set_gram_config('vcoef')
    search = build_pool_only(cfg, pipeline_settings('new'))
    fit_op = make_fit_operator()
    coords, data, variable_names, dim = cfg.load_data()
    all_vars = list(variable_names)
    truth = yaml.safe_load(open(os.path.join(_THIS, 'configs', f'{system}.yaml')))
    teqs = truth.get('truth_equations') or []
    seeded = (teqs[0] if len(all_vars) == 1
              else {v: teqs[i] for i, v in enumerate(all_vars)})
    seeded = _normalize_grid_labels(seeded)
    tsoeq = translate_equation(seeded, search.pool, all_vars=all_vars)

    g_fun = gv.grid_cache.g_func[gv.grid_cache.g_func_mask].reshape(-1)
    data_shape = gv.grid_cache.inner_shape

    rows = []
    for v in all_vars:
        eq = tsoeq.vals[v]
        eq.main_var_to_explain = v
        eq.weights_internal = np.ones(len(eq.structure) - 1)
        eq.weights_internal_evald = True
        eq.weights_final_evald = True
        # Run the real sparsity (force_out_of_place) -> sets _cached_vc_score.
        fit_op.apply(eq, {}, force_out_of_place=True)

        cached_vec = getattr(eq, '_cached_vc_score', None)
        cached = None if cached_vec is None else float(np.sum(cached_vec))

        fit_int = bool(eq.weights_internal[-1] != 0)
        _, t_n, f_n = eq.evaluate(normalize=True, return_val=False)
        _, t_r, f_r = eq.evaluate(normalize=False, return_val=False)
        fresh_norm = (None if f_n is None else
                      vc_stability_total_lr(f_n, t_n, g_fun, data_shape,
                                            main_var=v, fit_intercept=fit_int))
        fresh_raw = (None if f_r is None else
                     vc_stability_total_lr(f_r, t_r, g_fun, data_shape,
                                           main_var=v, fit_intercept=fit_int))
        rows.append((v, cached, fresh_norm, fresh_raw))
    return rows


def main():
    print(f"vc_score_formula = {getattr(gv, 'vc_score_formula', '?')}  tol={_TOL}\n")
    all_ok = True
    for system in _SYSTEMS:
        try:
            rows = check(system)
        except Exception as e:
            import traceback
            print(f"{system:8s} ERROR {type(e).__name__}: {str(e)[:80]}")
            traceback.print_exc()
            all_ok = False
            continue
        for (v, cached, fn, fr) in rows:
            # (a) wiring equality: cached == fresh_norm
            wire_ok = (cached is not None and fn is not None
                       and abs(cached - fn) <= _TOL * (1 + abs(fn)))
            # (b) scale-invariance: fresh_norm ~= fresh_raw
            si_rel = (abs(fn - fr) / (1 + abs(fn))) if (fn is not None and fr is not None) else float('nan')
            si_ok = si_rel <= 1e-6
            flag = '' if (wire_ok and si_ok) else '   <-- FAIL'
            if not (wire_ok and si_ok):
                all_ok = False
            print(f"{system:8s}[{v}] cached={cached!r} fresh_norm={fn!r} "
                  f"fresh_raw={fr!r} wire_ok={wire_ok} si_rel={si_rel:.2e}{flag}")
    print("\n" + ("ALL PASS" if all_ok else "SOME FAILED"))
    return 0 if all_ok else 1


if __name__ == '__main__':
    sys.exit(main())
