"""Extract distinct full discovered lorenz SYSTEMS and flag those whose EVERY
equation generalises out-of-sample as well as truth (CV-WAPE <= tol x truth-eq
CV-WAPE) -- candidate VALID analytical-alternative systems to credit in
lorenz.yaml truth_alternatives. Prints each candidate's 3 equations + per-eq
in-sample/CV WAPE so the algebra can be hand-verified before adding.
"""
import os
import sys
import json

_THIS = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.abspath(os.path.join(_THIS, '..', '..')), _THIS):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np

from term_contribution import build_pool
from analyze_pareto import _clean_lines, _canon_term, _truth_terms_by_target
from lorenz_dominator_check import _design, _insample_wape, _cv_wape
from thesis_runner import _load_yaml, CONFIGS_DIR
from epde.interface.equation_translator import parse_equation_str

TOL = 2.0   # an equation "generalises like truth" if CV-WAPE <= TOL x truth-eq CV


def main(system='lorenz'):
    rec = json.load(open(os.path.join(_THIS, 'results', system, 'new_rep00.json'), encoding='utf-8'))
    sols = rec.get('discovered_text_per_solution') or []
    hams = rec.get('hamming_per_solution') or []
    truth_eqs = _load_yaml(os.path.join(CONFIGS_DIR, f'{system}.yaml')).get('truth_equations') or []
    n_eq = len(truth_eqs)
    cfg, search, all_vars = build_pool(system)

    truth_cv = {}
    for k in range(n_eq):
        Xt, yt, _ = _design(truth_eqs[k], search.pool, all_vars)
        truth_cv[all_vars[k]] = _cv_wape(Xt, yt)
    print("truth per-eq CV-WAPE: " +
          "  ".join(f"{all_vars[k]}={truth_cv[all_vars[k]]:.2e}" for k in range(n_eq)))

    uniq = {}
    for i, s in enumerate(sols):
        key = tuple(_clean_lines(s))
        if len(key) == n_eq and key not in uniq:
            uniq[key] = i

    rows = []
    for key, i in uniq.items():
        per_eq, ratios = [], []
        for k in range(n_eq):
            try:
                X, y, _ = _design(key[k], search.pool, all_vars)
                inw, cvw = _insample_wape(X, y), _cv_wape(X, y)
            except Exception:
                inw, cvw = float('nan'), float('inf')
            per_eq.append((inw, cvw))
            ratios.append(cvw / truth_cv[all_vars[k]])
        rows.append((max(ratios), hams[i] if i < len(hams) else None, key, per_eq, ratios))

    n_ok = sum(1 for r in rows if r[0] <= TOL)
    print(f"\n{n_ok}/{len(rows)} systems have ALL eqs generalising (worst-eq CV <= {TOL}x truth).")
    print(f"Systems sorted by worst-equation CV-ratio (lower = closer to fully-valid); "
          f"the eq marked * is the offender:\n")
    for worst, ham, key, per_eq, ratios in sorted(rows)[:8]:
        print(f"{'='*92}\n  hamming={ham}   worst-eq CV-ratio={worst:.1f}x truth")
        for k in range(n_eq):
            inw, cvw = per_eq[k]
            flag = ' *OVERFIT' if ratios[k] > TOL else ''
            print(f"   [{all_vars[k]}] CV={cvw:.2e} ({ratios[k]:.1f}x){flag}   {key[k][:110]}")


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'lorenz')
