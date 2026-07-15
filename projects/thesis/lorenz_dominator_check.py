"""Are the lorenz front forms that DOMINATE truth (lower disc AND lower instab,
per equation) valid ANALYTICAL ALTERNATIVES, or genuine OVERFITS?

A valid analytical alternative is an algebraic identity for the data family -> it
holds out-of-sample (CV discrepancy ~ in-sample). A genuine overfit drives
IN-SAMPLE residual below truth by fitting region-specific FD-error/noise with
extra terms -> its CV discrepancy blows up (and exceeds truth's). For each
truth-dominating form we print the equation, its EXTRA terms, in-sample WAPE,
and 5-fold contiguous-block CV WAPE, vs truth's.
"""
import os
import sys
import json

_THIS = os.path.dirname(os.path.abspath(__file__))
for p in (os.path.abspath(os.path.join(_THIS, '..', '..')), _THIS):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
from sklearn.linear_model import LinearRegression

from term_contribution import build_pool, _term, _eval
from analyze_pareto import (_clean_lines, _truth_objectives, _canon_term,
                            _truth_terms_by_target)
from thesis_runner import _load_yaml, CONFIGS_DIR
from epde.interface.equation_translator import parse_equation_str, float_convertable


def _wape(y, yhat):
    return float(np.sum(np.abs(y - yhat)) / (np.sum(np.abs(y)) + 1e-12))


def _design(line, pool, all_vars):
    *left, right = parse_equation_str(line)
    cols, labels = [], []
    for term in left:
        factors = term[1:] if (float_convertable(term[0]) and len(term) > 1) else term
        if len(factors) == 1 and float_convertable(factors[0]):
            continue
        cols.append(_eval(_term(factors, pool, all_vars)))
        labels.append(' * '.join(factors))
    return np.column_stack(cols), _eval(_term(right, pool, all_vars)), labels


def _cv_wape(X, y, k=5):
    n = len(y)
    folds = np.array_split(np.arange(n), k)
    errs = []
    for f in folds:
        te = np.zeros(n, bool); te[f] = True; tr = ~te
        lr = LinearRegression().fit(X[tr], y[tr])
        errs.append(_wape(y[te], lr.predict(X[te])))
    return float(np.mean(errs))


def _insample_wape(X, y):
    lr = LinearRegression().fit(X, y)
    return _wape(y, lr.predict(X))


def main(system='lorenz'):
    rec = json.load(open(os.path.join(_THIS, 'results', system, 'new_rep00.json'), encoding='utf-8'))
    sols = rec.get('discovered_text_per_solution') or []
    objs = rec.get('objectives_per_solution') or []
    truth_eqs = _load_yaml(os.path.join(CONFIGS_DIR, f'{system}.yaml')).get('truth_equations') or []
    truth_map = _truth_terms_by_target(truth_eqs)
    n_eq = len(truth_eqs)

    cfg, search, all_vars = build_pool(system)
    truth_pe = _truth_objectives({all_vars[k]: truth_eqs[k] for k in range(n_eq)}, all_vars, search)

    # unique solutions
    uniq = {}
    for i, s in enumerate(sols):
        key = tuple(_clean_lines(s))
        if key and key not in uniq:
            uniq[key] = i

    for k in range(n_eq):
        var = all_vars[k]
        t_disc, t_instab = truth_pe.get(var, (float('nan'), float('nan')))
        Xt, yt, _ = _design(truth_eqs[k], search.pool, all_vars)
        t_in, t_cv = _insample_wape(Xt, yt), _cv_wape(Xt, yt)
        true_set = truth_map.get(_canon_term(parse_equation_str(truth_eqs[k])[-1]), set())
        print(f"\n{'='*90}\n### eq[{k}] {var}   truth: disc={t_disc:.2e} instab={t_instab:.2e}  "
              f"WAPE in={t_in:.2e} CV={t_cv:.2e}")

        seen = set()
        doms = []
        for key, i in uniq.items():
            if k >= len(key):
                continue
            ov = objs[i] if i < len(objs) else None
            if not (ov and len(ov) == 2 * n_eq):
                continue
            d, s = ov[k], ov[n_eq + k]
            if d < t_disc and s < t_instab:                  # dominates truth on BOTH
                line = key[k]
                if line in seen:
                    continue
                seen.add(line)
                doms.append((d, s, line))
        if not doms:
            print("  (no form dominates truth on BOTH objectives)")
            continue
        for d, s, line in sorted(doms):
            X, y, labels = _design(line, search.pool, all_vars)
            extra = [lb for lb in labels if _canon_term(lb.split(' * ')) not in true_set]
            in_w, cv_w = _insample_wape(X, y), _cv_wape(X, y)
            verdict = ('ANALYTICAL? generalises' if cv_w <= t_cv * 1.5
                       else 'OVERFIT (CV >> truth CV)')
            print(f"\n  dom: disc={d:.2e} instab={s:.2e}  WAPE in={in_w:.2e} CV={cv_w:.2e}  "
                  f"(truth CV={t_cv:.2e})  -> {verdict}")
            print(f"     EXTRA terms ({len(extra)}): {extra}")
            print(f"     {line}")


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'lorenz')
