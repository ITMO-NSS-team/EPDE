"""Signal check: is the known law visible in the data at a given noise level?

For every equation of the truth, its terms are evaluated on the data
(the native EPDE preprocessing or the supplied derivative stacks), the target is regressed on them by least squares, and we report

* R^2 of the true structure -- 1 means the law holds on the sampled data;
* the fitted coefficients next to the true ones;
* for every term, the R^2 lost when it is dropped -- a small loss suggests weak
  identifiability under this preprocessing and regression diagnostic.

This replaces the ``*_test`` functions of the old per-system scripts (which
compared a correct and an incorrect equation through EPDE's fitness
operators): it needs no search object, runs in seconds and explains WHY a
discovery fails ("the diffusion term of Allen-Cahn carries 0.01 % of the
variance") rather than only THAT it does.
"""

import numpy as np

from .library import _blocks, _interior
from .config import load_config, resolve_for_problem
from .datasets import load
from .metrics import _parse_factor, _parse_term_with_coef


def _factor_arrays(problem, data, search):
    blocks, lhs = _blocks(problem, data, search, 'epde')
    arrays = {}
    for name, array, *_ in blocks:
        arrays[_parse_factor(name)] = array
    for options in lhs.values():
        for name, array, _ in options:
            arrays[_parse_factor(name)] = array
    return arrays


def _term_array(term, arrays):
    out = None
    for factor in term:
        if factor not in arrays:
            raise KeyError(f'factor {factor[0]} {dict(factor[1])} is not in the evaluated pool')
        out = arrays[factor] if out is None else out * arrays[factor]
    return out


def check_equation(eq_text, arrays, inner):
    lhs_text, rhs_text = eq_text.split('=', 1)
    target_term, target_coef = _parse_term_with_coef(rhs_text)
    terms = [_parse_term_with_coef(t) for t in lhs_text.split('+')]
    terms = [t for t in terms if t is not None]
    y = (_term_array(target_term, arrays) * target_coef)[inner].ravel()
    if not terms or not y.size:
        raise ValueError('signal check requires a nonempty interior and at least one nonconstant term')
    X = np.stack([_term_array(t, arrays)[inner].ravel() for t, _ in terms], axis=1)
    X1 = np.column_stack([X, np.ones(len(y))])
    var_y = float(np.var(y)) or 1.0

    def r2(cols):
        coef, *_ = np.linalg.lstsq(X1[:, cols], y, rcond=None)
        return 1 - float(np.var(y - X1[:, cols] @ coef)) / var_y, coef

    full = list(range(X1.shape[1]))
    r2_full, coef = r2(full)
    rows = []
    for j, (term, true_coef) in enumerate(terms):
        r2_drop, _ = r2([c for c in full if c != j])
        rows.append({'term': ' * '.join(f'{name}{dict(params)}' if params else name
                                        for name, params in sorted(term, key=repr)),
                     'true': true_coef, 'fitted': float(coef[j]),
                     'r2_loss_if_dropped': r2_full - r2_drop})
    return {'equation': eq_text, 'r2': r2_full, 'intercept': float(coef[-1]), 'terms': rows}


def check(dataset: str, noise_levels=(0.0,), seed: int = 0, variant: str = 'default',
          overrides=None, config_path=None):
    cfg = load_config(dataset, variant, overrides, config_path)
    problem = load(dataset, **cfg.get('loader', {}))
    if not problem.truth:
        raise SystemExit(f'{dataset}: the truth is unknown, nothing to check')
    search = resolve_for_problem(cfg, problem)
    inner = _interior(problem, search)
    report = []
    for noise in noise_levels:
        arrays = _factor_arrays(problem, problem.noisy(noise, seed), search)
        report.append({'noise': noise,
                       'equations': [check_equation(eq, arrays, inner) for eq in problem.truth]})
    return problem, report


def print_check(problem, report):
    print(problem.summary())
    for block in report:
        print(f"\n=== noise {block['noise']:g} % ===")
        for eq in block['equations']:
            print(f"R^2 of the true structure: {eq['r2']:.6f}   ({eq['equation'].split('=')[1].strip()})")
            print(f"  {'term':58} {'true':>10} {'fitted':>12} {'R^2 lost if dropped':>20}")
            for row in eq['terms']:
                print(f"  {row['term'][:58]:58} {row['true']:>10.4g} {row['fitted']:>12.5g} "
                      f"{row['r2_loss_if_dropped']:>20.3e}")
