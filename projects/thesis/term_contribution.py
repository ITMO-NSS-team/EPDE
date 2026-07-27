"""Per-term coef x term contribution analysis, TRUE vs SPURIOUS, across the
thesis systems.

For each system we build the SAME EPDE token pool / preprocessing the discovery
uses (configs/<sys>.yaml + adapters/<sys>.py), then for every truth equation we
assemble a candidate library = {true LHS terms} + {a few injected SPURIOUS terms}
and OLS-refit it against the true target term. Each term's CONTRIBUTION is the
coefficient times the evaluated term field, c_j * theta_j(x), summarised as its
RMS over the domain (and normalised to a % share). A well-posed objective should
give the TRUE terms a large share and collapse the SPURIOUS ones to ~0; we also
report each term's leave-one-out Delta-R^2 (necessity).

Usage:
    python projects/thesis/term_contribution.py [sys1 sys2 ...]   # default: all
"""
from __future__ import annotations

import os
import re
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, '..', '..'))
for p in (_REPO_ROOT, _THIS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
from sklearn.linear_model import LinearRegression

from thesis_runner import (load_config, _build_token_pool, _construct_search,
                           _configure_preprocessor, pipeline_settings, _load_yaml,
                           CONFIGS_DIR)
from epde.interface.equation_translator import (parse_equation_str, parse_factor,
                                                float_convertable)
from epde.structure.main_structures import Term

# Injected spurious terms per system: plausible-but-wrong single factors that
# exist in the pool (derivatives at power 1 within max_deriv_order; polynomial
# powers <= data_fun_pow=3; deriv powers <= deriv_fun_pow=2). Skipped silently
# if a token is absent from a given system's pool.
SPURIOUS = {
    'ac':              ['du/dx1{power: 1.0}', 'd^2u/dx0^2{power: 1.0}', 'u{power: 2.0}'],
    'kdv':             ['d^2u/dx1^2{power: 1.0}', 'u{power: 2.0}', 'd^4u/dx1^4{power: 1.0}'],
    'kdv_cossin':      ['d^2u/dx1^2{power: 1.0}', 'u{power: 2.0}'],
    'wave':            ['du/dx1{power: 1.0}', 'u{power: 1.0}', 'du/dx0{power: 1.0}'],
    'burgers_viscous': ['du/dx1{power: 1.0}', 'u{power: 2.0}', 'd^3u/dx1^3{power: 1.0}'],
    'burgers_inviscid':['d^2u/dx1^2{power: 1.0}', 'u{power: 1.0}', 'du/dx1{power: 1.0}'],
    'ks':              ['u{power: 2.0}', 'd^3u/dx1^3{power: 1.0}', 'du/dx1{power: 1.0}'],
    'pde_compound':    ['du/dx1{power: 1.0}', 'u{power: 2.0}', 'd^2u/dx1^2{power: 1.0}'],
    'pde_divide':      ['du/dx1{power: 2.0}', 'd^2u/dx1^2{power: 1.0}'],
    'vdp':             ['u{power: 2.0}', 'du/dx0{power: 2.0}', 'u{power: 3.0}'],
    'ode':             ['u{power: 2.0}', 'du/dx0{power: 1.0}'],
    'lv':              ['u{power: 2.0}', 'v{power: 2.0}'],
    'lorenz':          ['u{power: 2.0}', 'v{power: 2.0}', 'w{power: 2.0}'],
    'ns':              ['du/dx1{power: 1.0}', 'u{power: 2.0}'],
}

# HARD spurious: highly-collinear composites + coordinate/trig-modulated
# degenerate forms (the cos(x)*u_xx class) -- the forms the objective work
# actually fights, vs the easy single-factor SPURIOUS above. NOT the credited
# truth_alternatives identities (those must not be penalised).
HARD_SPURIOUS = {
    'ac': ['sin{power: 1.0, freq: 2.0, dim: 1.0} * d^2u/dx1^2{power: 1.0}',
           'x{power: 1.0, dim: 1.0} * d^2u/dx1^2{power: 1.0}',
           'u{power: 1.0} * d^2u/dx1^2{power: 1.0}'],
    'burgers_viscous': ['x{power: 1.0, dim: 1.0} * d^2u/dx1^2{power: 1.0}',
                        'u{power: 2.0} * du/dx1{power: 1.0}',
                        'sin{power: 1.0, freq: 2.0, dim: 1.0} * du/dx1{power: 1.0}'],
    'ns': ['u{power: 1.0} * du/dx1{power: 1.0}',      # u*u_y  (truth has v*u_y)
           'v{power: 1.0} * du/dx2{power: 1.0}',      # v*u_x  (truth has u*u_x)
           'x{power: 1.0, dim: 2.0} * d^2u/dx2^2{power: 1.0}'],
    'lorenz': ['u{power: 1.0} * v{power: 1.0}',       # into eq0 (truth 10v-10u)
               'u{power: 1.0} * w{power: 1.0}'],
}

_GRID = re.compile(r'\bx_(\d+)\b')   # collapse x_0/x_1 -> the pool's single 'x' family


def _norm_factor(fs: str) -> str:
    """Map truth-string grid-coord labels (x_0, x_1) onto the pool's 'x' family,
    keeping the dim param that selects the axis."""
    return _GRID.sub('x', fs)


def build_pool(system: str):
    cfg = load_config(system)
    coords, data, variable_names, dim = cfg.load_data()
    tokens = _build_token_pool(cfg, coords, dim)
    search = _construct_search(cfg, coords, pipeline_settings('new'))
    _configure_preprocessor(search, cfg)
    f = cfg.hparams['fit']
    mdo = f.get('max_deriv_order')
    mdo = tuple(mdo) if mdo is not None else (2,) + (4,) * dim
    search.create_pool(data=data, variable_names=variable_names, max_deriv_order=mdo,
                       additional_tokens=tokens, data_fun_pow=f['data_fun_pow'],
                       deriv_fun_pow=f['deriv_fun_pow'])
    return cfg, search, variable_names


def _term(factor_strs, pool, all_vars):
    factors = [parse_factor(_norm_factor(fs), pool, all_vars) for fs in factor_strs]
    return Term(pool, passed_term=factors, collapse_powers=False)


def _eval(term):
    return np.asarray(term.evaluate(False)).reshape(-1).astype(np.float64)


def analyze_equation(eqstr, spurious, pool, all_vars):
    """Return per-term rows [(label, is_true, coef, rms, share, dR2)] + full R^2."""
    parsed = parse_equation_str(eqstr)
    *left, right = parsed
    labels, is_true, cols = [], [], []
    for term in left:
        # term is a list of tokens; a leading float is the (ignored) true coef.
        factors = term[1:] if (float_convertable(term[0]) and len(term) > 1) else term
        if len(factors) == 1 and float_convertable(factors[0]):
            continue                       # bare constant -> handled by intercept
        labels.append(' * '.join(factors)); is_true.append(True)
        cols.append(_eval(_term(factors, pool, all_vars)))
    for sp in spurious:
        factors = sp.split(' * ')
        try:
            v = _eval(_term(factors, pool, all_vars))
        except Exception:
            continue                       # token not in this system's pool
        lbl = ' * '.join(factors)
        if lbl in labels:                  # spurious coincides with a true term
            continue
        labels.append(lbl); is_true.append(False); cols.append(v)

    target = _eval(_term(right, pool, all_vars))
    X = np.column_stack(cols)
    lr = LinearRegression().fit(X, target)
    coef = lr.coef_
    full_r2 = lr.score(X, target)
    contrib = coef[None, :] * X            # (N, k) = c_j * theta_j(x)
    rms = np.sqrt(np.mean(contrib ** 2, axis=0))
    share = rms / rms.sum() if rms.sum() > 0 else rms     # L2 (magnitude-weighted)
    # PER-POINT magnitude share: at each x, |c_j theta_j| / sum_k|c_k theta_k|
    # (a fraction summing to 1 per point), then EQUAL-weight average over points.
    # Sums to 100% across terms, bounded, robust -- contrasts with L2 which is
    # dominated by the high-|target| points.
    absc = np.abs(contrib)
    mass = absc.sum(axis=1)
    good = mass > 0
    pp = (absc[good] / mass[good, None]).mean(axis=0) if good.any() else np.zeros(absc.shape[1])
    # RELATIVE per-point: each term's signed contribution as a fraction of the
    # TARGET at that point, c_j*theta_j(x)/y(x). Diverges where y~=0, so the mean
    # is meaningless -- summarise with the MEDIAN (robust to the zero-crossing
    # tails). At a typical point the per-term fractions sum to ~1.
    with np.errstate(divide='ignore', invalid='ignore'):
        rel = np.where(target[:, None] != 0, contrib / target[:, None], np.nan)
    rel_med = np.nanmedian(rel, axis=0)
    # DOMINANCE: fraction of points where term j is the LARGEST |contribution|
    # (division-free, fully robust, sums to 100% across terms).
    winner = np.argmax(absc, axis=1)
    dom = np.array([float(np.mean(winner == j)) for j in range(absc.shape[1])])
    # Alternative normalisation: each term's contribution RMS vs the TARGET's
    # RMS. Can exceed 100% when true terms cancel (large opposing terms).
    target_rms = float(np.sqrt(np.mean(target ** 2)))
    share_tgt = rms / target_rms if target_rms > 0 else rms * 0.0
    # SIGNED variance-explained share: c_j*<theta_j,y>/<y,y> (centered). These
    # sum to R^2 (so terms + residual = 100% OF THE TARGET) and respect sign /
    # cancellation. Under collinearity a term can exceed 100% or go negative
    # (suppressor) -- itself a flag that the equation is ill-conditioned.
    yc = target - target.mean()
    denom = float(np.sum(yc * yc)) or 1.0
    expl = np.array([coef[j] * float(np.sum((X[:, j] - X[:, j].mean()) * yc)) / denom
                     for j in range(X.shape[1])])
    dR2 = []
    for j in range(X.shape[1]):
        keep = [c for c in range(X.shape[1]) if c != j]
        if keep:
            r2j = LinearRegression().fit(X[:, keep], target).score(X[:, keep], target)
        else:
            r2j = 0.0   # dropping the only feature -> predict the mean -> R^2 = 0
        dR2.append(full_r2 - r2j)
    rows = list(zip(labels, is_true, coef, rms, share, share_tgt, expl, pp, rel_med, dom, dR2))
    return rows, full_r2, ' * '.join(right)


def main(systems, hard=False):
    for system in systems:
        print(f"\n{'='*78}\n### {system}{'  [HARD spurious]' if hard else ''}\n{'='*78}")
        try:
            cfg, search, all_vars = build_pool(system)
        except Exception as exc:
            print(f"  [skip] pool build failed: {exc!r}")
            continue
        truth = (_load_yaml(os.path.join(CONFIGS_DIR, f'{system}.yaml'))
                 .get('truth_equations') or [])
        spurious = (HARD_SPURIOUS.get(system, []) if hard else SPURIOUS.get(system, []))
        for ei, eqstr in enumerate(truth):
            try:
                rows, full_r2, tgt = analyze_equation(eqstr, spurious, search.pool, all_vars)
            except Exception as exc:
                print(f"  eq[{ei}] [skip] {exc!r}")
                continue
            print(f"\n  eq[{ei}]  target = {tgt}   (OLS full R^2 = {full_r2:.5f})")
            print(f"    {'term':<34}{'kind':>8}{'L2':>8}{'per-pt':>8}"
                  f"{'rel(med)':>10}{'dom%':>8}{'tgt%':>9}")
            for (lbl, tr, coef, rms, share, share_tgt, expl, pp,
                 relm, dom, dr2) in sorted(rows, key=lambda r: -r[3]):
                kind = 'TRUE' if tr else 'spur'
                print(f"    {lbl:<34}{kind:>8}{share:>8.1%}{pp:>8.1%}"
                      f"{relm:>10.1%}{dom:>8.1%}{expl:>9.1%}")
            expl_sum = sum(r[6] for r in rows)
            print(f"    -> L2/per-pt/dom% each sum to 100% among terms; "
                  f"rel(med)=median per-point fraction of target (need NOT sum to 100); "
                  f"tgt% sums to {expl_sum:.1%}=R^2")


if __name__ == '__main__':
    argv = sys.argv[1:]
    hard = '--hard' in argv
    args = [a for a in argv if a != '--hard'] or (
        list(HARD_SPURIOUS.keys()) if hard else list(SPURIOUS.keys()))
    main(args, hard=hard)
