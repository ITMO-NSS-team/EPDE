"""Compare candidate vcoef CV statistics on the 14 systems: TRUE vs SPURIOUS.

We are collapsing the vcoef estimator to a SINGLE per-term statistic ``S`` that
drives both the in-fit Lasso pruning (threshold ``S_j * max_corr``) and the
MOEA/D stability objective (now ``sum_j S_j``). This script decides which ``S``
to use by measuring, on seeded truth across all 14 systems, whether each
candidate scores TRUE terms LOW and SPURIOUS confuser terms HIGH.

Per system x variable:
  * seed the truth equation, evaluate -> truth feature columns + target;
  * fit ONE ``VaryingCoefSetup`` on the truth-only design -> per-true-term
    statistics (``sum_true`` = the objective the true equation gets;
    ``max_true`` = worst single true term);
  * for each curated SPURIOUS confuser, fit ``[truth | one spurious]`` (mirrors
    RPS adding one term at a time) and read the spurious column's statistics
    (``min_spur`` = best-separated confuser).

Candidate statistics (all from the SAME gammas, directly comparable; this script
builds ``VaryingCoefSetup`` DIRECTLY so ``_Bvals`` is present and the robust MAD
forms -- which the live ``from_full`` path cannot currently compute -- ARE
available here):

  var0_C     = Var(g0)/g0^2                      significance (current 'fit')
  NCdeb_C    = sum(max(g_k^2 - var_k, 0))/g0^2   noise-debiased region-variation
  NCraw_C    = sum(g_k^2)/g0^2                    raw region-variation (==(std/mu)^2)
  mad_med    = mad_x(beta)/|median_x(beta)|       robust region-variation
  mad_med_sq = (mad/median)^2

A GOOD single ``S`` gives LOW values to true terms and HIGH to spurious, so
``separation = min_spur - max_true`` should be > 0. The discriminators are the
collinear systems (lorenz, ac) and the soliton system (kdv).
"""
from __future__ import annotations
import argparse
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
from epde.interface.equation_translator import translate_equation, parse_factor
from epde.structure.main_structures import Term
from epde.operators.common.stability import VaryingCoefSetup, resolve_vc_modes_from_input
from kdv_sindy_test import build_pool_only, _normalize_grid_labels
from thesis_runner import load_config, pipeline_settings, _set_seeds

_ALL = ['ode', 'lv', 'vdp', 'lorenz', 'kdv', 'kdv_cossin', 'wave',
        'burgers_viscous', 'burgers_inviscid', 'pde_divide', 'pde_compound',
        'ac', 'ks', 'ns']

_EPS = 1e-12

# Candidate statistics, in the order tabulated. ``std_mu`` is printed once as a
# redundancy check that it matches sqrt(NCraw_C) (Parseval).
STAT_KEYS = ['var0_C', 'NCdeb_C', 'NCraw_C', 'mad_med', 'mad_med_sq']

# Curated SPURIOUS confuser terms per system, keyed by the equation's variable.
# Uses the EXACT token syntax from configs/<system>.yaml (grid labels already
# collapsed x_N{ -> x{ by _normalize_grid_labels). Terms that fail to build,
# evaluate degenerately, or coincide with a truth term are skipped at runtime,
# so the lists are deliberately generous.
_SPURIOUS = {
    'ode': {'u': [
        'u{power: 2.0}',
        'du/dx0{power: 1.0}',
        'x{power: 2.0, dim: 0.0}',
        'u{power: 1.0} * du/dx0{power: 1.0}',
        'sin{power: 1.0, freq: 2.0, dim: 0.0}',
    ]},
    'lv': {
        'u': ['v{power: 1.0}', 'u{power: 2.0}', 'v{power: 2.0}'],
        'v': ['u{power: 1.0}', 'u{power: 2.0}', 'v{power: 2.0}'],
    },
    'vdp': {'u': [
        'u{power: 2.0}', 'u{power: 3.0}', 'du/dx0{power: 2.0}',
        'u{power: 1.0} * du/dx0{power: 1.0}',
    ]},
    'lorenz': {
        'u': ['w{power: 1.0}', 'u{power: 1.0} * v{power: 1.0}',
              'u{power: 1.0} * w{power: 1.0}'],
        'v': ['w{power: 1.0}', 'v{power: 1.0} * w{power: 1.0}',
              'u{power: 1.0} * v{power: 1.0}'],
        'w': ['u{power: 1.0}', 'v{power: 1.0}',
              'u{power: 1.0} * w{power: 1.0}'],
    },
    'kdv': {'u': [
        'u{power: 1.0}', 'du/dx1{power: 1.0}', 'd^2u/dx1^2{power: 1.0}',
        'u{power: 2.0}', 'du/dx1{power: 2.0}',
    ]},
    'kdv_cossin': {'u': [
        'u{power: 1.0}', 'du/dx1{power: 1.0}', 'd^2u/dx1^2{power: 1.0}',
        'u{power: 2.0}',
    ]},
    'wave': {'u': [
        'u{power: 1.0}', 'du/dx1{power: 1.0}', 'du/dx0{power: 1.0}',
    ]},
    'burgers_viscous': {'u': [
        'u{power: 1.0}', 'du/dx1{power: 1.0}', 'du/dx1{power: 2.0}',
        'u{power: 2.0}', 'd^3u/dx1^3{power: 1.0}',
    ]},
    'burgers_inviscid': {'u': [
        'u{power: 1.0}', 'du/dx1{power: 1.0}', 'd^2u/dx1^2{power: 1.0}',
        'u{power: 2.0}',
    ]},
    'pde_divide': {'u': [
        'u{power: 1.0}', 'd^2u/dx1^2{power: 1.0}',
        'du/dx1{power: 1.0} * x{power: 1.0, dim: 1.0}',
        'u{power: 1.0} * x{power: 1.0, dim: 1.0}',
    ]},
    'pde_compound': {'u': [
        'u{power: 1.0}', 'du/dx1{power: 1.0}', 'd^2u/dx1^2{power: 1.0}',
        'u{power: 2.0}',
    ]},
    'ac': {'u': [
        'du/dx1{power: 1.0}', 'u{power: 2.0}', 'du/dx1{power: 2.0}',
        'd^3u/dx1^3{power: 1.0}',
    ]},
    'ks': {'u': [
        'u{power: 1.0}', 'du/dx1{power: 1.0}', 'd^3u/dx1^3{power: 1.0}',
        'u{power: 2.0}',
    ]},
    'ns': {
        'u': ['u{power: 1.0}', 'du/dx1{power: 1.0}', 'du/dx2{power: 1.0}',
              'v{power: 1.0} * du/dx2{power: 1.0}'],
        'v': ['v{power: 1.0}', 'dv/dx1{power: 1.0}', 'dv/dx2{power: 1.0}',
              'u{power: 1.0} * dv/dx1{power: 1.0}'],
        'p': ['du/dx1{power: 1.0}', 'dv/dx2{power: 1.0}', 'u{power: 1.0}',
              'v{power: 1.0}'],
    },
}

# Coordinate-MODULATED confusers -- the terms that actually survive in the
# 30x14 benchmark's coordinate-degeneracy failures. Kept SEPARATE from
# ``_SPURIOUS`` (plain confusers, caught by the significance A-term) because
# these need a non-constant ``beta(x)`` to be flagged at all: they are the
# B-term / mode-resolution question the K-sweep exists to answer. Axis
# convention: x0=t=dim0, x1=x=dim1. Focus systems only.
_COORD_SPURIOUS = {
    'ode': {'u': [
        'd^2u/dx0^2{power: 1.0} * sin{power: 1.0, freq: 2.0, dim: 0.0}',
    ]},
    'ac': {'u': [
        'd^2u/dx1^2{power: 1.0} * sin{power: 1.0, freq: 2.0, dim: 1.0}',
        'd^4u/dx1^4{power: 1.0} * cos{power: 1.0, freq: 2.0, dim: 1.0}',
        'd^2u/dx1^2{power: 1.0} * x{power: 1.0, dim: 1.0}',
    ]},
    'wave': {'u': [
        'd^2u/dx1^2{power: 1.0} * sin{power: 1.0, freq: 2.0, dim: 1.0}',
    ]},
    'burgers_inviscid': {'u': [
        'du/dx1{power: 1.0} * x{power: 1.0, dim: 1.0}',
        'du/dx1{power: 1.0} * sin{power: 1.0, freq: 2.0, dim: 1.0}',
    ]},
}

# REPLACEMENT specs: (distinctive label of the truth term to REMOVE, the
# coordinate-modulated term that replaces it). This is the LOAD-BEARING
# coordinate-degeneracy test -- the actual benchmark failure mode. Unlike the
# _COORD_SPURIOUS 'add' arm (confuser appended to COMPLETE truth, where it is
# redundant so the significance A-term flags it trivially), here the true term
# is GONE and its modulated twin must carry the signal (beta ~ 1/modulation).
# Whether that modulated term reads as unstable IS the K-sensitive question.
# The label is matched (space-insensitive substring, must be unique) against
# the truth feature-term names.
_REPLACE = {
    'ac': {'u': [
        ('d^2u/dx1^2', 'd^2u/dx1^2{power: 1.0} * sin{power: 1.0, freq: 2.0, dim: 1.0}'),
        ('d^2u/dx1^2', 'd^2u/dx1^2{power: 1.0} * cos{power: 1.0, freq: 2.0, dim: 1.0}'),
        ('d^2u/dx1^2', 'd^2u/dx1^2{power: 1.0} * x{power: 1.0, dim: 1.0}'),
    ]},
    'wave': {'u': [
        ('d^2u/dx1^2', 'd^2u/dx1^2{power: 1.0} * sin{power: 1.0, freq: 2.0, dim: 1.0}'),
    ]},
    'burgers_inviscid': {'u': [
        ('du/dx1', 'du/dx1{power: 1.0} * x{power: 1.0, dim: 1.0}'),
        ('du/dx1', 'du/dx1{power: 1.0} * sin{power: 1.0, freq: 2.0, dim: 1.0}'),
    ]},
}

# K (modes-per-axis incl. constant slot; K=2 -> 1 cosine, K=16 -> 15 cosines)
# swept by ``--k-sweep``. Spans the microscale's [2,6] band and well beyond.
_KSWEEP_DEFAULT = [2, 3, 4, 6, 10, 16]


def candidate_stats(setup: VaryingCoefSetup) -> dict:
    """All candidate statistics for every active feature of ``setup``, from a
    SINGLE gamma solve. Returns dict {stat_key: array(nf)} where nf includes the
    internal intercept column (last)."""
    sol = setup._solve_gammas(None)
    if sol is None:
        return None
    gamma, var = sol['gamma'], sol['var']
    nf, B, mk = sol['nf'], sol['B'], sol['mk']
    Bvals = getattr(setup, '_Bvals', None)
    is_const = mk == 0
    nonconst = ~is_const
    out = {k: np.full(nf, np.nan) for k in
           ['var0_C', 'NCdeb_C', 'NCraw_C', 'mad_med', 'mad_med_sq', 'std_mu']}
    for i in range(nf):
        sl = slice(i * B, (i + 1) * B)
        g = gamma[sl]
        v = var[sl]
        g0 = float(g[is_const][0]) if np.any(is_const) else 0.0
        var0 = float(v[is_const][0]) if np.any(is_const) else 0.0
        C = g0 * g0
        nc_raw = float(np.sum(g[nonconst] ** 2))
        nc_deb = float(np.sum(np.maximum(g[nonconst] ** 2 - v[nonconst], 0.0)))
        out['var0_C'][i] = var0 / (C + 1e-30)
        out['NCdeb_C'][i] = nc_deb / (C + 1e-30)
        out['NCraw_C'][i] = nc_raw / (C + 1e-30)
        if Bvals is not None:
            beta = Bvals @ g
            mu = float(np.mean(beta))
            sd = float(np.std(beta))
            med = float(np.median(beta))
            mad = float(np.median(np.abs(beta - med)))
            out['std_mu'][i] = sd / (abs(mu) + _EPS)
            out['mad_med'][i] = mad / (abs(med) + _EPS)
            out['mad_med_sq'][i] = (mad / (abs(med) + _EPS)) ** 2
    return {k: np.nan_to_num(v, nan=0.0, posinf=1e30) for k, v in out.items()}


def build_term_values(term_str, pool, all_vars):
    """Build a single Term from a symbolic string and evaluate it on the grid.
    Returns (term, flat_values) or raises."""
    factors = [parse_factor(f.strip(), pool, all_vars)
               for f in term_str.split(' * ')]
    term = Term(pool, passed_term=factors, collapse_powers=False)
    vals = np.asarray(term.evaluate(False), dtype=float).reshape(-1)
    return term, vals


def analyse_system(system):
    """Return {var: {'truth': stats_dict, 'spur': [stats_dict, ...],
    'n_truth': int}} for one system, or raise on a load failure."""
    cfg = load_config(system)
    _set_seeds(0)
    search = build_pool_only(cfg, pipeline_settings('new'))
    coords, data, variable_names, dim = cfg.load_data()
    truth = yaml.safe_load(open(os.path.join(_THIS, 'configs', f'{system}.yaml')))
    truth_eqs = truth.get('truth_equations') or []
    all_vars = list(variable_names)
    seeded = (truth_eqs[0] if len(all_vars) == 1
              else {v: truth_eqs[i] for i, v in enumerate(all_vars)})
    seeded = _normalize_grid_labels(seeded)
    soeq = translate_equation(seeded, search.pool, all_vars=all_vars)

    sw = gv.grid_cache.g_func[gv.grid_cache.g_func_mask]
    gshape = gv.grid_cache.inner_shape
    spur_cfg = _SPURIOUS.get(system, {})

    result = {}
    for v in all_vars:
        eq = soeq.vals[v]
        eq.main_var_to_explain = v
        eq.weights_internal = np.ones(len(eq.structure) - 1)
        eq.weights_internal_evald = True
        eq.weights_final_evald = True
        _, target, features = eq.evaluate(normalize=True, return_val=False)
        if features is None or np.asarray(features).ndim != 2:
            continue
        features = np.asarray(features, dtype=float)
        target = np.asarray(target, dtype=float).reshape(-1)
        n_truth = features.shape[1]
        truth_names = {t.name for i, t in enumerate(eq.structure)
                       if i != eq.target_idx}

        # truth-only fit -> per-true-term statistics (cols [0:n_truth]; the
        # last column is the internal intercept and is ignored).
        base = candidate_stats(
            VaryingCoefSetup(features, target, sw, gshape, main_var=v))
        if base is None:
            continue
        truth_stats = {k: base[k][:n_truth] for k in base}

        # each curated spurious confuser -> [truth | spurious] fit; read the
        # spurious column (index n_truth).
        spur_stats = []
        for s in spur_cfg.get(v, []):
            try:
                term, vals = build_term_values(s, search.pool, all_vars)
            except Exception:
                continue
            if term.name in truth_names:
                continue
            if vals.shape[0] != target.shape[0] or not np.any(np.abs(vals) > 0):
                continue
            aug = np.hstack([features, vals[:, None]])
            st = candidate_stats(
                VaryingCoefSetup(aug, target, sw, gshape, main_var=v))
            if st is None:
                continue
            spur_stats.append({'name': term.name,
                               **{k: float(st[k][n_truth]) for k in base}})
        result[v] = {'truth': truth_stats, 'spur': spur_stats,
                     'n_truth': n_truth}
    return result


def _agg_system(res):
    """Aggregate per-equation results into per-stat (sum_true, max_true,
    min_spur, sep) for one system. sep is the WORST (min) per-equation
    separation."""
    agg = {}
    for k in STAT_KEYS:
        sum_true_eq, max_true_eq, sep_eq, min_spur_all = [], [], [], []
        for v, d in res.items():
            tv = d['truth'][k]
            if tv.size == 0:
                continue
            sum_true_eq.append(float(np.sum(tv)))
            mt = float(np.max(tv))
            max_true_eq.append(mt)
            spur_vals = [s[k] for s in d['spur']]
            if spur_vals:
                ms = float(np.min(spur_vals))
                min_spur_all.append(ms)
                sep_eq.append(ms - mt)
        agg[k] = {
            'sum_true': float(np.sum(sum_true_eq)) if sum_true_eq else np.nan,
            'max_true': float(np.max(max_true_eq)) if max_true_eq else np.nan,
            'min_spur': float(np.min(min_spur_all)) if min_spur_all else np.nan,
            'sep': float(np.min(sep_eq)) if sep_eq else np.nan,
        }
    return agg


def dump_system(system):
    """Raw per-true-term beta(x) diagnostics: does gamma_0 recover the true
    constant coefficient, and how wildly does beta(x) swing? Verifies the beta
    reconstruction (mad/median) on a clean (ode) vs collinear (lorenz) system."""
    cfg = load_config(system)
    _set_seeds(0)
    search = build_pool_only(cfg, pipeline_settings('new'))
    coords, data, variable_names, dim = cfg.load_data()
    truth = yaml.safe_load(open(os.path.join(_THIS, 'configs', f'{system}.yaml')))
    truth_eqs = truth.get('truth_equations') or []
    all_vars = list(variable_names)
    seeded = (truth_eqs[0] if len(all_vars) == 1
              else {v: truth_eqs[i] for i, v in enumerate(all_vars)})
    seeded = _normalize_grid_labels(seeded)
    soeq = translate_equation(seeded, search.pool, all_vars=all_vars)
    sw = gv.grid_cache.g_func[gv.grid_cache.g_func_mask]
    gshape = gv.grid_cache.inner_shape
    print(f'\n##### {system}  grid_shape={gshape}  truth: {seeded}')
    for v in all_vars:
        eq = soeq.vals[v]
        eq.main_var_to_explain = v
        eq.weights_internal = np.ones(len(eq.structure) - 1)
        eq.weights_internal_evald = True
        eq.weights_final_evald = True
        _, target, features = eq.evaluate(normalize=True, return_val=False)
        if features is None or np.asarray(features).ndim != 2:
            continue
        features = np.asarray(features, dtype=float)
        target = np.asarray(target, dtype=float).reshape(-1)
        n_truth = features.shape[1]
        names = [t.name for i, t in enumerate(eq.structure)
                 if i != eq.target_idx]
        setup = VaryingCoefSetup(features, target, sw, gshape, main_var=v)
        sol = setup._solve_gammas(None)
        gamma, varr, B, mk = sol['gamma'], sol['var'], sol['B'], sol['mk']
        Bvals = setup._Bvals
        is_const = mk == 0
        nonconst = ~is_const
        # Constant-only weighted OLS (the standard SINDy coefficient): the
        # const columns of the super-Gram are at i*B for each feature i.
        const_global = np.arange(setup.n_features) * B
        Ac = setup.G[np.ix_(const_global, const_global)]
        bc = setup.Phiy[const_global]
        try:
            coef_c = np.linalg.solve(Ac + 1e-12 * np.eye(len(const_global)), bc)
        except np.linalg.LinAlgError:
            coef_c = np.full(len(const_global), np.nan)
        # conditioning of the constant block (features-only OLS):
        try:
            cond_c = float(np.linalg.cond(Ac))
        except np.linalg.LinAlgError:
            cond_c = float('inf')
        print(f'  [{v}] B={B} modes(k present)={sorted(set(mk.tolist()))} '
              f'N={Bvals.shape[0]} target|mean|={np.mean(np.abs(target)):.3g} '
              f'cond(const-block)={cond_c:.3g}')
        for i in range(n_truth):
            sl = slice(i * B, (i + 1) * B)
            g = gamma[sl]
            g0 = float(g[is_const][0])
            var0 = float(varr[sl][is_const][0])
            beta = Bvals @ g
            med = float(np.median(beta))
            mad = float(np.median(np.abs(beta - med)))
            vc = var0 / (g0 ** 2 + 1e-30)
            print(f'    {names[i][:30]:30s} g0={g0:>9.4g} var0={var0:>9.3g} '
                  f'vc=var0/g0^2={vc:>9.4g}  '
                  f'std={np.std(beta):>8.3g} NCraw={float(np.sum(g[nonconst]**2)):.3g}')
            print(f'        gamma_k(nonconst)={np.round(g[nonconst], 3).tolist()}')


def k_sweep_system(system, K_list):
    """Sweep the per-axis basis resolution ``K`` (overriding the Taylor
    microscale via ``modes=(K,)*D``) and measure, per K, whether the
    coordinate-modulated confusers become separable from the true terms.

    Returns ``{var: {'rows': [...], 'kstar': tuple|None, 'n_coord': int,
    'n_plain': int, 'grid': tuple}}``. Each row is one K with the B-term
    (region-variation ``NCdeb_C``) and full-score (``sum_asym`` = A+B)
    separations, plus the worst (min-B) coordinate confuser's beta-field
    spread ``std/|mu|`` -- the visual of whether its coefficient field starts
    to vary as K grows.
    """
    cfg = load_config(system)
    _set_seeds(0)
    search = build_pool_only(cfg, pipeline_settings('new'))
    coords, data, variable_names, dim = cfg.load_data()
    truth = yaml.safe_load(open(os.path.join(_THIS, 'configs', f'{system}.yaml')))
    truth_eqs = truth.get('truth_equations') or []
    all_vars = list(variable_names)
    seeded = (truth_eqs[0] if len(all_vars) == 1
              else {v: truth_eqs[i] for i, v in enumerate(all_vars)})
    seeded = _normalize_grid_labels(seeded)
    soeq = translate_equation(seeded, search.pool, all_vars=all_vars)

    sw = gv.grid_cache.g_func[gv.grid_cache.g_func_mask]
    gshape = gv.grid_cache.inner_shape
    D = len(gshape)
    plain_cfg = _SPURIOUS.get(system, {})
    coord_cfg = _COORD_SPURIOUS.get(system, {})

    result = {}
    for v in all_vars:
        eq = soeq.vals[v]
        eq.main_var_to_explain = v
        eq.weights_internal = np.ones(len(eq.structure) - 1)
        eq.weights_internal_evald = True
        eq.weights_final_evald = True
        _, target, features = eq.evaluate(normalize=True, return_val=False)
        if features is None or np.asarray(features).ndim != 2:
            continue
        features = np.asarray(features, dtype=float)
        target = np.asarray(target, dtype=float).reshape(-1)
        n_truth = features.shape[1]
        truth_names = {t.name for i, t in enumerate(eq.structure)
                       if i != eq.target_idx}
        names = [t.name for i, t in enumerate(eq.structure)
                 if i != eq.target_idx]  # ordered, aligned to features columns

        # K* the microscale would resolve (production reference line).
        try:
            kstar = resolve_vc_modes_from_input(gshape, main_var=v, k_max=6)
        except Exception:
            kstar = None

        # Build confuser value columns ONCE (independent of K); skip the ones
        # that fail to build / are degenerate / collide with a truth term.
        def build_cols(strs):
            cols = []
            for s in strs:
                try:
                    term, vals = build_term_values(s, search.pool, all_vars)
                except Exception:
                    continue
                if term.name in truth_names:
                    continue
                if vals.shape[0] != target.shape[0] or not np.any(np.abs(vals) > 0):
                    continue
                cols.append((term.name, vals))
            return cols
        coord_cols = build_cols(coord_cfg.get(v, []))
        plain_cols = build_cols(plain_cfg.get(v, []))

        def spur_rows(cols, modes):
            res = []
            for name, vals in cols:
                aug = np.hstack([features, vals[:, None]])
                st = candidate_stats(VaryingCoefSetup(
                    aug, target, sw, gshape, main_var=v, modes=modes))
                if st is None:
                    continue
                res.append({
                    'name': name,
                    'B': float(st['NCdeb_C'][n_truth]),
                    'S': float(st['var0_C'][n_truth] + st['NCdeb_C'][n_truth]),
                    'stdmu': float(st['std_mu'][n_truth]),
                })
            return res

        rows = []
        for K in K_list:
            modes = (int(K),) * D
            base = candidate_stats(VaryingCoefSetup(
                features, target, sw, gshape, main_var=v, modes=modes))
            if base is None:
                continue
            true_B = base['NCdeb_C'][:n_truth]
            true_S = base['var0_C'][:n_truth] + base['NCdeb_C'][:n_truth]
            max_true_B = float(np.max(true_B)) if true_B.size else float('nan')
            max_true_S = float(np.max(true_S)) if true_S.size else float('nan')

            cr = spur_rows(coord_cols, modes)
            pr = spur_rows(plain_cols, modes)
            min_coord = min(cr, key=lambda r: r['B']) if cr else None
            all_S = [r['S'] for r in cr + pr]

            rows.append({
                'K': int(K),
                'max_true_B': max_true_B,
                'min_spur_B': (min_coord['B'] if min_coord else float('nan')),
                'sep_B': ((min_coord['B'] - max_true_B) if min_coord
                          else float('nan')),
                'max_true_S': max_true_S,
                'min_spur_S': (min(all_S) if all_S else float('nan')),
                'sep_S': ((min(all_S) - max_true_S) if all_S
                          else float('nan')),
                'worst_coord': (min_coord['name'] if min_coord else '-'),
                'worst_stdmu': (min_coord['stdmu'] if min_coord
                                else float('nan')),
            })
        # --- REPLACEMENT arm: drop a true term, let its modulated twin carry
        # the load, and score the twin vs the remaining true terms per K.
        def _norm(s):
            return ''.join(s.split())

        repl_out = []
        for label, confuser_str in _REPLACE.get(system, {}).get(v, []):
            nl = _norm(label)
            hits = [j for j, nm in enumerate(names) if nl in _norm(nm)]
            if len(hits) != 1:
                continue  # ambiguous / absent -> skip
            j = hits[0]
            try:
                cterm, cvals = build_term_values(confuser_str, search.pool, all_vars)
            except Exception:
                continue
            if cvals.shape[0] != target.shape[0] or not np.any(np.abs(cvals) > 0):
                continue
            feats_rm = np.delete(features, j, axis=1)
            nrem = feats_rm.shape[1]
            rrows = []
            for K in K_list:
                modes = (int(K),) * D
                aug = np.hstack([feats_rm, cvals[:, None]])
                st = candidate_stats(VaryingCoefSetup(
                    aug, target, sw, gshape, main_var=v, modes=modes))
                if st is None:
                    continue
                true_S = st['var0_C'][:nrem] + st['NCdeb_C'][:nrem]
                true_B = st['NCdeb_C'][:nrem]
                spur_S = float(st['var0_C'][nrem] + st['NCdeb_C'][nrem])
                spur_B = float(st['NCdeb_C'][nrem])
                mts = float(np.max(true_S)) if true_S.size else float('nan')
                mtb = float(np.max(true_B)) if true_B.size else float('nan')
                rrows.append({
                    'K': int(K), 'true_S': mts, 'spur_S': spur_S,
                    'sep_S': spur_S - mts, 'true_B': mtb, 'spur_B': spur_B,
                    'sep_B': spur_B - mtb, 'stdmu': float(st['std_mu'][nrem]),
                })
            repl_out.append({'removed': label, 'confuser': cterm.name,
                             'rows': rrows})

        result[v] = {'rows': rows, 'kstar': kstar, 'grid': tuple(gshape),
                     'n_coord': len(coord_cols), 'n_plain': len(plain_cols),
                     'repl': repl_out}
    return result


def _print_k_sweep(system, res):
    """Per-system K-vs-separation table. ``sep_B`` is the coordinate-confuser
    region-variation gap (>0 => flaggable); ``sep_S`` the full sum_asym gap."""
    for v, d in res.items():
        ks = d['kstar']
        print(f"\n##### {system} [{v}]  grid={d['grid']}  K*(microscale)={ks}"
              f"  n_coord={d['n_coord']} n_plain={d['n_plain']}")
        if not d['rows']:
            print('   (no evaluable rows)')
            continue
        if d['n_coord'] == 0:
            print('   (no coordinate confusers for this system -- sep_B is NA;'
                  ' read sep_S only)')
        hdr = (f"   {'K':>3} {'max_true_B':>11} {'min_spur_B':>11} {'sep_B':>10}"
               f" | {'max_true_S':>11} {'min_spur_S':>11} {'sep_S':>10}"
               f" | {'beta_spur std/mu':>16}  worst_coord")
        print(hdr)
        print('   ' + '-' * (len(hdr) - 3))
        for r in d['rows']:
            fB = ' <0' if (r['sep_B'] == r['sep_B'] and r['sep_B'] < 0) else ''
            print(f"   {r['K']:>3} {r['max_true_B']:>11.3g} "
                  f"{r['min_spur_B']:>11.3g} {r['sep_B']:>10.3g}{fB:<3}"
                  f" | {r['max_true_S']:>11.3g} {r['min_spur_S']:>11.3g} "
                  f"{r['sep_S']:>10.3g} | {r['worst_stdmu']:>16.3g}  "
                  f"{r['worst_coord'][:34]}")

        # Replacement arm: the load-bearing coordinate-degeneracy test.
        for rp in d.get('repl', []):
            print(f"   -- REPLACE drop [{rp['removed']}] -> load-bearing "
                  f"{rp['confuser'][:48]}  (sep>0 => flagged)")
            rh = (f"   {'K':>3} {'true_S':>11} {'spur_S':>11} {'sep_S':>10}"
                  f" | {'true_B':>11} {'spur_B':>11} {'sep_B':>10}"
                  f" | {'beta std/mu':>12}")
            print(rh)
            for r in rp['rows']:
                fS = ' <0' if (r['sep_S'] == r['sep_S'] and r['sep_S'] < 0) else ''
                print(f"   {r['K']:>3} {r['true_S']:>11.3g} {r['spur_S']:>11.3g} "
                      f"{r['sep_S']:>10.3g}{fS:<3} | {r['true_B']:>11.3g} "
                      f"{r['spur_B']:>11.3g} {r['sep_B']:>10.3g} | "
                      f"{r['stdmu']:>12.3g}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--systems', default=','.join(_ALL))
    p.add_argument('--dump', action='store_true',
                   help='Raw per-true-term beta(x) diagnostics instead of the '
                        'comparison tables.')
    p.add_argument('--k-sweep', action='store_true',
                   help='Sweep basis modes K per axis (overriding the Taylor '
                        'microscale via modes=(K,)*D) and tabulate '
                        'true-vs-coordinate-confuser separation per K.')
    p.add_argument('--k-list', default=','.join(str(k) for k in _KSWEEP_DEFAULT),
                   help='Comma-separated K values for --k-sweep.')
    args = p.parse_args(argv)
    systems = [s.strip() for s in args.systems.split(',') if s.strip()]

    if args.k_sweep:
        K_list = [int(x) for x in args.k_list.split(',') if x.strip()]
        print(f'K-sweep (modes per axis overriding microscale): {K_list}')
        for sysn in systems:
            try:
                res = k_sweep_system(sysn, K_list)
            except Exception as e:
                print(f'{sysn:16s} ERROR {type(e).__name__}: {str(e)[:80]}')
                continue
            if not res:
                print(f'{sysn:16s} (no evaluable equations)')
                continue
            _print_k_sweep(sysn, res)
        return 0

    if args.dump:
        for sysn in systems:
            try:
                dump_system(sysn)
            except Exception as e:
                print(f'{sysn:16s} ERROR {type(e).__name__}: {str(e)[:80]}')
        return 0

    per_system = {}
    redundancy = []  # (system, var, max |std_mu - sqrt(NCraw_C)| over true)
    for sysn in systems:
        try:
            res = analyse_system(sysn)
        except Exception as e:
            print(f'{sysn:16s} ERROR {type(e).__name__}: {str(e)[:60]}')
            continue
        if not res:
            print(f'{sysn:16s} (no evaluable equations)')
            continue
        per_system[sysn] = _agg_system(res)
        for v, d in res.items():
            sm = d['truth']['std_mu']
            nc = np.sqrt(np.maximum(d['truth']['NCraw_C'], 0.0))
            if sm.size:
                redundancy.append((sysn, v, float(np.max(np.abs(sm - nc)))))

    # ---- per-statistic tables over systems --------------------------------
    for k in STAT_KEYS:
        print(f'\n=== {k} ===  (sum_true,max_true LOW good; min_spur HIGH good; '
              f'sep=min_spur-max_true >0 good)')
        hdr = (f"{'system':16s} {'sum_true':>11} {'max_true':>11} "
               f"{'min_spur':>11} {'sep':>11}")
        print(hdr)
        print('-' * len(hdr))
        for sysn in systems:
            if sysn not in per_system:
                continue
            a = per_system[sysn][k]
            flag = '' if not np.isfinite(a['sep']) else (' <-- NEG' if a['sep'] < 0 else '')
            print(f"{sysn:16s} {a['sum_true']:>11.3g} {a['max_true']:>11.3g} "
                  f"{a['min_spur']:>11.3g} {a['sep']:>11.3g}{flag}")

    # ---- decision summary: rank statistics --------------------------------
    print(f'\n{"=" * 72}\nDECISION SUMMARY (across {len(per_system)} systems)\n{"=" * 72}')
    hdr = (f"{'statistic':12s} {'#sep>0':>7} {'min_sep':>11} {'worst_sum_true':>15} "
           f"{'worst_max_true':>15}")
    print(hdr)
    print('-' * len(hdr))
    for k in STAT_KEYS:
        seps = [per_system[s][k]['sep'] for s in per_system
                if np.isfinite(per_system[s][k]['sep'])]
        sums = [per_system[s][k]['sum_true'] for s in per_system
                if np.isfinite(per_system[s][k]['sum_true'])]
        maxs = [per_system[s][k]['max_true'] for s in per_system
                if np.isfinite(per_system[s][k]['max_true'])]
        n_pos = sum(1 for x in seps if x > 0)
        print(f"{k:12s} {n_pos:>3}/{len(seps):<3} "
              f"{(min(seps) if seps else float('nan')):>11.3g} "
              f"{(max(sums) if sums else float('nan')):>15.3g} "
              f"{(max(maxs) if maxs else float('nan')):>15.3g}")

    if redundancy:
        worst = max(redundancy, key=lambda r: r[2])
        print(f'\nParseval check  max|std_mu - sqrt(NCraw_C)| over true terms = '
              f'{worst[2]:.2e}  ({worst[0]}/{worst[1]})  [should be ~0]')
    return 0


if __name__ == '__main__':
    sys.exit(main())
