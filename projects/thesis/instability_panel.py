"""Offline comparison of the instability-metric alternatives ("is vcoef best?").

For each benchmark system, assemble the candidate set (reusing
``hadamard/objective_experiment.py``): the seeded TRUE equation, valid
analytical identities (``truth_alternatives``), constructed wrong forms
(term dropped / spurious term added), and structurally-wrong equations the
pipeline actually discovered (``results/<sys>/*rep*.json``). Score every
candidate on CLEAN data (noisy scenarios are out of scope for the
solver-free search) under each ``instability_metric``:

    instab_vcoef     varying-coefficient NC/gamma_0^2 (the production default)
    instab_cv        axis-aligned sliding-window CV (var/mu^2)
    instab_survival  block-resampled coefficient survival (statistical)
    instab_tile      per-tile between-tile dispersion (basis-free spatial)

A metric PASSES a system iff the TRUE equation is non-dominated on
(disc, metric) by any truly-wrong form -- the standing principle: the true
equation must not be dominated by a NON-analytical form; identities are
credited valid and never counted against a metric. Strict separation
(TRUE strictly below ALL wrong on the metric axis alone) is reported as a
secondary, harder lens. vcoef's prior result (unique 14/14) is the bar.

    python projects/thesis/instability_panel.py                 # all systems
    python projects/thesis/instability_panel.py --systems kdv lv ac
    python projects/thesis/instability_panel.py --summarize     # from saved JSON
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_THIS = os.path.dirname(os.path.abspath(__file__))
_HADAMARD = os.path.join(_THIS, 'hadamard')
_ROOT = os.path.abspath(os.path.join(_THIS, '..', '..'))
for _p in (_ROOT, _THIS, _HADAMARD):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
import yaml  # noqa: E402
import epde.globals as gv  # noqa: E402
from epde.interface.equation_translator import translate_equation  # noqa: E402
from epde.operators.common.stability import (  # noqa: E402
    VaryingCoefSetup, calculate_weights,
)
from epde.operators.common.survival import survival_scores, tile_scores  # noqa: E402
from kdv_sindy_test import build_pool_only, _normalize_grid_labels  # noqa: E402
from thesis_runner import load_config, pipeline_settings, _set_seeds  # noqa: E402
from objective_experiment import (  # noqa: E402
    ALL_SYSTEMS, _as_eqdict, _constructed_wrong, _discovered_wrong,
    _features, _disc, _verdict, _verdict_pareto,
)

METRICS = ('vcoef', 'cv', 'survival', 'tile')
OUT = os.path.join(_THIS, 'instability_panel.json')
MD_OUT = os.path.join(_THIS, 'instability_panel.md')


def _metric_value(metric, f, t, w, gshape, var):
    """One instability metric on one equation's (features, target)."""
    if metric == 'vcoef':
        setup = VaryingCoefSetup(f, t, w, gshape, main_var=var,
                                 fit_intercept=False)
        return float(np.sum(np.asarray(setup.score(None), dtype=float)))
    if metric == 'cv':
        sw = calculate_weights(f, t, w, gshape, False,
                               gram_cls=None, gram_kwargs=None)
        sw_arr = np.array(sw)
        mu = sw_arr.mean(axis=0)
        std = sw_arr.std(axis=0, ddof=1)
        with np.errstate(divide='ignore', invalid='ignore'):
            cv = (std ** 2) / (mu ** 2)
            cv[mu == 0] = 0.0
        return float(np.sum(np.nan_to_num(cv)) / len(gshape))
    if metric == 'survival':
        return float(np.sum(survival_scores(f, t, w, gshape,
                                            fit_intercept=False)))
    if metric == 'tile':
        return float(np.sum(tile_scores(f, t, w, gshape,
                                        fit_intercept=False)))
    raise ValueError(f'unknown metric {metric!r}')


def score_candidate(eqdict, variable_names, pool, w, gshape):
    """disc + every instability metric, aggregated over the candidate's
    equations (mean for disc, sum for instability -- the
    objective_experiment convention)."""
    soeq = translate_equation(_normalize_grid_labels(eqdict), pool,
                              all_vars=list(variable_names))
    discs = []
    instabs = {m: [] for m in METRICS}
    for v in variable_names:
        f, t = _features(soeq, v)
        discs.append(_disc(f, t, w))
        for m in METRICS:
            try:
                instabs[m].append(_metric_value(m, f, t, w, gshape, v))
            except Exception:
                instabs[m].append(np.nan)
    out = {'disc': float(np.mean(discs))}
    for m in METRICS:
        out[f'instab_{m}'] = float(np.nansum(instabs[m]))
    return out


def run_system(system):
    cfg = load_config(system)
    _, _, variable_names, _ = cfg.load_data()
    variable_names = list(variable_names)

    y = yaml.safe_load(open(os.path.join(_THIS, 'configs', f'{system}.yaml'),
                            encoding='utf-8'))
    teqs = y.get('truth_equations') or []
    if not teqs:
        return {'system': system, 'error': 'no truth_equations (unknown-truth '
                                           'config); panel needs a seeded truth'}
    truth = _as_eqdict(teqs, variable_names)

    candidates = [('true', 'true', truth)]
    for alt in (y.get('truth_alternatives') or []):
        if alt:
            candidates.append((f'identity[{len(candidates)}]', 'identity',
                               _as_eqdict(alt, variable_names)))
    for label, eqd in _constructed_wrong(truth, variable_names).items():
        candidates.append((label, 'wrong', eqd))
    for fname, eqd in _discovered_wrong(system, variable_names):
        candidates.append((f'wrong:disc[{fname}]', 'wrong', eqd))

    cfg = load_config(system)
    _set_seeds(0)
    gv.set_gram_config('vcoef')
    search = build_pool_only(cfg, pipeline_settings('new'))
    w = np.asarray(gv.grid_cache.g_func[gv.grid_cache.g_func_mask]).reshape(-1)
    gshape = tuple(int(n) for n in gv.grid_cache.inner_shape)

    rows = []
    for label, cat, eqd in candidates:
        try:
            sc = score_candidate(eqd, variable_names, search.pool, w, gshape)
            sc.update({'label': label, 'category': cat})
            rows.append(sc)
        except Exception as exc:
            rows.append({'label': label, 'category': cat,
                         'error': f'{type(exc).__name__}: {str(exc)[:80]}'})
    return {'system': system, 'variables': variable_names,
            'n_candidates': len(candidates), 'rows': rows}


def render_markdown(recs):
    """Per-system pass matrix + N/total footer, both lenses."""
    recs = [r for r in recs if 'rows' in r]
    lines = [
        '# Instability-metric panel (clean data)',
        '',
        'PASS lens (primary): TRUE non-dominated on (disc, metric) by any '
        'truly-wrong form -- ties credited (what MOEA/D selection returns). '
        'strict (secondary): TRUE strictly below ALL wrong on the metric '
        'axis alone. Identities are never counted against a metric.',
        '',
        '| system | ' + ' | '.join(METRICS) + ' |',
        '|---' * (len(METRICS) + 1) + '|',
    ]
    nd_counts = {m: 0 for m in METRICS}
    strict_counts = {m: 0 for m in METRICS}
    for r in recs:
        cells = []
        for m in METRICS:
            key = f'instab_{m}'
            vp = _verdict_pareto(r['rows'], key)
            vs = _verdict(r['rows'], key)
            if vp is None:
                cells.append('n/a')
                continue
            nd = vp['true_nondominated']
            strict = bool(vs and vs['true_below_all_wrong'])
            nd_counts[m] += nd
            strict_counts[m] += strict
            mark = 'PASS' if nd else 'FAIL'
            cells.append(f"{mark}{' (strict)' if strict else ''}")
        lines.append(f"| {r['system']} | " + ' | '.join(cells) + ' |')
    n = len(recs)
    lines.append('| **N pass (ND)** | ' + ' | '.join(
        f"**{nd_counts[m]}/{n}**" for m in METRICS) + ' |')
    lines.append('| N strict | ' + ' | '.join(
        f"{strict_counts[m]}/{n}" for m in METRICS) + ' |')
    lines.append('')
    lines.append('vcoef is the production default and stays so unless '
                 'strictly beaten (prior bar: unique 14/14).')
    return '\n'.join(lines) + '\n'


def main(argv=None):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
    p = argparse.ArgumentParser()
    p.add_argument('--systems', nargs='+', default=ALL_SYSTEMS)
    p.add_argument('--out', default=OUT)
    p.add_argument('--md-out', default=MD_OUT)
    p.add_argument('--summarize', action='store_true',
                   help='Render the markdown table from an existing --out '
                        'JSON and exit.')
    args = p.parse_args(argv)

    if args.summarize:
        recs = json.load(open(args.out, encoding='utf-8'))
        md = render_markdown(recs)
        print(md)
        with open(args.md_out, 'w', encoding='utf-8') as fh:
            fh.write(md)
        return 0

    results = []
    for system in args.systems:
        print(f'=== {system} ===', flush=True)
        try:
            rec = run_system(system)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            rec = {'system': system, 'error': repr(exc)}
        results.append(rec)
        with open(args.out, 'w', encoding='utf-8') as fh:
            json.dump(results, fh, indent=2,
                      default=lambda o: float(o) if isinstance(
                          o, (np.floating, np.integer)) else str(o))
    md = render_markdown(results)
    print(md)
    with open(args.md_out, 'w', encoding='utf-8') as fh:
        fh.write(md)
    print('DONE ->', args.out, '|', args.md_out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
