"""Sweep (patch_radius, cv_method) on the seeded-truth KdV diagnostic.

Bypasses MOEA/D entirely: for each (r, cv_method) combination, seeds the
truth KdV equation, runs ``VWSRSparsity.apply`` once, and reports:

* M -- number of patches built (depends on r, stride, grid_shape).
* n_survived / n_truth -- how many of the two truth terms passed sparsity.
* recovered coefficients of the two truth terms.
* live ``coefficients_stability`` value (the live CV-based metric).
* per-pass wall.

Output columns are aligned for easy A/B reading. Default placement is
``sobol``, the lowest-discrepancy variant.

Usage:
    python projects/thesis/kdv_sindy_sweep.py [--system kdv]
        [--radii 2,4,8,auto] [--cv-methods mad_median,mad_median_lin,std_mean,std_mean_lin,iqr_sat]
        [--placement off|jitter|sobol] [--seed N] [--patch-stride N]
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time

# Truth YAMLs use ``x_0{... dim:0}`` / ``x_1{... dim:1}`` per the legacy
# grid-token labels, but ``thesis_runner._build_token_pool`` now registers
# a single ``x`` family with ``dim`` as an inner parameter. translate_equation
# wants the label to match the pool exactly, so we collapse ``x_N{...}`` ->
# ``x{...}`` before seeding. The ``dim:N`` parameter inside the braces
# already disambiguates the axis.
_GRID_LABEL_RE = re.compile(r'\bx_\d+(?=\{)')


def _normalize_grid_labels(symbolic):
    """Rewrite ``x_0{...}`` -> ``x{...}`` on a string or dict-of-strings."""
    if isinstance(symbolic, dict):
        return {k: _GRID_LABEL_RE.sub('x', v) for k, v in symbolic.items()}
    return _GRID_LABEL_RE.sub('x', symbolic)

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import numpy as np
import yaml

from epde import globals as gv
from epde.interface.equation_translator import translate_equation
from kdv_sindy_test import (  # noqa: E402
    build_pool_only, make_fit_operator,
)
from thesis_runner import (  # noqa: E402
    _set_seeds, load_config, pipeline_settings,
)


_FRAC_AUTO_RE = re.compile(r'^([0-9]*\.?[0-9]+)auto(_pa)?$', re.IGNORECASE)


def _parse_radius(v: str):
    """Accept several forms:
    * ``int``                  -- explicit scalar radius
    * ``'auto'``               -- isotropic r_auto from cached u
    * ``'<frac>auto'``         -- fraction of isotropic r_auto
    * ``'auto_pa'``            -- per-axis r_auto tuple
    * ``'<frac>auto_pa'``      -- fraction of per-axis r_auto tuple
    """
    s = v.strip()
    if s.lower() == 'auto':
        return 'auto'
    if s.lower() == 'auto_pa':
        return 'auto_pa'
    m = _FRAC_AUTO_RE.match(s)
    if m is not None:
        per_axis = bool(m.group(2))
        return ('frac_auto', float(m.group(1)), per_axis)
    return int(s)


def _parse_stride(v: str):
    s = v.lower()
    if s in ('auto', '2auto'):
        return s
    return int(v)


def _seed_and_eval(symbolic, search, fit_op, all_vars):
    """Seed truth then run sparsity+fitness once. Returns a summary dict.

    Mirrors ``kdv_sindy_test.inspect_truth`` but without the verbose
    per-term diagnostic; we only care about the post-sparsity state.
    """
    metaparams = {
        ('sparsity', v): {'optimizable': False, 'value': 1e-6}
        for v in all_vars
    }
    soeq = translate_equation(symbolic, search.pool, all_vars=list(all_vars))
    summaries = {}
    for v in all_vars:
        eq = soeq.vals[v]
        eq.main_var_to_explain = v
        eq.metaparameters = metaparams
        eq.weights_internal = np.ones(len(eq.structure) - 1)
        eq.weights_internal_evald = True
        eq.weights_final_evald = True

        target_idx = eq.target_idx
        feat_terms = [t for i, t in enumerate(eq.structure) if i != target_idx]
        names = [t.name for t in feat_terms]

        t0 = time.time()
        fit_op.apply(eq, {}, force_out_of_place=True)
        # Run a clean fitness pass so coefficients_stability gets the
        # live CV value on the surviving structure.
        eq.fitness_calculated = False
        eq.stability_calculated = False
        fit_op.apply(eq, {}, force_out_of_place=False)
        wall = time.time() - t0

        w = list(eq.weights_internal)
        # Pair (name, coef); only non-intercept feature terms.
        surviving = [(names[j], float(w[j])) for j in range(len(names))
                      if abs(w[j]) > 0.0]
        summaries[v] = dict(
            names=names,
            weights=[float(x) for x in w[: len(names)]],
            surviving=surviving,
            n_survived=len(surviving),
            n_truth=len(names),  # truth_eqs YAML lists the right
                                   # non-target terms exactly.
            fitness=float(eq.fitness_value),
            coef_stability=float(eq.coefficients_stability)
                if eq.coefficients_stability is not None else float('nan'),
            wall=wall,
        )
    return summaries


def _format_row(r, cv, M, total_survived, total_truth, fitness, stab,
                wall, coefs_str):
    return (f"{str(r):>4} {cv:<15} {M:>5} "
            f"{total_survived:>3}/{total_truth:<3} "
            f"{fitness:>10.4g} {stab:>10.4g} {wall:>7.2f}s  {coefs_str}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--system', default='kdv',
                    help="System name (configs/<system>.yaml). Default: kdv.")
    p.add_argument('--radii', default='2,4,8,auto',
                    help="Comma-separated patch_radius values; 'auto' allowed.")
    p.add_argument('--cv-methods',
                    default='mad_median,mad_median_lin,std_mean,std_mean_lin,iqr_sat',
                    help="Comma-separated CV metric keys.")
    p.add_argument('--placement', default='sobol',
                    choices=('off', 'jitter', 'sobol'),
                    help="Patch-center placement strategy. Default: sobol.")
    p.add_argument('--patch-stride', type=str, default='8',
                    help="(legacy single-value) used when --strides is not set.")
    p.add_argument('--strides', default=None,
                    help="Comma-separated patch_stride values; '2auto' allowed "
                         "(no-overlap stride from resolved r). When set, "
                         "overrides --patch-stride and produces a per-stride "
                         "table row.")
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args(argv)

    radii = [_parse_radius(x.strip()) for x in args.radii.split(',') if x.strip()]
    cvs = [x.strip() for x in args.cv_methods.split(',') if x.strip()]
    if args.strides:
        strides = [_parse_stride(x.strip()) for x in args.strides.split(',') if x.strip()]
    else:
        strides = [_parse_stride(args.patch_stride)]

    print(f'Sweep: system={args.system} placement={args.placement} '
          f'seed={args.seed}')
    print(f'  radii: {radii}')
    print(f'  strides: {strides}')
    print(f'  cv methods: {cvs}')
    print()

    cfg = load_config(args.system)
    cfg.hparams['moeadd']['early_stop_on_truth'] = False
    pipeline_kwargs = pipeline_settings('new')
    _set_seeds(args.seed)

    # Load symbolic truth from YAML (same as kdv_sindy_test).
    cfg_path = os.path.join(_THIS_DIR, 'configs', f'{args.system}.yaml')
    with open(cfg_path) as fh:
        truth_eqs = yaml.safe_load(fh).get('truth_equations') or []

    # Pre-resolve r_auto and per-axis r_auto so fractional sentinels
    # ('0.25auto', '0.5auto_pa', ...) get concrete values. Built once
    # against the shared patch_radius cache.
    need_auto = any(
        (r == 'auto' or r == 'auto_pa'
         or (isinstance(r, tuple) and r[0] == 'frac_auto'))
        for r in radii
    )
    r_auto_resolved = None
    r_auto_pa_resolved = None
    if need_auto:
        from epde.supplementary import (
            resolve_patch_radius_from_input,
            resolve_patch_radius_per_axis_from_input,
        )
        gv.set_gram_config('local', 'auto', 8,
                            ridge_lambda_=1e-8,
                            gram_jitter_=args.placement,
                            gram_jitter_seed_=args.seed)
        _set_seeds(args.seed)
        search_warm = build_pool_only(cfg, pipeline_kwargs)
        r_auto_resolved = resolve_patch_radius_from_input(
            gv.grid_cache.inner_shape, n_features_aug=11)
        r_auto_pa_resolved = resolve_patch_radius_per_axis_from_input(
            gv.grid_cache.inner_shape, n_features_aug=11)
        print(f'  (r_auto = {r_auto_resolved}; '
              f'r_auto_pa = {r_auto_pa_resolved})')

    # Header.
    header = (f"{'r':>5} {'s':>6} {'cv_method':<15} {'M':>5} "
              f"{'surv':>7} {'fitness':>10} {'stab':>10} {'wall':>8}  coefs")
    print(header)
    print('-' * (len(header) + 40))

    for r in radii:
     # Resolve fractional / per-axis sentinels -> concrete value(s).
     if isinstance(r, tuple) and r[0] == 'frac_auto':
         _, frac, per_axis = r
         if per_axis and r_auto_pa_resolved is not None:
             r_concrete = tuple(max(1, int(round(frac * rd)))
                                for rd in r_auto_pa_resolved)
             r_label = f'{frac:g}auto_pa{tuple(r_concrete)}'
         else:
             r_concrete = max(1, int(round(frac * r_auto_resolved)))
             r_label = f'{frac:g}auto({r_concrete})'
     elif r == 'auto':
         r_concrete = 'auto'
         r_label = (f'auto({r_auto_resolved})'
                    if r_auto_resolved is not None else 'auto')
     elif r == 'auto_pa':
         r_concrete = 'auto_per_axis'
         r_label = (f'auto_pa{tuple(r_auto_pa_resolved)}'
                    if r_auto_pa_resolved is not None else 'auto_pa')
     else:
         r_concrete = int(r)
         r_label = str(int(r))
     for s in strides:
        for cv in cvs:
            # Reconfigure globals before each combo. ``set_gram_config``
            # also resets gram_jitter_seed, so we keep it tied to args.seed
            # for reproducibility within this sweep.
            gv.set_gram_config('local', r_concrete, s,
                                ridge_lambda_=1e-8,
                                gram_jitter_=args.placement,
                                gram_jitter_seed_=args.seed)
            gv.set_cv_method(cv)
            _set_seeds(args.seed)

            # Pool needs to be rebuilt per combo because the patch
            # geometry caches inside the search builder. Cheap (~1s).
            search = build_pool_only(cfg, pipeline_kwargs)
            fit_op = make_fit_operator()
            coords, data, variable_names, dim = cfg.load_data()
            if len(variable_names) == 1:
                seeded = truth_eqs[0]
            else:
                seeded = {var: truth_eqs[i]
                          for i, var in enumerate(variable_names)}
            seeded = _normalize_grid_labels(seeded)

            try:
                summaries = _seed_and_eval(
                    seeded, search, fit_op, tuple(variable_names),
                )
            except Exception as exc:
                print(_format_row(r, cv, 0, 0, 0, float('nan'),
                                  float('nan'), 0.0, f'EXC: {exc!r}'))
                continue

            # Aggregate across all variables (KdV has 1; multi-eq
            # systems would sum survived/truth across vars). NOTE: the
            # iterator is renamed ``summ`` to avoid shadowing ``s`` (the
            # outer stride loop variable used in the print row).
            total_survived = sum(summ['n_survived'] for summ in summaries.values())
            total_truth = sum(summ['n_truth'] for summ in summaries.values())
            wall = sum(summ['wall'] for summ in summaries.values())
            fitness = float(np.mean([summ['fitness']
                                     for summ in summaries.values()]))
            stab = float(np.mean([summ['coef_stability']
                                  for summ in summaries.values()]))
            # Concatenate per-var surviving coef strings.
            coefs_strs = []
            for v, summ in summaries.items():
                for (n, c) in summ['surviving']:
                    coefs_strs.append(f'{n[:40]}={c:+.3f}')
            coefs_str = '; '.join(coefs_strs)

            M = '?'  # M is not surfaced through the sparsity op cleanly;
                     # we skip it here for compactness.
            # ``s`` may be the sentinel '2auto' -- print whichever the
            # caller passed; the actual stride number is derived inside
            # LocalPatchGramSetup against the resolved radius.
            print(f"{r_label:>14} {str(s):>6} {cv:<15} {M:>5} "
                  f"{total_survived:>3}/{total_truth:<3} "
                  f"{fitness:>10.4g} {stab:>10.4g} {wall:>7.2f}s  {coefs_str}")
    print()
    return 0


if __name__ == '__main__':
    sys.exit(main())
