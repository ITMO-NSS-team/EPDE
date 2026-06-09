"""True-vs-spurious stability MARGIN harness for ``gram_mode='vcoef'``.

Bypasses MOEA/D. For each system: seed the truth equation, pull its true
feature columns + target, then inject *spurious* feature columns built as
``true_feature * coordinate_ramp`` (a legitimate library product -- feature
times a coordinate token -- whose best constant coefficient is
region-dependent, exactly the kind of term a search wrongly proposes). A
good stability estimator scores the TRUE terms low and the SPURIOUS terms
high.

Reported per system:
* ``vc_true_max``   -- max varying-coefficient score over the true terms.
* ``vc_spur_min``   -- min vc score over the injected spurious terms.
* ``vc_margin``     -- ``vc_spur_min - vc_true_max`` (want > 0 on all systems).
* the patch-CV counterpart (``mad_median``, the current default) for A/B.

The margin being POSITIVE on every system -- with a single fixed config and
no per-dataset tuning -- is the dataset-independence the design targets.

Usage:
    python projects/thesis/vcoef_margin.py [--systems ode,lv,wave,kdv,...]
        [--axis 0] [--ramp-strength 1.0]
"""
from __future__ import annotations

import argparse
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import numpy as np
import yaml

import epde.globals as global_var
from epde.interface.equation_translator import translate_equation
from epde.operators.common.stability import VaryingCoefSetup, calculate_weights
from kdv_sindy_test import build_pool_only, make_fit_operator  # noqa: E402
from kdv_sindy_sweep import _normalize_grid_labels  # noqa: E402
from thesis_runner import _set_seeds, load_config, pipeline_settings  # noqa: E402

_ALL = ['ode', 'lv', 'vdp', 'lorenz', 'kdv', 'kdv_cossin', 'wave',
        'burgers_viscous', 'burgers_inviscid', 'pde_divide', 'pde_compound',
        'ac', 'ks', 'ns']


def _coord_ramps(grid_shape, axis):
    """Normalised coordinate ramp(s) in [-1, 1] flattened over the grid.

    Returns one ramp per requested axis (``axis=-1`` -> every axis), each a
    length-N vector matching the C-order flatten of the feature columns.
    """
    D = len(grid_shape)
    axes = range(D) if axis < 0 else [axis]
    ramps = {}
    for d in axes:
        if d >= D:
            continue
        n_d = grid_shape[d]
        line = np.linspace(-1.0, 1.0, n_d)
        shp = [1] * D
        shp[d] = n_d
        ramps[d] = np.broadcast_to(line.reshape(shp), grid_shape).reshape(-1)
    return ramps


def _patch_cv_scores(features, target, sw, grid_shape):
    """Per-feature patch-CV (mad_median, the current default) for A/B."""
    try:
        weights = np.array(calculate_weights(features, target, sw, grid_shape,
                                             True))
        center = np.median(weights, axis=0)
        mad = np.median(np.abs(weights - center), axis=0)
        with np.errstate(divide='ignore', invalid='ignore'):
            cv = np.nan_to_num((mad ** 2) / (center ** 2))
        return cv[:-1]  # drop intercept column
    except Exception:
        return None


def margin_for_system(system, axis, ramp_strength):
    cfg = load_config(system)
    pipeline_kwargs = pipeline_settings('new')
    _set_seeds(0)
    search = build_pool_only(cfg, pipeline_kwargs)

    cfg_path = os.path.join(_THIS_DIR, 'configs', f'{system}.yaml')
    with open(cfg_path) as fh:
        truth_eqs = yaml.safe_load(fh).get('truth_equations') or []
    _, _, variable_names, _ = cfg.load_data()
    all_vars = list(variable_names)
    if len(all_vars) == 1:
        seeded = truth_eqs[0]
    else:
        seeded = {var: truth_eqs[i] for i, var in enumerate(all_vars)}
    seeded = _normalize_grid_labels(seeded)
    metaparams = {('sparsity', v): {'optimizable': False, 'value': 1e-6}
                  for v in all_vars}
    soeq = translate_equation(seeded, search.pool, all_vars=all_vars)

    grid_shape = global_var.grid_cache.inner_shape
    sw = global_var.grid_cache.g_func[global_var.grid_cache.g_func_mask]
    ramps = _coord_ramps(grid_shape, axis)

    rows = []
    for v in all_vars:
        eq = soeq.vals[v]
        eq.main_var_to_explain = v
        eq.metaparameters = metaparams
        eq.weights_internal = np.ones(len(eq.structure) - 1)
        eq.weights_internal_evald = True
        eq.weights_final_evald = True
        _, target, features = eq.evaluate(normalize=True, return_val=False)
        if features is None or features.ndim != 2 or features.shape[1] == 0:
            continue
        n_true = features.shape[1]

        # Inject spurious columns: each true feature * each coordinate ramp.
        spur_cols = []
        for d, ramp in ramps.items():
            spur_cols.append(features * (ramp_strength * ramp)[:, None])
        spur = np.hstack(spur_cols) if spur_cols else np.zeros((len(target), 0))
        n_spur = spur.shape[1]
        aug = np.hstack([features, spur])

        vc = VaryingCoefSetup(aug, target, sw, grid_shape,
                              main_var=v).score(None)
        vc_true = vc[:n_true]
        vc_spur = vc[n_true:n_true + n_spur]

        pcv = _patch_cv_scores(aug, target, sw, grid_shape)
        if pcv is not None and len(pcv) >= n_true + n_spur:
            pcv_true, pcv_spur = pcv[:n_true], pcv[n_true:n_true + n_spur]
        else:
            pcv_true = pcv_spur = None

        rows.append(dict(
            var=v, n_true=n_true, n_spur=n_spur,
            vc_true_max=float(np.max(vc_true)),
            vc_spur_min=float(np.min(vc_spur)) if n_spur else float('nan'),
            vc_margin=(float(np.min(vc_spur) - np.max(vc_true))
                       if n_spur else float('nan')),
            pcv_true_max=(float(np.max(pcv_true))
                          if pcv_true is not None else float('nan')),
            pcv_spur_min=(float(np.min(pcv_spur))
                          if pcv_spur is not None and len(pcv_spur)
                          else float('nan')),
            pcv_margin=(float(np.min(pcv_spur) - np.max(pcv_true))
                        if pcv_true is not None and pcv_spur is not None
                        and len(pcv_spur) else float('nan')),
        ))
    return rows


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--systems', default=','.join(_ALL),
                   help="Comma-separated system list (default: all 14).")
    p.add_argument('--axis', type=int, default=-1,
                   help="Coordinate axis for the spurious ramp; -1 = every axis.")
    p.add_argument('--ramp-strength', type=float, default=1.0)
    args = p.parse_args(argv)

    systems = [s.strip() for s in args.systems.split(',') if s.strip()]
    hdr = (f"{'system':<16} {'var':<4} {'nT':>3} {'nS':>3} "
           f"{'vc_trueMax':>11} {'vc_spurMin':>11} {'vc_MARGIN':>11}   "
           f"{'pcv_trueMax':>11} {'pcv_spurMin':>11} {'pcv_MARGIN':>11}")
    print(hdr)
    print('-' * len(hdr))
    vc_pos = vc_tot = 0
    for s in systems:
        try:
            rows = margin_for_system(s, args.axis, args.ramp_strength)
        except Exception as e:
            print(f"{s:<16} ERROR: {type(e).__name__}: {e}")
            continue
        for r in rows:
            vc_tot += 1
            vc_pos += int(r['vc_margin'] > 0)
            print(f"{s:<16} {r['var']:<4} {r['n_true']:>3} {r['n_spur']:>3} "
                  f"{r['vc_true_max']:>11.4g} {r['vc_spur_min']:>11.4g} "
                  f"{r['vc_margin']:>11.4g}   "
                  f"{r['pcv_true_max']:>11.4g} {r['pcv_spur_min']:>11.4g} "
                  f"{r['pcv_margin']:>11.4g}")
    print('-' * len(hdr))
    print(f"vcoef positive margin: {vc_pos}/{vc_tot} equations")
    return 0


if __name__ == '__main__':
    sys.exit(main())
