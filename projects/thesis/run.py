"""Unified CLI entry for the thesis Section 4.5 main comparison.

Usage:
    python projects/thesis/run.py <system> [--reps N] [--pipelines legacy new]
                                            [--outdir TAG] [--no-resume]
                                            [--seed-base N]

``<system>`` matches one of the YAML files in ``projects/thesis/configs/``
(e.g. ``lv``, ``lorenz``, ``kdv``). The default pipelines are ``legacy`` and
``new``; pass ``--pipelines`` to override (the eight valid labels are
``legacy``, ``new``, and the six off-diagonal ablation labels -- see
``thesis_runner._PIPELINE_SETTINGS``).

``--outdir TAG`` redirects results to ``results/TAG/<system>/`` so a tagged
sweep across multiple systems stays grouped under one folder. ``--outdir
/abs/path`` lands there directly.
"""

from __future__ import annotations

import argparse
import os
import re
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from thesis_runner import (  # noqa: E402
    ABLATION_PIPELINES,
    CONFIGS_DIR,
    PIPELINES,
    _PIPELINE_SETTINGS,
    load_config,
    run_smoke,
)


def _available_systems() -> list:
    if not os.path.isdir(CONFIGS_DIR):
        return []
    # ``defaults.yaml`` is the shared baseline that load_config merges
    # under every per-system config; it's not itself a runnable system.
    return sorted(
        os.path.splitext(f)[0]
        for f in os.listdir(CONFIGS_DIR)
        if f.endswith('.yaml') and f != 'defaults.yaml'
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        'system',
        help=f"system name (looked up as configs/<system>.yaml). "
             f"Available: {', '.join(_available_systems()) or '(none yet)'}",
    )
    parser.add_argument('--reps', type=int, default=30,
                        help="reps per pipeline (default: 30)")
    parser.add_argument(
        '--pipelines', nargs='+', default=list(PIPELINES),
        choices=tuple(_PIPELINE_SETTINGS),
        help=f"pipeline labels (default: {' '.join(PIPELINES)})",
    )
    parser.add_argument('--outdir', default=None,
                        help="results tag (lands at results/<tag>/<system>/) "
                             "or absolute path; default reuses cfg's outdir")
    parser.add_argument('--no-resume', dest='resume', action='store_false', default=True,
                        help="overwrite existing per-rep JSONs instead of skipping")
    parser.add_argument('--seed-base', type=int, default=0,
                        help="seed for rep 0 (rep i uses seed_base + i)")
    parser.add_argument('--epochs', type=int, default=None,
                        help="override moeadd.training_epochs from the YAML.")
    parser.add_argument('--gram-mode', default='vcoef',
                        choices=('axis', 'vcoef'),
                        help="Gram / stability strategy (default: vcoef = "
                             "varying-coefficient stability, patch-free, "
                             "data-driven basis modes). 'axis' = legacy "
                             "axis-aligned sliding-window backup (var/mu^2 CV).")
    parser.add_argument('--noise-level', type=float, default=0.0,
                        help="Additive Gaussian noise applied to every data "
                             "array returned by ``cfg.load_data``. Convention "
                             "matches PySINDy's noisy benchmark: "
                             "``sigma = noise_level * 0.01 * std(data)``. "
                             "Noise re-applied per rep with seed "
                             "``seed_base + rep_idx`` so reps see "
                             "independent realisations.")
    parser.add_argument('--vc-coord-penalty', type=float, default=None,
                        help="kappa weight for the vcoef coordinate-modulation "
                             "penalty (globals.vc_coord_penalty). 0 disables; "
                             "larger penalises coordinate-modulated spurious "
                             "terms harder. Default: leave globals' value.")
    parser.add_argument('--single-objective', default=None,
                        choices=('discrepancy', 'instability'),
                        help="Run the SINGLE-objective evolutionary optimizer "
                             "driven by this objective alone (forces "
                             "multiobjective_mode=False). 'instability' tests "
                             "whether vcoef instability alone finds the true "
                             "equation; 'discrepancy' is the residual baseline. "
                             "Default: unset = multi-objective MOEA/D.")
    parser.add_argument('--anchor-on-residual', action='store_true',
                        default=False,
                        help="In 'max_corr' anchor mode, anchor the L1 "
                             "threshold to the working residual max|X^T r| "
                             "instead of the raw target max|X^T y|.")
    args = parser.parse_args(argv)

    try:
        cfg = load_config(args.system)
    except FileNotFoundError as exc:
        parser.error(str(exc))

    # Pin the gram mode before the batch starts; applies to every rep in
    # run_smoke since the setting is a process-level global.
    from epde import globals as _gv
    _gv.set_gram_config(args.gram_mode)
    _gv.set_anchor_on_residual(args.anchor_on_residual)
    if args.vc_coord_penalty is not None:
        _gv.vc_coord_penalty = float(args.vc_coord_penalty)

    # Single-objective mode: pin the objective and flip the search to the
    # single-criterion optimizer for every rep in this batch.
    if args.single_objective is not None:
        _gv.set_single_objective_metric(args.single_objective)
        cfg.hparams['search']['multiobjective_mode'] = False

    # When ``--noise-level`` is set, monkey-patch ``cfg.load_data`` so
    # every call inside ``build_search`` injects independent Gaussian
    # noise sized at PySINDy's convention. ``_gv.noise_seed`` is
    # advanced by run_smoke per rep so each rep sees a fresh draw.
    if args.noise_level > 0:
        import numpy as _np
        orig_load = cfg.load_data
        nl = float(args.noise_level)
        def _noisy_load():
            coords, data, vars_, dim = orig_load()
            seed = getattr(_gv, 'noise_seed', None)
            if seed is None:
                seed = args.seed_base
            rng = _np.random.default_rng(int(seed))
            def _noisy(arr):
                a = _np.asarray(arr, dtype=_np.float64)
                sigma = nl * 0.01 * float(_np.std(a))
                if sigma > 0:
                    a = a + rng.normal(0.0, sigma, size=a.shape)
                return a
            if isinstance(data, _np.ndarray):
                noisy = _noisy(data)
            elif isinstance(data, (list, tuple)):
                noisy = type(data)(_noisy(a) for a in data)
            else:
                noisy = data
            return coords, noisy, vars_, dim
        cfg.load_data = _noisy_load

    if args.epochs is not None:
        cfg.hparams['moeadd']['training_epochs'] = int(args.epochs)

    run_smoke(
        cfg,
        reps=args.reps,
        pipelines=tuple(args.pipelines),
        seed_base=args.seed_base,
        resume=args.resume,
        outdir=args.outdir,
    )
    return 0


if __name__ == '__main__':
    sys.exit(main())
