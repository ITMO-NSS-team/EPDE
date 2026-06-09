"""Run a single NEW-pipeline rep with loop counters enabled and report.

Sets ``EPDE_LOOP_STATS=1`` in the process environment **before** any
epde import, then runs ``build_search`` for each system (defaults:
lv + wave), prints the loop-stats table, and writes it to
``projects/thesis/profile_results/loop_stats_<system>.txt``. When
more than one system is profiled in a single invocation, also dumps
a side-by-side cross-system timer comparison to
``profile_results/timer_compare.txt``.

Usage:
    python projects/thesis/profile_loop_stats.py [system ...] [--epochs N] [--seed N]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time


# Enable instrumentation BEFORE any epde import — _loop_stats reads
# the env var at module-load time.
os.environ['EPDE_LOOP_STATS'] = '1'

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, '..', '..'))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from epde import _loop_stats  # noqa: E402
from thesis_runner import (  # noqa: E402
    _set_seeds,
    build_search,
    load_config,
    pipeline_settings,
)


def profile_system(system_name: str, pipeline: str = 'new', seed: int = 0,
                   epochs: int | None = None,
                   gram_mode: str = 'vcoef') -> tuple[float, dict]:
    """Profile one (system, pipeline, seed) rep.

    Returns ``(wall, timers)`` where ``wall`` is the outer wall-clock
    in seconds and ``timers`` is the snapshot returned by
    ``_loop_stats.timers_snapshot()`` so the caller can build a
    cross-system compare table.
    """
    cfg = load_config(system_name)
    cfg.hparams['moeadd']['early_stop_on_truth'] = False
    if epochs is not None:
        cfg.hparams['moeadd']['training_epochs'] = int(epochs)

    # Pin the gram config before any build_search call so all operator
    # paths see consistent settings for this rep.
    from epde import globals as _gv  # noqa: E402
    _gv.set_gram_config(gram_mode)

    pipeline_kwargs = pipeline_settings(pipeline)
    _set_seeds(seed)

    print(f"\n{'=' * 78}")
    print(f"LOOP-STATS  system={system_name}  pipeline={pipeline}  seed={seed}"
          f"  epochs={cfg.hparams['moeadd']['training_epochs']}")
    print(f"{'=' * 78}")

    _loop_stats.reset()
    t0 = time.time()
    build_search(cfg, pipeline_kwargs)
    wall = time.time() - t0
    print(f"\n[wall] build_search total: {wall:.2f}s\n")

    # Suffix output filenames with the gram mode so vcoef vs axis runs can
    # co-exist. The default (vcoef) gets no suffix.
    cfg_tag = "" if gram_mode == 'vcoef' else f"_{gram_mode}"

    out_dir = os.path.join(_THIS_DIR, 'profile_results')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"loop_stats_{system_name}{cfg_tag}.txt")
    text = _loop_stats.report(path=out_path)
    print(text)
    print(f"\n[saved] {out_path}")

    snapshot = _loop_stats.timers_snapshot()

    # JSON sidecar so parallel system runs can be aggregated post-hoc
    # by ``--compare-only``. One file per (system, seed, epochs, cfg)
    # tuple so concurrent processes never write the same path.
    json_path = os.path.join(
        out_dir,
        f"loop_stats_{system_name}_seed{seed}_ep{cfg.hparams['moeadd']['training_epochs']}{cfg_tag}.json"
    )
    with open(json_path, 'w') as f:
        json.dump({'system': system_name, 'pipeline': pipeline,
                   'seed': seed,
                   'epochs': cfg.hparams['moeadd']['training_epochs'],
                   'gram_mode': gram_mode,
                   'wall': wall, 'timers': snapshot}, f, indent=2)
    print(f"[saved] {json_path}")

    return wall, snapshot


def write_timer_compare(walls: dict, timers: dict, out_path: str) -> None:
    """Emit a side-by-side total_s per (site, system) table.

    ``walls`` maps system_name -> outer wall-clock seconds.
    ``timers`` maps system_name -> ``timers_snapshot()`` dict.
    Rows are sorted by the largest per-row total_s across systems
    (so the heaviest sites land at the top).
    """
    systems = list(timers.keys())
    all_sites = set()
    for system in systems:
        all_sites.update(timers[system].keys())

    def _row_max(site: str) -> float:
        return max(
            (timers[system].get(site, {}).get('total_s', 0.0) for system in systems),
            default=0.0,
        )

    sorted_sites = sorted(all_sites, key=_row_max, reverse=True)

    lines = []
    lines.append('TIMER COMPARE (total_s per system; share% = total_s / outer wall)')
    header_cells = [f"{'site':<35}"]
    for system in systems:
        header_cells.append(f"{system + ' s':>14}")
        header_cells.append(f"{system + ' %':>8}")
    header = '  '.join(header_cells)
    lines.append(header)
    lines.append('-' * len(header))
    lines.append(f"{'(outer wall)':<35}  " + '  '.join(
        f"{walls[system]:>13.2f}s  {100.0:>7.1f}%" for system in systems
    ))
    lines.append('-' * len(header))
    for site in sorted_sites:
        row_cells = [f"{site:<35}"]
        for system in systems:
            t = timers[system].get(site, {}).get('total_s', 0.0)
            share = (100.0 * t / walls[system]) if walls[system] > 0 else 0.0
            row_cells.append(f"{t:>13.2f}s")
            row_cells.append(f"{share:>7.1f}%")
        lines.append('  '.join(row_cells))

    text = '\n'.join(lines) + '\n'
    with open(out_path, 'w') as f:
        f.write(text)
    print(text)
    print(f"[saved] {out_path}")


def _load_sidecar_jsons(systems: list, seed: int, epochs: int) -> tuple[dict, dict]:
    """Read ``loop_stats_<sys>_seed<seed>_ep<epochs>.json`` sidecars
    written by prior parallel runs and return ``(walls, timers)``.
    Missing sidecars are reported and skipped.
    """
    out_dir = os.path.join(_THIS_DIR, 'profile_results')
    walls: dict[str, float] = {}
    timers: dict[str, dict] = {}
    for system_name in systems:
        json_path = os.path.join(
            out_dir, f"loop_stats_{system_name}_seed{seed}_ep{epochs}.json"
        )
        if not os.path.exists(json_path):
            print(f"[compare-only] missing sidecar: {json_path}")
            continue
        with open(json_path) as f:
            payload = json.load(f)
        walls[system_name] = float(payload['wall'])
        timers[system_name] = payload['timers']
    return walls, timers


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('systems', nargs='*', default=['lv', 'wave'])
    p.add_argument('--pipeline', default='new', choices=('legacy', 'new'))
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--epochs', type=int, default=5)
    p.add_argument('--compare-only', action='store_true',
                   help="Skip runs; aggregate existing sidecar JSONs into timer_compare.txt.")
    p.add_argument('--gram-mode', default='vcoef',
                   choices=('axis', 'vcoef'),
                   help="Gram / stability strategy (default: vcoef = "
                        "varying-coefficient stability). 'axis' = legacy "
                        "axis-aligned sliding-window backup (var/mu^2 CV).")
    args = p.parse_args(argv)

    if args.compare_only:
        walls, timers = _load_sidecar_jsons(args.systems, args.seed, args.epochs)
        if not timers:
            print("[compare-only] no sidecars found; nothing to compare.")
            return 1
        out_dir = os.path.join(_THIS_DIR, 'profile_results')
        os.makedirs(out_dir, exist_ok=True)
        compare_path = os.path.join(out_dir, 'timer_compare.txt')
        write_timer_compare(walls, timers, compare_path)
        return 0

    walls: dict[str, float] = {}
    timers: dict[str, dict] = {}
    for system_name in args.systems:
        wall, snap = profile_system(system_name, pipeline=args.pipeline,
                                    seed=args.seed, epochs=args.epochs,
                                    gram_mode=args.gram_mode)
        walls[system_name] = wall
        timers[system_name] = snap

    if len(args.systems) > 1:
        out_dir = os.path.join(_THIS_DIR, 'profile_results')
        os.makedirs(out_dir, exist_ok=True)
        compare_path = os.path.join(out_dir, 'timer_compare.txt')
        write_timer_compare(walls, timers, compare_path)
    return 0


if __name__ == '__main__':
    sys.exit(main())
