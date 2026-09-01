"""Controlled measurement of the per-equation RPS reset-coupling optimization.

Runs build_search for the given systems with EVERYTHING seeded (np.random,
Python ``random``, torch, PYTHONHASHSEED) and ``EPDE_LOOP_STATS=1`` so the run
is deterministic and the RPS term-sweep counters are recorded. Prints, per
system:

  * EqRPS.outer total_iters  -> total per-equation right-part SWEEPS actually
    run (a gated/unchanged equation contributes 0). This is the quantity the
    optimization reduces.
  * VWSRSparsity.apply entries -> number of sparse-regression fits (one per
    swept candidate target). Drops in lockstep with the sweeps.
  * front_sig -> md5 of the sorted discovered-equation text. With full seeding
    and an RNG-neutral change this MUST be identical before vs after, proving
    the optimization is result-preserving.

Usage:  python projects/thesis/_measure_reset_opt.py [sys ...] [--epochs N]
Run it once, ``git stash`` the operator edits, run again, ``git stash pop``;
diff the two outputs.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import os
import random
import sys

os.environ['EPDE_LOOP_STATS'] = '1'

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, '..', '..'))
for _p in (_REPO_ROOT, _THIS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402
import torch  # noqa: E402

from epde import _loop_stats  # noqa: E402
from epde import globals as _gv  # noqa: E402
from thesis_runner import build_search, load_config, pipeline_settings  # noqa: E402


def _seed_everything(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def _bucket(site: str):
    """Return (entries, total_iters) for a loop-stats LOOP site, or (0,0)."""
    s = _loop_stats._stats.get(site)
    if not s:
        return (0, 0)
    iters = s.get('iters', [])
    return (len(iters), int(sum(iters)))


def _timer_entries(site: str) -> int:
    b = _loop_stats._timers.get(site)
    return int(b['entries']) if b else 0


def measure(system: str, epochs: int, seed: int = 0) -> dict:
    _loop_stats.reset()
    _seed_everything(seed)
    # (Gram mode is derived from the instability metric now: the default
    # 'chi2' already resolves to the varying-coefficient Gram this asked for.)
    cfg = load_config(system)
    cfg.hparams['moeadd']['early_stop_on_truth'] = False
    cfg.hparams['moeadd']['training_epochs'] = int(epochs)
    with contextlib.redirect_stdout(io.StringIO()):
        search = build_search(cfg, pipeline_settings('new'))

    front = search.optimizer.pareto_levels.levels[0]
    texts = []
    for soeq in front:
        eqs = sorted(e.text_form for e in soeq)
        texts.append(' ;; '.join(eqs))
    texts.sort()
    sig = hashlib.md5('\n'.join(texts).encode()).hexdigest()[:12]

    outer_entries, outer_iters = _bucket('EqRPS.outer')
    return {
        'system': system,
        'n_eq': len(list(front[0])) if len(front) else 0,
        'n_front': len(front),
        'eqrps_calls': outer_entries,         # eq_selector.apply invocations
        'sweeps': outer_iters,                # actual term-sweeps (gated->0)
        'vwsr_fits': _timer_entries('VWSRSparsity.apply'),
        'front_sig': sig,
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument('systems', nargs='*', default=['ac', 'lv', 'lorenz'])
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args(argv)

    print(f"{'sys':<8}{'n_eq':>5}{'n_front':>8}{'eqrps_calls':>12}"
          f"{'sweeps':>8}{'vwsr_fits':>10}  front_sig")
    print('-' * 70)
    for s in args.systems:
        r = measure(s, args.epochs, args.seed)
        print(f"{r['system']:<8}{r['n_eq']:>5}{r['n_front']:>8}"
              f"{r['eqrps_calls']:>12}{r['sweeps']:>8}{r['vwsr_fits']:>10}"
              f"  {r['front_sig']}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
