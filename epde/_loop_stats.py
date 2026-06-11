"""Lightweight retry/condition-loop instrumentation.

Off by default. Set ``EPDE_LOOP_STATS=1`` to enable; ~30 source-level
``record(...)`` call sites then accumulate per-loop stats that
``report()`` formats as a table.

Cost when disabled: a single global-var read per ``record`` call.
Cost when enabled: a dict lookup + list append per loop exit.
"""
from __future__ import annotations

import os
import sys
from collections import defaultdict
from typing import Optional

_ENABLED = os.environ.get('EPDE_LOOP_STATS', '0') == '1'


def _new_bucket():
    return {'entries': 0, 'iters': [], 'hit_cap': 0, 'early_exit': 0, 'caps': set()}


_stats = defaultdict(_new_bucket)


def enabled() -> bool:
    return _ENABLED


def record(site: str, iters: int, cap: int) -> None:
    """Record one loop exit.

    ``site`` is a human label like ``"EqRPS.outer"``. ``iters`` is the
    number of iterations actually executed. ``cap`` is the loop's
    maximum (use ``sys.maxsize`` for condition-driven loops with no
    explicit cap).
    """
    if not _ENABLED:
        return
    b = _stats[site]
    b['entries'] += 1
    b['iters'].append(iters)
    b['caps'].add(cap)
    if iters >= cap:
        b['hit_cap'] += 1
    elif iters <= 1:
        b['early_exit'] += 1


def reset() -> None:
    _stats.clear()


def _stats_for(name: str) -> dict:
    b = _stats[name]
    n = b['entries']
    iters = b['iters']
    if n == 0:
        return {'entries': 0}
    iters_sorted = sorted(iters)
    median = iters_sorted[n // 2]
    return {
        'entries': n,
        'mean': sum(iters) / n,
        'median': median,
        'max': max(iters),
        'p95': iters_sorted[min(n - 1, int(n * 0.95))],
        'total_iters': sum(iters),
        'hit_cap_pct': 100.0 * b['hit_cap'] / n,
        'early_exit_pct': 100.0 * b['early_exit'] / n,
        'cap': max(b['caps']) if b['caps'] else 0,
    }


def report(path: Optional[str] = None) -> str:
    """Format all recorded loops as a table, sorted by total iterations.

    Writes to ``path`` if given AND also returns the string.
    """
    sites = sorted(_stats.keys(),
                   key=lambda s: -sum(_stats[s]['iters']) if _stats[s]['iters'] else 0)
    lines = []
    header = (f"{'site':<45} {'entries':>8} {'mean':>7} {'med':>5} "
              f"{'p95':>5} {'max':>5} {'cap':>6} {'%cap':>6} "
              f"{'%early':>7} {'totIters':>10}")
    lines.append(header)
    lines.append('-' * len(header))
    if not _ENABLED:
        lines.append('(EPDE_LOOP_STATS disabled -- set EPDE_LOOP_STATS=1 to record)')
    for site in sites:
        s = _stats_for(site)
        if s['entries'] == 0:
            continue
        cap_str = 'inf' if s['cap'] >= sys.maxsize else str(s['cap'])
        lines.append(
            f"{site:<45} {s['entries']:>8d} {s['mean']:>7.2f} "
            f"{s['median']:>5d} {s['p95']:>5d} {s['max']:>5d} "
            f"{cap_str:>6} {s['hit_cap_pct']:>5.1f}% "
            f"{s['early_exit_pct']:>6.1f}% {s['total_iters']:>10d}"
        )
    text = '\n'.join(lines)
    if path is not None:
        with open(path, 'w') as f:
            f.write(text + '\n')
    return text
