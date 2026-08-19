"""Lightweight retry/condition-loop instrumentation.

Off by default. Set ``EPDE_LOOP_STATS=1`` to enable; ~30 source-level
``record(...)`` call sites then accumulate per-loop stats that
``report()`` formats as a table.

A second metric class -- wall-clock timers -- is exposed via
``timer(site)``. Same env-var gate, same ``report()`` output (second
table). Both classes share the site namespace, so a site may appear in
both tables (e.g., ``EqRPS.outer`` records iters; ``EqRPS.apply``
records wall-clock).

Cost when disabled: a single global-var read per ``record`` / ``timer``
call (the latter returns a shared no-op context-manager singleton).
Cost when enabled: a dict lookup + list append per loop exit, plus a
``perf_counter`` pair + accumulate per ``timer`` exit.
"""
from __future__ import annotations

import os
import sys
import time
from collections import defaultdict
from typing import Optional

_ENABLED = os.environ.get('EPDE_LOOP_STATS', '0') == '1'


def _new_bucket():
    return {'entries': 0, 'iters': [], 'hit_cap': 0, 'early_exit': 0, 'caps': set()}


def _new_timer_bucket():
    return {'entries': 0, 'total_s': 0.0, 'max_s': 0.0}


_stats = defaultdict(_new_bucket)
_timers = defaultdict(_new_timer_bucket)


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


class _ActiveTimer:
    """Per-call active timer; one instance per ``with timer(site):``."""
    __slots__ = ('_site', '_t0')

    def __init__(self, site: str) -> None:
        self._site = site
        self._t0 = 0.0

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        dt = time.perf_counter() - self._t0
        b = _timers[self._site]
        b['entries'] += 1
        b['total_s'] += dt
        if dt > b['max_s']:
            b['max_s'] = dt
        return False


class _NoopTimer:
    """Singleton CM returned when ``EPDE_LOOP_STATS`` is off."""
    __slots__ = ()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


_NOOP_TIMER = _NoopTimer()


def timer(site: str):
    """Return a context manager that accumulates wall-clock under ``site``.

    Disabled fast path: returns a shared singleton no-op CM; the
    ``with`` block then costs one __enter__/__exit__ method call and
    nothing else. Enabled path: allocates a small ``_ActiveTimer``
    per call.
    """
    if not _ENABLED:
        return _NOOP_TIMER
    return _ActiveTimer(site)


def timed(site: str):
    """Decorator wrapping ``fn`` with ``timer(site)``.

    Avoids re-indenting large ``apply`` bodies when adding probes. When
    ``EPDE_LOOP_STATS`` is off, cost is one extra function call + the
    no-op CM enter/exit, all of which are negligible compared to the
    wrapped operator work.
    """
    def deco(fn):
        def wrapper(*args, **kwargs):
            with timer(site):
                return fn(*args, **kwargs)
        wrapper.__wrapped__ = fn
        wrapper.__name__ = getattr(fn, '__name__', 'wrapper')
        wrapper.__qualname__ = getattr(fn, '__qualname__', 'wrapper')
        wrapper.__doc__ = fn.__doc__
        return wrapper
    return deco


def reset() -> None:
    _stats.clear()
    _timers.clear()


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


def timers_snapshot() -> dict:
    """Return a copy of the timers dict for external consumers.

    Used by ``profile_loop_stats.py`` to build the cross-system
    compare table without re-parsing the report text.
    """
    return {site: dict(b) for site, b in _timers.items()}


def report(path: Optional[str] = None) -> str:
    """Format all recorded loops + timers as two tables.

    Writes to ``path`` if given AND also returns the string.
    """
    sites = sorted(_stats.keys(),
                   key=lambda s: -sum(_stats[s]['iters']) if _stats[s]['iters'] else 0)
    lines = []
    header = (f"{'site':<45} {'entries':>8} {'mean':>7} {'med':>5} "
              f"{'p95':>5} {'max':>5} {'cap':>6} {'%cap':>6} "
              f"{'%early':>7} {'totIters':>10}")
    lines.append('LOOPS')
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

    lines.append('')
    timer_sites = sorted(_timers.keys(),
                         key=lambda s: -_timers[s]['total_s'])
    timer_header = (f"{'site':<45} {'entries':>8} {'total_s':>10} "
                    f"{'mean_ms':>10} {'max_ms':>10} {'share%':>7}")
    lines.append('TIMERS')
    lines.append(timer_header)
    lines.append('-' * len(timer_header))
    if not _ENABLED:
        lines.append('(EPDE_LOOP_STATS disabled -- set EPDE_LOOP_STATS=1 to record)')
    if timer_sites:
        total_max = max(_timers[s]['total_s'] for s in timer_sites)
    else:
        total_max = 0.0
    for site in timer_sites:
        b = _timers[site]
        n = b['entries']
        if n == 0:
            continue
        mean_ms = 1000.0 * b['total_s'] / n
        max_ms = 1000.0 * b['max_s']
        share = (100.0 * b['total_s'] / total_max) if total_max > 0 else 0.0
        lines.append(
            f"{site:<45} {n:>8d} {b['total_s']:>10.2f} "
            f"{mean_ms:>10.2f} {max_ms:>10.2f} {share:>6.1f}%"
        )

    text = '\n'.join(lines)
    if path is not None:
        with open(path, 'w') as f:
            f.write(text + '\n')
    return text
