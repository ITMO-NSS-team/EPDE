"""One run = one (data set, variant, noise level, seed) -> one JSON record.

:func:`discover` is the plain EPDE workflow on a :class:`Problem` (the same
calls a notebook makes); :func:`run_one` wraps it with noise, seeding,
timing, scoring and bookkeeping. Campaigns (:mod:`epde_bench.campaign`)
call ``bench.py run`` in a fresh process per run: process isolation lets campaigns enforce per-run time and memory limits.
"""

import time
import traceback
from datetime import datetime, timezone
from typing import Optional

from . import env, metrics
from .config import load_config, resolve_for_problem, split_bench_tokens
from .datasets import load
from .paths import REPO_ROOT
from .problem import Problem
from .preprocessing import derivative_orders
from .tokens import stored_groups


def _inputs(problem: Problem, search_cfg: dict, data: Optional[dict] = None,
            device: str = 'auto', *, live_trajectory=False):
    from epde import EpdeSearch
    sample = problem.trajectory_data(data, derivative_orders(search_cfg, len(problem.grids)))
    search_cfg, bench_families = split_bench_tokens(search_cfg)
    solver = dict(search_cfg.get('solver', {}))
    solver['device'] = env.resolve_device(solver.get('device', device))
    search_cfg['solver'] = solver

    # Older callable families can read or capture the active domain during construction.
    live_trajectory = live_trajectory or problem.extra_tokens is not None
    if live_trajectory:
        from .library import _interior
        _interior(problem, search_cfg)
    search = EpdeSearch(config=search_cfg)
    if live_trajectory:
        _, domain = search.createDomain(sample.grids, ID=sample.trajectory_id)
    families = bench_families + stored_groups(problem, include_tensors=live_trajectory)
    if problem.extra_tokens is not None:
        families += problem.extra_tokens(problem)
    if live_trajectory:
        _, trajectory = search.createTrajectory(sample.fields, domain, cache_id=sample.trajectory_id,
                                                derivs=sample.derivatives or None)
        trajectory.axis_names = tuple(sample.coordinates)
        return search, trajectory, families
    return search, sample, families


def build_search(problem, search_cfg, data=None, device='auto'):
    """Compatibility API for callers that build a pool or inspect the live trajectory."""
    return _inputs(problem, search_cfg, data, device, live_trajectory=True)


def front_values(result):
    """Ordered equation texts and objectives used by existing reports."""
    return ([[eq['text'] for eq in system['equations']] for system in result.front],
            [system['objectives'] for system in result.front])


def pareto_front(search):
    return front_values(search.result)


def _discover(problem, search_cfg, data):
    search, sample, families = _inputs(problem, search_cfg, data)
    t0 = time.perf_counter()
    try:
        result = search.discover(sample, additional_tokens=families)
        return search, result, time.perf_counter() - t0
    except BaseException:
        search.close()
        raise


def discover(problem: Problem, search_cfg: dict, data: Optional[dict] = None):
    """Return the live search, ordered equation texts, objectives and elapsed seconds."""
    search, result, elapsed = _discover(problem, search_cfg, data)
    texts, objectives = front_values(result)
    return search, texts, objectives, elapsed


def record_front(record):
    """Read both versioned discovery results and historical text-only records."""
    if 'discovery' in record:
        from epde import DiscoveryResult
        return front_values(DiscoveryResult.from_dict(record['discovery']))[0]
    return record.get('front', [])


def run_one(dataset: str, variant: str = 'default', noise: float = 0.0, seed: int = 0,
            overrides: Optional[dict] = None, config_path=None) -> dict:
    """One scored run. Never raises: errors end up in the record."""
    from .baselines import UnsupportedProblem
    from .identity import experiment_identity
    cfg = load_config(dataset, variant, overrides, config_path)
    record = {'dataset': dataset, 'variant': variant, 'method': cfg['method'],
              'noise': noise, 'seed': seed,
              'started': datetime.now(timezone.utc).isoformat(timespec='seconds'),
              'config': cfg, 'environment': env.environment_info(REPO_ROOT)}
    t_start = time.perf_counter()
    try:
        record['identity'] = experiment_identity(dataset, variant, noise, seed, overrides, config_path)
        env.seed_everything(seed)
        problem = load(dataset, **cfg.get('loader', {}))
        record['problem'] = {'title': problem.title, 'kind': problem.kind,
                             'source': problem.source, 'shape': list(problem.shape),
                             'variables': problem.variables, 'meta': problem.meta,
                             'truth': problem.truth,
                             'n_truth_alternatives': len(problem.truth_alternatives)}
        if noise > 0 and problem.derivs is not None:
            raise UnsupportedProblem(dataset, 'Artificial noise with supplied derivatives requires '
                                     'a consistent noisy field/gradient protocol; use noise=0.')
        data = problem.noisy(noise, seed)
        if cfg['method'] == 'epde':
            search_cfg = resolve_for_problem(cfg, problem)
            record['search_config'] = search_cfg
            search, result, fit_seconds = _discover(problem, search_cfg, data)
            search.close()
            texts, objectives = front_values(result)
            record['discovery'] = result.as_dict()
        elif cfg['method'] == 'pysindy':
            from .baselines import run_pysindy
            texts, objectives, fit_seconds, info = run_pysindy(problem, data, cfg, seed)
            record['baseline'] = info
        else:
            raise ValueError(f"unknown method {cfg['method']!r}")
        record.update(status='ok', fit_seconds=fit_seconds, front=texts, objectives=objectives)
        score_front = [s['equations'] for s in result.front] if cfg['method'] == 'epde' else texts
        record['metrics'] = metrics.score_run(score_front, objectives, problem.truth_systems)
        sel = record['metrics'].get('selected_index')
        record['selected'] = texts[sel] if texts and sel is not None else None
    except UnsupportedProblem as exc:
        record.update(status='unsupported', reason=exc.reason)
    except Exception as exc:                                   # noqa: BLE001
        record.update(status='error', error=repr(exc), traceback=traceback.format_exc())
    record['total_seconds'] = time.perf_counter() - t_start
    return record
