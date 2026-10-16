"""One run = one (data set, variant, noise level, seed) -> one JSON record.

:func:`discover` is the plain EPDE workflow on a :class:`Problem` (the same
calls a notebook makes); :func:`run_one` wraps it with noise, seeding,
timing, scoring and bookkeeping. Campaigns (:mod:`epde_bench.campaign`)
call ``bench.py run`` in a fresh process per run: EPDE keeps its settings
in process globals, so separate processes are what makes runs independent.
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
from .preprocessing import make_preprocessor, validate_data, validate_derivatives
from .library import _interior
from .tokens import stored_groups


def build_search(problem: Problem, search_cfg: dict, data: Optional[dict] = None,
                 device: str = 'auto'):
    """``EpdeSearch`` + domain + trajectory for ``problem``, ready to ``fit``.

    Returns ``(search, trajectory, extra_token_families)``. ``data``
    replaces ``problem.data`` (e.g. a noisy copy).
    """
    from epde import EpdeSearch
    data = problem.data if data is None else data
    validate_data(problem, data)
    validate_derivatives(problem, data, search_cfg)
    _interior(problem, search_cfg)
    search_cfg, bench_families = split_bench_tokens(search_cfg)
    solver = dict(search_cfg.get('solver', {}))
    solver['device'] = env.resolve_device(solver.get('device', device))
    search_cfg['solver'] = solver

    search = EpdeSearch(config=search_cfg)
    _, domain = search.createDomain(list(problem.grids), ID=0)
    families = bench_families + stored_groups(problem)
    if problem.extra_tokens is not None:
        families += problem.extra_tokens(problem)
    _, trajectory = search.createTrajectory(data, domain, cache_id=0, derivs=problem.derivs,
                                            preprocessor=make_preprocessor(search_cfg))
    return search, trajectory, families


def pareto_front(search):
    """Pareto-0 as ``(texts, objectives)``: each text is the list of the
    system's equations (one per variable)."""
    population = search.equations(only_print=False, num=1)
    if not population:
        return [], []
    level = population[0] if search.multiobjective_mode else population
    texts, objectives = [], []
    for solution in level:
        # text_form ends with a line of metaparameters ("{'max_terms_number': ...}")
        texts.append([line.strip() for line in solution.text_form.split('\n')
                      if '=' in line and not line.strip().startswith('{')])
        obj = getattr(solution, 'obj_fun', None)
        objectives.append(None if obj is None else [float(v) for v in list(obj)])
    return texts, objectives


def discover(problem: Problem, search_cfg: dict, data: Optional[dict] = None):
    """Run EPDE once. Returns ``(search, texts, objectives, fit_seconds)``."""
    search, trajectory, families = build_search(problem, search_cfg, data)
    t0 = time.perf_counter()
    search.fit(data=[trajectory], additional_tokens=families)
    fit_seconds = time.perf_counter() - t0
    texts, objectives = pareto_front(search)
    return search, texts, objectives, fit_seconds


def run_one(dataset: str, variant: str = 'default', noise: float = 0.0, seed: int = 0,
            overrides: Optional[dict] = None, config_path=None) -> dict:
    """One scored run. Never raises: errors end up in the record."""
    cfg = load_config(dataset, variant, overrides, config_path)
    record = {'dataset': dataset, 'variant': variant, 'method': cfg['method'],
              'noise': noise, 'seed': seed,
              'started': datetime.now(timezone.utc).isoformat(timespec='seconds'),
              'config': cfg, 'environment': env.environment_info(REPO_ROOT)}
    t_start = time.perf_counter()
    try:
        env.seed_everything(seed)
        problem = load(dataset, **cfg.get('loader', {}))
        record['problem'] = {'title': problem.title, 'kind': problem.kind,
                             'source': problem.source, 'shape': list(problem.shape),
                             'variables': problem.variables, 'meta': problem.meta,
                             'truth': problem.truth,
                             'n_truth_alternatives': len(problem.truth_alternatives)}
        data = problem.noisy(noise, seed)
        if cfg['method'] == 'epde':
            search_cfg = resolve_for_problem(cfg, problem)
            record['search_config'] = search_cfg
            _, texts, objectives, fit_seconds = discover(problem, search_cfg, data)
        else:
            raise ValueError(f"unknown method {cfg['method']!r}")
        record.update(status='ok', fit_seconds=fit_seconds, front=texts, objectives=objectives)
        record['metrics'] = metrics.score_run(texts, objectives, problem.truth_systems)
        sel = record['metrics'].get('selected_index')
        record['selected'] = texts[sel] if texts and sel is not None else None
    except Exception as exc:                                   # noqa: BLE001
        record.update(status='error', error=repr(exc), traceback=traceback.format_exc())
    record['total_seconds'] = time.perf_counter() - t_start
    return record
