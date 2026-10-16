"""Cache of the discovery runs the notebooks show.

A notebook cell calls :func:`cached_run` instead of ``run_one``: when
``results/notebook_cache`` already holds a record of the same run -- same
data set, variant, noise, seed and overrides, the same merged configuration
and the same computation content -- it is loaded; otherwise the run is computed and
stored. Nothing about the search changes, only where it is computed.

Notebook files are maintained directly. Cells compute missing runs through
``cached_run``; the cache is an execution helper and does not generate notebooks.

Delete ``results/notebook_cache`` to force everything to be recomputed.
"""

import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

from . import env
from .identity import experiment_identity
from .paths import PIC_DIR, REPO_ROOT, RESULTS_DIR

CACHE_DIR = RESULTS_DIR / 'notebook_cache'


def _stem(variant: str, noise: float, seed: int, overrides: Optional[dict]) -> str:
    stem = f'{variant}__noise{float(noise):.17g}__seed{seed}'
    if overrides:
        digest = hashlib.sha1(json.dumps(overrides, sort_keys=True).encode()).hexdigest()[:10]
        stem += f'__{digest}'
    return stem


def cache_path(name, variant='default', noise=0.0, seed=0, overrides=None) -> Path:
    return CACHE_DIR / 'runs' / name / f'{_stem(variant, noise, seed, overrides)}.json'


def _valid(record: dict, name, variant, noise, seed, overrides) -> bool:
    """Only identical requests, source content, data and dependencies can be reused."""
    if record.get('status') != 'ok':
        return False
    if any(record.get(k) != v for k, v in
           [('dataset', name), ('variant', variant), ('noise', noise), ('seed', seed)]):
        return False
    return record.get('identity') == experiment_identity(name, variant, noise, seed, overrides)


def cached_run(name: str, variant: str = 'default', noise: float = 0.0, seed: int = 0,
               overrides: Optional[dict] = None) -> dict:
    """``run_one`` with a cache; the returned record says ``from_cache``."""
    path = cache_path(name, variant, noise, seed, overrides)
    if path.exists():
        try:
            record = json.loads(path.read_text(encoding='utf-8'))
            if _valid(record, name, variant, noise, seed, overrides):
                record['from_cache'] = True
                return record
        except (OSError, ValueError):
            pass
    path.unlink(missing_ok=True)
    from .runner import run_one
    identity = experiment_identity(name, variant, noise, seed, overrides)
    record = run_one(name, variant, noise, seed, overrides=overrides)
    if record.get('identity') != identity:
        raise ValueError('run identity changed during computation; result was not cached')
    if record['status'] == 'ok':
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(record, indent=1, default=str), encoding='utf-8')
    record['from_cache'] = False
    return record


def print_run(record: dict) -> None:
    """Status, fit time, the final Pareto front (compromise pick marked) and
    the structural metrics of one record."""
    origin = '(from cache)' if record.get('from_cache') else ''
    print(record['status'], f"fit {record.get('fit_seconds', 0):.0f} s", origin)
    selected = record.get('metrics', {}).get('selected_index')
    for i, system in enumerate(record.get('front', [])):
        print(f'[{i}]' + (' <- compromise pick' if i == selected else ''))
        for eq in system:
            print('    ', eq)
    print({k: v for k, v in record.get('metrics', {}).items() if k != 'best_index'})
    if record['status'] != 'ok':
        print('Search needs attention:', record.get('error', record['status']))


def _flatten(overrides: Optional[dict], prefix: str = ''):
    """``{'search': {'evolution': {'training_epochs': 2}}}`` ->
    ``['search.evolution.training_epochs=2']`` for ``bench.py run --set``."""
    out = []
    for key, value in (overrides or {}).items():
        path = f'{prefix}{key}'
        if isinstance(value, dict):
            out += _flatten(value, path + '.')
        else:
            out.append(f'{path}={json.dumps(value)}')
    return out


def _compute(job, timeout):
    from .campaign import MEMORY_RETRIES, MEMORY_WAIT, _kill_tree, _memory_refusal
    name, variant, noise, seed, overrides = job
    out = cache_path(name, variant, noise, seed, overrides)
    out.parent.mkdir(parents=True, exist_ok=True)
    identity = experiment_identity(name, variant, noise, seed, overrides)
    pending = out.with_suffix('.pending.json')
    pending.unlink(missing_ok=True)
    out.unlink(missing_ok=True)
    log = out.with_suffix('.log')
    cmd = [sys.executable, str(PIC_DIR / 'bench.py'), 'run', name, '--variant', variant,
           '--noise', str(noise), '--seed', str(seed), '--out', str(pending)]
    for item in _flatten(overrides):
        cmd += ['--set', item]
    child_env = dict(os.environ, PYTHONIOENCODING='utf-8', MPLBACKEND='Agg')
    t0 = time.perf_counter()
    for attempt in range(1, MEMORY_RETRIES + 2):
        pending.unlink(missing_ok=True)
        with open(log, 'w', encoding='utf-8') as fh:
            proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT,
                                    cwd=str(REPO_ROOT), env=child_env)
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                _kill_tree(proc.pid)
                pending.unlink(missing_ok=True)
                out.write_text(json.dumps(dict(dataset=name, variant=variant, noise=noise,
                    seed=seed, status='timeout', identity=identity)), encoding='utf-8')
                return job, 'timeout', time.perf_counter() - t0
        if attempt > MEMORY_RETRIES or not _memory_refusal(pending):
            break
        time.sleep(MEMORY_WAIT)
    try:
        record = json.loads(pending.read_text(encoding='utf-8'))
        if record.get('identity') != identity:
            raise ValueError('child identity differs from the planned notebook run')
        pending.write_text(json.dumps(record, indent=1), encoding='utf-8')
        pending.replace(out)
        status = record.get('status')
    except (OSError, ValueError) as exc:
        status = 'error'
        pending.unlink(missing_ok=True)
        out.write_text(json.dumps(dict(dataset=name, variant=variant, noise=noise, seed=seed,
            status=status, identity=identity, error=f'child produced no valid record: {exc}')), encoding='utf-8')
    return job, status, time.perf_counter() - t0


def precompute(jobs, workers: int = 4, timeout: float = 3 * 3600) -> None:
    """Compute every job not yet validly cached, ``workers`` at a time."""
    todo = []
    for job in dict.fromkeys((n, v, float(z), int(s), json.dumps(o or {}, sort_keys=True))
                             for n, v, z, s, o in jobs):
        name, variant, noise, seed, ov = job
        overrides = json.loads(ov) or None
        path = cache_path(name, variant, noise, seed, overrides)
        if path.exists():
            try:
                if _valid(json.loads(path.read_text(encoding='utf-8')), name, variant, noise, seed, overrides):
                    continue
            except (OSError, ValueError):
                pass
        todo.append((name, variant, noise, seed, overrides))
    print(f'notebook runs: {len(todo)} to compute with {workers} worker(s) -> {CACHE_DIR}', flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(_compute, job, timeout) for job in todo]
        for i, fut in enumerate(as_completed(futures), 1):
            (name, variant, noise, seed, _), status, seconds = fut.result()
            print(f'[{i}/{len(todo)}] {name:16} {variant:11} noise {noise:g} seed {seed}: '
                  f'{status} ({seconds:.0f} s)', flush=True)
