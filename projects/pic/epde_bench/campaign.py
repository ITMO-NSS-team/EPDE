"""Campaigns: the Cartesian product data sets x variants x noise x seeds.

Each run is ``bench.py run ... --out <file>`` in its own process:

* EPDE keeps search settings in process globals ("the last search
  constructed wins for the process"), so runs sharing a process could leak
  settings into each other;
* a run that hangs or blows up memory is killed by the timeout without
  taking the campaign down;
* it works the same under Windows' ``spawn`` and POSIX ``fork``.

Layout of ``results/<name>/``::

    campaign.json                       what was asked for, environment
    runs/<dataset>/<variant>__noise<n>__seed<s>.json   one record per run
    logs/<dataset>/<variant>__noise<n>__seed<s>.log    stdout/stderr of the run

Re-running the same command skips finished runs, so an interrupted campaign
resumes where it stopped (``--retry-errors`` also redoes failed ones).
"""

import json
import os
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from . import env
from .config import parse_set, variants
from .datasets import names
from .paths import PIC_DIR, REPO_ROOT, RESULTS_DIR
from .identity import experiment_identity, problem_metadata


#: EPDE's cache asserts that a fixed share of the machine's RAM is free when
#: tensors are uploaded (cache_refactored.py: memoryUsageProperties, 5 % in
#: Cache.add). On a shared machine that can fail at start-up for reasons that
#: have nothing to do with the run; such runs are retried after a pause.
MEMORY_RETRIES = 5
MEMORY_WAIT = 90


def _memory_refusal(out: Path) -> bool:
    try:
        rec = json.loads(out.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return False
    return rec.get('status') == 'error' and 'memoryUsageProperties' in (rec.get('traceback') or '')


def _kill_tree(pid: int) -> None:
    """Kill a process and all its descendants. Killing only ``pid`` is not
    enough on Windows: a venv's ``python.exe`` is a launcher that runs the
    real interpreter as a child, which would keep computing after a timeout.

    Listing processes can be refused (sandboxes deny ``psutil`` access to the
    process table); the run is then killed directly together with its process
    group, so a timeout never leaves a search computing in the background."""
    import psutil
    procs = []
    try:
        parent = psutil.Process(pid)
        procs = parent.children(recursive=True) + [parent]
    except psutil.NoSuchProcess:
        return
    except psutil.Error:
        pass
    for p in procs:
        try:
            p.kill()
        except psutil.Error:
            pass
    if os.name != 'nt':
        try:
            os.killpg(pid, signal.SIGKILL)   # runs start in their own session
        except OSError:
            pass
    try:
        os.kill(pid, getattr(signal, 'SIGKILL', signal.SIGTERM))
    except OSError:
        pass
    try:
        psutil.wait_procs(procs, timeout=30)
    except psutil.Error:
        pass


def run_stem(variant: str, noise: float, seed: int) -> str:
    return f'{variant}__noise{float(noise):.17g}__seed{seed}'


def _jobs(args):
    datasets = args.datasets.split(',') if args.datasets else names(args.suite)
    chosen = args.variants.split(',')
    unknown = set(chosen) - set(variants())
    if unknown:
        raise SystemExit(f'unknown variants: {", ".join(sorted(unknown))}')
    jobs = [(d, v, n, s) for d in datasets for v in chosen for n in args.noise for s in args.seeds]
    if len(set(jobs)) != len(jobs):
        raise SystemExit('campaign grid contains duplicate jobs')
    return jobs


def _finished(path: Path, retry_errors: bool, identity=None) -> bool:
    if not path.exists():
        return False
    try:
        record = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return False
    if identity is None or record.get('identity') != identity:
        return False
    status = record.get('status')
    return status in ('ok', 'unsupported') or (status in ('error', 'timeout') and not retry_errors)


def _planned_job(job, overrides):
    dataset, variant, noise, seed = job
    return dict(dataset=dataset, variant=variant, noise=noise, seed=seed,
                problem=problem_metadata(dataset),
                identity=experiment_identity(dataset, variant, noise, seed, overrides))


def _validate_manifest(path, planned, timeout):
    if not path.exists():
        return
    old = json.loads(path.read_text(encoding='utf-8'))
    if old.get('jobs') != planned or old.get('timeout') != timeout:
        raise SystemExit('Campaign identity changed; choose a new campaign name. '
                         'The existing manifest and results were preserved.')


def _run_job(root: Path, job, sets, timeout, planned=None):
    dataset, variant, noise, seed = job
    planned = planned or _planned_job(job, parse_set(sets))
    stem = run_stem(variant, noise, seed)
    out = root / 'runs' / dataset / f'{stem}.json'
    log = root / 'logs' / dataset / f'{stem}.log'
    out.parent.mkdir(parents=True, exist_ok=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, str(PIC_DIR / 'bench.py'), 'run', dataset, '--variant', variant,
           '--noise', str(noise), '--seed', str(seed), '--out', str(out)]
    for item in sets:
        cmd += ['--set', item]
    child_env = dict(os.environ, PYTHONIOENCODING='utf-8', MPLBACKEND='Agg')
    t0 = time.perf_counter()
    for attempt in range(1, MEMORY_RETRIES + 2):
        out.unlink(missing_ok=True)  # a retry must never accept the previous attempt's output
        with open(log, 'w', encoding='utf-8') as fh:
            try:
                proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT,
                                        cwd=str(REPO_ROOT), env=child_env,
                                        start_new_session=os.name != 'nt')
            except OSError as exc:
                out.write_text(json.dumps(dict(planned, status='error', error=str(exc))), encoding='utf-8')
                break
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                _kill_tree(proc.pid)
                out.write_text(json.dumps(dict(planned, status='timeout', total_seconds=timeout)), encoding='utf-8')
        if attempt > MEMORY_RETRIES or not _memory_refusal(out):
            break
        time.sleep(MEMORY_WAIT)                   # free RAM was short at start-up: try again
    try:
        record = json.loads(out.read_text(encoding='utf-8'))
        if not isinstance(record, dict) or record.get('status') not in ('ok', 'error', 'timeout', 'unsupported'):
            raise ValueError('child record has no recognized terminal status')
        if record.get('identity') != planned['identity']:
            raise ValueError('child record identity does not match the planned job')
    except (OSError, ValueError) as exc:
        record = dict(planned, status='error', error=f'no valid record written, see {log}: {exc}')
        out.write_text(json.dumps(record), encoding='utf-8')
    status = record.get('status')
    return job, status, time.perf_counter() - t0


def run_campaign(args) -> int:
    root = Path(RESULTS_DIR) / args.name
    jobs = _jobs(args)
    overrides = parse_set(args.set)
    planned = [_planned_job(job, overrides) for job in jobs]
    manifest = root / 'campaign.json'
    _validate_manifest(manifest, planned, args.timeout)
    todo = [(j, p) for j, p in zip(jobs, planned)
            if not _finished(root / 'runs' / j[0] / f'{run_stem(*j[1:])}.json',
                             args.retry_errors, p['identity'])]
    print(f'{len(jobs)} runs in the campaign, {len(jobs) - len(todo)} already done, '
          f'{len(todo)} to run with {args.workers} worker(s) -> {root}')
    if args.dry_run:
        for job, _ in todo:
            print('  ', *job)
        return 0
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / 'campaign.json'
    meta = {'name': args.name, 'updated': datetime.now(timezone.utc).isoformat(timespec='seconds'),
            'datasets': sorted({j[0] for j in jobs}), 'variants': args.variants.split(','),
            'noise': args.noise, 'seeds': args.seeds, 'set': args.set, 'timeout': args.timeout,
            'jobs': planned, 'command': ' '.join(sys.argv),
            'environment': env.environment_info(REPO_ROOT, probe_cuda=False)}
    if not manifest.exists():
        manifest.write_text(json.dumps(meta, indent=1), encoding='utf-8')
    done, failed = 0, False
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_run_job, root, job, args.set, args.timeout, plan): job
                   for job, plan in todo}
        for fut in as_completed(futures):
            try:
                job, status, seconds = fut.result()
            except Exception as exc:  # one broken run must not stop the others' bookkeeping
                job, status, seconds = futures[fut], f'error ({exc})', 0.
                failed = True
            done += 1
            print(f'[{done}/{len(todo)}] {job[0]:16} {job[1]:11} noise {job[2]:g} seed {job[3]}: '
                  f'{status} ({seconds:.0f} s)', flush=True)
    return 1 if failed else 0
