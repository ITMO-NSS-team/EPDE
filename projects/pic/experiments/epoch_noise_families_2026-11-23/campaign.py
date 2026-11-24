"""Schedule the matched campaign alongside the existing full-record searches."""
from functools import lru_cache
import os
import subprocess
import sys
from time import sleep

from filelock import FileLock, Timeout
import psutil
from cpu_budget import restrict

if __name__ == '__main__':
    restrict(psutil.Process())

from noise_suite import FAMILIES, ROOT, SEEDS, baseline_directory, directory, jobs, live
from noise_worker import verified
from noise_results import render
from compare import read_archive


def worker_locked(job):
    path = directory(*job) / 'worker.lock'
    if path.exists():
        try:
            with FileLock(path, timeout=0):
                pass
        except Timeout:
            return True
    return False


@lru_cache(maxsize=None)
def has_epoch_history(dest):
    return 'epoch_fronts' in read_archive(dest)


def ns_available():
    # A baseline archive with no recorded epoch history needs one full patience replay.
    for seed in SEEDS:
        for level in (0, 5):
            dest = baseline_directory('ns', seed, level)
            if not (dest / 'patience.json').is_file() and not has_epoch_history(dest):
                return False
    return True


def launch(job):
    dest = directory(*job)
    dest.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONHASHSEED='0', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', OMP_NUM_THREADS='1')
    with (dest / 'worker.log').open('a', encoding='utf8') as log:
        process = subprocess.Popen([sys.executable, '-u', str(ROOT / 'noise_worker.py'), *map(str, job)],
                                   stdout=log, stderr=subprocess.STDOUT, env=env)
    print('Started', *job, 'PID', process.pid, flush=True)
    return process


def campaign():
    all_jobs = jobs()
    controls = [baseline_directory(n, s, q) for n in live.DATASETS for s in SEEDS for q in (0, 5)]
    done = {j for j in all_jobs if verified(j)}
    pilots = {('ode_simple', SEEDS[0], f) for f in FAMILIES}
    active, attempts, failed = {}, {}, set()
    last_state = None
    while True:
        for j, p in list(active.items()):
            if p.poll() is None:
                continue
            del active[j]
            if p.returncode == 0 and verified(j):
                done.add(j)
                print('Verified', *j, flush=True)
            elif attempts[j] >= 2:
                failed.add(j)
                print('Failed twice', *j, directory(*j) / 'worker.log', flush=True)
            else:
                print('Retrying', *j, flush=True)
        # Account for a standalone pilot or workers surviving a scheduler restart.
        external = set()
        for j in all_jobs:
            if j in done or j in active or j in failed:
                continue
            if (directory(*j) / 'verified.json').exists() and verified(j):
                done.add(j)
            elif worker_locked(j):
                external.add(j)
        archive_count = sum((p / 'archive.json.gz').is_file() for p in controls)
        expanded = archive_count == len(controls) and pilots <= done
        occupied = set(active) | external
        regular = sum(j[0] != 'ns' for j in occupied)
        large = any(j[0] == 'ns' for j in occupied)
        for j in all_jobs:
            if j in done or j in occupied or j in failed or (pilots - done and j not in pilots):
                continue
            if j[0] == 'ns':
                if large or not expanded or not ns_available():
                    continue
                large = True
            else:
                if regular >= (4 if expanded else 1):
                    continue
                if not expanded and j[0] not in ('ode_simple', 'ode', 'vdp', 'duffing'):
                    continue
                regular += 1
            attempts[j] = attempts.get(j, 0) + 1
            active[j] = launch(j)
        if failed and not active and not external and (failed & pilots or len(done | failed) == len(all_jobs)):
            raise RuntimeError(f'Unfinished searches: {sorted(failed)}')
        scored_controls = sum((p / 'fronts.json').is_file() for p in controls)
        identities = sum((p / 'lorenz_identity.json').is_file() for p in controls if p.parent.parent.name == 'lorenz')
        state = len(done), scored_controls, identities
        if state != last_state:
            complete = render()
            print(f'{len(done)}/{len(all_jobs)} new searches verified, {scored_controls}/{len(controls)} controls scored',
                  flush=True)
            if complete:
                return
            last_state = state
        sleep(15)


if __name__ == '__main__':
    with FileLock(ROOT / 'campaign.lock', timeout=0):
        campaign()
