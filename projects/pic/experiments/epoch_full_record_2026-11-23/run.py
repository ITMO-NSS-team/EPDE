"""Full-record discovery with 50 scalar or 150 coupled epochs per run."""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import os
import subprocess
import sys
from unittest.mock import patch

from epoch_settings import OPTIONS, SEEDS, NOISE, directory, epochs, live, trial, whole_record


def discover(job):
    name, seed, noise = job
    dest = directory(*job)
    dest.mkdir(parents=True, exist_ok=True)
    with patch.object(live, 'split', whole_record), patch.object(trial, 'folder', return_value=dest):
        trial.worker(name, seed, 'candidate', noise_level=noise,
                     epoch_budget=epochs(name), loader_options=OPTIONS.get(name))


def launch(job):
    dest = directory(*job)
    if (dest/'archive.json.gz').is_file():
        return
    dest.mkdir(parents=True, exist_ok=True)
    with (dest/'search.log').open('w') as log:
        result = subprocess.run([sys.executable,'-u',__file__,'--worker',*map(str,job)],
                                stdout=log, stderr=subprocess.STDOUT,
                                env=dict(os.environ, PYTHONHASHSEED='0'))
    if result.returncode:
        tail = (dest/'search.log').read_text(errors='replace').splitlines()[-8:]
        print('FAILED', *job, *tail, sep='\n', flush=True)
        return job
    print('Completed', *job, flush=True)


def campaign(names, workers):
    jobs = [(n,s,q) for s in SEEDS for q in NOISE for n in names]
    # Separate lanes keep one large 3-D search resident and leave scalar jobs free to advance.
    failures = []
    with ThreadPoolExecutor(max_workers=1) as large, ThreadPoolExecutor(max_workers=workers) as regular:
        futures = [(large if job[0]=='ns' else regular).submit(launch, job) for job in jobs]
        for f in as_completed(futures):
            failure = f.result()
            if failure:
                failures.append(failure)
    if failures:
        raise SystemExit(f'Incomplete runs: {failures}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', nargs=3)
    parser.add_argument('--datasets', nargs='+', choices=live.DATASETS, default=live.DATASETS)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.worker:
        n, s, q = args.worker
        discover((n, int(s), int(q)))
    else:
        campaign(args.datasets, args.workers)
