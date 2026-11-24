"""Rerun both discovery designs on the original fields with matched search seeds."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import os
import subprocess
import sys
from unittest.mock import patch

from experiment import ROOT, live, trial, whole_record


def directory(name, seed, design):
    return ROOT/'clean_results'/design/name/str(seed)


def launch(job):
    name, seed, design = job
    dest = directory(name, seed, design)
    if (dest/'archive.json.gz').exists():
        return
    dest.mkdir(parents=True, exist_ok=True)
    with (dest/'search.log').open('w') as out:
        result = subprocess.run([sys.executable, '-u', __file__, '--worker', name, str(seed), design],
                                stdout=out, stderr=subprocess.STDOUT, env=dict(os.environ, PYTHONHASHSEED='0'))
    if result.returncode:
        print('Failed:', name, seed, design, flush=True)
        return job
    print(name, seed, design, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker', nargs=3)
    parser.add_argument('--workers', type=int, default=6)
    args = parser.parse_args()
    if args.worker:
        name, seed_text, design = args.worker
        seed = int(seed_text)
        split = {'full': whole_record, 'contiguous': live.split}[design]
        with patch.object(live, 'split', split), patch.object(trial, 'folder', return_value=directory(name, seed, design)):
            trial.worker(name, seed, 'candidate', noise_level=0)
    else:
        jobs = [(n, s, d) for s in trial.CONFIRMATION for n in live.DATASETS for d in ('full','contiguous')]
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            failures = [job for job in pool.map(launch, jobs) if job is not None]
        if failures:
            raise SystemExit(f'{len(failures)} failed searches: {failures}')
