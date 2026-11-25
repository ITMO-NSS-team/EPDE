"""Run the independent 5% correlated-noise cohort with the original epoch worker."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import os
import subprocess
import sys
from unittest.mock import patch

from filelock import FileLock
from settings import DATASETS, FRESH, ROOT, fresh_directory, limit_threads


def launch(job):
    name, seed = job
    dest = fresh_directory(name, seed, 'correlated')
    dest.mkdir(parents=True, exist_ok=True)
    with (dest / 'worker.log').open('a', encoding='utf8') as log:
        subprocess.run([sys.executable, '-u', __file__, '--worker', name, str(seed)],
                       env=dict(os.environ, PYTHONHASHSEED='0'), stdout=log, stderr=subprocess.STDOUT, check=True)
    print(f'Verified {name} {seed}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', nargs=2, metavar=('DATASET', 'SEED'))
    args = parser.parse_args()
    limit_threads()
    if args.worker:
        import noise_worker
        name, seed = args.worker
        assert name in DATASETS and int(seed) in FRESH
        with patch.object(noise_worker, 'directory', fresh_directory):
            noise_worker.run((name, int(seed), 'correlated'))
    else:
        with FileLock(ROOT / 'discovery.lock', timeout=0), ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(launch, [(n, s) for s in FRESH for n in DATASETS]))
