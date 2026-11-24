"""Run the paired protocol in fresh Python processes with a fixed hash seed."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def execute(job):
    system, seed, arm = job
    tag = f'{system}_{seed}_{arm}'
    if (ROOT / 'plain_results' / (tag + '.json')).exists():
        return 0
    with (ROOT / 'logs' / (tag + '.log')).open('w', encoding='utf-8') as log:
        result = subprocess.run([sys.executable, '-u', str(ROOT/'run.py'), system, str(seed), arm],
                                env=dict(os.environ, PYTHONHASHSEED='0'), stdout=log, stderr=subprocess.STDOUT)
    print(tag, result.returncode, flush=True)
    return result.returncode


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--workers', type=int, default=4)
    p.add_argument('--systems', nargs='+', default=['ode','vdp','wave','ac'])
    p.add_argument('--seeds', type=int, nargs='+', default=[73001,73002,73003])
    args = p.parse_args()
    (ROOT/'logs').mkdir(exist_ok=True)
    jobs = [(s, n, a) for n in args.seeds for s in args.systems for a in ['wape','weak']]
    with ThreadPoolExecutor(args.workers) as pool:
        status = list(pool.map(execute, jobs))
    sys.exit(int(any(status)))
