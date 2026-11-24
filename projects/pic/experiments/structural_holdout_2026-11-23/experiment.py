"""Matched discovery on alternating and contiguous samples of one regular grid."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
TRIAL = ROOT.parent / 'candidate_frequency_2026-11-23'
sys.path.insert(0, str(TRIAL))
import candidate_trial as trial
from candidate_trial import live, np


def alternate(axes):
    axis = max(range(len(axes)), key=lambda j: len(axes[j]))
    end = 2*(len(axes[axis])//2)
    def side(offset):
        slices = [slice(None)]*len(axes)
        slices[axis] = slice(offset, end, 2)
        return tuple(slices), [a[s] for a,s in zip(axes,slices)]
    train, held = side(0), side(1)
    assert not np.intersect1d(train[1][axis],held[1][axis]).size
    return axis, train, held


def directory(name, seed, arm):
    return ROOT/'results'/name/str(seed)/arm


def whole_record(axes):
    axis = max(range(len(axes)), key=lambda j: len(axes[j]))
    side = (tuple(slice(None) for _ in axes), axes)
    return axis, side, side


def full_directory(name, seed, arm):
    return ROOT/'full_results'/name/str(seed)/arm


def launch(job):
    name, seed, design = job
    dest = (full_directory if design=='full' else directory)(name, seed, 'candidate')
    if (dest/'archive.json.gz').exists():
        return
    dest.mkdir(parents=True, exist_ok=True)
    with (dest/'search.log').open('w') as log:
        child = subprocess.run([sys.executable, '-u', __file__, '--worker', name, str(seed), '--design',design],
                               stdout=log, stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONHASHSEED='0'))
    if child.returncode:
        raise RuntimeError((dest/'search.log').read_text(errors='replace')[-3000:])
    print(name,seed,flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--worker',nargs=2)
    parser.add_argument('--confirmation',action='store_true')
    parser.add_argument('--design',choices=['alternate','full'],default='alternate')
    parser.add_argument('--workers',type=int,default=4)
    args = parser.parse_args()
    if args.worker:
        name, seed = args.worker
        partition, dest = (whole_record,full_directory) if args.design=='full' else (alternate,directory)
        with patch.object(live,'split',partition), patch.object(trial,'folder',dest):
            trial.worker(name,int(seed),'candidate')
    else:
        seeds = trial.CONFIRMATION if args.confirmation else live.SEEDS
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(launch,[(name,seed,args.design) for seed in seeds for name in live.DATASETS]))
