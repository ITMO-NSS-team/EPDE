"""Discover on a prefix, then validate on previously unseen, equal-length time segments."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import replace
import json
import os
import subprocess
import sys
from unittest.mock import patch

from filelock import FileLock
from settings import FRESH, ROOT, limit_threads


def run(name, seed):
    from cpu_budget import await_capacity
    from noise_suite import atomic_text, epochs, live, measurement, trial, whole_record
    from compare import read_archive, structural_matches
    from score_archive import ArchiveScorer
    import numpy as np
    dest = ROOT/'time_results'/name/str(seed)
    dest.mkdir(parents=True, exist_ok=True)
    full = live.load_data(name)
    raw, info = measurement('correlated')(full, 5, seed)
    gap = 14
    size = (len(full.grids[0])-gap)//2
    def segment(start):
        sl = slice(start, start+size)
        p = replace(full, grids=tuple(g[sl] for g in full.grids), data={k:v[sl] for k,v in full.data.items()})
        return p, {k:v[sl].copy() for k,v in raw.items()}
    training, fields = segment(0)
    metadata = dict(info, segment=[0,size],
                    field_rms={k:float(np.sqrt(np.mean((v-training.data[k])**2))) for k,v in fields.items()})

    def observations(p, level, noise_seed):
        assert p is training and level == 5 and seed == noise_seed
        return fields, metadata

    with FileLock(dest/'worker.lock'):
        if not (dest/'archive.json.gz').exists():
            await_capacity()
            with (dest/'search.log').open('w', encoding='utf8', buffering=1) as log, \
                    redirect_stdout(log), redirect_stderr(log), patch.object(live,'load_data',return_value=training), \
                    patch.object(live,'split',whole_record), patch.object(trial,'folder',return_value=dest):
                trial.worker(name, seed, 'candidate', noise_level=5, epoch_budget=epochs(name), observation=observations)
        archive = read_archive(dest)
        assert archive['measurement'] == metadata and archive['epochs_completed'] == epochs(name)
        assert archive['train_hashes'] == {k:live.R.digest(v) for k,v in fields.items()}
        selected, digests = {}, {}
        for rule, start in [('train',0),('adjacent',size),('buffered',size+gap)]:
            p, record = segment(start)
            assert rule=='train' or not np.intersect1d(p.grids[0],training.grids[0]).size
            scorer = ArchiveScorer(p, archive)
            errors = scorer.trajectory_error([record])
            if rule == 'train':
                np.testing.assert_allclose(errors[scorer.valid],scorer.train[scorer.valid],rtol=1e-6,atol=2e-9)
                errors = scorer.train
            selected[rule] = scorer.select(errors)
            digests[rule] = {k:live.R.digest(v) for k,v in record.items()}
        _,_,hit = structural_matches(name, training, archive, np.asarray(archive['systems'],int))
        results = {key:dict(front=sel['front'],front_hit=hit(sel['front']),front_size=len(sel['front']))
                   for key,sel in selected.items()}
        atomic_text(dest/'pareto.json', json.dumps(dict(dataset=name, seed=seed, size=size, gap=gap,
                    generated=hit(range(len(archive['systems']))), hashes=digests, methods=results),allow_nan=False))


def launch(job):
    name, seed = job
    subprocess.run([sys.executable, '-u', __file__, '--worker',name,str(seed)],
                   env=dict(os.environ,PYTHONHASHSEED='0'),check=True)
    print(f'Validated unseen time segment for {name} {seed}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',nargs=2,metavar=('DATASET','SEED'))
    args = parser.parse_args()
    limit_threads()
    if args.worker:
        name, seed = args.worker
        assert name in ('ode','vdp') and int(seed) in FRESH
        run(name,int(seed))
    else:
        with FileLock(ROOT/'time_discovery.lock',timeout=0), ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(launch,[(name,seed) for seed in FRESH for name in ('ode','vdp')]))
