"""Run the paired frequency campaign one CPU-limited job at a time."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import psutil

ROOT = Path(__file__).resolve().parent
NAMES = ('duffing', 'kdv_cossin', 'ode', 'ode_simple', 'ac', 'burgers', 'burgers_inviscid',
         'kdv', 'ks', 'lorenz', 'lv', 'ns', 'pde_compound', 'pde_divide', 'vdp', 'wave')
THREADS = ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS',
           'BLIS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS')


def worker(name, seed, noise, arm, cpu):
    proc = psutil.Process()
    proc.cpu_affinity([cpu])
    if os.name == 'nt':
        proc.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    os.environ.update(dict.fromkeys(THREADS, '1'))
    from frequency_study import job
    import torch
    from threadpoolctl import threadpool_info, threadpool_limits
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    with threadpool_limits(1):
        pools = [(pool['internal_api'], pool['num_threads']) for pool in threadpool_info()]
        if proc.cpu_affinity() != [cpu] or any(count != 1 for _, count in pools):
            raise RuntimeError(f'Could not enforce one CPU per worker: {proc.cpu_affinity()}, {pools}')
        print(f'Worker affinity {proc.cpu_affinity()}, numerical thread pools {pools}.', flush=True)
        job(name, seed, noise, arm)


def launch(name, seed, noise, arm, ceiling):
    dest = ROOT/'results'/name/str(seed)/f'noise{noise}'/arm
    score = dest/'score.json'
    if score.is_file():
        return json.loads(score.read_text())
    eligible = psutil.Process().cpu_affinity()
    cpu_share = 100 / psutil.cpu_count()
    if ceiling < cpu_share:
        raise ValueError('The CPU ceiling must accommodate one logical processor.')
    admission = ceiling - cpu_share
    print(f'Waiting for host CPU <= {admission:g}% for {name} {seed} {noise}% {arm}.', flush=True)
    while True:
        usage = psutil.cpu_percent(interval=3, percpu=True)
        if sum(usage)/len(usage) <= admission:
            break
        time.sleep(12)
    cpu = min(eligible, key=lambda i: usage[i])
    dest.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONHASHSEED='0', **dict.fromkeys(THREADS,'1'))
    args = [sys.executable, '-u', __file__, '--worker', name, str(seed), str(noise), arm, str(cpu)]
    with (dest/'search.log').open('w') as log, subprocess.Popen(args, env=env, stdout=log,
                                                                stderr=subprocess.STDOUT) as child:
        process = psutil.Process(child.pid)
        paused = False
        try:
            while child.poll() is None:
                load = psutil.cpu_percent(interval=3)
                if child.poll() is not None:
                    break
                if load > ceiling and not paused:
                    process.suspend()
                    paused = True
                elif paused and load <= admission:
                    process.resume()
                    paused = False
                if paused:
                    time.sleep(3)
        except psutil.NoSuchProcess:
            pass
        finally:
            if child.poll() is None:
                if paused:
                    process.resume()
                child.terminate()
        if child.wait():
            raise RuntimeError(f'Worker failed. See {dest / "search.log"}')
    return json.loads(score.read_text())


def campaign(ceiling):
    arms = ('native', 'fft', 'fitted')
    results = []
    for seed in (923701, 923702, 923703):
        for name in NAMES:
            for noise in (0, 5):
                for arm in arms:
                    results.append(launch(name, seed, noise, arm, ceiling))
        if seed == 923701:
            screen = [r for r in results if r['dataset'] != 'ode_simple']
            total = {a: sum(r['matched'] for r in screen if r['arm'] == a) for a in arms}
            positive = [a for a in ('fft', 'fitted') if total[a] > total['native']]
            winner = max(positive, key=lambda a: (total[a],
                         -sum(r['total_work'] for r in screen if r['arm'] == a))) if positive else None
            (ROOT/'selection.json').write_text(json.dumps(dict(strict_matched=total, winner=winner), indent=2))
            if winner is None:
                break
            arms = ('native', winner)
    (ROOT/'results.json').write_text(json.dumps(results, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpu-ceiling', type=float, default=75.)
    parser.add_argument('--worker', nargs=5, metavar=('DATASET', 'SEED', 'NOISE', 'ARM', 'CPU'))
    args = parser.parse_args()
    if args.worker:
        name, seed, noise, arm, cpu = args.worker
        worker(name, int(seed), int(noise), arm, int(cpu))
    elif 0 < args.cpu_ceiling <= 100:
        campaign(args.cpu_ceiling)
    else:
        parser.error('--cpu-ceiling must be in (0, 100].')
