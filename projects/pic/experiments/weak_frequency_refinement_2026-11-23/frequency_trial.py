"""Run a native discovery search with an optional training-only frequency profile."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess
import sys
from time import monotonic

ROOT = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'guided_swap_2026-11-23'))
from swap_trial import Timeline, folder as archived_native, live
from refine import FrequencyProfile

CONFIRMATION = range(923601,923613)
ARMS = ('native','refined')


def folder(name, seed, arm):
    if arm=='native' and seed in live.SEEDS:
        return archived_native(name,seed,'native')
    return ROOT/'results'/name/str(seed)/arm


class Trial:
    def __init__(self):
        self.profiles = []
        self.work = {}
        self.seconds = 0.

    def observe(self, variables):
        self.timeline = Timeline(variables)
        return self.timeline

    def family(self, train, axes, plan, settings):
        family = live.U.TrigonometricTokens(freq_from_data=2)
        native = family.frequencies_from_data
        def estimate(trajectories):
            initial = native(trajectories)
            by_axis = initial if isinstance(initial,dict) else {0:initial}
            assert self.timeline.calls==0 and not self.profiles
            start = monotonic()
            profile = FrequencyProfile(train,axes,plan,settings)
            result = {}
            for axis,peaks in by_axis.items():
                rows = [profile.calibrate(axis,peaks,i) for i in range(len(peaks))]
                self.profiles.extend(rows)
                result[axis] = tuple(r['installed'] for r in rows)
                assert list(result[axis])==sorted(set(result[axis]))
            self.work = dict(profile.work)
            self.timeline.extra = profile.work['background_fits']+profile.work['profile_fits']
            self.seconds = monotonic()-start
            return result if isinstance(initial,dict) else result[0]
        family.frequencies_from_data = estimate
        return family

    def run(self, arm, seed, part, *, experiment):
        experiment()
        t = self.timeline
        return dict(arm=arm,profiles=self.profiles,work=self.work,frequency_seconds=self.seconds,
                    extra_fits=t.extra,gain_columns=0,hashes={},full_cost=t.full_cost,
                    system_cost=[t.system_cost[c] for c in sorted(t.systems)])


def launch(job):
    name,seed,arm = job
    dest = folder(*job)
    if (dest/'archive.json.gz').is_file():
        return
    if arm=='native' and seed in live.SEEDS:
        raise FileNotFoundError(f'Missing verified native archive at {dest}')
    dest.mkdir(parents=True,exist_ok=True)
    with (dest/'search.log').open('w') as log:
        cmd = [sys.executable,'-u',__file__,'--worker',name,str(seed),arm]
        result = subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONHASHSEED='0'))
    if result.returncode:
        raise RuntimeError((dest/'search.log').read_text(errors='replace')[-3500:])
    print(*job,flush=True)


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker',nargs=3)
    p.add_argument('--workers',type=int,default=6)
    p.add_argument('--datasets',nargs='+',choices=live.DATASETS,default=live.DATASETS)
    p.add_argument('--seeds',nargs='+',type=int,default=live.SEEDS)
    p.add_argument('--confirmation',action='store_true')
    args = p.parse_args()
    if args.worker:
        name,n,arm = args.worker
        t = Trial()
        live.run(name,int(n),arm,observer_factory=t.observe,operator_run=t.run,
                 destination=folder(name,int(n),arm),trig_factory=t.family if arm=='refined' else None)
    else:
        seeds = CONFIRMATION if args.confirmation else args.seeds
        jobs = [(s,n,a) for n in seeds for s in args.datasets for a in ARMS]
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(launch,jobs))
