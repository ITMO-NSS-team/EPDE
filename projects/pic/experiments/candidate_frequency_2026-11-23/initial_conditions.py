"""Paired rediscovery of the forced oscillator from two separately integrated trajectories."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import json
import os
import subprocess
import sys
from unittest.mock import patch

from candidate_trial import ROOT, Trial, live, np
from evaluate_candidates import ARMS, assess_run, check_pair, forcing_audit, read_archive, write_csv
from scipy.integrate import solve_ivp

INITIAL_STATES = {'displacement':(1.,0.), 'velocity':(0.,1.)}
SEEDS = range(923801,923804)
OUT = ROOT/'initial_conditions'


def data(state):
    original = live.load_data('ode')
    t, = live.R.axes_of(original)
    def derivative(time, values):
        u, v = values
        return v, 1.5*time-4*u-np.sin(2*time)*v
    def integrate(eps):
        sol = solve_ivp(derivative,(t[0],t[-1]),INITIAL_STATES[state],t_eval=t,method='DOP853',rtol=eps,atol=eps/10)
        assert sol.success and np.array_equal(sol.t,t)
        return sol.y[0]
    u = integrate(1.e-11)
    np.testing.assert_allclose(u,integrate(1.e-12),rtol=1.e-9,atol=1.e-9)
    assert not np.allclose(u,original.data['u'])
    return replace(original,data={'u':u})


def folder(state,seed,arm):
    return OUT/state/str(seed)/arm


def worker(state,seed,arm):
    problem = data(state)
    dest = folder(state,seed,arm)
    trial = Trial(problem.variables,arm=='candidate')
    with patch.object(live,'load_data',return_value=problem):
        if not (dest/'archive.json.gz').exists():
            live.run('ode',seed,arm,observer_factory=lambda _:trial,operator_run=trial.run,
                     destination=dest,trig_factory=trial.family)
        assess_run(('ode',seed,arm),destination=dest,moves=None,timing='frequency_seconds')


def launch(job):
    state,seed,arm = job
    dest = folder(*job)
    dest.mkdir(parents=True,exist_ok=True)
    if (dest/'assessment.json').exists():
        return
    mode = 'a' if (dest/'archive.json.gz').exists() else 'w'
    with (dest/'search.log').open(mode) as log:
        run = subprocess.run([sys.executable,'-u',__file__,'--worker',state,str(seed),arm],
                             env=dict(os.environ,PYTHONHASHSEED='0'),stdout=log,stderr=subprocess.STDOUT)
    if run.returncode:
        raise RuntimeError((dest/'search.log').read_text(errors='replace')[-2200:])


def publish():
    systems, forced, details = [], [], []
    for state in INITIAL_STATES:
        problem = data(state)
        with patch.object(live,'load_data',return_value=problem):
            for seed in SEEDS:
                dirs = [folder(state,seed,a) for a in ARMS]
                archives = [read_archive(d) for d in dirs]
                check_pair(*archives)
                reports = [json.loads((d/'assessment.json').read_text()) for d in dirs]
                assert reports[0]['peaks']==reports[1]['peaks']
                local = []
                result = forcing_audit('ode',seed,archives[1],reports[1],local)
                forced.append(dict(state=state,**result))
                details.extend(dict(state=state,**r) for r in local)
                for b in ('full','total'):
                    for arm,r in zip(ARMS,reports):
                        systems.append(dict(state=state,seed=seed,arm=arm,budget=b,held=r[b]['system']['held'],
                                            generated=r[b]['laws'][0]['generated'],calls=r[b]['system']['calls'],
                                            front_size=r[b]['system']['held_system_front_size'],
                                            extra_fits=r['extra_fits'],replay_error=r['replay_error'],
                                            train_hash=r['train_hashes']['u']))
    for name,rows in [('systems',systems),('forcing',forced),('forcing_profiles',details)]:
        if rows:
            write_csv(OUT/(name+'.csv'),rows)
    print([(s,a,sum(r['held'] for r in systems if r['state']==s and r['arm']==a and r['budget']=='total'))
           for s in INITIAL_STATES for a in ARMS],flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--worker',nargs=3)
    p.add_argument('--workers',type=int,default=2)
    args = p.parse_args()
    if args.worker:
        s,n,a = args.worker
        worker(s,int(n),a)
    else:
        jobs = [(s,n,a) for s in INITIAL_STATES for n in SEEDS for a in ARMS]
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(launch,jobs))
        publish()
