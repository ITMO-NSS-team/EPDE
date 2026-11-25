"""Paired all-record discovery pilot. Each worker owns one search and its atomic archive."""
import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import replace
from functools import partial
import gzip
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
from time import monotonic
from unittest.mock import patch

from filelock import FileLock

ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT.parent/'trajectory_validation_2026-11-24'
sys.path.insert(0, str(PREVIOUS))
from settings import DATASETS, limit_threads

ARMS = {'one':1, 'repeat3':3, 'diverse3':3, 'repeat5':5, 'diverse5':5}
DRAWS = range(5)
SEARCH_SEED = 947301
POLICIES = ('systematic', 'per_support', 'first12')


def output_root(policy):
    if policy not in POLICIES:
        raise ValueError(f'Unknown frequency policy {policy}')
    return ROOT if policy == 'first12' else ROOT/policy


def records(name, arm, draw):
    from validate_trajectories import dynamics, problems
    from noise_suite import live, measurement
    from scipy.integrate import solve_ivp
    original, alternatives = problems(name)
    t, = live.R.axes_of(original)
    initial = json.loads((PREVIOUS/'simulator_check.json').read_text())[name]['initial']
    sol = solve_ivp(dynamics(name,original), (t[0],t[-1]), initial, t_eval=t,
                    method='DOP853', rtol=1e-11, atol=1e-13)
    if not sol.success:
        raise RuntimeError(sol.message)
    base = replace(original, data={var:sol.y[i] for i,var in enumerate(original.variables)})
    sources = [base, *alternatives] if arm.startswith('diverse') else [base]*5
    samples, metadata = [], []
    for i,p in enumerate(sources[:ARMS[arm]]):
        seed = 948000 + draw*10 + i
        fields, _ = measurement('correlated')(p,5,seed)
        samples.append((p.grids,fields))
        metadata.append(dict(state=i if arm.startswith('diverse') else 0, noise_seed=seed,
                             hashes={k:live.R.digest(v) for k,v in fields.items()}))
    return original, samples, metadata


def search(name, arm, draw, budget, destination, *, policy='systematic'):
    from noise_suite import live, trial
    import epde
    from epde.optimizers.nsga.optimizer import NSGAOptimizer
    from epde.operators.common.objectives import LOSS_NAN_VAL
    from compare import system_front
    import numpy as np
    problem, samples, metadata = records(name, arm, draw)
    seed = SEARCH_SEED + DATASETS.index(name)
    random.seed(seed)
    np.random.seed(seed)
    class Observer(live.Observer):
        def term(self, term):
            # Fitted frequencies must retain full precision even when evolutionary
            # structure labels group nearby continuous parameter values.
            label = tuple(sorted(('x' if f.ftype == 'grids' else f.label,
                                  tuple(sorted((v['name'],float(f.params[i]))
                                               for i,v in f.params_description.items())))
                                 for f in term.structure))
            if label not in self.label_indices:
                self.label_indices[label] = len(self.labels)
                self.labels.append(label)
            return self.label_indices[label]
    observer = Observer(problem.variables)
    point_ids, points, systems = {}, [], set()
    def point(equation):
        key = (*observer.key(equation), tuple(equation._record_wape))
        if key not in point_ids:
            point_ids[key] = len(points)
            points.append(key)
        return point_ids[key]
    def member(system):
        return tuple(point(eq) for eq in system.vals)
    insert = NSGAOptimizer._archive_insert
    def track(optimizer, system):
        obj = system.obj_fun
        if np.isfinite(obj).all() and np.all(obj < LOSS_NAN_VAL):
            systems.add(member(system))
        return insert(optimizer, system)
    # The bounded profiler fits a continuous parameter of an encountered equation.
    # Candidate admission and final system comparison always use the full vector.
    from epde.operators.common import plain_fit
    native_handover = plain_fit._hand_over
    proposals = []
    class Profile(trial.Trial):
        def hand_over(self, child):
            proposals.append(child)
            return member(child)
    profiler = Profile(problem.variables, True)
    profiler.points = points
    fit_args = []
    profiled = set()
    def fit_versions(eq, *args):
        fit_args[:] = args
        return trial.plain_versions(eq, *args)
    def handover(system, siblings):
        if len(profiled) < trial.LIMIT and fit_args:
            for parent in (system, *siblings):
                if not all(plain_fit._fitted(eq) and max(eq.fitness_value,eq.coefficients_stability)<LOSS_NAN_VAL
                           for eq in parent.vals):
                    continue
                combo = member(parent)
                for point,eq in zip(combo,parent.vals):
                    if len(profiled) >= trial.LIMIT:
                        break
                    if any(trial.trig_terms(eq.target)):
                        continue
                    tones = {(axis,float(f.params[i])) for term in eq.structure for f,i,axis in trial.trig_terms(term)}
                    if len(tones)!=1:
                        continue
                    tone = next(iter(tones))
                    structure = (eq.main_var_to_explain, eq.structure_label, eq.target.factors_labels)
                    if tone not in profiler.ranges or structure in profiled:
                        continue
                    profiled.add(structure)
                    profiler.profile((None,point,combo,tone,parent), fit_args)
        extra = list(proposals)
        proposals.clear()
        native_handover(system, [*siblings,*extra])
    options = dict(use_solver=False, device='cpu', multiobjective_mode=True, engine='nsga2',
                   system_rps_mode='plain', system_rps_params={'variable_order_seed':seed},
                   evaluation_form='weak', discrepancy_metric='wape', instability_metric='dyad',
                   multisample_mode='joint', multisample_weighting='samples', multisample_discrepancy='per_record',
                   population_size=8, training_epochs=budget, early_stopping_patience=None,
                   default_preprocessor_type='FD', preprocessor_kwargs={'accuracy':4}, max_deriv_order=(2,),
                   data_fun_pow=3, deriv_fun_pow=2, equation_terms_max_number=7,
                   equation_factors_max_number={'factors_num':[1,2],'probas':[.65,.35]},
                   verbose_params={'show_iter_idx':True,'candidate_objectives':False})
    started = monotonic()
    output_root(policy)
    if policy == 'systematic':
        options['system_rps_params']['fill_probability'] = 0.
    with ExitStack() as stack:
        stack.enter_context(patch.object(NSGAOptimizer,'_archive_insert',track))
        if policy=='first12':
            stack.enter_context(patch.object(plain_fit,'_versions',fit_versions))
            stack.enter_context(patch.object(plain_fit,'_hand_over',handover))
        else:
            from support_frequency import SupportProfiles
            refiner = SupportProfiles(profiler.ranges, point, multistart=policy == 'systematic')
            def fit_supports(eq,*args):
                return refiner.refine(trial.plain_versions(eq,*args),args)
            native_args = NSGAOptimizer.form_processer_args
            def epoch_args(optimizer,*args,**kwargs):
                refiner.epoch = 0 if optimizer._mu is None else len(optimizer._pareto_history)+1
                return native_args(optimizer,*args,**kwargs)
            if policy == 'systematic':
                from support_proposals import SupportProposals
                schedule = SupportProposals(trial.plain_versions, refiner)
                apply = partial(schedule.apply, plain_fit.apply_plain)
                stack.enter_context(patch.object(plain_fit, 'apply_plain', apply))
            else:
                stack.enter_context(patch.object(plain_fit,'_versions',fit_supports))
            stack.enter_context(patch.object(NSGAOptimizer,'form_processer_args',epoch_args))
        model = stack.enter_context(epde.EpdeMultisample(**options))
        family = profiler.family(None,live.R.axes_of(problem),None,options)
        if policy!='first12':
            family.token_family.status['meaningful'] = True
        model.fit(samples=samples, additional_tokens=[family,
                                                      live.U.GridTokens(labels=['t'],max_power=2)])
        reported = {member(system) for system in model.optimizer.reported_front()}
        completed = len(model.pareto_history)
        working_population = model.optimizer._mu
        if working_population != options['population_size']:
            raise RuntimeError(f'Unexpected working population: {working_population}')
        if completed != budget or not systems:
            raise RuntimeError(f'Incomplete search: {completed}/{budget} epochs, {len(systems)} feasible systems')
        profiles = profiler.profiles if policy=='first12' else refiner.profiles
        if policy=='first12' and budget>1 and not profiles:
            raise RuntimeError('The historical pilot evaluated no shared-frequency profiles')
    ordered = sorted(systems)
    for profile in profiler.profiles:
        if tuple(profile['parent']) not in systems or any(tuple(child) not in systems for child in profile['children']):
            raise RuntimeError('A frequency profile references a system outside the evaluated population and archive')
    combos = np.array(ordered)
    values = np.column_stack([np.array([p[5][i] for p in points])[combos] for i in range(len(samples))]
                             + [np.array([p[4] for p in points])[combos]])
    front = np.flatnonzero(system_front(values))
    if {ordered[i] for i in front} != reported:
        raise AssertionError('Full-vector archive differs from independent nondominated sorting')
    data = dict(dataset=name, arm=arm, draw=draw, search_seed=seed, samples=metadata, options=options,
                epochs=completed, working_population=working_population,
                variables=list(problem.variables), labels=observer.labels, points=points,
                systems=ordered, front=front.tolist(), frequency_profiles=profiles,frequency_policy=policy,
                profile_cache_hits=0 if policy=='first12' else refiner.cache_hits,
                seconds=monotonic()-started)
    if policy == 'systematic':
        data['structure_proposals'] = schedule.counts
    pending = destination.with_suffix('.pending')
    with gzip.open(pending,'wt',encoding='utf8',compresslevel=3) as f:
        json.dump(data,f,allow_nan=False,separators=(',',':'))
    pending.replace(destination)


def make_job(name,arm,draw,smoke=False,policy='systematic'):
    from cpu_budget import await_capacity
    from noise_suite import epochs
    folder = output_root(policy)/('smoke' if smoke else 'results')/name/arm/str(draw)
    folder.mkdir(parents=True,exist_ok=True)
    with FileLock(folder/'worker.lock'):
        archive = folder/'archive.json.gz'
        if archive.is_file():
            with gzip.open(archive,'rt',encoding='utf8') as f:
                data=json.load(f)
            if ((data['dataset'],data['arm'],data['draw'],data['epochs']) != (
                    name,arm,draw,1 if smoke else epochs(name)) or
                    data.get('frequency_policy','first12')!=policy or
                    (not smoke and ((policy=='first12' and not data['frequency_profiles']) or data.get('working_population') != 8))):
                raise ValueError(f'Pilot archive metadata mismatch at {archive}')
        else:
            await_capacity()
            search(name,arm,draw,1 if smoke else epochs(name),archive,policy=policy)
        receipt = folder/'sha256.txt'
        with archive.open('rb') as f:
            digest = hashlib.file_digest(f,'sha256').hexdigest()
        if receipt.exists() and receipt.read_text().strip() != digest:
            raise ValueError(f'Pilot archive checksum mismatch at {archive}')
        temp = receipt.with_suffix('.pending')
        temp.write_text(digest+'\n',encoding='ascii')
        temp.replace(receipt)


def launch(job):
    name, arm, draw, policy = job
    folder = output_root(policy)/'results'/name/arm/str(draw)
    folder.mkdir(parents=True,exist_ok=True)
    with (folder/'worker.log').open('a', encoding='utf8') as log:
        subprocess.run([sys.executable,'-u',__file__,'--worker',name,arm,str(draw),'--frequency-policy',policy],
                       env=dict(os.environ,PYTHONHASHSEED='0'),stdout=log,stderr=subprocess.STDOUT,check=True)
    print('Verified',name,arm,draw,flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker',nargs=3,metavar=('DATASET','ARM','DRAW'))
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--frequency-policy',choices=POLICIES,default='systematic')
    parser.add_argument('--recheck-misses',action='store_true',help='Rerun the oscillator and Duffing misses from the preceding policy')
    args = parser.parse_args()
    limit_threads()
    if args.worker:
        name,arm,draw = args.worker
        if name not in DATASETS or arm not in ARMS or int(draw) not in DRAWS:
            parser.error('Unknown pilot job')
        make_job(name,arm,int(draw),args.smoke,args.frequency_policy)
    elif args.recheck_misses:
        if args.frequency_policy == 'first12':
            parser.error('--recheck-misses requires a preceding pilot')
        previous, metric = ('per_support', 'front_hit') if args.frequency_policy == 'systematic' else ('first12', 'generated')
        with (output_root(previous)/'runs.csv').open(encoding='utf8') as f:
            jobs = [(r['dataset'],r['arm'],int(r['draw']),args.frequency_policy) for r in csv.DictReader(f)
                    if r['dataset'] in ('ode','duffing') and not int(r[metric])]
        with FileLock(ROOT/'campaign.lock',timeout=0), ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(launch,jobs))
            subprocess.run([sys.executable,str(ROOT/'report.py'),'--partial','--frequency-policy',args.frequency_policy],check=True)
    else:
        with FileLock(ROOT/'campaign.lock',timeout=0), ThreadPoolExecutor(max_workers=2) as pool:
            for d in DRAWS:
                list(pool.map(launch,[(n,a,d,args.frequency_policy) for a in ARMS for n in DATASETS]))
                flags = ['--partial'] if d != DRAWS[-1] else []
                subprocess.run([sys.executable,str(ROOT/'report.py'),'--frequency-policy',args.frequency_policy,*flags],check=True)
