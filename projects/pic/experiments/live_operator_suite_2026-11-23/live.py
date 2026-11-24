"""Run matched native and historical fit-aware searches with a fixed fitness-call budget."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import replace
import gzip
import json
import os
from pathlib import Path
import random
import subprocess
import sys
from time import monotonic
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
sys.path.insert(0, str(ROOT.parent/'fit_aware_add_2026-11-23'))
import operator_trial as operators
from search import U, np
sys.path.insert(0, str(ROOT.parent/'general_weak_holdout_2026-11-23'))
from screen import R, SYSTEMS, split, training_plan
from epde.operators.common.survival import configured_width
from epde.operators.common import plain_fit
from epde.operators.common.fitness import SolverFreeFitness
from epde.operators.common.objectives import LOSS_NAN_VAL
from epde.operators.common.sparsity import VWSRSparsity
from epde_bench.problem import Problem

SEEDS = range(923301, 923304)
BUDGET = 3000
DATASETS = (*SYSTEMS, 'ode_simple')
ARMS = ('native', 'pvra')


def load_data(name, **loader_options):
    if name == 'ode_simple':
        axes, fields = U.load('ode')
        return Problem(name, name, 'ode', 'synthetic', tuple(np.meshgrid(*axes, indexing='ij')),
                       fields, ('t',), truth=['-1 * u{power: 1.0} = d^2u/dx0^2{power: 1.0}'])
    problem = R.datasets.load(name, **loader_options)
    axes = R.axes_of(problem)
    corrected = False
    for i,a in enumerate(axes):
        if not np.allclose(np.diff(a),np.diff(a)[0],rtol=1e-6,atol=0.):
            uniform = np.linspace(a[0],a[-1],len(a))
            if not np.array_equal(uniform.astype(np.float32).astype(float),a):
                raise ValueError('The coordinates cannot be recovered as a float32-rounded uniform grid.')
            axes[i] = uniform
            corrected = True
    return replace(problem,grids=tuple(np.meshgrid(*axes,indexing='ij'))) if corrected else problem


def folder(name, seed, arm):
    return ROOT/'results'/name/str(seed)/arm


class BudgetReached(BaseException):
    pass


class Observer:
    def __init__(self, variables):
        self.variables = variables
        self.labels = [()]
        self.label_indices = {(): 0}
        self.term_cache = {}
        self.point_ids = {}
        self.points = []
        self.systems = set()
        self.calls = 0
        self.system_events = 0

    def term(self, term):
        token = term.factors_labels
        if token not in self.term_cache:
            factors = []
            for f in term.structure:
                pars = tuple(sorted((v['name'], float(f.params[i])) for i,v in f.params_description.items()))
                factors.append(('x' if f.ftype == 'grids' else f.label, pars))
            label = tuple(sorted(factors))
            if label not in self.label_indices:
                self.label_indices[label] = len(self.labels)
                self.labels.append(label)
            self.term_cache[token] = self.label_indices[label]
        return self.term_cache[token]

    def key(self, eq):
        t = eq.target_idx
        pivot = self.term(eq.structure[t])
        terms = [(pivot, -1.)]
        for i,term in enumerate(eq.structure):
            if i != t:
                coeff = float(eq.weights_final[eq.weight_index(i,t)])
                if coeff:
                    terms.append((self.term(term),coeff))
        if eq.weights_final[-1]:
            terms.append((0,float(eq.weights_final[-1])))
        return (eq.main_var_to_explain, pivot, tuple(sorted(terms)), float(eq.fitness_value),
                float(eq.coefficients_stability))

    def evaluated(self, eq):
        self.calls += 1
        key = self.key(eq)
        if key not in self.point_ids:
            self.point_ids[key] = len(self.points)
            self.points.append((*key,self.calls))

    def hand_over(self, system):
        equations = [system.vals[v] for v in self.variables]
        if all(plain_fit._fitted(eq) and max(eq.fitness_value,eq.coefficients_stability)<LOSS_NAN_VAL
               for eq in equations):
            keys = [self.key(eq) for eq in equations]
            combo = tuple(self.point_ids[k] for k in keys)
            self.systems.add(combo)
            self.system_events += 1
            return combo


def run(name, seed, arm, *, observer_factory=Observer, operator_run=operators.run, destination=None,
        trig_factory=None, noise_level=5, epoch_budget=None, loader_options=None, observation=None):
    assert os.environ['PYTHONHASHSEED'] == '0'
    started = monotonic()
    problem = load_data(name, **(loader_options or {}))
    meta = {}
    if observation is None:
        raw = problem.noisy(noise_level, seed)
    else:
        raw, meta['measurement'] = observation(problem, noise_level, seed)
    axis,(cuts, axes),(_,_) = split(R.axes_of(problem))
    train = {k: v[cuts] for k,v in raw.items()}
    dims = len(axes)
    settings = dict(use_solver=False, device='cpu', multiobjective_mode=True, system_rps_mode='plain',
                    engine='nsga2', evaluation_form='weak', discrepancy_metric='wape', instability_metric='dyad',
                    system_rps_params={'variable_order_seed': seed},
                    training_epochs=100 if epoch_budget is None else epoch_budget, early_stopping_patience=None,
                    default_preprocessor_type='FD', preprocessor_kwargs={'accuracy':4}, max_deriv_order=(2,)+(4,)*(dims-1),
                    data_fun_pow=3, deriv_fun_pow=2, equation_terms_max_number=7, memory_for_cache=2,
                    equation_factors_max_number={'factors_num':[1,2], 'probas':[.65,.35]},
                    verbose_params={'show_iter_idx':epoch_budget is not None,'candidate_objectives':False})
    observer = observer_factory(problem.variables)
    apply, hand_over = SolverFreeFitness.apply, plain_fit._hand_over
    def record(host, eq, *args, **kwargs):
        if epoch_budget is None and observer.calls >= BUDGET:
            raise BudgetReached()
        result = apply(host,eq,*args,**kwargs)
        observer.evaluated(eq)
        return result
    def completed(system, siblings):
        for s in (system,*siblings):
            observer.hand_over(s)
        return hand_over(system,siblings)
    def build(with_trig):
        random.seed(seed)
        np.random.seed(seed)
        search = U.EpdeSearch(**settings)
        plan = training_plan(train, axes, configured_width(search._config.search_space))
        meta['plan'] = {k:list(getattr(plan,k)) for k in ('q','ms','ss','cut','shape')}
        meta['trig'] = with_trig
        grid = tuple(np.meshgrid(*axes,indexing='ij'))
        _, domain = search.createDomain(grid,ID=0)
        search.set_preprocessor(default_preprocessor_type='FD',preprocessor_kwargs={'accuracy':4})
        _, traj = search.createTrajectory(train,domain,cache_id=0)
        families = []
        if with_trig:
            family = (U.TrigonometricTokens(freq_from_data=2) if trig_factory is None else
                      trig_factory(train, axes, plan, settings))
            families.append(family)
        families.append(U.GridTokens(labels=['t','x','y'][:dims],max_power=2))
        if problem.extra_tokens:
            families.extend(problem.extra_tokens(problem))
        if epoch_budget is not None:
            history = []
            last_structures = None
            unchanged = 0
            def observe_front(snapshot, index):
                nonlocal last_structures, unchanged
                front = search.optimizer.reported_front()
                structures = frozenset(system.structure_labels for system in front)
                unchanged = unchanged + 1 if structures == last_structures else 0
                last_structures = structures
                members = [[observer.point_ids[observer.key(system.vals[v])] for v in observer.variables]
                           for system in front]
                history.append(dict(epoch=int(index) + 1, unchanged=unchanged, members=members))
                if unchanged == 5 and not any(row['unchanged'] >= 5 for row in history[:-1]):
                    print(f'Passive patience 5 first reached after epoch {int(index)+1}.', flush=True)
                return False
            search.optimizer_exec_params['early_stopping_callback'] = observe_front
        with patch.object(R.WF,'choose_window',return_value=plan):
            search.fit(data=[traj],additional_tokens=families)
        if epoch_budget is not None:
            epochs_completed = len(search.pareto_history)
            assert epochs_completed == epoch_budget == len(history), (epochs_completed, epoch_budget, len(history))
            native = search.optimizer.reported_front()
            meta.update(epochs_completed=epochs_completed, epoch_fronts=history, input_shape=list(problem.shape),
                        loader_options=loader_options or {}, budget_unit='epochs',
                        native_front=[[observer.point_ids[observer.key(s.vals[v])] for v in observer.variables]
                                      for s in native])
            assert set(map(tuple, history[-1]['members'])) == set(map(tuple, meta['native_front']))
    def experiment():
        with ExitStack() as patches:
            patches.enter_context(patch.object(SolverFreeFitness,'apply',record))
            patches.enter_context(patch.object(plain_fit,'_hand_over',completed))
            patches.enter_context(patch.object(VWSRSparsity,'apply',side_effect=AssertionError('Unexpected VWSR')))
            try:
                try:
                    build(True)
                except ValueError as err:
                    if 'freq_from_data' not in str(err) or observer.calls:
                        raise
                    build(False)
            except BudgetReached:
                pass
    telemetry = operator_run(arm,seed,'longest_axis',experiment=experiment)
    if epoch_budget is None:
        assert observer.calls == BUDGET
    rec = dict(system=name, seed=seed, arm=arm, variables=problem.variables, axis=axis, noise_percent=noise_level, **meta,
               train_hashes={k:R.digest(v) for k,v in train.items()}, settings=settings, calls=observer.calls,
               labels=observer.labels, points=observer.points, systems=sorted(observer.systems),
               system_events=observer.system_events, operator=telemetry, seconds=monotonic()-started)
    dest = (destination or folder(name,seed,arm))/'archive.json.gz'
    tmp = dest.with_suffix('.pending')
    with gzip.open(tmp,'wt',encoding='utf8',compresslevel=3) as stream:
        json.dump(rec,stream,allow_nan=False,separators=(',',':'))
    tmp.replace(dest)


def launch(job):
    name,seed,arm = job
    dest = folder(*job)
    dest.mkdir(parents=True,exist_ok=True)
    if (dest/'archive.json.gz').exists():
        return job, 0
    with (dest/'search.log').open('w') as stream:
        status = subprocess.run([sys.executable,'-u',__file__,name,str(seed),arm],stdout=stream,
                                stderr=subprocess.STDOUT,env=dict(os.environ,PYTHONHASHSEED='0')).returncode
    print(*job,status,flush=True)
    return job,status


if __name__ == '__main__':
    if len(sys.argv) == 4:
        run(sys.argv[1],int(sys.argv[2]),sys.argv[3])
    else:
        jobs = [(s,n,a) for n in SEEDS for s in DATASETS for a in ARMS]
        with ThreadPoolExecutor(max_workers=6) as pool:
            statuses = list(pool.map(launch,jobs))
        raise SystemExit(int(any(code for _,code in statuses)))
