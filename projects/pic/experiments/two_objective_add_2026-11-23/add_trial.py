"""Run the archived two-stage addition sampler using the live EPDE objective host."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from copy import deepcopy
import hashlib
import os
from pathlib import Path
import subprocess
import sys
from time import monotonic
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent/'guided_swap_2026-11-23'))
from swap_trial import Timeline, WeakRows, fill_prune, live, np, plain_fit, plain_moves
from epde.structure.main_structures import Equation

ARMS = ('pvra','pvxa')
CONFIRMATION = range(923501,923513)


def folder(name, seed, arm):
    return ROOT/'results'/name/str(seed)/arm


class AddTrial:
    def __init__(self):
        sys.path.append(str(live.operators.HISTORICAL))
        import arm_pvguide
        self.guide = arm_pvguide
        self.hashes = {}
        for module,expected in [
            (arm_pvguide,'9b2fc7c4261485f9e1afffd78c3e944b956154df19c6926f7defebb7c5fdc0b7'),
            (arm_pvguide.arm_xa,'2820149627c27941bf270dcd0a77e5823a3bfa07a0f65b6a36a689da03f256a2'),
            (arm_pvguide.SC,'76f50438258986f9726aef52277164c9ad0515b8500cef32b065783e7e6a660f')]:
            digest = hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
            if digest != expected:
                raise ValueError(f'Archived operator changed: {module.__file__}')
            self.hashes[module.__name__] = digest
        self.labels = live.Observer(())
        self.cache = {}
        self.events = []
        self.work = Counter()
        self.seconds = 0.

    def observe(self, variables):
        self.timeline = Timeline(variables)
        return self.timeline

    def capture_native(self, eq, *args):
        self.fit_args = args
        versions = self.native_versions(eq, *args)
        key = (eq.main_var_to_explain, eq.target.factors_labels,
               tuple(t.factors_labels for j,t in enumerate(eq.structure) if j!=eq.target_idx))
        self.cache[key] = 'discovery', self.summarize_versions(versions)
        return versions

    def summarize_versions(self, versions):
        return [dict(valid=True, wape=float(v.fitness_value), instability=float(v.coefficients_stability),
                     constant=bool(v.weights_final[-1]), kept=[self.labels.term(t) for j,t in enumerate(v.structure)
                                                                if j!=v.target_idx]) for v in versions]

    def screen_versions(self, design, term, col, cap):
        features = [*design['terms'],term]
        key = (design['var'],design['target_label'],tuple(t.factors_labels for t in features))
        before = self.timeline.extra
        if key in self.cache:
            source, versions = self.cache[key]
            self.work[source+'_cache_hits'] += 1
        else:
            self.work['shortlist_fit_attempts'] += 1
            eq = deepcopy(self.current_equation)
            eq.structure = [deepcopy(t) for t in [*features,design['target_term']]]
            for t in eq.structure:
                t.pool = eq.pool
            eq._invalidate_label_cache()
            eq.reset_state(True)
            plain_fit._install_target(eq,eq.structure[-1])
            native_cap, coeff, coeff_args, host, host_args = self.fit_args
            assert cap==native_cap
            class AuxiliaryHost:
                objectives = host.objectives
                def apply(_, variant, arguments):
                    self.native_apply(host,variant,arguments)
                    self.timeline.extra += 1
            versions = self.summarize_versions(self.native_versions(eq,cap,coeff,coeff_args,AuxiliaryHost(),host_args))
            self.cache[key] = 'shortlist',versions
            source = 'fresh'
        self.event['scores'].append(dict(term=self.labels.term(term),source=source,versions=versions,
                                         fits=self.timeline.extra-before))
        # Native scores need no historical surrogate parity prediction.
        return versions, 'native'

    def run(self, arm, seed, part, *, experiment):
        guide, label = self.guide,self.labels.term
        self.native_versions = plain_fit._versions
        self.native_apply = live.SolverFreeFitness.apply
        original_stage1, original_front = guide.stage1, guide.arm_xa.eligible
        chooser = guide.ra_choose if arm=='pvra' else guide.xa_choose
        original_columns = WeakRows.columns
        def counted_columns(store, term, slot):
            self.work['weak_column_requests'] += 1
            self.work['weak_column_cache_hits'] += int((slot,term.factors_labels) in store._columns)
            return original_columns(store,term,slot)
        def stage1(eq, design, partners):
            result = original_stage1(eq, design, partners)
            candidates,gains,top,_ = result
            e = self.event
            e['screened'] = True
            e['shortlist'] = [[label(candidates[i][0]),float(gains[i])] for i in top]
            if top and arm=='pvra':
                e['probabilities'] = (gains[top]/gains[top].sum()).tolist()
            if arm=='pvxa':
                e['repeat'] = [(design['var'],design['target_label'],
                               frozenset([*design['labels'],candidates[i][0].factors_labels])) in guide.CHOSEN for i in top]
            return result
        def front(version_lists):
            eligible = original_front(version_lists)
            self.event['front'] = eligible
            weights = np.zeros(len(version_lists))
            for i in eligible:
                weights[i] = guide.TABU_WEIGHT if self.event['repeat'][i] else 1.
            if eligible:
                weights /= weights.sum()
            self.event['probabilities'] = weights.tolist()
            return eligible
        def select(eq, design, partners):
            chosen = chooser(eq,design,partners)
            if chosen is not None:
                self.event['path'] = 'guided'
            return chosen
        def instrumented():
            ordinary = Equation.add_random_term
            def traced(eq, *args, **kwargs):
                system = plain_moves._MUTATING['system']
                if system is None or system.vals[eq.main_var_to_explain] is not eq or args or kwargs:
                    return ordinary(eq,*args,**kwargs)
                started = monotonic()
                partners = frozenset(fill_prune.partner_targets(system,eq.main_var_to_explain))
                target,terms,_ = guide.fit_set(eq,partners)
                self.current_equation = eq
                e = self.event = dict(slot=eq.main_var_to_explain, after_call=self.timeline.calls,
                                      fit_cost=self.timeline.cost, parent=[label(t) for t in eq.structure],
                                      target=label(target) if target is not None else None,
                                      eligible_parent=[label(t) for t in terms] if target is not None else None,
                                      screened=False,shortlist=[],front=[],scores=[],probabilities=[],path='ordinary')
                changed = ordinary(eq)
                e.update(child=[label(t) for t in eq.structure], changed=bool(changed),
                         selected=label(eq.structure[-1]) if changed else None, after_fit_cost=self.timeline.cost)
                self.events.append(e)
                self.seconds += monotonic()-started
                return changed
            with ExitStack() as patches:
                patches.enter_context(patch.object(Equation,'add_random_term',traced))
                patches.enter_context(patch.object(guide,'stage1',stage1))
                patches.enter_context(patch.object(guide,'ra_choose',select))
                patches.enter_context(patch.object(WeakRows,'columns',counted_columns))
                if arm=='pvxa':
                    patches.enter_context(patch.object(plain_fit,'_versions',self.capture_native))
                    patches.enter_context(patch.object(guide,'screen_versions',self.screen_versions))
                    patches.enter_context(patch.object(guide.arm_xa,'eligible',front))
                experiment()
        info = live.operators.run('pvra',seed,part,experiment=instrumented)
        t = self.timeline
        assert len(self.events)==info['counters']['ordinary_add_opportunities']
        return dict(info,arm=arm,additions=self.events,trace_labels=self.labels.labels,add_seconds=self.seconds,
                    extra_fits=t.extra,full_cost=t.full_cost,system_cost=[t.system_cost[c] for c in sorted(t.systems)],
                    work=self.work,hashes=self.hashes)


def launch(job):
    name,seed,arm = job
    dest = folder(*job)
    dest.mkdir(parents=True,exist_ok=True)
    if (dest/'archive.json.gz').exists():
        return
    with (dest/'search.log').open('w') as log:
        result = subprocess.run([sys.executable,'-u',__file__,'--worker',name,str(seed),arm],
                                env=dict(os.environ,PYTHONHASHSEED='0'),stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:
        raise RuntimeError((dest/'search.log').read_text(errors='replace')[-3500:])
    print(*job,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--datasets',nargs='+',choices=live.DATASETS,default=live.DATASETS)
    p.add_argument('--seeds',nargs='+',type=int,default=live.SEEDS)
    p.add_argument('--workers',type=int,default=6)
    p.add_argument('--worker',nargs=3)
    p.add_argument('--confirmation',action='store_true')
    args=p.parse_args()
    if args.worker:
        name,seed,arm=args.worker
        trial=AddTrial()
        live.run(name,int(seed),arm,observer_factory=trial.observe,operator_run=trial.run,
                 destination=folder(name,int(seed),arm))
    else:
        seeds=CONFIRMATION if args.confirmation else args.seeds
        jobs=[(s,n,a) for n in seeds for s in args.datasets for a in ARMS]
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(launch,jobs))
