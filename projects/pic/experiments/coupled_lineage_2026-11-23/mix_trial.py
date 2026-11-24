"""Use the archived donor-block sampler with the live weak-form fitness host."""
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

from trace_lineage import ROOT, Timeline, live, plain_fit
from epde.optimizers.nsga import per_variable
from epde.structure.weak_rows import WeakRows
from swap_trial import folder as native_folder

CONFIRMATION = range(923701,923713)
def folder(name,seed,arm):
    if arm=='native':
        return (native_folder(name,seed,'native') if seed in live.SEEDS else
                ROOT.parent/'candidate_frequency_2026-11-23'/'results'/name/str(seed)/'native')
    return ROOT/'mix_results'/name/str(seed)


class Trial:
    def __init__(self):
        sys.path.append(str(live.operators.HISTORICAL))
        import arm_pvmx
        self.mix = arm_pvmx
        self.hashes = {}
        for mod,expected in [(arm_pvmx,'1977428f5e96aa43241c2ddf463d6b9654a3992e76dcba9ed9a45ce99e38190b'),
                             (arm_pvmx.PC,'a471c034f715adb7f9bee39cb20de5b35d9cb1516914b1e7d6b9f01c2db316a7'),
                             (arm_pvmx.G,'9b2fc7c4261485f9e1afffd78c3e944b956154df19c6926f7defebb7c5fdc0b7')]:
            digest = hashlib.sha256(Path(mod.__file__).read_bytes()).hexdigest()
            assert digest==expected, mod.__file__
            self.hashes[mod.__name__] = digest
        self.cache = {}
        self.work = Counter()
        self.events = []
        self.labels = live.Observer(())
        self.seconds = 0.
        self.versions = plain_fit._versions
        self.apply = live.SolverFreeFitness.apply

    def observe(self, variables):
        self.timeline = Timeline(variables)
        return self.timeline

    def key(self, slot, target, terms):
        return slot,target,tuple(t.factors_labels for t in terms)

    def summaries(self, variants):
        return [dict(valid=True,wape=float(v.fitness_value),instability=float(v.coefficients_stability),
                     with_constant=bool(v.weights_final[-1])) for v in variants]

    def capture(self, eq, *args):
        self.fit_args = args
        output = self.versions(eq, *args)
        terms = [t for j,t in enumerate(eq.structure) if j!=eq.target_idx]
        self.cache[self.key(eq.main_var_to_explain,eq.target.factors_labels,terms)] = 'native',self.summaries(output)
        return output

    def exact(self, ctx, X, terms, labels, kind):
        key = self.key(ctx['var'],ctx['target_label'],terms)
        if key in self.cache:
            src, rows = self.cache[key]
            self.work[src+'_cache_hits'] += 1
            return rows, src
        candidate = deepcopy(self.equation)
        candidate.structure = [deepcopy(term) for term in [*terms, ctx['target_term']]]
        for term in candidate.structure:
            term.pool = candidate.pool
        candidate.reset_state(True)
        plain_fit._install_target(candidate,candidate.structure[-1])
        cap, coeff, coeff_args, host, host_args = self.fit_args
        trial = self
        class AuxiliaryHost:
            objectives = host.objectives
            def apply(self, variant, args):
                trial.apply(host,variant,args)
                trial.timeline.extra += 1
        rows = self.summaries(self.versions(candidate,cap,coeff,coeff_args,AuxiliaryHost(),host_args))
        self.cache[key] = 'proposal', rows
        self.work[kind+'_fresh_sets'] += 1
        return rows,'fresh'

    def parent(self, A, e, eq, ctx, target):
        terms = [t for j,t in enumerate(eq.structure) if j!=eq.target_idx]
        key = self.key(eq.main_var_to_explain,target,terms)
        point = self.mix._parent_point(A,e)
        current = []
        if live.np.isfinite(point).all():
            current.append(dict(valid=True,wape=float(point[0]),instability=float(point[1]),with_constant=None))
        if key in self.cache:
            self.work['parent_cache_hits'] += 1
            current += self.cache[key][1]
        if not current:
            current,_ = self.exact(ctx,None,terms,key[-1],'parent')
        return current

    def run(self, arm, seed, part, *, experiment):
        t = self.timeline
        mix = self.mix
        original_mix = mix.mix_equation
        native_column = WeakRows.columns
        def counted(rows, term, slot):
            hit = (slot, term.factors_labels) in rows._columns
            self.work['weak_hits' if hit else 'weak_columns'] += 1
            return native_column(rows,term,slot)
        def equation(A,child,e,k,pools,cap):
            self.equation = child.vals[k]
            row = dict(slot=k, parent=[self.labels.term(q) for q in A.vals[k].structure],
                       full_call=t.calls,start=t.cost)
            answer = original_mix(A,child,e,k,pools,cap)
            row.update(outcome=answer,child=[self.labels.term(q) for q in child.vals[k].structure],end=t.cost)
            self.events.append(row)
            return answer
        def cross(operator, objective, plan, coins, subop_args, duplicates=None):
            start = monotonic()
            for s in objective.population:
                s.reset_counter()
            objective.forced_pair = None
            keys = list(objective.population[0].vals.equation_keys)
            pool = mix.donor_pools(objective.population,keys)
            children = []
            for j,((a,ka,b,kb),coin) in enumerate(zip(plan,coins,strict=True)):
                first = per_variable.assemble(a,ka,b,kb,keys,coin,duplicates[j] if duplicates is not None else False)
                with patch.object(WeakRows,'columns',counted):
                    second = mix.optimal_mixing(a,keys,pool)
                if mix.ERRORS:
                    raise RuntimeError(mix.ERRORS[-1])
                if second.__dict__.pop('_pv_skip_mutation',False):
                    second.__dict__[per_variable.SKIP_FIRST_MUTATION] = True
                children.extend((first,second))
            objective.unplaced_candidates = children
            self.seconds += monotonic()-start
        with ExitStack() as stack:
            stack.enter_context(patch.object(plain_fit,'_versions',self.capture))
            stack.enter_context(patch.object(per_variable,'crossover',cross))
            stack.enter_context(patch.object(mix.arm_pv,'plane_table',per_variable.plane_table))
            stack.enter_context(patch.object(mix,'mix_equation',equation))
            stack.enter_context(patch.object(mix,'parent_versions',self.parent))
            stack.enter_context(patch.object(mix.PC,'set_versions',self.exact))
            stack.enter_context(patch.object(mix.PC,'predict',lambda *a: None))
            experiment()
        assert self.events and sum(mix.DECISIONS.values())==len(self.events)
        return dict(arm=arm,events=self.events,trace_labels=self.labels.labels,hashes=self.hashes,
                    work=dict(self.work,**mix.COUNTS),decisions=mix.DECISIONS,mix_seconds=self.seconds,
                    gain_columns=0,extra_fits=t.extra,full_cost=t.full_cost,
                    system_cost=[t.system_cost[c] for c in sorted(t.systems)])


def launch(job):
    name,seed = job
    dest = folder(name,seed,'mix')
    if (dest/'archive.json.gz').exists():
        return
    dest.mkdir(parents=True,exist_ok=True)
    with (dest/'search.log').open('w') as log:
        r = subprocess.run([sys.executable,'-u',__file__,'--worker',name,str(seed)],
                           env=dict(os.environ,PYTHONHASHSEED='0'),stdout=log,stderr=subprocess.STDOUT)
    if r.returncode:
        raise RuntimeError((dest/'search.log').read_text(errors='replace')[-2600:])
    print(*job,flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--worker',nargs=2)
    p.add_argument('--confirmation',action='store_true')
    args = p.parse_args()
    if args.worker:
        name,n = args.worker
        trial = Trial()
        live.run(name,int(n),'mix',observer_factory=trial.observe,operator_run=trial.run,destination=folder(name,int(n),'mix'))
    else:
        datasets = [s for s in live.DATASETS if len(live.load_data(s).variables)>1]
        seeds = CONFIRMATION if args.confirmation else live.SEEDS
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(launch, [(s,n) for n in seeds for s in datasets]))
