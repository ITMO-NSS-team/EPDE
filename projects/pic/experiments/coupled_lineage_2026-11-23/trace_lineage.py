"""Record live mating pools, selected native parents, crossed children and fitted descendants."""
import argparse
from contextlib import ExitStack
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import os
import subprocess
import sys
from unittest.mock import patch
import weakref

ROOT = Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'guided_swap_2026-11-23'))
from swap_trial import Timeline, live
from epde.optimizers.nsga import per_variable
from live import plain_fit


def folder(name, seed):
    return ROOT/'results'/name/str(seed)


class Trace(Timeline):
    def __init__(self, variables):
        super().__init__(variables)
        self.raw = live.Observer(variables)
        self.generations = []
        self.pairs = []
        self.descendants = {}
        self.active = None

    def identifiers(self, system):
        ids = []
        for slot in self.variables:
            eq = system.vals[slot]
            if (plain_fit._fitted(eq) and
                    max(eq.fitness_value,eq.coefficients_stability) < live.LOSS_NAN_VAL):
                ids.append(self.point_ids[self.key(eq)])
            else:
                ids.append(None)
        return ids

    def structures(self, system):
        return [[self.raw.term(t) for t in system.vals[v].structure] for v in self.variables]

    def hand_over(self, system):
        super().hand_over(system)
        if self.active is not None:
            self.active['outcomes'].append(dict(at=self.calls, points=self.identifiers(system)))

    def run(self, arm, seed, part, *, experiment):
        native_crossover = per_variable.crossover
        native_fit = plain_fit.apply_plain

        def cross(host, objective, plan, coins, subop_args, duplicates=None):
            self.generations.append(dict(at=self.calls, population=[self.identifiers(s) for s in objective.population]))
            native_crossover(host, objective, plan, coins, subop_args, duplicates)
            children = objective.unplaced_candidates
            assert len(children) == 2*len(plan)
            for i,(a,ka,b,kb) in enumerate(plan):
                event = dict(generation=len(self.generations)-1, at=self.calls,
                             parents=[self.identifiers(a),self.identifiers(b)],
                             slots=[ka,kb], from_b=sorted(coins[i]), children=[])
                self.pairs.append(event)
                for child,kind in zip(children[2*i:2*i+2], ('assembled','crossed')):
                    row = dict(kind=kind, structure=self.structures(child), fits=[])
                    event['children'].append(row)
                    self.descendants[id(child)] = weakref.ref(child), row

        def fit(objective, eq_selector, eq_args):
            entry = self.descendants.get(id(objective))
            if entry is not None and entry[0]() is objective:
                row = dict(at=self.calls, structure=self.structures(objective), outcomes=[])
                entry[1]['fits'].append(row)
                self.active = row
            try:
                return native_fit(objective, eq_selector, eq_args)
            finally:
                self.active = None

        with ExitStack() as patches:
            patches.enter_context(patch.object(per_variable, 'crossover', cross))
            patches.enter_context(patch.object(plain_fit, 'apply_plain', fit))
            experiment()
        assert self.generations and self.pairs and any(c['fits'] for e in self.pairs for c in e['children'])
        return dict(generations=self.generations, pairs=self.pairs, structure_labels=self.raw.labels,
                    full_cost=self.full_cost, system_cost=[self.system_cost[c] for c in sorted(self.systems)])


def worker(name, seed):
    trace = None
    def observe(variables):
        nonlocal trace
        trace = Trace(variables)
        return trace
    live.run(name, seed, 'native', observer_factory=observe,
             operator_run=lambda *a,**kw: trace.run(*a,**kw), destination=folder(name,seed))


def launch(job):
    name, seed = job
    dest = folder(*job)
    dest.mkdir(parents=True,exist_ok=True)
    if (dest/'archive.json.gz').exists():
        return
    with (dest/'search.log').open('w') as log:
        r = subprocess.run([sys.executable,'-u',__file__,'--worker',name,str(seed)],
                           env=dict(os.environ,PYTHONHASHSEED='0'), stdout=log, stderr=subprocess.STDOUT)
    if r.returncode:
        raise RuntimeError((dest/'search.log').read_text(errors='replace')[-2400:])
    print(*job,flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workers',type=int,default=2)
    p.add_argument('--worker',nargs=2)
    args = p.parse_args()
    if args.worker:
        worker(args.worker[0],int(args.worker[1]))
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(launch,[(s,n) for n in live.SEEDS for s in ('lorenz','ns')]))
