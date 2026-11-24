"""Trace ordinary swaps on the live suite. Each worker owns one trajectory and RNG stream."""
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
sys.path.insert(0, str(ROOT.parent / 'live_operator_suite_2026-11-23'))
import live
from live import np, plain_fit
from epde.operators.common import fill_prune, objectives
from epde.operators.multiobjective import plain_moves
from epde.structure.weak_rows import WeakRows

ARMS = ('native', 'guided_swap')
CONFIRMATION = range(923401, 923413)


def folder(name, seed, arm):
    return ROOT / 'results' / name / str(seed) / arm


class Timeline(live.Observer):
    def __init__(self, variables):
        super().__init__(variables)
        self.extra = 0
        self.full_cost = []
        self.system_cost = {}

    @property
    def cost(self):
        return self.calls + self.extra

    def evaluated(self, equation):
        super().evaluated(equation)
        self.full_cost.append(self.cost)

    def hand_over(self, system):
        combo = super().hand_over(system)
        if combo is not None:
            self.system_cost.setdefault(combo, self.cost)


class SwapTrial:
    def __init__(self):
        sys.path.append(str(live.operators.HISTORICAL))
        import arm_pvguide
        self.guide = arm_pvguide
        self.hashes = {}
        for name, expected in {
            'arm_pvls': '6ce6469c5595dd5dfbc2ae8677a6fc80f6c497cefae133f22ee4ea8c468f2da4',
            'arm_drop': 'e1dd6eddf53aee4a9a49079921198938b6c82fd0651ce1c66347e4fa8a227d85',
            'arm_pvguide': '9b2fc7c4261485f9e1afffd78c3e944b956154df19c6926f7defebb7c5fdc0b7',
            'setcond_core': '76f50438258986f9726aef52277164c9ad0515b8500cef32b065783e7e6a660f',
        }.items():
            path = live.operators.HISTORICAL / (name + '.py')
            self.hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
            if self.hashes[name] != expected:
                raise ValueError(f'Archived operator changed: {path}')
        self.labels = live.Observer(())
        self.events = []
        self.counts = Counter()
        self.seconds = 0.
        self.versions = {}
        self.components = {}
        self.fit_args = None
        self.last_scores = None

    def observe(self, variables):
        self.timeline = Timeline(variables)
        return self.timeline

    def support_key(self, slot, target, terms):
        return slot, self.labels.term(target), tuple(self.labels.term(t) for t in terms)

    def remember_fit(self, eq, *args):
        self.fit_args = args
        variants = self.native_versions(eq, *args)
        self.versions[self.support_key(eq.main_var_to_explain, eq.target,
                      [t for i,t in enumerate(eq.structure) if i != eq.target_idx])] = [
            (tuple(self.labels.term(t) for i,t in enumerate(v.structure) if i != v.target_idx),
             bool(v.weights_final[-1]), float(v.fitness_value), float(v.coefficients_stability))
            for v in variants]
        return variants

    def capture_scores(self, *args, **kwargs):
        self.last_scores = self.native_dyad(*args, **kwargs)
        return self.last_scores

    def remember_scores(self, objective, eq, ctx):
        self.last_scores = None
        value = self.native_compute(objective, eq, ctx)
        if self.last_scores is not None:
            terms = [t for i,t in enumerate(eq.structure) if i != eq.target_idx]
            key = self.support_key(eq.main_var_to_explain, eq.target, terms)
            constant = bool(eq.weights_internal[-1])
            assert len(self.last_scores) == len(terms) + int(constant)
            self.components[key, constant] = tuple(map(float, self.last_scores[:len(terms)]))
        return value

    def parent_scores(self, eq, design):
        terms, target = design['terms'], design['target_term']
        key = self.support_key(eq.main_var_to_explain, target, terms)
        fitted = (not design['stripped'] and plain_fit._fitted(eq) and eq.target is target)
        if fitted:
            constant = bool(eq.weights_final[-1])
            if (key, constant) in self.components:
                self.counts['fitted_cache_hits'] += 1
                return self.components[key, constant], constant
        if key in self.versions:
            self.counts['version_cache_hits'] += 1
        else:
            # Score only this parent on the native weak rows. A private fitness host
            # keeps proposal fits out of the discovery archive and full-call budget.
            cap, coeff, coeff_args, host, host_args = self.fit_args
            class AuxiliaryHost:
                objectives = host.objectives
                def apply(_, variant, arguments):
                    self.original_apply(host, variant, arguments)
                    self.timeline.extra += 1
            work = deepcopy(eq)
            indices = [next(i for i,t in enumerate(eq.structure) if t is term) for term in [*terms,target]]
            work.structure = [work.structure[i] for i in indices]
            work._invalidate_label_cache()
            work.reset_state(True)
            plain_fit._install_target(work, work.structure[-1])
            self.counts['parent_prefits'] += 1
            saved_args = self.fit_args
            try:
                self.remember_fit(work, cap, coeff, coeff_args, AuxiliaryHost(), host_args)
            finally:
                self.fit_args = saved_args
        versions = self.versions[key]
        if not versions:
            return None, None
        version = next((v for v in versions if fitted and v[1] == bool(eq.weights_final[-1])), versions[0])
        kept, constant = version[:2]
        scores = self.components[(key[0], key[1], kept), constant]
        by_term = dict(zip(kept, scores))
        return [by_term.get(i, 0.) for i in key[2]], constant

    def run(self, arm, seed, part, *, experiment):
        guide = self.guide
        timeline = self.timeline
        native_swap = plain_moves._swap
        self.native_versions = plain_fit._versions
        self.original_apply = live.SolverFreeFitness.apply
        self.native_dyad = objectives._BASIS_FREE_METRICS['dyad']
        self.native_compute = objectives.Instability.compute
        column = WeakRows.columns
        def counted_column(store, term, slot):
            self.counts['weak_column_requests'] += 1
            self.counts['weak_column_cache_hits'] += int((slot, term.factors_labels) in store._columns)
            return column(store, term, slot)

        def swap(eq, random_victim):
            started = monotonic()
            label = self.labels.term
            system = plain_moves._MUTATING['system']
            partners = frozenset(fill_prune.partner_targets(system, eq.main_var_to_explain))
            target, terms, stripped = guide.fit_set(eq, partners)
            event = dict(after_call=timeline.calls, fit_cost=timeline.cost, slot=eq.main_var_to_explain,
                         parent=[label(t) for t in eq.structure], target=label(target) if target is not None else None,
                         eligible_parent=[label(t) for t in terms] if target is not None else None,
                         shortlist=[], selected=None, constant=None)
            victim = random_victim
            choice = None
            path = 'native'
            if arm == 'guided_swap':
                if target is None or not terms:
                    path = terms if target is None else 'no_features'
                else:
                    design, reason = guide.child_design(eq, partners)
                    assert reason is None
                    scores, event['constant'] = self.parent_scores(eq, design)
                    if scores is None:
                        path = 'no_valid_fit'
                    else:
                        probs = guide.arm_drop.victim_probabilities(scores)
                        k = int(np.random.choice(len(terms), p=probs))
                        victim = terms[k]
                        event['victim_probabilities'] = probs.tolist()
                        rest = [j for j in range(len(terms)) if j != k]
                        reduced = dict(design, terms=[terms[j] for j in rest], labels=[design['labels'][j] for j in rest],
                                       X=design['X'][:, rest])
                        candidates, gains, top, _ = guide.stage1(eq, reduced, partners)
                        event['admissible'] = len(candidates)
                        event['shortlist'] = [(label(candidates[j][0]), float(gains[j])) for j in top]
                        if top:
                            weights = gains[top]
                            choice = candidates[int(np.random.choice(top, p=weights / weights.sum()))][0]
                            path = 'guided'
                        else:
                            path = 'empty_shortlist'
            event.update(path=path, outgoing=label(victim))
            index = next(i for i,t in enumerate(eq.structure) if t is victim)
            if choice is not None:
                new = deepcopy(choice)
                new.pool = eq.pool
                new.resetSavedState()
                eq.structure[index] = new
                eq._invalidate_label_cache()
                changed = True
            elif path == 'empty_shortlist':
                changed = False  # PVLS proposes no child when no gain is positive.
            else:
                changed = native_swap(eq, victim)
            event.update(changed=changed, selected=label(eq.structure[index]) if changed else None,
                         child=[label(t) for t in eq.structure], after_fit_cost=timeline.cost)
            self.events.append(event)
            self.seconds += monotonic() - started
            return changed

        with ExitStack() as stack:
            stack.enter_context(patch.object(plain_moves, '_swap', swap))
            stack.enter_context(patch.object(WeakRows, 'columns', counted_column))
            if arm == 'guided_swap':
                stack.enter_context(patch.object(plain_fit, '_versions', self.remember_fit))
                stack.enter_context(patch.dict(objectives._BASIS_FREE_METRICS, dyad=self.capture_scores))
                def compute(obj, eq, ctx):
                    return self.remember_scores(obj, eq, ctx)
                stack.enter_context(patch.object(objectives.Instability, 'compute', compute))
            info = live.operators.run('native', seed, part, experiment=experiment)
        return dict(info, arm=arm, trace_labels=self.labels.labels, swaps=self.events, work=self.counts,
                    extra_fits=timeline.extra, full_cost=timeline.full_cost,
                    system_cost=[timeline.system_cost[c] for c in sorted(timeline.systems)],
                    swap_seconds=self.seconds, hashes=self.hashes)


def worker(name, seed, arm):
    dest = folder(name, seed, arm)
    dest.mkdir(parents=True, exist_ok=True)
    trial = SwapTrial()
    live.run(name, seed, arm, observer_factory=trial.observe, operator_run=trial.run, destination=dest)


def launch(job):
    name, seed, arm = job
    dest = folder(*job)
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / 'archive.json.gz').exists():
        return 0
    with (dest / 'search.log').open('w') as log:
        result = subprocess.run([sys.executable, '-u', __file__, '--worker', name, str(seed), arm],
                                env=dict(os.environ, PYTHONHASHSEED='0'), stdout=log, stderr=subprocess.STDOUT)
    print(*job, result.returncode, flush=True)
    if result.returncode:
        raise RuntimeError((dest / 'search.log').read_text(errors='replace')[-3500:])
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', nargs=3, metavar=('SYSTEM', 'SEED', 'ARM'))
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--datasets', nargs='+', choices=live.DATASETS, default=live.DATASETS)
    parser.add_argument('--arms', nargs='+', choices=ARMS, default=ARMS)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--seeds', nargs='+', type=int, default=live.SEEDS)
    group.add_argument('--confirmation', action='store_true')
    args = parser.parse_args()
    if args.worker:
        name, seed, arm = args.worker
        worker(name, int(seed), arm)
    else:
        seeds = CONFIRMATION if args.confirmation else args.seeds
        jobs = [(s,n,a) for n in seeds for s in args.datasets for a in args.arms]
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(launch, jobs))


if __name__ == '__main__':
    main()
