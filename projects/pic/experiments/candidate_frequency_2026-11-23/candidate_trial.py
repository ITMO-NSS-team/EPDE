"""Refine a continuous trig parameter in an actually discovered parent equation."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import os
from pathlib import Path
import subprocess
import sys
from time import monotonic
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / 'guided_swap_2026-11-23'))
from swap_trial import Timeline, folder as native_folder, live
from live import np, plain_fit
from epde.structure.weak_rows import WeakRows
from scipy.optimize import minimize_scalar

CHECKPOINT = 2400
LIMIT = 12
CONFIRMATION = range(923701, 923713)


def folder(name, seed, arm):
    if arm == 'native' and seed in live.SEEDS:
        return native_folder(name, seed, arm)
    return ROOT / 'results' / name / str(seed) / arm


def trig_terms(term):
    for factor in term.structure:
        if factor.label in ('sin', 'cos'):
            p = {v['name']: i for i, v in factor.params_description.items()}
            yield factor, p['freq'], int(factor.params[p['dim']])


class Trial(Timeline):
    def __init__(self, variables, enabled):
        super().__init__(variables)
        self.enabled = enabled
        self.pool = {}
        self.seen = set()
        self.ranges = {}
        self.profiles = []
        self.started_at = None
        self.work = Counter()
        self.seconds = 0.

    def family(self, train, axes, plan, settings):
        family = live.U.TrigonometricTokens(freq_from_data=2)
        native = family.frequencies_from_data

        def estimate(trajectories):
            result = native(trajectories)
            for axis, peaks in (result if isinstance(result, dict) else {0: result}).items():
                dx = float(np.median(np.diff(axes[axis])))
                width = 2*np.pi/(len(axes[axis])*dx)
                for i, w in enumerate(peaks):
                    lo = max(np.finfo(float).eps, w-width/2)
                    hi = min(np.pi/dx, w+width/2)
                    if i:
                        lo = max(lo, np.nextafter((w+peaks[i-1])/2, w))
                    if i+1 < len(peaks):
                        hi = min(hi, np.nextafter((w+peaks[i+1])/2, w))
                    self.ranges[axis, float(w)] = float(lo), float(hi), width
            return result

        family.frequencies_from_data = estimate
        return family

    def hand_over(self, system):
        combo = live.Observer.hand_over(self, system)
        if combo is None:
            return None
        self.system_cost.setdefault(combo, self.cost)
        if not self.enabled or self.calls > CHECKPOINT or self.started_at is not None:
            return combo
        for index, point in enumerate(combo):
            if point in self.seen:
                continue
            self.seen.add(point)
            slot, _, pairs, wape, dyad, _ = self.points[point]
            eq = system.vals[slot]
            if wape > 1 or any(trig_terms(eq.target)):
                continue
            tones = {(axis, float(f.params[i])) for t in eq.structure for f,i,axis in trig_terms(t)}
            if len(tones) != 1:
                continue
            tone, = tones
            lo, hi, _ = self.ranges[tone]
            if not lo < tone[1] < hi:
                continue
            self.work['eligible_parent_points'] += 1
            snapshot = None
            for objective, score in enumerate(((wape, dyad), (dyad, wape))):
                key = len(pairs)-1, objective, index
                if key in self.pool and self.pool[key][0] <= score:
                    continue
                if snapshot is None:
                    snapshot = deepcopy(system)
                self.pool[key] = score, point, combo, tone, snapshot
        return combo

    def profile(self, item, fit_args):
        _, point, combo, tone, parent = item
        slot = self.points[point][0]
        eq = deepcopy(parent.vals[slot])
        intercept = bool(eq.weights_final[-1])
        eq.reset_state(False)
        factors = [(t,f,i) for t in eq.structure for f,i,_ in trig_terms(t)]
        origin = tone[1]
        lo, hi, width = self.ranges[tone]
        cache = {}

        def install(value):
            for term, factor, index in factors:
                desc = dict(factor.params_description)
                desc[index] = dict(desc[index], grid=(value,))
                factor.params_description = desc
                params = factor.params.copy()
                params[index] = value
                factor.params = params
                term.resetSavedState()
            eq._invalidate_label_cache()

        def score(value):
            value = float(value)
            if value in cache:
                self.work['profile_cache_hits'] += 1
                return cache[value]
            install(value)
            targets, features = eq.evaluate()
            key, = targets
            y = targets[key].ravel()
            x = features[key]
            if intercept:
                x = np.column_stack((x, np.ones(len(y))))
            self.extra += 1
            norms = np.linalg.norm(x, axis=0)
            sse = np.inf
            if np.all(norms > 0) and np.isfinite(x).all():
                unit = x/norms
                weights, _, rank, _ = np.linalg.lstsq(unit, y, rcond=None)
                if rank == len(norms):
                    residual = y-unit@weights
                    sse = float(residual@residual)
            cache[value] = sse
            return sse

        row = dict(parent=list(combo), point=point, slot=slot, axis=tone[0],
                   initial=origin, lower=lo, upper=hi, start_cost=self.cost, children=[])
        self.profiles.append(row)
        initial = score(origin)
        best = origin
        if np.isfinite(initial):
            score(lo)
            score(hi)
            optimum = minimize_scalar(score, method='bounded', bounds=(lo, hi),
                                      options=dict(xatol=width*1.e-4, maxiter=24))
            score(optimum.x)
            best = min(cache, key=lambda f: (cache[f], abs(f-origin)))
            if cache[best] < initial:
                install(best)
                for variant in plain_versions(eq, *fit_args):
                    child = deepcopy(parent)
                    child.vals.replace_gene(gene_key=slot, value=variant)
                    if not plain_fit.redundant_equations(list(child)):
                        child_combo = self.hand_over(child)
                        if child_combo is not None:
                            row['children'].append(list(child_combo))
        row.update(proposed=best, initial_sse=initial if np.isfinite(initial) else None,
                   best_sse=cache[best] if np.isfinite(cache[best]) else None,
                   fits=len(cache), end_cost=self.cost)

    def fit(self, eq, *args):
        if self.enabled and self.calls >= CHECKPOINT and self.started_at is None:
            self.started_at = self.calls
            start = monotonic()
            used = set()
            native_columns = WeakRows.columns

            def columns(rows, term, slot):
                hit = (slot, term.factors_labels) in rows._columns
                self.work['weak_cache_hits' if hit else 'weak_columns_built'] += 1
                return native_columns(rows, term, slot)

            with patch.object(WeakRows, 'columns', columns):
                for key, item in sorted(self.pool.items()):
                    if item[1] in used:
                        continue
                    used.add(item[1])
                    self.profile(item, args)
                    if len(used) == LIMIT:
                        break
            self.seconds = monotonic()-start
            self.pool.clear()
        return plain_versions(eq, *args)

    def run(self, arm, seed, part, *, experiment):
        with patch.object(plain_fit, '_versions', self.fit):
            experiment()
        assert not self.enabled or self.started_at is not None
        assert self.extra == sum(r['fits'] for r in self.profiles)
        return dict(arm=arm, profiles=self.profiles, checkpoint=self.started_at, work=self.work,
                    frequency_seconds=self.seconds, extra_fits=self.extra, gain_columns=0, hashes={},
                    full_cost=self.full_cost, system_cost=[self.system_cost[c] for c in sorted(self.systems)])


plain_versions = plain_fit._versions


def worker(name, seed, arm, *, noise_level=5, epoch_budget=None, loader_options=None, observation=None):
    def observe(variables):
        nonlocal trial
        trial = Trial(variables, arm == 'candidate')
        return trial
    trial = None
    live.run(name, seed, arm, observer_factory=observe, destination=folder(name, seed, arm),
             operator_run=lambda *a, **kw: trial.run(*a, **kw),
             trig_factory=lambda *a: trial.family(*a), noise_level=noise_level,
             epoch_budget=epoch_budget, loader_options=loader_options, observation=observation)


def launch(job):
    name, seed, arm = job
    dest = folder(*job)
    if (dest/'archive.json.gz').is_file():
        return
    if arm == 'native' and seed in live.SEEDS:
        raise FileNotFoundError(dest)
    dest.mkdir(parents=True, exist_ok=True)
    with (dest/'search.log').open('w') as log:
        proc = subprocess.run([sys.executable, '-u', __file__, '--worker', name, str(seed), arm],
                              env=dict(os.environ, PYTHONHASHSEED='0'), stdout=log, stderr=subprocess.STDOUT)
    if proc.returncode:
        raise RuntimeError((dest/'search.log').read_text(errors='replace')[-3200:])
    print(*job, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--worker', nargs=3)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--datasets', nargs='+', choices=live.DATASETS, default=live.DATASETS)
    parser.add_argument('--seeds', nargs='+', type=int, default=live.SEEDS)
    parser.add_argument('--confirmation', action='store_true')
    args = parser.parse_args()
    if args.worker:
        name, seed, arm = args.worker
        worker(name, int(seed), arm)
    else:
        seeds = CONFIRMATION if args.confirmation else args.seeds
        jobs = [(s,n,a) for n in seeds for s in args.datasets for a in ('native','candidate')]
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            list(pool.map(launch, jobs))
