"""One paired search and its strict and frequency-independent recovery scores."""
from dataclasses import replace
from functools import partial
from pathlib import Path
import json
import sys
from time import monotonic, process_time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / 'epoch_full_record_2026-11-23'))
from epoch_settings import OPTIONS, directory, epochs, live, np, trial, whole_record
from compare import read_archive, structural_matches
from recovery import lorenz_identity
from epde import globals as global_var
from epde.optimizers.nsga.optimizer import NSGAOptimizer


def blind(problem):
    if problem.name != 'kdv_cossin':
        return problem
    source = 'cos{power: 1.0, freq: 1.0, dim: 0.0} * sin{power: 1.0, freq: 1.0, dim: 1.0}'
    return replace(problem, truth=[law.replace('cos(t)sin(x){power: 1.0}', source) for law in problem.truth],
                   extra_tokens=None, extra_arrays={})


def archive_dir(name, seed, noise, arm):
    if arm == 'native' and name != 'kdv_cossin' and (directory(name, seed, noise)/'archive.json.gz').is_file():
        return directory(name, seed, noise)
    return ROOT/'results'/name/str(seed)/f'noise{noise}'/arm


def frozen_errors(problem, archive, fronts, seed, noise):
    """Evaluate archived coefficients and exact frequencies on a paired noise replicate."""
    ids = {p for front in fronts for system in front for p in system}
    points = archive['points']
    terms = sorted({j for i in ids for j, _ in points[i][2]})
    raw = problem.noisy(noise, seed)
    fields = live.R.WF.Fields(raw, live.R.axes_of(problem))
    plan = archive['plan']
    columns = {}
    for j in terms:
        label = archive['labels'][j]
        if not label:
            columns[j] = np.ones(int(np.prod(plan['shape'])))
            continue
        (term,), opaque = live.R.symbols(problem, [label])
        fields.opaque.update(opaque)
        try:
            matrix, _ = live.R.WF.weak_matrix([term.monomials(psi_cap=plan['q'])], [term.opaque],
                                               fields, plan['cut'], plan['ms'], plan['ss'], plan['q'])
            columns[j] = matrix[:, 0]
        finally:
            fields.opaque.clear()
    errors = {}
    for i in ids:
        _, target, pairs, *_ = points[i]
        residual = sum(w*columns[j] for j, w in pairs)
        scale = np.abs(columns[target]).mean()
        errors[i] = float(np.abs(residual).mean()/scale) if scale else None
    return {str(i): e for i,e in sorted(errors.items())}


def run(name, seed, noise, arm, dest):
    load = live.load_data
    generation_cost = []
    started = monotonic()
    cpu_started = process_time()
    cutoff = {}
    budget = None
    if arm != 'native':
        baseline = ROOT/'results'/name/str(seed)/f'noise{noise}'/'native'/'score.json'
        budget = json.loads(baseline.read_text())['total_work']
    state = None

    class Fitted(trial.Timeline):
        @property
        def cost(self):
            rows = getattr(global_var.samples_manager, 'weak', None)
            return self.calls + getattr(rows, 'frequency_solves', 0)

        def run(self, arm, seed, part, *, experiment):
            experiment()
            return dict(arm=arm, extra_fits=self.cost-self.calls, full_cost=self.full_cost,
                        system_cost=[self.system_cost[c] for c in sorted(self.systems)],
                        generation_cost=generation_cost)

    def observe(variables):
        nonlocal state
        state = Fitted(variables) if arm == 'fitted' else trial.Trial(variables, True)
        return state

    def family(*args):
        make = live.U.TrigonometricTokens
        if arm == 'fitted':
            return make(meaningful=True, allow_products=True, fit_frequencies=True)
        if arm == 'native':
            return state.family(*args)
        with patch.object(live.U, 'TrigonometricTokens', partial(make, meaningful=True, allow_products=True)):
            return state.family(*args)

    native_fit = live.U.EpdeSearch.fit
    def fit(search, *args, **kwargs):
        after = search.optimizer_exec_params['early_stopping_callback']
        def record(snapshot, index):
            stop = after(snapshot, index)
            generation_cost.append(state.cost)
            return stop
        search.optimizer_exec_params['early_stopping_callback'] = record
        return native_fit(search, *args, **kwargs)

    def operator(*args, **kwargs):
        result = state.run(*args, **kwargs)
        result['generation_cost'] = generation_cost
        result['cutoff'] = cutoff
        result['cpu_seconds'] = process_time() - cpu_started
        return result

    insert = NSGAOptimizer._archive_insert
    def archive_insert(optimizer, system):
        if budget is not None and not cutoff and state.cost > budget:
            cutoff['members'] = [[state.point_ids[state.key(s.vals[v])] for v in state.variables]
                                 for s in optimizer.reported_front()]
            cutoff['seconds_at_next_insertion'] = monotonic() - started
            cutoff['cpu_seconds_at_next_insertion'] = process_time() - cpu_started
        return insert(optimizer, system)

    with (patch.object(live, 'split', whole_record), patch.object(live.U.EpdeSearch, 'fit', fit),
          patch.object(NSGAOptimizer, '_archive_insert', archive_insert),
          patch.object(live, 'load_data', lambda *a, **kw: blind(load(*a, **kw)))):
        live.run(name, seed, arm, destination=dest, observer_factory=observe, operator_run=operator,
                 trig_factory=family, noise_level=noise, epoch_budget=epochs(name), loader_options=OPTIONS.get(name))


def score(name, seed, noise, arm, dest):
    archive = read_archive(dest)
    native = archive if arm == 'native' else read_archive(archive_dir(name, seed, noise, 'native'))
    budget = native['calls'] + native['operator']['extra_fits']
    problem = blind(live.load_data(name, **OPTIONS.get(name, {})))
    assert archive['train_hashes'] == native['train_hashes']
    assert archive['train_hashes'] == {k: live.R.digest(v) for k,v in problem.noisy(noise, seed).items()}
    assert archive['epochs_completed'] == epochs(name) and archive['settings']['early_stopping_patience'] is None
    assert all(factor[0] != 'cos(t)sin(x)' for label in archive['labels'] for factor in label)
    work = archive['operator']
    total = archive['calls'] + work['extra_fits']
    generated = [s for s,c in zip(archive['systems'], work['system_cost'], strict=True) if c <= budget]
    matched_epoch = epochs(name)
    matched = archive['native_front']
    if total > budget:
        costs = work['generation_cost']
        assert len(costs) == epochs(name) and costs == sorted(costs)
        matched_epoch = sum(c <= budget for c in costs)
        matched = work['cutoff'].get('members', archive['native_front'])

    identity = lorenz_identity(archive) if name == 'lorenz' else None
    def hits(front, relaxed):
        if not front:
            return 0
        labels = archive['labels']
        if relaxed:
            # For this suite each forcing problem registers one frequency per factor/axis.
            refs = {}
            for system in problem.truth_systems:
                for law in system:
                    target, features, _ = live.R.parse(law)
                    for term in [target, *features]:
                        for label, params in term:
                            p = dict(params)
                            if 'freq' in p:
                                refs.setdefault((label, p['dim']), set()).add(p['freq'])
            if any(len(ws) != 1 for ws in refs.values()):
                raise ValueError('Relaxed frequency matching requires an unambiguous factor and axis.')
            def relax(label, params):
                p = dict(params)
                ws = refs.get((label, p.get('dim')))
                if ws:
                    p['freq'], = ws
                return label, sorted(p.items())
            labels = [[relax(*factor) for factor in term] for term in labels]
            # Two independently fitted terms can become identical after removing
            # frequency information. Preserve the registered number of terms.
            signatures = [tuple((n, tuple(map(tuple, p))) for n,p in term) for term in labels]
            distinct = [len({signatures[j] for j,_ in p[2]}) == len(p[2]) for p in archive['points']]
            front = [system for system in front if all(distinct[i] for i in system)]
            if not front:
                return 0
        _, _, hit = structural_matches(name, problem, dict(archive, labels=labels), np.array(front))
        return int(hit(range(len(front))) or (identity is not None and identity(front)))

    result = dict(dataset=name, seed=seed, noise=noise, arm=arm, matched_budget=budget,
                  matched_epoch=matched_epoch, final_epoch=epochs(name), calls=archive['calls'],
                  frequency_solves=work['extra_fits'], total_work=total, seconds=archive['seconds'],
                  cpu_seconds=work.get('cpu_seconds'),
                  matched_cpu_seconds_upper_bound=work.get('cutoff', {}).get('cpu_seconds_at_next_insertion',
                                                                               work.get('cpu_seconds')),
                  matched_seconds_upper_bound=work.get('cutoff', {}).get('seconds_at_next_insertion',
                                                                          archive['seconds']),
                  matched_front_size=len(matched), final_front_size=len(archive['native_front']))
    for stage, front in [('generated_at_budget', generated), ('matched', matched),
                         ('generated', archive['systems']), ('final', archive['native_front'])]:
        result[stage] = hits(front, False)
        result[stage+'_form'] = hits(front, True)
    result['validation_seed'] = seed + 44000
    result['frozen_front_point_wape'] = frozen_errors(problem, archive, (matched, archive['native_front']),
                                                       result['validation_seed'], noise)
    return result


def job(name, seed, noise, arm):
    dest = archive_dir(name, seed, noise, arm)
    if not (dest/'archive.json.gz').is_file():
        dest.mkdir(parents=True, exist_ok=True)
        run(name, seed, noise, arm, dest)
    result = score(name, seed, noise, arm, dest)
    output = ROOT/'results'/name/str(seed)/f'noise{noise}'/arm/'score.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    temp = output.with_suffix('.pending')
    temp.write_text(json.dumps(result, indent=2, allow_nan=False))
    temp.replace(output)
