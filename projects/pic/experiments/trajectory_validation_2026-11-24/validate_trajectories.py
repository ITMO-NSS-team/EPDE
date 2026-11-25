"""Compare final scoring rules on identical discovery archives and independent records."""
import argparse
from dataclasses import replace
from functools import lru_cache
import json

from settings import DATASETS, FRESH, ROOT, fresh_directory, limit_threads

if __name__ == '__main__':
    limit_threads()

from noise_suite import SEEDS, atomic_text, baseline_directory, directory, live, measurement
from compare import read_archive, structural_matches
from score_archive import ArchiveScorer
from slabs import wape
import numpy as np
from scipy.integrate import solve_ivp


def dynamics(name, original):
    def rhs(time, state):
        u, v = state
        if name == 'lv':
            return 20*u*(1-v), 20*v*(u-1)
        if name == 'ode':
            return v, -4*u - np.sin(2*time)*v + 1.5*time
        if name == 'vdp':
            return v, .2*(1-u*u)*v - u
        p = original.meta
        return v, -p['delta']*v - p['alpha']*u - p['beta']*u**3 + p['gamma']*np.cos(p['omega']*time)
    return rhs


@lru_cache(maxsize=None)
def problems(name):
    original = live.load_data(name)
    t, = live.R.axes_of(original)
    states = [(1.25, -.75), (-.75, 1.25), (.6, 1.7), (-1.5, -.3)]
    if name == 'lv':
        states = [(1.4, .7), (.65, 1.35), (1.6, 1.15), (.85, .55)]
    rhs = dynamics(name, original)

    trajectories = []
    for initial in states:
        def integrate(rtol):
            sol = solve_ivp(rhs, (t[0], t[-1]), initial, t_eval=t, method='DOP853',
                            rtol=rtol, atol=rtol*.01)
            if not sol.success:
                raise RuntimeError(sol.message)
            return {k: sol.y[i] for i, k in enumerate(original.variables)}
        fields = integrate(1e-11)
        check = integrate(1e-12)
        for k in fields:
            np.testing.assert_allclose(fields[k], check[k], rtol=1e-6, atol=1e-6)
        trajectories.append(replace(original, data=fields))
    return original, trajectories


def draw(problem, seed, family):
    if family == 'correlated':
        return measurement(family)(problem, 5, seed)[0]
    return problem.noisy(5 if family == 'gaussian' else 0, seed)


def run(name, seed, family):
    if seed in FRESH:
        dest = fresh_directory(name, seed, family)
    elif family == 'correlated':
        dest = directory(name, seed, family)
    else:
        dest = baseline_directory(name, seed, 5 if family == 'gaussian' else 0)
    archive = read_archive(dest)
    assert (archive['system'], archive['seed']) == (name, seed)
    original, trajectories = problems(name)
    scorer = ArchiveScorer(original, archive)
    raw = draw(original, seed, family)
    assert archive['train_hashes'] == {k: live.R.digest(v) for k, v in raw.items()}
    matrix = scorer.matrix(raw)
    replay = wape(matrix, scorer.coefficients, scorer.targets)
    np.testing.assert_allclose(replay[scorer.valid], scorer.train[scorer.valid], rtol=1e-6, atol=2e-9)

    def measure(records, offset):
        return scorer.recording_errors(draw(p, seed + offset + i*10000, family) for i, p in enumerate(records))
    same = measure([original, original], 10000)
    new = measure(trajectories[:2], 10000)
    methods = dict(original=scorer.train, fresh_same=same.mean(axis=0),
                   fresh_new=new.mean(axis=0), disjoint=scorer.blocked_error(matrix, 0),
                   buffered=scorer.blocked_error(matrix, 14))
    selections = {m: scorer.select(errors) for m, errors in methods.items() if errors is not None}
    selections.update(fresh_same_front=scorer.select_multisample(same),
                      fresh_new_front=scorer.select_multisample(new))
    independent = measure(trajectories[2:], 30000)
    # All candidates are selected before consulting the catalog, including on generation misses.
    _, _, hit = structural_matches(name, original, archive, np.asarray(archive['systems'], int))
    records = {}
    for method, selection in selections.items():
        front = selection['front']
        records[method] = dict(front=front, front_hit=hit(front), front_size=len(front))
    members = sorted({i for value in selections.values() for i in value['front']})
    held = independent[:, scorer.systems[members]].transpose(1,0,2)
    diagnostics = dict(zip(members, np.where(np.isfinite(held), held, None).tolist()))
    result = dict(dataset=name, seed=seed, family=family, train_hashes=archive['train_hashes'],
                  generated=hit(range(len(archive['systems']))), methods=records,
                  held_out_wape_by_system=diagnostics)
    output = ROOT / 'pareto_scores' / name / str(seed) / (family + '.json')
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_text(output, json.dumps(result, allow_nan=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cohort', choices=('retrospective', 'fresh'))
    parser.add_argument('--datasets', nargs='+', choices=DATASETS, default=DATASETS)
    args = parser.parse_args()
    for name in args.datasets:
        for seed in (FRESH if args.cohort == 'fresh' else SEEDS):
            for family in (('correlated',) if args.cohort == 'fresh' else ('correlated','clean','gaussian')):
                run(name, seed, family)
                print('Scored', name, seed, family, flush=True)
