"""Compare first handover of a complete registered system at matched total-fit counts."""
import json
from itertools import permutations
from pathlib import Path
from statistics import median
import sys

ROOT = Path(__file__).resolve().parent
SWAP = ROOT.parent / 'guided_swap_2026-11-23'
ADD = ROOT.parent / 'two_objective_add_2026-11-23'
sys.path.insert(0, str(SWAP))
from assess_swap import live, np, previous, read_archive, write_csv


CHECKPOINTS = (500, 1000, 2000, live.BUDGET)


def checkpoints(dest, problem, raw, allowed, frequencies):
    rec = read_archive(dest)
    labels = [previous.label_key(term, frequencies) for term in rec['labels']]
    supports = [frozenset(labels[j] for j, _ in p[2]) for p in rec['points']]
    combos = np.asarray(rec['systems'], dtype=int)
    costs = np.array(rec['operator']['system_cost'])
    matched = np.array([tuple(supports[i] for i in combo) in allowed for combo in combos])
    assert len(costs) == len(matched)
    matches = costs[matched & (costs <= live.BUDGET)]
    best = int(matches.min()) if len(matches) else None
    existing = json.loads((dest / 'assessment.json').read_text())
    assert (best is not None) == bool(existing['total']['system']['assembled'])
    hits = dict.fromkeys(CHECKPOINTS, 0)
    if best is not None:
        _, held, dyad, valid, _ = previous.replay(problem, raw, rec)
        eligible = (valid & np.isfinite(held) & (held<=1.))[combos].all(axis=1)
        values = np.column_stack((held[combos], dyad[combos]))
        for budget in CHECKPOINTS:
            rows = eligible & (costs<=budget)
            rivals = values[rows]
            hits[budget] = int(any(not np.any((rivals <= v).all(axis=1) & (rivals < v).any(axis=1))
                                   for v in values[rows & matched]))
    assert hits[live.BUDGET] == existing['total']['system']['held']
    return best, hits


def main():
    rows, fronts = [], []
    for name in live.DATASETS:
        if name == 'ode_simple':
            continue
        problem = live.load_data(name)
        systems, freq = previous.registry(name, problem)
        allowed = {p for system in systems for p in permutations(system)}
        for seed in live.SEEDS:
            row = dict(dataset=name, seed=seed)
            raw = problem.noisy(5,seed)
            for arm, campaign in [('native', SWAP), ('pvra', ADD), ('guided_swap', SWAP), ('pvxa', ADD)]:
                dest = campaign / 'results' / name / str(seed) / arm
                row[arm], hits = checkpoints(dest, problem, raw, allowed, freq)
                fronts.extend(dict(dataset=name, seed=seed, arm=arm, total_fits=k, front_hit=v)
                              for k,v in hits.items())
            rows.append(row)
    write_csv(ROOT / 'system_convergence.csv', rows)
    write_csv(ROOT / 'system_front_convergence.csv', fronts)
    for arm in ('native', 'pvra', 'guided_swap', 'pvxa'):
        pair = [(r['native'], r[arm]) for r in rows if r[arm] is not None and r['native'] is not None]
        print(arm, {k: sum(r[arm] is not None and r[arm] <= k for r in rows)
                    for k in (500, 1000, 2000, live.BUDGET)},
              'paired', len(pair), 'earlier', sum(b < a for a,b in pair),
              'later', sum(b > a for a,b in pair), 'ties', sum(b == a for a,b in pair),
              'shared medians', median(a for a,b in pair), median(b for a,b in pair), flush=True)
        print('Held-out first front', arm, {k:sum(r['front_hit'] for r in fronts
              if r['arm']==arm and r['total_fits']==k) for k in CHECKPOINTS}, flush=True)


if __name__ == '__main__':
    main()
