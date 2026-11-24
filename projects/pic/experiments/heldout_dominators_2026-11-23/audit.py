"""Replay actual system handovers at the equal-fit endpoint with frozen coefficients."""
import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TRIAL = ROOT.parent / 'candidate_frequency_2026-11-23'
sys.path.insert(0, str(TRIAL))
from candidate_trial import folder, live
from assess_swap import previous, read_archive, write_csv
from live import np


def physical_check(problem, fields, archive):
    """Apply the published primary law to noisy and clean held-out weak columns."""
    target, terms, weights = live.R.parse(problem.truth[0])
    symbols, opaque = live.R.symbols(problem, [target, *terms])
    _, _, (slices, axes) = live.split(live.R.axes_of(problem))
    plan = {k: tuple(v) for k, v in archive['plan'].items()}
    bank = live.R.LocalBank(axes, symbols, {k: v[slices] for k, v in opaque.items()}, plan)
    clean, noisy = [bank.matrix({k: v[slices] for k, v in data.items()})
                    for data in (problem.data, fields)]
    def wape(matrix):
        return np.abs(matrix[:, 1:] @ weights - matrix[:, 0]).sum() / np.abs(matrix[:, 0]).sum()
    metrics = dict(physical_clean_wape=wape(clean), physical_noisy_wape=wape(noisy))
    for i, term in enumerate((target, *terms)):
        key = previous.label_key(term, ())
        metrics['noise_to_clean_' + key] = np.abs(noisy[:, i] - clean[:, i]).sum() / np.abs(clean[:, i]).sum()
        if key == 'd^2u/dx1^2' and problem.name == 'ac':
            metrics['clean_diffusion_to_target'] = np.abs(weights[i-1]*clean[:, i]).sum() / np.abs(clean[:, 0]).sum()
    return metrics


def audit(reference):
    name, seed = reference['dataset'], int(reference['seed'])
    directory = folder(name, seed, 'candidate')
    archive = read_archive(directory)
    problem = live.load_data(name)
    fields = problem.noisy(5, seed)
    train, held, dyad, valid, error = previous.replay(problem, fields, archive)
    points = archive['points']
    allowed, frequencies = previous.registry(name, problem)
    assert archive['variables'] == ['u']
    forms = {variant[0] for variant in allowed}
    labels = [previous.label_key(term, frequencies) for term in archive['labels']]
    pivots = [labels[p[1]] for p in points]
    supports = [frozenset(labels[j] for j, _ in p[2]) for p in points]
    actual = np.array(sorted({i for (i,), cost in zip(archive['systems'], archive['operator']['system_cost'], strict=True)
                              if cost <= live.BUDGET}), dtype=int)
    matches = [i for i in actual if supports[i] in forms]
    pool = actual[valid[actual] & np.isfinite(held[actual]) & (held[actual] <= 1.)]
    front = previous.nondominated(pool, held, dyad)
    assert len(front) == int(reference['held_system_front_size'])
    assert bool(matches) == bool(int(reference['assembled']))
    assert (any(supports[i] in forms for i in front)) == bool(int(reference['held']))
    groups = {}
    for i in pool:
        groups.setdefault(pivots[i], []).append(i)
    by_target = {int(i) for group in groups.values()
                 for i in previous.nondominated(np.asarray(group), held, dyad)}
    assert set(front).issubset(by_target)
    native_valid = [i for i in matches if valid[i]]
    eligible = sorted(set(matches).intersection(pool))
    fate = ('dominated' if eligible else 'outside held WAPE cutoff' if native_valid
            else 'failed native eligibility' if matches else 'no actual correct system')
    identity = dict(dataset=name, seed=seed)
    row = dict(**identity, fate=fate, actual=len(actual), correct=len(matches), eligible=len(eligible),
               held_front_size=len(front), target_front_size=len(by_target),
               recovered_by_target=int(bool(by_target.intersection(matches))), max_train_replay_error=error)
    descriptions, witnesses = [], []
    def describe(i):
        return dict(point=int(i), target=pivots[i], train_wape=float(train[i]), held_wape=float(held[i]),
                    dyad=float(dyad[i]), coefficients=json.dumps({labels[j]: w for j, w in points[i][2]}))
    for i in matches:
        rivals = pool[(held[pool] <= held[i]) & (dyad[pool] <= dyad[i]) &
                      ((held[pool] < held[i]) | (dyad[pool] < dyad[i]))] if i in eligible else []
        same = [j for j in rivals if pivots[j] == pivots[i]]
        short = ({'du/dx0', 'u', 'u^3'} if name == 'ac' else {'u', 'd^2u/dx1^2'})
        simple = [j for j in same if supports[j] == short]
        descriptions.append(dict(**identity, **describe(i), eligible=int(i in eligible),
                                 dominators=len(rivals), same_target=len(same), simple_same_target=len(simple)))
        if i in eligible:
            assert len(rivals)
            assert (i in by_target) == (len(same) == 0)
        for kind, ids in [('held_front', np.intersect1d(rivals, front)), ('same_target', same),
                          ('simple_same_target', simple)]:
            if len(ids):
                j = min(ids, key=lambda j: (held[j], dyad[j], j))
                witnesses.append(dict(**identity, correct_point=int(i), selection=kind, **describe(j)))
    if name in ('ac', 'wave'):
        row.update(physical_check(problem, fields, archive))
    return row, descriptions, witnesses


def write_table(filename, rows):
    columns = dict.fromkeys(k for row in rows for k in row)
    write_csv(ROOT / filename, [{**columns, **r} for r in rows])


if __name__ == '__main__':
    with (TRIAL / 'confirmation/systems.csv').open() as stream:
        records = [r for r in csv.DictReader(stream) if r['arm'] == 'candidate' and r['budget'] == 'total'
                   and r['dataset'] != 'ode_simple' and (r['dataset'] in ('ac','wave')
                   or (r['assembled'] == '1' and r['held'] == '0'))]
    runs, correct, examples = [], [], []
    for record in records:
        run, details, witnesses = audit(record)
        runs.append(run)
        correct.extend(details)
        examples.extend(witnesses)
    for filename, rows in [('runs.csv', runs), ('correct_points.csv', correct), ('witnesses.csv', examples)]:
        write_table(filename, rows)
    print(Counter((r['dataset'], r['fate']) for r in runs))
