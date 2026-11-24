"""Blindly choose weak rows, fit once, and compare held-out normalizations."""
from concurrent.futures import ProcessPoolExecutor
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent/'full_resolution_fit_2026-11-18'))
from study import R, SYSTEMS, np, wls
from epde.interface.search_config import load_search_config, set_active_config
from epde.operators.common.survival import configured_width, dyad_scores, row_floor

SEEDS = (923201, 923202, 923203)
CONFIRMATION_SEEDS = range(923211, 923223)
SCORES = ('source_wape', 'held_wape', 'held_train_target', 'held_all_terms')


def split(axes):
    axis = max(range(len(axes)), key=lambda k: len(axes[k]))
    half = len(axes[axis])//2
    def side(segment):
        cuts = [slice(None)]*len(axes)
        cuts[axis] = segment
        return tuple(cuts), [a[s] for a, s in zip(axes, cuts)]
    a, b = side(slice(0, half)), side(slice(-half, None))
    assert not np.intersect1d(a[1][axis], b[1][axis]).size
    return axis, a, b


def training_plan(fields, axes, width):
    dims = len(axes)
    zero = (0,)*dims
    library, targets = [], []
    for var in sorted(fields):
        library.append(R.WF.WeakTerm([(var, zero)], zero))
        for axis in range(dims):
            for order in range(1, 1+(2 if axis == 0 else 4)):
                deriv = tuple(order if j == axis else 0 for j in range(dims))
                targets.append(len(library))
                library.append(R.WF.WeakTerm([(var, deriv)], zero))
    for axis in range(dims):
        library.append(R.WF.WeakTerm([], tuple(int(axis == j) for j in range(dims))))
    return R.WF.choose_window(R.WF.Fields(fields, axes), library, targets, np.random.default_rng([0, 0]),
                              rows_ok=lambda shape: row_floor('dyad', shape, width) is None, gate=R.WF.HALF_POWER)


def relative_residual(bank, vector, scale):
    return float(np.mean(np.abs(bank @ vector))/scale) if scale > 0 else None


def score(train, held, candidate, shape):
    ids = [candidate['yi'], *candidate['xi']]
    A, B = train[:, ids], held[:, ids]
    v, status = wls(A, 0)
    if status != 'ok':
        return dict(status=status)
    dyad = float(np.sum(dyad_scores(A[:, 1:], A[:, 0], np.ones(len(A)), shape,
                                    fit_intercept=False))) if len(ids) > 1 else 0.
    if not np.isfinite(dyad) or dyad >= np.finfo(float).max:
        return dict(status='unresolved training dyad')
    v = -v/v[0]
    scale = float(np.mean(np.abs(A), axis=0) @ np.abs(v))
    train_target = np.mean(np.abs(A[:, 0]))
    values = [relative_residual(A, v, train_target), relative_residual(B, v, np.mean(np.abs(B[:, 0]))),
              relative_residual(B, v, train_target), relative_residual(B, v, scale)]
    if any(value is None or not np.isfinite(value) for value in values):
        return dict(status='unresolved discrepancy')
    units = np.geomspace(.05, 20., len(ids))*(-1.)**np.arange(len(ids))
    other_v = (-3.7*v/units)[::-1]
    other_train, other_held = (A*units)[:, ::-1], (B*units)[:, ::-1]
    check = relative_residual(other_held, other_v, np.mean(np.abs(other_train), axis=0) @ np.abs(other_v))
    np.testing.assert_allclose(check, values[-1], rtol=1e-11, atol=1e-13)
    return dict(status=status, dyad=dyad, weights=v.tolist(), **dict(zip(SCORES, values)))


def run(job):
    system, seed, destination = job
    cfg = load_search_config(overrides=dict(evaluation_form='weak', system_rps_mode='plain',
                             instability_metric='dyad', equation_terms_max_number=7))
    set_active_config(cfg)
    problem = R.datasets.load(system)
    raw = problem.noisy(5, seed)
    axis, (sl_a, axes_a), (sl_b, axes_b) = split(R.axes_of(problem))
    training = {k: v[sl_a] for k, v in raw.items()}
    validation = {k: v[sl_b] for k, v in raw.items()}
    result = dict(system=system, seed=seed, axis=axis, grid=list(next(iter(training.values())).shape),
                  train_hashes={k: R.digest(v) for k, v in training.items()})
    try:
        picked = training_plan(training, axes_a, configured_width(cfg.search_space))
    except ValueError as err:
        result.update(plan=None, reason=str(err), rows=[])
    else:
        plan = {k: getattr(picked, k) for k in ('q', 'ms', 'ss', 'cut', 'shape')}
        candidates, labels = R.panel(problem)
        terms, opaque = R.symbols(problem, labels)
        def matrix(fields, axes, cuts):
            bank = R.LocalBank(axes, terms, {k: v[cuts] for k, v in opaque.items()}, plan)
            mat = bank.matrix(fields)
            if () in labels:
                j = labels.index(())
                np.testing.assert_allclose(mat[:, j], 1., atol=1e-12)
                mat[:, j] = 1.
            return mat
        A, B = matrix(training, axes_a, sl_a), matrix(validation, axes_b, sl_b)
        rows = []
        for c in candidates:
            val = score(A, B, c, picked.shape)
            rows.append(dict(id=c['id'], slot=c['slot'], label=c['label'], registered=c['registered'], **val))
        result.update(plan=plan, used_ungated=picked.admissible == 0, rows=rows)
    result['slots'] = len(problem.truth)
    R.dump(destination/'results'/f'{system}_{seed}.json', result)
    return system, seed, bool(result['plan'])


def collect(seeds, destination):
    output = []
    for system in SYSTEMS:
        for seed in seeds:
            rec = json.loads((destination/'results'/f'{system}_{seed}.json').read_text())
            for slot in range(rec['slots']):
                all_rows = [r for r in rec['rows'] if r['slot'] == slot]
                valid = [r for r in all_rows if r['status'] == 'ok']
                for key in SCORES:
                    front = [r for r in valid if not any(s[key] <= r[key] and s['dyad'] <= r['dyad']
                             and (s[key] < r[key] or s['dyad'] < r['dyad']) for s in valid)]
                    output.append(dict(system=system, seed=seed, slot=slot, score=key, usable_plan=int(bool(rec['plan'])),
                                       candidates=len(all_rows), valid=len(valid), front=len(front),
                                       registered=int(any(r['registered'] for r in front)),
                                       exact=int(any(r['label']=='reference' for r in front)),
                                       additions=sum(r['label'].startswith('add_') and not r['registered'] for r in front)))
    with (destination/'slots.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=output[0])
        writer.writeheader()
        writer.writerows(output)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--confirm', action='store_true')
    args = parser.parse_args()
    seeds = CONFIRMATION_SEEDS if args.confirm else SEEDS
    destination = ROOT/'confirmation' if args.confirm else ROOT
    jobs = [(s,n,destination) for n in seeds for s in SYSTEMS]
    with ProcessPoolExecutor(max_workers=6) as pool:
        for result in pool.map(run, jobs):
            print(*result, flush=True)
    collect(seeds, destination)
