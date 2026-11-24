"""Repeat the minimum front discrepancy on independent noisy measurements."""
from pathlib import Path
import json
import re
from types import SimpleNamespace

import run
from live_score import LiveScore, Measurements, O, R, score
import numpy as np


def labels_of(equation):
    def label(term):
        factors = []
        for name, pars in re.findall(r'([\w\^/]+)\{([^}]+)\}', term):
            params = [p.split(':') for p in pars.split(',')]
            factors.append(('x' if name in ('t','x','y') else name,
                            tuple(sorted((k.strip(),float(v)) for k,v in params))))
        return tuple(sorted(factors))
    lhs, target = equation.split('=')
    labels = [label(target)]
    for term in re.split(r'(?<![eE])\+', lhs):
        factors = label(term)
        coeff = float(term.split('*')[0].strip())
        if coeff != 0:
            labels.append(factors)
    return tuple(sorted(labels))


def check(path):
    record = json.loads(path.read_text())
    row = min(record['front'], key=lambda r: r['objectives'][0])
    system, seed = record['system'], record['seed']
    axes, clean = run.U.load(system)
    code = list(run.U.TRUTH).index(system)
    def noisy(entropy):
        rng = np.random.default_rng(entropy)
        return {v: y + .05*y.std()*rng.standard_normal(y.shape) for v,y in clean.items()}
    original = noisy([seed, code])
    problem = SimpleNamespace(data=original, grids=np.meshgrid(*axes,indexing='ij'),
                              shape=next(iter(clean.values())).shape, named_arrays={})
    scorer = LiveScore(problem, original, seed, record['settings']['max_deriv_order'], 7)
    labels = labels_of(row['equation'])
    replay = scorer.support_score(labels)
    assert np.isclose(replay, row['objectives'][0], rtol=1.e-7, atol=1.e-10), (replay,row)
    fits = list(scorer.fold_fits(labels))
    fold_details = []
    for left, right, scales, vector in fits:
        size = np.linalg.norm(scales * vector)
        a, b = left @ vector / size, right @ vector / size
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        fold_details.append(dict(left_norm=float(na), right_norm=float(nb),
                                 cosine=float((a@b)/(na*nb)) if na*nb else None,
                                 signed_score=float(a@b)))
    terms, opaque = R.symbols(problem, labels)
    opaque = {k: y[scorer.segment] for k,y in opaque.items()}
    def evaluate(raw):
        columns = []
        for h, (_, held, _) in enumerate(scorer.folds):
            mapped = O.map_values(held.covariance.time_map, {v: y[h::3] for v,y in raw.items()})
            measure = Measurements(scorer.window,mapped,held.covariance)
            columns.append(np.column_stack([measure.column(t,opaque) for t in terms]))
        values = []
        for g, (_, _, scales, vector) in enumerate(fits):
            white = scorer.folds[g][2]
            left, right = [white @ columns[h] for h in range(3) if h != g]
            values.append(score(left,right,scales,vector))
        return float(np.mean(values))
    np.testing.assert_allclose(evaluate(original), replay, rtol=1.e-9, atol=1.e-10)
    fresh = [evaluate(noisy([910715,seed,code,b])) for b in range(8)]
    return dict(source=path.name, original=replay, folds=fold_details, fresh=fresh, fresh_median=float(np.median(fresh)),
                terms=row['terms'], distance=row['distance'])


if __name__ == '__main__':
    root = Path(__file__).resolve().parent
    (root/'rechecks').mkdir(exist_ok=True)
    for path in sorted((root/'plain_results').glob('*_weak.json')):
        dest = root/'rechecks'/path.name
        if not dest.exists() or 'folds' not in json.loads(dest.read_text()):
            result = check(path)
            dest.write_text(json.dumps(result, indent=1, allow_nan=False))
            print(path.stem, result['original'], result['fresh_median'], flush=True)
