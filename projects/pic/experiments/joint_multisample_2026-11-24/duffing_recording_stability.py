"""Replay three Duffing fits and their fixed-frequency controls on fresh recordings."""
import csv
import gzip
import json
from dataclasses import dataclass, replace

from pilot import ROOT, limit_threads, records
from report import describe_term, verify_receipt

if __name__ == '__main__':
    limit_threads()

from cpu_budget import await_capacity
from noise_suite import live
from compare import system_front
import numpy as np


RUN = ROOT/'systematic/results/duffing/one/1'
IDS = (862, 15463, 18952)
FIRST_DRAW = 100_000
DRAWS = 32
# The discovery log records this measurement operator for all three fits.
PLAN = dict(q=(2,), ms=(64,), ss=(8,), cut=(0,), shape=(110,))
EXTRA = 'sin[0]*x[0]^2'


@dataclass
class Fit:
    name: str
    target: int
    columns: list
    terms: list
    weights: np.ndarray


def dispersion(coefs):
    median = np.median(coefs, axis=0)
    mad = np.median(np.abs(coefs - median), axis=0)
    return np.divide(mad, np.abs(median), out=np.where(mad == 0., 0., np.inf), where=median != 0)


def solve(x, y):
    scale = np.linalg.norm(x, axis=0)
    normalized, _, rank, _ = np.linalg.lstsq(x / scale, y, rcond=None)
    if rank != x.shape[1]:
        raise ValueError('The fixed support is rank deficient on this recording')
    return normalized / scale


def measure(matrix, fit):
    y = matrix[:, fit.target]
    x = matrix[:, fit.columns]
    frozen_wape = np.abs(y - x @ fit.weights).sum() / np.abs(y).sum()
    # At this fixed grid and width, discovery used two halves of 55 weak rows.
    coefs = [solve(x[s], y[s]) for s in np.array_split(np.arange(len(y)), 2)]
    return frozen_wape, dispersion(np.array(coefs)).sum(), solve(x, y)


def write_csv(name, rows):
    path = RUN/'recording_stability'/name
    path.parent.mkdir(exist_ok=True)
    with path.with_suffix('.pending').open('w', encoding='utf8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted(set().union(*(r.keys() for r in rows))))
        writer.writeheader()
        writer.writerows(rows)
    path.with_suffix('.pending').replace(path)


def prepare():
    path = RUN/'archive.json.gz'
    verify_receipt(path)
    with gzip.open(path, 'rt', encoding='utf8') as stream:
        archive = json.load(stream)
    problem, training, metadata = records('duffing', 'one', 1)
    assert metadata == archive['samples']
    points = [archive['points'][archive['systems'][i][0]] for i in IDS]
    labels = sorted({k for pt in points for k, _ in pt[2]})
    terms, opaque = live.R.symbols(problem, [archive['labels'][k] for k in labels])
    bank = live.R.LocalBank(live.R.axes_of(problem), terms, opaque, PLAN)
    matrix = bank.matrix(training[0][1])
    fits = []
    for i, pt in zip(IDS, points):
        target = pt[1]
        assert dict(pt[2])[target] == -1.
        predictors = [(k, w) for k, w in pt[2] if k != target]
        fit = Fit(str(i), labels.index(target), [labels.index(k) for k, _ in predictors],
                  [describe_term(archive['labels'][k])[0] for k, _ in predictors],
                  np.array([w for _, w in predictors]))
        err, instability, beta = measure(matrix, fit)
        np.testing.assert_allclose([err, instability], [pt[5][0], pt[4]], rtol=1e-9)
        np.testing.assert_allclose(beta, fit.weights, rtol=1e-8, atol=1e-10)
        fits.append(fit)
        if EXTRA in fit.terms:
            keep = [j for j, t in enumerate(fit.terms) if t != EXTRA]
            cols = [fit.columns[j] for j in keep]
            fits.append(replace(fit, name=f'{i}_without_extra', columns=cols, terms=[fit.terms[j] for j in keep],
                                weights=solve(matrix[:, cols], matrix[:, fit.target])))
    return bank, fits, metadata[0]


def run():
    await_capacity()
    bank, fits, training = prepare()
    series = {f.name: [] for f in fits}
    measurements = []
    seeds, hashes, states = set(), set(), []
    for draw in range(FIRST_DRAW, FIRST_DRAW + DRAWS):
        _, samples, metadata = records('duffing', 'diverse5', draw)
        for (_, fields), record in zip(samples, metadata):
            state, seed, digest = record['state'], record['noise_seed'], record['hashes']['u']
            assert seed != training['noise_seed'] and seed not in seeds
            assert digest != training['hashes']['u'] and digest not in hashes
            seeds.add(seed)
            hashes.add(digest)
            states.append(state)
            matrix = bank.matrix(fields)
            for fit in fits:
                err, local, beta = measure(matrix, fit)
                series[fit.name].append((err, local, beta))
                measurements.append(dict(state=state, noise_seed=seed, u_sha256=digest, fit=fit.name,
                                         frozen_wape=err, within_record_dyad=local, **dict(zip(fit.terms, beta))))
    states = np.asarray(states)
    assert set(states) == set(range(5)) and all(sum(states == i) == DRAWS for i in range(5))
    summary, coefficients, comparisons = [], [], []
    groups = {f'state_{i}': states == i for i in range(5)}
    groups.update(changed_states=states > 0, all_states=states >= 0)
    for group, mask in groups.items():
        dyad_vectors, across_vectors = [], []
        for fit in fits:
            errors, dyads, coefs = (np.asarray(v)[mask] for v in zip(*series[fit.name]))
            spread = dispersion(coefs)
            values = dict(group=group, fit=fit.name, n=len(errors), median_frozen_wape=np.median(errors),
                          mean_within_record_dyad=np.mean(dyads),
                          median_within_record_dyad=np.median(dyads), across_record_instability=spread.sum())
            summary.append(values)
            if fit.name in {str(i) for i in IDS}:
                dyad_vectors.append([*errors, dyads.mean()])
                across_vectors.append([*errors, spread.sum()])
            for term, column, variability in zip(fit.terms, coefs.T, spread):
                coefficients.append(dict(group=group, fit=fit.name, term=term, median=np.median(column),
                                         mad=np.median(np.abs(column-np.median(column))), relative_mad=variability,
                                         positive=int(sum(column > 0)), negative=int(sum(column < 0))))
            if fit.name in (str(IDS[1]), str(IDS[2])):
                for reference in (str(IDS[0]), fit.name+'_without_extra'):
                    ref_errors, ref_dyads, _ = (np.asarray(v)[mask] for v in zip(*series[reference]))
                    comparisons.append(dict(group=group, fit=fit.name, reference=reference, n=len(errors),
                                            lower_frozen_wape=int(sum(errors < ref_errors)),
                                            lower_within_record_dyad=int(sum(dyads < ref_dyads)),
                                            lower_on_both=int(sum((dyads < ref_dyads) & (errors < ref_errors)))))
        for name, vectors in [('on_wape_dyad_front', dyad_vectors), ('on_wape_across_record_front', across_vectors)]:
            front = dict(zip(map(str, IDS), system_front(np.array(vectors))))
            for row in summary[-len(fits):]:
                row[name] = int(front[row['fit']]) if row['fit'] in front else None
    write_csv('recordings.csv', measurements)
    write_csv('summary.csv', summary)
    write_csv('coefficients.csv', coefficients)
    write_csv('paired_counts.csv', comparisons)
    print(f'{len(seeds)} held-out recordings, {len(fits)} fixed supports including controls.')


if __name__ == '__main__':
    run()
