"""Test cross-record smoothing of the weak dispersion column on a forced-KdV fixed support."""
import csv
from io import StringIO
from functools import lru_cache
from settings import ROOT, limit_threads

if __name__ == '__main__':
    limit_threads()

from noise_suite import SEEDS, atomic_text, directory, live, measurement
from compare import read_archive
import numpy as np
from numpy.polynomial.legendre import legvander


@lru_cache(maxsize=None)
def coordinate_basis(length):
    return np.linalg.qr(legvander(np.linspace(-1., 1., length), length-1))[0]


def repeatable_column(a, b, shape):
    # Orthogonal polynomial projections over weak-row centers. Select bandwidth by
    # cross-record prediction; the final projection uses both records symmetrically.
    qt, qx = (coordinate_basis(n) for n in shape)
    ca, cb = [qt.T @ col.reshape(shape) @ qx for col in (a, b)]
    degrees = [sorted({min(d, n-1) for d in (0, 1, 2, 3, 4, 6, n-1)}) for n in shape]
    best = None
    for dt in degrees[0]:
        for dx in degrees[1]:
            keep = np.s_[:dt+1, :dx+1]
            pa, pb = np.zeros_like(ca), np.zeros_like(cb)
            pa[keep], pb[keep] = ca[keep], cb[keep]
            risk = np.sum((pa-cb)**2 + (pb-ca)**2)
            if best is None or risk < best[0]:
                best = risk, dt, dx, (pa+pb)*.5
    _, dt, dx, chosen = best
    return (qt @ chosen @ qx.T).ravel(), dt, dx


def record(bank, clean, idx, held, name, degree):
    features = bank[:, 1:]
    beta = np.linalg.lstsq(features, bank[:, 0], rcond=None)[0]
    reduced = np.delete(features, idx-1, axis=1)
    reduced_beta = np.linalg.lstsq(reduced, bank[:, 0], rcond=None)[0]
    def error(mat, drop=False):
        design = np.delete(mat[:, 1:], idx-1, axis=1) if drop else mat[:, 1:]
        residual = mat[:, 0] - design @ (reduced_beta if drop else beta)
        return np.abs(residual).sum() / np.abs(mat[:, 0]).sum()
    return dict(method=name, degree_t=degree[0], degree_x=degree[1],
                coefficient=float(beta[idx-1]), absolute_coefficient_error=float(abs(beta[idx-1]+1)),
                relative_column_error=float(np.linalg.norm(bank[:,idx]-clean[:,idx])/np.linalg.norm(clean[:,idx])),
                clean_wape=error(clean), clean_drop_wape=error(clean, True),
                held_wape=error(held), held_drop_wape=error(held, True))


def main():
    name = 'kdv_cossin'
    p = live.load_data(name)
    axes = live.R.axes_of(p)
    target, features, truth = live.R.parse(p.truth[0])
    terms, opaque = live.R.symbols(p, [target, *features])
    idx, = [i for i, term in enumerate(terms) if term.data == (('u', (0,3)),)]
    assert truth[idx-1] == -1.
    plans = [read_archive(directory(name, s, 'correlated'))['plan'] for s in SEEDS]
    rows = []
    for i, seed in enumerate(range(935811, 935823)):
        plan = {k: tuple(v) for k, v in plans[i % len(plans)].items()}
        bank = live.R.LocalBank(axes, terms, opaque, plan)
        clean = bank.matrix(p.data)
        for family in ('gaussian','correlated'):
            def draw(offset):
                return (p.noisy(5, seed+offset) if family == 'gaussian'
                        else measurement(family)(p, 5, seed+offset)[0])
            a, b, held = draw(0), draw(10000), bank.matrix(draw(20000))
            first, second = bank.matrix(a), bank.matrix(b)
            average = bank.matrix({k:(a[k]+b[k])*.5 for k in a})
            smoothed = average.copy()
            smoothed[:,idx], dt, dx = repeatable_column(first[:,idx], second[:,idx], plan['shape'])
            np.testing.assert_allclose(average[:,idx], (first[:,idx]+second[:,idx])*.5, rtol=1e-8, atol=1e-8)
            for label, mat, degree in [('single', first, ('','')), ('average', average, ('','')),
                                       ('cross_record', smoothed, (dt,dx))]:
                rows.append(dict(family=family, seed=seed, geometry_seed=SEEDS[i % len(SEEDS)],
                                 **record(mat, clean, idx, held, label, degree)))
    out = StringIO(newline='')
    writer = csv.DictWriter(out, fieldnames=rows[0], lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(ROOT / 'kdv_measurement.csv', out.getvalue())
    print(f'Completed {len(rows)} fits on 24 paired observations.')


if __name__ == '__main__':
    main()
