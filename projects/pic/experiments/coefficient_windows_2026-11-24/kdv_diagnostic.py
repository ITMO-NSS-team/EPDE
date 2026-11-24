"""Isolate dispersion-column error after the fresh-noise window choices are frozen."""
import json
from pathlib import Path
import sys
from statistics import median

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent/'complementary_windows_2026-11-24'))
from cpu_limit import cap_worker

if __name__ == '__main__':
    cap_worker()

from run import R, WF, np, split
from selector import Geometry
from study import wls


def coefficient(bank, column):
    vector, status = wls(bank, 0)
    assert status == 'ok'
    return float(-vector[column]/vector[0])


def main():
    cases = []
    problem = R.datasets.load('kdv_cossin')
    panel, labels = R.panel(problem)
    ref, = [c for c in panel if c['label'] == 'reference']
    terms, opaque = R.symbols(problem, [labels[j] for j in [ref['yi'], *ref['xi']]])
    col, = [i for i, term in enumerate(terms) if term.data == (('u', (0, 3)),)]
    assert ref['exact'][col-1] == -1.
    _, (cuts, axes), _ = split(R.axes_of(problem))
    opaque = {k: v[cuts] for k, v in opaque.items()}

    def fields(raw):
        return WF.Fields({k: v[cuts] for k, v in raw.items()}, axes, opaque)

    clean = fields(problem.data)
    for seed in range(926301, 926307):
        case = json.loads((ROOT/'results'/f'kdv_cossin_{seed}.json').read_text())
        noisy = fields(problem.noisy(5, seed))
        plan = case['plan']
        base = Geometry(plan, clean.shape, plan['ms'], (0, 0))
        clean_base, noisy_base = [base.matrix(f, terms) for f in (clean, noisy)]
        for arm in ('native', 'complementary'):
            second = case['second'] if arm == 'complementary' else None
            a, b = clean_base, noisy_base
            if second:
                geom = Geometry(plan, clean.shape, second['ms'], second['shift'])
                a, b = geom.matrix(clean, terms, a), geom.matrix(noisy, terms, b)
            # Target and remaining predictors stay together when swapping one column.
            clean_disp, noisy_disp = b.copy(), a.copy()
            clean_disp[:, col], noisy_disp[:, col] = a[:, col], b[:, col]
            fitted = dict(noisy=coefficient(b, col), clean_dispersion_only=coefficient(clean_disp, col),
                          noisy_dispersion_only=coefficient(noisy_disp, col), clean=coefficient(a, col))
            np.testing.assert_allclose(fitted['noisy'], case[arm][0]['reference']['weights'][col])
            clean_others = a[:, [i for i in range(1, a.shape[1]) if i != col]]
            # Remove the component explainable by the other physical predictors before comparing energies.
            def residual(v):
                return v - clean_others @ np.linalg.lstsq(clean_others, v, rcond=None)[0]
            ratio = np.linalg.norm(residual(b[:, col]-a[:, col]))/np.linalg.norm(residual(a[:, col]))
            cases.append(dict(seed=seed, arm=arm, coefficients=fitted, noise_to_dispersion_signal=float(ratio)))
    summary = {}
    for arm in ('native', 'complementary'):
        subset = [c for c in cases if c['arm'] == arm]
        summary[arm] = {key: median(c['coefficients'][key] for c in subset) for key in subset[0]['coefficients']}
        summary[arm]['noise_to_dispersion_signal'] = median(c['noise_to_dispersion_signal'] for c in subset)
    (ROOT/'kdv_diagnostic.json').write_text(json.dumps(dict(summary=summary, cases=cases), indent=2, allow_nan=False))
    print(summary)


if __name__ == '__main__':
    main()
