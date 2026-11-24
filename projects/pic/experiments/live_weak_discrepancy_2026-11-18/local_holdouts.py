"""Paired fixed-coefficient readouts on the archived Pareto rivals and fresh observations."""
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from types import SimpleNamespace
from unittest.mock import patch

import run
import live_score as L
from failure_audit import REFERENCES, U, U3, UT, UTT, UXX
from recheck import labels_of
import numpy as np


REPEATS = 32
RIVALS = {
    'ode': {},
    'vdp': {'harmonic': f'-1 * {U} = {UTT}', 'linear_damping': f'-1 * {U} + 1 * {UT} = {UTT}'},
    'wave': {'cubic_space': f'1 * {U3} = {UXX}', 'affine_space': f'1 * {U} + 1 = {UXX}',
             'time_weighted': f'1 * t{{power: 2.0, dim: 0.0}} * {UXX} = {UT}'},
    'ac': {'reaction': f'5 * {U} + -5 * {U3} = {UT}'},
}


class LocalWindow(L.Window):
    """Unmodulated half-span windows with the same moment count as the broad bank."""
    def __init__(self, axes, orders, width):
        self.axes = axes
        self.q = [3] + [5] * (len(axes) - 1)
        k = 1
        while (2 * k + 1) ** len(axes) < width + 1:
            k += 1
        self.shape = (2 * k + 1,) * len(axes)
        self.basis = SimpleNamespace(size=int(np.prod(self.shape)))
        self.kernels = []
        self.geometry = []
        for d, (axis, order, q) in enumerate(zip(axes, orders, self.q)):
            cut = int(order) * L.R.WF.FD_REACH
            span = len(axis) - 1 - 2 * cut
            radius = span // 4
            centers = np.rint(np.linspace(cut + radius, len(axis) - 1 - cut - radius, self.shape[d])).astype(int)
            if radius < q or len(np.unique(centers)) != len(centers):
                raise ValueError('Insufficient local quadrature nodes')
            kernels, _ = L.Q.axis_weights(axis, cut, centers, radius, q, np.zeros(1), q, 3 if d == 0 else 5)
            np.testing.assert_allclose(kernels[0].sum(axis=1), 1, atol=1.e-12)
            self.kernels.append(kernels)
            self.geometry.append({'radius': radius, 'centers': centers.tolist()})

    def real(self, moments):
        return moments


def fixture(system, null):
    axes, clean = run.U.load(system)
    if null:
        t = axes[0][:, None]
        initial = clean['u'][:1]
        denominator = 1 + initial**2 * np.expm1(10*t)
        field = initial * np.exp(5*t) / np.sqrt(denominator)
        deriv = 5 * initial * np.exp(5*t) * (1 - initial**2) / denominator**1.5
        np.testing.assert_allclose(deriv, 5 * field * (1 - field**2), atol=2e-14)
        clean = {'u': field}
    return axes, clean


def support_panel(record, null):
    system = record['system']
    rows = {}
    def add(equation, role=None, dyad=None):
        key = labels_of(equation)
        row = rows.setdefault(key, {'equation': equation, 'roles': [], 'dyad': None})
        if role:
            row['roles'].append(role)
        if dyad is not None:
            row['dyad'] = dyad if row['dyad'] is None else min(dyad, row['dyad'])

    add(REFERENCES[system], 'reference', None if null else record['reference']['dyad'])
    for name, equation in RIVALS[system].items():
        add(equation, name)
    if not null:
        for candidate in record['candidates']:
            if candidate['dyad'] <= record['reference']['dyad']:
                add(candidate['equation'], dyad=candidate['dyad'])
    return rows


def observed_columns(scorer, terms, opaque, fields, correct_noise):
    """Common bank matrices for all candidates, indexed by interlaced observation group."""
    matrices = []
    for g, (_, held, _) in enumerate(scorer.folds):
        covariance = held.covariance
        mapped = [L.O.map_values(covariance.time_map, {v: y[g::3] for v, y in raw.items()}) for raw in fields]
        stacked = {v: np.stack([m[v] for m in mapped]) for v in mapped[0]}
        measure = L.Measurements(scorer.window, stacked, covariance if correct_noise else None)
        matrices.append(np.stack([measure.column(term, opaque) for term in terms], axis=-1))
    return matrices


def score_rows(scorer, matrices, terms, opaque, label_index, fits):
    """One homogeneous coefficient vector per training fold, fixed over all test windows and repeats."""
    sums = {key: np.zeros(matrices[0].shape[0]) for key in fits}
    for g, (train, _, white) in enumerate(scorer.folds):
        measure = L.Measurements(scorer.window, {v: field[0] for v, field in train.values.items()})
        pilot = np.column_stack([measure.column(term, opaque) for term in terms])
        scales = np.linalg.norm(white @ pilot, axis=0)
        held = [white @ matrices[h] for h in range(3) if h != g]
        for labels, vectors in fits.items():
            idx = [label_index[label] for label in labels]
            vector = vectors[g]
            norm = np.linalg.norm(scales[idx] * vector)
            if not np.isfinite(norm) or norm == 0:
                raise ValueError('Undefined training normalization')
            a, b = [m[:, :, idx] @ (vector / norm) for m in held]
            sums[labels] += np.einsum('br,br->b', a, b) / 3
    return sums


def study(job):
    record, null = job
    system, seed = record['system'], record['seed']
    axes, clean = fixture(system, null)
    code = list(run.U.TRUTH).index(system)
    def noisy(entropy):
        rng = np.random.default_rng(entropy)
        return {v: y + .05*y.std()*rng.standard_normal(y.shape) for v, y in clean.items()}
    raw = noisy([seed, code])
    source = json.loads((run.ROOT / 'plain_results' / f'{system}_{seed}_weak.json').read_text())
    if not null:
        assert hashlib.sha256(raw['u'].tobytes()).hexdigest() == source['raw_sha256']['u']
    fresh = [noisy([834271, seed, code, i]) for i in range(REPEATS)]
    panel = support_panel(record, null)
    orders = source['settings']['max_deriv_order']
    problem = SimpleNamespace(shape=raw['u'].shape, grids=np.meshgrid(*axes, indexing='ij'), data=raw, named_arrays={})
    broad = L.LiveScore(problem, raw, seed, orders, 7)
    fits = {labels: [vector for _, _, _, vector in broad.fold_fits(labels)] for labels in panel}
    assert all(len(vectors) == 3 for vectors in fits.values())
    with patch.object(L, 'Window', LocalWindow):
        local = L.LiveScore(problem, raw, seed, orders, 7)
    labels = sorted(set().union(*panel))
    terms, opaque = L.R.symbols(problem, labels)
    opaque = {k: v[broad.segment] for k, v in opaque.items()}
    label_index = {label: i for i, label in enumerate(labels)}
    for name, scorer in [('broad', broad), ('local', local)]:
        observations = observed_columns(scorer, terms, opaque, [raw, *fresh], True)
        scores = score_rows(scorer, observations, terms, opaque, label_index, fits)
        clean_cols = observed_columns(scorer, terms, opaque, [clean], False)
        clean_scores = score_rows(scorer, clean_cols, terms, opaque, label_index, fits)
        for key, row in panel.items():
            if name == 'broad':
                np.testing.assert_allclose(scores[key][0], broad.support_score(key), rtol=1e-7, atol=1e-9)
            assert np.all(np.isfinite(scores[key]))
            row[name] = {'original': float(scores[key][0]), 'fresh': scores[key][1:].tolist(),
                         'clean': float(clean_scores[key][0])}
    if not null:
        for candidate in record['candidates']:
            labels = labels_of(candidate['equation'])
            if labels in panel:
                np.testing.assert_allclose(panel[labels]['broad']['original'], candidate['signed'], rtol=1e-7, atol=1e-9)
    print(system, seed, 'null' if null else 'positive', len(panel), flush=True)
    return dict(system=system, seed=seed, null=null, local_geometry=local.window.geometry,
                unweighted={'local': local.unweighted_folds, 'broad': broad.unweighted_folds}, panel=list(panel.values()))


def summarize(records):
    totals = {}
    for record in records:
        name = record['system'] + ('_zero' if record['null'] else '')
        panel = record['panel']
        reference, = (row for row in panel if 'reference' in row['roles'])
        for method in ('broad', 'local'):
            summary = totals.setdefault(name, {}).setdefault(method, {})
            for kind in ('original', 'fresh', 'clean'):
                r = np.atleast_1d(reference[method][kind])
                if not record['null']:
                    survives = np.ones(r.shape, dtype=bool)
                    for rival in panel:
                        if rival is reference or rival['dyad'] is None:
                            continue
                        x = np.atleast_1d(rival[method][kind])
                        dominates = (x <= r) & ((x < r) | (rival['dyad'] < reference['dyad']))
                        survives &= ~dominates
                    label = 'front_' + kind
                    summary[label] = summary.get(label, 0) + int(survives.sum())
                for rival in panel:
                    for role in rival['roles']:
                        if role == 'reference':
                            continue
                        label = role + '_' + kind
                        count = int((r < np.atleast_1d(rival[method][kind])).sum())
                        summary[label] = summary.get(label, 0) + count
    return totals


if __name__ == '__main__':
    audit = json.loads((run.ROOT / 'failure_audit.json').read_text())
    jobs = [(record, False) for record in audit] + [(record, True) for record in audit if record['system'] == 'ac']
    with ProcessPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(study, jobs))
    summary = summarize(results)
    (run.ROOT / 'local_holdouts.json').write_text(json.dumps({'summary': summary, 'records': results}, allow_nan=False))
    print(json.dumps(summary, indent=1))
