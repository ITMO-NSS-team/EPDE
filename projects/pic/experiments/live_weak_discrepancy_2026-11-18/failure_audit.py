"""Replay signed and squared weak residuals on the two saved fronts and the reference."""
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stdout
import io
from types import SimpleNamespace
from unittest.mock import patch

import run
from live_score import LiveScore, R
from recheck import labels_of
import numpy as np

import epde.globals as global_var
from epde.operators.common.plain_fit import plain_params
from epde.operators.common.survival import dyad_scores


U = 'u{power: 1.0}'
UT = 'du/dx0{power: 1.0}'
UTT = 'd^2u/dx0^2{power: 1.0}'
UXX = 'd^2u/dx1^2{power: 1.0}'
U2 = 'u{power: 2.0}'
U3 = 'u{power: 3.0}'
REFERENCES = {
    'ode': f'-1 * {U} = {UTT}',
    'vdp': f'-1 * {U} + 0.2 * {UT} + -0.2 * {U2} * {UT} = {UTT}',
    'wave': f'0.04 * {UXX} = {UTT}',
    'ac': f'5 * {U} + -5 * {U3} + 0.0001 * {UXX} = {UT}',
}


def get_rows(record, raw, grids):
    settings = dict(record['settings'])
    settings['max_deriv_order'] = tuple(settings['max_deriv_order'])
    search = run.U.EpdeSearch(**settings)
    _, domain = search.createDomain(tuple(grids), ID=0)
    search.set_preprocessor(default_preprocessor_type='FD', preprocessor_kwargs={'accuracy': 4})
    _, traj = search.createTrajectory(raw, domain, cache_id=0)
    families = [run.U.TrigonometricTokens(freq_from_data=2),
                run.U.GridTokens(labels=['t', 'x'][:len(grids)], max_power=2)]
    with patch.object(search, '_run_optimization'):
        search.fit(data=[traj], additional_tokens=families)
    assert plain_params()['plain_target'] and plain_params()['target_axes'] is None
    return global_var.samples_manager.weak


def native_dyad_components(rows, labels, target):
    fields = rows.fields[0]
    plan = rows.plans[0, 'u']
    problem = SimpleNamespace(shape=fields.shape, grids=np.meshgrid(*fields.axes, indexing='ij'),
                              data=fields.values, named_arrays={})
    terms, opaque = R.symbols(problem, labels)
    matrix, shape = R.WF.weak_matrix(
        [t.monomials(psi_cap=plan.q) for t in terms], [t.opaque for t in terms],
        R.WF.Fields(fields.values, fields.axes, opaque), plan.cut, plan.ms, plan.ss, plan.q)
    if () in labels:
        matrix[:, labels.index(())] = 1
    yi = labels.index(target)
    X = np.delete(matrix, yi, axis=1)
    return dyad_scores(X, matrix[:, yi], np.ones(len(matrix)), shape, fit_intercept=False)


def weak_scores(scorer, labels):
    signed, energy = [], []
    for left, right, scales, vector in scorer.fold_fits(labels):
        unit = vector / np.linalg.norm(scales * vector)
        a, b = left @ unit, right @ unit
        signed.append(a @ b)
        energy.append((a @ a + b @ b) / 2)
    if len(signed) != 3:
        return None
    return {'signed': float(np.mean(signed)), 'energy': float(np.mean(energy))}


def clean_check(system, axes, clean, orders):
    """Best normalized implicit residual on the clean interior, allowing every coefficient to refit."""
    if system not in ('vdp', 'wave'):
        return {}
    supports = {'reference': REFERENCES[system]}
    if system == 'vdp':
        supports.update(harmonic=f'-1 * {U} = {UTT}', linear_damping=f'-1 * {U} + 0.14 * {UT} = {UTT}')
    else:
        supports.update(cubic_space=f'1 * {U3} = {UXX}', affine_space=f'1 * {U} + 1 = {UXX}',
                        time_weighted=f'1 * t{{power: 2.0, dim: 0.0}} * {UXX} = {UT}')
    problem = SimpleNamespace(shape=clean['u'].shape, grids=np.meshgrid(*axes, indexing='ij'),
                              data=clean, named_arrays={})
    interior = tuple(slice(R.WF.FD_REACH * order, -R.WF.FD_REACH * order) for order in orders)
    residuals = {}
    for name, equation in supports.items():
        terms, opaque = R.symbols(problem, labels_of(equation))
        fields = R.WF.Fields(clean, axes, opaque)
        matrix = np.column_stack([fields.point(t.data, t.coords, t.opaque)[interior].ravel() for t in terms])
        scaled = matrix / np.linalg.norm(matrix, axis=0)
        residuals[name] = float(np.linalg.svd(scaled, compute_uv=False)[-1])
    result = {'clean_relative_residual': residuals}
    if system == 'wave':
        field = clean['u']
        energy = np.linalg.svd(field - field.mean(axis=1, keepdims=True), compute_uv=False) ** 2
        result['leading_spatial_profile_fraction'] = float(energy[0] / energy.sum())
    return result


def audit(case):
    system, seed = case
    def load(arm):
        path = run.ROOT / 'plain_results' / f'{system}_{seed}_{arm}.json'
        return json.loads(path.read_text())
    weak, wape = load('weak'), load('wape')
    axes, clean = run.U.load(system)
    rng = np.random.default_rng([seed, list(run.U.TRUTH).index(system)])
    raw = {v: y + weak['noise'] * y.std() * rng.standard_normal(y.shape) for v, y in clean.items()}
    hashes = {v: hashlib.sha256(y.tobytes()).hexdigest() for v, y in raw.items()}
    assert hashes == weak['raw_sha256'] == wape['raw_sha256']
    grids = np.meshgrid(*axes, indexing='ij')
    problem = SimpleNamespace(shape=raw['u'].shape, grids=grids, data=raw, named_arrays={})
    scorer = LiveScore(problem, raw, seed, weak['settings']['max_deriv_order'], 7)
    with redirect_stdout(io.StringIO()):
        rows = get_rows(weak, raw, grids)
    truth = labels_of(REFERENCES[system])
    by_target = {}
    target_dyads = {}
    for target in truth:
        if len(target) != 1 or '/dx' not in target[0][0] or dict(target[0][1])['power'] != 1:
            continue
        components = native_dyad_components(rows, truth, target)
        by_target[target[0][0]] = list(zip([label for label in truth if label != target], components))
        target_dyads[target[0][0]] = float(np.sum(components))
    reference = dict(weak_scores(scorer, truth), dyad=min(target_dyads.values()))
    np.testing.assert_allclose(reference['signed'], weak['evaluated_laws']['exact'], rtol=1e-7, atol=1e-10)
    replayed = {}
    candidates = []
    unknown = []
    for arm, record in [('weak', weak), ('wape', wape)]:
        for row in record['front']:
            labels = labels_of(row['equation'])
            if labels not in replayed:
                replayed[labels] = weak_scores(scorer, labels)
            scores = replayed[labels]
            if scores is None:
                unknown.append(dict(arm=arm, equation=row['equation']))
                continue
            if arm == 'weak':
                np.testing.assert_allclose(scores['signed'], row['objectives'][0], rtol=1e-6, atol=1e-10)
                if row['distance'] == weak['min_distance']:
                    target, = labels_of('0 = ' + row['equation'].split('=')[1])
                    np.testing.assert_allclose(np.sum(native_dyad_components(rows, labels, target)),
                                               row['objectives'][1], rtol=1e-5, atol=1e-8)
            if labels != truth:
                candidates.append(dict(scores, dyad=row['objectives'][1], equation=row['equation'],
                                       terms=row['terms'], arm=arm, distance=row['distance']))
    counts = {}
    for name, pool in [('weak', [c for c in candidates if c['arm'] == 'weak']), ('union', candidates)]:
        counts[name] = {}
        for metric in ('signed', 'energy'):
            dominators = [c for c in pool if c[metric] <= reference[metric] and c['dyad'] <= reference['dyad']
                          and (c[metric] < reference[metric] or c['dyad'] < reference['dyad'])]
            counts[name][metric] = len({c['equation'] for c in dominators})
    return dict(system=system, seed=seed, reference=reference, reference_dyads=target_dyads,
                reference_dyad_components=by_target,
                distinct_supports=len(replayed), unresolved=unknown, dominator_counts=counts,
                candidates=candidates, clean_diagnostic=clean_check(system, axes, clean,
                                                                      weak['settings']['max_deriv_order']))


if __name__ == '__main__':
    cases = [(system, seed) for system in REFERENCES for seed in range(73001, 73004)]
    with ProcessPoolExecutor(max_workers=3) as pool:
        records = list(pool.map(audit, cases))
    (run.ROOT / 'failure_audit.json').write_text(json.dumps(records, indent=1, allow_nan=False), encoding='utf-8')
    for record in records:
        print(record['system'], record['seed'], record['reference'], record['dominator_counts'], flush=True)
