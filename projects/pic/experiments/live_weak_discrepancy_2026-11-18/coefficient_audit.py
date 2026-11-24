"""Cross reference and rival coefficient sources on independent observations and fixed dyad values."""
import hashlib
import io
import json
import re
from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch

import local_holdouts as H
from failure_audit import get_rows
from epde.operators.common.subset_selection import subset_coefficients
from recheck import labels_of
import numpy as np


def equation_vector(equation):
    labels = labels_of(equation)
    lhs, rhs = equation.split('=')
    pivot, = labels_of('0 = ' + rhs)
    vector = np.zeros(len(labels))
    vector[labels.index(pivot)] = -1
    for term in re.split(r'(?<![eE])\+', lhs):
        weight = float(term.split('*')[0].strip())
        if weight:
            parsed = labels_of(term + ' = ' + rhs)
            feature, = (label for label in parsed if label != pivot)
            vector[labels.index(feature)] += weight
    return labels, pivot, vector


def native_columns(rows, problem, labels):
    fields = rows.fields[0]
    native_problem = SimpleNamespace(shape=fields.shape, grids=np.meshgrid(*fields.axes, indexing='ij'),
                                     data=fields.values, named_arrays=problem.named_arrays)
    terms, opaque = H.L.R.symbols(native_problem, labels)
    plan = rows.plans[0, 'u']
    matrix, _ = H.L.R.WF.weak_matrix(
        [t.monomials(psi_cap=plan.q) for t in terms], [t.opaque for t in terms],
        H.L.R.WF.Fields(fields.values, fields.axes, opaque), plan.cut, plan.ms, plan.ss, plan.q)
    if () in labels:
        matrix[:, labels.index(())] = 1
    return matrix


def native_fit(matrix, labels, pivot):
    features = [j for j, label in enumerate(labels) if label and label != pivot]
    yi = labels.index(pivot)
    coefs = subset_coefficients(matrix[:, features], matrix[:, yi], np.ones(len(matrix)),
                                [True] * len(features) + [() in labels])
    vector = np.zeros(len(labels))
    vector[yi] = -1
    vector[features] = coefs[:-1]
    if () in labels:
        vector[labels.index(())] = coefs[-1]
    return vector


def native_training_matrices(source, raw, axes, problem, labels):
    """Rebuild native weak rows using only each training group's observed grid."""
    matrices = []
    geometry = []
    for g in range(3):
        part = {v: field[g::3] for v, field in raw.items()}
        grid = np.meshgrid(axes[0][g::3], *axes[1:], indexing='ij')
        floor = 'native'
        with redirect_stdout(io.StringIO()):
            try:
                rows = get_rows(source, part, grid)
            except ValueError as err:
                if 'no feasible weak-form window' not in str(err):
                    raise
                # Only coefficients are estimated on this short group. The original full-data
                # dyad remains fixed, so its two-slab row minimum is unnecessary for this fit.
                def wls_floor(metric, shape, width):
                    return None if np.prod(shape) >= width + 1 else 'insufficient rows for WLS'
                with patch('epde.operators.common.survival.row_floor', wls_floor):
                    rows = get_rows(source, part, grid)
                floor = 'wls'
        plan = rows.plans[0, 'u']
        matrices.append(native_columns(rows, problem, labels))
        geometry.append(dict(floor=floor, radius=list(plan.ms), shape=list(plan.shape)))
    return matrices, geometry


def unit_column_tls(matrix):
    scales = np.linalg.norm(matrix, axis=0)
    _, _, vh = np.linalg.svd(matrix / scales, full_matrices=False)
    return vh[-1] / scales


def in_physical_units(labels, pivot, vec):
    return (-vec / vec[labels.index(pivot)]).tolist()


def readouts(scorer, terms, opaque, raw_fields):
    """Concatenate fresh and clean measurements. Clean columns have no noise subtraction."""
    observations = H.observed_columns(scorer, terms, opaque, raw_fields[:-1], True)
    clean_cols = H.observed_columns(scorer, terms, opaque, raw_fields[-1:], False)
    matrices = [np.concatenate([a, b]) for a, b in zip(observations, clean_cols)]
    folds = []
    for g, (train, _, white) in enumerate(scorer.folds):
        measure = H.L.Measurements(scorer.window, {v: values[0] for v, values in train.values.items()})
        pilot = np.column_stack([measure.column(t, opaque) for t in terms])
        a, b = [white @ matrices[h] for h in range(3) if h != g]
        folds.append((np.linalg.norm(white @ pilot, axis=0), a, b))
    return folds


def discrepancy(folds, idx, vectors):
    values = []
    for (scales, left, right), vec in zip(folds, vectors):
        unit = vec / np.linalg.norm(scales[idx] * vec)
        a, b = left[:, :, idx] @ unit, right[:, :, idx] @ unit
        values.append(np.sum(a*b, axis=1))
    scores = np.mean(values, axis=0)
    assert np.isfinite(scores).all()
    return scores


def study(job):
    audit, holdout = job
    system, seed, null = holdout['system'], holdout['seed'], holdout['null']
    axes, clean = H.fixture(system, null)
    code = list(H.run.U.TRUTH).index(system)
    def noisy(entropy):
        rng = np.random.default_rng(entropy)
        return {v: field + .05 * field.std() * rng.standard_normal(field.shape) for v, field in clean.items()}
    source = json.loads((H.run.ROOT / 'plain_results' / f'{system}_{seed}_weak.json').read_text())
    raw = noisy([seed, code])
    if not null:
        assert hashlib.sha256(raw['u'].tobytes()).hexdigest() == source['raw_sha256']['u']
    fresh = [noisy([834271, seed, code, i]) for i in range(H.REPEATS)]
    problem = SimpleNamespace(shape=raw['u'].shape, grids=np.meshgrid(*axes, indexing='ij'), data=raw, named_arrays={})
    with redirect_stdout(io.StringIO()):
        rows = get_rows(source, raw, problem.grids)
    panel = {labels_of(row['equation']): row for row in holdout['panel']}
    columns = sorted(set().union(*panel))
    index = {label: j for j, label in enumerate(columns)}
    native = native_columns(rows, problem, columns)
    fold_native, geometry = native_training_matrices(source, raw, axes, problem, columns)
    broad = H.L.LiveScore(problem, raw, seed, source['settings']['max_deriv_order'], 7)
    with patch.object(H.L, 'Window', H.LocalWindow):
        local = H.L.LiveScore(problem, raw, seed, source['settings']['max_deriv_order'], 7)
    terms, opaque = H.L.R.symbols(problem, columns)
    opaque = {k: arr[broad.segment] for k, arr in opaque.items()}
    banks = {name: readouts(scorer, terms, opaque, [*fresh, clean]) for name, scorer in [('broad', broad), ('local', local)]}

    ref_labels, physical_target, physical = equation_vector(H.REFERENCES[system])
    if null:
        diffusion, = labels_of('0 = ' + H.UXX)
        physical[ref_labels.index(diffusion)] = 0
    target_name = physical_target[0][0] if null else min(audit['reference_dyads'], key=audit['reference_dyads'].get)
    best_target, = (label for label in ref_labels if len(label) == 1 and label[0][0] == target_name)
    eligible = [] if null else [c for c in audit['candidates'] if c['dyad'] <= audit['reference']['dyad']]
    archive = {}
    for row in eligible:
        archive.setdefault(labels_of(row['equation']), {})[row['equation']] = row['dyad']
    coefficient_info = {}
    results = []
    max_replay_error = 0.
    for labels, row in panel.items():
        idx = [index[label] for label in labels]
        matrix = native[:, idx]
        tls = unit_column_tls(matrix)
        fold_tls = [unit_column_tls(m[:, idx]) for m in fold_native]
        implicit = [v for _, _, _, v in broad.fold_fits(labels)]
        assert len(implicit) == 3
        is_ref = labels == ref_labels
        variants = archive.get(labels, {row['equation']: row['dyad']})
        for equation, dyad in variants.items():
            _, pivot, printed = equation_vector(equation)
            if is_ref:
                pivot = best_target
            wls = native_fit(matrix, labels, pivot)
            if labels in archive:
                max_replay_error = max(max_replay_error, float(np.max(np.abs(printed-wls)/(1+np.abs(wls)))))
                np.testing.assert_allclose(wls, printed, rtol=3e-5, atol=1e-7)
                wls = printed
            alternatives = {'implicit': implicit, 'native': [wls]*3, 'native_tls': [tls]*3,
                            'fold_wls': [native_fit(m[:, idx], labels, pivot) for m in fold_native],
                            'fold_tls': fold_tls}
            if is_ref:
                alternatives['physical'] = [physical]*3
                coefficient_info = {'labels': labels, 'pivot': physical_target, 'wls_target': pivot,
                                    'vectors': {key: [in_physical_units(labels, physical_target, v) for v in vecs]
                                                for key, vecs in alternatives.items()}}
            scores = {}
            for name, folds in banks.items():
                scores[name] = {key: discrepancy(folds, idx, vecs).tolist() for key, vecs in alternatives.items()}
                np.testing.assert_allclose(scores[name]['implicit'], [*row[name]['fresh'], row[name]['clean']],
                                           rtol=1e-7, atol=1e-10)
            results.append(dict(equation=equation, dyad=dyad, roles=row['roles'], scores=scores))
    print(system, seed, 'zero' if null else 'positive', len(results), flush=True)
    return dict(system=system, seed=seed, null=null, coefficients=coefficient_info, fold_geometry=geometry,
                printed_replay_max_scaled_error=max_replay_error, results=results)


def summarize(records):
    summary = {}
    for record in records:
        name = record['system'] + ('_zero' if record['null'] else '')
        ref, = (r for r in record['results'] if 'reference' in r['roles'])
        others = [r for r in record['results'] if r is not ref]
        for bank in ('broad', 'local'):
            out = summary.setdefault(name, {}).setdefault(bank, {})
            methods = ('implicit', 'native', 'native_tls', 'fold_wls', 'fold_tls')
            for rival_fit in methods:
                for ref_fit in (*methods, 'physical'):
                    reference = np.array(ref['scores'][bank][ref_fit])
                    wins = np.ones(len(reference), dtype=bool)
                    for rival in others:
                        if rival['dyad'] is None:
                            continue
                        scores = np.array(rival['scores'][bank][rival_fit])
                        wins &= ~((scores <= reference) & ((scores < reference) | (rival['dyad'] < ref['dyad'])))
                    totals = out.setdefault(ref_fit + '/' + rival_fit, {})
                    if not record['null']:
                        counts = totals.setdefault('front', [0, 0])
                        counts[0] += int(wins[:-1].sum())
                        counts[1] += int(wins[-1])
                    for role in H.RIVALS[record['system']]:
                        variants = [r for r in others if role in r['roles']]
                        # The named diagnostic uses the most competitive stored orientation for each recording.
                        scores = np.min([r['scores'][bank][rival_fit] for r in variants], axis=0)
                        counts = totals.setdefault(role, [0, 0])
                        counts[0] += int((reference[:-1] < scores[:-1]).sum())
                        counts[1] += int(reference[-1] < scores[-1])
    return summary


if __name__ == '__main__':
    audits = {(r['system'], r['seed']): r for r in json.loads((H.run.ROOT/'failure_audit.json').read_text())}
    holdouts = json.loads((H.run.ROOT/'local_holdouts.json').read_text())['records']
    jobs = [(audits[r['system'], r['seed']], r) for r in holdouts]
    with ProcessPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(study, jobs))
    summary = summarize(results)
    (H.run.ROOT/'coefficient_audit.json').write_text(json.dumps({'summary': summary, 'records': results}, allow_nan=False))
    for name, banks in summary.items():
        print(name, {m: banks['broad'][m] for m in ('implicit/implicit', 'native/native', 'physical/implicit',
                                                     'physical/native', 'native_tls/native_tls')})
