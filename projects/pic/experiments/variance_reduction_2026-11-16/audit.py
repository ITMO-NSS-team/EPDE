"""Audit stored results without regenerating fits or validation operators."""
import moments as M
from pathlib import Path
import csv
import hashlib
import json
import math

ROOT, np = M.ROOT, M.np


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite(value):
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, list):
        return all(finite(v) for v in value)
    return not isinstance(value, float) or math.isfinite(value)


def main():
    core = json.loads((ROOT / 'core_audit.json').read_text())
    summary = json.loads((ROOT / 'core_summary.json').read_text())
    checks = json.loads((ROOT / 'checks.json').read_text())
    invariants = json.loads((ROOT / 'invariant_checks.json').read_text())
    names = M.R.datasets.names('core')
    expected_files = {f'{name}_seed{seed}.json' for name in names for seed in range(3)}
    files = sorted((ROOT / 'results').glob('*_seed*.json'))
    assert {p.name for p in files} == expected_files
    assert set(core['result_sha256']) == expected_files
    sources = [ROOT / 'run_broad_validation.py', Path(M.__file__), Path(M.O.__file__),
               Path(M.Q.__file__), Path(M.B.__file__), Path(M.V.__file__)]
    source_hashes = {p.name: sha(p) for p in sources}
    sample_hashes, result_hashes = {}, {}
    cases, coefficient_rows, sample_arrays, whitening_matrices = 0, 0, 0, 0
    retained, old_retained = {}, 0
    primary_failures = []
    for path in files:
        data = json.loads(path.read_text())
        assert finite(data), path.name
        assert sha(path) == core['result_sha256'][path.name]
        assert data['source_sha256'] == source_hashes
        assert all(sha(Path(p)) == h for p, h in data['input_sha256'].items())
        joint_path = next(Path(p) for p in data['input_sha256'] if Path(p).parent.parent == M.J.ROOT)
        base_path = next(Path(p) for p in data['input_sha256'] if Path(p) != joint_path)
        joint, base = json.loads(joint_path.read_text()), json.loads(base_path.read_text())
        assert data['candidates'] == joint['candidates']
        assert data['instability'] == base['instability'] == joint['instability']
        for c, b in zip(data['candidates'], base['candidates'], strict=True):
            for key in ('id', 'slot', 'terms', 'target', 'registered', 'exact'):
                assert c[key] == b[key]
        idx = [g['indices'] for g in base['groups']]
        assert data['segment'] == [max(i[0] for i in idx), min(i[-1] for i in idx) + 1]
        assert not data['infeasible']
        assert set(data['rows']) == set(summary)
        for g, fold in enumerate(data['folds']):
            expected = M.V.catalog(data['candidates'], joint['folds'][g])
            for arm, rows in fold['arms'].items():
                for row, saved in zip(rows, expected, strict=True):
                    for key, value in saved.items():
                        assert row[key] == value, (path.name, g, arm, row['id'], key)
                    coefficient_rows += 1
        for bank in data['weighting'].values():
            for group in bank.values():
                for weight in group.values():
                    assert weight['feasible'] and weight['rank'] == weight['rows']
                    assert weight['whitened_covariance_max_error'] < 2e-5
                    whitening_matrices += 1
        old_local = json.loads((M.J.L.ROOT / 'results' / path.name).read_text())
        for slot in sorted({c['slot'] for c in data['candidates']}):
            cases += 1
            old_retained += bool(M.A.frontier(old_local['aggregates']['local_continuous']['cross']['scores'],
                                            data['instability'], data['candidates'], slot))
            for arm in summary:
                for kind in ('baseline', 'joint'):
                    keep = bool(M.A.frontier(data['aggregates'][arm][kind]['cross']['scores'],
                                            data['instability'], data['candidates'], slot))
                    retained[arm, kind] = retained.get((arm, kind), 0) + keep
                    if arm == 'q0_white128' and kind == 'joint' and not keep:
                        primary_failures.append(dict(system=data['system'], seed=data['seed'], slot=slot))
        sample_path = path.with_name(path.stem + '_samples.npz')
        with np.load(sample_path) as samples:
            assert set(samples.files) == {f'{arm}_{n}' for arm in summary for n in (64, 128)}
            frozen = None
            for arm, rows in data['rows'].items():
                current_frozen = np.array([r['frozen_clean_energy'] for r in rows])
                if frozen is None:
                    frozen = current_frozen
                else:
                    np.testing.assert_array_equal(current_frozen, frozen)
                for n in (64, 128):
                    sample = samples[f'{arm}_{n}']
                    assert sample.shape == (512, len(rows)) and np.isfinite(sample).all()
                    np.testing.assert_allclose(sample.std(axis=0, ddof=1), [r[f'sd{n}'] for r in rows],
                                               rtol=1e-13, atol=1e-15)
                    sample_arrays += 1
                for metric in ('cross', 'ordinary', 'clean_full', 'clean_subsample', 'frozen_clean_energy'):
                    average = np.mean([[r[metric] for r in f['arms'][arm]] for f in data['folds']], axis=0)
                    np.testing.assert_allclose(average, [r[metric] for r in rows], rtol=1e-13, atol=1e-15)
                for kind, aggregate in data['aggregates'][arm].items():
                    for metric, item in aggregate.items():
                        np.testing.assert_array_equal(item['scores'], [r[metric] for r in rows if r['kind'] == kind])
        result_hashes[path.name] = sha(path)
        sample_hashes[sample_path.name] = sha(sample_path)
    assert cases == core['equation_seed_cases'] == 60
    assert old_retained == core['old_local_retained'] == 54
    for (arm, kind), count in retained.items():
        assert count == summary[arm][kind]['retained']
    assert retained['q0_white128', 'joint'] == retained['q0_white64', 'joint'] == 56
    assert checks['real_basis_isometry'] and checks['feature_units_invariant']
    assert checks['zero_covariance_explicitly_infeasible']
    assert invariants['all_constants_preserved'] and invariants['maximum_derivative_order'] == 1
    assert len(invariants['rows']) == 12
    assert max(r['clean_energy'] for r in invariants['rows']) < 2e-12
    assert min(r['constant_deleted_energy'] for r in invariants['rows']) > 3.5
    margins = list(csv.DictReader((ROOT / 'core_primary_margins.csv').open()))
    assert {r['system'] for r in margins} == set(names)
    assert (ROOT / 'findings.txt').is_file() and (ROOT / 'validation_comparison.png').is_file()
    artifacts = {p.name: sha(p) for p in sorted(ROOT.iterdir()) if p.is_file() and p.name != 'completion_audit.json'}
    out = dict(completed_system_runs=len(files), equation_seed_cases=cases, noise_percent=5,
               old_local_retained=old_retained, primary_joint_retained=retained['q0_white128', 'joint'],
               primary_failures=primary_failures, all_result_numbers_finite=True,
               source_and_input_hashes_verified=True, candidate_panel_and_dyad_unchanged=True,
               coefficient_catalog_rows_exactly_verified=coefficient_rows,
               common_interior_geometry_verified=True, all_saved_arms_feasible=True,
               full_rank_weighting_matrices=whitening_matrices,
               finite_paired_sample_arrays=sample_arrays, paired_sample_rows_per_array=512,
               saved_score_means_and_sds_verified=True, frozen_fit_accuracy_identical_across_arms=True,
               registered_equivalent_retention_recomputed=True, ode_simple_identity_controls=12,
               independent_new_noise_records=0, new_evolutionary_searches=0,
               result_sha256=result_hashes, sample_sha256=sample_hashes, artifact_sha256=artifacts)
    (ROOT / 'completion_audit.json').write_text(json.dumps(out, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: v for k, v in out.items() if not k.endswith('_sha256')}, indent=2))


if __name__ == '__main__':
    main()
