"""Verify frozen sources, fresh records, numerical outputs and summaries."""
import run_eiv as V
E, R, np, ROOT = V.E, V.R, V.np, V.ROOT
from pathlib import Path
import json, hashlib
from collections import Counter


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    files = sorted((ROOT/'results').glob('*.json'))
    assert {p.stem for p in files} == {f'{name}_seed{seed}' for name in ('ac','kdv_cossin','wave','ks') for seed in (3,4,5)}
    current = V.source_hashes(); statuses = Counter(); checked_fits = 0; iteration_count = 0
    raw_hashes = {}; geometry_inputs = {}; registered_cases = 0; result_hashes = {}
    for path in files:
        d = json.loads(path.read_text())
        assert d['source_sha256'] == current
        assert all(sha(Path(p)) == h for p,h in d['input_sha256'].items())
        assert d['seed'] in (3,4,5) and d['geometry_seed']==0 and d['draws']==512
        p = R.datasets.load(d['system']); raw = p.noisy(5, d['seed'])
        assert d['raw_data_sha256'] == {v:R.digest(y) for v,y in raw.items()}
        assert d['clean_data_sha256'] == {v:R.digest(y) for v,y in p.data.items()}
        _, old = V.F.baseline(d['system'],0)
        for g, group in enumerate(d['groups']):
            idx = old['groups'][g]['indices']
            assert group['observed_sha256'] == {v:R.digest(y[idx]) for v,y in raw.items()}
            assert group['weight_meta']['feasible']
        for other_seed in (0,1,2):
            previous = p.noisy(5,other_seed)
            assert any(R.digest(previous[v]) != h for v,h in d['raw_data_sha256'].items())
        raw_hashes[path.stem] = d['raw_data_sha256']
        assert all(c['registered'] for c in d['candidates'])
        registered_cases += len(d['candidates'])
        for f in d['folds']:
            assert np.array_equal(f['local_natural_indices'],V.natural_rows(f['local_plan']))
            for row in f['rows']:
                status = row['fit']['status']; statuses[row['method'],status] += 1
                if row['coefficients'] is None:
                    assert status != 'ok'
                    assert 'frozen_clean_energy' not in row
                else:
                    assert np.isfinite(row['coefficients']).all()
                    for metric in ('cross','ordinary','plain_cross','clean_validation','frozen_clean_energy','relative_coefficient_error'):
                        assert np.isfinite(row[metric]), (path.name,row['bank'],row['method'],metric)
                    checked_fits += 1
                if row['method']=='raw_irls':
                    for step in row['fit']['trace']:
                        iteration_count += 1
                        assert step['covariance_rank']==step['rows']
                        assert step['covariance_eigen_ratio'] > E.RCOND
                        assert np.isfinite(step['prediction_change']) and np.isfinite(step['weighted_loss'])
                    if status=='ok':
                        assert row['fit']['trace'][-1]['prediction_change']<=1e-8
                        np.testing.assert_array_equal(row['coefficients'],row['fit']['trace'][-1]['coefficients'])
                    elif status=='nonconverged':
                        assert row['fit']['iterations']==100 and row['coefficients'] is None
        for aggregate in d['aggregates']:
            rr = [r for f in d['folds'] for r in f['rows'] if all(r[k]==aggregate[k] for k in ('bank','method','candidate'))]
            assert len(rr)==3
            good = [r for r in rr if r['coefficients'] is not None and r['fit']['status']=='ok']
            assert len(good)==aggregate['successful_folds']
            if len(good)==3:
                for metric in ('cross','ordinary','plain_cross','clean_validation','frozen_clean_energy','relative_coefficient_error'):
                    np.testing.assert_allclose(aggregate[metric],np.mean([r[metric] for r in good]),rtol=1e-14,atol=1e-15)
            else:
                assert 'frozen_clean_energy' not in aggregate
        result_hashes[path.name] = sha(path)
        geometry_inputs.update(d['input_sha256'])
    checks=json.loads((ROOT/'checks.json').read_text()); invariants=json.loads((ROOT/'invariant_checks.json').read_text())
    assert checks['known_covariance_gls'] and checks['joint_feature_units_invariant'] and checks['joint_covariance_algebra']
    assert checks['zero_covariance_explicitly_infeasible'] and invariants['all_constants_preserved']
    assert len(invariants['rows'])==3 and max(r['clean_energy'] for r in invariants['rows'])<1e-8
    assert (ROOT/'findings.txt').exists()
    out = dict(system_runs=len(files), reference_seed_cases=12, registered_equation_seed_cases=registered_cases,
               fresh_noise_seeds=[3,4,5], frozen_geometry_seed=0, frozen_source_and_input_hashes_verified=True,
               fresh_raw_and_clean_data_hashes_verified=True, natural_row_subsets_verified=True,
               finite_evaluated_fits=checked_fits, checked_irls_iterations=iteration_count,
               statuses=[dict(method=m,status=s,count=n) for (m,s),n in sorted(statuses.items())],
               nonconverged_fits_excluded_from_primary_averages=True, aggregate_means_verified=True,
               ode_simple_invariant_controls=3, structural_searches=0,
               result_sha256=result_hashes, geometry_input_sha256=geometry_inputs,
               artifact_sha256={p.name:sha(p) for p in sorted(ROOT.iterdir()) if p.is_file() and p.name!='completion_audit.json'})
    R.dump(ROOT/'completion_audit.json',out)
    print(json.dumps({k:v for k,v in out.items() if not k.endswith('_sha256')},indent=2))


if __name__=='__main__': main()
