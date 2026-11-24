"""Audit the frozen decision boundary and independent-noise evaluation."""
import run_outer as S
np,R,ROOT=S.np,S.R,S.ROOT
import json
from collections import Counter


def main():
    files=sorted((ROOT/'results').glob('*.json'));assert len(files)==12
    assert {p.stem for p in files}=={f'{n}_seed{s}' for n in ('ac','kdv_cossin','wave','ks') for s in (6,7,8)}
    statuses=Counter();selections=0;forced=0;finite=0;hashes={};inputs={}
    for path in files:
        d=json.loads(path.read_text());fpath=ROOT/'frozen'/path.name;frozen=json.loads(fpath.read_text())
        assert S.sha(fpath)==d['selection_freeze_sha256']
        assert d['source_sha256']==S.sources()
        assert all(S.sha(p)==h for p,h in d['input_sha256'].items());inputs.update(d['input_sha256'])
        assert d['outer_seed']==d['seed']+3 and d['seed'] in (6,7,8) and d['outer_seed'] in (9,10,11)
        p=R.datasets.load(d['system'])
        for seed,key in [(d['seed'],'raw_data_sha256'),(d['outer_seed'],'outer_data_sha256')]:
            assert d[key]=={v:R.digest(y) for v,y in p.noisy(5,seed).items()}
        assert d['raw_data_sha256']!=d['outer_data_sha256']
        _,base=S.F.baseline(d['system'],0);idx=[np.asarray(g['indices']) for g in base['groups']]
        assert all(not np.intersect1d(a,b).size for i,a in enumerate(idx) for b in idx[:i])
        previous=json.loads((S.N.ROOT/'results'/f"{d['system']}_seed3.json").read_text())
        assert d['candidates']==previous['candidates']
        for g,fold in enumerate(d['folds']):
            before=frozen['folds'][g];assert fold['geometry']==previous['folds'][g]['geometry']
            assert fold['validation_weighting']==before['validation_weighting']
            assert all(m['feasible'] for m in fold['validation_weighting'].values())
            for row,original in zip(fold['rows'],before['rows']):
                assert row['fit']==original['fit'] and row['coefficients']==original['coefficients']
                assert not any(k.startswith(('outer_','clean_','frozen_clean')) for k in original)
                for k,v in original.items():assert row[k]==v
                statuses[row['bank'],row['method'],row['fit']['status']]+=1
                if row['fit']['status']=='ok':
                    assert row['coefficients'] is not None and np.isfinite(row['coefficients']).all();finite+=1
                    if row['method']=='irls':assert row['fit']['trace'][-1]['prediction_change']<=1e-8
                    for key,value in row.items():
                        if key.startswith(('inner_','outer_','clean_','frozen_clean')):assert np.isfinite(value)
                else:assert row['coefficients'] is None and 'frozen_clean_energy' not in row
            for decision,original in zip(fold['decisions'],before['decisions']):
                assert 'evaluation' not in original
                reproduced=S.choose(original['options'])
                for k,v in reproduced.items():assert v==original[k]
                for k,v in original.items():assert decision[k]==v
                selections+=1;forced+=decision['forced']
                if decision['status']=='ok':
                    row=next(r for r in fold['rows'] if r['candidate']==decision['candidate'] and r['bank']==decision['bank'] and r['method']=='irls')
                    for k,v in decision['evaluation'].items():assert row[k]==v
        hashes[path.name]=S.sha(path)
    sr=0;bs=Counter()
    for seed in (6,7,8):
        path=ROOT/'sensitivity'/f'seed{seed}.json';d=json.loads(path.read_text())
        assert all(S.sha(p)==h for p,h in d['input_sha256'].items())
        assert all(S.sha(p)==h for p,h in d['source_sha256'].items())
        for row in d['rows']:
            sr+=1;assert len(row['samples'])==64
            count=Counter(r['status'] for r in row['samples']);assert dict(count)==row['bootstrap_statuses']
            assert count['ok']==row['bootstrap_successes']
            for sample in row['samples']:
                bs[row['bank'],row['method'],sample['status']]+=1
                if sample['status']=='ok':assert np.isfinite(sample['coefficients']).all()
                else:assert sample['coefficients'] is None
    assert sr==54
    R.dump(ROOT/'completion_audit.json',dict(system_runs=12,reference_seed_cases=12,registered_equation_seed_cases=24,
        fresh_seed_sets_disjoint=True,raw_noise_replay_verified=True,source_and_input_hashes_verified=True,
        original_registered_supports_preserved=True,geometry_matches_native_study=True,
        frozen_selection_replayed=True,selection_records=selections,forced_selections=forced,
        training_coefficients_unchanged_by_outer_evaluation=True,all_validation_weights_feasible=True,finite_fits=finite,
        fit_statuses=[dict(bank=b,method=m,status=s,count=n) for (b,m,s),n in sorted(statuses.items())],
        conditional_sensitivity_rows=sr,independent_bootstrap_fits=sum(bs.values()),
        bootstrap_statuses=[dict(bank=b,method=m,status=s,count=n) for (b,m,s),n in sorted(bs.items())],
        result_sha256=hashes,input_sha256=inputs))
    print('PASS: 12 runs; frozen decisions, fresh noise replay, unchanged fits, geometry, supports and bootstrap outcomes verified.')


if __name__=='__main__':main()
