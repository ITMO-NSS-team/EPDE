"""Completion/provenance and independent arithmetic checks for both stages."""
import criteria as K
R,np,ROOT=K.R,K.np,K.ROOT
import json,csv
from pathlib import Path
from functools import lru_cache
from run_suite import SYSTEMS


@lru_cache(None)
def sha(path):return K.S.sha(path)


def check_hashes(record):
    for label in ('input_sha256','source_sha256'):
        for path,digest in record[label].items():assert sha(path)==digest,(label,path)


def independent_front(discrepancy,instability,cs,slot):
    ix=np.array([j for j,c in enumerate(cs) if c['slot']==slot]);v=np.column_stack([discrepancy,instability])[ix]
    dominated=(np.all(v[:,None,:]<=v[None,:,:],axis=2)&np.any(v[:,None,:]<v[None,:,:],axis=2)).any(axis=0)
    return list(ix[~dominated])


def main():
    expected={(n,s) for n in SYSTEMS for s in range(3)};runs=json.loads((ROOT/'process_results.json').read_text())
    assert {(r['system'],r['seed']) for r in runs}==expected and len(runs)==45 and all(r['exit_code']==0 for r in runs)
    front_count=0;case_count=0;instability_nonfinite=0;max_error=0.;primary_retained=0
    for name,seed in sorted(expected):
        d=json.loads((ROOT/'results'/f'{name}_seed{seed}.json').read_text());check_hashes(d)
        p=R.datasets.load(name);raw=p.noisy(5,seed);cs,labels=R.panel(p)
        assert {v:R.digest(y) for v,y in raw.items()}==d['raw_data_sha256']
        assert [c['id'] for c in cs]==[c['id'] for c in d['candidates']]
        assert [[len(c['xi']),c['registered']] for c in cs]==[[len(c['xi']),c['registered']] for c in d['candidates']]
        for vals in d['instability'].values():instability_nonfinite+=int(np.sum(~np.isfinite(vals)))
        max_error=max(max_error,d['maximum_dyad_replay_error'])
        for row in d['fronts']:
            f=independent_front(d['discrepancy'][row['arm']],d['instability'][row['metric']],cs,row['slot'])
            assert [cs[j]['id'] for j in f]==row['candidate_ids']
            assert row['retained']==any(cs[j]['registered'] for j in f)
            if row['arm']=='q0_white128/joint' and row['metric']=='dyad_sum':
                case_count+=1;primary_retained+=row['retained']
            front_count+=1
    assert case_count==60 and primary_retained==56 and instability_nonfinite==0
    checks=json.loads((ROOT/'checks.json').read_text());assert checks['source_sha256']==sha(str(ROOT/'criteria.py'))
    assert all(checks[k] for k in ('exact_constant_law_stable','feature_target_units_invariant','axis_transposition_invariant','spatial_drift_detected'))
    identities=json.loads((ROOT/'identity_checks.json').read_text());assert identities['analytic_identities_exact'] and identities['no_constant_removed']
    for path,digest in identities['source_sha256'].items():assert sha(path)==digest
    jobs=json.loads((ROOT/'uncertainty_jobs.json').read_text());runs=json.loads((ROOT/'uncertainty_process_results.json').read_text())
    assert len(jobs)==len(runs)==20 and all(r['exit_code']==0 for r in runs)
    assert {(r['system'],r['seed']) for r in runs}=={tuple(j) for j in jobs}
    pairs=list(csv.DictReader((ROOT/'uncertainty_pairs.csv').open()));fronts=list(csv.DictReader((ROOT/'uncertainty_fronts.csv').open()))
    checked_coverage=0;checked_fronts=0;noise_nonfinite=0;ureplay=0.;original_sums={}
    for name,seed in jobs:
        path=ROOT/'uncertainty'/f'{name}_seed{seed}.json';d=json.loads(path.read_text());check_hashes(d)
        assert sha(str(path.with_suffix('.npz')))==d['array_sha256'];a=np.load(path.with_suffix('.npz'));cs=d['candidates']
        original=json.loads((ROOT/'results'/path.name).read_text());original_sums[name,seed]=original
        assert d['raw_data_sha256']==original['raw_data_sha256']
        assert [c['id'] for c in cs]==[c['id'] for c in original['candidates']]
        np.testing.assert_allclose(a['observed'][0],original['discrepancy']['q0_white128/joint'],rtol=2e-9,atol=2e-11)
        for axis in (1,2):np.testing.assert_allclose(a['observed'][axis],original['instability'][d['axes'][axis]],rtol=2e-10,atol=2e-11)
        ureplay=max(ureplay,d['score_reproduction_max_error'])
        assert d['actual_noise_seeds']==list(range(390000+1000*seed,390128+1000*seed))
        for key in a.files:noise_nonfinite+=int(np.sum(~np.isfinite(a[key])))
        assert a['bootstrap'].shape==a['benchmark'].shape==(128,3,len(cs))
        assert d['bootstrap_nonfinite']==d['benchmark_nonfinite']==0
        for row in [r for r in pairs if r['system']==name and int(r['seed'])==seed]:
            ref=next(j for j,c in enumerate(cs) if c['id']==row['reference']);j=next(j for j,c in enumerate(cs) if c['id']==row['candidate'])
            for axis,label in enumerate(d['axes']):
                boot=a['bootstrap'][:,axis,j]-a['bootstrap'][:,axis,ref]-(a['pilot_center'][axis,j]-a['pilot_center'][axis,ref])
                actual=a['benchmark'][:,axis,j]-a['benchmark'][:,axis,ref]-(a['clean_center'][axis,j]-a['clean_center'][axis,ref])
                q=np.quantile(boot,[.025,.975]);assert int(row[label+'_coverage_count'])==np.sum((actual>=q[0])&(actual<=q[1]))
                np.testing.assert_allclose([float(row[label+'_bootstrap_error_lo']),float(row[label+'_bootstrap_error_hi'])],q)
                np.testing.assert_allclose([float(row[label+'_bootstrap_sd']),float(row[label+'_benchmark_sd'])],[np.std(boot,ddof=1),np.std(actual,ddof=1)])
                checked_coverage+=1
        for row in [r for r in fronts if r['system']==name and int(r['seed'])==seed]:
            axis=d['axes'].index(row['metric']);slot=int(row['slot'])
            for key in ('bootstrap','benchmark'):
                count=sum(any(cs[j]['registered'] for j in independent_front(b[0],b[axis],cs,slot)) for b in a[key])
                assert count==int(row[key+'_registered_count']);checked_fronts+=128
    assert noise_nonfinite==0
    out=dict(status='PASS',stage1_records=45,stage1_equation_seed_cases=case_count,stage1_fronts_checked=front_count,
        primary_retained=primary_retained,maximum_dyad_replay_error=max_error,stage1_nonfinite=instability_nonfinite,
        stage2_records=len(jobs),stage2_measurements_each=128*len(jobs),stage2_score_replay_error=ureplay,
        stage2_pair_axis_coverage_checks=checked_coverage,stage2_measurement_front_checks=checked_fronts,stage2_nonfinite=noise_nonfinite,
        source_input_array_hashes_verified=True,raw_measurement_replay=True,constants_and_identity_checks=True,
        caveats=['Reused stage1 records; selected stage2 cases; dependent pair summaries.',
            'Main fits, validation weights/corrections and geometry fixed in stage2.',
            'Stage1 clean_fit_energy fields describe historical local baseline fits, not the frozen joint-risk fits.'],
        analysis_sha256={str(p):sha(str(p)) for p in (Path(__file__),ROOT/'analyze_uncertainty.py',ROOT/'summarize.py')})
    R.dump(ROOT/'audit.json',out);print(json.dumps(out,indent=2))


if __name__=='__main__':main()
