from pathlib import Path
import sys,json,datetime
ROOT=Path(__file__).resolve().parent
import coverage as C
S,R,np=C.S,C.R,C.np
from scipy.linalg import eigvalsh
from functools import lru_cache

@lru_cache(None)
def sha(path):return S.E.sha(path)

def walk(d):
    if isinstance(d,dict):
        yield d
        for v in d.values():yield from walk(v)
    elif isinstance(d,list):
        for v in d:yield from walk(v)

def main():
    for name in ('checks.json','checks_correlated.json','audit.json','audit_followups.json','audit_profile.json','audit_attribution.json','audit_ks_attribution.json'):
        d=json.loads((ROOT/name).read_text());assert d['status']=='PASS',name
    summary=json.loads((ROOT/'summary.json').read_text());assert summary['status']=='COMPLETE' and summary['records']==138
    runs=json.loads((ROOT/'processes.json').read_text());assert len(runs['rows'])==138 and all(r['exit_code']==0 for r in runs['rows'])
    counts={name:len(list((ROOT/name).glob('*.json'))) for name in ('results','frozen','explore','explore_frozen','correlated_explore','correlated_frozen')}
    assert counts['results']==138
    assert all(counts[n]==69 for n in ('explore','explore_frozen','correlated_explore','correlated_frozen'))
    statuses=[json.loads(p.read_text())['status'] for p in (ROOT/'results').glob('*.json')];assert counts['frozen']==statuses.count('COMPLETE')
    links=0
    for path in ROOT.rglob('*.json'):
        if path.name in ('completion_audit.json','final_manifest.json'):continue
        for d in walk(json.loads(path.read_text())):
            for key in ('source_sha256','input_sha256','prior_source_sha256','generated_sha256'):
                for p,h in d.get(key,{}).items():assert sha(p)==h,(path,key,p);links+=1
    for n in ('noise_trials.json','noise_trials_correlated.json'):assert len(json.loads((ROOT/n).read_text())['rows'])==576
    distortion=json.loads((ROOT/'calibration_distortion.json').read_text())
    for row in distortion['rows']:
        name,seed,bank=row['system'],row['seed'],row['bank'];frozen=json.loads((ROOT/'frozen'/f'{name}_iid_curvature_seed{seed}.json').read_text());active=frozen['active']
        with np.load(ROOT/f'{name}_attribution_seed{seed}.npz') as z:g=z[bank+'_oracle_error_grams'].mean(axis=0)[np.ix_(active,active)]
        with np.load(ROOT/'results'/f'{name}_iid_curvature_seed{seed}.npz') as z:s=z[bank+'_sigma'][np.ix_(active,active)]
        scale=np.sqrt(np.diag(s));g=g/scale[:,None]/scale[None,:];s=s/scale[:,None]/scale[None,:];eig=eigvalsh((g+g.T)/2,s)
        np.testing.assert_allclose([eig[0],eig[-1]],[row['min_actual_to_calibration_second_moment'],row['max_actual_to_calibration_second_moment']],rtol=3e-8,atol=3e-10)
    for n in ('findings.txt','coverage_comparison.png','coverage_comparison.pdf','iid_systems.png','pure_noise_comparison.png'):assert (ROOT/n).stat().st_size>1000
    master=ROOT.parent/'solver_free_research_2026-11-16.txt';assert 'latent_signal_noise_2026-11-17/findings.txt' in master.read_text()
    result=dict(status='PASS',records=138,complete=statuses.count('COMPLETE'),incomplete=statuses.count('incomplete_noise_model'),
        exploration_records=138,paired_pure_noise_draws=576,linked_hashes=links,
        main_audit=json.loads((ROOT/'audit.json').read_text())['totals'],followup_audit=json.loads((ROOT/'audit_followups.json').read_text()),
        master_log_sha256=sha(master),findings_sha256=sha(ROOT/'findings.txt'),source_sha256={str(Path(__file__)):sha(Path(__file__))})
    R.dump(ROOT/'completion_audit.json',result)
    artifacts=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(ROOT.rglob('*')) if p.is_file() and p.name!='final_manifest.json']
    manifest=dict(status='COMPLETE',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),artifact_count=len(artifacts),total_bytes=sum(r['bytes'] for r in artifacts),artifacts=artifacts)
    R.dump(ROOT/'final_manifest.json',manifest)
    for row in artifacts:assert S.E.sha(ROOT/row['path'])==row['sha256']
    print(json.dumps({k:v for k,v in result.items() if k!='followup_audit'},indent=2));print('ARTIFACTS',len(artifacts),'BYTES',manifest['total_bytes'])

if __name__=='__main__':main()
