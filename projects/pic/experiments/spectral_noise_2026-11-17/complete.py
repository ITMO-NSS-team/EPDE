from pathlib import Path
import sys,json,datetime,hashlib
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'joint_moment_regions_2026-11-17'))
import study as S
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
    for name in ('checks.json','check_stable.json','audit.json','audit_followups.json'):
        d=json.loads((ROOT/name).read_text());assert d['status']=='PASS',name
    summary=json.loads((ROOT/'summary.json').read_text());assert summary['status']=='COMPLETE' and summary['records']==162
    runs=json.loads((ROOT/'processes.json').read_text());assert len(runs['rows'])==162 and all(r['exit_code']==0 for r in runs['rows'])
    counts={name:len(list((ROOT/name).glob('*.json'))) for name in ('results','frozen','explore','explore_frozen')}
    assert counts==dict(results=162,frozen=162,explore=69,explore_frozen=69)
    links=0
    for path in ROOT.rglob('*.json'):
        if path.name in ('completion_audit.json','final_manifest.json'):continue
        for d in walk(json.loads(path.read_text())):
            for key in ('source_sha256','input_sha256'):
                for p,h in d.get(key,{}).items():assert sha(p)==h,(path,key,p);links+=1
    # Exact reported numeric claims, not a test for favorable outcomes.
    assert not summary['incomplete']
    for seed in (33,34,35):
        d=json.loads((ROOT/f'compound_attribution_seed{seed}.json').read_text());r=next(r for r in d['rows'] if r['bank']=='broad')
        assert r['original_coverage_count']==0 and r['oracle_correction_coverage_count']==256
        assert 4<r['analytic_bias_cutoff_ratio']<5
    ident=json.loads((ROOT/'identification.json').read_text());assert len(ident['rows'])==576
    assert sum(r['spectral_status']!='ok' for r in ident['rows'])==26
    assert sum(r['spectral_status']!='ok' and r['true_rho']==.4 for r in ident['rows'])==23
    for name in ('findings.txt','coverage_comparison.png','coverage_comparison.pdf','iid_systems.png'):assert (ROOT/name).stat().st_size>5000
    master=ROOT.parent/'solver_free_research_2026-11-16.txt';assert 'spectral_noise_2026-11-17/findings.txt' in master.read_text()
    result=dict(status='PASS',records=162,exploratory_records=69,pure_noise_draws=576,linked_hashes=links,
        main_audit=json.loads((ROOT/'audit.json').read_text())['totals'],followup_audit=json.loads((ROOT/'audit_followups.json').read_text()),
        master_log_sha256=sha(master),findings_sha256=sha(ROOT/'findings.txt'),source_sha256={str(Path(__file__)):sha(Path(__file__))})
    S.R.dump(ROOT/'completion_audit.json',result)
    artifacts=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(ROOT.rglob('*')) if p.is_file() and p.name!='final_manifest.json']
    manifest=dict(status='COMPLETE',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),artifact_count=len(artifacts),total_bytes=sum(r['bytes'] for r in artifacts),artifacts=artifacts)
    S.R.dump(ROOT/'final_manifest.json',manifest)
    for row in artifacts:assert S.E.sha(ROOT/row['path'])==row['sha256']
    print(json.dumps({k:v for k,v in result.items() if k!='followup_audit'},indent=2));print('ARTIFACTS',len(artifacts),'BYTES',manifest['total_bytes'])

if __name__=='__main__':main()
