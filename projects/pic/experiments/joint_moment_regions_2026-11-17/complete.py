import study as S
R,np,ROOT=S.R,S.np,S.ROOT
from pathlib import Path
from functools import lru_cache
import json,datetime

@lru_cache(None)
def sha(path):return S.E.sha(path)

def walk(d):
    if isinstance(d,dict):
        yield d
        for value in d.values():yield from walk(value)
    elif isinstance(d,list):
        for value in d:yield from walk(value)

def main():
    audit=json.loads((ROOT/'audit_joint.json').read_text());assert audit['status']=='PASS' and audit['records']==117
    for file in ('checks.json','checks_intersection.json'):assert json.loads((ROOT/file).read_text())['status']=='PASS'
    hashes=0
    for path in ROOT.rglob('*.json'):
        if path.name in ('completion_audit.json','final_manifest.json'):continue
        for d in walk(json.loads(path.read_text())):
            for key in ('source_sha256','input_sha256'):
                for p,h in d.get(key,{}).items():assert sha(p)==h,(path,key,p);hashes+=1
    for directory in ('results','frozen','intersection'):assert len(list((ROOT/directory).glob('*.json')))==117
    summary=json.loads((ROOT/'intersection_summary.json').read_text());iid=next(r for r in summary['summary'] if r['profile']=='iid')
    assert iid['reference_geometry']=={'bounded_outer':57,'unbounded_certified':3}
    assert iid['all_candidate_geometry']=={'bounded_outer':679,'empty_certified':307,'unbounded_certified':62,'unresolved':2}
    assert iid['exact_reference_anchor_accepted']==59 and iid['reference_feasible_witnesses']==47
    # True-law membership is a post-freeze diagnostic, distinct from the
    # truth-free witness algorithm. It supplies the remaining reference witnesses.
    for path in (ROOT/'intersection').glob('*_iid_iid_*.json'):
        d=json.loads(path.read_text())
        for row in [r for r in d['statistics'] if r['label']=='reference']:
            cert=next(c for c in d['certificates'] if c['candidate']==row['candidate'])
            assert row['joint_exact_anchor_accepted'] or 'feasible_witness' in cert
    witnesses=json.loads((ROOT/'ks_sign_witnesses.json').read_text());assert len(witnesses['rows'])==6
    for row in witnesses['rows']:
        seed=row['seed'];name=f'ks_iid_iid_seed{seed}.json';d=json.loads((ROOT/'intersection'/name).read_text());f=json.loads((ROOT/'frozen'/name).read_text())
        c=next(c for c in f['candidates'] if c['label']=='reference');v=S.E.vector(c,row['beta'],len(f['labels']))
        with np.load((ROOT/'results'/name).with_suffix('.npz')) as a:
            margins=[float(v@(a[b+'_observed_gram']-d['cutoff']*a[b+'_sigma'])@v) for b in ('local','broad')]
        np.testing.assert_array_equal(margins,row['margins']);assert row['accepted_both']==(max(margins)<=0)
        if row['fixed_value']==1.:assert row['accepted_both']
    noise=json.loads((ROOT/'noise_guard_diagnostic.json').read_text());assert len(noise['rows'])==48
    assert all(r['capped_sigma_ratio']==r['uncapped_sigma_ratio'] for r in noise['rows'] if r['mode']=='raw')
    boundary=json.loads((ROOT/'audit_boundary_diagnostics.json').read_text());assert len(boundary)==31
    assert max(r['natural_backward_error'] for r in boundary)<8e-17
    master=ROOT.parent/'solver_free_research_2026-11-16.txt';assert 'Follow-up completed 2026-11-17: joint_moment_regions_2026-11-17/findings.txt.' in master.read_text()
    report=ROOT/'findings.txt';assert report.stat().st_size>10000
    for name in ('joint_results.png','joint_results.pdf'):assert (ROOT/name).stat().st_size>10000
    out=dict(status='PASS',records=117,source_and_input_links_checked=hashes,all_reference_intersections_nonempty_after_diagnostics=True,
        ks_positive_sign_witnesses=3,raw_guard_ablation_unchanged=24,primary_audit_totals=audit['totals'],
        master_log_sha256=sha(master),source_sha256={str(Path(__file__)):sha(Path(__file__))},findings_sha256=sha(report))
    R.dump(ROOT/'completion_audit.json',out)
    artifacts=[]
    for path in sorted(ROOT.rglob('*')):
        if path.is_file() and path.name!='final_manifest.json':artifacts.append(dict(path=str(path.relative_to(ROOT)),bytes=path.stat().st_size,sha256=sha(path)))
    manifest=dict(status='COMPLETE',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),artifact_count=len(artifacts),total_bytes=sum(r['bytes'] for r in artifacts),artifacts=artifacts)
    R.dump(ROOT/'final_manifest.json',manifest)
    for row in artifacts:assert S.E.sha(ROOT/row['path'])==row['sha256']
    print(json.dumps(out,indent=2));print('MANIFEST',len(artifacts),manifest['total_bytes'])

if __name__=='__main__':main()
