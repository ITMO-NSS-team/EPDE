"""Final aggregation, provenance and artifact audit; no estimator refitting."""
from pathlib import Path
from functools import lru_cache
from collections import Counter,defaultdict
import json,hashlib,csv,datetime
import score_fit as F
np=F.np
ROOT=Path(__file__).resolve().parent
@lru_cache(None)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def walk(d):
    if isinstance(d,dict):
        yield d
        for v in d.values():yield from walk(v)
    elif isinstance(d,list):
        for v in d:yield from walk(v)
def dump(name,d):(ROOT/name).write_text(json.dumps(d,indent=2),encoding='utf-8')

def main():
    audits=['checks.json','checks_score.json','audit.json','audit_followups.json','audit_axis.json','audit_attribution.json']
    assert all(read(ROOT/n)['status']=='PASS' for n in audits)
    counts={n:len(list((ROOT/n).glob('*.json'))) for n in ('results','frozen','audit_records','screen','screen_frozen','score_screen','score_frozen','axis_frozen')}
    assert all(counts[n]==138 for n in ('results','frozen','audit_records')),counts
    assert all(counts[n]==23 for n in ('screen','screen_frozen','score_screen','score_frozen')),counts
    assert counts['axis_frozen']==13
    processes=read(ROOT/'processes.json');assert len(processes['rows'])==138 and all(r['exit_code']==0 for r in processes['rows'])
    records={p.stem:read(p) for p in (ROOT/'results').glob('*.json')}
    assert all(d['status']=='COMPLETE' for d in records.values())
    audit=read(ROOT/'audit.json');assert audit['records']==138 and audit['incomplete']==0
    for r in audit['rows']:assert sha(ROOT/'results'/(r['record']+'.json'))==r['result_sha256']

    # Reconstruct aggregation independently from the primary result counts.
    rates={};geometry=defaultdict(Counter);accepted=Counter();ref_counts=Counter()
    for d in records.values():
        for bank in ('local','broad'):
            refs=[r for r in d['statistics'] if r['bank']==bank and r['label']=='reference']
            assert d['metrics'][bank]['status']=='ok' and refs
            assert len({r['operator_coverage_count'] for r in refs})==1
            rates[d['system'],d['profile'],d['seed'],d['model_kind'],bank]=refs[0]['operator_coverage_count']/256
            k=(d['profile'],d['model_kind'],bank)
            for r in refs:
                region=next(c for c in d['regions'][bank] if c['candidate']==r['candidate'])
                geometry[k][region['status']]+=1;accepted[k]+=r['anchor_exact_accepted'];ref_counts[k]+=1
        rates[d['system'],d['profile'],d['seed'],d['model_kind'],'both']=d['joint']['operator_coverage_count']/256
        k=(d['profile'],d['model_kind'],'both')
        for r in d['joint']['reference_statistics']:
            cert=next(c for c in d['joint']['certificates'] if c['candidate']==r['candidate'])
            geometry[k][cert['geometry']]+=1;accepted[k]+=r['anchor_accepted'];ref_counts[k]+=1
    summary=read(ROOT/'summary.json');assert summary['records']==138 and not summary['incomplete']
    for r in summary['summary']:
        k=(r['profile'],r['model_kind'],r['bank']);rr=[v for (n,p,s,m,b),v in rates.items() if (p,m,b)==k]
        np.testing.assert_allclose(r['operator_coverage'],np.mean(rr),rtol=0,atol=1e-15)
        assert r['completed_records']==len(rr) and r['geometry']==dict(geometry[k])
        assert r['exact_anchor_accepted']==accepted[k] and r['reference_cases']==ref_counts[k]
    for r in read(ROOT/'paired_summary.json')['summary']:
        p,b=r['profile'],r['bank'];control='iid' if p=='iid' else 'known_rho'
        keys=[(n,s) for n,profile,s,m,bank in rates if (profile,m,bank)==(p,'ar2',b)]
        candidate=[rates[n,p,s,'ar2',b] for n,s in keys];base=[rates[n,p,s,control,b] for n,s in keys]
        assert r['complete_pairs']==r['planned_pairs']==len(keys) and not r['missing']
        np.testing.assert_allclose([r['candidate_coverage'],r['control_coverage'],r['paired_difference']],[np.mean(candidate),np.mean(base),np.mean(np.array(candidate)-base)],rtol=0,atol=1e-15)
    for r in csv.DictReader((ROOT/'system_coverage.csv').open(encoding='utf-8')):
        rr=[v for (n,p,s,m,b),v in rates.items() if (n,p,m,b)==(r['system'],r['profile'],r['model_kind'],r['bank'])]
        assert len(rr)==int(r['records'])==3
        np.testing.assert_allclose(float(r['operator_coverage']),np.mean(rr),rtol=0,atol=1e-15)

    for n in ('noise_trials.json','noise_trials_score.json'):
        d=read(ROOT/n);assert d['status']=='COMPLETE' and len(d['rows'])==576
        assert all(r['info']['status']=='ok' for r in d['rows'])
    axis=read(ROOT/'axis_diagnostic.json');assert len(axis['rows'])==13 and sum(r['status']=='boundary' for r in axis['rows'])==1
    attr=read(ROOT/'compound_attribution_summary.json')['rows'];assert len(attr)==6
    broad=[r for r in attr if r['bank']=='broad'];assert sum(r['original_coverage_count'] for r in broad)==28 and sum(r['oracle_correction_coverage_count'] for r in broad)==758
    detail=read(ROOT/'detailed_analysis.json');assert len(detail['primary_optimizer'])==84
    for r in detail['primary_optimizer']:
        info=records[r['record']]['fit_info'][r['variable']];selected=min([f for f in info['fits'] if f['success'] and np.isfinite(f['value']) and f['value']<1e40],key=lambda f:f['value'])
        t=np.array(selected['theta']);g=np.array(selected['gradient']);dim=len(info['rhos']);res=g.copy()
        lo=np.r_[np.full(dim,-1+1e-6),np.full(dim,-40),np.full(2*dim,-1+1e-6)]
        hi=np.r_[np.full(dim,1-1e-6),np.full(dim,20),np.full(2*dim,1-1e-6)]
        # Independent componentwise projected step calculation.
        res=np.minimum(np.maximum(g,t-hi),t-lo)
        np.testing.assert_allclose(r['projected_gradient_mapping'],np.max(np.abs(res)),rtol=0,atol=4e-15)
    assert detail['summary']['primary_projected_gradient_above_1e_minus3']==14
    assert detail['summary']['primary_selected_gap_above_1e_minus5']==6

    links=0
    for path in ROOT.rglob('*.json'):
        if path.name in ('completion_audit.json','final_manifest.json'):continue
        for d in walk(read(path)):
            for key in ('source_sha256','input_sha256','prior_source_sha256','generated_sha256'):
                for p,h in d.get(key,{}).items():assert sha(p)==h,(path,key,p);links+=1
    for n in ('findings.txt','method_notes.txt','coverage_comparison.png','coverage_comparison.pdf','iid_systems.png'):
        assert (ROOT/n).stat().st_size>1000
    master=ROOT.parent/'solver_free_research_2026-11-16.txt'
    assert 'ar2_signal_noise_2026-11-17/findings.txt' in master.read_text(encoding='utf-8')
    result=dict(status='PASS',primary_records=138,primary_complete=138,counts=counts,distinct_fitted_anchors=69,paired_pure_noise_draws=576,
        aggregation_rates_replayed=len(rates),optimizer_diagnostics_replayed=84,linked_hashes=links,main_audit=audit['totals'],
        audit_files={n:sha(ROOT/n) for n in audits},master_log_sha256=sha(master),findings_sha256=sha(ROOT/'findings.txt'),source_sha256={str(Path(__file__)):sha(Path(__file__))})
    dump('completion_audit.json',result)
    artifacts=[dict(path=str(p.relative_to(ROOT)),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(ROOT.rglob('*')) if p.is_file() and p.name!='final_manifest.json']
    manifest=dict(status='COMPLETE',created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),artifact_count=len(artifacts),total_bytes=sum(r['bytes'] for r in artifacts),artifacts=artifacts)
    dump('final_manifest.json',manifest)
    for r in artifacts:assert hashlib.sha256((ROOT/r['path']).read_bytes()).hexdigest()==r['sha256']
    print(json.dumps(result,indent=2));print('ARTIFACTS',len(artifacts),'BYTES',manifest['total_bytes'])

if __name__=='__main__':main()
