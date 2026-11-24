from research_context import *
from functools import lru_cache
from datetime import datetime,timezone

@lru_cache(None)
def digest(path):return sha(path)

def main():
    assert read(ROOT/'design_process.json')['status']=='COMPLETE'
    assert read(ROOT/'repair_process.json')['status']=='COMPLETE'
    for name,n in [('audit_design.json',9),('audit_profiles.json',84),('audit_repair.json',3)]:
        d=read(ROOT/name);assert d['status']=='PASS' and len(d['rows'])==n
    assert read(ROOT/'audit_optimizer.json')['status']=='PASS'
    links=0;jsons=0
    def visit(obj):
        nonlocal links
        if isinstance(obj,list):
            for x in obj:visit(x)
        elif isinstance(obj,dict):
            for k,v in obj.items():
                if k in ('source_sha256','input_sha256'):
                    for p,h in v.items():assert digest(p)==h,(k,p);links+=1
                else:visit(v)
    for p in ROOT.rglob('*.json'):
        if p.name in ('completion_audit.json','artifact_manifest.json'):continue
        visit(read(p));jsons+=1
    previous=read(PREVIOUS/'artifact_manifest.json')
    for rel,h in previous['sha256'].items():assert sha(PREVIOUS/rel)==h,('previous artifact changed',rel)
    screen=read(ROOT/'design_screen.json')
    with np.load(ROOT/'design_screen.npz') as z:
        assert sha(ROOT/'design_screen.npz')==screen['array_sha256'];e,d,_=primary('kdv_cossin','iid',51)
        for row in screen['rows']:
            if row['status']!='ok':continue
            m=z[row['bank']+'_pilot'];cv=z[row['bank']+'_covariance'];x=m[:,e.active];det=m[:,e.deterministic]
            if len(e.deterministic):x=x-det@np.linalg.lstsq(det,x,rcond=1e-12)[0]
            # Independent generalized eigensystem for the frozen proxy.
            from scipy.linalg import eigvalsh
            c=cv[np.ix_(e.active,e.active)];g=S.gram(x);sc=np.sqrt(np.diag(c));c=c/sc[:,None]/sc;g=g/sc[:,None]/sc
            ev=eigvalsh((g+g.T)/2,(c+c.T)/2);proxy=float(np.log1p(np.maximum(ev,0)).sum());np.testing.assert_allclose(proxy,row['score'],rtol=1e-9,atol=1e-9)
    expected=max((r for r in screen['rows'] if r['status']=='ok' and r['bank']!='local'),key=lambda r:r['score'])['bank'];assert expected==screen['selected']
    dump(ROOT/'completion_audit.json',dict(status='PASS',json_records=jsons,hash_links=links,distinct_linked_files=digest.cache_info().currsize,previous_artifacts_unchanged=len(previous['sha256']),design_scores_checked=len(screen['rows']),source_sha256=sources('complete.py'),previous_manifest_sha256=sha(PREVIOUS/'artifact_manifest.json')))
    s=read(ROOT/'summary.json');log=ROOT.parent/'solver_free_research_2026-11-16.txt';marker='2026-11-17: Noise optimizer basin repair and frozen weak-design test'
    entry='\n\n'+marker+'\nidentifiability_design_2026-11-17/findings.txt\n'
    entry+='Profile grid84 fits shows old ode seed52 bootstrap112 was a suboptimal basin,\nnot the best likelihood fit. Refining all original starts changes sigma\n10.3437->0.160199 and rho0.999916->0.662192; -2loglik improves19.6677.\n'
    entry+=f"Default15 regression:45 datasets/60 fields, all selected fits okay; max relative sigma change {s['regression']['max_sigma_relative_change']:.6g}.\n"
    entry+=f"Archived128 paired refits cutoff ratios {s['archive']['cutoff_ratios']}; no draws dropped.\n"
    entry+=f"Fresh ODE seeds54..56: {s['fresh_ode_coverage']} out of3;384 refits; diagnostic extremes {s['fresh_ode_extreme_diagnostic']}.\n"
    entry+=f"Frozen12-bank KdV screen selects full-time/half-space k2 windows. Fresh9 cases:\n{s['design_coverage']}; geometry {s['design_geometry']}.\n"
    entry+='Smaller dimensionless cutoffs do not resolve unbounded coefficient precision.\nAll numerical/replay audits PASS; previous773 artifacts unchanged.\nNo solver, production edit or dyad replacement. Larger outer validation and\nweak-design criteria targeting poorly identified directions remain next.\n'
    if marker not in log.read_text(encoding='utf-8'):
        with log.open('a',encoding='utf-8') as f:f.write(entry)
    files=[p for p in ROOT.rglob('*') if p.is_file() and p.name!='artifact_manifest.json'];dump(ROOT/'artifact_manifest.json',dict(status='COMPLETE',utc=datetime.now(timezone.utc).isoformat(),files=len(files),bytes=sum(p.stat().st_size for p in files),sha256={str(p.relative_to(ROOT)):sha(p) for p in sorted(files)}))
    print('COMPLETE',jsons,'JSON',links,'links; previous',len(previous['sha256']),'unchanged;',len(files),'artifacts',flush=True)

if __name__=='__main__':main()
