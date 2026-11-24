from compat_context import *
from functools import lru_cache
from datetime import datetime,timezone

@lru_cache(None)
def digest(p):return sha(p)

def main():
    for name in ('process','summary','development'):assert read(ROOT/(name+'.json'))['status']=='COMPLETE'
    for name in ('checks','audit','dimension_diagnostic'):assert read(ROOT/(name+'.json'))['status']=='PASS'
    expected=dict(control_calibration=96,control_validation=48,signal_frozen=1288,signal_results=1288,control_audits=144,signal_audits=1288)
    for folder,n in expected.items():assert len(list((ROOT/folder).glob('*.json')))==n,(folder,n)
    links=jsons=arrays=0
    def visit(x):
        nonlocal links
        if isinstance(x,list):
            for v in x:visit(v)
        elif isinstance(x,dict):
            for k,v in x.items():
                if k in ('source_sha256','input_sha256'):
                    for p,h in v.items():assert digest(p)==h,(k,p);links+=1
                else:visit(v)
    for p in ROOT.rglob('*.json'):
        if p.name in ('completion_audit.json','artifact_manifest.json'):continue
        d=read(p);visit(d);jsons+=1
        if 'array_sha256' in d:assert digest(str(p.with_suffix('.npz')))==d['array_sha256'];arrays+=1
    # Check calibration rank and max explicitly, independently of the study driver.
    cal=read(ROOT/'calibration_frozen.json');records=[read(p) for p in (ROOT/'control_calibration').glob('*.json')]
    for n in (320,1001):
        cuts=[]
        for k in range(6):
            scores=[r['lr'] for d in records if d['n']==n and d['rho_index']==k for r in d['rows']];assert len(scores)==128;cuts.append(sorted(scores)[125])
        assert max(cuts)==cal['cutoffs'][str(n)]
    unchanged={}
    for name in ('spline_noise_2026-11-17','nuisance_calibration_2026-11-17','safeguard_envelope_2026-11-17','nuisance_direction_2026-11-17','identifiability_design_2026-11-17','full_pipeline_uncertainty_2026-11-17'):
        folder=ROOT.parent/name;manifest=read(folder/'artifact_manifest.json')
        for rel,h in manifest['sha256'].items():assert digest(str(folder/rel))==h,(name,rel)
        unchanged[name]=len(manifest['sha256'])
    audit=read(ROOT/'audit.json');s=read(ROOT/'summary.json');diag=read(ROOT/'dimension_diagnostic.json');assert len(audit['rows'])==1432 and len(diag['rows'])==1416
    dump(ROOT/'completion_audit.json',dict(status='PASS',json_records=jsons,array_hashes=arrays,hash_links=links,distinct_linked_files=digest.cache_info().currsize,counts=expected,audit_totals={k:sum(r.get(k,0) for r in audit['rows']) for k in ('cases','selection_replays','dense_truth_checks','dense_selected_checks','independent_limit_checks')},fine_grid_improvement_records=len(s['fine_grid_improvements']),posthoc_dimension_comparisons=3*len(diag['rows']),previous_artifacts_unchanged=unchanged,source_sha256=src('compat_complete.py')))
    marker='2026-11-17: Common-contrast noise compatibility and noncompact uncertainty';log=ROOT.parent/'solver_free_research_2026-11-16.txt';entry='\n\n'+marker+'\nspline_compatibility_2026-11-17/findings.txt\n'
    entry+='1536 controlled calibration fields,768 controlled validation fields,\n264 fresh ODE/VDP/Duffing fields and1024 positive-noise ODE stress fields.\nCompare CV against a fixed-p64 common-contrast compatibility screen.\nBoth archived boundary failures repaired, but fresh stress boundary failures\nonly2 ->1; five selections changed, median error essentially unchanged.\nNo selection changed on the264-field primary cohort. Not a complete repair.\n'
    entry+='p64 nominal noise-region coverage995/1024; amplitude unbounded458/1024.\nGrid calibration:996/1024 covered,473/1024 unbounded. Analytic rho->1,\nsigma^2(1-rho)->eta limit verified independently with first differences.\nExact controlled n320/rho.4 also unbounded27/64: excess mean flexibility\nremoves information needed to distinguish low-frequency signal and noise.\n'
    entry+='Posthoc fixed-dimension diagnostic, not adaptive-method validation:\n'+str(s['posthoc_dimensions']['stress/ode/positive'])+'\n'
    entry+='All audits and prior manifests pass. Numerical warning records retained.\nNo production edit, equation solver, dropped field, coefficient propagation\nrerun or full15-system claim. Constants/valid identities retained.\n'
    if marker not in log.read_text(encoding='utf-8'):
        with log.open('a',encoding='utf-8') as f:f.write(entry)
    files=[p for p in ROOT.rglob('*') if p.is_file() and p.name!='artifact_manifest.json'];dump(ROOT/'artifact_manifest.json',dict(status='COMPLETE',utc=datetime.now(timezone.utc).isoformat(),files=len(files),bytes=sum(p.stat().st_size for p in files),sha256={str(p.relative_to(ROOT)):sha(p) for p in sorted(files)}));print('COMPLETE',jsons,'JSON',links,'hash links',arrays,'arrays',unchanged,flush=True)

if __name__=='__main__':main()
