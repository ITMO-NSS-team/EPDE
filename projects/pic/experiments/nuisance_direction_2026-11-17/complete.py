from context import *
import audit_new as A
from functools import lru_cache
from datetime import datetime,timezone

@lru_cache(None)
def digest(path):return sha(path)

def main():
    for name,n in [('audit.json',18),('audit_survey.json',96),('audit_guard.json',144)]:
        d=read(ROOT/name);assert d['status']=='PASS' and len(d['rows'])==n
    assert read(ROOT/'audit_screen.json')['status']=='PASS'
    for name in ('process.json','survey_process.json','guard_process.json','profile_probe.json','local_comparison.json'):assert read(ROOT/name)['status']=='COMPLETE'
    # Independently replay the retrospective fixed-metric generalized scores.
    ret=read(ROOT/'retrospective.json');nr=0
    for row in ret['rows']:
        root=PREVIOUS/'repair_results' if row['cohort']=='repaired_ode' else C.PREVIOUS/'results';path=root/f"{row['system']}_{row['profile']}_seed{row['seed']}.json";d=read(path);a=np.load(path.with_suffix('.npz'));qs={}
        for b in ('local','broad'):
            cv=a[b+('_covariance' if row['cohort']=='repaired_ode' else '_anchor_covariance')];q=[A.eigen_score(g,cv,d['active']) for g in a[b+'_bootstrap_grams']];qs[b]=q;A.close(sorted(q)[122],row['cuts'][b]);nr+=128
        A.close(sorted(np.maximum(qs['local'],qs['broad']))[122],row['cuts']['joint'])
    # The profile probe's released fits use the same exact likelihood as its
    # comparison. Replay model, objective and projected gradient independently.
    for p in (ROOT/'profile_probe').glob('*.json'):
        d=read(p);e,_,_=C.primary('ode','iid',51);e.pid,e.rho=S.PROFILES[d['profile']];e.seed=d['seed'];y=e.true_field(0,True)['u'];A.exactfit(y,d['new_model'],dict(d['new_info'],status=d['status']));A.close(d['likelihood_gain'],(len(y)-2)*(d['old_objective']-d['new_info']['objective']))
    links=0;jsons=0
    def visit(obj):
        nonlocal links
        if isinstance(obj,list):
            for v in obj:visit(v)
        elif isinstance(obj,dict):
            for k,v in obj.items():
                if k in ('source_sha256','input_sha256'):
                    for p,h in v.items():assert digest(p)==h,(k,p);links+=1
                else:visit(v)
    for p in ROOT.rglob('*.json'):
        if p.name in ('completion_audit.json','artifact_manifest.json'):continue
        visit(read(p));jsons+=1
    unchanged={}
    for prev in (PREVIOUS,C.PREVIOUS):
        d=read(prev/'artifact_manifest.json')
        for rel,h in d['sha256'].items():assert sha(prev/rel)==h,(prev,rel)
        unchanged[prev.name]=len(d['sha256'])
    dump(ROOT/'completion_audit.json',dict(status='PASS',json_records=jsons,hash_links=links,distinct_linked_files=digest.cache_info().currsize,retrospective_generalized_scores=nr,previous_artifacts_unchanged=unchanged,source_sha256=sources('complete.py')))
    s=read(ROOT/'summary.json');log=ROOT.parent/'solver_free_research_2026-11-16.txt';marker='2026-11-17: Nuisance normalization, weak directions and profile-start guard'
    entry='\n\n'+marker+'\nnuisance_direction_2026-11-17/findings.txt\n'
    entry+='18 fresh datasets,128 refits each: studentized versus observed fixed-metric\nbootstrap normalization, paired on identical draws; all-start temporal fitter.\n'
    entry+=f"Coverage: {s['coverage']}\nReference geometry: {s['reference_geometry']}\n"
    entry+='KDv full_k4 selected by fixed-candidate weak-direction criterion.\nIndependent96-field ODE survey exposes19/32 positive-correlation near-unit\nfits versus0/32 IID and0/32 negative. Gradient convergence missed poor basins.\n'
    entry+=f"Conditional profile-start guard, independent48-field test: {s['guard']['fresh']}\n"
    entry+='The guard is a noise-fitting safeguard; full bootstrap uncertainty with it\nhas not been evaluated here. Original18-case results remain unchanged.\n'
    entry+='Local6 median positive amplitude ratio1.05, but10/32 near-unit estimates;\nno global replacement. Exact Gaussian information calculation: at rho+.4,\nlocal quintic removal raises local rho scale0.0513->0.314 and log-sigma/rho\ncorrelation0.984. This is a local bound under exact polynomial elimination,\nnot a confidence interval or proof of every temporal-model bias.\n'
    entry+='Audits PASS. Prior experiment manifests unchanged. No solver, production\nedit, omitted draw or dyad replacement; constants and valid identities retained.\n'
    if marker not in log.read_text(encoding='utf-8'):
        with log.open('a',encoding='utf-8') as f:f.write(entry)
    files=[p for p in ROOT.rglob('*') if p.is_file() and p.name!='artifact_manifest.json'];dump(ROOT/'artifact_manifest.json',dict(status='COMPLETE',utc=datetime.now(timezone.utc).isoformat(),files=len(files),bytes=sum(p.stat().st_size for p in files),sha256={str(p.relative_to(ROOT)):sha(p) for p in sorted(files)}))
    print('COMPLETE',jsons,'JSON',links,'hash links',unchanged,flush=True)

if __name__=='__main__':main()
