from calibration_context import *
from functools import lru_cache
from datetime import datetime,timezone


@lru_cache(None)
def digest(path):return sha(path)


def main():
    for name in ('process','bootstrap_process','propagation_process','paired_process','profile_stress','posthoc_repair'):assert read(ROOT/(name+'.json'))['status']=='COMPLETE'
    for name in ('profile_checks','audit','bootstrap_audit','propagation_audit'):assert read(ROOT/(name+'.json'))['status']=='PASS'
    expected=dict(calibration=384,model_validation=192,ode=192,bootstrap_anchors=24,bootstrap_chunks=384,bootstrap_frozen=24,bootstrap_results=24,propagation_primary=24,propagation_chunks=528,propagation_frozen=24,propagation_results=24,paired_profiles=24,paired_audits=24)
    for folder,n in expected.items():assert len(list((ROOT/folder).glob('*.json')))==n,(folder,n)
    stress=read(ROOT/'profile_stress.json');assert len(stress['rows'])==88;assert stress['nominal_membership_changes']==stress['calibrated_membership_changes']==0
    import profile_region as Q
    from run_profiles import generate
    for row in stress['rows']:
        y,theta,design=generate('ode',row['profile'],row['seed']);data=F.A.Contrasts(y).fields[0];v,g=Q.likelihood(data,np.r_[theta,row['selected']['nuisance']],True);np.testing.assert_allclose(v,row['selected_original_value'],rtol=1e-10,atol=1e-10);np.testing.assert_allclose(g,row['selected_original_gradient'],rtol=1e-7,atol=1e-9);assert row['selected_projected_gradient']<=2e-5
    negative=[r['key'] for r in stress['rows'] if r['new_raw_lr']<-1e-5];repair=read(ROOT/'posthoc_repair.json');assert negative==[r['key'] for r in repair['rows']];assert len(negative)==1
    # Rejected initialization changes remain explicitly failed and unused.
    rejected={}
    for name,count in [('exact_start_checks',1),('expanded_checks',2)]:
        d=read(ROOT/(name+'.json'));assert d['status']=='FAIL';bad=[r for r in d['rows'] if r['status']=='FAIL'];assert len(bad)==count;rejected[name]=[dict(record=r['record'],likelihood_difference=r['likelihood_difference']) for r in bad]
    for folder in ('calibration','model_validation','ode'):
        for path in (ROOT/folder).glob('*.json'):assert 'initial_likelihood_backend' not in read(path)['fit_info']
    for path in (ROOT/'bootstrap_frozen').glob('*.json'):
        for r in read(path)['refits']:assert 'initial_likelihood_backend' not in r['info']
    links=jsons=0
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
    for name in ('safeguard_envelope_2026-11-17','nuisance_direction_2026-11-17','identifiability_design_2026-11-17','full_pipeline_uncertainty_2026-11-17'):
        folder=ROOT.parent/name;manifest=read(folder/'artifact_manifest.json')
        for rel,h in manifest['sha256'].items():assert sha(folder/rel)==h,(name,rel)
        unchanged[name]=len(manifest['sha256'])
    pa=read(ROOT/'propagation_audit.json');ba=read(ROOT/'bootstrap_audit.json');summary=read(ROOT/'summary.json')
    dump(ROOT/'completion_audit.json',dict(status='PASS',json_records=jsons,hash_links=links,distinct_linked_files=digest.cache_info().currsize,cohort_counts=expected,rejected_initializers=rejected,restricted_stress_replays=88,newly_exposed_unrestricted_basin=negative,original_bootstrap_fit_replays=sum(r['original_fit_replays'] for r in ba['rows']),propagation_totals={k:sum(r[k] for r in pa['rows']) for k in ('generalized_scores','certificates','loewner_bounds')},previous_artifacts_unchanged=unchanged,source_sha256=src('complete.py')))
    marker='2026-11-17: Noise-region calibration and signal-model stress test';log=ROOT.parent/'solver_free_research_2026-11-16.txt'
    entry='\n\n'+marker+'\nnuisance_calibration_2026-11-17/findings.txt\n'
    entry+='384 controlled calibration fields,192 controlled validation fields,\n192 fresh ODE fields. Nominal97.5% noise-pair LR/Wald regions. Controlled\ncalibration frozen before validation. Profile LR model-calibrated coverage\n(iid,+.4,-.4): controlled63/64,64/64,64/64; ODE59/64,7/64,63/64.\nPositive noise-amplitude median/truth0.9998 controlled versus1.4242 ODE.\nAll768 records passed original-Python likelihood/gradient audits. This\nis a statistical model-transfer failure, not just optimizer convergence.\n'
    entry+='Independent24-field bootstrap follow-up,128 refits each. Continuous\nbasic ellipses in(log sigma,atanh rho), fixed before truth scoring.\n'
    entry+='Noise coverage: '+str(summary['bootstrap_noise_coverage'])+'\n'
    entry+='Paired likelihood coverage: '+str(summary['paired_likelihood_coverage'])+'\n'
    entry+='Finite66-scenario propagation per ellipse,128 conditional draws/point;\nkeep both original regions in union. This does NOT certify the continuum.\n'
    entry+='Weak-error coverage: '+str(summary['equation_error_coverage'])+'\nReference geometry: '+str(summary['reference_geometry'])+'\n'
    entry+='Rejected2 faster initializer variants after archived basin failures.\nOriginal guarded fitter retained. Audits and prior manifest checks pass.\nNo solver, production edits, clipped parameters or discarded draws.\nConstants/valid identities retained. KdV remains unresolved, dyad provisional.\n'
    entry+='An80-start nuisance-profile stress test on88 fields improved some\nprofile likelihoods(max LR reduction6.0042) but changed no membership\ndecisions; numerical robustness evidence, not a global-optimality proof.\n'
    entry+='The stress test exposed one deficient unrestricted fit(positive/234).\nPosthoc release improved-2loglik25.4127 but increased noise bias and\nchanged that case to noncoverage(LR20.0859). Original7/64 count retained;\na truth-seeded retrospective replacement would give6/64, not validation.\n'
    if marker not in log.read_text(encoding='utf-8'):
        with log.open('a',encoding='utf-8') as f:f.write(entry)
    files=[p for p in ROOT.rglob('*') if p.is_file() and p.name!='artifact_manifest.json'];dump(ROOT/'artifact_manifest.json',dict(status='COMPLETE',utc=datetime.now(timezone.utc).isoformat(),files=len(files),bytes=sum(p.stat().st_size for p in files),sha256={str(p.relative_to(ROOT)):sha(p) for p in sorted(files)}));print('COMPLETE',jsons,'JSON',links,'hash links',unchanged,flush=True)


if __name__=='__main__':main()
