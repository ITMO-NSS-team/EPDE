from bridge import *
from datetime import datetime,timezone

def main():
    required=['metric_checks.json','acceleration_checks.json','audit_factorial_saved.json','audit_canonical.json','preflight_saved/check.json','preflight_saved/audit.json','preflight_fast/check.json','preflight_fast/audit.json','audit.json','panel_audit.json','diagnostic_audit.json']
    for name in required:assert read(ROOT/name)['status']=='PASS',name
    process=read(ROOT/'processes_fast.json');assert len(process['rows'])==69 and all(r['exit_code']==0 for r in process['rows'])
    for folder,n in [('results',69),('frozen',69),('audit_records',69),('panel_results',69),('panel_frozen',69),('panel_audit_records',69),('factorial',3),('factorial_frozen',3),('factorial_audit_saved',3),('pilot_replays',12)]:assert len(list((ROOT/folder).glob('*.json')))==n,(folder,n)
    for name in ('summary.json','factorial_analysis.json','pilot_effect.json','failure_geometry.json','precision_summary.json','panel_summary.json','panel_process.json','panel_enclosures.json','forced_geometry_probe.json','correction_diagnostic_summary.json','noise_tail_diagnostic.json'):assert read(ROOT/name)['status']=='COMPLETE'
    assert read(ROOT/'panel_enclosures.json')['audit_status']=='PASS'
    failed=sorted(p.stem for p in (ROOT/'results').glob('*.json') if any(not x for counts in read(p)['coverage'].values() for x in counts.values()))
    assert sorted(read(ROOT/'correction_diagnostic_summary.json')['diagnosed_records'])==failed
    assert sorted(p.stem for p in (ROOT/'correction_diagnostics').glob('*.json'))==failed
    assert (ROOT/'findings.txt').exists();summary=read(ROOT/'summary.json');assert summary['datasets']==69
    hashes={};links=0;jsons=0
    def digest(path):
        key=str(Path(path).resolve())
        if key not in hashes:hashes[key]=sha(key)
        return hashes[key]
    def visit(obj):
        nonlocal links
        if isinstance(obj,dict):
            for key,val in obj.items():
                if key in ('source_sha256','input_sha256','generated_sha256'):
                    for path,h in val.items():assert digest(path)==h,(key,path);links+=1
                else:visit(val)
        elif isinstance(obj,list):
            for val in obj:visit(val)
    for path in ROOT.rglob('*.json'):
        if path.name in ('completion_audit.json','artifact_manifest.json'):continue
        visit(read(path));jsons+=1
    warnings=sum(len(read(p)['warnings']) for p in (ROOT/'results').glob('*.json'));assert warnings==len(summary['warnings'])
    initial=read(ROOT/'initial_execution.json');assert initial['completed_cases']==initial['coverage_outcomes_seen']==0
    for path,info in initial['partial_logs'].items():assert sha(path)==info['sha256']
    out=dict(status='PASS',scope='Full nuisance-procedure research screen; not precise per-system coverage certification',utc=datetime.now(timezone.utc).isoformat(),outer_datasets=69,paired_methods=2,bootstrap_draws=128,metric_draws=16,required_checks=required,fit_warnings=warnings,verified_json_records=jsons,verified_hash_links=links,distinct_linked_files=len(hashes),historical_failures_preserved=['initial exact pilot-hash replay mismatch','failed cold-process replay before saved-input audit','direct execution interrupted before outcomes for exact algebra acceleration'],source_sha256=sources('complete_round.py','write_findings.py'),input_sha256={str(ROOT/n):sha(ROOT/n) for n in ['findings.txt','summary.json','factorial_analysis.json','pilot_effect.json','failure_geometry.json',*required]})
    if (ROOT/'audit_live.json').exists() and read(ROOT/'audit_live.json')['status']!='PASS':
        out['historical_failures_preserved'].append('Initial live audit report and any failed records preserved; finish_audit.py drains the queue and audit_verified.py independently checks ill-conditioned ellipsoids at 80 digits')
    out['statistical_noise_flags']=read(ROOT/'noise_tail_diagnostic.json')['flagged_fits']
    dump(ROOT/'completion_audit.json',out)
    log=ROOT.parent/'solver_free_research_2026-11-16.txt';text=log.read_text(encoding='utf-8');marker='full_pipeline_uncertainty_2026-11-17/findings.txt'
    if marker not in text:
        entry='\n\n2026-11-17: KS pilot attribution and complete nuisance-procedure screen\n'+marker+'\n'
        entry+='Paired KS-positive48 bootstrap fields: correlation-aware GCV reduces median\nnoise amplitude ratios1.349->1.035 (temporal),1.080->1.0003 (local6); near-unit\ntemporal fits6/48->0. Prior raw fields/noise increments paired exactly.\n'
        entry+='69 independent observed datasets, seeds51..53; each recalibrates128 bootstrap\nfields and16-field normalization. Noise fitting/reconstruction repeated within\nbootstrap; fixed/refitted-pilot normalization compared on identical data.\n'
        for profile in ('iid','positive','negative'):
            rr=[next(r for r in summary['summary'] if (r['profile'],r['mode'],r['bank'])==(profile,'refit_pilot',b)) for b in ('local','broad','both')];entry+=profile+' refitted-pilot local/broad/joint: '+' / '.join(f"{r['covered']}/{r['datasets']}" for r in rr)+'.\n'
        rr=[next(r for r in summary['systems'] if (r['system'],r['profile'],r['mode'],r['bank'])==('ks','positive','refit_pilot',b)) for b in ('local','broad','both')];entry+='KS-positive: '+' / '.join(f"{r['covered']}/{r['datasets']}" for r in rr)+'.\n'
        entry+=f'Final numerical/replay audits PASS; retained fit warnings {warnings}.\n'
        precision=read(ROOT/'precision_summary.json');panel=read(ROOT/'panel_summary.json')
        entry+='Primary reference-region geometry: '+str(summary['geometry'])+'.\n'
        entry+=f"Directly checked zero-coefficient witnesses: {precision['zero_witnesses']}/{precision['nonzero_truth_coordinates']} nonzero reference coordinates within bounded outer regions.\n"
        entry+='Exploratory fixed-panel calibration uses the same archived draws and a\nmaximum over all candidate supports and both banks. Forced-KdV regions remain\nunbounded in all 9 cases: panel cutoff reductions about 1-10% are insufficient\nto remove observed recession directions (about 59-83% reduction needed even\nfor those particular directions). This is not a justification to lower cutoffs.\n'
        entry+='Repeated-pilot versus fixed-pilot coverage discordance: '+str(summary['discordance'])+'.\n'
        entry+='Statistical noise-identification warning: ode/iid/seed52 bootstrap draw112\nhas sigma72.47x and rho0.999916 despite a small optimizer gradient. It selects\nthe local cutoff; diagnostic deletion lowers it14.46%/5.11% for fixed/refit\npilots, leaving joint cutoffs unchanged. No draw was removed. The anchor also\noverestimates iid noise correlation (rho0.516). Profile-based identification\nchecks remain necessary; numerical convergence alone is not identification.\n'
        entry+='Exact QR/Wick algebra speeds the same statistical procedure; initial direct\nrun stopped before outcomes, sources/logs preserved. Tiny pilot replay variation\nquantified and exact saved inputs used. No production change, solver or dyad\nreplacement. Three outer datasets/case are a screen, not95% certification.\n'
        log.write_text(text+entry,encoding='utf-8')
    paths=sorted(p for p in ROOT.rglob('*') if p.is_file() and p.name!='artifact_manifest.json' and '__pycache__' not in p.parts)
    dump(ROOT/'artifact_manifest.json',dict(status='COMPLETE',utc=out['utc'],files=len(paths),bytes=sum(p.stat().st_size for p in paths),sha256={str(p.relative_to(ROOT)):sha(p) for p in paths},external_log_sha256={str(log):sha(log)}));print(json.dumps({k:v for k,v in out.items() if not k.endswith('sha256')},indent=2));print('MANIFEST',len(paths),flush=True)
if __name__=='__main__':main()
