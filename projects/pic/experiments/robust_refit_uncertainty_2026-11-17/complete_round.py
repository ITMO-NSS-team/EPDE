from common import *
from collections import Counter
from datetime import datetime,timezone

def main():
    required=['checks.json','checks6.json','state_checks.json','fast_checks.json','fast_tail_checks.json','batch_checks.json','time_checks.json','pilot_checks.json','audit_followups.json','audit_state.json','audit_fast.json','audit_attribution.json','audit_student.json','audit_time.json','audit_pilot_screen.json','audit_time_pilot.json','audit_time_pilot_student.json']
    for name in required:assert read(ROOT/name)['status']=='PASS',name
    expected={'results':69,'frozen_fast':69,'studentized_all':69,'studentized_all_frozen':69,'time_screen':23,'time_screen_frozen':23,'time_validation':69,'time_validation_frozen':69,'time_noise':9,'pilot_screen':3,'pilot_screen_frozen':3,'time_pilot_results':3,'time_pilot_frozen':3,'time_pilot_studentized':3,'time_pilot_student_frozen':3}
    for folder,n in expected.items():assert len(list((ROOT/folder).glob('*.json')))==n,(folder,n)
    for name in ('processes_fast.json','student_processes.json'):
        rows=read(ROOT/name)['rows'];assert len(rows)==69 and all(r['exit_code']==0 for r in rows),name
    assert read(ROOT/'time_fresh_processes.json')['status']=='COMPLETE' and len(read(ROOT/'time_fresh_processes.json')['rows'])==69
    assert read(ROOT/'time_noise_processes.json')['status']=='COMPLETE' and len(read(ROOT/'time_noise_processes.json')['rows'])==9
    rows=read(ROOT/'time_pilot_processes.json')['rows'];assert len(rows)==3 and all(len(r['rows'])==2 and all(x['exit_code']==0 for x in r['rows']) for r in rows)
    for name in ('summary.json','studentized_summary.json','time_summary.json','time_pilot_summary.json'):assert read(ROOT/name)['status']=='COMPLETE'
    assert (ROOT/'findings.txt').exists()
    cache={};verified=0;jsonfiles=0
    def digest(path):
        key=str(Path(path).resolve())
        if key not in cache:cache[key]=sha(key)
        return cache[key]
    def visit(obj):
        nonlocal verified
        if isinstance(obj,dict):
            for key,value in obj.items():
                if key in ('source_sha256','input_sha256','generated_sha256'):
                    assert isinstance(value,dict),(key,type(value))
                    for path,expected_hash in value.items():assert digest(path)==expected_hash,(key,path);verified+=1
                else:visit(value)
        elif isinstance(obj,list):
            for item in obj:visit(item)
    for path in ROOT.rglob('*.json'):
        if path.name in ('artifact_manifest.json','completion_audit.json'):continue
        visit(read(path));jsonfiles+=1
    a=read(ROOT/'summary.json');s=read(ROOT/'studentized_summary.json');t=read(ROOT/'time_summary.json');p=read(ROOT/'time_pilot_summary.json')
    assert a['anchors']==s['anchors']==t['fresh_anchors']==69 and p['anchors']==3
    out=dict(status='PASS',scope='Completed research round; universal coverage is not established',utc=datetime.now(timezone.utc).isoformat(),artifact_counts=expected,required_checks=required,verified_json_records=jsonfiles,verified_hash_links=verified,distinct_linked_files=len(cache),primary_fit_warnings=len(a['warnings']),targeted_fit_warnings=len(p['warnings']),source_sha256=hashes('complete_round.py','write_findings.py'),input_sha256=hashes('findings.txt','summary.json','studentized_summary.json','time_summary.json','time_pilot_summary.json',*required))
    dump(ROOT/'completion_audit.json',out)
    marker='robust_refit_uncertainty_2026-11-17/findings.txt';log=ROOT.parent/'solver_free_research_2026-11-16.txt';text=log.read_text(encoding='utf-8')
    if marker not in text:
        selected=[r for r in p['summary'] if r['mode']=='studentized'];entry='\n\n2026-11-17: numerical repair, robust refitting and studentization\n'+marker+'\n'
        entry+='All84 previous AR2 fits pass verified transformed-score convergence after\nexact innovations evaluation/polishing. Compound correction bias reduced by\nlocal quintic contrasts, but576 pure-noise controls expose lost positive-rho\nidentification.69 fresh anchors42..44,138 paired fixed/refit records;42 boundary\nwarnings retained. Correction refit alone does not propagate normalization.\n'
        entry+='Studentization tested/audited on all69 anchors: KS iid local86.46->90.89%,\nNS broad100->94.01%; AC-positive fails due bootstrap parameter drift.\n'
        entry+='Temporal likelihood + spatial sign correlations tested on69 fresh parameter\ncases45..47,192 bootstrap fields and576 paired pure-noise fields. Correlation-\naware GCV pilot reduces paired AC bootstrap amplitude bias24..30% to0.4..1.4%.\n'
        entry+='Targeted fresh AC-positive coverage48..50,256 draws per stream; studentized\n'+', '.join(r['bank']+' '+str(round(100*r['coverage'],2))+'%' for r in selected)+'.\n'
        entry+='All recorded numerical/replay audits PASS. Research artifacts only; constants\nand identities retained; no solver, production change or dyad replacement.\nFull-set coverage for the combined temporal estimator/correlated pilot remains\nunvalidated; do not promote globally or treat pooled draws as fitted anchors.\n'
        log.write_text(text+entry,encoding='utf-8')
    paths=sorted(p for p in ROOT.rglob('*') if p.is_file() and p.name!='artifact_manifest.json' and '__pycache__' not in p.parts)
    dump(ROOT/'artifact_manifest.json',dict(status='COMPLETE',utc=out['utc'],files=len(paths),bytes=sum(p.stat().st_size for p in paths),sha256={str(p.relative_to(ROOT)):sha(p) for p in paths},external_log_sha256={str(log):sha(log)}))
    print(json.dumps(out,indent=2));print('MANIFEST',len(paths),flush=True)
if __name__=='__main__':main()
