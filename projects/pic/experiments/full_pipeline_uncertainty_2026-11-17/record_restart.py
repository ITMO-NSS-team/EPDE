from bridge import *
from datetime import datetime,timezone

def main():
    assert not list((ROOT/'results').glob('*.json'));assert not list((ROOT/'frozen').glob('*.json'));assert read(ROOT/'acceleration_checks.json')['status']=='PASS';assert read(ROOT/'preflight_fast/audit.json')['status']=='PASS'
    logs=sorted((ROOT/'logs').glob('*.txt'));dump(ROOT/'initial_execution.json',dict(status='INTERRUPTED_FOR_EXACT_ALGEBRA_ACCELERATION',utc=datetime.now(timezone.utc).isoformat(),completed_cases=0,coverage_outcomes_seen=0,identified_runner_pid=137988,identified_auditor_pid=144088,identified_worker_pids=[136276,143956,129140,136064,137892,142708,12144],partial_logs={str(p):dict(sha256=sha(p),tail=p.read_text(encoding='utf-8').splitlines()[-3:]) for p in logs},source_sha256=sources('record_restart.py','full_pipeline_saved.py','run_full_saved.py','audit_full.py'),input_sha256={str(ROOT/p):sha(ROOT/p) for p in ('acceleration_checks.json','preflight_fast/audit.json','fast_provenance.json')}))
    print('Recorded interrupted execution; no completed coverage cases.',flush=True)
if __name__=='__main__':main()
