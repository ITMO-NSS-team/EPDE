"""Audit completed immutable records while the final studentization jobs run."""
from common import *
import audit_student as Q
from concurrent.futures import ProcessPoolExecutor
import time,argparse

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=1);a=ap.parse_args();seen=set();pending={};rows=[]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        while len(rows)<69:
            for path in sorted((ROOT/'studentized_all').glob('*.json')):
                if path.name not in seen and len(pending)<a.workers:
                    pending[pool.submit(Q.one,str(path))]=path.name;seen.add(path.name)
            for f in list(pending):
                if f.done():r=f.result();pending.pop(f);rows.append(r);print(r['record'],'PASS',flush=True)
            process=ROOT/'student_processes.json'
            if process.exists():assert all(r['exit_code']==0 for r in read(process)['rows'])
            time.sleep(1)
    totals={k:sum(r[k] for r in rows) for k in ('operator_scores','regions','direct_metric_checks','hash_links')}
    out=dict(status='PASS',records=69,totals=totals,max_direct_metric_relative_error=max(r['max_direct_metric_relative_error'] for r in rows),rows=rows,source_sha256=hashes('audit_student.py','audit_student_available.py'))
    dump(ROOT/'audit_student.json',out);print(totals,flush=True)
if __name__=='__main__':main()
