from common import *
from concurrent.futures import ThreadPoolExecutor
import subprocess,time
def one(job):
    n,p,s=job;key=f'{n}_{p}_seed{s}';log=ROOT/'student_logs'/(key+'.txt');log.parent.mkdir(exist_ok=True);start=time.monotonic()
    with log.open('w',encoding='utf-8') as f:r=subprocess.run([sys.executable,'-B','-X','utf8',str(ROOT/'studentize_all.py'),n,p,str(s)],stdout=f,stderr=subprocess.STDOUT)
    return dict(system=n,profile=p,seed=s,exit_code=r.returncode,seconds=time.monotonic()-start)
def main():
    jobs=[(n,p,s) for n,p in JOBS for s in (42,43,44)];jobs.sort(key=lambda j:(j[0] not in ('ks','ns','pde_compound'),j));pending={};submitted=set();rows=[]
    with ThreadPoolExecutor(max_workers=7) as pool:
        while len(rows)<69:
            limit=7 if (ROOT/'processes_fast.json').exists() else 2
            for job in jobs:
                n,p,s=job
                if job not in submitted and len(pending)<limit and (ROOT/'results'/f'{n}_{p}_seed{s}.json').exists():pending[pool.submit(one,job)]=job;submitted.add(job)
            for f in list(pending):
                if f.done():r=f.result();rows.append(r);pending.pop(f);print(r,flush=True)
            time.sleep(1)
    dump(ROOT/'student_processes.json',dict(rows=rows,source_sha256=hashes('run_student_suite.py')));assert all(r['exit_code']==0 for r in rows)
if __name__=='__main__':main()
