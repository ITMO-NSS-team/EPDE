from bridge import *
from concurrent.futures import ThreadPoolExecutor,as_completed
import subprocess,time,argparse

def worker(job):
    n,p,s=job;key=f'{n}_{p}_seed{s}';log=ROOT/'logs'/(key+'.txt');log.parent.mkdir(exist_ok=True);start=time.monotonic()
    with log.open('w',encoding='utf-8') as f:r=subprocess.run([sys.executable,'-B','-X','utf8',str(ROOT/'run_full.py'),'--case',n,p,str(s)],stdout=f,stderr=subprocess.STDOUT)
    return dict(system=n,profile=p,seed=s,exit_code=r.returncode,seconds=time.monotonic()-start)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--case',nargs=3);ap.add_argument('--workers',type=int,default=7);a=ap.parse_args()
    if a.case:
        from full_pipeline import run
        run(a.case[0],a.case[1],int(a.case[2]));return
    assert read(ROOT/'metric_checks.json')['status']=='PASS';assert read(ROOT/'preflight/check.json')['status']=='PASS';jobs=[(n,p,s) for n,p in JOBS for s in (51,52,53)];jobs.sort(key=lambda j:(j[0]!='ks',j));rows=[]
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        for f in as_completed([pool.submit(worker,j) for j in jobs]):r=f.result();rows.append(r);print(r,flush=True)
    dump(ROOT/'processes.json',dict(rows=rows,source_sha256=sources('run_full.py','coverage_decision.txt')))
    assert all(r['exit_code']==0 for r in rows)
if __name__=='__main__':main()
