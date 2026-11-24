from common import *
from concurrent.futures import ProcessPoolExecutor,as_completed
import subprocess,time,argparse
def one(job):
    n,s,p=job;key=f'{n}_{p}_seed{s}';log=ROOT/'logs_fast'/(key+'.txt');log.parent.mkdir(exist_ok=True);start=time.monotonic()
    with log.open('w',encoding='utf-8') as f:out=subprocess.run([sys.executable,'-B','-X','utf8',str(ROOT/'run_case_fast.py'),n,str(s),p],stdout=f,stderr=subprocess.STDOUT)
    return dict(system=n,seed=s,profile=p,exit_code=out.returncode,seconds=time.monotonic()-start)
def main():
    p=argparse.ArgumentParser();p.add_argument('--workers',type=int,default=7);a=p.parse_args();assert read(ROOT/'checks6.json')['status']=='PASS'
    jobs=[(n,s,p) for n,p in JOBS for s in (42,43,44)];jobs.sort(key=lambda j:(j[0] not in ('ns','ks','pde_compound'),j));rows=[]
    with ProcessPoolExecutor(max_workers=a.workers) as pool:
        for f in as_completed([pool.submit(one,j) for j in jobs]):r=f.result();rows.append(r);print(r,flush=True)
    dump(ROOT/'processes_fast.json',dict(rows=rows,source_sha256=hashes('run_suite_fast.py')));assert all(r['exit_code']==0 for r in rows)
if __name__=='__main__':main()
