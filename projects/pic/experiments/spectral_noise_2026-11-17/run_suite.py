from pathlib import Path
import sys,json,time,subprocess,argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
ROOT=Path(__file__).resolve().parent
SYSTEMS=['ac','burgers','burgers_inviscid','duffing','kdv','kdv_cossin','ks','lorenz','lv','ns','ode','pde_compound','pde_divide','vdp','wave']

def jobs():
    return [(n,s,'iid',m) for s in (33,34,35) for n in SYSTEMS for m in ('spectral','iid')]+[(n,s,p,m) for s in (33,34,35) for n in ('ac','kdv_cossin','ks','wave') for p in ('positive','negative') for m in ('spectral','ar','known_rho')]

def worker(job):
    n,s,p,m=job;start=time.monotonic();f=ROOT/'logs'/f'{n}_{p}_{m}_seed{s}.txt';f.parent.mkdir(exist_ok=True)
    with f.open('w',encoding='utf-8') as out:r=subprocess.run([sys.executable,'-B','-X','utf8',str(ROOT/'run_case.py'),n,str(s),p,m],stdout=out,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    row=dict(system=n,seed=s,profile=p,model=m,exit_code=r.returncode,seconds=time.monotonic()-start);print(json.dumps(row),flush=True);return row

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=7);args=ap.parse_args()
    for p in ('checks.json','check_stable.json'):assert json.loads((ROOT/p).read_text())['status']=='PASS'
    with ThreadPoolExecutor(max_workers=args.workers) as pool:rows=[f.result() for f in as_completed([pool.submit(worker,j) for j in jobs()])]
    (ROOT/'processes.json').write_text(json.dumps(dict(workers=args.workers,rows=rows),indent=2),encoding='utf-8');sys.exit(any(r['exit_code'] for r in rows))
