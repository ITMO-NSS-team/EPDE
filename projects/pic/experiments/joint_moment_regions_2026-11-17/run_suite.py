from pathlib import Path
import sys,json,time,subprocess,argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
ROOT=Path(__file__).resolve().parent
SYSTEMS=['ac','burgers','burgers_inviscid','duffing','kdv','kdv_cossin','ks','lorenz','lv','ns','ode','pde_compound','pde_divide','vdp','wave']

def worker(job):
    n,s,p,m=job;start=time.monotonic();f=ROOT/'logs'/f'{n}_{p}_{m}_seed{s}.txt';f.parent.mkdir(parents=True,exist_ok=True)
    with f.open('w',encoding='utf-8') as out:r=subprocess.run([sys.executable,'-B','-X','utf8',str(ROOT/'run_case.py'),n,str(s),p,m],stdout=out,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    row=dict(system=n,seed=s,profile=p,model=m,exit_code=r.returncode,seconds=time.monotonic()-start);print(json.dumps(row),flush=True);return row

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--cohort',choices=['iid','correlated'],default='iid');args=ap.parse_args()
    assert json.loads((ROOT/'checks.json').read_text())['status']=='PASS'
    jobs=[(n,s,'iid','iid') for s in (30,31,32) for n in SYSTEMS] if args.cohort=='iid' else [(n,s,p,m) for s in (30,31,32) for n in ('ac','kdv_cossin','ks','wave') for p in ('positive','negative') for m in ('iid','ar','known_rho')]
    with ThreadPoolExecutor(max_workers=7) as pool:rows=[f.result() for f in as_completed([pool.submit(worker,j) for j in jobs])]
    (ROOT/(args.cohort+'_processes.json')).write_text(json.dumps(rows,indent=2),encoding='utf-8');sys.exit(any(r['exit_code'] for r in rows))
