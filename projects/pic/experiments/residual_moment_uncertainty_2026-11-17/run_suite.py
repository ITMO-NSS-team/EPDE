from pathlib import Path
import sys,subprocess,json,time,argparse
from concurrent.futures import ThreadPoolExecutor,as_completed
ROOT=Path(__file__).resolve().parent
SYSTEMS=['ac','burgers','burgers_inviscid','duffing','kdv','kdv_cossin','ks','lorenz','lv','ns','ode','pde_compound','pde_divide','vdp','wave']


def run(job):
    name,seed=job;start=time.monotonic();p=ROOT/'logs'/f'{name}_seed{seed}.txt';p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('w',encoding='utf-8') as f:
        r=subprocess.run([sys.executable,'-B','-X','utf8',str(ROOT/'run_case.py'),'--system',name,'--seed',str(seed)],stdout=f,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    row=dict(system=name,seed=seed,exit_code=r.returncode,seconds=time.monotonic()-start);print(json.dumps(row),flush=True);return row


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--systems',nargs='+',default=SYSTEMS);ap.add_argument('--seeds',nargs='+',type=int,default=[24,25,26]);a=ap.parse_args()
    assert json.loads((ROOT/'checks.json').read_text())['status']=='PASS'
    with ThreadPoolExecutor(max_workers=7) as pool:rows=[f.result() for f in as_completed([pool.submit(run,(n,s)) for s in a.seeds for n in a.systems])]
    (ROOT/'process_results.json').write_text(json.dumps(rows,indent=2),encoding='utf-8');sys.exit(any(r['exit_code'] for r in rows))
