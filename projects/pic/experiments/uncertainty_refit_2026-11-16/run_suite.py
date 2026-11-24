from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import subprocess,sys,time,json,argparse
ROOT=Path(__file__).resolve().parent
SYSTEMS=['ac','burgers','burgers_inviscid','duffing','kdv','kdv_cossin','ks','lorenz','lv','ns','ode','pde_compound','pde_divide','vdp','wave']


def worker(job,replay):
    name,seed=job;start=time.monotonic();log=ROOT/('replay_logs' if replay else 'logs')/f'{name}_seed{seed}.txt';log.parent.mkdir(parents=True,exist_ok=True)
    cmd=[sys.executable,'-B','-X','utf8',str(ROOT/'experiment.py'),'--system',name,'--seed',str(seed)]
    if replay:cmd.append('--replay')
    with log.open('w',encoding='utf-8') as f:result=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    row=dict(system=name,seed=seed,exit_code=result.returncode,seconds=time.monotonic()-start);print(json.dumps(row),flush=True);return row


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--replay',action='store_true');a=ap.parse_args()
    jobs=[(n,0) for n in ('ac','kdv_cossin','ks','wave')] if a.replay else [(n,s) for s in (18,19,20) for n in SYSTEMS]
    if not a.replay:
        assert all(json.loads((ROOT/'replay'/f'{n}.json').read_text())['status']=='PASS' for n in ('ac','kdv_cossin','ks','wave'))
    with ThreadPoolExecutor(max_workers=7) as pool:rows=[f.result() for f in as_completed([pool.submit(worker,j,a.replay) for j in jobs])]
    (ROOT/('replay_processes.json' if a.replay else 'process_results.json')).write_text(json.dumps(rows,indent=2),encoding='utf-8')
    sys.exit(any(r['exit_code'] for r in rows))
