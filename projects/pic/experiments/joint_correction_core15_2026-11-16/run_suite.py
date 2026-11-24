"""Run independent cases in parallel with per-case logs and bounded memory."""
from pathlib import Path
import sys,subprocess,json,time
from concurrent.futures import ThreadPoolExecutor,as_completed
ROOT=Path(__file__).resolve().parent
SYSTEMS=['ac','burgers','burgers_inviscid','duffing','kdv','kdv_cossin','ks','lorenz','lv','ns','ode','pde_compound','pde_divide','vdp','wave']


def run(job):
    name,seed=job;start=time.monotonic();log=ROOT/'logs'/f'{name}_seed{seed}.txt';log.parent.mkdir(parents=True,exist_ok=True)
    if (ROOT/'results'/f'{name}_seed{seed}.json').exists():return dict(system=name,seed=seed,exit_code=0,skipped=True)
    with log.open('w',encoding='utf-8') as stream:
        result=subprocess.run([sys.executable,'-B','-X','utf8',str(ROOT/'run_case.py'),'--system',name,'--seed',str(seed)],
                              stdout=stream,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    out=dict(system=name,seed=seed,exit_code=result.returncode,seconds=time.monotonic()-start)
    print(json.dumps(out),flush=True);return out


if __name__=='__main__':
    jobs=[(n,s) for s in (12,13,14) for n in SYSTEMS]
    with ThreadPoolExecutor(max_workers=7) as pool:results=[f.result() for f in as_completed([pool.submit(run,j) for j in jobs])]
    (ROOT/'process_results.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    print('SUITE COMPLETE',sum(r['exit_code']==0 for r in results),'/',len(results),flush=True)
    sys.exit(any(r['exit_code']!=0 for r in results))
