from pathlib import Path
import subprocess,sys,json,time
from concurrent.futures import ThreadPoolExecutor,as_completed
ROOT=Path(__file__).resolve().parent


def run(job):
    name,seed=job;start=time.monotonic();path=ROOT/'uncertainty_logs'/f'{name}_seed{seed}.txt';path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('w',encoding='utf-8') as f:
        result=subprocess.run([sys.executable,'-B','-X','utf8',str(ROOT/'uncertainty.py'),'--system',name,'--seed',str(seed)],
                              stdout=f,stderr=subprocess.STDOUT,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    row=dict(system=name,seed=seed,exit_code=result.returncode,seconds=time.monotonic()-start);print(json.dumps(row),flush=True);return row


if __name__=='__main__':
    changes=json.loads((ROOT/'changes.json').read_text())
    jobs={(n,s) for n in ('ac','kdv_cossin','ks','wave') for s in range(3)}
    jobs.update((r['system'],r['seed']) for r in changes if r['arm']=='q0_white128/joint' and r['metric']!='discrepancy_only')
    jobs=sorted(jobs);(ROOT/'uncertainty_jobs.json').write_text(json.dumps(jobs,indent=2),encoding='utf-8')
    with ThreadPoolExecutor(max_workers=7) as pool:rows=[f.result() for f in as_completed([pool.submit(run,j) for j in jobs])]
    (ROOT/'uncertainty_process_results.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    sys.exit(any(r['exit_code'] for r in rows))
