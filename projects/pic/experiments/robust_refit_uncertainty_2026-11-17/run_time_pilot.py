from common import *
from concurrent.futures import ThreadPoolExecutor,as_completed
import subprocess,time

def one(seed):
    key=f'ac_positive_seed{seed}';start=time.monotonic();rows=[]
    for script in ('time_pilot_case.py','time_pilot_student.py'):
        log=ROOT/'time_pilot_logs'/(script+'_'+key+'.txt');log.parent.mkdir(exist_ok=True);args=['ac',str(seed),'positive'] if script=='time_pilot_case.py' else ['ac','positive',str(seed)]
        with log.open('w',encoding='utf-8') as f:r=subprocess.run([sys.executable,'-B','-X','utf8',str(ROOT/script),*args],stdout=f,stderr=subprocess.STDOUT)
        rows.append(dict(script=script,seed=seed,exit_code=r.returncode));print(rows[-1],flush=True)
        if r.returncode:break
    return dict(seed=seed,rows=rows,seconds=time.monotonic()-start)

def main():
    assert read(ROOT/'pilot_checks.json')['status']=='PASS';rows=[]
    with ThreadPoolExecutor(max_workers=3) as pool:
        for f in as_completed([pool.submit(one,s) for s in (48,49,50)]):rows.append(f.result())
    dump(ROOT/'time_pilot_processes.json',dict(rows=rows,source_sha256=hashes('run_time_pilot.py')));assert all(len(r['rows'])==2 and all(x['exit_code']==0 for x in r['rows']) for r in rows)
if __name__=='__main__':main()
