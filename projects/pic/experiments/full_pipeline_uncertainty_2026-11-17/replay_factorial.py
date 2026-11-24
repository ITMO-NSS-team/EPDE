from bridge import *
import subprocess,time

def main():
    rows=[]
    for j in range(3):
        path=ROOT/'factorial_replay_logs'/f'pass{j+1}.txt';path.parent.mkdir(exist_ok=True);start=time.monotonic()
        with path.open('w',encoding='utf-8') as f:r=subprocess.run([sys.executable,'-B','-X','utf8',str(ROOT/'audit_factorial.py')],stdout=f,stderr=subprocess.STDOUT)
        rows.append(dict(replay=j+1,exit_code=r.returncode,seconds=time.monotonic()-start,log_sha256=sha(path)));print(rows[-1],flush=True)
        if r.returncode:break
    dump(ROOT/'factorial_reproducibility.json',dict(status='PASS' if len(rows)==3 and all(r['exit_code']==0 for r in rows) else 'FAIL',rows=rows,source_sha256=sources('replay_factorial.py','audit_factorial.py')))
if __name__=='__main__':main()
