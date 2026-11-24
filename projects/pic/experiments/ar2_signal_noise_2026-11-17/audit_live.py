"""Run the unchanged comprehensive case auditor as immutable results arrive."""
from pathlib import Path
import sys,json,importlib.util,time,argparse
ROOT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('latent_case_auditor',ROOT/'audit.py');A=importlib.util.module_from_spec(spec);sys.modules[spec.name]=A;spec.loader.exec_module(A)
S,R=A.S,A.R
from concurrent.futures import ProcessPoolExecutor,wait,FIRST_COMPLETED

def check(path):
    out=A.safe_one(path);out['result_sha256']=S.E.sha(path);out['auditor_sha256']=S.E.sha(ROOT/'audit.py');return out

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workers',type=int,default=7);parser.add_argument('--initial-workers',type=int,default=2);args=parser.parse_args();seen=set();pending={};rows=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        while len(rows)<138:
            limit=args.workers if (ROOT/'processes.json').exists() else args.initial_workers
            files=sorted((ROOT/'results').glob('*.json'),key=lambda p:(not p.stem.startswith(('ks_','ns_')),p.name))
            for p in files:
                if p.stem not in seen and len(pending)<limit:
                    seen.add(p.stem);pending[pool.submit(check,str(p))]=p
            if pending:
                done,_=wait(pending,timeout=10,return_when=FIRST_COMPLETED)
                for future in done:
                    p=pending.pop(future);row=future.result();rows.append(row);R.dump(ROOT/'audit_records'/p.name,row);print(row['record'],row['status'],flush=True)
            else:time.sleep(5)
    while not (ROOT/'processes.json').exists():time.sleep(5)
    process=json.loads((ROOT/'processes.json').read_text())['rows'];assert len(process)==138 and all(r['exit_code']==0 for r in process)
    files=sorted((ROOT/'results').glob('*.json'));assert len(files)==138
    expected={f"{r['system']}_{r['profile']}_{r['model']}_seed{r['seed']}" for r in process};assert {p.stem for p in files}==expected=={r['record'] for r in rows}
    paired={}
    for p in files:
        d=json.loads(p.read_text());key=(d['system'],d['profile'],d['seed']);hashes=(d['raw_sha256'],d.get('actual_field_hashes'))
        if key in paired:
            assert paired[key][0]==hashes[0]
            if paired[key][1] and hashes[1]:assert paired[key][1]==hashes[1]
        else:paired[key]=hashes
    for row in rows:
        assert S.E.sha(ROOT/'results'/(row['record']+'.json'))==row['result_sha256'];assert S.E.sha(ROOT/'audit.py')==row['auditor_sha256']
    failed=[r for r in rows if r['status']!='PASS']
    if failed:R.dump(ROOT/'audit_failed.json',dict(rows=rows));raise AssertionError(f'{len(failed)} audit failures')
    keys=('hash_links','operator_scores','registered_scores','regions','intersection_certificates','matrix_replays');totals={k:sum(r[k] for r in rows) for k in keys}
    out=dict(status='PASS',records=138,workers=args.workers,incomplete=sum(r['incomplete'] for r in rows),paired_fields=True,totals=totals,rows=rows,
        source_sha256={str(p):S.E.sha(p) for p in (Path(__file__),ROOT/'audit.py',S.ROOT/'audit_joint.py')})
    R.dump(ROOT/'audit.json',out);print(totals,flush=True)

if __name__=='__main__':main()
