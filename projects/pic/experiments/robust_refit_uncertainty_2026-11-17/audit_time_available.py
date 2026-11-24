from common import *
import audit_time as Q
from concurrent.futures import ProcessPoolExecutor
import time

def main():
    seen=set();pending={};rows=[]
    with ProcessPoolExecutor(max_workers=1) as pool:
        while len(rows)<101:
            files=[p for folder in ('time_screen','time_validation','time_noise') for p in sorted((ROOT/folder).glob('*.json'))]
            for path in files:
                if str(path) not in seen and not pending:pending[pool.submit(Q.one,str(path))]=path;seen.add(str(path))
            for f in list(pending):
                if f.done():r=f.result();pending.pop(f);rows.append(r);print(r['record'],'PASS',flush=True)
            time.sleep(.2)
    out=dict(status='PASS',records=len(rows),totals={k:sum(r[k] for r in rows) for k in ('hash_links','field_hashes','score_replays')},rows=rows,source_sha256=hashes('audit_time.py','audit_time_available.py','audit_followups.py'))
    dump(ROOT/'audit_time.json',out);print(out['totals'],flush=True)
if __name__=='__main__':main()
