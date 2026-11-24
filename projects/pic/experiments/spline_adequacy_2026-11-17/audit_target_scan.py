from target_bootstrap import *
from audit_bootstrap import replay
import traceback

def one(key):
    d,a,sid,pid=load(key);y,clean,sigma,rho,inputs=field(d['job']);assert np.array_equal(y,a['raw']);replay(y,d['info'],True);return dict(status='PASS',key=key,seed=d['job']['seed'],selected_p=d['info']['selection']['selected_p'])

def main():
    assert read(ROOT/'target_process.json')['status']=='COMPLETE';selection=read(ROOT/'target_selection.json');rows=[]
    with ProcessPoolExecutor(max_workers=8) as pool:
        for f in as_completed([pool.submit(one,key) for key in selection['scanned']]):
            rows.append(f.result())
            if len(rows)%64==0:print('target scan audit',len(rows),'of256',flush=True)
    eligible=sorted([r for r in rows if r['selected_p']==24],key=lambda r:r['seed']);assert [r['key'] for r in eligible[:8]]==selection['selected'];dump(ROOT/'target_scan_audit.json',dict(status='PASS',rows=rows,eligible=len(eligible),selected=selection['selected'],source_sha256=src('audit_target_scan.py','audit_bootstrap.py','target_bootstrap.py'),input_sha256={str(ROOT/'target_selection.json'):sha(ROOT/'target_selection.json')}));print('TARGET SCAN AUDIT PASS',len(eligible),'eligible',flush=True)

if __name__=='__main__':main()
