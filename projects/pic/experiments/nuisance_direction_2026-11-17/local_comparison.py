from context import *
from concurrent.futures import ProcessPoolExecutor,as_completed
import local6_fit as L

def one(path):
    path=Path(path);d=read(path);e,_,_=C.primary('ode','iid',51);e.seed=d['seed'];e.pid,e.rho=S.PROFILES[d['profile']];y=e.true_field(0,True)['u'];assert R.digest(y)==d['raw_sha256']['u'];m,i=L.estimate(y)
    # Independent gradient replay at the stored optimizer solution.
    best=min(i['fits'],key=lambda r:r['value']);cc=L.Contrasts(y);v,g=cc.evaluate(np.array(best['theta']));np.testing.assert_allclose(v,i['objective'],rtol=1e-12);np.testing.assert_allclose(g,best['gradient'],atol=1e-11)
    out=dict(status='COMPLETE',profile=d['profile'],seed=d['seed'],model=m,info=i,sigma_ratio=m[0]/d['truth_for_evaluation']['u'][0],rho_error=m[1][0]-e.rho,raw_sha256=d['raw_sha256']['u'],source_sha256={**sources('local_comparison.py','local_comparison.txt'),str(Path(L.__file__)):sha(L.__file__)},input_sha256={str(path):sha(path)})
    dump(ROOT/'local_comparison'/path.name,out);return dict(profile=d['profile'],seed=d['seed'],status=i['status'],sigma_ratio=out['sigma_ratio'],rho_error=out['rho_error'])

def main():
    rows=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(one,str(p)) for p in sorted((ROOT/'survey').glob('*.json'))]):rows.append(f.result())
    dump(ROOT/'local_comparison.json',dict(status='COMPLETE',rows=rows,source_sha256=sources('local_comparison.py','local_comparison.txt')))
    for p in ('iid','positive','negative'):
        rr=[r for r in rows if r['profile']==p];print('local6',p,'median sigma',np.median([r['sigma_ratio'] for r in rr]),'max',max(r['sigma_ratio'] for r in rr),'median rho error',np.median([r['rho_error'] for r in rr]),'warnings',[r for r in rr if r['status']!='ok'],flush=True)

if __name__=='__main__':main()
