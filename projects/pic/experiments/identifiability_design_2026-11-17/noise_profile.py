from research_context import *
from scipy.optimize import minimize
from concurrent.futures import ProcessPoolExecutor,as_completed
import argparse,time

GRID=[-.8,-.4,0.,.2,.4,.6,.8,.9,.97,.99,.999,.9999,.99999,.999999]


def one(job):
    key,index=job;rho=GRID[index];start=time.monotonic();path=ROOT/'noise_inputs'/(key+'.json');d=read(path)
    with np.load(path.with_suffix('.npz')) as z:y=z['field']
    assert R.digest(y)==d['field_sha256'];cc=F.A.Contrasts(y);cc.fields=[cc.fields[0]];cc.count=cc.fields[0].size;scale=np.sqrt(np.mean(cc.fields[0]**2));cc.fields=[cc.fields[0]/scale]
    lo=np.array([-40.,-F.P.BOUND,-F.P.BOUND]);hi=np.array([20.,F.P.BOUND,F.P.BOUND]);bounds=list(zip(lo,hi))
    def objective(u,state=False):
        full=np.r_[np.arctanh(rho),u];t,jac=F.P.inverse(full,1)
        try:
            if state:value,g,q=F.State.profile(cc,t[:1],np.exp(t[1:2]),t[2:3],t[3:])
            else:value,g=F.A.profile_score(cc,t[:1],np.exp(t[1:2]),t[2:3],t[3:],F.A.covariance_band)
            return value,(g*jac)[1:]
        except (np.linalg.LinAlgError,FloatingPointError):return 1e30,np.zeros(3)
    starts=[np.array(d['fit_info']['transformed'])[1:]]
    starts += [F.P.transform([rho,*s],1)[1:] for s in ((-5.,.99,-.8),(-3.,.95,-.5),(-11.,.9,0.),(-15.,.9,0.))]
    candidates=[]
    for u in starts:
        fit=minimize(objective,u,jac=True,method='L-BFGS-B',bounds=bounds,options=dict(ftol=1e-14,gtol=2e-7,maxiter=500,maxls=40))
        value,g=objective(fit.x,True);pg=float(np.max(abs(fit.x-np.clip(fit.x-g,lo,hi))))
        candidates.append(dict(transformed=fit.x,value=value,projected_gradient=pg,optimizer_success=bool(fit.success)))
    best=min(candidates,key=lambda r:r['value']);u=np.array(best['transformed']);value,g=objective(u,True);pg=float(np.max(abs(u-np.clip(u-g,lo,hi))))
    if pg>2e-5:
        fit=minimize(lambda x:objective(x,True),u,jac=True,method='L-BFGS-B',bounds=bounds,options=dict(ftol=1e-15,gtol=2e-7,maxiter=700,maxls=50));u=fit.x;value,g=objective(u,True);pg=float(np.max(abs(u-np.clip(u-g,lo,hi))))
    t,jac=F.P.inverse(np.r_[np.arctanh(rho),u],1);_,_,q=F.State.profile(cc,t[:1],np.exp(t[1:2]),t[2:3],t[3:]);sigma=float(scale*np.sqrt(q/cc.count))
    out=dict(status='ok' if pg<=2e-5 else 'unresolved_optimization',key=key,index=index,rho=rho,sigma=sigma,
             innovation_sigma=sigma*np.sqrt(1-rho*rho),objective=value,projected_gradient=pg,theta=t,transformed=u,gradient=g,
             contrast_count=cc.count,scale=scale,starts=candidates,seconds=time.monotonic()-start,
             source_sha256=sources('noise_profile.py','protocol.txt'),input_sha256={str(path):sha(path),str(path.with_suffix('.npz')):sha(path.with_suffix('.npz'))})
    dump(ROOT/'noise_profiles'/f'{key}_rho{index:02d}.json',out);return dict(key=key,rho=rho,status=out['status'],seconds=out['seconds'])


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=6);args=ap.parse_args()
    keys=read(ROOT/'noise_preparation.json')['objects'];jobs=[(k,i) for k in keys for i in range(len(GRID))];rows=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for f in as_completed([pool.submit(one,j) for j in jobs]):r=f.result();rows.append(r);print(r,flush=True)
    dump(ROOT/'noise_process.json',dict(status='COMPLETE',rows=rows,source_sha256=sources('noise_profile.py')))


if __name__=='__main__':main()
