from context import *
from scipy.optimize import minimize
from concurrent.futures import ProcessPoolExecutor,as_completed
import noise_profile as PreviousProfile
import time

GRID=PreviousProfile.GRID
def one(profile,seed):
    start=time.monotonic();p=ROOT/'survey'/f'ode_{profile}_seed{seed}.json';d=read(p);e,_,_=C.primary('ode','iid',51);e.profile=profile;e.pid,e.rho=S.PROFILES[profile];e.seed=seed;y=e.true_field(0,True)['u'];assert R.digest(y)==d['raw_sha256']['u']
    cc=F.A.Contrasts(y);cc.fields=[cc.fields[0]];cc.count=cc.fields[0].size;scale=np.sqrt(np.mean(cc.fields[0]**2));cc.fields=[cc.fields[0]/scale]
    lo=np.array([-40.,-F.P.BOUND,-F.P.BOUND]);hi=np.array([20.,F.P.BOUND,F.P.BOUND]);bounds=list(zip(lo,hi));rows=[]
    def exact(u):
        t,jac=F.P.inverse(u,1)
        try:v,g,q=F.State.profile(cc,t[:1],np.exp(t[1:2]),t[2:3],t[3:]);return v,g*jac
        except (np.linalg.LinAlgError,FloatingPointError):return 1e30,np.zeros_like(u)
    for rho in GRID:
        def fun(u,state=False):
            full=np.r_[np.arctanh(rho),u]
            if state:v,g=exact(full);return v,g[1:]
            t,jac=F.P.inverse(full,1)
            try:v,g=F.A.profile_score(cc,t[:1],np.exp(t[1:2]),t[2:3],t[3:],F.A.covariance_band);return v,(g*jac)[1:]
            except (np.linalg.LinAlgError,FloatingPointError):return 1e30,np.zeros(3)
        starts=[np.array(d['fit_info']['u']['transformed'])[1:]]+[F.P.transform([rho,*s],1)[1:] for s in ((-5.,.99,-.8),(-3.,.95,-.5),(-11.,.9,0.),(-15.,.9,0.))];candidates=[]
        for u in starts:
            fit=minimize(fun,u,jac=True,method='L-BFGS-B',bounds=bounds,options=dict(ftol=1e-14,gtol=2e-7,maxiter=500,maxls=40));val,g=fun(fit.x,True);candidates.append((val,fit.x))
        val,u=min(candidates,key=lambda r:r[0]);v,g=fun(u,True);pg=max(abs(u-np.clip(u-g,lo,hi)))
        if pg>2e-5:
            fit=minimize(lambda u:fun(u,True),u,jac=True,method='L-BFGS-B',bounds=bounds,options=dict(ftol=1e-15,gtol=2e-7,maxiter=700,maxls=50));u=fit.x;v,g=fun(u,True);pg=max(abs(u-np.clip(u-g,lo,hi)))
        rows.append(dict(rho=rho,objective=v,transformed=np.r_[np.arctanh(rho),u],projected_gradient=pg))
    best=min(rows,key=lambda r:r['objective']);bl=np.r_[-F.P.BOUND,lo];bh=np.r_[F.P.BOUND,hi];fit=minimize(exact,best['transformed'],jac=True,method='L-BFGS-B',bounds=list(zip(bl,bh)),options=dict(ftol=1e-15,gtol=2e-7,maxiter=700,maxls=50));u=fit.x;v,g=exact(u);t,jac=F.P.inverse(u,1);_,_,q=F.State.profile(cc,t[:1],np.exp(t[1:2]),t[2:3],t[3:]);pg=float(max(abs(u-np.clip(u-g,bl,bh))));sig=float(scale*np.sqrt(q/cc.count))
    out=dict(status='ok' if pg<=2e-5 else 'unresolved_optimization',profile=profile,seed=seed,old_model=d['model'],old_objective=d['fit_info']['u']['objective'],grid=rows,new_model=[sig,[float(t[0])],[2]],new_info=dict(transformed=u,objective=v,projected_gradient=pg),likelihood_gain=cc.count*(d['fit_info']['u']['objective']-v),seconds=time.monotonic()-start,source_sha256=sources('profile_probe.py','profile_probe.txt'),input_sha256={str(p):sha(p)})
    dump(ROOT/'profile_probe'/p.name,out);print('PROFILE',profile,seed,'old',d['model']['u'][:2],'new',out['new_model'][:2],'gain',out['likelihood_gain'],'pg',pg,flush=True);return dict(profile=profile,seed=seed,likelihood_gain=out['likelihood_gain'],status=out['status'])

def main():
    rows=[]
    with ProcessPoolExecutor(max_workers=2) as pool:
        for f in as_completed([pool.submit(one,p,s) for p,s in [('positive',100),('positive',102),('positive',103),('positive',121),('iid',112),('negative',101)]]):rows.append(f.result())
    dump(ROOT/'profile_probe.json',dict(status='COMPLETE',rows=rows,source_sha256=sources('profile_probe.py','profile_probe.txt')))

if __name__=='__main__':main()
