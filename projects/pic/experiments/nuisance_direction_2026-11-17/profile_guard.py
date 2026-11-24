from context import *
from scipy.optimize import minimize

def estimate(y,original=None):
    model,info=Repair.estimate(y) if original is None else original
    trigger=abs(model[1][0])>.95 or info['status']!='ok'
    if not trigger:return model,dict(info,guard_triggered=False,guard_rows=[],guard_improvement=0.,unguarded_model=model)
    cc=F.A.Contrasts(y);cc.fields=[cc.fields[0]];cc.count=cc.fields[0].size;sc=np.sqrt(np.mean(cc.fields[0]**2));cc.fields=[cc.fields[0]/sc]
    lo=np.array([-F.P.BOUND,-40.,-F.P.BOUND,-F.P.BOUND]);hi=np.array([F.P.BOUND,20.,F.P.BOUND,F.P.BOUND]);bounds=list(zip(lo,hi));rows=[]
    def objective(u,exact=True):
        t,jac=F.P.inverse(u,1)
        try:
            if exact:v,g,q=F.State.profile(cc,t[:1],np.exp(t[1:2]),t[2:3],t[3:])
            else:v,g=F.A.profile_score(cc,t[:1],np.exp(t[1:2]),t[2:3],t[3:],F.A.covariance_band)
            return v,g*jac
        except (np.linalg.LinAlgError,FloatingPointError):return 1e30,np.zeros_like(u)
    def record(tag,u):
        v,g=objective(u);pg=float(max(abs(u-np.clip(u-g,lo,hi))));return dict(start=tag,transformed=u,objective=v,projected_gradient=pg,status='ok' if pg<=2e-5 and v<1e29 else 'unresolved_optimization')
    rows.append(record('unguarded_final',np.array(info['transformed'])))
    for rho in (-.8,-.4,0.,.4,.8):
        def fn(u,exact=False):v,g=objective(np.r_[np.arctanh(rho),u],exact);return v,g[1:]
        starts=[np.array(info['transformed'])[1:]]+[F.P.transform([rho,*s],1)[1:] for s in ((-5.,.99,-.8),(-3.,.95,-.5),(-11.,.9,0.),(-15.,.9,0.))];choices=[]
        for u in starts:
            opt=minimize(fn,u,jac=True,method='L-BFGS-B',bounds=bounds[1:],options=dict(ftol=1e-14,gtol=2e-7,maxiter=500,maxls=40));v,g=fn(opt.x,True);choices.append(dict(transformed=opt.x,objective=v))
        best=min(choices,key=lambda r:r['objective']);u=np.array(best['transformed']);v,g=fn(u,True)
        if max(abs(u-np.clip(u-g,lo[1:],hi[1:])))>2e-5:
            opt=minimize(lambda u:fn(u,True),u,jac=True,method='L-BFGS-B',bounds=bounds[1:],options=dict(ftol=1e-15,gtol=2e-7,maxiter=700,maxls=50));u=opt.x
        fixed=record(f'fixed_rho_{rho}',np.r_[np.arctanh(rho),u]);opt=minimize(objective,fixed['transformed'],jac=True,method='L-BFGS-B',bounds=bounds,options=dict(ftol=1e-15,gtol=2e-7,maxiter=700,maxls=50));r=record(f'released_rho_{rho}',opt.x);r['fixed_rho']=fixed;r['nuisance_starts']=choices;rows.append(r)
    best=min(rows,key=lambda r:r['objective']);u=np.array(best['transformed']);t,jac=F.P.inverse(u,1);v,g,q=F.State.profile(cc,t[:1],np.exp(t[1:2]),t[2:3],t[3:]);sigma=float(sc*np.sqrt(q/cc.count));rhos=[float(t[0])]+list(model[1][1:]);status=best['status']
    if max(abs(r) for r in rhos)>1-1e-5 or t[1]>19.99:status='boundary'
    assert v<=info['objective']+1e-12
    return [sigma,rhos,model[2]],dict(info,status=status,sigma=sigma,rhos=rhos,transformed=u,theta=t,objective=v,projected_gradient=best['projected_gradient'],guard_triggered=True,guard_rows=rows,guard_improvement=info['objective']-v,unguarded_model=model,selected_guard_start=best['start'])
