"""Exact initial score with three additional persistent-signal starts."""
from calibration_context import *
import exact_start_fit as Fast
from scipy.optimize import minimize


def estimate(y):
    old_initial=F.estimate
    def initial(values):
        model,info=old_initial(values)
        cc=F.A.Contrasts(values);cc.fields=[cc.fields[0]];cc.count=cc.fields[0].size;sc=np.sqrt(np.mean(cc.fields[0]**2));cc.fields=[cc.fields[0]/sc]
        bounds=[(-F.P.BOUND,F.P.BOUND),(-40.,20.),(-F.P.BOUND,F.P.BOUND),(-F.P.BOUND,F.P.BOUND)]
        def objective(u):
            t,jac=F.P.inverse(u,1)
            try:v,g,q=F.State.profile(cc,t[:1],np.exp(t[1:2]),t[2:3],t[3:]);return v,g*jac
            except (np.linalg.LinAlgError,FloatingPointError):return 1e30,np.zeros_like(u)
        for rho in (-.8,0.,.8):
            opt=minimize(objective,F.P.transform([rho,-5.,.99,-.95],1),jac=True,method='L-BFGS-B',bounds=bounds,options=dict(ftol=1e-14,gtol=2e-7,maxiter=500,maxls=40))
            info['fits'].append(dict(transformed=opt.x,value=float(opt.fun),success=bool(opt.success),iterations=int(opt.nit),message=str(opt.message)))
        return model,info
    F.estimate=initial
    try:model,info=Fast.estimate(y)
    finally:F.estimate=old_initial
    return model,dict(info,initial_start_count=7)
