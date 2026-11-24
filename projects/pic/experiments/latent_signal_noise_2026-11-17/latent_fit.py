import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
import numpy as np
from scipy.linalg import cholesky_banded,cho_solve_banded
from scipy.optimize import minimize

def covariance_band(n,rho,kappa):
    """A*(D2 R D2.T + kappa I)*A.T; det(A)=1, A=I-rho*lag."""
    h=(1-rho)*(1+rho);ab=np.zeros((3,n))
    ab[0]=6*h+kappa*(1+rho*rho);ab[0,0]=2*(1-rho)*(3-rho)+kappa
    ab[1,:-1]=-4*h-kappa*rho
    if n>1:ab[1,0]=-h*(4-rho)-kappa*rho
    ab[2,:-2]=h
    return ab

class Contrasts:
    def __init__(self,y):
        self.shape=y.shape;self.fields=[];self.count=0
        for axis in range(y.ndim):
            d=np.moveaxis(np.diff(y,n=2,axis=axis),axis,0);d=d.reshape(d.shape[0],-1)
            if len(d)<3:raise ValueError('Axis too short for curvature contrast')
            self.fields.append(d);self.count+=d.size

    def evaluate(self,rhos,kappas):
        total=0.;logdet=0.
        for d,r,k in zip(self.fields,rhos,kappas):
            ab=covariance_band(len(d),r,k);chol=cholesky_banded(ab,lower=True,check_finite=False)
            filtered=d.copy();filtered[1:]-=r*d[:-1]
            sol=cho_solve_banded((chol,True),filtered,check_finite=False)
            total+=float(np.sum(filtered*sol));logdet+=d.shape[1]*2*float(np.log(chol[0]).sum())
        if total<=0 or not np.isfinite(total):raise FloatingPointError('Invalid contrast quadratic')
        return total,logdet

def estimate(y):
    cc=Contrasts(y);scale=np.sqrt(sum(float(np.sum(d*d)) for d in cc.fields)/cc.count)
    if not np.isfinite(scale) or scale<=np.finfo(float).eps*max(float(np.max(np.abs(y))),np.finfo(float).tiny):return None,dict(status='no_variation')
    for i in range(len(cc.fields)):cc.fields[i]=cc.fields[i]/scale
    dim=y.ndim;failed_evaluations=0
    def objective(theta):
        nonlocal failed_evaluations
        try:q,ld=cc.evaluate(theta[:dim],np.exp(theta[dim:]))
        except (np.linalg.LinAlgError,FloatingPointError):failed_evaluations+=1;return 1e50
        return float(np.log(q/cc.count)+ld/cc.count)
    fits=[]
    for r,lk in ((0.,-7.),(.4,-3.),(-.4,-11.)):
        start=np.r_[np.full(dim,r),np.full(dim,lk)]
        opt=minimize(objective,start,method='L-BFGS-B',bounds=[(-1+1e-6,1-1e-6)]*dim+[(-40.,20.)]*dim,
            options={'ftol':1e-12,'gtol':2e-7,'maxiter':250,'maxls':40})
        fits.append(dict(success=bool(opt.success),value=float(opt.fun),theta=opt.x.tolist(),iterations=int(opt.nit),message=str(opt.message)))
    good=[f for f in fits if f['success'] and np.isfinite(f['value']) and f['value']<1e40]
    if not good:return None,dict(status='optimizer_failure',fits=fits,failed_evaluations=failed_evaluations)
    best=min(good,key=lambda f:f['value']);theta=np.asarray(best['theta']);rhos=theta[:dim];kappas=np.exp(theta[dim:]);q,ld=cc.evaluate(rhos,kappas);sigma=scale*np.sqrt(q/cc.count)
    status='boundary' if np.any(np.abs(rhos)>1-1e-5) or np.any(theta[dim:]>19.99) else 'ok'
    info=dict(status=status,sigma=float(sigma),rhos=rhos.tolist(),kappas=kappas.tolist(),signal_curvature_sigma=(sigma*np.sqrt(kappas)).tolist(),
        log_kappas=theta[dim:].tolist(),shape=list(y.shape),contrast_count=cc.count,objective=best['value'],fits=fits,failed_evaluations=failed_evaluations,
        successful_objective_spread=max(f['value'] for f in good)-min(f['value'] for f in good),likelihood='exact' if dim==1 else 'axis_composite')
    return [float(sigma),rhos.tolist(),[2]*dim],info
