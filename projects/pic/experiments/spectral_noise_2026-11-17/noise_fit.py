import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='1'
import numpy as np
from scipy.fft import dctn
from scipy.optimize import minimize

EPS_RHO=1e-6

def axis_precision(n,rho):
    k=np.arange(n);cos=np.cos(np.pi*k/n)
    diagonal=(1+rho*rho-2*rho*cos)/(1-rho*rho)
    weights=np.sqrt(2./n)*np.ones(n);weights[0]=np.sqrt(1./n)
    endpoints=np.column_stack((weights*np.cos(np.pi*k/(2*n)),weights*np.cos(np.pi*k*(n-.5)/n)))
    return diagonal,endpoints,rho/(1+rho)

def small_logdet(a):
    sign,value=np.linalg.slogdet(a)
    if sign<=0:raise FloatingPointError('Nonpositive determinant in stationary covariance')
    return float(value)

def block(diag,u,k):
    # A=diag(diag)+u k u.T, with k symmetric2x2.
    middle=np.eye(2)+k@(u.T@(u/diag[:,None]))
    logdet=float(np.log(diag).sum())+small_logdet(middle)
    return (diag,u,k),logdet

def apply(a,x,axis,inverse=False):
    d,u,k=a;shape=x.shape;y=np.moveaxis(x,axis,0).reshape(len(d),-1)
    if inverse:
        inv=1/d;v=u*inv[:,None];middle=np.eye(2)+k@(u.T@v)
        result=inv[:,None]*y-v@np.linalg.solve(middle,k@(v.T@y))
    else:result=d[:,None]*y+u@(k@(u.T@y))
    return np.moveaxis(result.reshape((shape[axis],)+tuple(shape[j] for j in range(len(shape)) if j!=axis)),0,axis)

class Contrast:
    def __init__(self,y,design='strips'):
        self.shape=y.shape;self.design=design;self.low=tuple(slice(0,(n+1)//2) for n in y.shape)
        self.high=tuple(slice((n+1)//2,n) for n in y.shape)
        z=dctn(y,type=2,norm='ortho')
        if design=='corner':self.z=z[self.high].copy();self.count=self.z.size
        elif design=='strips':
            z[self.low]=0.;self.z=z;self.count=z.size-int(np.prod([(n+1)//2 for n in y.shape]))
        else:raise ValueError(design)
        assert self.count>0

    def evaluate(self,rhos):
        full=[];low=[];high=[];logdet=0.;nfull=int(np.prod(self.shape));nlow=int(np.prod([(n+1)//2 for n in self.shape]))
        for n,rho in zip(self.shape,rhos):
            d,u,alpha=axis_precision(n,rho);l=(n+1)//2;k=alpha*np.eye(2)
            if self.design=='strips':
                full.append((d,u,k));ll,ld=block(d[:l],u[:l],k);low.append(ll)
                logdet+=(nfull/n)*(n-1)*np.log1p(-rho*rho)+(nlow/l)*ld
            else:
                middle=np.eye(2)+alpha*(u[:l].T@(u[:l]/d[:l,None]))
                kh=alpha*np.linalg.inv(middle);kh=(kh+kh.T)/2
                hh,ld=block(d[l:],u[l:],kh);high.append(hh);logdet-=self.count/(n-l)*ld
        q=self.z.copy()
        for axis,p in enumerate(full if self.design=='strips' else high):q=apply(p,q,axis)
        quadratic=float(np.sum(self.z*q))
        if self.design=='strips':
            b=q[self.low].copy();sol=b.copy()
            for axis,p in enumerate(low):sol=apply(p,sol,axis,inverse=True)
            quadratic-=float(np.sum(b*sol))
        if not np.isfinite(quadratic) or quadratic<=0:raise FloatingPointError('Nonpositive contrast quadratic')
        return quadratic,float(logdet)

def estimate(y,design='strips'):
    scale=float(np.std(y))
    if not np.isfinite(scale) or scale<=0:return None,dict(status='no_variation')
    # Global centering and scaling affect only removed DC/unit convention.
    contrast=Contrast((y-np.mean(y))/scale,design);d=y.ndim
    def objective(rhos):
        q,ld=contrast.evaluate(rhos)
        return np.log(q/contrast.count)+ld/contrast.count
    fits=[]
    for start in (0.,.5,-.5):
        result=minimize(objective,np.full(d,start),method='L-BFGS-B',bounds=[(-1+EPS_RHO,1-EPS_RHO)]*d,
            options={'ftol':1e-12,'gtol':2e-7,'maxiter':250,'maxls':50,'finite_diff_rel_step':1e-5})
        fits.append(dict(success=bool(result.success),value=float(result.fun),rho=result.x.tolist(),message=str(result.message),iterations=int(result.nit)))
    good=[f for f in fits if f['success'] and np.isfinite(f['value'])]
    if not good:return None,dict(status='optimizer_failure',fits=fits)
    best=min(good,key=lambda f:f['value']);rho=np.asarray(best['rho']);q,ld=contrast.evaluate(rho);sigma=scale*np.sqrt(q/contrast.count)
    boundary=bool(np.any(np.abs(rho)>1-10*EPS_RHO))
    info=dict(status='boundary' if boundary else 'ok',sigma=float(sigma),rhos=rho.tolist(),design=design,
        retained=contrast.count,shape=list(y.shape),objective=best['value'],fits=fits,successful_objective_spread=max(f['value'] for f in good)-min(f['value'] for f in good))
    return [float(sigma),rho.tolist(),[0]*d],info
