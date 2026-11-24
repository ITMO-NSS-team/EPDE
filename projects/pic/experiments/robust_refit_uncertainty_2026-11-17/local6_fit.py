"""Gaussian-consistent robust AR1 fit after local polynomial elimination."""
from common import np
from scipy.linalg import cholesky,cho_solve
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.integrate import quad
from functools import lru_cache
NU=4.
BOUND=float(np.arctanh(1-1e-6))

@lru_cache(None)
def geometry(n):
    x=np.linspace(-1,1,n);q=np.linalg.qr(np.vander(x,6,increasing=True),mode='complete')[0][:,6:]
    lag=np.abs(np.arange(n)[:,None]-np.arange(n)[None,:]);m=n-6
    def integrand(x):
        if x<=0:return 0.
        density=np.exp((m/2-1)*np.log(x)-x/2-(m/2)*np.log(2)-gammaln(m/2))
        return (NU+m)*x/(NU+x)*density/m
    alpha=quad(integrand,0,np.inf,epsabs=2e-12,epsrel=2e-12)[0]
    return q,lag,alpha

class Contrasts:
    def __init__(self,y):
        self.fields=[];self.count=0;self.dim=y.ndim
        for axis,n in enumerate(y.shape):
            if n<8:raise ValueError('Axis too short for quintic contrasts')
            z=np.moveaxis(y,axis,0).reshape(n,-1);edges=list(range(0,n,12))+[n]
            if len(edges)>2 and edges[-1]-edges[-2]<8:edges.pop(-2)
            groups={}
            for a,b in zip(edges[:-1],edges[1:]):groups.setdefault(b-a,[]).append(z[a:b])
            for length,blocks in groups.items():
                basis,lag,alpha=geometry(length);d=basis.T@np.concatenate(blocks,axis=1)
                self.fields.append((axis,d,basis,lag,alpha));self.count+=d.size
        self.scale=np.sqrt(sum(float(np.sum(d*d)) for _,d,*_ in self.fields)/self.count)
        if not np.isfinite(self.scale) or self.scale<=np.finfo(float).tiny:raise ValueError('No contrast variation')
        self.fields=[(a,d/self.scale,q,l,alpha) for a,d,q,l,alpha in self.fields]

    def evaluate(self,t,gradient=True):
        sig=np.exp(t[0]);rho=np.tanh(t[1:]);total=0.;g=np.zeros(self.dim+1)
        for axis,d,basis,lag,alpha in self.fields:
            r=rho[axis];cov=r**lag;deriv=np.zeros_like(cov);mask=lag>0;deriv[mask]=lag[mask]*r**(lag[mask]-1)
            v=basis.T@cov@basis;dv=basis.T@deriv@basis;chol=cholesky(v,lower=True,check_finite=False)
            sol=cho_solve((chol,True),d,check_finite=False);quadval=np.sum(d*sol,axis=0)/sig**2
            m,nrep=d.shape;w=(NU+m)/(NU+quadval);ld=2*np.log(np.diag(chol)).sum()
            total+=alpha*nrep*(ld+2*m*t[0])+(NU+m)*np.log1p(quadval/NU).sum()
            g[0]+=2*(alpha*nrep*m-np.sum(w*quadval))
            if gradient:
                trace=float(np.trace(cho_solve((chol,True),dv,check_finite=False)))
                g[axis+1]+=(alpha*nrep*trace-np.sum(w*np.sum(sol*(dv@sol),axis=0))/sig**2)*(1-r*r)
        return float(total/self.count),g/self.count

def estimate(y,warm=None):
    cc=Contrasts(y);bounds=[(-20.,10.)]+[(-BOUND,BOUND)]*cc.dim
    d=np.concatenate([d.ravel() for _,d,*_ in cc.fields]);initial=max(float(np.median(np.abs(d-np.median(d)))/.6744897501960817),1e-8)
    starts=[np.r_[np.log(initial),np.full(cc.dim,np.arctanh(r))] for r in (0.,.4,-.4)]
    if warm is not None:starts=[np.r_[np.log(warm[0]/cc.scale),np.arctanh(np.clip(warm[1],-1+1e-6,1-1e-6))]]+starts
    fits=[]
    def fun(t):
        try:return cc.evaluate(t)
        except (np.linalg.LinAlgError,FloatingPointError):return 1e30,np.zeros_like(t)
    for start in starts:
        opt=minimize(fun,start,jac=True,method='L-BFGS-B',bounds=bounds,options=dict(ftol=1e-14,gtol=2e-7,maxiter=300,maxls=40))
        value,grad=fun(opt.x);lo,hi=np.array(bounds).T;pg=float(np.max(np.abs(opt.x-np.clip(opt.x-grad,lo,hi))))
        fits.append(dict(theta=opt.x,value=value,gradient=grad,projected_gradient=pg,success=bool(opt.success),message=str(opt.message),iterations=int(opt.nit)))
    best=min(fits,key=lambda f:f['value']);t=np.array(best['theta']);sigma=cc.scale*np.exp(t[0]);rhos=np.tanh(t[1:])
    status='ok' if best['projected_gradient']<=2e-5 and best['value']<1e20 else 'unresolved_optimization'
    if np.any(np.abs(rhos)>1-1e-5):status='boundary'
    model=[float(sigma),rhos.tolist(),[3]*cc.dim]
    return model,dict(status=status,sigma=sigma,rhos=rhos,objective=best['value'],projected_gradient=best['projected_gradient'],fits=fits,contrast_count=cc.count,shape=y.shape)
