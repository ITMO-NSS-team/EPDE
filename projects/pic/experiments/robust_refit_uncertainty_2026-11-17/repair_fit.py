"""Same AR2 objective, transformed parameters and explicit stationarity test."""
from common import A,np
from scipy.optimize import minimize
BOUND=float(np.arctanh(1-1e-6))
def transform(t,dim):
    u=np.array(t,float).copy();idx=np.r_[np.arange(dim),np.arange(2*dim,4*dim)];u[idx]=np.arctanh(np.clip(u[idx],-1+1e-6,1-1e-6));return u
def inverse(u,dim):
    t=np.array(u,float).copy();jac=np.ones_like(t);idx=np.r_[np.arange(dim),np.arange(2*dim,4*dim)];t[idx]=np.tanh(t[idx]);jac[idx]=1-t[idx]**2;return t,jac
def estimate(y,old_info):
    cc=A.Contrasts(y);scale=np.sqrt(sum(float(np.sum(d*d)) for d in cc.fields)/cc.count);cc.fields=[d/scale for d in cc.fields];dim=y.ndim
    lo=np.r_[np.full(dim,-BOUND),np.full(dim,-40.),np.full(2*dim,-BOUND)];hi=np.r_[np.full(dim,BOUND),np.full(dim,20.),np.full(2*dim,BOUND)]
    starts=[f['theta'] for f in old_info['fits'] if np.isfinite(f['value']) and f['value']<1e30]
    fits=[]
    def fun(u):
        t,jac=inverse(u,dim)
        try:
            value,g=A.profile_score(cc,t[:dim],np.exp(t[dim:2*dim]),t[2*dim:3*dim],t[3*dim:],A.covariance_band)
            return value,g*jac
        except (np.linalg.LinAlgError,FloatingPointError):return 1e30,np.zeros_like(u)
    for start in starts:
        opt=minimize(fun,transform(start,dim),jac=True,method='L-BFGS-B',bounds=list(zip(lo,hi)),options=dict(ftol=1e-15,gtol=2e-7,maxiter=700,maxls=50))
        value,g=fun(opt.x);t,jac=inverse(opt.x,dim);pg=float(np.max(np.abs(opt.x-np.clip(opt.x-g,lo,hi))))
        fits.append(dict(theta=t,transformed=opt.x,value=value,gradient=g,projected_gradient=pg,success=bool(opt.success),message=str(opt.message),iterations=int(opt.nit)))
    best=min(fits,key=lambda f:f['value']);t=np.array(best['theta']);q,ld=cc.evaluate(t[:dim],np.exp(t[dim:2*dim]),t[2*dim:3*dim],t[3*dim:]);sig=scale*np.sqrt(q/cc.count)
    status='ok' if best['projected_gradient']<=2e-5 else 'unresolved_optimization'
    if np.any(np.abs(t[:dim])>1-1e-5) or np.any(t[dim:2*dim]>19.99):status='boundary'
    return [float(sig),t[:dim].tolist(),[2]*dim],dict(status=status,objective=best['value'],old_objective=old_info['objective'],improvement=old_info['objective']-best['value'],projected_gradient=best['projected_gradient'],fits=fits)
