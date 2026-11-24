import latent_fit as Base
np=Base.np
from scipy.linalg import cholesky_banded,cho_solve_banded
from scipy.optimize import minimize

def covariance_band(n,rho,kappa,phi):
    h=(1-rho)*(1+rho);hp=(1-phi)*(1+phi);b=np.array([1.,-2-phi,1+2*phi,-phi]);ab=np.zeros((4,n))
    for lag in range(4):ab[lag,:n-lag]=h*np.dot(b[:4-lag],b[lag:])
    def old(i,j):
        lag=abs(i-j)
        if lag>2:return 0.
        if i==j:return 2*(1-rho)*(3-rho) if i==0 else 6*h
        if lag==1:return -h*(4-rho) if min(i,j)==0 else -4*h
        return h
    for j in (0,1):
        for lag in range(min(4,n-j)):
            i=j+lag;ab[lag,j]=sum(fi*fj*old(i-a,j-bb) for a,fi in [(0,1.)]+([(1,-phi)] if i else []) for bb,fj in [(0,1.)]+([(1,-phi)] if j else []))
    ab[0]+=kappa*hp*(1+rho*rho);ab[0,0]+=kappa*(1-hp*(1+rho*rho))
    if n>1:ab[0,1]+=kappa*(hp+rho*rho-hp*(1+rho*rho))
    ab[1,:-1]-=kappa*rho*hp
    if n>1:ab[1,0]-=kappa*rho*(1-hp)
    return ab

class Contrasts(Base.Contrasts):
    def evaluate(self,rhos,kappas,phis):
        total=0.;logdet=0.
        for d,r,k,p in zip(self.fields,rhos,kappas,phis):
            ab=covariance_band(len(d),r,k,p);chol=cholesky_banded(ab,lower=True,check_finite=False)
            z=d.copy();z[1:]-=r*d[:-1];filtered=z.copy();filtered[1:]-=p*z[:-1]
            sol=cho_solve_banded((chol,True),filtered,check_finite=False);total+=float(np.sum(filtered*sol));logdet+=d.shape[1]*2*float(np.log(chol[0]).sum())
        if total<=0 or not np.isfinite(total):raise FloatingPointError('Invalid quadratic')
        return total,logdet

def estimate(y):
    cc=Contrasts(y);scale=np.sqrt(sum(float(np.sum(d*d)) for d in cc.fields)/cc.count)
    if not np.isfinite(scale) or scale<=np.finfo(float).eps*max(float(np.max(np.abs(y))),np.finfo(float).tiny):return None,dict(status='no_variation')
    cc.fields=[d/scale for d in cc.fields];dim=y.ndim;failed=0
    def objective(theta):
        nonlocal failed
        try:q,ld=cc.evaluate(theta[:dim],np.exp(theta[dim:2*dim]),theta[2*dim:])
        except (np.linalg.LinAlgError,FloatingPointError):failed+=1;return 1e50
        return float(np.log(q/cc.count)+ld/cc.count)
    fits=[]
    for r,lk,p in ((0.,-5.,.9),(.4,-3.,.99),(-.4,-11.,0.)):
        start=np.r_[np.full(dim,r),np.full(dim,lk),np.full(dim,p)]
        opt=minimize(objective,start,method='L-BFGS-B',bounds=[(-1+1e-6,1-1e-6)]*dim+[(-40.,20.)]*dim+[(-1+1e-6,1-1e-6)]*dim,
            options={'ftol':1e-12,'gtol':2e-7,'maxiter':300,'maxls':40})
        fits.append(dict(success=bool(opt.success),value=float(opt.fun),theta=opt.x.tolist(),iterations=int(opt.nit),message=str(opt.message)))
    good=[f for f in fits if f['success'] and np.isfinite(f['value']) and f['value']<1e40]
    if not good:return None,dict(status='optimizer_failure',fits=fits,failed_evaluations=failed)
    best=min(good,key=lambda f:f['value']);t=np.array(best['theta']);rho=t[:dim];kappa=np.exp(t[dim:2*dim]);phi=t[2*dim:];q,ld=cc.evaluate(rho,kappa,phi);sig=scale*np.sqrt(q/cc.count)
    status='boundary' if np.any(np.abs(rho)>1-1e-5) or np.any(t[dim:2*dim]>19.99) else 'ok'
    info=dict(status=status,sigma=float(sig),rhos=rho.tolist(),kappas=kappa.tolist(),phis=phi.tolist(),signal_phi_boundary=bool(np.any(np.abs(phi)>1-1e-5)),
        contrast_count=cc.count,shape=list(y.shape),objective=best['value'],fits=fits,failed_evaluations=failed,successful_objective_spread=max(f['value'] for f in good)-min(f['value'] for f in good),likelihood='exact' if dim==1 else 'axis_composite')
    return [float(sig),rho.tolist(),[2]*dim],info
