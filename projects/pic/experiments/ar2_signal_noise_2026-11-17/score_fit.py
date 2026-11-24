from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
PREVIOUS=ROOT.parent/'latent_signal_noise_2026-11-17'
sys.path.insert(0,str(PREVIOUS))
import correlated_fit as Old
np=Old.np
from scipy.linalg import cholesky_banded,cho_solve_banded
from scipy.optimize import minimize
from score_core import profile_score

def ar2_correlation(n,p1,p2):
    a1=p1*(1-p2);out=np.empty(n);out[0]=1.
    if n>1:out[1]=p1
    for i in range(2,n):out[i]=a1*out[i-1]+p2*out[i-2]
    return out

def covariance_band(n,rho,kappa,p1,p2):
    a1=p1*(1-p2);a2=p2;h=(1-rho)*(1+rho)
    b=np.array([1.,-2-a1,1+2*a1-a2,-a1+2*a2,-a2]);ab=np.zeros((5,n),dtype=np.result_type(rho,kappa,p1,p2,float))
    for lag in range(5):ab[lag,:n-lag]=h*np.dot(b[:5-lag],b[lag:])
    def old(i,j):
        lag=abs(i-j)
        if lag>2:return 0.
        if i==j:return 2*(1-rho)*(3-rho) if i==0 else 6*h
        if lag==1:return -h*(4-rho) if min(i,j)==0 else -4*h
        return h
    filt=[1.,-a1,-a2]
    for j in range(min(3,n)):
        for lag in range(min(5,n-j)):
            i=j+lag;ab[lag,j]=sum(filt[u]*filt[v]*old(i-u,j-v) for u in range(min(2,i)+1) for v in range(min(2,j)+1))
    # Initial AR2 innovations have a correlated 2x2 block; later ones are iid.
    c=p1*p2;t1=1-p1*p1*(1-p2*p2);v=(1-p1)*(1+p1)*(1-p2)*(1+p2)
    ab[0]+=kappa*v*(1+rho*rho);ab[0,0]+=kappa*(1-v*(1+rho*rho))
    if n>1:ab[0,1]+=kappa*(t1-2*rho*c+rho*rho-v*(1+rho*rho))
    if n>2:ab[0,2]+=kappa*rho*rho*(t1-v)
    ab[1,:-1]-=kappa*rho*v
    if n>1:ab[1,0]+=kappa*(c-rho+rho*v)
    if n>2:ab[1,1]+=kappa*(-rho*t1+rho*rho*c+rho*v);ab[2,0]-=kappa*rho*c
    return ab

class Contrasts(Old.Contrasts):
    def evaluate(self,rhos,kappas,p1s,p2s):
        total=0.;logdet=0.
        for d,r,k,p1,p2 in zip(self.fields,rhos,kappas,p1s,p2s):
            band=covariance_band(len(d),r,k,p1,p2);chol=cholesky_banded(band,lower=True,check_finite=False)
            z=d.copy();z[1:]-=r*d[:-1];filtered=z.copy();filtered[1:]-=p1*(1-p2)*z[:-1];filtered[2:]-=p2*z[:-2]
            sol=cho_solve_banded((chol,True),filtered,check_finite=False);total+=float(np.sum(filtered*sol));logdet+=d.shape[1]*2*float(np.log(chol[0]).sum())
        if total<=0 or not np.isfinite(total):raise FloatingPointError('Invalid contrast quadratic')
        return total,logdet

def estimate(y,warm=None):
    cc=Contrasts(y);scale=np.sqrt(sum(float(np.sum(d*d)) for d in cc.fields)/cc.count)
    if not np.isfinite(scale) or scale<=np.finfo(float).eps*max(float(np.max(np.abs(y))),np.finfo(float).tiny):return None,dict(status='no_variation')
    cc.fields=[d/scale for d in cc.fields];dim=y.ndim;failed=0
    if warm is None:_,warm=Old.estimate(y)
    def objective(theta):
        nonlocal failed
        try:return profile_score(cc,theta[:dim],np.exp(theta[dim:2*dim]),theta[2*dim:3*dim],theta[3*dim:],covariance_band)
        except (np.linalg.LinAlgError,FloatingPointError):failed+=1;return 1e50,np.zeros(4*dim)
    starts=[];warmgood=[f for f in warm.get('fits',[]) if f['success'] and f['value']<1e40]
    if warmgood:
        best=min(warmgood,key=lambda f:f['value']);starts.append(('nested_ar1',np.r_[best['theta'],np.zeros(dim)]))
    for i,(r,lk,p1,p2) in enumerate(((0.,-5.,.99,-.8),(.4,-3.,.95,-.5),(-.4,-11.,.9,0.))):starts.append((f'fixed_{i}',np.r_[np.full(dim,r),np.full(dim,lk),np.full(dim,p1),np.full(dim,p2)]))
    fits=[]
    for name,start in starts:
        opt=minimize(objective,start,method='L-BFGS-B',jac=True,bounds=[(-1+1e-6,1-1e-6)]*dim+[(-40.,20.)]*dim+[(-1+1e-6,1-1e-6)]*(2*dim),
            options={'ftol':1e-12,'gtol':2e-6,'maxiter':500,'maxls':40})
        fits.append(dict(start=name,success=bool(opt.success),value=float(opt.fun),theta=opt.x.tolist(),iterations=int(opt.nit),gradient=opt.jac.tolist(),message=str(opt.message)))
    good=[f for f in fits if f['success'] and np.isfinite(f['value']) and f['value']<1e40]
    if not good:return None,dict(status='optimizer_failure',fits=fits,failed_evaluations=failed,warm=warm)
    best=min(good,key=lambda f:f['value']);t=np.array(best['theta']);rho=t[:dim];k=np.exp(t[dim:2*dim]);p1=t[2*dim:3*dim];p2=t[3*dim:];q,ld=cc.evaluate(rho,k,p1,p2);sig=scale*np.sqrt(q/cc.count)
    status='boundary' if np.any(np.abs(rho)>1-1e-5) or np.any(t[dim:2*dim]>19.99) else 'ok'
    info=dict(status=status,sigma=float(sig),rhos=rho.tolist(),kappas=k.tolist(),p1s=p1.tolist(),p2s=p2.tolist(),ar1=(p1*(1-p2)).tolist(),ar2=p2.tolist(),
        signal_boundary=bool(np.any(np.abs(t[2*dim:])>1-1e-5)),contrast_count=cc.count,shape=list(y.shape),objective=best['value'],fits=fits,warm=warm,
        failed_evaluations=failed,successful_objective_spread=max(f['value'] for f in good)-min(f['value'] for f in good),likelihood='exact' if dim==1 else 'axis_composite')
    return [float(sig),rho.tolist(),[2]*dim],info
