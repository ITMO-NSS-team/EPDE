from common import A,np
import repair_fit as P
import state_score as State
from scipy.optimize import minimize
from scipy.linalg import qr
from copy import copy
def estimate(y):
    cc=A.Contrasts(y);cc.fields=[cc.fields[0]];cc.count=cc.fields[0].size;scale=np.sqrt(np.mean(cc.fields[0]**2));cc.fields=[cc.fields[0]/scale]
    original_cc=cc;compressed=False
    data=cc.fields[0]
    if data.shape[1]>2*data.shape[0]:
        _,rmat=qr(data.T,mode='economic',check_finite=False);cc=copy(original_cc);cc.fields=[rmat.T*np.sqrt(data.shape[0]/data.shape[1])];cc.count=data.shape[0]**2;compressed=True
    lo=np.array([-P.BOUND,-40.,-P.BOUND,-P.BOUND]);hi=np.array([P.BOUND,20.,P.BOUND,P.BOUND]);bounds=list(zip(lo,hi))
    def objective(u,state=False):
        t,jac=P.inverse(u,1)
        try:
            if state:value,g,q=State.profile(cc,t[:1],np.exp(t[1:2]),t[2:3],t[3:])
            else:value,g=A.profile_score(cc,t[:1],np.exp(t[1:2]),t[2:3],t[3:],A.covariance_band)
            return value,g*jac
        except (np.linalg.LinAlgError,FloatingPointError):return 1e30,np.zeros_like(u)
    fits=[]
    for start in ((0.,-5.,.99,-.8),(.4,-3.,.95,-.5),(-.4,-11.,.9,0.),(0.,-15.,.9,0.)):
        opt=minimize(objective,P.transform(start,1),jac=True,method='L-BFGS-B',bounds=bounds,options=dict(ftol=1e-14,gtol=2e-7,maxiter=500,maxls=40))
        fits.append(dict(transformed=opt.x,value=float(opt.fun),success=bool(opt.success),iterations=int(opt.nit),message=str(opt.message)))
    best=min(fits,key=lambda f:f['value']);cc=original_cc;u=np.array(best['transformed']);value,g=objective(u,True);pg=float(np.max(np.abs(u-np.clip(u-g,lo,hi))));polished=False
    if pg>2e-5:
        opt=minimize(lambda x:objective(x,True),u,jac=True,method='L-BFGS-B',bounds=bounds,options=dict(ftol=1e-15,gtol=2e-7,maxiter=500,maxls=40));u=opt.x;value,g=objective(u,True);pg=float(np.max(np.abs(u-np.clip(u-g,lo,hi))));polished=True
    t,jac=P.inverse(u,1);v,gg,q=State.profile(cc,t[:1],np.exp(t[1:2]),t[2:3],t[3:]);sigma=scale*np.sqrt(q/cc.count);rhos=[float(t[0])]
    signal=None
    if y.ndim>1:
        residual=np.diff(y,n=4,axis=0);residual=residual-np.median(residual);sign=np.sign(residual);signal=[]
        for axis in range(1,y.ndim):
            left=[slice(None)]*y.ndim;right=list(left);left[axis]=slice(None,-1);right[axis]=slice(1,None);tau=float(np.mean(sign[tuple(left)]*sign[tuple(right)]));rhos.append(float(np.sin(np.pi*tau/2)));signal.append(tau)
    status='ok' if pg<=2e-5 else 'unresolved_optimization'
    if max(abs(r) for r in rhos)>1-1e-5 or t[1]>19.99:status='boundary'
    return [float(sigma),rhos,[2]+[4]*(y.ndim-1)],dict(compressed=compressed,status=status,sigma=float(sigma),rhos=rhos,objective=value,projected_gradient=pg,gradient=g,transformed=u,theta=t,polished=polished,fits=fits,spatial_sign_moments=signal,contrast_count=cc.count,shape=y.shape)
