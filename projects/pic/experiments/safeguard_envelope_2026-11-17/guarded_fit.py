from study_context import *
import native_score as Native
from scipy.optimize import minimize
Native.install()

def reference(cc,u):
    t,jac=F.P.inverse(u,1);q,ld,dq,dld=Native.ORIGINAL(cc.fields[0],t[0],np.exp(t[1]),t[2],t[3]);return float(np.log(q/cc.count)+ld/cc.count),(dq/q+dld/cc.count)*jac,q

def estimate(y):
    model,info=Guard.estimate(y)
    cc=F.A.Contrasts(y);cc.fields=[cc.fields[0]];cc.count=cc.fields[0].size;scale=np.sqrt(np.mean(cc.fields[0]**2));cc.fields=[cc.fields[0]/scale]
    u=np.array(info['transformed']);before=info['objective'];v,g,q=reference(cc,u);lo=np.array([-F.P.BOUND,-40.,-F.P.BOUND,-F.P.BOUND]);hi=np.array([F.P.BOUND,20.,F.P.BOUND,F.P.BOUND]);pg=float(max(abs(u-np.clip(u-g,lo,hi))));polished=False
    if pg>2e-5:
        fit=minimize(lambda u:reference(cc,u)[:2],u,jac=True,method='L-BFGS-B',bounds=list(zip(lo,hi)),options=dict(ftol=1e-15,gtol=2e-7,maxiter=700,maxls=50));u=fit.x;v,g,q=reference(cc,u);pg=float(max(abs(u-np.clip(u-g,lo,hi))));polished=True
    t,jac=F.P.inverse(u,1);sigma=float(scale*np.sqrt(q/cc.count));rhos=[float(t[0])]+list(model[1][1:]);status='ok' if pg<=2e-5 else 'unresolved_optimization'
    if max(abs(r) for r in rhos)>1-1e-5 or t[1]>19.99:status='boundary'
    return [sigma,rhos,model[2]],dict(info,status=status,sigma=sigma,rhos=rhos,objective=v,gradient=g,projected_gradient=pg,transformed=u,theta=t,
        native_certification=dict(original_value_error=abs(v-before),original_projected_gradient=pg,original_polish=polished,native_model=model))
