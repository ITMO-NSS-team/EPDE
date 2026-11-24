from common import *
import robust_fit as F
import repair_fit as P
from scipy.optimize._numdiff import approx_derivative
def main():
    rng=np.random.default_rng(980100);errors=[]
    for shape in ((31,),(100,),(47,20),(20,17,9)):
        y=rng.normal(size=shape);cc=F.Contrasts(y)
        for rho in (-.7,0.,.6):
            t=np.r_[.1,np.full(len(shape),np.arctanh(rho))];value,g=cc.evaluate(t)
            numerical=approx_derivative(lambda t:cc.evaluate(t)[0],t,method='3-point').ravel()
            np.testing.assert_allclose(g,numerical,rtol=3e-5,atol=2e-7);errors.append(float(np.max(np.abs(g-numerical))))
        for _,d,q,lag,alpha in cc.fields:
            np.testing.assert_allclose(q.T@q,np.eye(q.shape[1]),atol=2e-15)
            x=np.linspace(-1,1,len(q));np.testing.assert_allclose(q.T@np.vander(x,4,increasing=True),0,atol=3e-15)
    y=R.WF.ar_noise(rng,(4001,),.2,[.4])+np.linspace(-1,1,4001)**3
    m,i=F.estimate(y);assert i['status']=='ok' and abs(m[0]/.2-1)<.1 and abs(m[1][0]-.4)<.1,(m,i)
    x=np.linspace(-1,1,len(y));mm,ii=F.estimate(-3*y+2+x+3*x*x-2*x**3)
    assert ii['status']=='ok';np.testing.assert_allclose(mm[0],3*m[0],rtol=1e-6);np.testing.assert_allclose(mm[1],m[1],atol=1e-6)
    cc=A.Contrasts(rng.normal(size=(31,20)));dim=2;t=np.r_[[.2,-.3],[-4.,-.5],[.8,.7],[-.5,.2]];u=P.transform(t,dim);tt,jac=P.inverse(u,dim);np.testing.assert_allclose(t,tt,atol=1e-15)
    def fun(u):
        t,j=P.inverse(u,dim);val,g=A.profile_score(cc,t[:dim],np.exp(t[dim:2*dim]),t[2*dim:3*dim],t[3*dim:],A.covariance_band);return val,g*j
    np.testing.assert_allclose(fun(u)[1],approx_derivative(lambda u:fun(u)[0],u).ravel(),atol=2e-6,rtol=2e-5)
    dump(ROOT/'checks.json',dict(status='PASS',gradient_checks=len(errors)+1,max_robust_gradient_error=max(errors),synthetic_model=m,unit_polynomial_invariance=True,source_sha256=hashes('checks.py','robust_fit.py','repair_fit.py','common.py','protocol.txt')))
    print('PASS',len(errors)+1,m,flush=True)
if __name__=='__main__':main()
