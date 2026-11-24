from common import *
import state_score as F
from scipy.linalg import toeplitz
from scipy.optimize._numdiff import approx_derivative
def main():
    rng=np.random.default_rng(980112);err=[]
    for n in (11,31):
        d=np.diff(np.eye(n),2,axis=0);data=rng.normal(size=(n-2,3))
        for r,k,p1,p2 in [(.4,.01,.9,-.8),(-.4,2.,-.5,.4),(0.,1e-5,.9999,-.99),(.7,1e-10,.999999,-.9)]:
            cov=d@toeplitz(r**np.arange(n))@d.T+k*toeplitz(A.ar2_correlation(n-2,p1,p2));q=float(np.sum(data*np.linalg.solve(cov,data)));ld=3*np.linalg.slogdet(cov)[1]
            qq,ll,dq,dl=F.axis_score(data,r,k,p1,p2);np.testing.assert_allclose([qq,ll],[q,ld],rtol=2e-9,atol=2e-8)
            pars=np.array([r,np.log(k),p1,p2]);fun=lambda t:np.array(F.axis_score(data,t[0],np.exp(t[1]),t[2],t[3])[:2]);deriv=approx_derivative(fun,pars,method='3-point',abs_step=1e-6)
            analytic=np.stack([dq,dl]);np.testing.assert_allclose(analytic,deriv,rtol=3e-4,atol=2e-4);err.append(float(np.max(np.abs(analytic-deriv))))
    dump(ROOT/'state_checks.json',dict(status='PASS',dense_checks=len(err),max_absolute_gradient_error=max(err),source_sha256=hashes('state_checks.py','state_score.py')));print('PASS',len(err),max(err),flush=True)
if __name__=='__main__':main()
