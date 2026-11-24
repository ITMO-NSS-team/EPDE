from pathlib import Path
ROOT=Path(__file__).resolve().parent
import score_fit as F
from score_core import selected_inverse,profile_score
from scipy.linalg import cholesky_banded,toeplitz
np=F.np
import sys,importlib.util
spec=importlib.util.spec_from_file_location('score_check_coverage',F.PREVIOUS/'coverage.py');C=importlib.util.module_from_spec(spec);spec.loader.exec_module(C)
S,R=C.S,C.R

def main():
    rng=np.random.default_rng(970106);max_inverse=0.;max_gradient=0.;count=0
    for n in (11,31):
        for rho,p1,p2,k in ((.4,.8,-.5,.1),(-.4,.99,-.9,.001),(0.,.9,0.,.01),(.2,-.5,.5,3.)):
            band=F.covariance_band(n,rho,k,p1,p2);chol=cholesky_banded(band,lower=True);inv=selected_inverse(chol);dense=np.diag(band[0])
            for lag in range(1,5):dense+=np.diag(band[lag,:n-lag],lag)+np.diag(band[lag,:n-lag],-lag)
            inverse=np.linalg.inv(dense)
            for lag in range(5):
                truth=np.diag(inverse,-lag);np.testing.assert_allclose(inv[lag,:n-lag],truth,rtol=2e-8,atol=3e-9);max_inverse=max(max_inverse,float(np.max(np.abs(inv[lag,:n-lag]-truth))))
            y=rng.normal(size=n+2);cc=F.Contrasts(y);theta=np.array([rho,np.log(k),p1,p2]);obj,g=profile_score(cc,[rho],[k],[p1],[p2],F.covariance_band)
            def direct(t):
                d=np.diff(np.eye(n+2),n=2,axis=0);cov=d@toeplitz(t[0]**np.arange(n+2))@d.T+np.exp(t[1])*toeplitz(F.ar2_correlation(n,t[2],t[3]));z=d@y;q=z@np.linalg.solve(cov,z);return np.log(q/n)+np.linalg.slogdet(cov)[1]/n
            np.testing.assert_allclose(obj,direct(theta),rtol=2e-8,atol=3e-9)
            for j in range(4):
                h=1e-5;delta=np.eye(4)[j]*h;fd=(direct(theta+delta)-direct(theta-delta))/(2*h);np.testing.assert_allclose(g[j],fd,rtol=2e-4,atol=8e-6);max_gradient=max(max_gradient,abs(g[j]-fd));count+=1
    n=1001;p1=.99;p2=-.8;curv=np.empty(n);curv[:2]=np.linalg.cholesky(np.array([[1.,p1],[p1,1.]]))@rng.normal(size=2);sd=np.sqrt((1-p1*p1)*(1-p2*p2))
    for i in range(2,n):curv[i]=p1*(1-p2)*curv[i-1]+p2*curv[i-2]+sd*rng.normal()
    y=np.cumsum(np.cumsum(.001*curv))+R.WF.ar_noise(rng,(n,),.2,[.4]);m,info=F.estimate(y);assert info['status']=='ok' and abs(m[0]/.2-1)<.2 and abs(m[1][0]-.4)<.2
    mm,ii=F.estimate(-3*y+np.linspace(7,11,n));assert ii['status']=='ok';np.testing.assert_allclose(mm[0],3*m[0],rtol=8e-4);np.testing.assert_allclose(mm[1],m[1],atol=8e-4)
    R.dump(ROOT/'checks_score.json',dict(status='PASS',gradient_checks=count,max_inverse_error=max_inverse,max_gradient_error=max_gradient,synthetic=dict(model=m,info=info),trend_units_invariant=True,
        source_sha256={str(p):S.E.sha(p) for p in (Path(__file__),ROOT/'score_fit.py',ROOT/'score_core.py',ROOT/'fit.py',ROOT/'score_protocol.txt',F.PREVIOUS/'correlated_fit.py',F.PREVIOUS/'latent_fit.py')}))
    print('PASS',count,max_inverse,max_gradient,m)

if __name__=='__main__':main()
