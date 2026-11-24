from pathlib import Path
import sys,json,hashlib
ROOT=Path(__file__).resolve().parent
import fit as F
np=F.np
from scipy.linalg import toeplitz,cholesky
sys.path.insert(0,str(ROOT.parent/'latent_signal_noise_2026-11-17'))
import coverage as C
S,R=C.S,C.R

def main():
    rng=np.random.default_rng(970100);rows=[];nested=[]
    for n in (11,31):
        d=np.diff(np.eye(n),n=2,axis=0);m=n-2
        for rho in (-.4,0.,.4):
            for p1 in (-.5,0.,.8,.99):
                for p2 in (-.9,0.,.5):
                    for k in (0.,1e-4,.1,10.):
                        cov=d@toeplitz(rho**np.arange(n))@d.T+k*toeplitz(F.ar2_correlation(m,p1,p2));a=(np.eye(m)-rho*np.eye(m,k=-1))@(np.eye(m)-p1*(1-p2)*np.eye(m,k=-1)-p2*np.eye(m,k=-2));ab=F.covariance_band(m,rho,k,p1,p2)
                        band=np.diag(ab[0])
                        for lag in range(1,5):band+=np.diag(ab[lag,:m-lag],lag)+np.diag(ab[lag,:m-lag],-lag)
                        np.testing.assert_allclose(band,a@cov@a.T,rtol=2e-11,atol=3e-11)
                        y=rng.normal(size=n);q,ld=F.Contrasts(y).evaluate([rho],[k],[p1],[p2]);z=d@y;truth=float(z@np.linalg.solve(cov,z));det=np.linalg.slogdet(cov)[1]
                        np.testing.assert_allclose(q,truth,rtol=5e-8,atol=5e-9);np.testing.assert_allclose(ld,det,rtol=5e-8,atol=5e-9)
                        rows.append(dict(n=n,rho=rho,p1=p1,p2=p2,kappa=k,quadratic_error=abs(q-truth),logdet_error=abs(ld-det)))
                        if p2==0:
                            oq,ol=F.Old.Contrasts(y).evaluate([rho],[k],[p1]);np.testing.assert_allclose([q,ld],[oq,ol],rtol=1e-8,atol=2e-9);nested.append(abs(q-oq))
    n=1001;p1=.99;p2=-.8;a1=p1*(1-p2);curv=np.empty(n);curv[:2]=cholesky(np.array([[1.,p1],[p1,1.]]),lower=True)@rng.normal(size=2);innov=np.sqrt((1-p1*p1)*(1-p2*p2))
    for i in range(2,n):curv[i]=a1*curv[i-1]+p2*curv[i-2]+innov*rng.normal()
    signal=np.cumsum(np.cumsum(curv*.001));y=signal+R.WF.ar_noise(rng,(n,),.2,[.4]);model,info=F.estimate(y)
    assert info['status']=='ok' and abs(model[0]/.2-1)<.2 and abs(model[1][0]-.4)<.2,info
    other,oi=F.estimate(-3*y+np.linspace(7,11,n));assert oi['status']=='ok';np.testing.assert_allclose(other[0],3*model[0],rtol=8e-4);np.testing.assert_allclose(other[1],model[1],atol=8e-4)
    sources={str(p):S.E.sha(p) for p in (Path(__file__),ROOT/'fit.py',ROOT/'protocol.txt',F.PREVIOUS/'correlated_fit.py',F.PREVIOUS/'latent_fit.py')}
    R.dump(ROOT/'checks.json',dict(status='PASS',dense_checks=len(rows),nested_checks=len(nested),max_quadratic_error=max(r['quadratic_error'] for r in rows),max_logdet_error=max(r['logdet_error'] for r in rows),max_nested_error=max(nested),synthetic=dict(model=model,info=info),trend_units_invariant=True,source_sha256=sources))
    print('PASS',len(rows),len(nested),'synthetic',model,flush=True)

if __name__=='__main__':main()
