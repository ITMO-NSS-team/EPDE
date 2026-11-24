from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent
import correlated_fit as F
np=F.np
from scipy.linalg import toeplitz
sys.path.insert(0,str(ROOT.parent/'spectral_noise_2026-11-17'))
import coverage as C
S,R=C.S,C.R

def main():
    rng=np.random.default_rng(960106);rows=[]
    for n in (11,31):
        d=np.diff(np.eye(n),n=2,axis=0);m=n-2
        for rho in (-.4,0.,.4):
            for phi in (-.5,0.,.8,.99):
                for k in (0.,1e-4,.1,10.):
                    cov=d@toeplitz(rho**np.arange(n))@d.T+k*toeplitz(phi**np.arange(m));a=(np.eye(m)-rho*np.eye(m,k=-1))@(np.eye(m)-phi*np.eye(m,k=-1));ab=F.covariance_band(m,rho,k,phi)
                    band=np.diag(ab[0])
                    for lag in range(1,4):band+=np.diag(ab[lag,:m-lag],lag)+np.diag(ab[lag,:m-lag],-lag)
                    np.testing.assert_allclose(band,a@cov@a.T,rtol=1e-11,atol=2e-11)
                    y=rng.normal(size=n);q,ld=F.Contrasts(y).evaluate([rho],[k],[phi]);z=d@y;truth=float(z@np.linalg.solve(cov,z));det=np.linalg.slogdet(cov)[1]
                    np.testing.assert_allclose(q,truth,rtol=3e-8,atol=3e-9);np.testing.assert_allclose(ld,det,rtol=3e-8,atol=3e-9);rows.append(dict(n=n,rho=rho,phi=phi,kappa=k,error=abs(q-truth)))
    n=1001;curv=R.WF.ar_noise(rng,(n,),.001,[.9]);signal=np.cumsum(np.cumsum(curv));y=signal+R.WF.ar_noise(rng,(n,),.2,[.4]);model,info=F.estimate(y)
    assert info['status']=='ok' and abs(model[0]/.2-1)<.2 and abs(model[1][0]-.4)<.2,info
    other,oi=F.estimate(3*y+np.linspace(7,11,n));assert oi['status']=='ok';np.testing.assert_allclose(other[0],3*model[0],rtol=8e-4);np.testing.assert_allclose(other[1],model[1],atol=8e-4)
    R.dump(ROOT/'checks_correlated.json',dict(status='PASS',dense_checks=len(rows),max_error=max(r['error'] for r in rows),synthetic_signal=dict(model=model,info=info),trend_unit_invariant=True,
        source_sha256={str(p):S.E.sha(p) for p in (Path(__file__),ROOT/'correlated_fit.py',ROOT/'latent_fit.py',ROOT/'correlated_curvature_protocol.txt')}))
    print('PASS',len(rows),'synthetic',model)

if __name__=='__main__':main()
