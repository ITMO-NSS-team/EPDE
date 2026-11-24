from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parent
import latent_fit as F
np=F.np
from scipy.linalg import toeplitz
sys.path.insert(0,str(ROOT.parent/'spectral_noise_2026-11-17'))
import coverage as C
S,R=C.S,C.R

def main():
    rng=np.random.default_rng(960100);dense=0;maximum=0.;recovery=[]
    for n in (9,31):
        d=np.diff(np.eye(n),n=2,axis=0)
        for rho in (-.8,-.4,0.,.4,.8):
            for kappa in (0.,1e-6,.1,10.):
                cc=d@toeplitz(rho**np.arange(n))@d.T+kappa*np.eye(n-2);aa=np.eye(n-2)-rho*np.eye(n-2,k=-1);ab=F.covariance_band(n-2,rho,kappa)
                band=np.diag(ab[0])+np.diag(ab[1,:-1],-1)+np.diag(ab[1,:-1],1)+np.diag(ab[2,:-2],-2)+np.diag(ab[2,:-2],2)
                np.testing.assert_allclose(band,aa@cc@aa.T,rtol=2e-12,atol=2e-12)
                y=rng.normal(size=n);obj=F.Contrasts(y);q,ld=obj.evaluate([rho],[kappa]);z=d@y
                truth=float(z@np.linalg.solve(cc,z));det=np.linalg.slogdet(cc)[1]
                np.testing.assert_allclose(q,truth,rtol=2e-8,atol=2e-9);np.testing.assert_allclose(ld,det,rtol=2e-8,atol=2e-9);dense+=1;maximum=max(maximum,abs(q-truth))
    for shape in ((1001,),(81,128)):
        for rho in (-.4,0.,.4):
            y=R.WF.ar_noise(rng,shape,.2,[rho]*len(shape));model,info=F.estimate(y)
            recovery.append(dict(shape=shape,rho=rho,model=model,info=info))
            assert info['status']=='ok' and abs(model[0]/.2-1)<.2 and max(abs(r-rho) for r in model[1])<.2,(shape,rho,model,info)
    n=1001;eta=rng.normal(size=n)*.001;signal=np.cumsum(np.cumsum(eta));y=signal+R.WF.ar_noise(rng,(n,),.2,[.4]);model,info=F.estimate(y)
    assert info['status']=='ok' and abs(model[0]/.2-1)<.2 and abs(model[1][0]-.4)<.2,info
    transformed,ti=F.estimate(-3*y+np.linspace(4,9,n));assert ti['status']=='ok';np.testing.assert_allclose(transformed[0],3*model[0],rtol=3e-4);np.testing.assert_allclose(transformed[1],model[1],atol=3e-4)
    R.dump(ROOT/'checks.json',dict(status='PASS',dense_checks=dense,max_quadratic_error=maximum,recovery=recovery,latent_signal=dict(model=model,info=info),
        trend_units_invariant=True,source_sha256={str(p):S.E.sha(p) for p in (Path(__file__),ROOT/'latent_fit.py',ROOT/'protocol.txt')}))
    print('PASS',dense,maximum,'latent',model)

if __name__=='__main__':main()
