from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parent
import noise_fit as F
np=F.np
from scipy.fft import dct
from scipy.linalg import toeplitz
sys.path.insert(0,str(ROOT.parent/'joint_moment_regions_2026-11-17'))
import study as S

def main():
    rng=np.random.default_rng(950100);dense=0;maxerror=0
    for shape in ((17,),(8,9),(6,7,8)):
        for rho in ([0.]*len(shape),[.4]*len(shape),[-.4]*len(shape),[(-1)**j*.6 for j in range(len(shape))]):
            covariance=np.array([[1.]])
            for n,r in zip(shape,rho):
                t=dct(np.eye(n),type=2,norm='ortho',axis=0);c=t@toeplitz(r**np.arange(n))@t.T;covariance=np.kron(covariance,c)
            for design in ('strips','corner'):
                y=rng.normal(size=shape);contrast=F.Contrast(y,design);q,ld=contrast.evaluate(rho)
                mask=np.zeros(shape,bool)
                if design=='strips':mask[:]=True;mask[contrast.low]=False
                else:mask[contrast.high]=True
                selected=covariance[np.ix_(mask.flatten(),mask.flatten())];z=F.dctn(y,type=2,norm='ortho')[mask]
                truth=float(z@np.linalg.solve(selected,z));logdet=np.linalg.slogdet(selected)[1]
                np.testing.assert_allclose(q,truth,rtol=2e-12,atol=1e-12);np.testing.assert_allclose(ld,logdet,rtol=2e-12,atol=1e-12)
                maxerror=max(maxerror,abs(q-truth));dense+=1
    recovery=[]
    for shape in ((4096,),(128,128),(20,24,28)):
        for rho in (-.4,0.,.4):
            y=S.R.WF.ar_noise(rng,shape,.2,[rho]*len(shape))
            for design in ('strips','corner'):
                model,info=F.estimate(y,design);assert info['status']=='ok',info
                assert abs(model[0]/.2-1)<.25 and max(abs(r-rho) for r in model[1])<.3,(shape,rho,model)
                other,ii=F.estimate(7-3*y,design);assert ii['status']=='ok'
                np.testing.assert_allclose(other[0],3*model[0],rtol=2e-5);np.testing.assert_allclose(other[1],model[1],rtol=1e-4,atol=3e-5)
                recovery.append(dict(shape=shape,true_rho=rho,design=design,model=model))
    out=dict(status='PASS',dense_covariance_checks=dense,max_quadratic_error=maxerror,recovery=recovery,
        source_sha256={str(p):S.E.sha(p) for p in (ROOT/'noise_fit.py',Path(__file__),ROOT/'protocol.txt')})
    S.R.dump(ROOT/'checks.json',out);print(json.dumps(out,indent=2))

if __name__=='__main__':main()
