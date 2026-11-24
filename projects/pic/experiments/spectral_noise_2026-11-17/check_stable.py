from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parent
import stable_fit as F
np=F.np
from scipy.fft import dct
from scipy.linalg import toeplitz
sys.path.insert(0,str(ROOT.parent/'joint_moment_regions_2026-11-17'))
import study as S

def main():
    rng=np.random.default_rng(950105);rows=[]
    for shape in ((31,),(11,13)):
        for r in (-1+1e-6,-.99,-.4,0.,.4,.99,1-1e-6):
            covariance=np.array([[1.]])
            for n in shape:
                t=dct(np.eye(n),norm='ortho',axis=0);c=t@toeplitz(r**np.arange(n))@t.T;covariance=np.kron(covariance,c)
            y=rng.normal(size=shape)
            for design in ('strips','corner'):
                cc=F.Contrast(y,design);q,ld=cc.evaluate([r]*len(shape));mask=np.zeros(shape,bool)
                if design=='strips':mask[:]=True;mask[cc.low]=False
                else:mask[cc.high]=True
                z=F.base.dctn(y,norm='ortho')[mask];a=covariance[np.ix_(mask.flatten(),mask.flatten())];truth=z@np.linalg.solve(a,z);det=np.linalg.slogdet(a)[1]
                # Tensor covariance condition can approach1e14 at this limit.
                rel=abs(q/truth-1);err=abs(ld-det);tol=2e-3 if abs(r)>.999 else 2e-8
                assert rel<tol and err<tol*len(z),(shape,r,design,rel,err)
                rows.append(dict(shape=shape,rho=r,design=design,quadratic_relative_error=rel,logdet_error=err))
    replay=[]
    for name,seed,profile in [('lv',32,'iid'),('ac',31,'positive'),('wave',30,'negative')]:
        e=S.Study(name,seed,profile,'ar');old=json.loads((ROOT/'explore'/f'{name}_{profile}_seed{seed}.json').read_text())
        for v,y in e.raw.items():
            m,info=F.estimate(y,'strips');r=next(r for r in old['rows'] if r['mode']=='raw' and r['design']=='strips' and r['variable']==v)
            np.testing.assert_allclose(m[0],r['model'][0],rtol=2e-5);np.testing.assert_allclose(m[1],r['model'][1],rtol=2e-5,atol=2e-6)
            replay.append(dict(system=name,variable=v,model=m,status=info['status']))
    out=dict(status='PASS',rows=rows,replay=replay,source_sha256={str(p):S.E.sha(p) for p in (Path(__file__),ROOT/'stable_fit.py',ROOT/'noise_fit.py')})
    S.R.dump(ROOT/'check_stable.json',out);print(json.dumps(out,indent=2))

if __name__=='__main__':main()
