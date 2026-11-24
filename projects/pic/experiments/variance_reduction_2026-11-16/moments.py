"""Real moment basis and a common training-only operator-noise metric."""
import os,sys
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
sys.dont_write_bytecode=True
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'raw_validation_constants_2026-11-16'))
import raw_operator as O
import run_validation as V
J,R,Q,F,A,np=O.J,O.R,O.Q,O.F,O.A,O.np
B=J.B
from itertools import product


class RealBasis:
    def __init__(self,k,dim):
        signed=list(range(k+1))+list(range(-1,-k-1,-1))
        modes=list(product(signed,repeat=dim));index={key:i for i,key in enumerate(modes)}
        self.zero=index[(0,)*dim];self.pairs=[];self.size=len(modes)
        for key,i in index.items():
            first=next((x for x in key if x),0)
            if first>0:self.pairs.append((i,index[tuple(-x for x in key)]))

    def convert(self,m):
        assert m.shape[-2]==self.size
        rows=[m[...,self.zero,:].real]
        for i,j in self.pairs:
            np.testing.assert_allclose(m[...,j,:],m[...,i,:].conj(),rtol=3e-10,atol=3e-10)
            rows.extend([np.sqrt(2)*m[...,i,:].real,np.sqrt(2)*m[...,i,:].imag])
        out=np.stack(rows,axis=-2)
        np.testing.assert_allclose(np.sum(out**2,axis=-2),np.sum(np.abs(m)**2,axis=-2),rtol=3e-10,atol=3e-10)
        return out


def whitening(draws,pilot):
    n=len(draws);scale=np.linalg.norm(pilot,axis=0);active=scale>0
    e=(draws-draws.mean(axis=0))[:,:,active]/scale[active]
    c=np.einsum('brj,bsj->rs',e,e,optimize=True)/((n-1)*active.sum())
    eig,u=np.linalg.eigh((c+c.T)/2);threshold=R.SV._DYAD_RCOND*max(float(eig[-1]),0.)
    retained=eig>threshold;meta=dict(rank=int(retained.sum()),rows=len(eig),eigenvalues=eig,
        active_columns=int(active.sum()),draws=n,feasible=bool(np.all(retained)))
    if not meta['feasible']:
        meta['reason']='pooled covariance not positive full rank at numerical tolerance';return None,meta
    w=(u/np.sqrt(eig)).T
    identity=w@c@w.T;error=float(np.max(np.abs(identity-np.eye(len(eig)))))
    meta['whitened_covariance_max_error']=error
    assert error<2e-5
    return w,meta


def transform(m,w):
    return m if w is None else np.einsum('ij,...jk->...ik',w,m,optimize=True)


def checks():
    rng=np.random.default_rng(1331);k=2;dim=2;basis=RealBasis(k,dim)
    real=rng.normal(size=(25,4));complex_m=np.zeros((25,4),complex);complex_m[basis.zero]=real[0]
    for n,(i,j) in enumerate(basis.pairs):
        complex_m[i]=(real[1+2*n]+1j*real[2+2*n])/np.sqrt(2);complex_m[j]=complex_m[i].conj()
    recovered=basis.convert(complex_m);np.testing.assert_allclose(recovered,real,atol=1e-15)
    np.testing.assert_allclose(recovered.T@recovered,(complex_m.conj().T@complex_m).real,atol=2e-14)
    draws=rng.normal(size=(128,25,4));pilot=rng.normal(size=(25,4));w,meta=whitening(draws,pilot)
    units=np.array([.01,100.,.2,3.]);other,_=whitening(draws*units,pilot*units)
    np.testing.assert_allclose(w.T@w,other.T@other,rtol=2e-12,atol=2e-12)
    zero,zm=whitening(np.zeros_like(draws),pilot);assert zero is None and not zm['feasible']
    # A constant's actual modulated column is preserved by both transforms.
    t=np.linspace(0,1,41);term=R.WF.WeakTerm([],(0,));plan=dict(q=(3,),cut=(0,),ms=(20,),ss=(20,),shape=(1,))
    bank=Q.Bank([t],[term],{},plan=plan,frequencies=[np.array([0,1,2,-1,-2])*np.pi/.5])
    cm=bank.matrix({'u':np.ones(41)});rb=RealBasis(2,1).convert(cm)
    np.testing.assert_allclose(cm[0,0],1.,atol=5e-14)
    assert not np.allclose(rb[:,0],1.)
    R.dump(ROOT/'checks.json',dict(real_basis_isometry=True,feature_units_invariant=True,
        whitening_identity_max_error=meta['whitened_covariance_max_error'],zero_covariance_explicitly_infeasible=True,
        constant_dc=float(cm[0,0].real),constant_real_moments=rb[:,0]))


if __name__=='__main__':checks()
