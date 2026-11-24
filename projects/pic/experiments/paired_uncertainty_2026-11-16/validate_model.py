"""Exhaustive complex-vector check of paired and rotation covariance."""
import run_paired as P
np=P.np
from itertools import product


def main():
    rng=np.random.default_rng(621)
    # Three correlated complex raw-group error pools; each maps into different
    # fold-specific operators. Last two columns are identical candidates.
    base=rng.normal(size=(3,3,4));base-=base.mean(axis=1,keepdims=True)
    combos=np.array(list(product(range(3),repeat=3)))
    signs=np.array(list(product([-1.,1.],repeat=3)))
    indices=np.repeat(combos,len(signs),axis=0);signs=np.tile(signs,(len(combos),1))
    shifts=[];linear_by_group=np.zeros((3,3,3));quadratic_cov=np.zeros((3,3))
    for g in range(3):
        left,right=[i for i in range(3) if i!=g]
        z=[rng.normal(size=(2,3))+1j*rng.normal(size=(2,3)) for _ in range(2)]
        errors=[]
        for h in [left,right]:
            mapping=rng.normal(size=(4,2,3))+1j*rng.normal(size=(4,2,3))
            e=np.einsum('bd,drc->brc',base[h],mapping);e[:,:,2]=e[:,:,1];errors.append(e)
        for v in z:v[:,2]=v[:,1]
        zl,zr=z;el,er=errors
        _,out,_=P.sample_scores(zl,zr,el,er,np.ones(3),indices,signs,left,right)
        shifts.append(out)
        linear_by_group[left]+=np.sum((el.conj()*zr[None]).real,axis=1)
        linear_by_group[right]+=np.sum((zl[None].conj()*er).real,axis=1)
        qq=np.einsum('arc,brc->abc',el.conj(),er).real.reshape(-1,3)
        quadratic_cov+=qq.T@qq/len(qq)
    samples=np.mean(shifts,axis=0)
    analytic=(sum(v.T@v/len(v) for v in linear_by_group)+quadratic_cov)/9
    actual=np.cov(samples,rowvar=False,ddof=0)
    np.testing.assert_allclose(actual,analytic,rtol=3e-14,atol=3e-14)
    np.testing.assert_allclose(samples[:,2]-samples[:,1],0.,atol=1e-14)
    # Direct and covariance-derived pair variance must agree as well.
    pair_var=actual[0,0]+actual[1,1]-2*actual[0,1]
    np.testing.assert_allclose(pair_var,np.var(samples[:,0]-samples[:,1]),rtol=3e-14)
    P.R.dump(P.ROOT/'complex_covariance_checks.json',dict(exhaustive_joint_draws=len(samples),
        max_covariance_error=float(np.max(np.abs(actual-analytic))),
        identical_candidate_paired_error=0.,paired_variance_verified=True))
    print('Complex-vector covariance checks passed:',len(samples),'joint draws')


if __name__=='__main__':main()
