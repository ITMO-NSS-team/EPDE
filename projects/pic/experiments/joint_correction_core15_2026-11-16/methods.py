"""Joint-noise normal correction and support-specific transfer validation."""
import os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
sys.dont_write_bytecode=True
sys.path.insert(0,str(ROOT.parent/'outer_selection_2026-11-16'))
import run_outer as S
N,E,M,R,O,F,np=S.N,S.E,S.M,S.R,S.O,S.F,S.np


def prepare(draws,pilot,white=None):
    delta=draws-draws.mean(axis=0)
    if white is not None:delta=np.einsum('rs,bsj->brj',white,delta,optimize=True)
    noise=np.einsum('brj,brk->jk',delta,delta,optimize=True)/(len(draws)-1)
    return dict(white=white,noise_gram=noise)


def normal(point,c,prepared,corrected=True):
    w=prepared['white'];a=point if w is None else w@point
    xi,yi=c['xi'],c['yi']
    if not xi:return np.array([]),dict(status='ok',rank=0)
    scale=np.linalg.norm(a[:,xi],axis=0)
    if np.any(scale==0):return None,dict(status='rank_deficient_design')
    gram=a.T@a-(prepared['noise_gram'] if corrected else 0)
    xx=gram[np.ix_(xi,xi)]/scale[:,None]/scale[None,:];xy=gram[xi,yi]/scale
    eig,u=np.linalg.eigh((xx+xx.T)/2);threshold=E.RCOND*max(float(np.max(np.abs(eig))),0.)
    info=dict(status='ok',eigenvalues=eig,rank=int(np.sum(eig>threshold)),negative=int(np.sum(eig < -threshold)))
    if np.any(eig<=threshold):
        info['status']='indefinite_corrected_design' if np.any(eig < -threshold) else 'rank_deficient_design'
        return None,info
    coef=(u@((u.T@xy)/eig))/scale
    assert np.isfinite(coef).all()
    return coef,info


def transfer(pilot,white,rich,c):
    if white is None:return None,dict(status='infeasible_covariance')
    x=white@pilot[:,c['xi']];scale=np.linalg.norm(x,axis=0)
    if not len(scale) or np.any(scale==0):return None,dict(status='rank_deficient_validation_design')
    u,s,vh=np.linalg.svd(x/scale,full_matrices=False)
    if len(s)<len(scale) or s[-1]**2<=E.RCOND*s[0]**2:return None,dict(status='rank_deficient_validation_design')
    left=((vh.T/s)@u.T@white)/scale[:,None]
    h=rich[:,c['xi']];gram=(h.conj().T@h).real;hs=np.sqrt(np.diag(gram))
    if np.any(hs==0):return None,dict(status='rank_deficient_rich_design')
    ge,gu=np.linalg.eigh((gram/hs[:,None]/hs[None,:]))
    if ge[0]<=E.RCOND*ge[-1]:return None,dict(status='rank_deficient_rich_design')
    factor=(np.sqrt(ge)[:,None]*gu.T)*hs[None,:]
    transform=factor@left;den=float(np.sum(np.abs(rich[:,c['yi']])**2))
    if not den>0:return None,dict(status='zero_target_scale')
    np.testing.assert_allclose(left@pilot[:,c['xi']],np.eye(len(scale)),rtol=2e-7,atol=2e-7)
    return transform,dict(status='ok',scale=den,validation_eigen_ratio=float((s[-1]/s[0])**2),rich_eigen_ratio=float(ge[0]/ge[-1]))


def score(points,transform,den,c,coef):
    v=S.vector(c,coef,points[0].shape[1]);a=transform@points[0]@v;b=transform@points[1]@v
    return float(a@b/den)


def choose(options):
    feasible=[r for r in options if r['status']=='ok' and r.get('inner') is not None and np.isfinite(r['inner'])]
    if not feasible:return dict(status='no_feasible_choice',option=None,forced=False)
    best=min(feasible,key=lambda r:r['inner'])
    return dict(status='ok',option=best['option'],forced=len(feasible)==1,inner=best['inner'],feasible_options=len(feasible))


def checks():
    rng=np.random.default_rng(917);x=np.column_stack([np.ones(11),np.linspace(-1,1,11)])
    pilot=np.column_stack([x,1.3*x[:,0]-.6*x[:,1]]);c=dict(xi=[0,1],yi=2)
    draws=np.broadcast_to(pilot,(512,*pilot.shape)).copy();draws[:,:,1:]+=rng.normal(size=(512,11,2))*.04
    w=np.diag(np.linspace(1,2,11));prepared=prepare(draws,pilot,w)
    centered=draws-draws.mean(axis=0)
    direct=sum((w@a).T@(w@a) for a in centered)/511
    np.testing.assert_allclose(prepared['noise_gram'],direct,rtol=2e-13,atol=2e-13)
    assert np.max(np.abs(prepared['noise_gram'][0]))==0
    actual,info=normal(pilot,c,prepared);assert info['status']=='ok'
    units=np.array([.1,30.,2.]);other,meta=normal(pilot*units,c,prepare(draws*units,pilot*units,w))
    assert meta['status']=='ok';np.testing.assert_allclose(other*units[:2]/units[2],actual,rtol=2e-12)
    z=np.zeros_like(draws);kept,km=normal(pilot,c,prepare(z,pilot,w));np.testing.assert_allclose(kept,[1.3,-.6],atol=2e-14)
    bad,bm=normal(pilot,c,dict(white=None,noise_gram=1000*np.eye(3)));assert bad is None and bm['status']=='indefinite_corrected_design'
    t,tm=transfer(pilot,w,pilot,c);delta=np.array([.2,-.1]);r=pilot[:,:2]@delta
    np.testing.assert_allclose(np.sum((t@r)**2),np.sum(r**2),rtol=2e-12)
    tu,tum=transfer(pilot*units,w,pilot*units,c)
    np.testing.assert_allclose(np.sum((tu@(r*units[2]))**2)/tum['scale'],np.sum((t@r)**2)/tm['scale'],rtol=2e-12)
    lost,lm=transfer(np.column_stack([x[:,0],x[:,0],pilot[:,2]]),w,pilot,c);assert lost is None
    a=dict(option='first',status='ok',inner=1.);b=dict(option='second',status='ok',inner=2.)
    assert choose([dict(a,outer=10),dict(b,outer=-100)])==choose([a,b])
    assert choose([a,dict(b,inner=1.)])['option']=='first'
    assert choose([dict(a,status='failed'),b])['forced']
    R.dump(ROOT/'checks.json',dict(joint_normal_correction_algebra=True,constant_noise_zero=True,
        feature_units_invariant=True,zero_noise_exact_fit=True,indefinite_design_rejected=True,
        transfer_recovers_coefficient_effect=True,transfer_units_invariant=True,rank_failure_explicit=True,
        selection_isolated=True,source_sha256=S.sha(__file__)))
    print('Correction, transfer, constants, units, infeasibility and selection checks: PASS')


if __name__=='__main__':checks()
