"""Conditional paired uncertainty with shared operator perturbations."""
import os,sys
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
sys.dont_write_bytecode=True
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'operator_weak_2026-11-16'))
import run_operator as E
Q,F,R,A,np=E.Q,E.F,E.R,E.A,E.np
import json,hashlib,time,argparse

FAILED={('ac',0),('ac',2),('kdv_cossin',0),('kdv_cossin',2),('ks',0),('pde_compound',1),('pde_compound',2)}


def sample_scores(zl,zr,el,er,scale,indices,signs,left,right):
    """Error in exact bilinear score; common indices retain joint dependence."""
    count=len(indices);out=np.empty((count,zl.shape[1]));linear=np.empty_like(out)
    point=np.sum((zl.conj()*zr).real,axis=0)/scale
    for start in range(0,count,8):
        sl=slice(start,start+8)
        ll=el[indices[sl,left]]*signs[sl,left,None,None]
        rr=er[indices[sl,right]]*signs[sl,right,None,None]
        lin=np.sum((ll.conj()*zr[None]+zl[None].conj()*rr).real,axis=1)/scale
        linear[sl]=lin
        out[sl]=lin+np.sum((ll.conj()*rr).real,axis=1)/scale
    return point,out,linear


def interval(gap,errors,level):
    a=(1-level)/2;q=np.quantile(errors,[a,1-a])
    return [float(gap-q[1]),float(gap-q[0])]


def summarize_model(cs,dyad,point,oracle,fold_noise,linear_samples,exact_linear):
    noise=fold_noise.mean(axis=0);linear=linear_samples.mean(axis=0)
    covariance=np.atleast_2d(np.cov(noise,rowvar=False,ddof=1))
    # Each empirical group error has zero mean by sign symmetrization.
    exact_linear_cov=sum(v.T@v/len(v) for v in exact_linear)/9
    pairs=[]
    refs=[i for i,c in enumerate(cs) if c.get('registered',False) and c['label']!='exact_control']
    for j in refs:
        for i,c in enumerate(cs):
            if i==j or c['slot']!=cs[j]['slot'] or c['label']=='exact_control':continue
            err=noise[:,i]-noise[:,j];gap=float(point[i]-point[j]);clean=float(oracle[i]-oracle[j])
            sd=float(np.std(err,ddof=1));lc=exact_linear_cov
            eligible=bool(dyad[i]<=dyad[j])
            mc_linear=linear[:,i]-linear[:,j]
            pair=dict(reference=cs[j]['id'],candidate=c['id'],slot=c['slot'],
                dyad_eligible=eligible,point_dominator=bool(eligible and gap<0),gap=gap,oracle_gap=clean,
                paired_sd=sd,standardized_margin=gap/sd if sd>0 else None,
                independent_candidate_sd=float(np.sqrt(max(covariance[i,i]+covariance[j,j],0))),
                independent_rotation_sd=float(np.sqrt(sum(np.var(f[:,i]-f[:,j],ddof=1) for f in fold_noise)/9)),
                first_order_sd=float(np.sqrt(max(lc[i,i]+lc[j,j]-2*lc[i,j],0))),
                quadratic_only_sd=float(np.std(err-mc_linear,ddof=1)),
                model_lower_score_fraction=float(np.mean(gap+err<0)),
                error_mean=float(np.mean(err)),error_mean_mcse=sd/np.sqrt(len(err)),
                interval90=interval(gap,err,.9),interval95=interval(gap,err,.95))
            pair['oracle_in_95']=bool(pair['interval95'][0]<=clean<=pair['interval95'][1])
            pairs.append(pair)
    exact=[]
    for i,c in enumerate(cs):
        if c['label']!='exact_control':continue
        band=interval(float(point[i]),noise[:,i],.95)
        exact.append(dict(slot=c['slot'],score=float(point[i]),oracle=float(oracle[i]),
            sd=float(np.std(noise[:,i],ddof=1)),interval95=band,
            oracle_in_95=bool(band[0]<=oracle[i]<=band[1])))
    return dict(pairs=pairs,exact=exact,score_covariance=covariance,
                exact_linear_covariance=exact_linear_cov,
                max_score_mcse=float(np.sqrt(np.diag(covariance).max()/len(noise))))


def run_case(name,seed,cov_draws=128,mc_draws=512):
    started=time.monotonic();path=Q.ROOT/'unit_results'/f'{name}_seed{seed}.json'
    previous=json.loads(path.read_text());_,old=F.baseline(name,seed)
    b=old['draws'];assert cov_draws>=b
    problem=R.datasets.load(name);axes=R.axes_of(problem)
    cs,labels=R.panel(problem);terms,opaque=R.symbols(problem,labels);noriginal=len(cs)
    cs=[dict(c) for c in cs]
    for c in list(cs):
        if c['label']=='reference':
            cc=dict(c);cc.update(id=c['id']+':exact',label='exact_control',registered=False);cs.append(cc)
    print(name,seed,'reconstruct',cov_draws,'covariance draws',flush=True)
    replay=dict(old);replay['draws']=cov_draws
    fields=E.reconstruct_groups(problem,replay)
    counts=[cov_draws]
    if (name,seed) in FAILED:counts.insert(0,cov_draws//2)
    rng=np.random.default_rng(np.random.SeedSequence([82317,seed,mc_draws]))
    uniform=rng.random((mc_draws,3));signs=rng.choice(np.array([-1.,1.]),size=(mc_draws,3))
    state={n:dict(noise=np.zeros((3,mc_draws,len(cs))),linear=np.zeros((3,mc_draws,len(cs))),
                  exact_linear=np.zeros((3,n,len(cs)))) for n in counts}
    point=np.zeros((3,len(cs)));oracle=np.zeros_like(point);reproduction=0.
    for g in range(3):
        frequency=[]
        for selection in old['groups'][g]['selection']:
            k=selection['loo']['max_mode'];frequency.append(np.r_[np.arange(k+1),-np.arange(1,k+1)]*selection['spacing'])
        bank=Q.Bank(axes,terms,opaque,frequencies=frequency)
        left,right=[i for i in range(3) if i!=g]
        train=bank.matrix(fields[g][0]);scales=np.sum(np.abs(train)**2,axis=0)
        # Process one bank at a time. Keep operator matrices, not a dense covariance.
        ml=np.stack([bank.matrix(v) for v in fields[left]])
        mr=np.stack([bank.matrix(v) for v in fields[right]])
        zl=2*ml[0]-ml[1:b+1].mean(axis=0);zr=2*mr[0]-mr[1:b+1].mean(axis=0)
        coeff=np.zeros((len(terms),len(cs)))
        rows=previous['folds'][g]['arms'][f'full_continuous_clsm_{b}']
        for i,c in enumerate(cs):
            coef=rows[i]['coefficients'] if i<noriginal else c['exact']
            coeff[c['yi'],i]=1.;coeff[c['xi'],i]-=coef
        # Clean controls are read only after all coefficients/banks are frozen.
        clean=bank.matrix(problem.data)
        for start in range(0,len(cs),4):
            sl=slice(start,start+4);cc=coeff[:,sl];target=np.array([c['yi'] for c in cs[sl]])
            scale=scales[target];ll=zl@cc;rr=zr@cc
            point[g,sl]=np.sum((ll.conj()*rr).real,axis=0)/scale
            truth=clean@cc;oracle[g,sl]=np.sum(np.abs(truth)**2,axis=0)/scale
            for n in counts:
                el=ml[1:n+1]@cc;er=mr[1:n+1]@cc
                el-=el.mean(axis=0);er-=er.mean(axis=0)
                factor=np.sqrt((1+1/b)*n/(n-1));el*=factor;er*=factor
                indices=(uniform*n).astype(int)
                _,perturb,linear=sample_scores(ll,rr,el,er,scale,indices,signs,left,right)
                state[n]['noise'][g,:,sl]=perturb;state[n]['linear'][g,:,sl]=linear
                state[n]['exact_linear'][left,:,sl]+=np.sum((el.conj()*rr[None]).real,axis=1)/scale
                state[n]['exact_linear'][right,:,sl]+=np.sum((ll[None].conj()*er).real,axis=1)/scale
                del el,er
        target=np.array([r['cross'] for r in rows])
        np.testing.assert_allclose(point[g,:noriginal],target,rtol=3e-7,atol=1e-8)
        reproduction=max(reproduction,float(np.max(np.abs(point[g,:noriginal]-target)/np.maximum(1,np.abs(target)))))
        np.testing.assert_allclose(oracle[g,:noriginal],[r['oracle'] for r in rows],rtol=3e-7,atol=1e-8)
        del ml,mr
        print(' fold',g,'elapsed',round(time.monotonic()-started,1),'s',flush=True)
    pscore=point.mean(axis=0);oscore=oracle.mean(axis=0)
    dyad=np.r_[previous['instability'],np.full(len(cs)-noriginal,np.nan)]
    results={n:summarize_model(cs,dyad,pscore,oscore,v['noise'],v['linear'],v['exact_linear']) for n,v in state.items()}
    out=dict(system=name,seed=seed,original_draws=b,covariance_draws=cov_draws,mc_draws=mc_draws,
        candidates=cs,point_scores=pscore,oracle_scores=oscore,instability=previous['instability'],models=results,
        frozen_score_max_scaled_error=reproduction,elapsed_seconds=time.monotonic()-started,
        baseline_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(Q.__file__),Path(E.__file__)]})
    R.dump(ROOT/'results'/f'{name}_seed{seed}.json',out)
    np.savez_compressed(ROOT/'results'/f'{name}_seed{seed}_samples.npz',
        **{f'{n}_{kind}':v[kind] for n,v in state.items() for kind in ('noise','linear')})
    print('DONE',name,seed,round(out['elapsed_seconds'],1),'s',flush=True)


def checks():
    from itertools import product
    # Exhaust all independent signs for scalar group perturbations. Rotations
    # share a group; duplicated candidate columns must have zero paired error.
    signs=np.array(list(product([-1.,1.],repeat=3)));idx=np.zeros_like(signs,dtype=int)
    mu=np.array([1.,2.,3.]);sig=np.array([.2,.3,.4]);folds=[];linear=[]
    for g in range(3):
        l,r=[i for i in range(3) if i!=g]
        _,noise,lin=sample_scores(np.full((1,2),mu[l]),np.full((1,2),mu[r]),
            np.full((1,1,2),sig[l]),np.full((1,1,2),sig[r]),np.ones(2),idx,signs,l,r)
        np.testing.assert_array_equal(noise[:,0],noise[:,1]);folds.append(noise[:,0]);linear.append(lin[:,0])
    avg=np.mean(folds,axis=0)
    analytic=(sum(sig[i]**2*(mu.sum()-mu[i])**2 for i in range(3))+
              sum(sig[i]**2*sig[j]**2 for i in range(3) for j in range(i+1,3)))/9
    np.testing.assert_allclose(np.var(avg),analytic,rtol=1e-14)
    assert np.var(avg)>sum(np.var(f) for f in folds)/9
    assert np.var(avg)>np.var(np.mean(linear,axis=0))
    R.dump(ROOT/'checks.json',dict(exact_bilinear_variance=float(analytic),
        duplicate_pair_error=0.,shared_rotation_covariance_verified=True,
        quadratic_variance_verified=True))


def main():
    if os.name=='nt':R.ctypes.windll.kernel32.SetPriorityClass(R.ctypes.windll.kernel32.GetCurrentProcess(),0x4000)
    ap=argparse.ArgumentParser();ap.add_argument('--systems',nargs='+',default=R.datasets.names('core'))
    ap.add_argument('--seeds',nargs='+',type=int,default=[0,1,2]);ap.add_argument('--checks-only',action='store_true')
    args=ap.parse_args();checks()
    if args.checks_only:return
    for name in args.systems:
        for seed in args.seeds:
            if (ROOT/'results'/f'{name}_seed{seed}.json').exists():continue
            run_case(name,seed)


if __name__=='__main__':main()
