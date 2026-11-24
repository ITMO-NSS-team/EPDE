"""Training-only joint prediction risk using common-bank bootstrap optimism."""
import os,sys
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
sys.dont_write_bytecode=True
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'broad_moments_2026-11-16'))
import run_broad as B
L,E,Q,F,R,A,np=B.L,B.E,B.Q,B.F,B.R,B.A,B.np
import json,hashlib,time,argparse


def grams(m):
    return np.matmul(np.swapaxes(m.conj(),-2,-1),m).real


def prepare(mm,b):
    p=mm[0];n=len(mm)-1
    # Preserve the arithmetic of the existing real/complex fitting baselines.
    total=np.zeros_like(p);gram=np.zeros((p.shape[1],p.shape[1]))
    for mat in mm[1:b+1]:total+=mat;gram+=grams(mat)
    mean=total/b;z=2*p-mean
    centered_sum=gram-b*grams(mean) if np.iscomplexobj(p) else gram-b*mean.T@mean
    correction=(1+1/b)*(centered_sum/(b-1))
    errors=mm[1:]-mm[1:].mean(axis=0)
    errors*=np.sqrt((1+1/b)*n/(n-1))
    noise=grams(errors).mean(axis=0)
    observed=p[None]+errors;corrected=z[None]+errors
    return dict(p=p,z=z,correction=correction,noise=noise,
        point_ols=grams(p),point_clsm=grams(z)-correction,
        draw_ols=grams(observed),draw_clsm=grams(corrected)-correction,
        scales=np.linalg.norm(observed,axis=1),point_scale=np.linalg.norm(p,axis=0),
        reference_gram=grams(z),reference_training=grams(corrected)-noise)


def fit_scaled(gram,scale,xi,yi):
    if not xi:return np.array([]),dict(rank=0,negative=0)
    ss=scale[xi];ss=np.where(ss>0,ss,1.)
    gg=gram[np.ix_(xi,xi)]/ss[:,None]/ss[None,:];rhs=gram[xi,yi]/ss
    eig,u=np.linalg.eigh((gg+gg.T)/2);threshold=R.SV._DYAD_RCOND*np.max(np.abs(eig))
    active=np.abs(eig)>threshold;inverse=np.divide(1.,eig,out=np.zeros_like(eig),where=active)
    coef=(u@(inverse*(u.T@rhs)))/ss
    assert np.isfinite(coef).all()
    return coef,dict(rank=int(active.sum()),negative=int(np.sum(eig < -threshold)))


def assess(option,reference,c,kind):
    xi,yi=c['xi'],c['yi'];width=len(reference['point_scale']);n=len(option['scales'])
    coef,fit=R.signed_fit(option['point_'+kind],option['p'],xi,yi)
    v=np.zeros(width);v[yi]=1.;v[xi]-=coef
    vectors=np.zeros((n,width));vectors[:,yi]=1.;negative=0;deficient=0
    for j in range(n):
        cc,info=fit_scaled(option['draw_'+kind][j],option['scales'][j],xi,yi)
        vectors[j,xi]-=cc;negative+=info['negative']>0;deficient+=info['rank']<len(xi)
    denom=float(reference['point_scale'][yi]**2)
    proxy=np.einsum('bi,ij,bj->b',vectors,reference['reference_gram'],vectors,optimize=True)/denom
    training=np.einsum('bi,bij,bj->b',vectors,reference['reference_training'],vectors,optimize=True)/denom
    optimism=proxy-training
    apparent=float(v@(reference['reference_gram']-reference['noise'])@v/denom)
    diff=vectors-v
    spread=np.einsum('bi,ij,bj->b',diff,reference['reference_gram'],diff,optimize=True)/denom
    diagnostics={}
    for count in (n//2,n):
        oo=optimism[:count]
        diagnostics[count]=dict(risk=float(apparent+oo.mean()),apparent=apparent,
            optimism=float(oo.mean()),optimism_mcse=float(np.std(oo,ddof=1)/np.sqrt(count)),
            prediction_spread=float(spread[:count].mean()),
            median_proxy_loss=float(np.median(proxy[:count])))
    assert all(np.isfinite(r['risk']) and np.isfinite(r['prediction_spread']) for r in diagnostics.values())
    return dict(coefficients=coef,fit=fit,diagnostics=diagnostics,
                negative_perturbed_fits=negative,rank_deficient_perturbed_fits=deficient)


def select(assessments,count,mode):
    allowed=[(tag,x) for tag,x in assessments.items() if mode not in ('ols','clsm') or tag.endswith('_'+mode)]
    key='prediction_spread' if mode=='spread' else 'risk'
    return min(allowed,key=lambda pair:pair[1]['diagnostics'][count][key])[0]


def run_case(name,seed):
    started=time.monotonic();_,old=F.baseline(name,seed);b=old['draws']
    paths=dict(local=L.ROOT/'results'/f'{name}_seed{seed}.json',
        continuous=Q.ROOT/'unit_results'/f'{name}_seed{seed}.json',plan=F.ROOT/'fit_results'/f'{name}_seed{seed}.json')
    saved={k:json.loads(p.read_text()) for k,p in paths.items()}
    problem=R.datasets.load(name);axes=R.axes_of(problem);cs,labels=R.panel(problem);terms,opaque=R.symbols(problem,labels)
    frozen=F.evaluation_grams(problem,old,terms,opaque,labels)
    print(name,seed,'joint-risk fit; 128 conditional draws',flush=True)
    replay=dict(old);replay['draws']=128;groups=E.reconstruct_groups(problem,replay)
    folds=[];verification=0.
    for g in range(3):
        segment,ax,opq,plan=Q.mapped_plan(old,g,saved['plan']['plans'][g]['plan'],axes,opaque)
        banks={'local':Q.Bank(ax,terms,opq,plan=plan)};bank_meta={}
        pilot={v:y[segment] for v,y in groups[g][0].items()}
        kmax=min(s['loo']['max_mode'] for s in old['groups'][g]['selection'])
        for extra in (0,1):
            bank,meta=B.identify_bank(ax,terms,opq,pilot,cs,kmax,extra);bank_meta[f'broad{extra}']=meta
            if bank is not None:banks[f'broad{extra}']=bank
        prepared={};local_matrices=None
        for tag,bank in banks.items():
            matrix=np.stack([bank.matrix({v:y[segment] for v,y in values.items()}) for values in groups[g]])
            prepared[tag]=prepare(matrix,b)
            if tag=='local':local_matrices=matrix
        options=[];fits={f'{mode}_{n}':[] for n in (64,128) for mode in ('joint_all','joint_ols','joint_clsm','spread_all')}
        choices=[]
        for i,c in enumerate(cs):
            candidates={f'{tag}_{kind}':assess(p,prepared['local'],c,kind) for tag,p in prepared.items() for kind in ('ols','clsm')}
            for kind in ('ols','clsm'):
                expected=saved['continuous']['folds'][g]['arms'][f"legacy_continuous_{'ols' if kind=='ols' else 'clsm_'+str(b)}"][i]['coefficients']
                np.testing.assert_allclose(candidates['local_'+kind]['coefficients'],expected,rtol=5e-7,atol=2e-8)
            for n in (64,128):
                picks={}
                for mode in ('all','ols','clsm','spread'):
                    picked=select(candidates,n,mode);picks[mode]=picked;entry=candidates[picked]
                    tag='spread_all' if mode=='spread' else 'joint_'+mode
                    fits[f'{tag}_{n}'].append((entry['coefficients'],entry['fit']))
                choices.append(dict(candidate=c['id'],draws=n,selected=picks))
            options.append(dict(candidate=c['id'],options=candidates))
        for kind in ('ols','clsm'):
            fits['fixed_local_'+kind]=[(x['options']['local_'+kind]['coefficients'],x['options']['local_'+kind]['fit']) for x in options]
        # All fitting and risk choices are frozen before held-out and clean scoring.
        local=banks['local'];get=lambda values:local.matrix({v:y[segment] for v,y in values.items()})
        mm=[]
        for h in range(3):
            matrix=local_matrices if h==g else np.stack([get(v) for v in groups[h][:b+1]])
            mm.append(dict(pilot=matrix[0],corrected=2*matrix[0]-matrix[1:b+1].mean(axis=0)))
        clean=get(problem.data);left,right=[h for h in range(3) if h!=g]
        bundle=E.gram_bundle(mm[g],mm[left],mm[right],clean,np.ones(len(clean),bool));arms={}
        for method,fit in fits.items():
            arms['legacy_'+method]=F.scores(cs,fit,frozen,f'{g}_loo_')
            arms['local_'+method]=F.scores(cs,fit,bundle,'')
        actual=np.array([r['cross'] for r in arms['local_fixed_local_clsm']])
        expected=np.array([r['cross'] for r in saved['local']['folds'][g]['arms']['local_continuous']])
        np.testing.assert_allclose(actual,expected,rtol=5e-7,atol=2e-8)
        verification=max(verification,float(np.max(np.abs(actual-expected)/np.maximum(1,np.abs(expected)))))
        folds.append(dict(training=g,banks=bank_meta,options=options,choices=choices,arms=arms))
        print(' fold',g,'elapsed',round(time.monotonic()-started,1),'s',flush=True)
    aggregates={}
    for method in folds[0]['arms']:
        entry={}
        for metric in ('cross','ordinary','oracle','clean_energy'):
            scores=np.mean([[r[metric] for r in f['arms'][method]] for f in folds],axis=0)
            entry[metric]=dict(scores=scores,dominance=R.dominance(scores,old['instability'],cs))
        aggregates[method]=entry
    result=dict(system=name,seed=seed,draws=b,risk_draws=128,candidates=cs,instability=old['instability'],
        folds=folds,aggregates=aggregates,frozen_score_max_scaled_error=verification,elapsed_seconds=time.monotonic()-started,
        input_sha256={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in paths.items()},
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(B.__file__),Path(Q.__file__)]})
    R.dump(ROOT/'results'/f'{name}_seed{seed}.json',result)
    print('DONE',name,seed,round(result['elapsed_seconds'],1),'s',flush=True)


def checks():
    from itertools import product
    x=np.column_stack([np.ones(4),np.arange(4.)]);y=x@np.array([.2,.7])+np.array([.1,-.2,.3,-.1])
    m=np.column_stack([x,y]);sig=.13
    signs=np.array(list(product([-1.,1.],repeat=4)));n=len(signs)
    draws=np.repeat(m[None],n,axis=0);draws[:,:,-1]+=sig*signs
    pool=np.concatenate([m[None],draws]);a=prepare(pool,n);c=dict(xi=[0,1],yi=2)
    result=assess(a,a,c,'ols');den=float(a['point_scale'][2]**2)
    expected=2*(1+1/n)*n/(n-1)*sig**2*2
    actual=result['diagnostics'][n]['optimism']*den
    np.testing.assert_allclose(actual,expected,rtol=3e-13,atol=3e-14)
    v=np.array([-.3,.2,1.])
    fixed=v@a['reference_gram']@v-np.mean(np.einsum('i,bij,j->b',v,a['reference_training'],v))
    np.testing.assert_allclose(fixed,0.,atol=3e-14)
    scale=np.array([.01,100.,1.]);other=prepare(pool*scale,n)
    scaled=assess(other,other,c,'ols')
    np.testing.assert_allclose(scaled['diagnostics'][n]['risk'],result['diagnostics'][n]['risk'],rtol=3e-12,atol=3e-14)
    # Verify the fast numerical solve against the established fitting helper.
    fast,_=fit_scaled(a['point_clsm'],a['point_scale'],c['xi'],c['yi'])
    slow,_=R.signed_fit(a['point_clsm'],a['p'],c['xi'],c['yi'])
    np.testing.assert_allclose(fast,slow,atol=3e-14)
    R.dump(ROOT/'checks.json',dict(fixed_x_optimism=actual,expected_optimism=expected,
        fixed_coefficient_optimism=float(fixed),feature_rescaling_invariant=True,established_solve_matches=True))


def main():
    if os.name=='nt':R.ctypes.windll.kernel32.SetPriorityClass(R.ctypes.windll.kernel32.GetCurrentProcess(),0x4000)
    ap=argparse.ArgumentParser();ap.add_argument('--systems',nargs='+',default=['ac','wave','kdv_cossin','ks'])
    ap.add_argument('--seeds',nargs='+',type=int,default=[0,1,2]);ap.add_argument('--checks-only',action='store_true')
    args=ap.parse_args();checks()
    if args.checks_only:return
    for name in args.systems:
        for seed in args.seeds:
            if (ROOT/'results'/f'{name}_seed{seed}.json').exists():continue
            run_case(name,seed)


if __name__=='__main__':main()
