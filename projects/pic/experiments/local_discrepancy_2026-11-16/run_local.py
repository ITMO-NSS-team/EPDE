"""Local weak cross-discrepancy and continuous quadrature on raw observations."""
import os,sys
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
sys.dont_write_bytecode=True
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'paired_uncertainty_2026-11-16'))
import run_paired as P
E,Q,F,R,A,np=P.E,P.Q,P.F,P.R,P.A,P.np
import json,hashlib,time,argparse


def raw_fits(problem,old,g,plan,terms,opaque,pilot,cs):
    axes=R.axes_of(problem);idx=np.array(old['groups'][g]['indices'])
    ax=[axes[0][idx],*axes[1:]];opq={k:v[idx] for k,v in opaque.items()}
    bank=Q.Bank(ax,terms,opq,plan=plan)
    observed={v:y[idx] for v,y in problem.noisy(5,old['seed']).items()}
    pp={v:y[idx] for v,y in pilot.items()}
    mat=bank.matrix(observed);pm=bank.matrix(pp)
    methods={'raw_ols':[R.signed_fit(mat.T@mat,mat,c['xi'],c['yi']) for c in cs]}
    total=np.zeros_like(mat);gram=np.zeros((mat.shape[1],mat.shape[1]))
    rng=np.random.default_rng(np.random.SeedSequence([1171,old['seed'],g]));model=old['groups'][g]['noise']
    for b in range(1,old['draws']+1):
        draw={v:y+R.WF.ar_noise(rng,y.shape,model[v][0],model[v][1]) for v,y in pp.items()}
        m=bank.matrix(draw);total+=m;gram+=m.T@m
        if b in (old['draws']//2,old['draws']):
            mean=total/b;shift=mat-mean+pm
            covariance=(gram-b*mean.T@mean)/(b-1)
            corrected=shift.T@shift-(1+1/b)*covariance
            methods[f'raw_clsm_{b}']=[R.signed_fit(corrected,mat,c['xi'],c['yi']) for c in cs]
    return methods,bank.metadata


def uncertainty_fold(matrices,vectors,scales,g,b,uniform,signs):
    left,right=[i for i in range(3) if i!=g];n=len(matrices[0])-1
    ml,mr=matrices[left],matrices[right]
    zl=2*ml[0]-ml[1:b+1].mean(axis=0);zr=2*mr[0]-mr[1:b+1].mean(axis=0)
    noise=np.zeros((len(uniform),vectors.shape[1]));linear=np.zeros_like(noise)
    exact_linear=np.zeros((3,n,vectors.shape[1]));indices=(uniform*n).astype(int)
    for start in range(0,vectors.shape[1],4):
        sl=slice(start,start+4);cc=vectors[:,sl];scale=scales[sl]
        ll=zl@cc;rr=zr@cc;el=ml[1:]@cc;er=mr[1:]@cc
        el-=el.mean(axis=0);er-=er.mean(axis=0)
        factor=np.sqrt((1+1/b)*n/(n-1));el*=factor;er*=factor
        _,noise[:,sl],linear[:,sl]=P.sample_scores(ll,rr,el,er,scale,indices,signs,left,right)
        exact_linear[left,:,sl]=np.sum((el.conj()*rr[None]).real,axis=1)/scale
        exact_linear[right,:,sl]=np.sum((ll[None].conj()*er).real,axis=1)/scale
    return noise,linear,exact_linear


def run_case(name,seed):
    started=time.monotonic();_,old=F.baseline(name,seed);b=old['draws']
    prior_path=Q.ROOT/'unit_results'/f'{name}_seed{seed}.json';previous=json.loads(prior_path.read_text())
    plan_path=F.ROOT/'fit_results'/f'{name}_seed{seed}.json';plans=json.loads(plan_path.read_text())
    problem=R.datasets.load(name);axes=R.axes_of(problem);cs,labels=R.panel(problem);terms,opaque=R.symbols(problem,labels)
    frozen=F.evaluation_grams(problem,old,terms,opaque,labels)
    refs=[dict(c,id=c['id']+':exact',label='exact_control',registered=False) for c in cs if c['label']=='reference']
    allcs=cs+refs;noriginal=len(cs)
    print(name,seed,'local bank; covariance B=128',flush=True)
    replay=dict(old);replay['draws']=128;fields=E.reconstruct_groups(problem,replay)
    rng=np.random.default_rng(np.random.SeedSequence([82317,seed,512]))
    uniform=rng.random((512,3));signs=rng.choice(np.array([-1.,1.]),size=(512,3))
    methods_unc=['continuous',f'raw_clsm_{b}']
    states={m:dict(noise=np.zeros((3,512,len(allcs))),linear=np.zeros((3,512,len(allcs))),
                   exact_linear=np.zeros((3,128,len(allcs))),point=np.zeros((3,len(allcs))),
                   oracle=np.zeros((3,len(allcs)))) for m in methods_unc}
    folds=[];validation=0.
    for g in range(3):
        plan=plans['plans'][g]['plan']
        fits,rawmeta=raw_fits(problem,old,g,plan,terms,opaque,fields[g][0],cs)
        for tag,rows in [('old',old['folds'][g]['arms']['loo']['counts'][str(b)]),
                         ('continuous',previous['folds'][g]['arms'][f'full_continuous_clsm_{b}'])]:
            fits[tag]=[(np.array(r['coefficients']),r['fit']) for r in rows]
        segment,ax,opq,fine=Q.mapped_plan(old,g,plan,axes,opaque);bank=Q.Bank(ax,terms,opq,plan=fine)
        get=lambda values:bank.matrix({v:y[segment] for v,y in values.items()})
        matrices=[np.stack([get(v) for v in group]) for group in fields]
        mm=[dict(pilot=m[0],corrected=2*m[0]-m[1:b+1].mean(axis=0)) for m in matrices]
        clean=get(problem.data);left,right=[i for i in range(3) if i!=g]
        bundle=E.gram_bundle(mm[g],mm[left],mm[right],clean,np.ones(len(clean),bool))
        arm={}
        for method,fit in fits.items():
            arm['legacy_'+method]=F.scores(cs,fit,frozen,f'{g}_loo_')
            fullfit=fit+[(np.array(c['exact']),{}) for c in refs]
            rows=F.scores(allcs,fullfit,bundle,'')
            arm['local_'+method]=rows[:noriginal]
            for a,c in zip(arm['local_'+method],arm['legacy_'+method]):a['frozen_clean_energy']=c['clean_energy']
            if method in methods_unc:
                vectors=np.zeros((len(terms),len(allcs)))
                for i,(c,(co,_)) in enumerate(zip(allcs,fullfit)):
                    vectors[c['yi'],i]=1.;vectors[c['xi'],i]-=co
                scales=bundle['scale'][[c['yi'] for c in allcs]]
                noise,linear,exact_linear=uncertainty_fold(matrices,vectors,scales,g,b,uniform,signs)
                state=states[method];state['noise'][g]=noise;state['linear'][g]=linear
                state['exact_linear']+=exact_linear
                state['point'][g]=[r['cross'] for r in rows];state['oracle'][g]=[r['oracle'] for r in rows]
        expected=previous['folds'][g]['arms'][f'legacy_continuous_clsm_{b}']
        actual=np.array([r['cross'] for r in arm['legacy_continuous']]);target=np.array([r['cross'] for r in expected])
        np.testing.assert_allclose(actual,target,rtol=2e-7,atol=1e-9)
        validation=max(validation,float(np.max(np.abs(actual-target)/np.maximum(1,np.abs(target)))))
        # Same exact coefficients and common bank for every fitting arm.
        exact=F.scores(refs,[(np.array(c['exact']),{}) for c in refs],bundle,'')
        folds.append(dict(training=g,raw_quadrature=rawmeta,local_quadrature=bank.metadata,arms=arm,exact_control=exact))
        print(' fold',g,'elapsed',round(time.monotonic()-started,1),'s',flush=True)
    aggregate={}
    for method in folds[0]['arms']:
        entry={}
        for metric in ('cross','ordinary','oracle','uncorrected_cross','clean_energy'):
            scores=np.mean([[r[metric] for r in f['arms'][method]] for f in folds],axis=0)
            entry[metric]=dict(scores=scores,dominance=R.dominance(scores,old['instability'],cs))
        aggregate[method]=entry
    dyad=np.r_[old['instability'],np.full(len(refs),np.nan)]
    unc={m:P.summarize_model(allcs,dyad,v['point'].mean(axis=0),v['oracle'].mean(axis=0),
                           v['noise'],v['linear'],v['exact_linear']) for m,v in states.items()}
    result=dict(system=name,seed=seed,draws=b,covariance_draws=128,mc_draws=512,candidates=cs,
        instability=old['instability'],folds=folds,aggregates=aggregate,uncertainty=unc,
        frozen_score_max_scaled_error=validation,elapsed_seconds=time.monotonic()-started,
        input_sha256=dict(continuous_result=hashlib.sha256(prior_path.read_bytes()).hexdigest(),
                          training_plan=hashlib.sha256(plan_path.read_bytes()).hexdigest()),
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(Q.__file__),Path(P.__file__)]})
    R.dump(ROOT/'results'/f'{name}_seed{seed}.json',result)
    print('DONE',name,seed,round(result['elapsed_seconds'],1),'s',flush=True)


def main():
    if os.name=='nt':R.ctypes.windll.kernel32.SetPriorityClass(R.ctypes.windll.kernel32.GetCurrentProcess(),0x4000)
    ap=argparse.ArgumentParser();ap.add_argument('--systems',nargs='+',default=R.datasets.names('core'))
    ap.add_argument('--seeds',nargs='+',type=int,default=[0,1,2]);args=ap.parse_args()
    for name in args.systems:
        for seed in args.seeds:
            if (ROOT/'results'/f'{name}_seed{seed}.json').exists():continue
            run_case(name,seed)


if __name__=='__main__':main()
