"""Training-only choice of weak fitting bank by its worst feature-noise direction."""
import os,sys
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
sys.dont_write_bytecode=True
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'broad_moments_2026-11-16'))
import run_broad as B
L,E,Q,F,R,A,np=B.L,B.E,B.Q,B.F,B.R,B.A,B.np
import json,hashlib,time,argparse


def worst_noise(gram,noise,xi):
    if not xi:return dict(score=0.,rank=0)
    gg=gram[np.ix_(xi,xi)];nn=noise[np.ix_(xi,xi)]
    ss=np.sqrt(np.maximum(np.diag(gg),0.))
    if np.any(ss==0):return dict(score=None,rank=0)
    gg=gg/ss[:,None]/ss[None,:];nn=nn/ss[:,None]/ss[None,:]
    eig,u=np.linalg.eigh((gg+gg.T)/2);keep=eig>R.SV._DYAD_RCOND*eig.max()
    rank=int(keep.sum())
    if rank<len(xi):return dict(score=None,rank=rank)
    white=u/np.sqrt(eig);relative=white.T@nn@white
    value=float(np.linalg.eigvalsh((relative+relative.T)/2)[-1])
    return dict(score=max(value,0.),rank=rank)


def bank_moments(bank,fields,segment):
    get=lambda v:bank.matrix({k:y[segment] for k,y in v.items()})
    mm=np.stack([get(v) for v in fields]);b=len(fields)-1
    mean=mm[1:].mean(axis=0);shift=2*mm[0]-mean
    gram=np.real(shift.conj().T@shift);noise={}
    for n in (b//2,b):
        ee=mm[1:n+1]-mm[1:n+1].mean(axis=0)
        noise[n]=(1+1/b)*np.einsum('brj,brk->jk',ee.conj(),ee).real/(n-1)
    return gram,noise,mm


def run_case(name,seed):
    started=time.monotonic();_,old=F.baseline(name,seed);b=old['draws']
    paths=dict(local=L.ROOT/'results'/f'{name}_seed{seed}.json',broad=B.ROOT/'results'/f'{name}_seed{seed}.json',
        continuous=Q.ROOT/'unit_results'/f'{name}_seed{seed}.json',plan=F.ROOT/'fit_results'/f'{name}_seed{seed}.json')
    saved={k:json.loads(p.read_text()) for k,p in paths.items()}
    p=R.datasets.load(name);axes=R.axes_of(p);cs,labels=R.panel(p);terms,opaque=R.symbols(p,labels)
    frozen=F.evaluation_grams(p,old,terms,opaque,labels);groups=E.reconstruct_groups(p,old)
    folds=[];print(name,seed,'feature-information bank selection',flush=True)
    for g in range(3):
        segment,ax,opq,plan=Q.mapped_plan(old,g,saved['plan']['plans'][g]['plan'],axes,opaque)
        banks={'local':Q.Bank(ax,terms,opq,plan=plan)}
        for extra in (0,1):
            selection=saved['broad']['folds'][g]['selection'][str(extra)]
            if not selection['feasible']:continue
            pp=selection['plan'];k=selection['k'];signed=np.r_[np.arange(k+1),-np.arange(1,k+1)]
            freq=[signed*np.pi/(m*(a[1]-a[0])) for m,a in zip(pp['ms'],ax)]
            banks[f'broad{extra}']=Q.Bank(ax,terms,opq,plan=pp,frequencies=freq)
        moments={tag:bank_moments(bank,groups[g],segment) for tag,bank in banks.items()}
        options={}
        for tag in banks:
            if tag=='local':
                options[tag]={kind:saved['continuous']['folds'][g]['arms'][f"legacy_continuous_{'ols' if kind=='ols' else 'clsm_'+str(b)}"] for kind in ('ols','clsm')}
            else:
                extra=tag[-1];options[tag]={kind:saved['broad']['folds'][g]['arms'][f'legacy_broad_q{extra}_{kind}'] for kind in ('ols','clsm')}
        fits={};choices=[];selection_failures=[]
        for n in (b//2,b):
            picked=[]
            for i,c in enumerate(cs):
                scores={tag:worst_noise(gg,nn[n],c['xi']) for tag,(gg,nn,_) in moments.items()}
                feasible=[tag for tag,s in scores.items() if s['score'] is not None]
                choice=min(feasible,key=lambda tag:scores[tag]['score']) if feasible else None
                picked.append(choice);choices.append(dict(draws=n,candidate=c['id'],choice=choice,scores=scores))
            if None in picked:
                selection_failures.append(dict(draws=n,candidates=[c['id'] for c,tag in zip(cs,picked) if tag is None]));continue
            for kind in ('ols','clsm'):
                fits[f'selected_{kind}_{n}']=[(np.array(options[tag][kind][i]['coefficients']),options[tag][kind][i]['fit']) for i,tag in enumerate(picked)]
        fits['continuous']=[(np.array(r['coefficients']),r['fit']) for r in options['local']['clsm']]
        # All choices are now fixed. The held-out bank remains shared.
        local=banks['local'];get=lambda v:local.matrix({k:y[segment] for k,y in v.items()})
        mm=[]
        for h in range(3):
            mat=moments['local'][2] if h==g else np.stack([get(v) for v in groups[h]])
            mm.append(dict(pilot=mat[0],corrected=2*mat[0]-mat[1:].mean(axis=0)))
        clean=get(p.data);left,right=[i for i in range(3) if i!=g]
        bundle=E.gram_bundle(mm[g],mm[left],mm[right],clean,np.ones(len(clean),bool));arms={}
        for method,fit in fits.items():
            arms['legacy_'+method]=F.scores(cs,fit,frozen,f'{g}_loo_')
            arms['local_'+method]=F.scores(cs,fit,bundle,'')
        np.testing.assert_allclose([r['cross'] for r in arms['local_continuous']],
            [r['cross'] for r in saved['local']['folds'][g]['arms']['local_continuous']],rtol=3e-7,atol=1e-8)
        folds.append(dict(training=g,choices=choices,selection_failures=selection_failures,arms=arms))
        print(' fold',g,'elapsed',round(time.monotonic()-started,1),'s',flush=True)
    aggregates={}
    for method in sorted(set.intersection(*(set(f['arms']) for f in folds))):
        entry={}
        for metric in ['cross','ordinary','oracle','clean_energy']:
            scores=np.mean([[r[metric] for r in f['arms'][method]] for f in folds],axis=0)
            entry[metric]=dict(scores=scores,dominance=R.dominance(scores,old['instability'],cs))
        aggregates[method]=entry
    result=dict(system=name,seed=seed,draws=b,candidates=cs,instability=old['instability'],folds=folds,
        aggregates=aggregates,elapsed_seconds=time.monotonic()-started,
        input_sha256={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in paths.items()},
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(Q.__file__)]})
    R.dump(ROOT/'results'/f'{name}_seed{seed}.json',result)
    print('DONE',name,seed,round(result['elapsed_seconds'],1),'s',flush=True)


def checks():
    rng=np.random.default_rng(723);x=rng.normal(size=(20,4));e=rng.normal(size=(20,4))
    gg=x.T@x;nn=e.T@e;scale=np.array([.01,.1,10.,100.]);s1=worst_noise(gg,nn,list(range(4)))['score']
    s2=worst_noise(gg*scale[:,None]*scale[None,:],nn*scale[:,None]*scale[None,:],list(range(4)))['score']
    np.testing.assert_allclose(s1,s2,rtol=2e-13)
    assert worst_noise(np.ones((2,2)),np.eye(2),[0,1])['score'] is None
    R.dump(ROOT/'checks.json',dict(feature_rescaling_invariant=True,numerical_rank_failure_explicit=True,max_scaling_error=abs(s1-s2)))


def main():
    if os.name=='nt':R.ctypes.windll.kernel32.SetPriorityClass(R.ctypes.windll.kernel32.GetCurrentProcess(),0x4000)
    ap=argparse.ArgumentParser();ap.add_argument('--systems',nargs='+',default=['ac','wave','kdv_cossin','ks'])
    ap.add_argument('--seeds',nargs='+',type=int,default=[0,1,2]);args=ap.parse_args();checks()
    for name in args.systems:
        for seed in args.seeds:
            if (ROOT/'results'/f'{name}_seed{seed}.json').exists():continue
            run_case(name,seed)


if __name__=='__main__':main()
