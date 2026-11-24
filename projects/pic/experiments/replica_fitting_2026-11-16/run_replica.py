"""Weak coefficient fitting with independently reconstructed training replicas."""
import os,sys
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
sys.dont_write_bytecode=True
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'local_discrepancy_2026-11-16'))
import run_local as L
E,Q,F,R,A,np=L.E,L.Q,L.F,L.R,L.A,L.np
import json,hashlib,time,argparse


def make_fits(cs,pa,pb,za,zb):
    obs=(pa+pb)/2;pooled=np.concatenate([pa,pb])
    grams=dict(replica_ols=(pooled.T@pooled,pooled),
        replica_iv=((pa.T@pb+pb.T@pa)/2,obs),
        replica_iv_mean=((za.T@zb+zb.T@za)/2,obs))
    return {name:[R.signed_fit(gram,mat,c['xi'],c['yi']) for c in cs] for name,(gram,mat) in grams.items()}


def training_replicas(problem,old,g,bank,segment,cs):
    axes=R.axes_of(problem);idx=np.array(old['groups'][g]['indices']);raw=problem.noisy(5,old['seed'])
    get=lambda values:bank.matrix({v:y[segment] for v,y in values.items()})
    point=[];corrected=[];meta=[]
    for h in range(2):
        ii=idx[h::2];observed={v:y[ii] for v,y in raw.items()}
        pilot,_=R.reconstruct(observed,axes,ii);pm=get(pilot)
        model={v:R.WF.noise_model(y) for v,y in observed.items()}
        rng=np.random.default_rng(np.random.SeedSequence([99127,old['seed'],g,h]));total=np.zeros_like(pm)
        for b in range(old['draws']):
            draw={v:pilot[v][ii]+R.WF.ar_noise(rng,y.shape,model[v][0],model[v][1]) for v,y in observed.items()}
            total+=get(R.reconstruct(draw,axes,ii)[0])
        point.append(pm);corrected.append(2*pm-total/old['draws'])
        meta.append(dict(indices=ii,noise=model))
    fits=make_fits(cs,*point,*corrected)
    # All noisy fits are frozen before these interpolation controls.
    clean=[]
    for h in range(2):
        ii=idx[h::2]
        clean.append(get(R.reconstruct({v:y[ii] for v,y in problem.data.items()},axes,ii)[0]))
    control=make_fits(cs,*clean,*clean)['replica_iv']
    return fits,meta,control


def run_case(name,seed):
    started=time.monotonic();_,old=F.baseline(name,seed);b=old['draws']
    local_path=L.ROOT/'results'/f'{name}_seed{seed}.json';previous=json.loads(local_path.read_text())
    plan_path=F.ROOT/'fit_results'/f'{name}_seed{seed}.json';plans=json.loads(plan_path.read_text())
    problem=R.datasets.load(name);axes=R.axes_of(problem);cs,labels=R.panel(problem);terms,opaque=R.symbols(problem,labels)
    frozen=F.evaluation_grams(problem,old,terms,opaque,labels)
    print(name,seed,'training replica fit B',b,flush=True)
    groups=E.reconstruct_groups(problem,old);folds=[]
    for g in range(3):
        segment,ax,opq,plan=Q.mapped_plan(old,g,plans['plans'][g]['plan'],axes,opaque)
        bank=Q.Bank(ax,terms,opq,plan=plan)
        fits,meta,control=training_replicas(problem,old,g,bank,segment,cs)
        rows=previous['folds'][g]['arms']['local_continuous']
        fits['continuous']=[(np.array(r['coefficients']),r['fit']) for r in rows]
        get=lambda values:bank.matrix({v:y[segment] for v,y in values.items()})
        mm=[]
        for group in groups:
            pilot=get(group[0]);mean=sum(get(v) for v in group[1:])/b
            mm.append(dict(pilot=pilot,corrected=2*pilot-mean))
        clean=get(problem.data);left,right=[i for i in range(3) if i!=g]
        bundle=E.gram_bundle(mm[g],mm[left],mm[right],clean,np.ones(len(clean),bool));arms={}
        for method,fit in fits.items():
            arms['legacy_'+method]=F.scores(cs,fit,frozen,f'{g}_loo_')
            arms['local_'+method]=F.scores(cs,fit,bundle,'')
        for metric in ['cross','oracle','ordinary']:
            np.testing.assert_allclose([r[metric] for r in arms['local_continuous']],
                [r[metric] for r in rows],rtol=3e-7,atol=1e-8)
        clean_control=F.scores(cs,control,frozen,f'{g}_loo_')
        folds.append(dict(training=g,replicas=meta,arms=arms,clean_replica_control=clean_control))
        print(' fold',g,'elapsed',round(time.monotonic()-started,1),'s',flush=True)
    aggregates={}
    for method in folds[0]['arms']:
        entry={}
        for metric in ['cross','ordinary','oracle','clean_energy']:
            scores=np.mean([[r[metric] for r in f['arms'][method]] for f in folds],axis=0)
            entry[metric]=dict(scores=scores,dominance=R.dominance(scores,old['instability'],cs))
        aggregates[method]=entry
    result=dict(system=name,seed=seed,draws=b,candidates=cs,instability=old['instability'],folds=folds,
        aggregates=aggregates,elapsed_seconds=time.monotonic()-started,
        input_sha256=dict(local_result=hashlib.sha256(local_path.read_bytes()).hexdigest(),
                          training_plan=hashlib.sha256(plan_path.read_bytes()).hexdigest()),
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(Q.__file__)]})
    R.dump(ROOT/'results'/f'{name}_seed{seed}.json',result)
    print('DONE',name,seed,round(result['elapsed_seconds'],1),'s',flush=True)


def checks():
    from itertools import product
    rng=np.random.default_rng(3381);truth=rng.normal(size=(7,3));a=rng.normal(size=truth.shape);b=rng.normal(size=truth.shape)
    cross=[];ordinary=[]
    for sa,sb in product([-1,1],repeat=2):
        x,y=truth+sa*a,truth+sb*b
        cross.append((x.T@y+y.T@x)/2);ordinary.append((x.T@x+y.T@y)/2)
    np.testing.assert_allclose(np.mean(cross,axis=0),truth.T@truth,atol=1e-14)
    assert np.trace(np.mean(ordinary,axis=0)-truth.T@truth)>0
    R.dump(ROOT/'checks.json',dict(independent_linear_operator_cross_gram_unbiased=True,
        within_replica_gram_noise_bias_verified=True))


def main():
    if os.name=='nt':R.ctypes.windll.kernel32.SetPriorityClass(R.ctypes.windll.kernel32.GetCurrentProcess(),0x4000)
    ap=argparse.ArgumentParser();ap.add_argument('--systems',nargs='+',default=['ac','wave','kdv_cossin','ks'])
    ap.add_argument('--seeds',nargs='+',type=int,default=[0,1,2]);args=ap.parse_args();checks()
    for name in args.systems:
        for seed in args.seeds:
            if (ROOT/'results'/f'{name}_seed{seed}.json').exists():continue
            run_case(name,seed)


if __name__=='__main__':main()
