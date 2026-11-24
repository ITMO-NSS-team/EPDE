"""Broad weak fitting with the first numerically identifiable moment bank."""
import os,sys
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
sys.dont_write_bytecode=True
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'local_discrepancy_2026-11-16'))
import run_local as L
E,Q,F,R,A,np=L.E,L.Q,L.F,L.R,L.A,L.np
import json,hashlib,time,argparse


def identify_bank(axes,terms,opaque,pilot,cs,kmax,extra):
    dim=len(axes);monos=[t.monomials() for t in terms]
    base=np.max([R.WF.psi_orders(m,dim) for m in monos],axis=0).astype(int)
    q=base+extra;monos=[t.monomials(psi_cap=q) for t in terms]
    cut=np.max([R.WF.leftover_orders(m,dim) for m in monos],axis=0).astype(int)*R.WF.FD_REACH
    ms=np.array([(len(a)-2*c-1)//2 for a,c in zip(axes,cut)])
    plan=dict(q=tuple(q),cut=tuple(cut),ms=tuple(ms),ss=tuple(ms),shape=(1,)*dim)
    if np.any(ms<1):return None,dict(feasible=False,reason='no spatial/time support',plan=plan)
    trial=[]
    for k in range(kmax+1):
        signed=np.r_[np.arange(k+1),-np.arange(1,k+1)]
        frequency=[signed*np.pi/(m*(a[1]-a[0])) for m,a in zip(ms,axes)]
        bank=Q.Bank(axes,terms,opaque,plan=plan,frequencies=frequency);mat=bank.matrix(pilot)
        failures=[];ratios=[]
        for c in cs:
            if not c['xi']:continue
            xx=mat[:,c['xi']];scale=np.linalg.norm(xx,axis=0)
            if np.any(scale==0):failures.append(c['id']);continue
            gram=np.real(xx.conj().T@xx)/scale[:,None]/scale[None,:]
            eig=np.linalg.eigvalsh((gram+gram.T)/2)
            ratio=float(eig[0]/eig[-1]) if eig[-1]>0 else 0.;ratios.append(ratio)
            if ratio<=R.SV._DYAD_RCOND:failures.append(c['id'])
        trial.append(dict(k=k,rows=len(mat),unidentified=failures,min_eigen_ratio=min(ratios) if ratios else None))
        if not failures:return bank,dict(feasible=True,k=k,q=q,plan=plan,trials=trial,quadrature=bank.metadata)
    return None,dict(feasible=False,reason='no bank identifies every panel feature matrix',plan=plan,trials=trial)


def train(bank,fields,segment,cs):
    get=lambda values:bank.matrix({v:y[segment] for v,y in values.items()})
    mat=get(fields[0]);gram=lambda m:np.real(m.conj().T@m)
    fits={'ols':[R.signed_fit(gram(mat),mat,c['xi'],c['yi']) for c in cs]}
    total=np.zeros_like(mat);gg=np.zeros((mat.shape[1],mat.shape[1]))
    for values in fields[1:]:
        m=get(values);total+=m;gg+=gram(m)
    b=len(fields)-1;mean=total/b;corrected=2*mat-mean
    cov=(gg-b*gram(mean))/(b-1);gc=gram(corrected)-(1+1/b)*cov
    fits['clsm']=[R.signed_fit(gc,mat,c['xi'],c['yi']) for c in cs]
    return fits


def run_case(name,seed):
    started=time.monotonic();_,old=F.baseline(name,seed);b=old['draws']
    local_path=L.ROOT/'results'/f'{name}_seed{seed}.json';previous=json.loads(local_path.read_text())
    plan_path=F.ROOT/'fit_results'/f'{name}_seed{seed}.json';plans=json.loads(plan_path.read_text())
    p=R.datasets.load(name);axes=R.axes_of(p);cs,labels=R.panel(p);terms,opaque=R.symbols(p,labels)
    frozen=F.evaluation_grams(p,old,terms,opaque,labels);groups=E.reconstruct_groups(p,old)
    folds=[];print(name,seed,'broad weak fits B',b,flush=True)
    for g in range(3):
        idx=np.array(old['groups'][g]['indices']);segment=slice(idx[0],idx[-1]+1)
        ax=[axes[0][segment],*axes[1:]];opq={k:v[segment] for k,v in opaque.items()}
        pilot={v:y[segment] for v,y in groups[g][0].items()}
        kmax=min(s['loo']['max_mode'] for s in old['groups'][g]['selection'])
        selected={extra:identify_bank(ax,terms,opq,pilot,cs,kmax,extra) for extra in (0,1)}
        fits={}
        for extra,(bank,meta) in selected.items():
            if bank is None:continue
            for label,fit in train(bank,groups[g],segment,cs).items():fits[f'broad_q{extra}_{label}']=fit
        baseline=previous['folds'][g]['arms']['local_continuous']
        fits['continuous']=[(np.array(r['coefficients']),r['fit']) for r in baseline]
        _,ax_local,opq_local,plan=Q.mapped_plan(old,g,plans['plans'][g]['plan'],axes,opaque)
        local=Q.Bank(ax_local,terms,opq_local,plan=plan)
        get=lambda values:local.matrix({v:y[segment] for v,y in values.items()})
        mm=[]
        for fields in groups:
            m=get(fields[0]);mean=sum(get(v) for v in fields[1:])/b
            mm.append(dict(pilot=m,corrected=2*m-mean))
        clean=get(p.data);left,right=[i for i in range(3) if i!=g]
        bundle=E.gram_bundle(mm[g],mm[left],mm[right],clean,np.ones(len(clean),bool))
        arms={}
        for method,fit in fits.items():
            arms['legacy_'+method]=F.scores(cs,fit,frozen,f'{g}_loo_')
            arms['local_'+method]=F.scores(cs,fit,bundle,'')
        np.testing.assert_allclose([r['cross'] for r in arms['local_continuous']],
                                   [r['cross'] for r in baseline],rtol=3e-7,atol=1e-8)
        controls={}
        for extra,(bank,meta) in selected.items():
            if bank is None:continue
            matrix=bank.matrix({v:y[segment] for v,y in p.data.items()})
            gram=np.real(matrix.conj().T@matrix)
            cf=[R.signed_fit(gram,matrix,c['xi'],c['yi']) for c in cs]
            controls[extra]=F.scores(cs,cf,frozen,f'{g}_loo_')
        folds.append(dict(training=g,selection={e:meta for e,(_,meta) in selected.items()},arms=arms,clean_fit_control=controls))
        print(' fold',g,'k',{e:m.get('k') for e,(_,m) in selected.items()},'elapsed',round(time.monotonic()-started,1),'s',flush=True)
    common=set.intersection(*(set(f['arms']) for f in folds));aggregates={}
    for method in sorted(common):
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


def main():
    if os.name=='nt':R.ctypes.windll.kernel32.SetPriorityClass(R.ctypes.windll.kernel32.GetCurrentProcess(),0x4000)
    ap=argparse.ArgumentParser();ap.add_argument('--systems',nargs='+',default=['ac','wave','kdv_cossin','ks'])
    ap.add_argument('--seeds',nargs='+',type=int,default=[0,1,2]);args=ap.parse_args()
    for name in args.systems:
        for seed in args.seeds:
            if (ROOT/'results'/f'{name}_seed{seed}.json').exists():continue
            run_case(name,seed)


if __name__=='__main__':main()
