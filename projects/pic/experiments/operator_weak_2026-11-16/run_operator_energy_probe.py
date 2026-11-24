"""Quadrature-controlled fitting and training-only operator-risk test banks."""
import quadrature as Q
F,R,A,np=Q.F,Q.R,Q.A,Q.np
from pathlib import Path
import hashlib
import json
import argparse
import time


def reconstruct_groups(problem,old):
    axes=R.axes_of(problem);raw=problem.noisy(5,old['seed']);groups=[]
    for g,meta in enumerate(old['groups']):
        idx=np.array(meta['indices']);observed={v:y[idx] for v,y in raw.items()}
        assert all(R.digest(y)==meta['observed_sha256'][v] for v,y in observed.items())
        pilot,_=R.reconstruct(observed,axes,idx);rng=np.random.default_rng(np.random.SeedSequence([1171,old['seed'],g]))
        model=meta['noise'];fields=[pilot]
        for b in range(old['draws']):
            draw={v:pilot[v][idx]+R.WF.ar_noise(rng,y.shape,model[v][0],model[v][1]) for v,y in observed.items()}
            fields.append(R.reconstruct(draw,axes,idx)[0])
        groups.append(fields)
    return groups


def local_fit(bank,fields,segment,candidates):
    get=lambda values:bank.matrix({v:y[segment] for v,y in values.items()})
    pilot=get(fields[0]);total=np.zeros_like(pilot);gram=np.zeros((pilot.shape[1],pilot.shape[1]))
    methods={'continuous_ols':[R.signed_fit(pilot.T@pilot,pilot,c['xi'],c['yi']) for c in candidates]}
    for b,values in enumerate(fields[1:],1):
        mat=get(values);total+=mat;gram+=mat.T@mat
        if b in ((len(fields)-1)//2,len(fields)-1):
            mean=total/b;shifted=2*pilot-mean
            covariance=(gram-b*mean.T@mean)/(b-1)
            corrected=shifted.T@shifted-(1+1/b)*covariance
            methods[f'continuous_clsm_{b}']=[R.signed_fit(corrected,pilot,c['xi'],c['yi']) for c in candidates]
    return methods


def moments(bank,fields):
    pilot=bank.matrix(fields[0]);total=np.zeros_like(pilot);squares=np.zeros(pilot.shape)
    for values in fields[1:]:
        mat=bank.matrix(values);total+=mat;squares+=np.abs(mat)**2
    b=len(fields)-1;mean=total/b
    variance=np.maximum((squares-b*np.abs(mean)**2)/(b-1),0.)
    return dict(pilot=pilot,corrected=2*pilot-mean,variance=(1+1/b)*variance)


def risk_prefix(training,shape):
    """Approximate unbiased projection risk; same bank for every candidate.

Retain-row minus drop-row risk is 2*variance-|bias-corrected value|^2.
Normalize operator columns by their TRAINING energy; minimize the sum over
the nested rectangular frequency prefixes. Covariance between rows does
not enter this trace/MSE risk. Bootstrap bias accuracy remains an assumption.
"""
    z=training['corrected'];v=training['variance']
    scale=np.mean(np.abs(training['pilot'])**2,axis=0)
    active=scale>np.finfo(float).tiny
    delta=np.sum((2*v[:,active]-np.abs(z[:,active])**2)/scale[active],axis=1).reshape(shape)
    indices=[np.r_[np.arange((n+1)//2),np.arange(1,(n+1)//2)] for n in shape]
    unsigned=tuple((n+1)//2 for n in shape)
    risk=np.zeros(unsigned)
    np.add.at(risk,np.ix_(*indices),delta)
    for axis in range(len(shape)):risk=np.cumsum(risk,axis=axis)
    pick=np.unravel_index(np.argmin(risk),risk.shape)
    mask=np.ones(shape,bool)
    for axis,(idx,k) in enumerate(zip(indices,pick)):
        view=[1]*len(shape);view[axis]=len(idx);mask &= (idx<=k).reshape(view)
    return mask.reshape(-1),dict(max_modes=pick,full_max_modes=tuple(n-1 for n in unsigned),
        retained_rows=int(mask.sum()),full_rows=int(mask.size),relative_retained=float(mask.mean()),
        estimated_risk_difference_from_dc=float(risk[pick]-risk[(0,)*len(shape)]),
        training_column_scales=scale,active_columns=int(active.sum()))


def gram_bundle(train,left,right,clean,mask):
    aa,bb=left['corrected'][mask],right['corrected'][mask]
    pa,pb=left['pilot'][mask],right['pilot'][mask]
    cc=clean[mask];tt=train['pilot'][mask]
    bundle={}
    for name,g in [('cross',np.real(aa.conj().T@bb)),('ordinary',np.real((aa.conj().T@aa+bb.conj().T@bb)/2)),
                   ('oracle',np.real(cc.conj().T@cc)),('uncorrected_cross',np.real(pa.conj().T@pb))]:
        bundle[name]=(g+g.T)/2
    bundle['scale']=np.sum(np.abs(tt)**2,axis=0);bundle['clean_scale']=np.sum(np.abs(cc)**2,axis=0)
    return bundle


def run_case(name,seed,output):
    started=time.monotonic();source,old=F.baseline(name,seed)
    p=R.datasets.load(name);axes=R.axes_of(p);cs,labels=R.panel(p);terms,opaque=R.symbols(p,labels)
    previous=json.loads((F.ROOT/'fit_results'/f'{name}_seed{seed}.json').read_text())
    frozen=F.evaluation_grams(p,old,terms,opaque,labels)
    print(name,seed,'B',old['draws'],'reconstructing independent groups',flush=True)
    groups=reconstruct_groups(p,old);folds=[]
    for g in range(3):
        segment,ax,opq,plan=Q.mapped_plan(old,g,previous['plans'][g]['plan'],axes,opaque)
        local=Q.Bank(ax,terms,opq,plan=plan)
        fits=local_fit(local,groups[g],segment,cs)
        saved=old['folds'][g]['arms']['loo']['counts'][str(old['draws'])]
        fits['old_clsm']=[(np.array(r['coefficients']),r['fit']) for r in saved]
        arm={}
        for method,fit in fits.items():arm['legacy_'+method]=F.scores(cs,fit,frozen,f'{g}_loo_')
        frequencies=[]
        for sel in old['groups'][g]['selection']:
            k=sel['loo']['max_mode'];frequencies.append(np.r_[np.arange(k+1),-np.arange(1,k+1)]*sel['spacing'])
        spectral=Q.Bank(axes,terms,opaque,frequencies=frequencies)
        mm=[moments(spectral,fields) for fields in groups]
        mask,risk=risk_prefix(mm[g],spectral.shape)
        # All fitting and bank-selection decisions are frozen before clean controls.
        clean=spectral.matrix(p.data)
        others=[i for i in range(3) if i!=g]
        for label,keep in [('full',np.ones(len(clean),bool)),('risk',mask)]:
            bundle=gram_bundle(mm[g],mm[others[0]],mm[others[1]],clean,keep)
            for method,fit in fits.items():
                row=F.scores(cs,fit,bundle,'')
                # A common frozen clean bank measures fit accuracy, so an easier
                # selected test set cannot masquerade as better coefficients.
                for a,b in zip(row,arm['legacy_'+method]):a['frozen_clean_energy']=b['clean_energy']
                arm[label+'_'+method]=row
            references=[c for c in cs if c['label']=='reference']
            truthfits=[(np.array(c['exact']),{}) for c in references]
            arm[label+'_exact_control']=F.scores(references,truthfits,bundle,'')
        folds.append(dict(training=g,local_plan=plan,quadrature=local.metadata,
                          spectral_quadrature=spectral.metadata,risk_selection=risk,arms=arm))
        print(' fold',g,'risk modes',risk['max_modes'],'/',risk['full_max_modes'],
              'elapsed',round(time.monotonic()-started,1),'s',flush=True)
    aggregate={}
    for method in folds[0]['arms']:
        if method.endswith('exact_control'):continue
        entry={}
        for metric in ('cross','ordinary','oracle','uncorrected_cross','clean_energy'):
            scores=np.mean([[r[metric] for r in f['arms'][method]] for f in folds],axis=0)
            entry[metric]=dict(scores=scores,dominance=R.dominance(scores,old['instability'],cs))
        aggregate[method]=entry
    result=dict(system=name,seed=seed,draws=old['draws'],candidates=cs,instability=old['instability'],
        folds=folds,aggregates=aggregate,elapsed_seconds=time.monotonic()-started,
        baseline_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(__file__),Path(Q.__file__)]})
    R.dump(output/f'{name}_seed{seed}.json',result)
    print('DONE',name,seed,round(result['elapsed_seconds'],1),'s',flush=True)


def check_risk():
    # Exhaustive independent +/- noise validates the prefix-risk difference
    # against actual projection MSE in a small orthogonal moment space.
    from itertools import product
    theta=np.array([2.,.4,0.,.4,0.])[:,None];sigma=np.full_like(theta,.3)
    risk=[];mse=[]
    for signs in product([-1.,1.],repeat=5):
        obs=theta+sigma*np.array(signs)[:,None]
        delta=2*sigma**2-obs**2
        actual=[];estimate=[]
        for k in range(3):
            keep=np.array([0,1,2,1,2])<=k
            actual.append(float(np.sum(np.where(keep[:,None],obs-theta,-theta)**2)))
            estimate.append(float(np.sum(delta[keep])))
        mse.append(actual);risk.append(estimate)
    em=np.mean(mse,axis=0);er=np.mean(risk,axis=0)
    np.testing.assert_allclose(em-em[0],er-er[0],atol=1e-14)
    R.dump(Q.ROOT/'risk_checks.json',dict(max_prefix_risk_difference_error=float(np.max(np.abs((em-em[0])-(er-er[0]))))))


def main():
    if R.os.name=='nt':R.ctypes.windll.kernel32.SetPriorityClass(R.ctypes.windll.kernel32.GetCurrentProcess(),0x4000)
    ap=argparse.ArgumentParser();ap.add_argument('--systems',nargs='+',default=R.datasets.names('core'))
    ap.add_argument('--seeds',nargs='+',type=int,default=[0,1,2]);ap.add_argument('--output',type=Path,default=Q.ROOT/'results')
    ap.add_argument('--checks-only',action='store_true');args=ap.parse_args()
    Q.checks();check_risk()
    if args.checks_only:return
    for name in args.systems:
        for seed in args.seeds:
            path=args.output/f'{name}_seed{seed}.json'
            if path.exists():print('SKIP',path.name,flush=True);continue
            run_case(name,seed,args.output)


if __name__=='__main__':main()
