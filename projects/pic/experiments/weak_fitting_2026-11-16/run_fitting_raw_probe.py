"""Train-only adaptive raw weak fitting; frozen evaluation and candidate panel.

No candidate equation is integrated. See protocol.txt and failure_analysis.txt.
"""
import probe as P
R,A,np=P.R,P.A,P.np
import argparse
import hashlib
import json
import time
from pathlib import Path

ROOT=P.ROOT


def baseline(name,seed):
    follow=R.ROOT/'followup128'/f'{name}_seed{seed}_b128.json'
    path=follow if follow.exists() else R.ROOT/'audited'/f'{name}_seed{seed}_b32.json'
    return path,json.loads(path.read_text(encoding='utf-8'))


def evaluation_grams(problem,old,terms,opaque,labels):
    """Sufficient quadratic forms for EXACTLY the earlier held-out scores."""
    path=ROOT/'evaluation_cache'/f"{old['system']}_s{old['seed']}_b{old['draws']}.npz"
    if path.exists():
        with np.load(path) as z:
            assert str(z['label_digest'])==hashlib.sha256(repr(labels).encode()).hexdigest()
            return {k:z[k].copy() for k in z.files if k!='label_digest'}
    axes=R.axes_of(problem);raw=problem.noisy(5,old['seed']);groups=[]
    for g,metadata in enumerate(old['groups']):
        idx=np.array(metadata['indices']);observed={v:y[idx] for v,y in raw.items()}
        assert all(R.digest(y)==metadata['observed_sha256'][v] for v,y in observed.items())
        pilot,_=R.reconstruct(observed,axes,idx)
        model=metadata['noise'];rng=np.random.default_rng(np.random.SeedSequence([1171,old['seed'],g]))
        fields=[pilot]
        for b in range(old['draws']):
            draw={v:pilot[v][idx]+R.WF.ar_noise(rng,y.shape,model[v][0],model[v][1]) for v,y in observed.items()}
            fields.append(R.reconstruct(draw,axes,idx)[0])
        groups.append(fields)
    grams={}
    for g in range(3):
        others=[i for i in range(3) if i!=g]
        for method in ('gcv','loo'):
            frequencies=[]
            for sel in old['groups'][g]['selection']:
                k=sel[method]['max_mode'];frequencies.append(np.r_[np.arange(k+1),-np.arange(1,k+1)]*sel['spacing'])
            bank=R.SpectralBank(axes,terms,opaque,frequencies)
            train=bank.matrix(groups[g][0]);clean=bank.matrix(problem.data)
            left=R.spectral_means(bank,groups[others[0]],[old['draws']])
            right=R.spectral_means(bank,groups[others[1]],[old['draws']])
            aa,bb=2*left[0]-left[1][old['draws']],2*right[0]-right[1][old['draws']]
            key=f'{g}_{method}_'
            for name,mat in [('cross',np.real(aa.conj().T@bb)),
                             ('ordinary',np.real((aa.conj().T@aa+bb.conj().T@bb)/2)),
                             ('oracle',np.real(clean.conj().T@clean)),
                             ('uncorrected_cross',np.real(left[0].conj().T@right[0]))]:
                grams[key+name]=(mat+mat.T)/2
            grams[key+'scale']=np.sum(np.abs(train)**2,axis=0)
            grams[key+'clean_scale']=np.sum(np.abs(clean)**2,axis=0)
        print(f"  frozen evaluation fold {g} cached",flush=True)
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,label_digest=hashlib.sha256(repr(labels).encode()).hexdigest(),**grams)
    return grams


def scores(candidates,fits,grams,key):
    rows=[]
    for c,(coef,diag) in zip(candidates,fits):
        vector=np.zeros(len(grams[key+'scale']));vector[c['yi']]=1.;vector[c['xi']]-=coef
        row=dict(id=c['id'],coefficients=coef,fit=diag)
        for metric in ('cross','ordinary','oracle','uncorrected_cross'):
            row[metric]=float(vector@grams[key+metric]@vector/grams[key+'scale'][c['yi']])
        row['clean_energy']=float(vector@grams[key+'oracle']@vector/grams[key+'clean_scale'][c['yi']])
        rows.append(row)
    return rows


def fit_group(problem,old,g,candidates,terms,opaque,raw):
    axes=R.axes_of(problem);idx=np.array(old['groups'][g]['indices'])
    observed={v:y[idx] for v,y in raw.items()};ax=[axes[0][idx],*axes[1:]]
    opq={k:v[idx] for k,v in opaque.items()};width=max(len(c['terms']) for c in candidates)
    chosen,seconds=P.selector(ax,terms,opq,observed,width,np.random.SeedSequence([692,old['seed'],g]),
                             sorted({c['yi'] for c in candidates}))
    plan=P.plan_dict(chosen)
    assert R.SV.row_floor('dyad',plan['shape'],width) is None
    bank=R.LocalBank(ax,terms,opq,plan)
    full_pilot,_=R.reconstruct(observed,axes,idx);pilot={v:y[idx] for v,y in full_pilot.items()}
    model={v:R.WF.noise_model(y) for v,y in observed.items()}
    mat,pm,moments=P.raw_moments(bank,observed,pilot,model,old['draws'],
        np.random.default_rng(np.random.SeedSequence([693,old['seed'],g])))
    methods={'weak_ols':[R.signed_fit(mat.T@mat,mat,c['xi'],c['yi']) for c in candidates]}
    for b,entry in moments.items():
        methods[f'weak_clsm_{b}']=[R.signed_fit(entry['corrected'],mat,c['xi'],c['yi']) for c in candidates]
        methods[f'weak_mean_{b}']=[R.signed_fit(entry['shifted'].T@entry['shifted'],mat,c['xi'],c['yi']) for c in candidates]
    # Oracle data are reached only after every method's coefficients are frozen.
    clean_mat=bank.matrix({v:y[idx] for v,y in problem.data.items()})
    diagnostics=[]
    for c in candidates:
        if c['label']!='reference':continue
        indices=[c['yi'],*c['xi']]
        def normratio(a,b):return np.linalg.norm(a,axis=0)/np.maximum(np.linalg.norm(b,axis=0),np.finfo(float).tiny)
        signal=clean_mat[:,indices]
        diagnostics.append(dict(slot=c['slot'],columns=indices,
            raw_column_relative_error=normratio(mat[:,indices]-signal,signal),
            pilot_column_relative_error=normratio(pm[:,indices]-signal,signal),
            estimated_noise_energy=np.diag(moments[old['draws']]['covariance'])[indices],
            clean_column_energy=np.sum(signal**2,axis=0)))
    return methods,dict(plan=plan,selector_seconds=seconds,selector_score=chosen.score,
                        selector_table=chosen.table,noise=model,diagnostics=diagnostics)


def run_case(name,seed,output):
    started=time.monotonic();source,old=baseline(name,seed)
    p=R.datasets.load(name);cs,labels=R.panel(p);terms,opaque=R.symbols(p,labels)
    A.common_width(max(len(c['terms']) for c in cs));raw=p.noisy(5,seed)
    print(f"{name} seed={seed} B={old['draws']}",flush=True)
    fits=[];plans=[]
    for g in range(3):
        fit,plan=fit_group(p,old,g,cs,terms,opaque,raw);fits.append(fit);plans.append(plan)
        print('  training',g,plan['plan'],'selector',round(plan['selector_seconds'],2),'s',flush=True)
    grams=evaluation_grams(p,old,terms,opaque,labels)
    folds=[];validation=0.
    for g in range(3):
        fold={}
        for bandwidth in ('gcv','loo'):
            key=f'{g}_{bandwidth}_';methods={}
            saved=old['folds'][g]['arms'][bandwidth]['counts'][str(old['draws'])]
            oldfits=[(np.array(r['coefficients']),r['fit']) for r in saved]
            methods['old_clsm']=scores(cs,oldfits,grams,key)
            for a,b in zip(methods['old_clsm'],saved):
                for metric in ('cross','ordinary','oracle','uncorrected_cross'):
                    delta=abs(a[metric]-b[metric]);validation=max(validation,delta/max(1,abs(b[metric])))
                    np.testing.assert_allclose(a[metric],b[metric],rtol=2e-7,atol=1e-9)
            for name_,fit in fits[g].items():methods[name_]=scores(cs,fit,grams,key)
            fold[bandwidth]=methods
        folds.append(fold)
    aggregates={}
    for bandwidth in ('gcv','loo'):
        aggregates[bandwidth]={}
        for method in folds[0][bandwidth]:
            entry={}
            for metric in ('cross','ordinary','oracle','uncorrected_cross','clean_energy'):
                d=np.mean([[r[metric] for r in f[bandwidth][method]] for f in folds],axis=0)
                entry[metric]=dict(scores=d,dominance=R.dominance(d,old['instability'],cs))
            aggregates[bandwidth][method]=entry
    result=dict(system=name,seed=seed,draws=old['draws'],candidates=cs,instability=old['instability'],
                plans=plans,folds=folds,aggregates=aggregates,baseline=str(source),
                baseline_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                frozen_score_max_scaled_difference=validation,elapsed_seconds=time.monotonic()-started)
    R.dump(output/f'{name}_seed{seed}.json',result)
    print('DONE',name,seed,round(result['elapsed_seconds'],1),'s; frozen-score error',validation,flush=True)


def main():
    if R.os.name=='nt':R.ctypes.windll.kernel32.SetPriorityClass(R.ctypes.windll.kernel32.GetCurrentProcess(),0x4000)
    ap=argparse.ArgumentParser();ap.add_argument('--systems',nargs='+',default=R.datasets.names('core'))
    ap.add_argument('--seeds',nargs='+',type=int,default=[0,1,2]);ap.add_argument('--output',type=Path,default=ROOT/'results')
    args=ap.parse_args()
    for name in args.systems:
        for seed in args.seeds:
            path=args.output/f'{name}_seed{seed}.json'
            if path.exists():print('SKIP',path.name,flush=True);continue
            run_case(name,seed,args.output)


if __name__=='__main__':main()
