"""Three-way cross-fitted residual energy, with optional bootstrap mean correction.

One existing Kepler record; three interlaced, disjoint raw-sample groups.
Train coefficients, frequency slots and spectral bandwidth on one group;
evaluate residuals on the two other, independently reconstructed groups.
Compare Re<r_A,r_B>/S_train with (||r_A||^2+||r_B||^2)/(2*S_train).
No clipping, square root, noise-variance normalization or equation integration.
"""

import argparse
import copy
import ctypes
import json
import os
from pathlib import Path
import time

import window_profiles as W
import signal_refits as S
import spectral_weak_probe as P
import corrected_estimator as E
from epde.operators.common import survival as SV
import numpy as np
from scipy.interpolate import make_smoothing_spline
from scipy.fft import dct


IDS = P.REPRESENTATIVES


def cache_group(root, t, observed, seed, group, count):
    cache_dir=root/'residual_product_results'/'cache'
    cache_dir.mkdir(parents=True,exist_ok=True)
    path=cache_dir/f'kepler_seed{seed}_group{group}.npz'
    ix=np.arange(group,len(t),3)
    metadata=dict(version=1,seed=seed,group=group,observation_digest=W.digest(observed[ix]),
                  times_digest=W.digest(t),scipy=W.scipy.__version__,
                  weak_form_sha256=W.source_hash(W.REPO/'epde/preprocessing/weak_form.py'))
    rng=np.random.default_rng([20261116,907,seed,group])
    draws=[]
    if path.exists():
        with np.load(path) as z:
            if json.loads(str(z['metadata']))!=metadata:
                raise ValueError(f'Stale cache: {path}')
            pilot=z['pilot'].copy()
            draws=list(z['draws'].copy())
            model=(float(z['sigma']),z['rho'].tolist(),z['difference_order'].tolist())
            rng.bit_generator.state=json.loads(str(z['rng_state']))
        print(f'  cache seed={seed} group={group}: {len(draws)} draws',flush=True)
    else:
        pilot=np.asarray(make_smoothing_spline(t[ix],observed[ix],lam=None)(t))
        model=W.WF.noise_model(observed[ix])
    previous_count=len(draws)
    start=time.perf_counter()
    for draw in range(len(draws),count):
        error=W.WF.ar_noise(rng,(len(ix),),model[0],model[1])
        curve=make_smoothing_spline(t[ix],pilot[ix]+error,lam=None)
        draws.append(np.asarray(curve(t)))
        if (draw+1)%32==0:
            print(f'  seed={seed} group={group}: {draw+1}/{count} bootstrap reconstructions, {time.perf_counter()-start:.1f}s',flush=True)
    payload=dict(pilot=pilot,draws=np.stack(draws),indices=ix,sigma=np.asarray(model[0]),rho=np.asarray(model[1]),
                 difference_order=np.asarray(model[2]),metadata=json.dumps(metadata),rng_state=json.dumps(rng.bit_generator.state))
    if not path.exists() or len(draws)>previous_count:
        np.savez_compressed(path,**payload)
    return dict(pilot=pilot,draws=np.asarray(draws[:count]),indices=ix,
                noise_model=dict(sigma=model[0],rho=model[1],difference_order=model[2]),cache=str(path))


def population_check():
    # Enumerate independent finite distributions instead of relying on MC luck.
    ea=np.repeat(np.array([-2.,2.]),2)[:,None]*np.array([[1.,.5]])
    eb=np.tile(np.array([-3.,3.]),2)[:,None]*np.array([[.2,1.]])
    delta=np.array([.3,-.4])
    product=np.mean(np.sum((delta+ea)*(delta+eb),axis=1))
    energy=np.mean((np.sum((delta+ea)**2,axis=1)+np.sum((delta+eb)**2,axis=1))/2)
    np.testing.assert_allclose(product,delta@delta,atol=1e-14)
    assert energy>product
    ba,bb=np.array([.7,-.1]),np.array([-.2,.6])
    corrected=np.mean(np.sum(((delta+ba+ea)-ba)*((delta+bb+eb)-bb),axis=1))
    np.testing.assert_allclose(corrected,delta@delta,atol=1e-14)
    return dict(clean_squared_error=float(delta@delta),cross_product=float(product),
                ordinary_squared_error=float(energy),known_bias_corrected_cross_product=float(corrected))


def select_bandwidth_loo(t, observed):
    """Exact leave-one-out error for the same nested DCT projection family.

Secondary sensitivity added after a GCV split selected N-2 modes. It changes
no candidate ranking or threshold directly and uses the training group only.
"""
    n=len(observed)
    basis=np.sqrt(2/n)*np.cos(np.pi*(np.arange(n)[:,None]+.5)*np.arange(n)[None,:]/n)
    basis[:,0]=1/np.sqrt(n)
    transformed=dct(observed,type=2,norm='ortho')
    residuals=observed[:,None]-np.cumsum(basis*transformed[None,:],axis=1)[:,:-1]
    remaining=1-np.cumsum(basis*basis,axis=1)[:,:-1]
    loo=np.mean((residuals/remaining)**2,axis=0)
    k=int(np.argmin(loo))
    spacing=np.pi/(n*float(t[1]-t[0]))
    return np.arange(k+1)*spacing,dict(method='Exact leave-one-out DCT-II projection error on raw training u',
        selected_max_mode=k,selected_df=k+1,frequency_spacing=spacing,maximum_frequency=k*spacing,
        loo=loo.tolist(),dct_power=(transformed**2).tolist())


def loo_check():
    n=24
    t=np.arange(n,dtype=float)
    y=np.sin(t*.5)+.03*np.random.default_rng(3).normal(size=n)
    _,selection=select_bandwidth_loo(t,y)
    errors=[]
    for k in (0,4,20):
        design=np.sqrt(2/n)*np.cos(np.pi*(np.arange(n)[:,None]+.5)*np.arange(k+1)[None,:]/n)
        design[:,0]=1/np.sqrt(n)
        residuals=[]
        for i in range(n):
            keep=np.arange(n)!=i
            coef=np.linalg.lstsq(design[keep],y[keep],rcond=None)[0]
            residuals.append(y[i]-design[i]@coef)
        actual=float(np.mean(np.square(residuals)))
        predicted=selection['loo'][k]
        np.testing.assert_allclose(actual,predicted,rtol=1e-9,atol=1e-11)
        errors.append(abs(actual-predicted))
    return dict(maximum_brute_refit_difference=max(errors))


def residual(item, matrix, mass, index):
    return matrix[...,index[item['target']]]-matrix[...,[index[s] for s in item['terms']]]@np.asarray(item['coefficients'])-item['intercept']*mass


def summarize_pairs(case, fits, matrices, bank, index, train, evaluation, clean, old_i, count, corrected):
    left,right=evaluation
    weights=np.full(len(bank.frequencies),2.)
    weights[0]=1.
    norm=lambda a: np.sum(weights[None,:]*np.abs(a)**2,axis=(-2,-1))
    product=lambda a,b: np.sum(weights[None,:]*np.real(np.conjugate(a)*b),axis=(-2,-1))
    fitted={m['id']:m for m in fits}
    group_data=[]
    for m in matrices:
        draw=m['draws'][:count]
        average=draw.mean(axis=0)
        bias=average-m['pilot']
        group_data.append(dict(value=m['pilot']-bias if corrected else m['pilot'],
                               centered=draw-average,bias=bias))
    out=[]
    for template in case['models']:
        f=fitted[template['id']]
        item=dict(template,coefficients=f['coefficients'],intercept=f['intercept'])
        scale=float(norm(matrices[train]['pilot'][...,index[item['target']]]))
        if scale<=0: raise ValueError('Nonpositive training signal scale')
        ra=residual(item,group_data[left]['value'],bank.mass,index)
        rb=residual(item,group_data[right]['value'],bank.mass,index)
        # The bootstrap deviations have zero mean and no constant term.
        contrast=np.zeros(len(index))
        contrast[index[item['target']]]=1
        for label,coef in zip(item['terms'],item['coefficients']): contrast[index[label]]-=coef
        na=group_data[left]['centered']@contrast
        nb=group_data[right]['centered']@contrast
        point=float(product(ra,rb)/scale)
        ordinary=float((norm(ra)+norm(rb))/(2*scale))
        conditional=product(ra[None,...]+na,rb[None,...]+nb)/scale
        # Pair the SAME record-level replicate index in all rotations, preserving
        # reuse of an evaluation group across folds. Training fits are fixed here.
        rc=residual(item,clean,bank.mass,index)
        oracle=float(norm(rc)/scale)
        row=dict(id=item['id'],cross_product=point,ordinary_energy=ordinary,training_signal_scale=scale,
                 frozen_dyad=old_i[item['id']],training_dyad=f['objectives'][1],
                 clean_field_squared_error_same_coefficients=oracle,
                 conditional_bootstrap_scores=conditional.tolist(),
                 conditional_quantiles=dict(zip(('q05','median','q95'),np.quantile(conditional,[.05,.5,.95]).tolist())),
                 mean_correction_residual_norms=[float(np.sqrt(norm(group_data[g]['bias']@contrast)/scale)) for g in evaluation])
        np.testing.assert_allclose(point,float(product(rb,ra)/scale),rtol=1e-12,atol=1e-14)
        np.testing.assert_allclose(float(product(ra,ra)/scale),float(norm(ra)/scale),rtol=1e-12,atol=1e-14)
        if ordinary-point < -1e-12: raise ValueError('Squared difference identity failed')
        np.testing.assert_allclose(ordinary-point,float(norm(ra-rb)/(2*scale)),rtol=1e-10,atol=1e-12)
        out.append(row)
    return dict(models=out)


def run_fold(case, old_i, t, groups, clean, train, count, bandwidth='gcv'):
    ix=groups[train]['indices']
    # observed raw training values are stored separately; never use evaluation
    # groups for frequency selection, bandwidth selection, or coefficient fits.
    observed=groups[train]['observed']
    frequencies=S.mapped_frequencies(t[ix],observed,case['original_frequencies'])
    selector=select_bandwidth_loo if bandwidth=='loo' else P.select_bandwidth
    nu,selection=selector(t[ix],observed)
    labels=sorted({s for m in case['models'] for s in [m['target'],*m['terms']]})
    terms=[W.parse_term(s) for s in labels]
    index={s:i for i,s in enumerate(labels)}
    # All groups bracket indices 2..N-3. Bank's reach=1 enforces exactly that.
    banks=dict(local=P.Bank(t,case,terms,np.array([0.]),'local',1),
               spectral=P.Bank(t,case,terms,nu,'global',1))
    matrices={name:[] for name in banks}
    for group in groups:
        pilot_fields=W.fields(t,group['pilot'],frequencies,terms)
        build={name:dict(pilot=bank.matrix(pilot_fields),draws=[]) for name,bank in banks.items()}
        for field in group['draws'][:count]:
            values=W.fields(t,field,frequencies,terms)
            for name,bank in banks.items(): build[name]['draws'].append(bank.matrix(values))
        for name in banks:
            build[name]['draws']=np.stack(build[name]['draws'])
            matrices[name].append(build[name])
    local=matrices['local'][train]
    rows=len(local['pilot'])
    parts=SV._dyadic_slab_edges(rows,(rows,),SV.dyad_t_min_rows(case['width']))
    evaluation=[g for g in range(3) if g!=train]
    clean_matrices={name:bank.matrix(W.fields(t,clean,frequencies,terms)) for name,bank in banks.items()}
    result=dict(training_group=train,evaluation_groups=evaluation,token_frequencies=frequencies,
                selection=selection,banks={name:b.metadata for name,b in banks.items()},draw_counts={})
    checks=sorted({max(2,count//4),max(2,count//2),count})
    for b in checks:
        bias,covariance,n=E.moments(local['draws'][:b,:,0,:].real,local['pilot'][:,0,:].real)
        fits=[E.fit_equation(item,local['pilot'][:,0,:].real,index,bias,covariance,n,parts,'clsm') for item in case['models']]
        block=dict(training_fits=fits,arms={})
        for name,bank in banks.items():
            block['arms'][name]={}
            for method,corrected in (('uncorrected',False),('mean_corrected',True)):
                block['arms'][name][method]=summarize_pairs(case,fits,matrices[name],bank,index,train,evaluation,
                                                            clean_matrices[name],old_i,b,corrected)
        result['draw_counts'][str(b)]=block
    # Clean-input control: use identical noisy-training-selected frequencies and
    # banks, fit clean training reconstruction and evaluate clean groups. No truth
    # coefficient is supplied; this is only a labelled diagnostic.
    clean_group_fields=[W.fields(t,g['clean_pilot'],frequencies,terms) for g in groups]
    clean_local=banks['local'].matrix(clean_group_fields[train])[:,0,:].real
    clean_fits=[E.fit_equation(item,clean_local,index,np.zeros_like(clean_local),
                              np.zeros((len(clean_local),len(index),len(index))),count,parts,'clsm') for item in case['models']]
    result['clean_control']=dict(training_fits=clean_fits,arms={})
    for name,bank in banks.items():
        mm=[bank.matrix(f) for f in clean_group_fields]
        # Two identical pseudo-draws give zero bias/variance; no extra noise is
        # invented for this deterministic control.
        simple=[dict(pilot=a,draws=np.stack([a,a])) for a in mm]
        result['clean_control']['arms'][name]=summarize_pairs(case,clean_fits,simple,bank,index,train,evaluation,
                                                             clean_matrices[name],old_i,2,False)
    main=result['draw_counts'][str(count)]['arms']['spectral']['mean_corrected']['models']
    print(f"  train={train}, bandwidth={selection['selected_max_mode']}, corrected spectral products="
          f"{[(r['id'],round(r['cross_product'],7)) for r in main]}",flush=True)
    return result


def aggregate_records(records):
    out=dict(models=[])
    for position in range(len(records[0]['models'])):
        rows=[r['models'][position] for r in records]
        bootstrap=np.mean([r['conditional_bootstrap_scores'] for r in rows],axis=0)
        out['models'].append(dict(id=rows[0]['id'],frozen_dyad=rows[0]['frozen_dyad'],
            cross_product=float(np.mean([r['cross_product'] for r in rows])),
            ordinary_energy=float(np.mean([r['ordinary_energy'] for r in rows])),
            clean_field_squared_error_same_coefficients=float(np.mean([r['clean_field_squared_error_same_coefficients'] for r in rows])),
            fold_scores=[r['cross_product'] for r in rows],conditional_bootstrap_scores=bootstrap.tolist(),
            conditional_quantiles=dict(zip(('q05','median','q95'),np.quantile(bootstrap,[.05,.5,.95]).tolist()))))
    ref=out['models'][0]
    for r in out['models']:
        gaps=np.array(r['conditional_bootstrap_scores'])-np.array(ref['conditional_bootstrap_scores'])
        r['difference_from_reference']=r['cross_product']-ref['cross_product']
        r['conditional_gap_quantiles']=dict(zip(('q05','median','q95'),np.quantile(gaps,[.05,.5,.95]).tolist()))
        r['conditional_fraction_lower_than_reference']=float(np.mean(gaps<0))
    out['cross_product_dominators']=[r['id'] for r in out['models'][1:] if W.dominates(
        [r['cross_product'],r['frozen_dyad']],[ref['cross_product'],ref['frozen_dyad']])]
    out['ordinary_energy_dominators']=[r['id'] for r in out['models'][1:] if W.dominates(
        [r['ordinary_energy'],r['frozen_dyad']],[ref['ordinary_energy'],ref['frozen_dyad']])]
    out['clean_field_dominators']=[r['id'] for r in out['models'][1:] if W.dominates(
        [r['clean_field_squared_error_same_coefficients'],r['frozen_dyad']],
        [ref['clean_field_squared_error_same_coefficients'],ref['frozen_dyad']])]
    return out


def aggregate(folds, count):
    result={}
    for b in folds[0]['draw_counts']:
        result[b]={name:{method:aggregate_records([f['draw_counts'][b]['arms'][name][method] for f in folds])
                        for method in ('uncorrected','mean_corrected')} for name in ('local','spectral')}
    clean={name:aggregate_records([f['clean_control']['arms'][name] for f in folds]) for name in ('local','spectral')}
    return result,clean


def write(report, output):
    (output/'residual_cross_product.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    lines=['Three-way cross-fitted residual product','',report['protocol'],'',
           'Primary: spectral, mean-corrected residual product, mean of all three training rotations.',
           'Discrepancies are signed relative SQUARED errors; do not compare numerically to earlier L1 scores.',
           'Conditional bootstrap intervals omit fitting/selection uncertainty and are not calibrated confidence bounds.','']
    for trial in report['trials']:
        lines.append(f"seed {trial['seed']}")
        for b,arms in trial['aggregate'].items():
            for name,methods in arms.items():
                for method,result in methods.items():
                    lines.append(f"  B={b} {name} {method}: ordinary={result['ordinary_energy_dominators']}; product={result['cross_product_dominators']}; clean fields={result['clean_field_dominators']}")
        primary=trial['aggregate'][str(report['draws'])]['spectral']['mean_corrected']
        for r in primary['models']:
            lines.append(f"  {r['id']}: product={r['cross_product']:.8g}, ordinary={r['ordinary_energy']:.8g}, "
                         f"folds={r['fold_scores']}, I={r['frozen_dyad']:.7g}, gap_bootstrap={r['conditional_gap_quantiles']}")
        lines.append(f"  clean controls: { {k:v['cross_product_dominators'] for k,v in trial['clean_control'].items()} }")
    lines += ['', 'Limitations:', *report['limitations']]
    (output/'residual_cross_product.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def plot(report,output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,len(report['trials']),figsize=(4.7*len(report['trials']),4.8),constrained_layout=True)
    axes=np.atleast_1d(axes)
    colors=['#b12a31','#1976b9','#d78c21','#588d42','#7954aa']
    labels=['Ref','A','B','C','D']
    for trial,ax in zip(report['trials'],axes):
        rows=trial['aggregate'][str(report['draws'])]['spectral']['mean_corrected']['models']
        for k,row in enumerate(rows):
            lo,hi=row['conditional_quantiles']['q05'],row['conditional_quantiles']['q95']
            ax.vlines(k,lo,hi,color=colors[k],lw=3,alpha=.55)
            ax.scatter(k,row['cross_product'],color=colors[k],s=55)
            ax.scatter(k,row['ordinary_energy'],color=colors[k],s=40,marker='x')
        ax.axhline(0,color='black',lw=.8)
        ax.set_xticks(range(5),labels)
        ax.set(title=f"Seed {trial['seed']}",ylabel='Relative squared residual')
        ax.grid(axis='y',alpha=.2)
    fig.suptitle('Independent residual cross-products: Kepler\nDots: products; crosses: ordinary energy; lines: conditional bootstrap 5–95%')
    fig.savefig(output/'residual_products.png',dpi=180)
    fig.savefig(output/'residual_products.svg')
    plt.close(fig)


def main():
    if os.name=='nt': ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(),0x4000)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument('--draws',type=int,default=128)
    parser.add_argument('--bandwidth',choices=['gcv','loo'],default='gcv')
    parser.add_argument('--seeds',type=int,nargs='+',choices=[0,1,2])
    parser.add_argument('--output-dir',type=Path)
    args=parser.parse_args()
    if args.draws<8 or args.draws%4: raise ValueError('Use a draw budget divisible by four, >=8')
    root=args.root
    output=args.output_dir or root/('residual_product_results_loo' if args.bandwidth=='loo' else 'residual_product_results')
    output.mkdir(parents=True,exist_ok=True)
    source=root/'signal_results/signal_refits.json'
    corrected=root/'estimator_results/corrected_estimator.json'
    case=copy.deepcopy(json.loads(source.read_text(encoding='utf-8'))['cases'][1])
    original=json.loads(corrected.read_text(encoding='utf-8'))['cases'][1]
    case['models']=[m for m in case['models'] if m['id'] in IDS]
    assert [m['id'] for m in case['models']]==IDS
    S.set_active_config(S.load_search_config(overrides={'equation_terms_max_number':case['width'],
                          'system_rps_mode':'plain','instability_metric':'dyad'}))
    report=dict(protocol=__doc__,draws=args.draws,bandwidth_method=args.bandwidth,
        validation=dict(population=population_check(),loo=loo_check()),trials=[],
        sources={str(p):W.source_hash(p) for p in [source,corrected,Path(__file__),Path(P.__file__),Path(E.__file__),
                                                  W.REPO/'epde/preprocessing/weak_form.py']},
        limitations=[
            'Fixed archived supports were selected on the full record; this does not validate selection of new structures.',
            'Original full-record CLSM raw dyad coordinates are frozen solely to isolate the discrepancy axis.',
            'Independence of interlaced raw groups is valid for these iid-noise synthetic records, not arbitrary colored measurements.',
            'Conditional bootstrap spreads hold training coefficients, frequency slots, bandwidth and bias estimates fixed; they omit training/selection uncertainty.',
            'Rotations share observations. Their scores are not independent replicates; no standard error divided by sqrt(3) is used.',
            'Bootstrap means are plug-in bias estimates, not guarantees of unbiased nonlinear reconstruction.',
            'Training signal normalization is an estimated scale, fixed conditional on training data; it is not noise-variance normalization.',
            'Signed products can be negative. Values are not clipped; apparent ranking gains from negative estimates require uncertainty checks.',
            f'{args.draws} draws is a computational budget; quarter/half/full sensitivity is reported before any claim of MC stability.',
            'No candidate differential equation solve, extra experimental record, token ban, or production change.'])
    start=time.perf_counter()
    for trial,old in zip(case['trials'],original['trials']):
        seed=trial['seed']
        if args.seeds is not None and seed not in args.seeds:
            continue
        print(f'Kepler seed {seed}: three independent groups',flush=True)
        path=root/'signal_results'/f"{case['run']}_seed{seed}_signal.npz"
        with np.load(path) as z: t,observed,clean=[z[k].copy() for k in ('t','noisy','clean')]
        assert W.digest(observed)==trial['noise_digest']
        old_i={m['id']:m['objectives'][1] for m in old['stages']['raw']['methods']['clsm']['full_baseline']['models']}
        groups=[cache_group(root,t,observed,seed,g,args.draws) for g in range(3)]
        joined=np.concatenate([g['indices'] for g in groups])
        np.testing.assert_array_equal(np.sort(joined),np.arange(len(t)))
        assert len(np.unique(joined))==len(t)
        for g in groups:
            ix=g['indices']
            g['observed']=observed[ix]
            g['clean_pilot']=np.asarray(make_smoothing_spline(t[ix],clean[ix],lam=None)(t))
        folds=[run_fold(case,old_i,t,groups,clean,train,args.draws,args.bandwidth) for train in range(3)]
        combined,clean_result=aggregate(folds,args.draws)
        row=dict(seed=seed,source_hash=W.source_hash(path),
                 groups=[dict(indices=g['indices'].tolist(),noise_model=g['noise_model'],cache_sha256=W.source_hash(g['cache']),
                              used_draw_count=len(g['draws']),used_draws_digest=W.digest(g['draws']),pilot_digest=W.digest(g['pilot'])) for g in groups],
                 folds=folds,aggregate=combined,clean_control=clean_result)
        report['trials'].append(row)
        write(report,output)
        primary=combined[str(args.draws)]['spectral']['mean_corrected']
        print(f"  seed {seed}: ordinary dominators={primary['ordinary_energy_dominators']}; cross-product dominators={primary['cross_product_dominators']}",flush=True)
    report['runtime_seconds']=time.perf_counter()-start
    write(report,output)
    plot(report,output)
    print(f"Completed {report['runtime_seconds']:.1f}s: {output}",flush=True)


if __name__=='__main__': main()
