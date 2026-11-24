"""Analytic zero-coefficient diagnostic and rank-preserving coordinate study."""
import os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT.parent/'uncertainty_refit_2026-11-16'))
import experiment as E
R,np,K=E.R,E.np,E.K
import json,csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def write(path,rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)


def analytic():
    K.common_width(1);n=2*R.SV._block_rows(1);assert n==16
    assert len(R.SV._dyadic_slab_edges(n,(n,),n//2))==1
    rng=np.random.default_rng(170917);ys=rng.standard_normal((256,n));observed=[];expected=[];scaled=[];constant=[]
    for y in ys:
        cc=y.reshape(2,-1).mean(axis=1);expected.append(abs(cc[0]-cc[1])/abs(cc.sum()))
        observed.append(float(R.SV.dyad_scores(np.ones((n,1)),y,np.ones(n),(n,),fit_intercept=False)[0]))
        scaled.append(float(R.SV.dyad_scores(np.ones((n,1)),1e-8*y,np.ones(n),(n,),fit_intercept=False)[0]))
        constant.append(float(R.SV.dyad_scores(np.ones((n,1)),1.3+1e-8*y,np.ones(n),(n,),fit_intercept=False)[0]))
    np.testing.assert_allclose(observed,expected,rtol=2e-9,atol=2e-10);np.testing.assert_allclose(scaled,observed,rtol=2e-9,atol=2e-10)
    exactzero=float(R.SV.dyad_scores(np.ones((n,1)),np.zeros(n),np.ones(n),(n,),fit_intercept=False)[0]);assert exactzero==0
    c=rng.standard_normal((1048576,2));ratio=np.abs(c[:,0]-c[:,1])/np.abs(c[:,0]+c[:,1]);q=np.array([.5,.9,.95,.99])
    empirical=np.quantile(ratio,q);theory=np.tan(np.pi*q/2)
    # Distributional checks use fixed deterministic simulations, not a candidate acceptance gate.
    assert np.max(np.abs(np.mean(ratio[:,None]<=theory,axis=0)-q))<.003
    prefix=[dict(n=m,mean=float(np.mean(ratio[:m])),sd=float(np.std(ratio[:m],ddof=1)),median=float(np.median(ratio[:m]))) for m in (128,512,4096,65536,1048576)]
    batches=ratio.reshape(-1,128);sample_sd=np.std(batches,axis=1,ddof=1)
    out=dict(production_rows=n,production_draws=len(ys),exact_zero_score=exactzero,
        score_at_sigma1_median=float(np.median(observed)),score_at_sigma_1e_minus8_median=float(np.median(scaled)),
        nonzero_constant_score_at_sigma_1e_minus8_median=float(np.median(constant)),
        production_formula_max_relative_error=float(np.max(np.abs(np.array(observed)-expected)/np.maximum(1.,expected))),
        quantile_levels=q,empirical_quantiles=empirical,analytic_quantiles=theory,prefix_statistics=prefix,
        sd_across_128_draw_batches_quantiles=np.quantile(sample_sd,[.1,.5,.9,.99,.999]),
        uncapped_analytic_mean='infinite',uncapped_analytic_variance='does not exist',
        caveat='Production float cap is finite; the uncapped model describes the near-zero-median ratio pathology, not every real candidate.')
    R.dump(ROOT/'analytic_checks.json',out)
    xx=np.logspace(-1,4,100);tail=np.array([np.mean(ratio>x) for x in xx]);fig,axs=plt.subplots(1,2,figsize=(11,4.5))
    axs[0].loglog(xx,tail,label='Gaussian slab simulation');axs[0].loglog(xx,2/np.pi*np.arctan(1/xx),'--',label='Absolute-Cauchy tail')
    axs[0].set_xlabel('Dyad score');axs[0].set_ylabel('Tail probability');axs[0].legend()
    sigmas=np.logspace(-8,0,9);zero=[];nonzero=[]
    for sig in sigmas:
        cc=sig*c/np.sqrt(n//2);zero.append(float(np.median(np.abs(cc[:,0]-cc[:,1])/np.abs(cc.sum(axis=1)))))
        cc=cc+1.3;nonzero.append(float(np.median(np.abs(cc[:,0]-cc[:,1])/np.abs(cc.sum(axis=1)))))
    axs[1].loglog(sigmas,zero,'o-',label='True coefficient 0');axs[1].loglog(sigmas,nonzero,'o-',label='True constant coefficient 1.3')
    axs[1].set_xlabel('Observation noise SD');axs[1].set_ylabel('Median two-slab dyad');axs[1].legend()
    fig.suptitle('A zero coefficient is a nonregular point of the relative-MAD statistic')
    fig.tight_layout();fig.savefig(ROOT/'zero_coefficient.png',dpi=180);fig.savefig(ROOT/'zero_coefficient.svg');plt.close(fig)
    return out


def main():
    theory=analytic();files=sorted((E.ROOT/'results').glob('*.json'));assert len(files)==45
    transforms={'raw':lambda x:x,'log1p':np.log1p,'bounded':lambda x:x/(1+x)};rows=[];ordering_checks=0;inputs={}
    for path in files:
        d=json.loads(path.read_text());a=np.load(path.with_suffix('.npz'));cs=d['candidates'];inputs[str(path.with_suffix('.npz'))]=E.S.sha(path.with_suffix('.npz'))
        raw={tag:a[tag+'_dyad'] for tag in ('observed','standard_center','twice_center','clean_center','standard','twice','benchmark')}
        for transform,fn in transforms.items():
            values={k:fn(v) for k,v in raw.items()}
            for tag,x in raw.items():
                y=values[tag];np.testing.assert_array_equal(np.sign(x[...,None,:]-x[...,:,None]),np.sign(y[...,None,:]-y[...,:,None]))
                ordering_checks+=int(np.prod(x.shape[:-1]))*len(cs)**2
            for slot in sorted({c['slot'] for c in cs}):
                ref=next(j for j,c in enumerate(cs) if c['slot']==slot and c['label']=='reference')
                for j,c in enumerate(cs):
                    if c['slot']!=slot or c['registered']:continue
                    actual=values['benchmark'][:,j]-values['benchmark'][:,ref]-(values['clean_center'][j]-values['clean_center'][ref])
                    for pilot in ('standard','twice'):
                        boot=values[pilot][:,j]-values[pilot][:,ref]-(values[pilot+'_center'][j]-values[pilot+'_center'][ref])
                        q=np.quantile(boot,[.025,.975]);sa=float(np.std(actual,ddof=1));sb=float(np.std(boot,ddof=1))
                        rows.append(dict(system=d['system'],seed=d['seed'],slot=slot,candidate=c['id'],transform=transform,pilot=pilot,
                            coverage_count=int(np.sum((actual>=q[0])&(actual<=q[1]))),sd_ratio=sb/sa if sa else None,
                            bootstrap_sd=sb,benchmark_sd=sa,bootstrap_bias=float(np.mean(boot)),benchmark_bias=float(np.mean(actual)),
                            error_lo=float(q[0]),error_hi=float(q[1])))
    write(ROOT/'pair_calibration.csv',rows);summary=[]
    for name in ['ALL',*sorted({r['system'] for r in rows})]:
        for transform in transforms:
            for pilot in ('standard','twice'):
                rr=[r for r in rows if (name=='ALL' or r['system']==name) and r['transform']==transform and r['pilot']==pilot]
                keys={(r['system'],r['seed'],r['slot']) for r in rr}
                summary.append(dict(system=name,transform=transform,pilot=pilot,pairs=len(rr),mean_coverage=float(np.mean([r['coverage_count']/128 for r in rr])),
                    equal_case_coverage=float(np.mean([np.mean([r['coverage_count']/128 for r in rr if (r['system'],r['seed'],r['slot'])==key]) for key in keys])),
                    median_sd_ratio=float(np.median([r['sd_ratio'] for r in rr if r['sd_ratio'] is not None]))))
    write(ROOT/'summary.csv',summary)
    baseline=json.loads((E.ROOT/'summary.json').read_text())
    for pilot in ('standard','twice'):
        actual=next(r for r in summary if r['system']=='ALL' and r['transform']=='raw' and r['pilot']==pilot)
        previous=next(r for r in baseline['calibration'] if r['axis']=='dyad' and r['pilot']==pilot)
        np.testing.assert_allclose(actual['mean_coverage'],previous['mean_coverage'],rtol=0,atol=1e-15)
    out=dict(status='PASS',analytic_model_checked=True,production_formula_verified=True,pairwise_order_checks=ordering_checks,
        all_observed_pareto_orderings_preserved=True,raw_calibration_reproduced=True,
        scope='Exploratory reanalysis of the same45 fresh-anchor records; not another independent replication.',
        summary=[r for r in summary if r['system']=='ALL'],source_sha256={str(p):E.S.sha(p) for p in [Path(__file__),ROOT/'protocol.txt',Path(R.SV.__file__)]},input_sha256=inputs)
    R.dump(ROOT/'audit_summary.json',out);print(json.dumps(out['summary'],indent=2));print('PAIR ORDER CHECKS',ordering_checks);print('THEORY',json.dumps(theory,default=lambda x:x.tolist() if hasattr(x,'tolist') else str(x)))


if __name__=='__main__':main()
