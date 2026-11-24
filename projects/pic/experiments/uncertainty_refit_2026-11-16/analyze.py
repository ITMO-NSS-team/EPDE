"""Summaries of frozen pilot/refit ablations; no selected uncertainty gate."""
import experiment as E
R,np,ROOT=E.R,E.np,E.ROOT
import json,csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def write(name,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (ROOT/name).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)


def stats(boot,actual,bc,ac):
    be=boot-bc;ae=actual-ac;q=np.quantile(be,[.025,.975]);sb=float(np.std(be,ddof=1));sa=float(np.std(ae,ddof=1))
    return dict(bootstrap_center=float(bc),benchmark_center=float(ac),bootstrap_bias=float(np.mean(be)),benchmark_bias=float(np.mean(ae)),
        bootstrap_sd=sb,benchmark_sd=sa,sd_ratio=sb/sa if sa else None,
        coverage_count=int(np.sum((ae>=q[0])&(ae<=q[1]))),error_lo=float(q[0]),error_hi=float(q[1]),
        bootstrap_negative_count=int(np.sum(boot<0)),benchmark_negative_count=int(np.sum(actual<0)))


def allscores(a,tag):
    d=a[tag+'_discrepancy'];i=a[tag+'_dyad'];return np.stack([d[...,0,:],d[...,1,:],i],axis=-2)


def main():
    files=sorted((ROOT/'results').glob('*.json'));assert len(files)==45
    pairs=[];fronts=[];energies=[];coefficients=[];spreads=[];numerical=[];pilots=[]
    for path in files:
        d=json.loads(path.read_text());a=np.load(path.with_suffix('.npz'));name,seed=d['system'],d['seed'];cs=d['candidates']
        frozen=json.loads((ROOT/'frozen'/path.name).read_text());scores={tag:allscores(a,tag) for tag in ('observed','standard_center','twice_center','clean_center','standard','twice','benchmark')}
        assert all(np.isfinite(a[k]).all() for k in a.files)
        for var,v in d['pilot_relative_l2_error']['standard'].items():
            pilots.append(dict(system=name,seed=seed,variable=var,standard=v,twice=d['pilot_relative_l2_error']['twice'][var]))
        for tag in ('standard','twice','benchmark'):
            numerical.append(dict(system=name,seed=seed,ensemble=tag,main_refits=128*3*len(cs),
                indefinite_refits=int(np.sum(a[tag+'_negative'])),rank_deficient_refits=int(np.sum(a[tag+'_deficient'])),
                measurements_with_indefinite_refits=int(np.sum(a[tag+'_negative']>0))))
        for slot in sorted({c['slot'] for c in cs}):
            ref=next(j for j,c in enumerate(cs) if c['slot']==slot and c['label']=='reference')
            for mode,axis in [('fixed',0),('refit',1)]:
                for criterion in ('dyad','complexity'):
                    row=dict(system=name,seed=seed,slot=slot,mode=mode,criterion=criterion)
                    for tag in scores:
                        arr=scores[tag][None] if scores[tag].ndim==2 else scores[tag]
                        ff=[E.K.front(b[axis],b[2] if criterion=='dyad' else [len(c['xi']) for c in cs],cs,slot) for b in arr]
                        row[tag+'_registered_count']=sum(any(cs[j]['registered'] for j in f) for f in ff)
                        row[tag+'_reference_count']=sum(ref in f for f in ff)
                        row[tag+'_mean_size']=float(np.mean([len(f) for f in ff]))
                    fronts.append(row)
            for j,c in enumerate(cs):
                if c['slot']!=slot or c['registered']:continue
                margins={k:v[...,j]-v[...,ref] for k,v in scores.items()}
                sd=np.std(margins['benchmark'],axis=0,ddof=1)
                spreads.append(dict(system=name,seed=seed,slot=slot,candidate=c['id'],fixed_sd=sd[0],refit_sd=sd[1],ratio=sd[1]/sd[0] if sd[0] else None))
                for axis,label in enumerate(('fixed_D','refit_D','dyad')):
                    for pilot in ('standard','twice'):
                        st=stats(margins[pilot][:,axis],margins['benchmark'][:,axis],margins[pilot+'_center'][axis],margins['clean_center'][axis])
                        row=dict(system=name,seed=seed,slot=slot,candidate=c['id'],axis=label,pilot=pilot,
                            observed=float(margins['observed'][axis]),**st)
                        if axis<2:
                            for tag in (pilot,'benchmark'):
                                dd,ii=margins[tag][:,axis],margins[tag][:,2]
                                row[tag+'_joint_domination_count']=int(np.sum((dd<=0)&(ii<=0)&((dd<0)|(ii<0))))
                        pairs.append(row)
            c=cs[ref]
            energies.append(dict(system=name,seed=seed,slot=slot,candidate=c['id'],anchor_energy=float(a['anchor_energy'][ref]),
                **{tag+'_'+stat:float(fun(a[tag+'_energy'][:,ref])) for tag in ('standard','twice','benchmark') for stat,fun in [('median',np.median),('mean',np.mean),('maximum',np.max)]}))
            for g in range(3):
                for term,idx in enumerate(c['xi']):
                    ac=-a['clean_center_vectors'][g,idx,ref];actual=-a['benchmark_vectors'][:,g,idx,ref]
                    for pilot in ('standard','twice'):
                        bc=-a[pilot+'_center_vectors'][g,idx,ref];boot=-a[pilot+'_vectors'][:,g,idx,ref]
                        coefficients.append(dict(system=name,seed=seed,slot=slot,fold=g,term=term,pilot=pilot,
                            observed=-a['anchor_vectors'][g,idx,ref],exact=c['exact'][term],**stats(boot,actual,bc,ac)))
    for filename,rr in [('pair_calibration.csv',pairs),('fronts.csv',fronts),('reference_energy.csv',energies),
                        ('coefficient_calibration.csv',coefficients),('refit_spread.csv',spreads),('numerical_status.csv',numerical),('pilot_error.csv',pilots)]:write(filename,rr)
    calibration=[]
    for name in ['ALL',*sorted({r['system'] for r in pairs})]:
        for axis in ('fixed_D','refit_D','dyad'):
            for pilot in ('standard','twice'):
                rr=[r for r in pairs if (name=='ALL' or r['system']==name) and r['axis']==axis and r['pilot']==pilot]
                casekeys={(r['system'],r['seed'],r['slot']) for r in rr}
                casecoverage=[np.mean([r['coverage_count']/128 for r in rr if (r['system'],r['seed'],r['slot'])==key]) for key in casekeys]
                calibration.append(dict(system=name,axis=axis,pilot=pilot,pairs=len(rr),cases=len(casekeys),
                    mean_coverage=float(np.mean([r['coverage_count']/128 for r in rr])),
                    equal_case_coverage=float(np.mean(casecoverage)),median_coverage=float(np.median([r['coverage_count']/128 for r in rr])),
                    median_sd_ratio=float(np.median([r['sd_ratio'] for r in rr if r['sd_ratio'] is not None]))))
    write('calibration_summary.csv',calibration)
    reference_cal=[]
    for pilot in ('standard','twice'):
        rr=[r for r in coefficients if r['pilot']==pilot]
        reference_cal.append(dict(pilot=pilot,coefficients=len(rr),mean_coverage=float(np.mean([r['coverage_count']/128 for r in rr])),
            median_sd_ratio=float(np.median([r['sd_ratio'] for r in rr if r['sd_ratio'] is not None]))))
    fs=[]
    for mode in ('fixed','refit'):
        for criterion in ('dyad','complexity'):
            rr=[r for r in fronts if r['mode']==mode and r['criterion']==criterion]
            fs.append(dict(mode=mode,criterion=criterion,cases=len(rr),anchor_retained=sum(r['observed_registered_count'] for r in rr),
                **{tag+'_mean_retention':float(np.mean([r[tag+'_registered_count']/128 for r in rr])) for tag in ('standard','twice','benchmark')},
                benchmark_mean_size=float(np.mean([r['benchmark_mean_size'] for r in rr]))))
    ratios=[r['ratio'] for r in spreads if r['ratio'] is not None]
    summary=dict(status='COMPLETE',records=45,equation_seed_cases=60,measurements=45*3*128,
        pairs=len(spreads),score_nonfinite=0,calibration=[r for r in calibration if r['system']=='ALL'],
        coefficient_calibration=reference_cal,fronts=fs,
        empirical_refit_over_fixed_sd_quantiles=np.quantile(ratios,[.1,.5,.9,.99]),
        pilot_field_error_improved=sum(r['twice']<r['standard'] for r in pilots),pilot_field_error_total=len(pilots),
        numerical={tag:dict(main_refits=sum(r['main_refits'] for r in numerical if r['ensemble']==tag),
                           indefinite=sum(r['indefinite_refits'] for r in numerical if r['ensemble']==tag),
                           rank_deficient=sum(r['rank_deficient_refits'] for r in numerical if r['ensemble']==tag)) for tag in ('standard','twice','benchmark')})
    R.dump(ROOT/'summary.json',summary)
    labels=sorted({r['system'] for r in pairs});fig,axs=plt.subplots(1,3,figsize=(16,5),sharey=True)
    for ax,axis in zip(axs,('fixed_D','refit_D','dyad')):
        for pilot,color in [('standard','#087f8c'),('twice','#b74929')]:
            rr=[next(r for r in calibration if r['system']==name and r['axis']==axis and r['pilot']==pilot) for name in labels]
            ax.plot(labels,[r['equal_case_coverage'] for r in rr],'o-',color=color,label=pilot)
        ax.axhline(.95,color='grey',ls='--');ax.set_ylim(0,1.02);ax.set_title(axis);ax.tick_params(axis='x',rotation=70);ax.spines[['top','right']].set_visible(False)
    axs[0].legend();axs[0].set_ylabel('Mean conditional error-range coverage (equal cases)')
    fig.suptitle('Fresh anchors 18-20: standard versus twice-smoothed bias correction')
    fig.tight_layout();fig.savefig(ROOT/'calibration.png',dpi=180);fig.savefig(ROOT/'calibration.svg');plt.close(fig)
    fig,ax=plt.subplots(figsize=(11,4.5))
    distributions=[[r['ratio'] for r in spreads if r['system']==name and r['ratio'] is not None] for name in labels]
    ax.boxplot(distributions,tick_labels=labels,showfliers=False);ax.axhline(1,color='grey',ls='--');ax.set_yscale('log')
    ax.tick_params(axis='x',rotation=60);ax.set_ylabel('Actual refitted / fixed coefficient discrepancy SD')
    ax.set_title('Effect of main coefficient refits; boxes omit outlier marks, CSV retains every value')
    fig.tight_layout();fig.savefig(ROOT/'refit_spread.png',dpi=180);fig.savefig(ROOT/'refit_spread.svg');plt.close(fig)
    print(json.dumps(summary,indent=2,default=lambda x:x.tolist() if hasattr(x,'tolist') else str(x)))


if __name__=='__main__':main()
