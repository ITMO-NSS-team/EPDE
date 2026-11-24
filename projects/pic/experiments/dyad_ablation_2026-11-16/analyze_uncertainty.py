"""Descriptive paired calibration; no uncertainty penalty or selection gate."""
import criteria as K
R,np,ROOT=K.R,K.np,K.ROOT
import json,csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def write(name,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with (ROOT/name).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows)


def correlate(x,y):
    return float(np.corrcoef(x,y)[0,1]) if np.std(x)>0 and np.std(y)>0 else None


def main():
    jobs=json.loads((ROOT/'uncertainty_jobs.json').read_text());assert len(jobs)==20
    closest=list(csv.DictReader((K.M.ROOT/'core_closest_eligible_competitor.csv').open()))
    closest={(r['system'],int(r['seed']),int(r['slot']),r['candidate']) for r in closest}
    pairs=[];fronts=[];noise=[];cache={}
    for name,seed in jobs:
        path=ROOT/'uncertainty'/f'{name}_seed{seed}.json';meta=json.loads(path.read_text());a=np.load(path.with_suffix('.npz'))
        cs=meta['candidates'];cache[name,seed]=(meta,a)
        assert a['bootstrap'].shape==a['benchmark'].shape==(128,3,len(cs))
        assert all(np.isfinite(a[k]).all() for k in a.files),'Nonfinite results require explicit separate reporting'
        for var,model in meta['estimated_noise'].items():
            noise.append(dict(system=name,seed=seed,variable=var,estimated_sigma=model[0],actual_sigma=meta['actual_iid_sigma'][var],
                sigma_ratio=model[0]/meta['actual_iid_sigma'][var],estimated_rhos=json.dumps(model[1])))
        for slot in sorted({c['slot'] for c in cs}):
            ref=next(j for j,c in enumerate(cs) if c['slot']==slot and c['label']=='reference')
            for axis in (1,2):
                row=dict(system=name,seed=seed,slot=slot,metric=meta['axes'][axis])
                for key in ('observed','pilot_center','clean_center'):
                    ff=K.front(a[key][0],a[key][axis],cs,slot)
                    row[key+'_registered']=any(cs[j]['registered'] for j in ff)
                    row[key+'_reference']=ref in ff
                for key in ('bootstrap','benchmark'):
                    ff=[K.front(b[0],b[axis],cs,slot) for b in a[key]]
                    row[key+'_registered_count']=sum(any(cs[j]['registered'] for j in f) for f in ff)
                    row[key+'_reference_count']=sum(ref in f for f in ff)
                    row[key+'_mean_front_size']=float(np.mean([len(f) for f in ff]))
                fronts.append(row)
            for j,c in enumerate(cs):
                if c['slot']!=slot or c['registered']:continue
                margins={k:a[k][...,j]-a[k][...,ref] for k in a.files}
                errors={key:margins[key]-margins[center] for key,center in [('bootstrap','pilot_center'),('benchmark','clean_center')]}
                row=dict(system=name,seed=seed,slot=slot,reference=cs[ref]['id'],candidate=c['id'],
                    previous_closest=(name,seed,slot,c['id']) in closest)
                for axis,label in enumerate(meta['axes']):
                    boot,actual=errors['bootstrap'][:,axis],errors['benchmark'][:,axis]
                    lo,hi=np.quantile(boot,[.025,.975]);sdb=float(np.std(boot,ddof=1));sda=float(np.std(actual,ddof=1))
                    row.update({label+'_observed':float(margins['observed'][axis]),label+'_pilot_center':float(margins['pilot_center'][axis]),
                        label+'_clean_center':float(margins['clean_center'][axis]),label+'_bootstrap_sd':sdb,label+'_benchmark_sd':sda,
                        label+'_sd_ratio':sdb/sda if sda>0 else None,label+'_bootstrap_bias':float(np.mean(boot)),
                        label+'_benchmark_bias':float(np.mean(actual)),label+'_bootstrap_error_lo':float(lo),label+'_bootstrap_error_hi':float(hi),
                        label+'_coverage_count':int(np.sum((actual>=lo)&(actual<=hi))),
                        label+'_benchmark_negative_count':int(np.sum(margins['benchmark'][:,axis]<0)),
                        label+'_bootstrap_negative_count':int(np.sum(margins['bootstrap'][:,axis]<0)),
                        label+'_observed_error_over_sd':float((margins['observed'][axis]-margins['clean_center'][axis])/sda) if sda>0 else None})
                for axis,label in enumerate(meta['axes'][1:],1):
                    for key in ('bootstrap','benchmark'):
                        d,i=margins[key][:,0],margins[key][:,axis]
                        row[key+'_'+label+'_dominates_count']=int(np.sum((d<=0)&(i<=0)&((d<0)|(i<0))))
                        row[key+'_'+label+'_correlation']=correlate(d,i)
                pairs.append(row)
    write('uncertainty_pairs.csv',pairs);write('uncertainty_fronts.csv',fronts);write('uncertainty_noise.csv',noise)
    write('uncertainty_closest.csv',[r for r in pairs if r['previous_closest']])
    rows=[]
    for name in ['ALL',*sorted({r['system'] for r in pairs})]:
        pp=[r for r in pairs if name=='ALL' or r['system']==name]
        for axis in ('discrepancy','dyad_sum','dyad_prediction'):
            ratio=np.array([r[axis+'_sd_ratio'] for r in pp if r[axis+'_sd_ratio'] is not None])
            rows.append(dict(system=name,axis=axis,pairs=len(pp),
                mean_coverage=float(np.mean([r[axis+'_coverage_count']/128 for r in pp])),
                median_coverage=float(np.median([r[axis+'_coverage_count']/128 for r in pp])),
                median_sd_ratio=float(np.median(ratio)),minimum_sd_ratio=float(np.min(ratio)),maximum_sd_ratio=float(np.max(ratio))))
    write('uncertainty_calibration.csv',rows)
    summary=dict(records=len(jobs),equation_seed_cases=len(fronts)//2,pairs=len(pairs),draws_each=128,
        bootstrap_measurements=128*len(jobs),independent_benchmark_measurements=128*len(jobs),
        nonfinite=0,interpretation='Descriptive dependent-pair conditional diagnostics on selected records; no discovery confidence claim.',calibration=rows,
        noise_sigma_ratio_range=[min(r['sigma_ratio'] for r in noise),max(r['sigma_ratio'] for r in noise)])
    R.dump(ROOT/'uncertainty_summary.json',summary)
    labels=sorted({r['system'] for r in pairs});fig,axs=plt.subplots(1,2,figsize=(13,5))
    for axis,color in [('discrepancy','#087f8c'),('dyad_sum','#b74929'),('dyad_prediction','#6053a1')]:
        rr=[next(r for r in rows if r['system']==name and r['axis']==axis) for name in labels]
        axs[0].plot(labels,[r['mean_coverage'] for r in rr],'o-',label=axis,color=color)
        axs[1].plot(labels,[r['median_sd_ratio'] for r in rr],'o-',label=axis,color=color)
    axs[0].axhline(.95,color='grey',ls='--');axs[0].set_ylim(0,1.02);axs[0].set_ylabel('Mean paired error-range coverage')
    axs[1].axhline(1.,color='grey',ls='--');axs[1].set_yscale('log');axs[1].set_ylabel('Median bootstrap / benchmark SD')
    for ax in axs:
        ax.tick_params(axis='x',rotation=55);ax.spines[['top','right']].set_visible(False)
    axs[0].legend(fontsize=9);fig.suptitle('Conditional joint-objective uncertainty: 128 + 128 measurements per record')
    fig.tight_layout();fig.savefig(ROOT/'uncertainty_calibration.png',dpi=180);fig.savefig(ROOT/'uncertainty_calibration.svg');plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(13,5),sharey=True)
    rr=[r for r in fronts if r['system'] in ('ac','kdv_cossin','ks','wave')]
    for ax,metric in zip(axs,('dyad_sum','dyad_prediction')):
        ff=[r for r in rr if r['metric']==metric];xx=np.arange(len(ff))
        ax.bar(xx-.18,[r['bootstrap_registered_count']/128 for r in ff],.36,color='#087f8c',label='Pilot bootstrap')
        ax.bar(xx+.18,[r['benchmark_registered_count']/128 for r in ff],.36,color='#b74929',label='Actual repeated noise')
        ax.set_xticks(xx,[r['system']+' '+str(r['seed']) for r in ff],rotation=60,ha='right');ax.set_title(metric);ax.set_ylim(0,1.05)
        ax.spines[['top','right']].set_visible(False)
    axs[0].set_ylabel('Fraction retaining a registered equation');axs[0].legend(fontsize=9)
    fig.suptitle('Frozen main fits; dyad slab refits vary with every measurement')
    fig.tight_layout();fig.savefig(ROOT/'uncertainty_retention.png',dpi=180);fig.savefig(ROOT/'uncertainty_retention.svg');plt.close(fig)
    print(json.dumps({k:v for k,v in summary.items() if k!='calibration'},indent=2))
    print(json.dumps([r for r in rows if r['system']=='ALL'],indent=2))
    print('CLOSEST PAIRS')
    for r in pairs:
        if r['previous_closest'] and r['system'] in ('ac','kdv_cossin','ks','wave'):
            print(r['system'],r['seed'],r['candidate'],'D observed/SD',r['discrepancy_observed']/r['discrepancy_benchmark_sd'],
                  'D coverage',r['discrepancy_coverage_count']/128,'I coverage',r['dyad_sum_coverage_count']/128,
                  'joint dominations boot/actual',r['bootstrap_dyad_sum_dominates_count'],r['benchmark_dyad_sum_dominates_count'])


if __name__=='__main__':main()
