"""Post-hoc descriptive decomposition of already frozen uncertainty results."""
import criteria as K
R,np,ROOT=K.R,K.np,K.ROOT
import csv,json


def main():
    pairs=list(csv.DictReader((ROOT/'uncertainty_pairs.csv').open()))
    fronts=list(csv.DictReader((ROOT/'uncertainty_fronts.csv').open()))
    groups={'all':pairs,'previous_closest':[r for r in pairs if r['previous_closest']=='True'],
        'added_term':[r for r in pairs if ':add_' in r['candidate']],
        'dropped_term':[r for r in pairs if ':drop_' in r['candidate']],
        'reduced_form':[r for r in pairs if ':add_' not in r['candidate'] and ':drop_' not in r['candidate']]}
    coverage=[]
    for label,rows in groups.items():
        coverage.append(dict(group=label,pairs=len(rows),**{axis:float(np.mean([int(r[axis+'_coverage_count'])/128 for r in rows]))
            for axis in ('discrepancy','dyad_sum','dyad_prediction')}))
    ensemble_summary=[]
    for metric in ('dyad_sum','dyad_prediction'):
        rr=[r for r in fronts if r['metric']==metric]
        ensemble_summary.append(dict(metric=metric,cases=len(rr),
            bootstrap_mean_retention=float(np.mean([int(r['bootstrap_registered_count'])/128 for r in rr])),
            benchmark_mean_retention=float(np.mean([int(r['benchmark_registered_count'])/128 for r in rr])),
            benchmark_mean_front_size=float(np.mean([float(r['benchmark_mean_front_size']) for r in rr]))))
    examples=[]
    for name,seed,cid in [('ac',1,'e0:drop_0'),('ks',0,'e0:add_7'),('kdv_cossin',0,'e0:drop_0'),('wave',0,'e0:add_0')]:
        path=ROOT/'uncertainty'/f'{name}_seed{seed}.json';d=json.loads(path.read_text());a=np.load(path.with_suffix('.npz'))
        p=R.datasets.load(name);raw=p.noisy(5,seed);pilot,_=R.reconstruct(raw,R.axes_of(p),np.arange(len(R.axes_of(p)[0])))
        assert {v:R.digest(y) for v,y in pilot.items()}==d['pilot_sha256']
        cs=d['candidates'];ref=next(j for j,c in enumerate(cs) if c['id']=='e0:reference')
        j=next(j for j,c in enumerate(cs) if c['id']==cid)
        rr=next(r for r in pairs if r['system']==name and int(r['seed'])==seed and r['candidate']==cid)
        effects=[]
        for axis,label in enumerate(d['axes']):
            boot=a['bootstrap'][:,axis,j]-a['bootstrap'][:,axis,ref]
            actual=a['benchmark'][:,axis,j]-a['benchmark'][:,axis,ref]
            clean=a['clean_center'][axis,j]-a['clean_center'][axis,ref]
            center=a['pilot_center'][axis,j]-a['pilot_center'][axis,ref]
            sd=float(np.std(actual,ddof=1));center_shift=float(center-clean)
            effects.append(dict(axis=label,observed_margin=float(a['observed'][axis,j]-a['observed'][axis,ref]),
                pilot_margin=float(center),clean_margin=float(clean),pilot_center_shift=center_shift,
                pilot_center_shift_over_benchmark_sd=center_shift/sd if sd else None,
                bootstrap_bias=float(np.mean(boot)-center),benchmark_bias=float(np.mean(actual)-clean),
                bootstrap_sd=float(np.std(boot,ddof=1)),benchmark_sd=sd,coverage=int(rr[label+'_coverage_count'])/128,
                bootstrap_raw_margin_median=float(np.median(boot)),benchmark_raw_margin_median=float(np.median(actual))))
        examples.append(dict(system=name,seed=seed,candidate=cid,pilot_sha_reproduced=True,
            pilot_relative_l2_error={v:float(np.linalg.norm(pilot[v]-y)/np.linalg.norm(y)) for v,y in p.data.items()},
            reference_scores={k:a[k][:,ref].tolist() for k in ('observed','pilot_center','clean_center')},
            effects=effects))
    tails=[]
    for name,seed in json.loads((ROOT/'uncertainty_jobs.json').read_text()):
        path=ROOT/'uncertainty'/f'{name}_seed{seed}.json';d=json.loads(path.read_text());a=np.load(path.with_suffix('.npz'))
        for j,c in enumerate(d['candidates']):
            for key in ('bootstrap','benchmark'):
                x=a[key][:,1,j];median=float(np.median(x))
                tails.append(dict(system=name,seed=seed,candidate=c['id'],ensemble=key,
                    median=median,maximum=float(np.max(x)),maximum_over_median=float(np.max(x)/median) if median else None))
    tails.sort(key=lambda r:r['maximum_over_median'] or 0,reverse=True)
    intervals=[]
    for r in pairs:
        if r['previous_closest']!='True':continue
        gap=float(r['discrepancy_observed']);lo=gap-float(r['discrepancy_bootstrap_error_hi']);hi=gap-float(r['discrepancy_bootstrap_error_lo'])
        intervals.append(dict(system=r['system'],seed=int(r['seed']),slot=int(r['slot']),candidate=r['candidate'],
            observed_discrepancy_margin=gap,nominal_conditional_lower=lo,nominal_conditional_upper=hi,
            includes_zero=lo<=0<=hi,benchmark_error_coverage=int(r['discrepancy_coverage_count'])/128))
    with (ROOT/'closest_discrepancy_intervals.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,list(intervals[0]));w.writeheader();w.writerows(intervals)
    out=dict(scope='Post-hoc descriptions; no method or threshold selected using these diagnostics.',coverage_by_structure=coverage,
        selected_ensemble_summary=ensemble_summary,examples=examples,largest_dyad_tails=tails[:20],
        discrepancy_intervals=intervals,interval_scope='Observed margin minus bootstrap error quantiles; conditional diagnostic, not a production gate.',
        source_sha256={__file__:K.S.sha(__file__)},input_sha256={str(p):K.S.sha(p) for p in (ROOT/'uncertainty_pairs.csv',ROOT/'uncertainty_fronts.csv')})
    R.dump(ROOT/'uncertainty_diagnostics.json',out)
    print(json.dumps(coverage,indent=2));print(json.dumps(ensemble_summary,indent=2))
    for row in examples:
        print(row['system'],row['seed'],'pilot relative error',row['pilot_relative_l2_error'])
        for ef in row['effects']:print(ef)
    print('Largest tails',json.dumps(tails[:3],indent=2))


if __name__=='__main__':main()
