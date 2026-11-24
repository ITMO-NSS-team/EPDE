"""Summaries at record level; rotated folds are not independent replicates."""
import run_outer as S
np,R,ROOT=S.np,S.R,S.ROOT
import json,csv
from collections import Counter,defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def csvout(name,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with (ROOT/name).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows)


def main():
    files=sorted((ROOT/'results').glob('*.json'));assert len(files)==12
    folds=[];cases=[];decisions=[];failures=[];statuses=Counter()
    metrics=['frozen_clean_energy']+[f'{stage}_{rule}' for stage in ('inner','outer','clean') for rule in ('native_broad','old_broad')]
    for path in files:
        d=json.loads(path.read_text());records=defaultdict(list)
        for g,fold in enumerate(d['folds']):
            for row in fold['rows']:
                statuses[row['bank'],row['method'],row['fit']['status']]+=1
                out=dict(system=d['system'],seed=d['seed'],outer_seed=d['outer_seed'],training=g,
                         candidate=row['candidate'],strategy=row['bank']+'_'+row['method'],status=row['fit']['status'])
                for key in metrics:
                    if key in row:out[key]=row[key]
                folds.append(out);records[out['candidate'],out['strategy']].append(out)
                if row['fit']['status']!='ok':failures.append(dict(out,iterations=row['fit'].get('iterations')))
            for decision in fold['decisions']:
                out=dict(system=d['system'],seed=d['seed'],outer_seed=d['outer_seed'],training=g,
                         candidate=decision['candidate'],strategy='select_'+decision['rule'],status=decision['status'],
                         selected_bank=decision['bank'],forced=decision['forced'])
                for key in metrics:
                    if key in decision.get('evaluation',{}):out[key]=decision['evaluation'][key]
                folds.append(out);records[out['candidate'],out['strategy']].append(out)
                rr=[r for r in fold['rows'] if r['candidate']==decision['candidate'] and r['method']=='irls'
                    and r['bank'] in ('local','broad') and r['fit']['status']=='ok']
                diag=dict(out)
                if out['status']=='ok':
                    best=min(rr,key=lambda r:r['frozen_clean_energy'])
                    diag.update(available=len(rr),clean_best_bank=best['bank'],clean_choice_correct=best['bank']==out['selected_bank'],
                                clean_best_energy=best['frozen_clean_energy'],clean_regret=out['frozen_clean_energy']-best['frozen_clean_energy'])
                    metric=decision['rule'];outerbest=min(rr,key=lambda r:r['outer_'+metric]);metricbest=min(rr,key=lambda r:r['clean_'+metric])
                    diag.update(outer_best_bank=outerbest['bank'],outer_choice_agrees=outerbest['bank']==out['selected_bank'],
                                clean_metric_best_bank=metricbest['bank'],clean_metric_choice_agrees=metricbest['bank']==out['selected_bank'],
                                outer_minus_inner=out['outer_'+metric]-out['inner_'+metric])
                decisions.append(diag)
        for (candidate,strategy),rows in records.items():
            good=[r for r in rows if r['status']=='ok'];out=dict(system=d['system'],seed=d['seed'],outer_seed=d['outer_seed'],
                candidate=candidate,strategy=strategy,successful_folds=len(good))
            if len(good)==3:
                for key in metrics:
                    if all(key in r for r in good):out[key]=float(np.mean([r[key] for r in good]))
                out['forced_folds']=sum(r.get('forced',False) for r in good)
                if strategy.startswith('select_'):
                    dd=[r for r in decisions if r['system']==d['system'] and r['seed']==d['seed'] and r['candidate']==candidate and r['strategy']==strategy]
                    out['clean_oracle_energy']=float(np.mean([r['clean_best_energy'] for r in dd]))
                    out['clean_regret']=out['frozen_clean_energy']-out['clean_oracle_energy']
            cases.append(out)
    summary=[];comparisons=[]
    for name in ('ac','kdv_cossin','wave','ks'):
        for strategy in sorted({r['strategy'] for r in cases}):
            rr=[r for r in cases if r['system']==name and r['strategy']==strategy and r['candidate']=='e0:reference']
            good=[r for r in rr if r['successful_folds']==3]
            out=dict(system=name,strategy=strategy,complete_cases=len(good),successful_folds=sum(r['successful_folds'] for r in rr))
            if len(good)==3:
                for metric in metrics:out['median_'+metric]=float(np.median([r[metric] for r in good]))
            summary.append(out)
        for rule in ('native_broad','old_broad'):
            for control in ('local_irls','broad_irls','mixed_irls'):
                pairs=[]
                for seed in (6,7,8):
                    a=next(r for r in cases if r['system']==name and r['seed']==seed and r['candidate']=='e0:reference' and r['strategy']=='select_'+rule)
                    b=next(r for r in cases if r['system']==name and r['seed']==seed and r['candidate']=='e0:reference' and r['strategy']==control)
                    if a['successful_folds']!=3 or b['successful_folds']!=3:continue
                    delta=a['frozen_clean_energy']-b['frozen_clean_energy']
                    tol=1e-12*max(1.,abs(b['frozen_clean_energy']))
                    pairs.append(dict(seed=seed,clean_delta=delta,clean_ratio=a['frozen_clean_energy']/b['frozen_clean_energy'],
                                      clean_outcome='better' if delta < -tol else 'worse' if delta>tol else 'same',
                                      outer_delta=a['outer_'+rule]-b['outer_'+rule]))
                comparisons.append(dict(system=name,rule=rule,control=control,complete_pairs=len(pairs),
                                        outcomes=dict(Counter(p['clean_outcome'] for p in pairs)),pairs=pairs))
    choice_summary=[]
    for rule in ('native_broad','old_broad'):
        for name in ('ac','kdv_cossin','wave','ks'):
            rr=[r for r in decisions if r['candidate']=='e0:reference' and r['strategy']=='select_'+rule and r['system']==name and r['status']=='ok']
            choice_summary.append(dict(system=name,rule=rule,folds=len(rr),forced=sum(r['forced'] for r in rr),
                clean_best_choices=sum(r['clean_choice_correct'] for r in rr),outer_agrees=sum(r['outer_choice_agrees'] for r in rr),
                clean_metric_agrees=sum(r['clean_metric_choice_agrees'] for r in rr),
                choices=dict(Counter(r['selected_bank'] for r in rr)),
                mean_outer_minus_inner=float(np.mean([r['outer_minus_inner'] for r in rr]))))
    csvout('per_fold.csv',folds);csvout('per_case.csv',cases);csvout('decisions.csv',decisions);csvout('failed_fits.csv',failures);csvout('summary.csv',summary)
    R.dump(ROOT/'summary.json',dict(system_runs=12,registered_equation_seed_cases=24,reference_seed_cases=12,
        fit_statuses=[dict(bank=b,method=m,status=s,count=n) for (b,m,s),n in sorted(statuses.items())],
        accuracy=summary,comparisons=comparisons,choice_summary=choice_summary))
    strategies=['local_irls','broad_irls','mixed_irls','select_native_broad','select_old_broad']
    titles=['AC','Forced KdV','Wave','KS'];xt=['Local','Broad','Mixed','Select: native','Select: interp.']
    fig,axs=plt.subplots(1,4,figsize=(14,4.5))
    for ax,name,title in zip(axs,('ac','kdv_cossin','wave','ks'),titles):
        for i,strategy in enumerate(strategies):
            rr=[r for r in cases if r['system']==name and r['strategy']==strategy and r['candidate']=='e0:reference' and r['successful_folds']==3]
            values=[r['frozen_clean_energy'] for r in rr];color='#078492' if strategy.startswith('select') else '#64748b'
            ax.scatter(i+np.linspace(-.09,.09,len(values)),values,s=17,alpha=.45,color=color)
            if len(values)==3:ax.scatter(i,np.median(values),s=65,color=color)
            else:ax.text(i,.1,f'{sum(r["successful_folds"] for r in cases if r["system"]==name and r["strategy"]==strategy and r["candidate"]=="e0:reference")}/9 fits',ha='center',transform=ax.get_xaxis_transform(),color='#a43b17')
        ax.set_yscale('log');ax.set_title(title);ax.set_xticks(range(5),xt,rotation=50,ha='right');ax.spines[['top','right']].set_visible(False)
    axs[0].set_ylabel('Frozen clean residual energy (lower is better)')
    fig.suptitle('Fresh noise seeds 6–8: fixed fits and frozen inner selection')
    fig.text(.5,.015,'Small points: individual records. Large points: three-record median. Outer records use independent noise seeds 9–11.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.03,1,.95));fig.savefig(ROOT/'comparison.png',dpi=180);fig.savefig(ROOT/'comparison.svg');plt.close(fig)
    print('Record summaries and plots written.')
    print(json.dumps(choice_summary,indent=2))


if __name__=='__main__':main()
