"""Paired reference-support summaries; keep incomplete cases explicit."""
import run_case as C
np,R,ROOT=C.np,C.R,C.ROOT
import json,csv
from collections import Counter,defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def write(name,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with (ROOT/name).open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,keys);writer.writeheader();writer.writerows(rows)


def main():
    files=sorted((ROOT/'results').glob('*.json'));assert len(files)==45
    folds=[];cases=[];failures=[];decisions=[];statuses=Counter();metric_status=Counter()
    metrics=['frozen_clean_energy']+[f'{stage}_{rule}' for stage in ('inner','outer','clean') for rule in C.RULES]
    for path in files:
        d=json.loads(path.read_text());groups=defaultdict(list);by_id={c['id']:c for c in d['candidates']}
        for g,fold in enumerate(d['folds']):
            for candidate,rules in fold['metrics'].items():
                for rule,meta in rules.items():metric_status[rule,meta['status']]+=1
            for row in fold['rows']:
                c=by_id[row['candidate']];statuses[row['bank'],row['method'],row['fit']['status']]+=1
                out=dict(system=d['system'],seed=d['seed'],outer_seed=d['outer_seed'],training=g,candidate=c['id'],slot=c['slot'],
                         reference=c['label']=='reference',strategy=row['option'],status=row['fit']['status'])
                out.update({k:row[k] for k in metrics if k in row});folds.append(out);groups[c['id'],out['strategy']].append(out)
                if row['fit']['status']!='ok':failures.append(dict(out,iterations=row['fit'].get('iterations'),negative=row['fit'].get('negative')))
            for decision in fold['decisions']:
                c=by_id[decision['candidate']];out=dict(system=d['system'],seed=d['seed'],outer_seed=d['outer_seed'],training=g,
                    candidate=c['id'],slot=c['slot'],reference=c['label']=='reference',strategy='select/'+decision['rule'],
                    status=decision['status'],selected_option=decision['option'],forced=decision['forced'])
                out.update({k:v for k,v in decision.get('evaluation',{}).items() if k in metrics})
                folds.append(out);groups[c['id'],out['strategy']].append(out)
                diag=dict(out)
                if decision['status']=='ok':
                    feasible={o['option'] for o in decision['options'] if o['status']=='ok' and o['inner'] is not None}
                    available=[r for r in fold['rows'] if r['candidate']==c['id'] and r['option'] in feasible]
                    best=min(available,key=lambda r:r['frozen_clean_energy'])
                    outer=min(available,key=lambda r:r['outer_'+decision['rule']])
                    diag.update(feasible_options=len(available),clean_best_option=best['option'],clean_best_energy=best['frozen_clean_energy'],
                                clean_regret=out['frozen_clean_energy']-best['frozen_clean_energy'],
                                outer_best_option=outer['option'],outer_choice_agrees=outer['option']==out['selected_option'])
                decisions.append(diag)
        for (cid,strategy),rows in groups.items():
            good=[r for r in rows if r['status']=='ok'];c=by_id[cid]
            out=dict(system=d['system'],seed=d['seed'],candidate=cid,slot=c['slot'],reference=c['label']=='reference',strategy=strategy,
                     successful_folds=len(good),forced_folds=sum(r.get('forced',False) for r in good))
            if len(good)==3:
                for m in metrics:
                    if all(m in r for r in good):out[m]=float(np.mean([r[m] for r in good]))
            cases.append(out)
    refs=[r for r in cases if r['reference']];summaries=[];comparison_rows=[];comparisons=[]
    pairs=[('mixed/corrected_gls','mixed/gls'),('mixed/corrected_ols','mixed/ols'),('mixed/corrected_gls','mixed/irls')]
    pairs += [('select/'+r,'mixed/irls') for r in C.RULES]
    pairs += [('select/'+r,'select/broad_white') for r in ('broad_transfer','mixed_transfer')]
    for method,baseline in pairs:
        rr=[]
        for a in refs:
            if a['strategy']!=method:continue
            b=next(r for r in refs if all(r[k]==a[k] for k in ('system','seed','candidate')) and r['strategy']==baseline)
            out=dict(system=a['system'],seed=a['seed'],candidate=a['candidate'],method=method,baseline=baseline,
                     method_folds=a['successful_folds'],baseline_folds=b['successful_folds'])
            if a['successful_folds']==3 and b['successful_folds']==3:
                aa,bb=a['frozen_clean_energy'],b['frozen_clean_energy'];tol=1e-12*max(1.,abs(bb))
                out.update(method_energy=aa,baseline_energy=bb,ratio=aa/bb if bb>0 else None,
                           outcome='better' if aa<bb-tol else 'worse' if aa>bb+tol else 'same')
            else:out['outcome']='incomplete'
            rr.append(out);comparison_rows.append(out)
        ratios=[r['ratio'] for r in rr if r.get('ratio') is not None]
        comparisons.append(dict(method=method,baseline=baseline,cases=len(rr),outcomes=dict(Counter(r['outcome'] for r in rr)),
                                median_ratio=float(np.median(ratios)) if ratios else None))
    for name in C.SYSTEMS:
        for cid in sorted({r['candidate'] for r in refs if r['system']==name}):
            for strategy in sorted({r['strategy'] for r in refs}):
                rr=[r for r in refs if r['system']==name and r['candidate']==cid and r['strategy']==strategy]
                good=[r for r in rr if r['successful_folds']==3]
                out=dict(system=name,candidate=cid,strategy=strategy,complete_cases=len(good),successful_folds=sum(r['successful_folds'] for r in rr))
                if len(good)==3:
                    for m in metrics:
                        if all(m in r for r in good):out['median_'+m]=float(np.median([r[m] for r in good]))
                summaries.append(out)
    write('per_fold.csv',folds);write('per_case.csv',cases);write('decisions.csv',decisions);write('failed_fits.csv',failures)
    write('paired_comparisons.csv',comparison_rows);write('system_summary.csv',summaries)
    out=dict(system_runs=45,reference_seed_cases=len(refs)//18,registered_equation_seed_cases=len(cases)//18,
             fit_statuses=[dict(bank=b,method=m,status=s,count=n) for (b,m,s),n in sorted(statuses.items())],
             metric_statuses=[dict(rule=r,status=s,count=n) for (r,s),n in sorted(metric_status.items())],comparisons=comparisons,
             selected_options={rule:dict(Counter(r.get('selected_option') for r in decisions if r['reference'] and r['strategy']=='select/'+rule)) for rule in C.RULES})
    R.dump(ROOT/'summary.json',out)
    # Comparable, dimensionless case ratios rather than pooling heterogeneous energies.
    show=[('mixed/corrected_gls','Corrected mixed'),('select/broad_white','Broad score'),('select/broad_transfer','Broad transfer'),('select/mixed_transfer','Mixed transfer')]
    fig,axs=plt.subplots(1,4,figsize=(14,5),sharey=True)
    for ax,(method,title) in zip(axs,show):
        rr=[r for r in comparison_rows if r['method']==method and r['baseline']=='mixed/irls']
        for i,name in enumerate(C.SYSTEMS):
            vals=[r['ratio'] for r in rr if r['system']==name and r.get('ratio') is not None and r['ratio']>0]
            if vals:ax.scatter(vals,i+np.linspace(-.15,.15,len(vals)),s=20,alpha=.75,color='#078492')
            missing=sum(r['outcome']=='incomplete' for r in rr if r['system']==name)
            if missing:ax.text(.98,i,f'{missing} unpaired',transform=ax.get_yaxis_transform(),ha='right',va='center',fontsize=7,color='#9c3415')
        ax.axvline(1,color='#64748b',lw=1);ax.set_xscale('log');ax.set_title(title);ax.set_xlabel('Energy / fixed mixed IRLS');ax.spines[['top','right']].set_visible(False)
    axs[0].set_yticks(range(15),C.SYSTEMS);axs[0].invert_yaxis();fig.suptitle('Core15: fresh noise seeds 12–14, known reference supports')
    fig.text(.5,.01,'Unpaired: method or baseline lacks a complete three-fold fit. Every selection rule completed every case.',ha='center',fontsize=8)
    fig.tight_layout(rect=(0,.035,1,1));fig.savefig(ROOT/'comparison.png',dpi=180);fig.savefig(ROOT/'comparison.svg');plt.close(fig)
    print(json.dumps(out,indent=2))


if __name__=='__main__':main()
