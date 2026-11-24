import criteria as K
R,np,ROOT=K.R,K.np,K.ROOT
import json,csv
from collections import Counter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def write(name,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with (ROOT/name).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows)


def main():
    files=sorted((ROOT/'results').glob('*.json'));assert len(files)==45
    rows=[];changes=[];candidate_rows=[]
    for path in files:
        d=json.loads(path.read_text())
        for row in d['fronts']:
            base=next(r for r in d['fronts'] if r['arm']==row['arm'] and r['slot']==row['slot'] and r['metric']=='dyad_sum')
            rows.append(dict(system=d['system'],seed=d['seed'],**{k:v for k,v in row.items() if k!='candidate_ids'},
                             baseline_retained=base['retained'],front_delta=row['size']-base['size']))
            if row['retained']!=base['retained']:
                changes.append(dict(system=d['system'],seed=d['seed'],slot=row['slot'],arm=row['arm'],metric=row['metric'],
                    change='gain' if row['retained'] else 'loss',front_ids=row['candidate_ids'],baseline_ids=base['candidate_ids']))
        for j,c in enumerate(d['candidates']):
            candidate_rows.append(dict(system=d['system'],seed=d['seed'],candidate=c['id'],slot=c['slot'],registered=c['registered'],
                features=len(c['xi']),**{k:v[j] for k,v in d['instability'].items()}))
    summary=[]
    for arm in sorted({r['arm'] for r in rows}):
        for metric in d['instability']:
            rr=[r for r in rows if r['arm']==arm and r['metric']==metric]
            summary.append(dict(arm=arm,metric=metric,cases=len(rr),retained=sum(r['retained'] for r in rr),reference_retained=sum(r['reference_retained'] for r in rr),
                gains=sum(r['retained'] and not r['baseline_retained'] for r in rr),losses=sum(not r['retained'] and r['baseline_retained'] for r in rr),
                mean_front_size=float(np.mean([r['size'] for r in rr])),median_front_size=float(np.median([r['size'] for r in rr])),
                mean_unregistered_front=float(np.mean([r['unregistered'] for r in rr])),
                mean_front_features=float(np.mean([r['mean_features'] for r in rr]))))
    primary='q0_white128/joint';baseline=next(r for r in summary if r['arm']==primary and r['metric']=='dyad_sum')
    assert baseline['retained']==56
    write('per_equation.csv',rows);write('summary.csv',summary);write('candidate_scores.csv',candidate_rows)
    R.dump(ROOT/'changes.json',changes);R.dump(ROOT/'summary.json',dict(system_runs=45,equation_seed_cases=60,primary=primary,summary=summary,
        maximum_dyad_replay_error=max(json.loads(p.read_text())['maximum_dyad_replay_error'] for p in files)))
    rr=[r for r in summary if r['arm']==primary];labels=[r['metric'] for r in rr];xx=np.arange(len(rr))
    fig,axs=plt.subplots(1,2,figsize=(12,5))
    bars=axs[0].bar(xx,[r['retained'] for r in rr],color=['#087f8c' if r['metric'].startswith('dyad') else '#64748b' for r in rr])
    axs[0].bar_label(bars,padding=3);axs[0].set_ylim(0,65);axs[0].set_ylabel('Registered equation retained / 60');axs[0].axhline(56,color='#a34a28',ls='--',lw=1)
    axs[1].bar(xx,[r['mean_front_size'] for r in rr],color='#64748b');axs[1].set_ylabel('Mean Pareto front size')
    for ax in axs:
        ax.set_xticks(xx,labels,rotation=45,ha='right');ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Dyad ablation: identical candidates, fits and discrepancy; reused seeds 0–2')
    fig.tight_layout();fig.savefig(ROOT/'comparison.png',dpi=180);fig.savefig(ROOT/'comparison.svg');plt.close(fig)
    print(json.dumps(summary,indent=2));print('PRIMARY CHANGES',json.dumps([r for r in changes if r['arm']==primary],indent=2))


if __name__=='__main__':main()
