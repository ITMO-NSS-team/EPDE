"""Per-equation outcomes and scientific figures for the full fitting ablation."""
import run_fitting as F
R,A,np=F.R,F.A,F.np
import csv
import json
import hashlib
from pathlib import Path


def main():
    if R.os.name=='nt':R.ctypes.windll.kernel32.SetPriorityClass(R.ctypes.windll.kernel32.GetCurrentProcess(),0x4000)
    paths=sorted((F.ROOT/'fit_results').glob('*_seed*.json'))
    data=[json.loads(p.read_text(encoding='utf-8')) for p in paths]
    assert len(data)==45
    assert {(x['system'],x['seed']) for x in data} == {(n,s) for n in R.datasets.names('core') for s in range(3)}
    source=hashlib.sha256((F.ROOT/'run_fitting.py').read_bytes()).hexdigest()
    assert all(x['source_sha256']==source for x in data)
    names=['old_clsm','raw_ols','smooth_ols','raw_mean','smooth_mean','raw_clsm','smooth_clsm']
    rows=[]
    for x in data:
        assert x['baseline_sha256']==hashlib.sha256(Path(x['baseline']).read_bytes()).hexdigest()
        cs=x['candidates']
        for bandwidth in ('gcv','loo'):
            for method in names:
                actual=method if method in ('old_clsm','raw_ols','smooth_ols') else f"{method}_{x['draws']}"
                entry=x['aggregates'][bandwidth][actual]
                for slot in sorted({c['slot'] for c in cs}):
                    i=next(i for i,c in enumerate(cs) if c['slot']==slot and c['label']=='reference')
                    row=dict(system=x['system'],seed=x['seed'],slot=slot,bandwidth=bandwidth,method=method,
                             draws=x['draws'],reference_clean_energy=entry['clean_energy']['scores'][i],
                             reference_I=x['instability'][i])
                    for metric in ('cross','ordinary','oracle','uncorrected_cross'):
                        row[metric+'_valid_front']='|'.join(A.frontier(entry[metric]['scores'],x['instability'],cs,slot))
                        row[metric+'_reference']=entry[metric]['scores'][i]
                        row[metric+'_dominators']='|'.join(entry[metric]['dominance'][slot]['unregistered'])
                    rows.append(row)
    with (F.ROOT/'per_equation.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summary={}
    for bandwidth in ('gcv','loo'):
        summary[bandwidth]={}
        for method in names:
            selected=[r for r in rows if r['bandwidth']==bandwidth and r['method']==method]
            assert len(selected)==60
            summary[bandwidth][method]={metric:sum(bool(r[metric+'_valid_front']) for r in selected)
                for metric in ('cross','ordinary','oracle','uncorrected_cross')}
            ratios=[]
            for r in selected:
                old=next(o for o in rows if (o['system'],o['seed'],o['slot'],o['bandwidth'],o['method'])==
                         (r['system'],r['seed'],r['slot'],bandwidth,'old_clsm'))
                ratios.append(r['reference_clean_energy']/old['reference_clean_energy'])
            summary[bandwidth][method]['clean_energy_improved_cases']=sum(v<1 for v in ratios)
            summary[bandwidth][method]['median_clean_energy_ratio']=float(np.median(ratios))
    assert summary['loo']['old_clsm']['cross']==50
    R.dump(F.ROOT/'summary.json',summary)
    # Bootstrap-prefix changes describe computational sensitivity, not CIs.
    sensitivity=[]
    for x in data:
        for method in ('raw_clsm','smooth_clsm'):
            for bandwidth in ('gcv','loo'):
                entries=[x['aggregates'][bandwidth][f'{method}_{b}'] for b in (x['draws']//2,x['draws'])]
                for slot in sorted({c['slot'] for c in x['candidates']}):
                    front=[bool(A.frontier(e['cross']['scores'],x['instability'],x['candidates'],slot)) for e in entries]
                    if front[0]!=front[1]:sensitivity.append(dict(system=x['system'],seed=x['seed'],slot=slot,
                        method=method,bandwidth=bandwidth,half=front[0],full=front[1]))
    R.dump(F.ROOT/'bootstrap_sensitivity.json',sensitivity)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ordered=[(n,s) for n in R.datasets.names('core') for s in sorted({r['slot'] for r in rows if r['system']==n})]
    fig,ax=plt.subplots(figsize=(10,8))
    colors=['#777777','#147ba6','#cf7a20']
    for offset,method,color in zip([-.24,0,.24],['old_clsm','raw_clsm','smooth_clsm'],colors):
        values=[sum(bool(r['cross_valid_front']) for r in rows if r['system']==n and r['slot']==s
                    and r['bandwidth']=='loo' and r['method']==method) for n,s in ordered]
        ax.barh(np.arange(len(ordered))+offset,values,height=.22,color=color,label=method)
    ax.set_yticks(np.arange(len(ordered)),[f'{n} [{s}]' for n,s in ordered]);ax.invert_yaxis()
    ax.set_xticks([0,1,2,3]);ax.set_xlabel('Seeds retaining a catalog-valid equation / 3')
    ax.set_xlim(0,3.2);ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
    ax.set_title('Weak fitting ablation: all 15 systems, 5% noise\nFixed residual cross-product and dyad; LOO evaluation bank')
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.09),ncol=3)
    fig.tight_layout();fig.savefig(F.ROOT/'structural_results.png',dpi=180);fig.savefig(F.ROOT/'structural_results.svg');plt.close(fig)
    subset=['ks','kdv_cossin','pde_compound','pde_divide','ns','ac']
    fig,axes=plt.subplots(2,3,figsize=(13,7))
    for ax,name in zip(axes.flat,subset):
        for method,color in zip(['old_clsm','raw_clsm','smooth_clsm'],colors):
            rr=[r for r in rows if r['system']==name and r['bandwidth']=='loo' and r['method']==method]
            xx=[r['seed']+(.13*(r['slot']-1) if name=='ns' else 0.) for r in rr]
            ax.scatter(xx,[r['reference_clean_energy'] for r in rr],label=method,color=color,s=34)
        ax.set_title(name);ax.set_yscale('log');ax.set_xticks([0,1,2]);ax.set_xlabel('Noise seed (NS slots offset)');ax.grid(alpha=.2)
    axes[0,0].set_ylabel('Clean weak residual energy\n(noisy-fitted reference coefficients)')
    axes[1,0].set_ylabel('Clean weak residual energy')
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=3)
    fig.suptitle('Coefficient accuracy can improve while structural ranking worsens')
    fig.tight_layout(rect=(0,.04,1,.96));fig.savefig(F.ROOT/'coefficient_results.png',dpi=180)
    fig.savefig(F.ROOT/'coefficient_results.svg');plt.close(fig)
    audit=dict(cases=len(data),equation_seed_cases=60,
        max_frozen_score_scaled_difference=max(x['frozen_score_max_scaled_difference'] for x in data),
        same_panel_and_scoring=True,source_hashes_match=True,
        source_sha256={str(p.name):hashlib.sha256(p.read_bytes()).hexdigest() for p in F.ROOT.glob('*.py')},
        calculation_seconds=sum(x['elapsed_seconds'] for x in data))
    R.dump(F.ROOT/'audit.json',audit)
    print(json.dumps(summary,indent=2));print('Prefix sensitivity cases:',len(sensitivity))


if __name__=='__main__':main()
