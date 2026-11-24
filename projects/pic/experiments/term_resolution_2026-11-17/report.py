import csv
import json
from collections import Counter, defaultdict
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from study import ROOT


def load(name):
    return json.loads((ROOT/name).read_text())


def main():
    certs=load('certificates.json')['records']
    core=[r for r in certs if r['cohort']=='core']
    by_system=defaultdict(Counter);old_counts=Counter();rows=[]
    old_file=ROOT.parent/'full_pipeline_uncertainty_2026-11-17'/'coefficient_precision.csv'
    with old_file.open() as stream:
        old={(r['system'],r['profile'],int(r['seed']),r['candidate'],int(r['coefficient'])):
             r['zero_witness_found'] for r in csv.DictReader(stream) if r['coefficient']}
    for rec in core:
        for t in rec['rows']:
            if t['omitted'] is None:
                assert t['status']=='compatible'
                continue
            parent=next(x for x in rec['rows'] if x['candidate']==t['candidate'] and x['omitted'] is None)
            j=parent['free'].index(t['omitted'])
            key=(rec['system'],rec['profile'],rec['seed'],t['candidate'],j)
            status=old.get(key,'unbounded_parent')
            old_counts[status,t['status']]+=1
            by_system[rec['system'],rec['profile']][t['status']]+=1
            rows.append(dict(system=key[0],profile=key[1],seed=key[2],candidate=key[3],term=t['term'],
                             resolution=t['status'],old_zero_witness=status))
    with (ROOT/'core_deletions.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,rows[0]);w.writeheader();w.writerows(rows)
    totals=Counter(t['resolution'] for t in rows)
    scalar=load('kdv_scalar.json'); fresh=load('fresh_analysis.json')['records']
    projected=load('projected_audit.json')['records']
    prof_order={'iid':0,'positive':1,'negative':2}
    projected.sort(key=lambda r:(prof_order[r['profile']],r['seed']))
    points=np.array([r['ratios'] for r in projected])
    accepted=(points<=1).sum(axis=0)
    raw=load('fresh_replay.json'); replay=load('projected_replay.json')
    assert raw['status']==replay['status']=='PASS'
    assert load('audit.json')['exact_binary64_certificates'].get('unresolved',0)==0
    transfer=list(csv.DictReader((ROOT/'heldout_transfer.csv').open()))
    held=[r for r in transfer if r['cohort']=='core' and r['parent']=='False']
    summary=dict(primary=dict(fields=len(core),parents=sum(x['omitted'] is None for r in core for x in r['rows']),
                 terms=totals,old_checks={str(k):v for k,v in old_counts.items()},
                 transport=[sum(r['accepted']=='True' for r in held),len(held)],
                 by_system={f'{s}/{p}':c for (s,p),c in by_system.items()}),
                 scalar=dict(common=scalar['all_records_and_designs_intersection'],ols=scalar['locked']),
                 first_fresh={tag:{k:sum(r['decisions'][tag][k] for r in fresh) for k in fresh[0]['decisions'][tag]}
                              for tag in fresh[0]['decisions']},
                 second_fresh=dict(arms=['whole','projected'],candidates=['physical','dispersion_only'],
                 accepted=accepted.tolist(),noise_covered=np.asarray([r['covered_noise'] for r in projected]).sum(axis=0).tolist(),
                 warnings=sum(len(r['warnings']) for r in projected)),
                 first_cutoff_fraction_range=[min(r['cutoff_ratio'] for r in fresh),max(r['cutoff_ratio'] for r in fresh)])
    (ROOT/'summary.json').write_text(json.dumps(summary,indent=2))

    fig,axes=plt.subplots(1,2,figsize=(10.3,4.2),sharey=True,layout='constrained')
    x=np.arange(len(projected));labels=[r['profile']+'\n'+str(r['seed']) for r in projected]
    for i,ax in enumerate(axes):
        for j,(name,marker,color) in enumerate([('Physical KdV','o','#17675d'),
                                                ('Dispersion alone at −0.25','D','#993b14')]):
            ax.scatter(x+(j-.5)*.20,points[:,i,j],s=43,color=color,marker=marker,label=name,zorder=3)
        ax.axhline(1,color='#303030',ls='--',lw=1.1)
        ax.set_xticks(x,labels,fontsize=8);ax.set_xlim(-.5,5.5);ax.set_yscale('log')
        ax.set_title(('All residual components','Projection along the locked contrast')[i])
        ax.grid(axis='y',alpha=.2);ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Observed statistic / joint cutoff\n(values above 1 reject the equation)')
    fig.legend(*axes[0].get_legend_handles_labels(),loc='outside lower center',ncol=2,frameon=False)
    fig.suptitle('Forced KdV on six new noise fields, with the same calibration in each arm')
    fig.savefig(ROOT/'fresh_comparison.png',dpi=190);fig.savefig(ROOT/'fresh_comparison.pdf');plt.close(fig)
    print(json.dumps(summary['second_fresh']))


if __name__=='__main__':
    main()
