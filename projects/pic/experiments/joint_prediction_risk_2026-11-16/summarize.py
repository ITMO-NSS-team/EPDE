import run_joint as J
from pathlib import Path
import csv,json,hashlib,argparse


def write_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--suite',choices=['targeted','core'],default='targeted');args=ap.parse_args()
    names=['ac','wave','kdv_cossin','ks'] if args.suite=='targeted' else J.R.datasets.names('core')
    data=[json.loads((J.ROOT/'results'/f'{n}_seed{s}.json').read_text()) for n in names for s in range(3)]
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(J.__file__),Path(J.B.__file__),Path(J.Q.__file__)]}
    rows=[];choices=[];risk_options=[];failed_banks=[]
    for d in data:
        assert d['source_sha256']==hashes
        files=dict(local=J.L.ROOT/'results'/f"{d['system']}_seed{d['seed']}.json",continuous=J.Q.ROOT/'unit_results'/f"{d['system']}_seed{d['seed']}.json",
                   plan=J.F.ROOT/'fit_results'/f"{d['system']}_seed{d['seed']}.json")
        assert d['input_sha256']=={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in files.items()}
        cs=d['candidates']
        for method,a in d['aggregates'].items():
            norm=method.replace('_128','_full').replace('_64','_half');fit=method.split('_',1)[1]
            for slot in sorted({c['slot'] for c in cs}):
                i=next(i for i,c in enumerate(cs) if c['slot']==slot and c['label']=='reference')
                row=dict(system=d['system'],seed=d['seed'],slot=slot,method=norm,
                         frozen_clean_energy=d['aggregates']['legacy_'+fit]['clean_energy']['scores'][i])
                for metric in ('cross','ordinary','oracle'):
                    assert J.np.isfinite(a[metric]['scores']).all()
                    row[metric+'_reference']=a[metric]['scores'][i]
                    row[metric+'_valid_front']='|'.join(J.A.frontier(a[metric]['scores'],d['instability'],cs,slot))
                    row[metric+'_dominators']='|'.join(a[metric]['dominance'][slot]['unregistered'])
                rows.append(row)
        for fold in d['folds']:
            half={x['candidate']:x for x in fold['choices'] if x['draws']==64}
            for x in fold['choices']:
                if x['draws']!=128:continue
                choices.append(dict(system=d['system'],seed=d['seed'],training=fold['training'],candidate=x['candidate'],
                    selected=x['selected']['all'],half_selected=half[x['candidate']]['selected']['all'],
                    ols_only=x['selected']['ols'],clsm_only=x['selected']['clsm'],spread_only=x['selected']['spread']))
            for c in fold['options']:
                for option,entry in c['options'].items():
                    v=entry['diagnostics']['128']
                    risk_options.append(dict(system=d['system'],seed=d['seed'],training=fold['training'],candidate=c['candidate'],option=option,
                        risk=v['risk'],apparent=v['apparent'],optimism=v['optimism'],optimism_mcse=v['optimism_mcse'],
                        spread=v['prediction_spread'],negative_perturbed_fits=entry['negative_perturbed_fits'],
                        rank_deficient_perturbed_fits=entry['rank_deficient_perturbed_fits']))
            for tag,meta in fold['banks'].items():
                if not meta['feasible']:failed_banks.append(dict(system=d['system'],seed=d['seed'],training=fold['training'],bank=tag,reason=meta['reason']))
    summary={m:{metric:sum(bool(r[metric+'_valid_front']) for r in rows if r['method']==m)
                for metric in ('cross','ordinary','oracle')} for m in sorted({r['method'] for r in rows})}
    expected=6 if args.suite=='targeted' else 54
    assert summary['local_fixed_local_clsm']['cross']==expected
    prefix=args.suite+'_'
    write_csv(J.ROOT/(prefix+'per_equation.csv'),rows);write_csv(J.ROOT/(prefix+'choices.csv'),choices)
    write_csv(J.ROOT/(prefix+'risk_options.csv'),risk_options)
    J.R.dump(J.ROOT/(prefix+'infeasible_banks.json'),failed_banks)
    J.R.dump(J.ROOT/(prefix+'summary.json'),summary)
    lookup={(r['system'],r['seed'],r['slot'],r['method']):r for r in rows}
    selected=[r for r in rows if r['method']=='local_joint_all_full']
    stats=dict(system_runs=len(data),equation_seed_cases=len(selected),retention=summary['local_joint_all_full']['cross'],
        baseline_retention=expected,bank_choices=len(choices),half_full_bank_changes=sum(c['selected']!=c['half_selected'] for c in choices),
        clean_energy_improved=sum(r['frozen_clean_energy']<lookup[r['system'],r['seed'],r['slot'],'local_fixed_local_clsm']['frozen_clean_energy'] for r in selected),
        half_full_retention_changes=sum(bool(r['cross_valid_front'])!=bool(lookup[r['system'],r['seed'],r['slot'],'local_joint_all_half']['cross_valid_front']) for r in selected),
        broad_infeasible_fold_arms=len(failed_banks),max_baseline_scaled_error=max(d['frozen_score_max_scaled_error'] for d in data),
        source_and_input_hashes_verified=True,calculation_seconds=sum(d['elapsed_seconds'] for d in data))
    J.R.dump(J.ROOT/(prefix+'audit.json'),stats)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axs=plt.subplots(1,4,figsize=(13,3.6))
    methods=['local_fixed_local_clsm','local_fixed_local_ols','local_joint_all_full','local_spread_all_full']
    for ax,name in zip(axs,['ac','wave','kdv_cossin','ks']):
        values=[J.np.median([r['frozen_clean_energy'] for r in rows if r['system']==name and r['method']==m]) for m in methods]
        ax.bar(range(4),values,color=['#888888','#a7bfd2','#2176ae','#d78c2f']);ax.set_yscale('log')
        ax.set_xticks(range(4),['Local\nCLSM','Local\nOLS','Joint\nrisk','Spread\nonly']);ax.set_title(name);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    axs[0].set_ylabel('Median residual energy on fixed clean bank')
    fig.suptitle('Training-only selection: joint prediction risk vs coefficient spread')
    fig.tight_layout();fig.savefig(J.ROOT/(prefix+'accuracy.png'),dpi=180);fig.savefig(J.ROOT/(prefix+'accuracy.svg'));plt.close(fig)
    print(json.dumps(summary,indent=2));print('Audit:',json.dumps(stats))
    for name in ['ac','wave','kdv_cossin','ks']:
        print(name,{m:float(J.np.median([r['frozen_clean_energy'] for r in rows if r['system']==name and r['method']==m])) for m in methods})


if __name__=='__main__':main()
