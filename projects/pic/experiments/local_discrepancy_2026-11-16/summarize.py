import run_local as L
R,A,np=L.R,L.A,L.np
from pathlib import Path
import csv,json,hashlib


def write_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    data=[json.loads(p.read_text()) for p in (L.ROOT/'results').glob('*_seed*.json')]
    assert len(data)==45
    assert {(d['system'],d['seed']) for d in data}=={(n,s) for n in R.datasets.names('core') for s in range(3)}
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(L.__file__),Path(L.Q.__file__),Path(L.P.__file__)]}
    rows=[];pairs=[];exact=[]
    for d in data:
        assert d['source_sha256']==hashes
        files=dict(continuous_result=L.Q.ROOT/'unit_results'/f"{d['system']}_seed{d['seed']}.json",
                   training_plan=L.F.ROOT/'fit_results'/f"{d['system']}_seed{d['seed']}.json")
        assert d['input_sha256']=={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in files.items()}
        global_unc=json.loads((L.P.ROOT/'results'/f"{d['system']}_seed{d['seed']}.json").read_text())['models']['128']
        global_pairs={(p['reference'],p['candidate']):p for p in global_unc['pairs']}
        global_exact={p['slot']:p for p in global_unc['exact']}
        b=d['draws'];cs=d['candidates']
        for method,entry in d['aggregates'].items():
            normalized=method.replace(f'_clsm_{b}','_clsm').replace(f'_clsm_{b//2}','_clsm_half')
            fit=method.split('_',1)[1];frozen='legacy_'+fit
            for slot in sorted({c['slot'] for c in cs}):
                i=next(i for i,c in enumerate(cs) if c['slot']==slot and c['label']=='reference')
                row=dict(system=d['system'],seed=d['seed'],slot=slot,method=normalized,
                    frozen_clean_energy=d['aggregates'][frozen]['clean_energy']['scores'][i],
                    selected_clean_energy=entry['clean_energy']['scores'][i])
                for metric in ('cross','ordinary','oracle','uncorrected_cross'):
                    assert np.isfinite(entry[metric]['scores']).all()
                    row[metric+'_reference']=entry[metric]['scores'][i]
                    row[metric+'_valid_front']='|'.join(A.frontier(entry[metric]['scores'],d['instability'],cs,slot))
                    row[metric+'_dominators']='|'.join(entry[metric]['dominance'][slot]['unregistered'])
                rows.append(row)
        for method,model in d['uncertainty'].items():
            normalized=method.replace(f'_clsm_{b}','_clsm')
            for p in model['pairs']:
                if not p['reference'].endswith(':reference'):continue
                old=global_pairs[p['reference'],p['candidate']]
                pairs.append(dict(system=d['system'],seed=d['seed'],method=normalized,slot=p['slot'],
                    reference=p['reference'],candidate=p['candidate'],dyad_eligible=p['dyad_eligible'],
                    local_point_dominator=p['point_dominator'],local_gap=p['gap'],local_clean_gap=p['oracle_gap'],
                    local_sd=p['paired_sd'],local_lo95=p['interval95'][0],local_hi95=p['interval95'][1],
                    local_oracle_in_95=p['oracle_in_95'],spectral_continuous_gap=old['gap'],
                    spectral_continuous_clean_gap=old['oracle_gap'],spectral_continuous_sd=old['paired_sd'],
                    spectral_continuous_point_dominator=old['point_dominator'],
                    local_to_spectral_continuous_sd=p['paired_sd']/old['paired_sd'] if old['paired_sd'] else None))
            for p in model['exact']:
                if method!='continuous':continue # identical exact controls across fit arms
                old=global_exact[p['slot']]
                exact.append(dict(system=d['system'],seed=d['seed'],slot=p['slot'],
                    local_score=p['score'],local_oracle=p['oracle'],local_sd=p['sd'],
                    local_lo95=p['interval95'][0],local_hi95=p['interval95'][1],local_oracle_in_95=p['oracle_in_95'],
                    spectral_score=old['score'],spectral_oracle=old['oracle'],spectral_sd=old['sd']))
    summary={}
    for method in sorted({r['method'] for r in rows}):
        selected=[r for r in rows if r['method']==method];assert len(selected)==60
        summary[method]={m:sum(bool(r[m+'_valid_front']) for r in selected) for m in ('cross','ordinary','oracle','uncorrected_cross')}
    assert summary['legacy_old']['cross']==50 and summary['legacy_continuous']['cross']==53
    write_csv(L.ROOT/'per_equation.csv',rows);write_csv(L.ROOT/'paired_comparison.csv',pairs)
    write_csv(L.ROOT/'exact_controls.csv',exact);R.dump(L.ROOT/'summary.json',summary)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    keys=[(n,s) for n in R.datasets.names('core') for s in sorted({r['slot'] for r in rows if r['system']==n})]
    methods=['legacy_old','legacy_continuous','local_continuous','local_raw_clsm']
    labels=['Old fit / spectral','Continuous fit / spectral','Continuous fit / local','Raw continuous fit / local']
    fig,ax=plt.subplots(figsize=(11,8))
    for off,m,label in zip([-.3,-.1,.1,.3],methods,labels):
        values=[sum(bool(r['cross_valid_front']) for r in rows if r['method']==m and r['system']==n and r['slot']==s) for n,s in keys]
        ax.barh(np.arange(len(keys))+off,values,height=.18,label=label)
    ax.set_yticks(range(len(keys)),[f'{n} [{s}]' for n,s in keys]);ax.invert_yaxis();ax.set_xticks([0,1,2,3])
    ax.set_xlim(0,3.15);ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
    ax.set_xlabel('Seeds retaining a registered equation / 3');ax.set_title('Core 15: local weak evaluation and raw fitting ablation')
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.09),ncol=2,fontsize=9)
    fig.tight_layout();fig.savefig(L.ROOT/'structural_results.png',dpi=180);fig.savefig(L.ROOT/'structural_results.svg');plt.close(fig)
    R.dump(L.ROOT/'audit.json',dict(cases=45,equation_seed_trials=60,source_and_input_hashes_verified=True,
        finite_scores=True,baselines_reproduced=True,max_frozen_score_scaled_error=max(d['frozen_score_max_scaled_error'] for d in data),
        calculation_seconds=sum(d['elapsed_seconds'] for d in data),
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in L.ROOT.glob('*.py')}))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
