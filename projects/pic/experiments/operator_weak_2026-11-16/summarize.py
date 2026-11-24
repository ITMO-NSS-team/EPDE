import run_operator as E
Q,F,R,A,np=E.Q,E.F,E.R,E.A,E.np
from pathlib import Path
import csv,json,hashlib


def write_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    if R.os.name=='nt':R.ctypes.windll.kernel32.SetPriorityClass(R.ctypes.windll.kernel32.GetCurrentProcess(),0x4000)
    data=[json.loads(p.read_text(encoding='utf-8')) for p in (Q.ROOT/'unit_results').glob('*_seed*.json')]
    assert len(data)==45
    assert {(x['system'],x['seed']) for x in data}=={(n,s) for n in R.datasets.names('core') for s in range(3)}
    hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(E.__file__),Path(Q.__file__)]}
    rows=[];selections=[];exact=[];verification=0.
    for x in data:
        assert x['source_sha256']==hashes
        source,old=F.baseline(x['system'],x['seed'])
        assert x['baseline_sha256']==hashlib.sha256(source.read_bytes()).hexdigest()
        cs=x['candidates'];b=x['draws']
        for method,entry in x['aggregates'].items():
            normalized=method.replace(f'_clsm_{b}', '_clsm').replace(f'_clsm_{b//2}', '_clsm_half')
            fit=method.split('_continuous',1)[-1] if '_continuous' in method else None
            frozen_method='legacy_continuous'+fit if fit is not None else 'legacy_old_clsm'
            for slot in sorted({c['slot'] for c in cs}):
                i=next(i for i,c in enumerate(cs) if c['slot']==slot and c['label']=='reference')
                row=dict(system=x['system'],seed=x['seed'],slot=slot,method=normalized,draws=b,
                    frozen_clean_energy=x['aggregates'][frozen_method]['clean_energy']['scores'][i],
                    selected_clean_energy=entry['clean_energy']['scores'][i])
                for metric in ('cross','ordinary','oracle','uncorrected_cross'):
                    assert np.isfinite(entry[metric]['scores']).all()
                    row[metric+'_reference']=entry[metric]['scores'][i]
                    row[metric+'_valid_front']='|'.join(A.frontier(entry[metric]['scores'],x['instability'],cs,slot))
                    row[metric+'_dominators']='|'.join(entry[metric]['dominance'][slot]['unregistered'])
                rows.append(row)
        for metric in ('cross','ordinary','oracle','uncorrected_cross'):
            a=np.array(x['aggregates']['legacy_old_clsm'][metric]['scores'])
            target=np.array(old['aggregates']['loo'][str(b)][metric]['scores'])
            np.testing.assert_allclose(a,target,rtol=2e-7,atol=1e-9)
            verification=max(verification,float(np.max(np.abs(a-target)/np.maximum(1,np.abs(target)))))
        for fold in x['folds']:
            for label,field in [('risk_energy','risk_selection'),('risk_units','unit_risk_selection')]:
                s=fold[field]
                selections.append(dict(system=x['system'],seed=x['seed'],training=fold['training'],
                    method=label,max_modes=str(s['max_modes']),full_max_modes=str(s['full_max_modes']),
                    retained_rows=s['retained_rows'],full_rows=s['full_rows'],retained_fraction=s['relative_retained']))
        for bank in ('full','risk_energy','risk_units'):
            for i,c in enumerate([c for c in cs if c['label']=='reference']):
                r=[f['arms'][bank+'_exact_control'][i] for f in x['folds']]
                exact.append(dict(system=x['system'],seed=x['seed'],slot=c['slot'],bank=bank,
                    cross=float(np.mean([v['cross'] for v in r])),ordinary=float(np.mean([v['ordinary'] for v in r])),
                    clean_energy=float(np.mean([v['clean_energy'] for v in r]))))
    write_csv(Q.ROOT/'per_equation.csv',rows);write_csv(Q.ROOT/'selected_banks.csv',selections)
    write_csv(Q.ROOT/'exact_coefficient_controls.csv',exact)
    summary={}
    for method in sorted({r['method'] for r in rows}):
        rs=[r for r in rows if r['method']==method];assert len(rs)==60
        summary[method]={m:sum(bool(r[m+'_valid_front']) for r in rs) for m in ('cross','ordinary','oracle','uncorrected_cross')}
    assert summary['legacy_old_clsm']['cross']==50
    R.dump(Q.ROOT/'summary.json',summary)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ordered=[(n,s) for n in R.datasets.names('core') for s in sorted({r['slot'] for r in rows if r['system']==n})]
    methods=['legacy_old_clsm','legacy_continuous_clsm','risk_energy_continuous_clsm','risk_units_continuous_clsm']
    colors=['#888888','#087caf','#db8a2f','#408a49']
    labels=['Previous fit/score','Continuous fit; previous score','Continuous + energy risk','Continuous + operator-unit risk']
    fig,ax=plt.subplots(figsize=(11,8))
    for offset,method,color,label in zip([-.3,-.1,.1,.3],methods,colors,labels):
        values=[sum(bool(r['cross_valid_front']) for r in rows if r['system']==n and r['slot']==s and r['method']==method)
                for n,s in ordered]
        ax.barh(np.arange(len(ordered))+offset,values,height=.18,color=color,label=label)
    ax.set_yticks(np.arange(len(ordered)),[f'{n} [{s}]' for n,s in ordered]);ax.invert_yaxis()
    ax.set_xticks([0,1,2,3]);ax.set_xlim(0,3.15);ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
    ax.set_xlabel('Noise seeds retaining a catalog-valid equation / 3')
    ax.set_title('Core 15: continuous weak fitting and operator-risk evaluation\n5% noise, fixed candidate panel and dyad axis')
    ax.legend(loc='upper center',bbox_to_anchor=(.5,-.09),ncol=2,fontsize=9)
    fig.tight_layout();fig.savefig(Q.ROOT/'structural_results.png',dpi=180);fig.savefig(Q.ROOT/'structural_results.svg');plt.close(fig)
    fig,axs=plt.subplots(1,3,figsize=(12,3.8))
    for ax,name in zip(axs,['kdv_cossin','pde_compound','ks']):
        for offset,bank,color in zip([-.12,0,.12],['full','risk_energy','risk_units'],['#888888','#db8a2f','#408a49']):
            rr=sorted([r for r in exact if r['system']==name],key=lambda r:r['seed'])
            rs=[r for r in rr if r['bank']==bank]
            ax.scatter([r['seed']+offset for r in rs],[r['cross'] for r in rs],color=color,label=bank,s=40)
        ax.axhline(0,color='k',linewidth=.6);ax.set_xticks([0,1,2]);ax.set_xlabel('Noise seed');ax.set_title(name);ax.grid(alpha=.2)
    axs[0].set_ylabel('Cross-product at EXACT coefficients')
    handles,lab=axs[0].get_legend_handles_labels();fig.legend(handles,lab,loc='lower center',ncol=3)
    fig.tight_layout(rect=(0,.08,1,1));fig.savefig(Q.ROOT/'exact_score_controls.png',dpi=180)
    fig.savefig(Q.ROOT/'exact_score_controls.svg');plt.close(fig)
    R.dump(Q.ROOT/'audit.json',dict(cases=45,equation_trials=60,baseline_max_scaled_error=verification,
        source_hashes_match=True,finite_scores=True,calculation_seconds=sum(x['elapsed_seconds'] for x in data),
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in Q.ROOT.glob('*.py')}))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
