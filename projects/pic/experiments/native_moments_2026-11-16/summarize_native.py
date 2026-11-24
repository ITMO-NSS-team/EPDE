"""All paired cases, incomplete fits, coefficients, and validation rankings."""
import native_quadrature as N
np,ROOT,R=N.np,N.ROOT,N.R
import csv,json
from collections import Counter


def write(name,rows):
    if not rows:return
    with (ROOT/name).open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader();writer.writerows(rows)


def main():
    files=sorted((ROOT/'results').glob('*.json'))
    assert {p.stem for p in files}=={f'{n}_seed{s}' for n in ('ac','kdv_cossin','wave','ks') for s in (3,4,5)}
    cases=[];folds=[];weights=[];coefs=[];status=Counter();ranking=[]
    for path in files:
        d=json.loads(path.read_text());meta=dict(system=d['system'],seed=d['seed'])
        by_id={c['id']:c for c in d['candidates']}
        for a in d['aggregates']:
            row=dict(meta,**a);row['statuses']='|'.join(row['statuses'])
            baseline=next(x for x in d['aggregates'] if x['bank']=='prior_local_dense' and x['method']=='recon_clsm' and x['candidate']==a['candidate'])
            row['baseline_energy']=baseline['frozen_clean_energy']
            if 'frozen_clean_energy' in a:row['energy_ratio_to_baseline']=a['frozen_clean_energy']/row['baseline_energy']
            cases.append(row)
        for f in d['folds']:
            for tag,w in f['validation_weighting'].items():
                weights.append(dict(meta,training=f['training'],bank=tag,feasible=w['feasible'],rank=w['rank'],rows=w['rows'],
                                    eigen_ratio=w['eigenvalues'][0]/w['eigenvalues'][-1] if w['eigenvalues'][-1]>0 else 0.,
                                    identity_error=w.get('whitened_covariance_max_error')))
            for row in f['rows']:
                status[row['bank'],row['method'],row['fit']['status']]+=1
                rr={k:v for k,v in row.items() if k not in ('fit','coefficients','coefficient_error')}
                rr.update(meta,training=f['training'],status=row['fit']['status'],iterations=row['fit'].get('iterations'),
                          coefficients=json.dumps(row['coefficients']))
                folds.append(rr)
                if row['coefficients'] is not None:
                    for i,(actual,truth) in enumerate(zip(row['coefficients'],by_id[row['candidate']]['exact'])):
                        coefs.append(dict(meta,training=f['training'],bank=row['bank'],method=row['method'],candidate=row['candidate'],
                                          index=i,estimate=actual,exact=truth,error=actual-truth))
        for c in d['candidates']:
            options=[a for a in d['aggregates'] if a['candidate']==c['id'] and a['bank'].startswith(('native_','interpolated_'))
                     and a['method'] in ('ols','irls') and a['successful_folds']==3]
            if not options:continue
            for score in ('old_cross','native_local_white_cross','native_broad_white_cross','native_mixed_white_cross'):
                allowed=[r for r in options if score in r]
                if not allowed:continue
                best=min(allowed,key=lambda r:r[score]);oracle=min(allowed,key=lambda r:r['frozen_clean_energy'])
                baseline=next(a['frozen_clean_energy'] for a in d['aggregates'] if a['bank']=='prior_local_dense' and a['method']=='recon_clsm' and a['candidate']==c['id'])
                ranking.append(dict(meta,candidate=c['id'],score=score,options=len(allowed),
                                    chosen_bank=best['bank'],chosen_method=best['method'],chosen_energy=best['frozen_clean_energy'],
                                    baseline_energy=baseline,oracle_option_energy=oracle['frozen_clean_energy'],
                                    ratio_to_baseline=best['frozen_clean_energy']/baseline,
                                    ratio_to_best_saved=best['frozen_clean_energy']/oracle['frozen_clean_energy']))
    summaries=[]
    for name in ('ac','kdv_cossin','wave','ks'):
        for bank in ('native_local','native_broad','native_mixed','interpolated_local','interpolated_broad','interpolated_mixed','prior_local_dense','prior_broad'):
            methods=('ols','irls') if not bank.startswith('prior') else ('recon_clsm','raw_irls')
            for method in methods:
                rows=[r for r in cases if r['system']==name and r['bank']==bank and r['method']==method and r['candidate']=='e0:reference']
                if not rows:continue
                good=[r for r in rows if r['successful_folds']==3]
                summary=dict(system=name,bank=bank,method=method,successful_folds=sum(r['successful_folds'] for r in rows),complete_cases=len(good))
                if len(good)==3:
                    summary.update(median_energy=float(np.median([r['frozen_clean_energy'] for r in good])),
                                   median_baseline_energy=float(np.median([r['baseline_energy'] for r in good])),
                                   improved_cases=sum(r['energy_ratio_to_baseline']<1-1e-8 for r in good),
                                   worsened_cases=sum(r['energy_ratio_to_baseline']>1+1e-8 for r in good))
                summaries.append(summary)
    for name,rows in [('per_case.csv',cases),('per_fold.csv',folds),('validation_weighting.csv',weights),('coefficient_errors.csv',coefs),
                      ('ranking_diagnostic.csv',ranking),('summary.csv',summaries)]:write(name,rows)
    R.dump(ROOT/'summary.json',dict(system_runs=12,reference_seed_cases=12,paired_reused_seeds=[3,4,5],accuracy=summaries,
           fit_statuses=[dict(bank=b,method=m,status=s,count=n) for (b,m,s),n in sorted(status.items())],
           infeasible_validation_weights=[r for r in weights if not r['feasible']],
           ranking_limitation='Post-hoc diagnostic only; selecting a fit with these scores requires separate outer validation.'))
    for r in summaries:
        if r['method'] in ('irls','raw_irls') or (r['bank']=='prior_local_dense' and r['method']=='recon_clsm'):print(r)
    plot(summaries)


def plot(rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,4,figsize=(13,4.5))
    arms=[('prior_local_dense','recon_clsm','Old local'),('interpolated_broad','irls','Interp. broad'),
          ('native_broad','irls','Native broad'),('native_local','irls','Native local'),
          ('interpolated_mixed','irls','Interp. mixed'),('native_mixed','irls','Native mixed')]
    for ax,name,title in zip(axes,('ac','kdv_cossin','wave','ks'),('AC','Forced KdV','Wave','KS')):
        for x,(bank,method,label) in enumerate(arms):
            row=next(r for r in rows if r['system']==name and r['bank']==bank and r['method']==method)
            if 'median_energy' in row:ax.scatter(x,row['median_energy'],s=50,c='#087e8b' if bank.startswith('native') else '#64748b')
            else:ax.text(x,.05,f"{row['successful_folds']}/9\nfits",transform=ax.get_xaxis_transform(),ha='center',fontsize=8,color='#a34b21')
        ax.set_yscale('log');ax.set_xticks(range(len(arms)),[a[2] for a in arms],rotation=55,ha='right',fontsize=9)
        ax.set_title(title);ax.spines[['right','top']].set_visible(False);ax.set_xlim(-.5,len(arms)-.5)
    axes[0].set_ylabel('Median frozen clean residual energy')
    fig.suptitle('Native quadrature and mixed moments: paired noise seeds 3–5')
    fig.text(.5,.02,'Lower is better. Known supports; no selection by clean data. Missing dots indicate incomplete three-seed results.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.09,1,.94))
    for ext in ('png','svg'):fig.savefig(ROOT/('comparison.'+ext),dpi=180)
    plt.close(fig)


if __name__=='__main__':main()
