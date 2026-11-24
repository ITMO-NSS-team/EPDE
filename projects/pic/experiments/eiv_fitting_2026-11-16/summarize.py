"""Paired fresh-seed summaries; incomplete and nonconverged fits stay visible."""
import estimator as E
np, ROOT, R = E.np, E.ROOT, E.R
from pathlib import Path
import csv, json, hashlib
from collections import Counter


def write_csv(name, rows):
    with (ROOT/name).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader(); writer.writerows(rows)


def main():
    files = sorted((ROOT/'results').glob('*.json'))
    assert {p.stem for p in files} == {f'{name}_seed{seed}' for name in ('ac','kdv_cossin','wave','ks') for seed in (3,4,5)}
    cases, folds, statuses, traces, floor = [], [], Counter(), [], []
    for path in files:
        data = json.loads(path.read_text())
        ref = next(c for c in data['candidates'] if c['label']=='reference')
        baseline = next(a for a in data['aggregates'] if a['bank']=='local_dense' and a['method']=='recon_clsm' and a['candidate']==ref['id'])
        for a in data['aggregates']:
            row = dict(system=data['system'], seed=data['seed'], **a)
            row['statuses'] = '|'.join(row['statuses'])
            if 'frozen_clean_energy' in row:
                row['baseline_energy'] = baseline['frozen_clean_energy']
                row['energy_ratio_to_baseline'] = row['frozen_clean_energy']/row['baseline_energy']
            cases.append(row)
        for f in data['folds']:
            for row in f['rows']:
                fit = row['fit']; statuses[row['bank'], row['method'], fit['status']] += 1
                rr = {k: v for k,v in row.items() if k not in ('fit','coefficients','coefficient_error')}
                rr.update(system=data['system'], seed=data['seed'], training=f['training'],
                          status=fit['status'], iterations=fit.get('iterations'),
                          coefficients=json.dumps(row['coefficients']),
                          coefficient_error=json.dumps(row.get('coefficient_error')))
                folds.append(rr)
                for step in fit.get('trace',[]):
                    traces.append(dict(system=data['system'], seed=data['seed'], training=f['training'],
                                       bank=row['bank'], candidate=row['candidate'],
                                       **{k:v for k,v in step.items() if k!='coefficients'}))
                if row['method'] in ('clean_fit','interpolated_clean_fit'):
                    floor.append(rr)
    write_csv('per_case.csv', cases); write_csv('per_fold.csv', folds)
    write_csv('iteration_trace.csv', traces); write_csv('clean_fit_floors.csv', floor)
    summaries = []
    for name in ('ac','kdv_cossin','wave','ks'):
        for bank in ('local_dense','local','broad'):
            for method in ('recon_clsm','recon_ols','raw_ols','corrected_raw_ols','raw_irls'):
                rows = [r for r in cases if r['system']==name and r['bank']==bank and r['method']==method and r['candidate']=='e0:reference']
                complete = [r for r in rows if r['successful_folds']==3]
                summary = dict(system=name, bank=bank, method=method,
                               complete_seed_cases=len(complete), successful_folds=sum(r['successful_folds'] for r in rows))
                # A median of successful cases alone is not comparable when some seeds fail.
                if len(complete)==3:
                    summary.update(median_energy=float(np.median([r['frozen_clean_energy'] for r in complete])),
                                   median_baseline_energy=float(np.median([r['baseline_energy'] for r in complete])),
                                   improved_seeds=sum(r['energy_ratio_to_baseline'] < 1-1e-8 for r in complete),
                                   worsened_seeds=sum(r['energy_ratio_to_baseline'] > 1+1e-8 for r in complete))
                summaries.append(summary)
    write_csv('summary.csv', summaries)
    status = [dict(bank=b, method=m, status=s, count=n) for (b,m,s),n in sorted(statuses.items())]
    R.dump(ROOT/'summary.json', dict(system_runs=len(files), reference_seed_cases=12,
           systems=list(('ac','kdv_cossin','wave','ks')), fresh_seeds=[3,4,5],
           accuracy_summary=summaries, convergence=status,
           limitation='Frozen known supports and historical geometries. No candidate selection, structural-retention count, or evolutionary search.'))
    for row in summaries:
        if row['method'] in ('raw_irls','corrected_raw_ols') or (row['bank']=='local_dense' and row['method']=='recon_clsm'):
            print(row)
    plot(summaries)


def plot(rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(13,4.4))
    labels = [('local_dense','recon_clsm','Old local\nCLSM'), ('local_dense','corrected_raw_ols','Dense\nraw OLS'),
              ('local_dense','raw_irls','Dense\nIRLS'), ('local','raw_irls','Spaced\nIRLS'),
              ('broad','corrected_raw_ols','Broad\nraw OLS'), ('broad','raw_irls','Broad\nIRLS')]
    for ax, name, title in zip(axes, ('ac','kdv_cossin','wave','ks'), ('AC','Forced KdV','Wave','KS')):
        for x,(bank,method,label) in enumerate(labels):
            r = next(r for r in rows if r['system']==name and r['bank']==bank and r['method']==method)
            if 'median_energy' in r:
                ax.scatter(x,r['median_energy'],s=55,c='#087e8b' if method=='raw_irls' else '#64748b')
            else:
                ax.text(x,.04,f"{r['successful_folds']}/9\nfits",transform=ax.get_xaxis_transform(),ha='center',fontsize=8,color='#ad542d')
        ax.set_yscale('log');ax.set_title(title);ax.set_xticks(range(len(labels)),[l[2] for l in labels],rotation=55,ha='right',fontsize=8)
        ax.spines[['top','right']].set_visible(False)
        ax.set_xlim(-.5,len(labels)-.5)
    axes[0].set_ylabel('Median frozen clean residual energy')
    fig.suptitle('Known-support coefficient test: fresh 5% noise seeds 3–5',fontsize=13)
    fig.text(.5,.015,'Lower is better. Missing dots: at least one nonconverged/infeasible fold; no median computed over successes only.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.08,1,.94))
    for suffix in ('png','svg'):fig.savefig(ROOT/('accuracy.'+suffix),dpi=180)
    plt.close(fig)


if __name__=='__main__':
    main()
