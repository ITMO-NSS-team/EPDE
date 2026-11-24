"""Paired discretization controls and descriptive validation-ranking agreement."""
import native_quadrature as N
np,ROOT,R=N.np,N.ROOT,N.R
import json,csv
from itertools import combinations


def write(name,rows):
    with (ROOT/name).open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader();writer.writerows(rows)


def main():
    clean=[];paired=[];agreement=[];failed=[]
    for path in sorted((ROOT/'results').glob('*.json')):
        d=json.loads(path.read_text());meta=dict(system=d['system'],seed=d['seed'])
        rr=[r for r in d['aggregates'] if r['candidate']=='e0:reference']
        for row in rr:
            if row['method'] in ('native_clean','interpolated_clean','full_clean'):
                clean.append(dict(meta,bank=row['bank'],method=row['method'],energy=row.get('frozen_clean_energy'),
                                  successful_folds=row['successful_folds']))
        for bank in ('local','broad','mixed'):
            for method in ('ols','irls'):
                native=next(r for r in rr if r['bank']=='native_'+bank and r['method']==method)
                interp=next(r for r in rr if r['bank']=='interpolated_'+bank and r['method']==method)
                out=dict(meta,bank=bank,method=method,native_successful_folds=native['successful_folds'],
                         interpolated_successful_folds=interp['successful_folds'])
                if native['successful_folds']==interp['successful_folds']==3:
                    out.update(native_energy=native['frozen_clean_energy'],interpolated_energy=interp['frozen_clean_energy'],
                               native_over_interpolated=native['frozen_clean_energy']/interp['frozen_clean_energy'])
                paired.append(out)
        options=[r for r in rr if r['bank'].startswith(('native_','interpolated_')) and r['method'] in ('ols','irls') and r['successful_folds']==3]
        for metric in ('old_cross','native_local_white_cross','native_broad_white_cross','native_mixed_white_cross'):
            pairs=[(a,b) for a,b in combinations(options,2) if metric in a and metric in b]
            count=correct=0
            for a,b in pairs:
                gap=a['frozen_clean_energy']-b['frozen_clean_energy'];score=a[metric]-b[metric]
                if abs(gap)<=1e-12*max(a['frozen_clean_energy'],b['frozen_clean_energy']):continue
                count+=1;correct+=gap*score>0
            agreement.append(dict(meta,metric=metric,pairs=count,agree=int(correct),fraction=correct/count if count else None))
        for f in d['folds']:
            for row in f['rows']:
                if row['bank'].startswith(('native_','interpolated_')) and row['method']=='irls' and row['fit']['status']!='ok':
                    trace=row['fit']['trace'];last=trace[-1]
                    failed.append(dict(meta,training=f['training'],bank=row['bank'],candidate=row['candidate'],
                                       status=row['fit']['status'],iterations=row['fit']['iterations'],
                                       last_prediction_change=last.get('prediction_change'),
                                       covariance_eigen_ratio=last.get('covariance_eigen_ratio')))
    for name,rows in [('clean_controls.csv',clean),('paired_input_comparison.csv',paired),('ranking_agreement.csv',agreement),('failed_fits.csv',failed)]:
        if rows:write(name,rows)
    totals=[]
    for metric in sorted({r['metric'] for r in agreement}):
        rows=[r for r in agreement if r['metric']==metric];count=sum(r['pairs'] for r in rows);agree=sum(r['agree'] for r in rows)
        totals.append(dict(metric=metric,pairs=count,agree=agree,fraction=agree/count if count else None))
    R.dump(ROOT/'diagnosis.json',dict(failed_fits=failed,ranking_agreement=totals,
           limitation='Pairs share records and fitted coefficients; these are descriptive diagnostics, not independent tests or final fit selection.'))
    print('Failed new IRLS fits:',failed)
    print('Descriptive ranking agreement:',totals)
    for name in ('ac','kdv_cossin','wave','ks'):
        print(name,'clean broad controls',[(r['method'],r['energy']) for r in clean if r['system']==name and r['seed']==3 and r['bank']=='broad'])


if __name__=='__main__':main()
