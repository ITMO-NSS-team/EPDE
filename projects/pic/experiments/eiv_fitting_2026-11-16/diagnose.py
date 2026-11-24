"""Coefficient and fixed-point diagnostics for the frozen fresh-seed study."""
import estimator as E
import json, csv
from collections import Counter
np, ROOT = E.np, E.ROOT


def write_csv(name, rows):
    with (ROOT/name).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader(); writer.writerows(rows)


def main():
    coefficients, incomplete = [], []
    for path in sorted((ROOT/'results').glob('*.json')):
        d = json.loads(path.read_text())
        ref = next(c for c in d['candidates'] if c['label']=='reference')
        for fold in d['folds']:
            for row in fold['rows']:
                if row['candidate'] == ref['id'] and row['coefficients'] is not None:
                    for index, (truth, estimate) in enumerate(zip(ref['exact'], row['coefficients'])):
                        coefficients.append(dict(system=d['system'], seed=d['seed'], training=fold['training'],
                                                 bank=row['bank'], method=row['method'], index=index,
                                                 exact=truth, estimate=estimate, error=estimate-truth,
                                                 relative_error=(estimate-truth)/abs(truth) if truth else None))
                if row['method'] == 'raw_irls' and row['fit']['status'] != 'ok':
                    trace = row['fit']['trace']
                    rr = dict(system=d['system'], seed=d['seed'], training=fold['training'],
                              bank=row['bank'], candidate=row['candidate'], status=row['fit']['status'],
                              iterations=row['fit']['iterations'])
                    if len(trace) >= 3 and all('coefficients' in step for step in trace[-3:]):
                        tail = np.array([step['coefficients'] for step in trace[-3:]])
                        one, two = np.linalg.norm(tail[-1]-tail[-2]), np.linalg.norm(tail[-1]-tail[-3])
                        rr.update(last_prediction_change=trace[-1]['prediction_change'],
                                  last_two_step_change_over_one_step=float(two/one) if one else None,
                                  last_coefficients=json.dumps(trace[-1]['coefficients']),
                                  previous_coefficients=json.dumps(trace[-2]['coefficients']))
                    incomplete.append(rr)
    write_csv('coefficient_errors.csv', coefficients)
    if incomplete:
        write_csv('nonconvergence.csv', incomplete)
    groups = []
    for name in ('ac','kdv_cossin','wave','ks'):
        for bank in ('local_dense','local','broad'):
            cc = [r for r in coefficients if r['system']==name and r['bank']==bank and r['method']=='raw_irls']
            for index in sorted({r['index'] for r in cc}):
                rr = [r for r in cc if r['index']==index]
                groups.append(dict(system=name, bank=bank, index=index, fitted_folds=len(rr),
                                   exact=rr[0]['exact'], median=float(np.median([r['estimate'] for r in rr])),
                                   minimum=min(r['estimate'] for r in rr), maximum=max(r['estimate'] for r in rr)))
    write_csv('coefficient_summary.csv', groups)
    E.R.dump(ROOT/'diagnosis.json', dict(nonconverged=incomplete, coefficients=groups,
             note='Coefficient summaries include only successful individual folds; fitted_folds is explicit. '
                  'They are diagnostics, not complete-case accuracy summaries.'))
    print('Incomplete fits:',Counter((r['system'],r['bank']) for r in incomplete))
    for row in groups:
        if row['bank']=='broad': print(row)


if __name__=='__main__': main()
