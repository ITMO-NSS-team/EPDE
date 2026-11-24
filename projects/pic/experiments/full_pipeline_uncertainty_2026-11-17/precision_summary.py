"""Post-calibration coefficient widths and directly checked zero witnesses.

A zero inside an outer interval alone proves no joint feasibility. A recorded
zero witness must satisfy both original quadratic inequalities directly.
"""
from bridge import *
from collections import Counter
import csv


def main():
    files = sorted((ROOT / 'results').glob('*.json'))
    assert len(files) == 69
    rows, recession = [], []
    for path in files:
        d = read(path)
        meta = {k: d[k] for k in ('system', 'profile', 'seed')}
        with np.load(path.with_suffix('.npz')) as z:
            grams = {b: S.gram(z[b + '_observed']) for b in ('local', 'broad')}
            covs = {b: z[b + '_anchor_covariance'] for b in grams}
            cut = d['joint']['refit_pilot']['cutoff']
            for c in d['candidates']:
                if c['label'] != 'reference':
                    continue
                cert = next(r for r in d['joint']['refit_pilot']['certificates'] if r['candidate'] == c['id'])
                if cert['geometry'] != 'bounded_outer':
                    rows.append(dict(**meta, candidate=c['id'], geometry=cert['geometry']))
                    if cert['geometry'] == 'unbounded_certified':
                        direction=np.array(cert['common_negative_direction']); v=np.zeros(len(d['labels'])); v[c['xi']]=direction
                        banks={}
                        for bank in grams:
                            signal=float(v@grams[bank]@v); error=float(cut*(v@covs[bank]@v))
                            assert signal-error<0
                            clean=S.gram(z[bank+'_clean']); denominator=sum(abs(direction[j])*np.sqrt(max(0.,clean[k,k])) for j,k in enumerate(c['xi']))**2
                            banks[bank]=dict(observed_curvature=signal,error_allowance=error,
                                             curvature_ratio=signal/error,
                                             clean_cancellation_ratio=float(v@clean@v)/denominator if denominator else None)
                        direction=direction/np.max(abs(direction))
                        recession.append(dict(**meta,candidate=c['id'],banks=banks,
                                              normalized_direction=[dict(term=d['labels'][k],weight=float(direction[j])) for j,k in enumerate(c['xi'])]))
                    continue
                region = cert['outer_region']
                for j, truth in enumerate(c['exact']):
                    lo, hi = region['lower'][j], region['upper'][j]
                    proposals = [('reference', c['exact']), ('joint_witness', cert.get('feasible_witness')),
                                 ('outer_center', region.get('center'))]
                    witness = None
                    for origin, coefficients in proposals:
                        if coefficients is None:
                            continue
                        coefficients = np.array(coefficients, dtype=float).copy(); coefficients[j] = 0.
                        v = S.E.vector(c, coefficients, len(d['labels']))
                        margins = {b: float(v @ (grams[b] - cut * covs[b]) @ v) for b in grams}
                        if all(x <= 0. for x in margins.values()):
                            witness = dict(origin=origin, coefficients=coefficients.tolist(), margins=margins)
                            break
                    rows.append(dict(**meta, candidate=c['id'], geometry=cert['geometry'],
                                     coefficient=j, term=d['labels'][c['xi'][j]], truth=truth,
                                     lower=lo, upper=hi, full_width=hi-lo,
                                     width_over_absolute_truth=(hi-lo)/abs(truth) if truth else None,
                                     outer_contains_zero=lo <= 0 <= hi,
                                     zero_witness_found=witness is not None, zero_witness=witness))
    bounded = [r for r in rows if r['geometry'] == 'bounded_outer']
    nonzero = [r for r in bounded if r['truth'] != 0]
    summaries = []
    for name, profile in JOBS:
        rr = [r for r in nonzero if (r['system'], r['profile']) == (name, profile)]
        summaries.append(dict(system=name, profile=profile, nonzero_coefficients=len(rr),
                              zero_witnesses=sum(r['zero_witness_found'] for r in rr),
                              relative_full_width_quantiles=np.quantile([r['width_over_absolute_truth'] for r in rr], [0, .5, 1]).tolist() if rr else []))
    out = dict(status='COMPLETE', datasets=len(files), coordinate_records=len(bounded),
               nonzero_truth_coordinates=len(nonzero), zero_witnesses=sum(r['zero_witness_found'] for r in nonzero),
               summaries=summaries, rows=rows, recession_diagnostics=recession,
               interpretation='Widths belong to certified outer regions. Zero feasibility is claimed only '
               'for an explicit vector satisfying both original joint inequalities. Failure to find a '
               'witness among three proposals is not proof of exclusion. References supply diagnostic '
               'test vectors only after calibration. Coefficients and equations within a dataset are '
               'not independent statistical trials.',
               source_sha256=sources('precision_summary.py'),
               input_sha256={str(p): sha(p) for p in files})
    dump(ROOT / 'precision_summary.json', out)
    flat = [{k: v for k, v in r.items() if k != 'zero_witness'} for r in rows]
    with (ROOT / 'coefficient_precision.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, list(dict.fromkeys(k for r in flat for k in r)))
        writer.writeheader(); writer.writerows(flat)
    print('COMPLETE', len(nonzero), 'nonzero coordinates;', out['zero_witnesses'], 'verified zero witnesses', flush=True)


if __name__ == '__main__':
    main()
