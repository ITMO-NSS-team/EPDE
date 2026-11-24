"""Post-calibration diagnostics; never changes fits, cutoffs, or coverage."""
from bridge import *
from scipy.linalg import eigh
import csv


def main():
    files = sorted((ROOT / 'results').glob('*.json'))
    assert len(files) == 69
    failures, reference_failures, margins = [], [], []
    for path in files:
        d = read(path)
        meta = {k: d[k] for k in ('system', 'profile', 'seed')}
        with np.load(path.with_suffix('.npz')) as z:
            for mode in ('fixed_pilot', 'refit_pilot'):
                for bank in ('local', 'broad', 'both'):
                    actual = max(d['actual_scores'].values()) if bank == 'both' else d['actual_scores'][bank]
                    cut = d['joint'][mode]['cutoff'] if bank == 'both' else d['cuts'][mode][bank]
                    boot = np.maximum(z[mode + '_local_scores'], z[mode + '_broad_scores']) if bank == 'both' else z[mode + '_' + bank + '_scores']
                    count = int(np.sum(boot >= actual))
                    row = dict(**meta, mode=mode, bank=bank, covered=d['coverage'][mode][bank],
                               observed_over_cutoff=actual / cut, bootstrap_exceedances=count,
                               diagnostic_tail_rank=(count + 1) / (len(boot) + 1))
                    margins.append(row)
                    if row['covered'] or bank == 'both':
                        continue
                    idx = d['active']
                    cov = z[bank + '_anchor_covariance'][np.ix_(idx, idx)]
                    gram = z[bank + '_error_gram'][np.ix_(idx, idx)]
                    scale = np.sqrt(np.diag(cov))
                    cc, gg = cov / scale[:, None] / scale[None, :], gram / scale[:, None] / scale[None, :]
                    values, vectors = eigh((gg + gg.T) / 2, (cc + cc.T) / 2)
                    np.testing.assert_allclose(values[-1], actual, rtol=3e-7)
                    direction = vectors[:, -1]
                    direction /= np.linalg.norm(direction)
                    top = np.argsort(abs(direction))[::-1][:5]
                    failures.append(dict(**row, leading_standardized_coordinates=[
                        dict(term=d['labels'][idx[i]], weight=float(direction[i])) for i in top]))
            for r in d['references']:
                if r['noise_covered'] and r['zero_accepted']:
                    continue
                c = next(c for c in d['candidates'] if c['id'] == r['candidate'])
                v = S.E.vector(c, c['exact'], len(d['labels']))
                bank = r['bank']
                clean, observed = z[bank + '_clean'] @ v, z[bank + '_observed'] @ v
                error = observed - clean
                radius2 = d['cuts'][r['mode']][bank] * float(v @ z[bank + '_anchor_covariance'] @ v)
                reference_failures.append(dict(**meta, **r, squared_radius=radius2,
                    clean_residual_squared=float(clean @ clean), noise_squared=float(error @ error),
                    observed_residual_squared=float(observed @ observed),
                    cross_term=float(2 * clean @ error)))
    out = dict(status='COMPLETE', datasets=len(files), failures=failures,
               reference_failures=reference_failures,
               interpretation='Post hoc diagnostics. Tail ranks are not certified p-values. '
               'Coordinates use marginal-covariance scaling and can mix correlated terms. '
               'Separate the clean weak-equation residual from noise and their cross term.',
               source_sha256=sources('failure_geometry.py'),
               input_sha256={str(p): sha(p) for p in files})
    dump(ROOT / 'failure_geometry.json', out)
    with (ROOT / 'coverage_margins.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, list(margins[0])); writer.writeheader(); writer.writerows(margins)
    print('COMPLETE', len(failures), 'bank failures;', len(reference_failures), 'reference failures', flush=True)


if __name__ == '__main__':
    main()
