"""Score fixed diffusion and reaction supports against the frozen search fronts."""
from concurrent.futures import ProcessPoolExecutor
import csv
import gzip
import json

from operator_trial import PARTS, ROOT, SEEDS, SYSTEM, folder, measurements
from validate import make_banks, np
from search import U
from epde.interface.search_config import load_search_config, set_active_config
from epde.operators.common.objectives import LOSS_NAN_VAL
from epde.operators.common.subset_selection import subset_table
from epde.operators.common.survival import dyad_scores


NAMES = ['1', 'du/dx0{power: 1.0}', 'u{power: 1.0} * d^2u/dx1^2{power: 1.0}',
         'du/dx1{power: 2.0}', 'u{power: 2.0}']


def evaluate(job):
    seed, part = job
    dest = folder('pvpa', part)
    record = json.loads((dest/f'{SYSTEM}_{seed}_wape.json').read_text())
    settings = record['settings']
    # The native dyad partitions use the maximum width of this search space.
    config = load_search_config(overrides={k: settings[k] for k in
                                ('system_rps_mode', 'system_rps_params', 'equation_terms_max_number',
                                 'discrepancy_metric', 'instability_metric', 'evaluation_form')})
    set_active_config(config)
    fronts = json.loads((dest/f'pair_fronts_{seed}.json').read_text())
    with gzip.open(dest/f'archive_{seed}.json.gz', 'rt') as stream:
        archive = json.load(stream)
    fields = measurements(SYSTEM, seed)
    clean_axes, clean_fields = U.load(SYSTEM)
    def matrix(half):
        axes, raw = fields[half]
        return make_banks(axes, raw['u'], NAMES, record['geometry'])(raw['u'])[0]
    M = matrix(part)
    V = matrix('late' if part == 'early' else 'early')
    axes, _ = fields[part]
    clean_indices = np.searchsorted(clean_axes[0], axes[0])
    np.testing.assert_array_equal(clean_axes[0][clean_indices], axes[0])
    clean_field = clean_fields['u'][clean_indices]
    clean = make_banks(axes, clean_field, NAMES, record['geometry'])(clean_field)[0]
    y = M[:, 1]
    weights = np.ones(len(y))
    shape = tuple(record['geometry']['native']['shape'])
    rows = []
    for label, indices in [('diffusion', [2, 3]), ('reaction', [4])]:
        X = M[:, indices]
        _, amp, _, coef = subset_table(X, y, weights, discrepancy='wape', subsets=[2**len(indices)-1],
                                      return_coefs=True)
        cap = config.search_space.rps_amplification_cap
        assert cap is None or amp[0] <= cap
        coeffs = coef[0][:-1]
        assert coef[0][-1] == 0
        dyad = float(dyad_scores(X, y, weights, shape, fit_intercept=False).sum())
        def discrepancy(bank):
            return float(np.mean(np.abs(bank[:, indices] @ coeffs - bank[:, 1]))/np.mean(np.abs(bank[:, 1])))
        metrics = dict(source=discrepancy(M), held=discrepancy(V))
        # Verify the independently reconstructed objectives on scored versions, allowing
        # the two orders in which EPDE can store a product's factors.
        def terms(name):
            return tuple(sorted(name.split(' * ')))
        expected = {terms(NAMES[j]) for j in [1, *indices]}
        names = [terms(name) for name in archive['labels']]
        equivalents = {i for i, r in enumerate(archive['rows']) if names[r['pivot']] == terms(NAMES[1])
                       and {names[j] for j in r['ids']} == expected and r['dyad'] < LOSS_NAN_VAL}
        for i in equivalents:
            r = archive['rows'][i]
            np.testing.assert_allclose([metrics['source'], dyad], [r['wape'], r['dyad']], rtol=1e-7, atol=1e-11)
        row = dict(seed=seed, part=part, model=label, fit_coefficients=json.dumps(coeffs.tolist()),
                   amplification=float(amp[0]), dyad=dyad, source_wape=metrics['source'], held_wape=metrics['held'],
                   scored_variants=len(equivalents))
        clean_X, clean_y = clean[:, indices], clean[:, 1]
        clean_coef = np.linalg.lstsq(clean_X, clean_y, rcond=None)[0]
        def relative_residual(X, target, c):
            return float(np.mean(np.abs(X @ c - target))/np.mean(np.abs(target)))
        row['clean_refit_wape'] = relative_residual(clean_X, clean_y, clean_coef)
        row['clean_refit_dyad'] = float(dyad_scores(clean_X, clean_y, weights, shape, fit_intercept=False).sum())
        row['clean_wape_with_noisy_fit'] = relative_residual(clean_X, clean_y, coeffs)
        row['noisy_wape_with_known_coefficients'] = (relative_residual(X, y, np.ones(len(indices)))
                                                     if label == 'diffusion' else None)
        row['clean_wape_with_known_coefficients'] = (relative_residual(clean_X, clean_y, np.ones(len(indices)))
                                                     if label == 'diffusion' else None)
        for front, score in metrics.items():
            points = fronts[front]
            # Any dominating archived point is itself dominated by (or is) a front point.
            dominators = [h for h in points if h['archive_id'] not in equivalents
                          and h['wape'] <= score and h['dyad'] <= dyad
                          and (h['wape'] < score or h['dyad'] < dyad)]
            row[front+'_nondominated'] = int(score <= 1. and np.isfinite(dyad) and dyad < LOSS_NAN_VAL and not dominators)
            row[front+'_dominators'] = json.dumps([h['archive_id'] for h in dominators])
            for h in points:
                if h['archive_id'] in equivalents:
                    np.testing.assert_allclose(h['wape'], score, rtol=1e-7, atol=1e-11)
                    assert row[front+'_nondominated']
        rows.append(row)
    return rows


if __name__ == '__main__':
    with ProcessPoolExecutor(max_workers=3) as pool:
        rows = [r for group in pool.map(evaluate, [(s, p) for s in SEEDS for p in PARTS]) for r in group]
    with (ROOT/'reference_front.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
