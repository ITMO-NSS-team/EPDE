"""Read-only analysis of frozen fits and paired conditional score errors."""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
from pathlib import Path
import csv
import json
import numpy as np

ROOT = Path(__file__).resolve().parent
ARM = 'q0_white128'


def write_csv(name, rows):
    with (ROOT / name).open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    retention = list(csv.DictReader((ROOT / 'core_retention.csv').open()))
    rankings = list(csv.DictReader((ROOT / 'core_ranking.csv').open()))
    pairs, closest, fitted = [], [], []
    for path in sorted((ROOT / 'results').glob('*_seed*.json')):
        data = json.loads(path.read_text())
        candidates, dyad = data['candidates'], data['instability']
        rows = data['rows'][ARM]
        by_candidate = {r['candidate']: i for i, r in enumerate(rows) if r['kind'] == 'joint'}
        with np.load(path.with_name(path.stem + '_samples.npz')) as samples:
            for ref_index, ref in enumerate(candidates):
                if ref['label'] != 'reference':
                    continue
                ri = by_candidate[ref['id']]
                comparisons = []
                for j, candidate in enumerate(candidates):
                    if candidate['slot'] != ref['slot'] or candidate['registered']:
                        continue
                    ci = by_candidate[candidate['id']]
                    gap = rows[ci]['cross'] - rows[ri]['cross']
                    sd = {n: float(np.std(samples[f'{ARM}_{n}'][:, ci] -
                                         samples[f'{ARM}_{n}'][:, ri], ddof=1))
                          for n in (64, 128)}
                    eligible = dyad[j] <= dyad[ref_index]
                    row = dict(system=data['system'], seed=data['seed'], slot=ref['slot'],
                               reference=ref['id'], candidate=candidate['id'],
                               dyad_eligible=eligible, reference_dyad=dyad[ref_index],
                               candidate_dyad=dyad[j], gap=gap, paired_sd64=sd[64],
                               paired_sd128=sd[128], gap_over_sd=gap / sd[128] if sd[128] else None,
                               clean_full_gap=rows[ci]['clean_full'] - rows[ri]['clean_full'],
                               clean_subsample_gap=rows[ci]['clean_subsample'] - rows[ri]['clean_subsample'],
                               reference_frozen_energy=rows[ri]['frozen_clean_energy'],
                               candidate_frozen_energy=rows[ci]['frozen_clean_energy'])
                    pairs.append(row)
                    if eligible:
                        comparisons.append(row)
                if comparisons:
                    closest.append(min(comparisons, key=lambda r: r['gap']))
                if data['system'] in ('wave', 'ks', 'ac', 'kdv_cossin'):
                    fitted.append(dict(system=data['system'], seed=data['seed'], slot=ref['slot'],
                                       exact=ref['exact'], terms=ref['terms'],
                                       coefficients=[next(r['coefficients'] for r in f['arms'][ARM]
                                                          if r['id'] == 'joint/' + ref['id'])
                                                     for f in data['folds']]))
    write_csv('core_primary_margins.csv', pairs)
    write_csv('core_closest_eligible_competitor.csv', closest)
    changes = {}
    for arm, fit in sorted({(r['arm'], r['fit']) for r in retention}):
        selected = [r for r in retention if r['arm'] == arm and r['fit'] == fit]
        case = lambda r: dict(system=r['system'], seed=int(r['seed']), slot=int(r['slot']))
        changes[arm + '/' + fit] = dict(
            failures=[case(r) for r in selected if r['cross_retained'] == 'False'],
            gains=[case(r) for r in selected if r['cross_retained'] == 'True' and r['old_local_retained'] == 'False'],
            losses=[case(r) for r in selected if r['cross_retained'] == 'False' and r['old_local_retained'] == 'True'])
    choices = [r for r in rankings if r['arm'] == ARM]
    accuracy = dict(improved=sum(float(r['accuracy_ratio']) < 1 - 1e-8 for r in choices),
                    worsened=sum(float(r['accuracy_ratio']) > 1 + 1e-8 for r in choices),
                    unchanged=sum(abs(float(r['accuracy_ratio']) - 1) <= 1e-8 for r in choices))
    out = dict(primary_arm=ARM, fitting='joint', structural_changes=changes,
               diagnostic_option_accuracy_vs_baseline=accuracy,
               targeted_coefficients=fitted,
               margin_definition='Competitor minus reference score; positive favors reference. '
                                 'Dyad-eligible unregistered candidates only for closest comparisons. '
                                 'These reference-pair margins do not replace equivalent-aware retention.',
               uncertainty_limit='Conditional paired SD with fixed coefficients, weights, geometry and noise estimates; '
                                 'not calibrated confidence, a significance test, or fresh-seed replication.',
               ranking_limit='Lowest validation score among saved reference fits is a ranking diagnostic. '
                             'Using it for selection requires a separate final evaluation split.')
    (ROOT / 'margin_summary.json').write_text(json.dumps(out, indent=2) + '\n', encoding='utf-8')
    print('Diagnostic saved-option accuracy:', accuracy)
    print('Primary changes:', changes[ARM + '/joint'])
    for row in closest:
        if (row['system'], row['seed']) in [('ks', 0), ('kdv_cossin', 2), ('ac', 0),
                                            ('ac', 2), ('kdv_cossin', 0), ('kdv_cossin', 1)]:
            print(row['system'], row['seed'], row['candidate'],
                  'gap', row['gap'], 'SD128', row['paired_sd128'],
                  'SD64', row['paired_sd64'], 'clean gap', row['clean_full_gap'])
    plot(rankings)


def plot(rankings):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    summary = json.loads((ROOT / 'core_summary.json').read_text())
    arms = ['q0_plain', 'q0_white64', 'q0_white128', 'q1_plain', 'q1_white64', 'q1_white128']
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), gridspec_kw={'width_ratios': [1.45, 1]})
    x = np.arange(len(arms))
    for dx, kind, color, label in [(-.19, 'baseline', '#64748b', 'Frozen local fits'),
                                  (.19, 'joint', '#087e8b', 'Frozen joint-risk fits')]:
        values = [summary[arm][kind]['retained'] for arm in arms]
        bars = axes[0].bar(x + dx, values, width=.36, color=color, label=label)
        axes[0].bar_label(bars, padding=3, fontsize=9)
    axes[0].axhline(54, color='#bb4d22', ls='--', lw=1.3, label='Previous local validation: 54')
    axes[0].set_xticks(x, ['q0\nPlain', 'q0\nWhite 64', 'q0\nWhite 128',
                         'q1\nPlain', 'q1\nWhite 64', 'q1\nWhite 128'])
    axes[0].set_ylim(0, 62)
    axes[0].set_ylabel('Registered structure retained / 60 cases')
    axes[0].set_title('15 systems, three reused noise seeds')
    axes[0].legend(loc='lower left', frameon=False, fontsize=9)
    for i, (name, label) in enumerate([('wave', 'Wave'), ('ks', 'KS'), ('ac', 'AC'), ('kdv_cossin', 'Forced KdV')]):
        rr = sorted([r for r in rankings if r['arm'] == ARM and r['system'] == name], key=lambda r: int(r['seed']))
        for j, row in enumerate(rr):
            xpos = i + (j - 1) * .17
            baseline, selected = float(row['baseline_clean_energy']), float(row['lowest_score_clean_energy'])
            axes[1].plot([xpos, xpos], [baseline, selected], color='#94a3b8', lw=1.5)
            axes[1].scatter(xpos, baseline, c='#64748b', s=36, marker='o', label='Local fit' if i == j == 0 else None)
            axes[1].scatter(xpos, selected, c='#087e8b', s=48, marker='x', label='Lowest validation score' if i == j == 0 else None)
    axes[1].set_yscale('log')
    axes[1].set_xticks(range(4), ['Wave', 'KS', 'AC', 'Forced\nKdV'])
    axes[1].set_ylabel('Frozen clean residual energy (lower is better)')
    axes[1].set_title('Saved-fit ranking diagnostic, q0 / White 128')
    axes[1].legend(loc='lower right', frameon=False, fontsize=8)
    fig.text(.5, .025, 'Fixed candidate panels. Right: diagnostic choice among six saved fits; separate outer validation is still required.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .07, 1, 1))
    for suffix in ('png', 'svg'):
        fig.savefig(ROOT / ('validation_comparison.' + suffix), dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
