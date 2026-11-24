"""Plot and summarize the collision fronts using validation-only model choices."""
import csv
from collections import Counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from physics import NAMES


def rms(rows, key):
    return float(np.sqrt(np.mean([r[key]**2 for r in rows])))


def common_structures(stage, fronts, chosen, tags):
    """Compare the exact intersections and the terms shared by the validation winners."""
    columns = ['form', 'heldout', 'mask', 'terms', 'heldout_r2', 'heldout_rmse', 'heldout_pair_normal_rmse',
               'refit_heldout_r2', 'refit_heldout_rmse', 'refit_heldout_pair_normal_rmse']
    archive = pd.read_csv(stage/'all_candidates.csv.gz', usecols=columns)
    rows = []
    for form in ('strong', 'weak'):
        winners = [r['mask'] for r in chosen if r['form'] == form]
        core = int(np.bitwise_and.reduce(winners))
        counts = []
        for name in ('native_pareto', 'validation_pareto'):
            counts.append(Counter(r['mask'] for r in fronts[name] if r['form'] == form))
        masks = {m for tally in counts for m, n in tally.items() if n == len(tags)} | {0}
        if core:
            masks.add(core)
        data = archive.loc[(archive.form == form) & archive['mask'].isin(masks)]
        for mask, block in data.groupby('mask'):
            b = block.set_index('heldout').sort_index()
            assert list(b.index) == tags
            row = dict(form=form, mask=mask, terms=b.terms.iloc[0], native_front_folds=counts[0][mask],
                       validation_front_folds=counts[1][mask], exact_winner_core=bool(mask and mask == core))
            for source, prefix in (('', 'screen_'), ('refit_', 'refit_')):
                r2 = b[source+'heldout_r2']
                row[prefix+'mean_r2'] = r2.mean()
                row[prefix+'worst_r2'] = r2.min()
                for metric in ('heldout_rmse', 'heldout_pair_normal_rmse'):
                    row[prefix+metric] = np.sqrt((b[source+metric]**2).mean())
                row.update({prefix+'r2_'+tag: r2[tag] for tag in tags})
            rows.append(row)
    pd.DataFrame(rows).to_csv(stage/'common_structures.csv', index=False)


def plot_merged(stage, fronts):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3), constrained_layout=True)
    scores = [r['mean_heldout_r2'] for rows in fronts.values() for r in rows]
    for ax, form, title in zip(axes, ('strong', 'weak'), ('Forward increments', 'Weak form')):
        rows = fronts[form]
        x = [r['mean_screen_wape'] for r in rows]
        y = [r['mean_screen_dyad'] for r in rows]
        ax.plot(x, y, color='0.65', linewidth=1, zorder=1)
        points = ax.scatter(x, y, c=[r['mean_heldout_r2'] for r in rows], cmap='viridis',
                            vmin=min(scores), vmax=max(scores), edgecolors='0.18', linewidths=.4, s=32)
        ax.set(title=title, xlabel='Mean training WAPE', ylabel='Mean coefficient instability', yscale='log')
        ax.grid(alpha=.18)
    fig.colorbar(points, ax=axes, label='Mean held-out R²')
    fig.savefig(stage/'merged_fronts.png', dpi=180)
    plt.close(fig)


def report(stage, fronts):
    native = fronts['native_pareto']
    physical = fronts['validation_pareto']
    tags = sorted({r['heldout'] for r in native})
    chosen, budgets = [], []
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for col, form in enumerate(('strong', 'weak')):
        for tag in tags:
            front = sorted((r for r in native if r['form'] == form and r['heldout'] == tag),
                           key=lambda r: r['screen_wape'])
            line, = axes[0,col].plot([r['screen_wape'] for r in front], [r['screen_dyad'] for r in front],
                                     marker='.', linewidth=1.2, label=tag)
            best = min(front, key=lambda r: (r['validation_rmse'], r['size'], r['mask']))
            chosen.append(best)
            axes[0,col].scatter(best['screen_wape'], best['screen_dyad'], s=45, marker='D',
                                color=line.get_color(), edgecolors='black', linewidth=.45, zorder=4)
        for budget in range(len(NAMES) + 1):
            selected = [min((r for r in physical if r['form'] == form and r['heldout'] == tag and r['size'] <= budget),
                            key=lambda r: (r['validation_rmse'], r['size'], r['mask'])) for tag in tags]
            mode, count = Counter(r['terms'] for r in selected).most_common(1)[0]
            budgets.append(dict(form=form, max_terms=budget, validation_rmse=rms(selected, 'validation_rmse'),
                                heldout_rmse=rms(selected, 'heldout_rmse'),
                                heldout_pair_normal_rmse=rms(selected, 'heldout_pair_normal_rmse'),
                                mean_heldout_r2=float(np.mean([r['heldout_r2'] for r in selected])),
                                most_common_structure=mode, folds_selecting_structure=count,
                                selected_sizes=','.join(str(r['size']) for r in selected)))
        curve = [r for r in budgets if r['form'] == form]
        for key, label in [('validation_rmse', 'Validation'), ('heldout_rmse', 'Held-out recording')]:
            axes[1,col].plot([r['max_terms'] for r in curve], [r[key] for r in curve], '.-', label=label)
        axes[0,col].set(title=('Forward increments' if form == 'strong' else 'Weak form'),
                        xlabel='Training-screen WAPE', ylabel='Dyadic coefficient instability', yscale='log')
        axes[1,col].set(xlabel='Maximum contact terms in refit', ylabel='Velocity RMSE (px/s)',
                        xticks=np.arange(0, len(NAMES)+1, 2))
        for ax in axes[:,col]:
            ax.grid(alpha=.2)
    axes[0,1].legend(fontsize=8)
    axes[1,1].legend()
    fig.savefig(stage/'fronts.png', dpi=175)
    plt.close(fig)
    with (stage/'term_budgets.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=budgets[0])
        writer.writeheader()
        writer.writerows(budgets)
    common_structures(stage, fronts, chosen, tags)

    lines = ['# Pareto fronts for the robot collision law', '',
             f'The search covers all {len(NAMES)} original contact terms, every subset of those terms, and both native intercept choices. '
             'The target is the contact contribution to the 0.4 s velocity after subtracting the frozen self-motion and wall model. '
             'All test errors use observed forward velocities on the omitted recording, including for weak-form equations.', '',
             'The across-recording fronts average the two training-screen objectives for each common term set over all five folds, '
             'retain structures eligible in all five and recompute nondominance. Each recording has equal weight. The tables give mean and per-recording held-out R², '
             'recording-balanced RMSE in px/s and a separate equation refitted on all five recordings. The cross-validation scores use the original fold-specific coefficients. '
             '[Merged forward front](merged_strong.csv), [merged weak front](merged_weak.csv) and [merged front plot](merged_fronts.png).', '',
             'For transfer, the common-structure table gives exact front membership in all five folds and the shared terms of the five validation winners. '
             'It also gives the no-contact baseline and each recording’s held-out errors for the fixed structures. Native fits use the training screen. '
             'Physical refits use all training rows with zero intercept. [Common structures and worst-recording scores](common_structures.csv).', '',
             '![Native fronts and the effect of the term budget on velocity error](fronts.png)', '',
             'Each upper curve is one recording held out. Both axes are minimized. Diamonds mark the equation on each native front with lowest internal validation RMSE. '
             'Each lower curve uses the lowest-validation-error full-training refit within each term budget. The lower panels weight each recording equally.', '',
             '## Equations on the expanded native front selected by validation', '',
             '| Held-out recording | Form | Contact terms (plus any intercept) | Screen WAPE | Instability | Validation RMSE | Test RMSE | Test R² | Test pair-normal RMSE |',
             '|---|---|---|---:|---:|---:|---:|---:|---:|']
    for r in chosen:
        lines.append(f"| {r['heldout']} | {r['form']} | {r['terms']} | {r['screen_wape']:.4f} | "
                     f"{r['screen_dyad']:.4g} | {r['validation_rmse']:.2f} | {r['heldout_rmse']:.2f} | "
                     f"{r['heldout_r2']:.3f} | {r['heldout_pair_normal_rmse']:.2f} |")
    lines += ['', 'RMSE is in camera px/s (divide by 45.08199 for cm/s). Pair-normal RMSE measures the error in the two robots’ relative velocity along '
              'the line joining their centers for pairs within 253 px at prediction time. Held-out R² uses the two Cartesian components. '
              'Predictions use the frozen equation before hard-disk position projection.', '',
              '## Error by term budget', '',
              '| Term budget | Most frequent forward structure | Folds selecting it | Forward test RMSE | Forward pair-normal RMSE | Weak test RMSE | Weak pair-normal RMSE |',
              '|---:|---|---:|---:|---:|---:|---:|']
    for size in (0, 1, 2, 3, 4, 6, 8, 10, 14):
        a, b = [r for r in budgets if r['max_terms'] == size]
        lines.append(f"| {size} | {a['most_common_structure']} | {a['folds_selecting_structure']}/{len(tags)} | "
                     f"{a['heldout_rmse']:.2f} | {a['heldout_pair_normal_rmse']:.2f} | "
                     f"{b['heldout_rmse']:.2f} | {b['heldout_pair_normal_rmse']:.2f} |")
    lines += ['', '## Terms, coverage and scoring', '',
              'The 14 terms are `push` (radial push), `overlap` (distance-weighted radial response), `front` and `neighbor_front` '
              '(headings toward the other robot), `approach` (closing speed), `slide` (tangential relative velocity), '
              '`gear` (tangential response to summed spin), `drive` (neighbor propulsion), `drive_in` (its inward component), and '
              '`band0` through `band4` (radial push in disjoint center-distance intervals). The band edges in units of 230 px are 0, 0.93, 0.97, 1.00, 1.05 and 1.10. '
              'All terms are additive vector columns with power one. The scalar native intercept adds the same constant to each Cartesian component.', '',
              'The original 30 NSGA-II runs allowed six right-hand-side terms. We retain every returned member and replay both recorded objectives, preserving each seed and its coefficients. '
              'The expanded search exhaustively applies EPDE’s weighted least-squares subset fitter and dyadic instability scorer to the same 2,048 training-screen rows per fold and form. '
              'The term budget is wide enough for all 14 terms, the optional intercept and the target. EPDE sizes the dyadic partitions from that budget, so compare instability values within each search setting. '
              'WAPE is the sum of absolute residuals divided by the sum of absolute target values. Dyadic instability measures how much fitted coefficients vary across dyadic partitions of those screen rows.', '',
              'There are 16,384 physical subsets and 32,768 variants including the intercept choice per fold and form. The empty and intercept-only variants serve as controls. '
              'Because `push = band0 + band1 + band2 + band3 + band4`, coefficients are non-unique whenever all six appear. The candidate table reports their prediction errors from '
              'the equivalent representation with `push` removed. Those duplicate representations are excluded from front selection. Native eligibility also applies EPDE’s amplification guard '
              'and requires a finite dyadic score below its overflow sentinel. A blank dyadic score denotes an ineligible equation. Every variant retains its prediction error.', '',
              'The physical refit uses all training rows, recording-balanced least squares and zero intercept to preserve the vector law under rotation. '
              'Each fold fits on the first 75% of four recordings, uses their remaining quarters for validation and evaluates the frozen equation on the fifth recording. '
              'Physical refit fronts minimize term count and validation RMSE. Held-out errors are displayed only after front construction. '
              'A recording can share individual robot IDs with the training recordings.', '',
              '## Data files', '',
              f'- [Expanded native Pareto front](native_pareto.csv) contains {len(native)} equations with coefficients, both objectives and held-out errors.',
              f'- [Original NSGA-II fronts](nsga2_fronts.csv) contains all {len(fronts["nsga2_fronts"])} returned entries across seeds, including repeated equations.',
              f'- [Full-training validation Pareto front](validation_pareto.csv) contains {len(physical)} equations with coefficients and held-out errors.',
              '- [Each of the 14 terms fitted alone](single_terms.csv) gives both formulations, all five test recordings, coefficients and errors.',
              '- [Every candidate and its errors](all_candidates.csv.gz) includes the original support, its representative, eligibility and front flags, coefficients and errors. '
              'Native screen fits use the unprefixed columns. The `refit_` columns give full-training physical fits for the no-intercept variant. The `equation` column contains the additive contact contribution.',
              '- [All 15 term budgets](term_budgets.csv) gives recording-balanced errors. Selected sizes follow the alphabetical order of held-out recording names.', '',
              'Run `python projects/robot/collision/run.py --stage pareto` from the repository root to reproduce the tables and figure.']
    (stage/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
