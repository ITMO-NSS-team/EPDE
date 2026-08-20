"""Figures for the EPDE-vs-GOLEM comparison.

    python plots.py results/benchmark_instability.jsonl results/benchmark_complexity.jsonl
"""

import argparse
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ARM_ORDER = ['native', 'golem_eq_gen', 'golem_eq_budget']
ARM_LABEL = {'native': 'EPDE (MOEA/D-DD)',
             'golem_eq_gen': 'GOLEM, same generations',
             'golem_eq_budget': 'GOLEM, same eval. budget'}
ARM_COLOR = {'native': '#2b6cb0',
             'golem_eq_gen': '#c05621',
             'golem_eq_budget': '#276749'}


def load(paths):
    records = []
    for path in paths:
        with open(path, encoding='utf-8') as handle:
            for line in handle:
                line = line.strip()
                if line:
                    record = json.loads(line)
                    if 'error' not in record:
                        record['_source'] = os.path.basename(path)
                        records.append(record)
    return records


def bar_figure(records, title, out_path):
    scenarios = sorted({r['scenario'] for r in records})
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    width = 0.26
    x = np.arange(len(scenarios))

    panels = [
        ('recovery of the true equation, %', lambda items: 100 * np.mean(
            [r['structure_match'] for r in items])),
        ('wall-clock time, s', lambda items: np.mean([r['elapsed_sec'] for r in items])),
        ('equation evaluations', lambda items: np.mean(
            [r['fitness_evaluations'] for r in items])),
    ]

    for ax, (label, reducer) in zip(axes, panels):
        for offset, arm in enumerate(ARM_ORDER):
            values = []
            for scenario in scenarios:
                items = [r for r in records
                         if r['scenario'] == scenario and r.get('arm') == arm]
                values.append(reducer(items) if items else 0.0)
            ax.bar(x + (offset - 1) * width, values, width,
                   label=ARM_LABEL[arm], color=ARM_COLOR[arm])
        ax.set_xticks(x)
        ax.set_xticklabels([s.replace('_', '\n') for s in scenarios], fontsize=8)
        ax.set_ylabel(label, fontsize=9)
        ax.grid(axis='y', alpha=0.25)
        if label != 'recovery of the true equation, %':
            ax.set_yscale('log')
    axes[0].legend(fontsize=8, loc='upper left')
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def anytime_figure(records, title, out_path):
    scenarios = sorted({r['scenario'] for r in records if r.get('anytime')})
    if not scenarios:
        return None
    cols = min(3, len(scenarios))
    rows = int(np.ceil(len(scenarios) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.2 * cols, 3.6 * rows),
                             squeeze=False)
    for idx, scenario in enumerate(scenarios):
        ax = axes[idx // cols][idx % cols]
        for arm in ARM_ORDER:
            curves = [r['anytime'] for r in records
                      if r['scenario'] == scenario and r.get('arm') == arm
                      and r.get('anytime')]
            if not curves:
                continue
            grid = np.unique(np.concatenate(
                [np.array(c, dtype=float)[:, 0] for c in curves]))
            stacked = []
            for curve in curves:
                arr = np.array(curve, dtype=float)
                # step function: best-so-far, flat between improvements
                values = np.array([arr[arr[:, 0] <= g][-1, 1]
                                   if np.any(arr[:, 0] <= g) else np.nan
                                   for g in grid])
                stacked.append(values)
            stacked = np.array(stacked, dtype=float)
            median = np.nanmedian(stacked, axis=0)
            ax.step(grid, median, where='post', label=ARM_LABEL[arm],
                    color=ARM_COLOR[arm], linewidth=1.6)
        ax.set_yscale('log')
        ax.set_xscale('log')
        ax.set_title(scenario, fontsize=10)
        ax.set_xlabel('equation evaluations', fontsize=8)
        ax.set_ylabel('best discrepancy', fontsize=8)
        ax.grid(alpha=0.25)
    axes[0][0].legend(fontsize=8)
    for idx in range(len(scenarios), rows * cols):
        axes[idx // cols][idx % cols].axis('off')
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('files', nargs='+')
    parser.add_argument('--outdir', default=None)
    args = parser.parse_args()

    outdir = args.outdir or os.path.join(
        os.path.dirname(os.path.abspath(__file__)), 'results')
    os.makedirs(outdir, exist_ok=True)

    by_source = defaultdict(list)
    for record in load(args.files):
        by_source[record['_source']].append(record)

    for source, records in by_source.items():
        stem = os.path.splitext(source)[0]
        second = records[0].get('second_objective', 'instability')
        title = f'{stem}  (objectives: discrepancy + {second})'
        print(bar_figure(records, title, os.path.join(outdir, f'{stem}_summary.png')))
        path = anytime_figure(records, title,
                              os.path.join(outdir, f'{stem}_anytime.png'))
        if path:
            print(path)


if __name__ == '__main__':
    main()
