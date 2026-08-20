"""Aggregate benchmark records into a per-scenario comparison table."""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

ARM_ORDER = ['native', 'golem_eq_gen', 'golem_eq_budget']
ARM_LABEL = {'native': 'EPDE (MOEA/D-DD)',
             'golem_eq_gen': 'GOLEM, same generations',
             'golem_eq_budget': 'GOLEM, same eval. budget'}


def load(paths):
    records = []
    for path in paths:
        with open(path, encoding='utf-8') as handle:
            for line in handle:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return [r for r in records if 'error' not in r]


def aggregate(records):
    groups = defaultdict(list)
    for record in records:
        groups[(record['scenario'], record.get('arm', record.get('engine')))].append(record)

    rows = []
    for (scenario, arm), items in sorted(groups.items()):
        discrepancies = [r['best_discrepancy'] for r in items
                         if r['best_discrepancy'] is not None]
        rows.append(dict(
            scenario=scenario, arm=arm, runs=len(items),
            recovery=float(np.mean([r['structure_match'] for r in items])),
            recovery_kept=float(np.mean([r.get('structure_match_kept',
                                               r['structure_match'])
                                         for r in items])),
            recovery_ever=float(np.mean([r.get('recovery_ever', False) for r in items])),
            recovery_factor=float(np.mean([r.get('match_up_to_factor_kept',
                                                 r.get('match_up_to_factor',
                                                       r['structure_match']))
                                           for r in items])),
            time_mean=float(np.mean([r['elapsed_sec'] for r in items])),
            time_std=float(np.std([r['elapsed_sec'] for r in items])),
            evals_mean=float(np.mean([r['fitness_evaluations'] for r in items])),
            ms_per_eval=float(np.mean([1000 * r['elapsed_sec'] / r['fitness_evaluations']
                                       for r in items if r['fitness_evaluations']])),
            disc_median=float(np.median(discrepancies)) if discrepancies else float('nan'),
            disc_best=float(np.min(discrepancies)) if discrepancies else float('nan'),
            front_mean=float(np.mean([r['front_size'] for r in items])),
        ))
    return rows


def print_table(rows):
    header = (f"{'scenario':16s} {'arm':26s} {'runs':>4s} {'front':>6s} "
              f"{'kept':>6s} {'x-fact':>7s} {'ever':>6s} {'time,s':>12s} "
              f"{'evals':>8s} {'ms/eval':>8s} {'disc.median':>12s}")
    print(header)
    print('-' * len(header))
    print('recovery of the ground truth: on the final front / anywhere in the '
          'returned set / up to a redundant common factor / on the front at '
          'ANY generation')
    print('-' * len(header))
    for scenario in sorted({r['scenario'] for r in rows}):
        for arm in ARM_ORDER:
            for row in rows:
                if row['scenario'] != scenario or row['arm'] != arm:
                    continue
                print(f"{row['scenario']:16s} {ARM_LABEL.get(arm, arm):26s} "
                      f"{row['runs']:4d} {row['recovery']*100:5.0f}% "
                      f"{row['recovery_kept']*100:5.0f}% "
                      f"{row['recovery_factor']*100:6.0f}% "
                      f"{row['recovery_ever']*100:5.0f}% "
                      f"{row['time_mean']:8.1f}+-{row['time_std']:<4.1f} "
                      f"{row['evals_mean']:8.0f} {row['ms_per_eval']:8.2f} "
                      f"{row['disc_median']:12.3e}")
        print()


def print_totals(rows):
    print('Overall (mean over scenarios):')
    for arm in ARM_ORDER:
        subset = [r for r in rows if r['arm'] == arm]
        if not subset:
            continue
        print(f"  {ARM_LABEL.get(arm, arm):26s} "
              f"front={np.mean([r['recovery'] for r in subset])*100:5.1f}%  "
              f"kept={np.mean([r['recovery_kept'] for r in subset])*100:5.1f}%  "
              f"up-to-factor={np.mean([r['recovery_factor'] for r in subset])*100:5.1f}%  "
              f"ever={np.mean([r['recovery_ever'] for r in subset])*100:5.1f}%  "
              f"time={np.mean([r['time_mean'] for r in subset]):7.1f}s  "
              f"evals={np.mean([r['evals_mean'] for r in subset]):8.0f}")


def markdown_table(rows) -> str:
    lines = ['| scenario | engine | runs | recovery (front) | recovery (kept) '
             '| recovery (ever) | time, s | evaluations | ms/eval | median discrepancy |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for scenario in sorted({r['scenario'] for r in rows}):
        for arm in ARM_ORDER:
            for row in rows:
                if row['scenario'] != scenario or row['arm'] != arm:
                    continue
                lines.append(
                    f"| {row['scenario']} | {ARM_LABEL.get(arm, arm)} | {row['runs']} "
                    f"| {row['recovery']*100:.0f}% | {row['recovery_kept']*100:.0f}% "
                    f"| {row['recovery_ever']*100:.0f}% "
                    f"| {row['time_mean']:.1f} +/- {row['time_std']:.1f} "
                    f"| {row['evals_mean']:.0f} | {row['ms_per_eval']:.2f} "
                    f"| {row['disc_median']:.3e} |")
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('files', nargs='+')
    parser.add_argument('--json-out', default=None)
    parser.add_argument('--markdown', action='store_true')
    args = parser.parse_args()

    records = load(args.files)
    if not records:
        print('no successful records found')
        return
    rows = aggregate(records)
    if args.markdown:
        print(markdown_table(rows))
        return
    print_table(rows)
    print_totals(rows)
    if args.json_out:
        with open(args.json_out, 'w', encoding='utf-8') as handle:
            json.dump(rows, handle, indent=2)
        print('\nwritten', args.json_out)


if __name__ == '__main__':
    main()
