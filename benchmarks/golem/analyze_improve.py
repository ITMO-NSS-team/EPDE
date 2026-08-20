"""Aggregate the GOLEM-configuration sweep from ``improve.py``."""

import argparse
import json
from collections import defaultdict

import numpy as np

ORDER = ['base', 'mating', 'nsga2', 'ops', 'bandit', 'cache', 'tuned',
         'pop32', 'pop48', 'restart3', 'restart5', 'restart8', 'restart5_nsga2']
LABEL = {
    'base': 'baseline (steady-state + SPEA-2)',
    'mating': '+ mating tournament w/ replacement',
    'nsga2': '+ NSGA-II survival selection',
    'ops': '+ wider mutation action set',
    'bandit': '+ wide set, bandit agent',
    'cache': '+ structure-keyed fitness cache',
    'tuned': 'all of the above',
    'pop32': 'population 32 instead of 16',
    'pop48': 'population 48 instead of 16',
    'restart3': '3 restarts, budget split',
    'restart5': '5 restarts, budget split',
    'restart8': '8 restarts, budget split',
    'restart5_nsga2': '5 restarts + NSGA-II',
}


def load(paths):
    records = []
    for path in paths:
        with open(path, encoding='utf-8') as handle:
            for line in handle:
                line = line.strip()
                if line:
                    record = json.loads(line)
                    if 'error' not in record:
                        records.append(record)
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('files', nargs='+')
    parser.add_argument('--per-scenario', action='store_true')
    args = parser.parse_args()

    records = load(args.files)
    if not records:
        print('no successful records')
        return

    if args.per_scenario:
        groups = defaultdict(list)
        for record in records:
            groups[(record['scenario'], record['config'])].append(record)
        header = (f"{'scenario':14s} {'config':36s} {'runs':>4s} {'front':>6s} "
                  f"{'kept':>6s} {'ever':>6s} {'time,s':>9s} {'evals':>7s} "
                  f"{'cached':>7s} {'disc.median':>12s}")
        print(header)
        print('-' * len(header))
        for scenario in sorted({r['scenario'] for r in records}):
            for config in ORDER:
                items = groups.get((scenario, config))
                if not items:
                    continue
                print(_row(f'{scenario:14s} {LABEL.get(config, config):36s}', items))
            print()

    groups = defaultdict(list)
    for record in records:
        groups[record['config']].append(record)
    header = (f"{'config':36s} {'runs':>4s} {'front':>6s} {'kept':>6s} "
              f"{'ever':>6s} {'time,s':>9s} {'evals':>7s} {'front':>6s} "
              f"{'kept':>6s} {'disc.median':>12s}")
    print('Averaged over scenarios and seeds:')
    print(header)
    print('-' * len(header))
    for config in ORDER:
        items = groups.get(config)
        if not items:
            continue
        print(_row(f'{LABEL.get(config, config):36s}', items))


def _row(prefix, items):
    discrepancies = [r['best_discrepancy'] for r in items
                     if r['best_discrepancy'] is not None]
    return (f"{prefix} {len(items):4d} "
            f"{100 * np.mean([r['structure_match'] for r in items]):5.0f}% "
            f"{100 * np.mean([r['structure_match_kept'] for r in items]):5.0f}% "
            f"{100 * np.mean([r.get('recovery_ever', False) for r in items]):5.0f}% "
            f"{np.mean([r['elapsed_sec'] for r in items]):9.1f} "
            f"{np.mean([r['fitness_evaluations'] for r in items]):7.0f} "
            f"{np.mean([r['front_size'] for r in items]):6.1f} "
            f"{np.mean([r['kept_size'] for r in items]):6.1f} "
            f"{np.median(discrepancies) if discrepancies else float('nan'):12.3e}")


if __name__ == '__main__':
    main()
