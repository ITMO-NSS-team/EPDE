"""How does the cost of one configured run scale with population size?

MOEA/D-DD processes every weight sector separately, and there is one weight
vector per population member, so a single EPDE "epoch" costs O(pop_size)
sector passes -- i.e. O(pop_size^2) candidate evaluations. A conventional
generational EA costs O(pop_size) per generation. This measures both.

    PYTHONHASHSEED=0 python scaling.py --pop-sizes 8 16 24 32
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

from _common import check_hash_seed, RESULTS_DIR  # noqa: E402
import scenarios as scenarios_module  # noqa: E402
from run_benchmark import run_once  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenario', default='wave',
                        choices=sorted(scenarios_module.ALL))
    parser.add_argument('--pop-sizes', nargs='+', type=int,
                        default=[6, 10, 16, 24, 32])
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    parser.add_argument('--out', default=os.path.join(RESULTS_DIR,
                                                      'scaling.jsonl'))
    args = parser.parse_args()
    check_hash_seed()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    rows = []
    for pop_size in args.pop_sizes:
        for seed in args.seeds:
            for arm in ('native', 'golem_eq_gen'):
                scenario = scenarios_module.ALL[args.scenario]()
                scenario['pop_size'] = pop_size
                scenario['epochs'] = args.epochs
                record = run_once(arm, scenario, seed)
                record['configured_pop_size'] = pop_size
                rows.append(record)
                with open(args.out, 'a', encoding='utf-8') as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + '\n')
                print(f"{arm:14s} pop={pop_size:3d} seed={seed} "
                      f"evals={record['fitness_evaluations']:6d} "
                      f"time={record['elapsed_sec']:7.2f}s", flush=True)

    print(f"\n{'pop':>4s} {'native evals':>13s} {'golem evals':>12s} {'ratio':>7s}")
    for pop_size in args.pop_sizes:
        def mean(arm):
            values = [r['fitness_evaluations'] for r in rows
                      if r['configured_pop_size'] == pop_size and r['arm'] == arm]
            return float(np.mean(values)) if values else float('nan')
        native, golem = mean('native'), mean('golem_eq_gen')
        print(f'{pop_size:4d} {native:13.0f} {golem:12.0f} {native / golem:7.1f}x')


if __name__ == '__main__':
    main()
