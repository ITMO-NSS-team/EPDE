"""Probe: how much selection pressure does GOLEM actually apply?

Wraps ``Selection.__call__`` and records (input size, requested size, output
size) for every invocation of a run.  A step where ``output == input`` is a
no-op: every parent bred, regardless of fitness.
"""

import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import scenarios as scenarios_module  # noqa: E402
from run_benchmark import run_once  # noqa: E402

from golem.core.optimisers.genetic.operators.selection import Selection  # noqa: E402


def main():
    scheme = sys.argv[1] if len(sys.argv) > 1 else 'generational'
    name = sys.argv[2] if len(sys.argv) > 2 else 'wave'
    pop = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    epochs = int(sys.argv[4]) if len(sys.argv) > 4 else 5

    from golem.core.optimisers.genetic.operators.inheritance import GeneticSchemeTypesEnum
    stats = Counter()
    original = Selection.__call__

    def probed(self, population, pop_size=None):
        result = original(self, population, pop_size)
        stats[(len(population), pop_size, len(result))] += 1
        return result

    Selection.__call__ = probed

    scenario = scenarios_module.ALL[name]()
    scenario['pop_size'] = pop
    scenario['epochs'] = epochs
    record = run_once('golem_eq_gen', scenario, 0,
                      golem_params={'genetic_scheme': GeneticSchemeTypesEnum[scheme]})

    Selection.__call__ = original

    print(f'\nscheme={scheme} scenario={name} pop={pop} epochs={epochs}')
    print(f"evals={record['fitness_evaluations']} front={record['front_size']} "
          f"best_disc={record['best_discrepancy']:.4e} match={record['structure_match']}")
    print('\n(input, requested, output) -> times')
    no_op = total = 0
    for key, count in sorted(stats.items()):
        marker = ''
        if key[2] >= key[0]:
            marker = '   <-- no-op: every parent survives'
            no_op += count
        total += count
        print(f'  {key} -> {count}{marker}')
    print(f'\nno-op selections: {no_op}/{total}')


if __name__ == '__main__':
    main()
