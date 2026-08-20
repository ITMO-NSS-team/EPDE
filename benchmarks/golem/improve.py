"""Which algorithmic changes make GOLEM better *for this domain*?

Every configuration runs under the same equation-evaluation budget, so the
comparison is of search quality per unit of work, not of how much work each one
happens to do. The budget is a fraction of what the native EPDE arm spent on
the same (scenario, seed), read from an earlier benchmark file.

Configurations
--------------
base
    The integration's defaults: steady-state scheme, SPEA-2 survival, uniform
    choice between EPDE's term-replacement mutation and a term-addition one.
mating
    Adds a mating-pool selection (binary tournament with replacement). Without
    it GOLEM's reproduction applies no selection pressure at all -- the
    requested pool is never smaller than the population, and every selection
    operator answers that by returning everyone.
nsga2
    Swaps SPEA-2 for NSGA-II rank + crowding distance as survival selection.
ops
    Widens the mutation action set: term add, term drop, whole-equation reroll,
    sparsity jitter.
bandit
    The wide action set plus GOLEM's multi-armed-bandit agent choosing between
    the actions, instead of uniform random.
cache
    Memoizes evaluated chromosomes by (structure, metaparameters), so a
    re-derived structure costs a copy instead of a full fit and right-part
    sweep.
tuned
    Everything that helped, together.

    PYTHONHASHSEED=0 python improve.py --budget-from ../results/benchmark_instability.jsonl
"""

import argparse
import json
import os
import sys
import traceback
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _common import check_hash_seed, RESULTS_DIR  # noqa: E402
import scenarios as scenarios_module  # noqa: E402
from run_benchmark import run_once  # noqa: E402

from golem.core.optimisers.genetic.operators.selection import SelectionTypesEnum  # noqa: E402
from golem.core.optimisers.adaptive.operator_agent import MutationAgentTypeEnum  # noqa: E402

WIDE_OPS = ('add', 'drop', 'reroll', 'sparsity')

CONFIGS = {
    'base': {},
    'mating': {'mating_selection': SelectionTypesEnum.tournament_with_replacement},
    'nsga2': {'selection': SelectionTypesEnum.nsga2},
    'ops': {'extra_mutations': WIDE_OPS},
    'bandit': {'extra_mutations': WIDE_OPS,
               'adaptive_mutation_type': MutationAgentTypeEnum.bandit},
    'cache': {'fitness_cache_size': 20000},
    'tuned': {'mating_selection': SelectionTypesEnum.tournament_with_replacement,
              'selection': SelectionTypesEnum.nsga2,
              'extra_mutations': WIDE_OPS,
              'adaptive_mutation_type': MutationAgentTypeEnum.bandit,
              'fitness_cache_size': 20000},
    # Restarts: the same budget spent on several independent short searches,
    # whose archives are unioned. GOLEM has no restart mechanism.
    'restart3': {'restarts': 3},
    'restart5': {'restarts': 5},
    'restart8': {'restarts': 8},
    # Restarts on top of the two changes that did not hurt at 10% budget.
    'restart5_nsga2': {'restarts': 5, 'selection': SelectionTypesEnum.nsga2},
    # More diversity per generation instead of more generations.
    'pop32': {'pop_size_override': 32},
    'pop48': {'pop_size_override': 48},
}


def load_budgets(path):
    budgets = {}
    with open(path, encoding='utf-8') as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get('arm') == 'native':
                budgets[(record['scenario'], record['seed'])] = \
                    record['fitness_evaluations']
    return budgets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenarios', nargs='+',
                        default=['wave', 'kdv', 'allen_cahn', 'van_der_pol'],
                        choices=sorted(scenarios_module.ALL))
    parser.add_argument('--configs', nargs='+', default=list(CONFIGS),
                        choices=list(CONFIGS))
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    parser.add_argument('--pop-size', type=int, default=16)
    parser.add_argument('--budget-scale', type=float, default=0.2,
                        help='fraction of the native evaluation budget to allow')
    parser.add_argument('--budget-from', default=os.path.join(
        RESULTS_DIR, 'benchmark_instability.jsonl'))
    parser.add_argument('--second-objective', default=None,
                        choices=['instability', 'complexity'])
    parser.add_argument('--max-epochs', type=int, default=2000)
    parser.add_argument('--tag', default='')
    args = parser.parse_args()
    check_hash_seed()

    budgets = load_budgets(args.budget_from)
    out_path = os.path.join(
        RESULTS_DIR,
        f'improve_{args.tag or datetime.now().strftime("%Y%m%d_%H%M%S")}.jsonl')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    for scenario_name in args.scenarios:
        for seed in args.seeds:
            native = budgets.get((scenario_name, seed))
            if native is None:
                print(f'!!! no native budget for {scenario_name} seed={seed}', flush=True)
                continue
            budget = max(50, int(round(args.budget_scale * native)))
            for config_name in args.configs:
                params = dict(CONFIGS[config_name])
                scenario = scenarios_module.ALL[scenario_name]()
                scenario['pop_size'] = params.pop('pop_size_override', args.pop_size)
                print(f'>>> {scenario_name} | {config_name} | seed={seed} '
                      f'| budget={budget} ({args.budget_scale:.0%} of {native})',
                      flush=True)
                try:
                    record = run_once('golem_eq_budget', scenario, seed,
                                      eval_budget=budget, epochs=args.max_epochs,
                                      golem_params=params,
                                      second_objective=args.second_objective)
                    record['config'] = config_name
                    record['budget_scale'] = args.budget_scale
                    record['native_budget'] = native
                except Exception as exc:
                    traceback.print_exc()
                    record = dict(config=config_name, scenario=scenario_name,
                                  seed=seed, error=f'{type(exc).__name__}: {exc}')
                with open(out_path, 'a', encoding='utf-8') as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + '\n')
                print(json.dumps({k: v for k, v in record.items()
                                  if k not in ('front_equations', 'anytime',
                                               'kept_equations', 'recovery_trace')},
                                 ensure_ascii=False), flush=True)
    print('written to', out_path)


if __name__ == '__main__':
    main()
