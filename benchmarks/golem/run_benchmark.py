"""Head-to-head benchmark: EPDE's native MOEA/D-DD engine vs the GOLEM engine.

Both arms share the data, the token pool, the chromosome representation, the
mutation/crossover operators and the fitness chain; only the population engine
differs.  Three arms are run per (scenario, seed):

``native``
    EPDE as shipped: MOEA/D-DD, ``pop_size`` sectors x ``epochs`` generations.
``golem_eq_gen``
    GOLEM with the *same* population size and generation count -- what a user
    gets by swapping the optimizer and changing nothing else.
``golem_eq_budget``
    GOLEM run until it has spent the same number of equation-level fitness
    evaluations the native arm consumed.  This is the like-for-like comparison:
    one MOEA/D generation costs several GOLEM generations, because MOEA/D
    processes every weight sector separately and retries for unique offspring.

Each record captures wall-clock time, the evaluation count, the resulting
Pareto front, and whether the analytic ground-truth equation was recovered.

Usage::

    python run_benchmark.py --scenarios wave kdv --seeds 0 1 2
"""

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

from _common import set_seeds, check_hash_seed, RESULTS_DIR  # noqa: E402
import scenarios as scenarios_module  # noqa: E402
import metrics  # noqa: E402

ARMS = ('native', 'golem_eq_gen', 'golem_eq_budget')


def instrument_fitness(search_obj):
    """Count equation-level fitness computations performed during a run.

    The chromosome-level fitness operator is an ``OperatorMapper`` around a
    single ``SolverFreeFitness`` instance that is *also* the right-part
    selector's inner evaluator, so wrapping it captures the whole evaluation
    budget -- offspring scoring and right-part sweeps alike.
    """
    blocks = search_obj.director.builder.blocks_labeled
    inner = blocks['initial_sorter']._operator.suboperators['chromosome_fitness']
    while True:
        subops = getattr(getattr(inner, 'suboperators', None), 'suboperators', {})
        if 'to_map' not in subops:
            break
        inner = inner.suboperators['to_map']
    counter = {'n': 0}
    original = inner.apply

    def counted(objective, arguments, **kwargs):
        counter['n'] += 1
        return original(objective=objective, arguments=arguments, **kwargs)

    inner.apply = counted
    return counter


def instrument_trace(search_obj, counter, trace):
    """Record ``(evaluations_so_far, objective_vector)`` for every candidate.

    Hooks the chromosome-level fitness operator, which both engines call once
    per candidate system, so the resulting anytime curves are directly
    comparable. Reading ``obj_fun`` can raise on a candidate the chain left
    unfitted; such candidates are simply not recorded.
    """
    blocks = search_obj.director.builder.blocks_labeled
    mapper = blocks['initial_sorter']._operator.suboperators['chromosome_fitness']
    original = mapper.apply

    def traced(objective, arguments, **kwargs):
        result = original(objective=objective, arguments=arguments, **kwargs)
        try:
            trace.append((counter['n'], [float(v) for v in np.asarray(objective.obj_fun)]))
        except Exception:
            pass
        return result

    mapper.apply = traced
    return trace


def build_search(engine, scenario, golem_params=None):
    if engine == 'native':
        from epde.interface.interface import EpdeSearch as Cls
        extra = {}
    else:
        from epde_golem import EpdeGolemSearch as Cls
        extra = {'golem_params': dict(golem_params or {})}

    search = Cls(use_solver=False, multiobjective_mode=True,
                 boundary=scenario['boundary'],
                 coordinate_tensors=scenario['coordinate_tensors'],
                 verbose_params={'show_iter_idx': False},
                 device='cpu', **scenario['search_kwargs'], **extra)
    search.set_preprocessor(default_preprocessor_type='FD', preprocessor_kwargs={})
    return search


def run_once(arm, scenario, seed, eval_budget=None, epochs=None,
             golem_params=None, trace=False, second_objective=None):
    """Execute one arm and score the resulting Pareto front."""
    engine = 'native' if arm == 'native' else 'golem'
    set_seeds(seed)
    # Must be set before EpdeSearch construction: the fitness fillers are
    # assembled in __init__ and a later change desyncs them from the axis
    # readers.
    import epde.globals as epde_globals
    epde_globals.set_second_objective(second_objective)
    search = build_search(engine, scenario, golem_params)
    counter = instrument_fitness(search)
    trace_records = []
    if trace:
        instrument_trace(search, counter, trace_records)
    if engine == 'golem' and eval_budget:
        search.golem_params.update(eval_budget=eval_budget,
                                   eval_counter=lambda: counter['n'])
    search.set_moeadd_params(population_size=scenario['pop_size'],
                             training_epochs=epochs or scenario['epochs'])
    tokens = scenario['tokens']()

    t0 = time.perf_counter()
    search.fit(data=scenario['data'],
               variable_names=scenario['variable_names'],
               additional_tokens=tokens,
               **scenario['fit_kwargs'])
    elapsed = time.perf_counter() - t0

    levels = search.optimizer.pareto_levels
    front = list(levels.levels[0])
    kept = list(levels.population)            # front + dominated survivors / archive
    objectives = np.array([np.asarray(sol.obj_fun, dtype=float) for sol in front]) \
        if front else np.zeros((0, 2))
    # Two recovery criteria. `structure_match` is what a user sees -- EPDE's
    # ``equations()`` prints the non-dominated level. `structure_match_kept`
    # asks the weaker question "did the run hold the true equation anywhere in
    # what it returned", which separates "never found it" from "found it but
    # ranked something else first".
    match = metrics.best_match(front, scenario['ground_truth'])
    match_kept = metrics.best_match(kept, scenario['ground_truth'])

    # Per-generation flag: was the true equation on the non-dominated front at
    # the end of that generation? Both engines expose the same per-epoch
    # snapshot list, so the trajectories are directly comparable. This is what
    # separates "never found it" from "found it early, then lost it".
    history = getattr(search.optimizer, '_pareto_history', None)
    if history is None:
        history = getattr(search.optimizer, 'pareto_history', []) or []
    recovery_trace = [metrics.snapshot_hits(snapshot, scenario['ground_truth'])
                      for snapshot in history]

    anytime = None
    if trace and trace_records:
        # Running best discrepancy as a function of evaluations spent, thinned
        # to the points where it improves.
        best = float('inf')
        anytime = []
        for n_evals, values in trace_records:
            if values and np.isfinite(values[0]) and values[0] < best:
                best = values[0]
                anytime.append([n_evals, best])

    return dict(
        arm=arm, engine=engine, scenario=scenario['name'], seed=seed,
        anytime=anytime,
        recovery_trace=recovery_trace,
        recovery_ever=bool(any(recovery_trace)),
        pop_size=scenario['pop_size'],
        epochs=epochs or scenario['epochs'],
        eval_budget=eval_budget,
        elapsed_sec=round(elapsed, 3),
        fitness_evaluations=counter['n'],
        front_size=len(front),
        kept_size=len(kept),
        cache_hits=getattr(getattr(search.optimizer, 'evaluator', None),
                           'n_cache_hits', 0),
        best_discrepancy=float(np.min(objectives[:, 0])) if len(front) else None,
        median_discrepancy=float(np.median(objectives[:, 0])) if len(front) else None,
        structure_match=bool(match['structure_match']),
        structure_match_kept=bool(match_kept['structure_match']),
        # Weaker criterion: the true law recovered up to a redundant common
        # factor (e.g. u^3 * (u_tt - 0.04 u_xx) = 0), which both engines
        # produce regularly and which a term-set comparison scores as a miss.
        match_up_to_factor=bool(match['match_up_to_factor']),
        match_up_to_factor_kept=bool(match_kept['match_up_to_factor']),
        coef_error=(None if not np.isfinite(match['coef_error'])
                    else round(float(match['coef_error']), 5)),
        best_equation=(metrics.system_text_forms(front[match['index']])
                       if match['index'] is not None else None),
        front_equations=[metrics.system_text_forms(sol) for sol in front],
        kept_equations=[metrics.system_text_forms(sol) for sol in kept],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenarios', nargs='+', default=sorted(scenarios_module.ALL),
                        choices=sorted(scenarios_module.ALL))
    parser.add_argument('--arms', nargs='+', default=list(ARMS), choices=ARMS)
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    parser.add_argument('--pop-size', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--golem-max-epochs', type=int, default=500,
                        help='generation cap for the budget-matched GOLEM arm')
    parser.add_argument('--golem-scheme', default='steady_state',
                        choices=['steady_state', 'generational', 'parameter_free'],
                        help='GOLEM genetic scheme; "generational" is the '
                             'no-selection-pressure configuration kept for ablation')
    parser.add_argument('--golem-adaptive-mutation', default='default',
                        choices=['default', 'random', 'bandit'],
                        help='GOLEM mutation-choice agent')
    parser.add_argument('--out', default=None)
    parser.add_argument('--tag', default='')
    parser.add_argument('--trace', action='store_true',
                        help='record the anytime curve (best discrepancy vs '
                             'evaluations spent) for every run')
    parser.add_argument('--budget-from', default=None,
                        help='JSONL from an earlier sweep; supplies the native '
                             'evaluation budget per (scenario, seed) so an '
                             'ablation can run the budget-matched GOLEM arm '
                             'without re-running the native one')
    parser.add_argument('--second-objective', default=None,
                        choices=['instability', 'complexity'],
                        help="override EPDE's second Pareto axis; the "
                             'scenarios use instability (use_pic=True) by '
                             'default, which puts no parsimony pressure on '
                             'the search')
    args = parser.parse_args()
    check_hash_seed()

    from golem.core.optimisers.genetic.operators.inheritance import GeneticSchemeTypesEnum
    from golem.core.optimisers.adaptive.operator_agent import MutationAgentTypeEnum
    golem_params = dict(
        genetic_scheme=GeneticSchemeTypesEnum[args.golem_scheme],
        adaptive_mutation_type=MutationAgentTypeEnum[args.golem_adaptive_mutation],
    )

    out_path = args.out or os.path.join(
        RESULTS_DIR,
        f'benchmark_{args.tag or datetime.now().strftime("%Y%m%d_%H%M%S")}.jsonl')
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    known_budgets = {}
    if args.budget_from:
        with open(args.budget_from, encoding='utf-8') as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get('arm') == 'native':
                    known_budgets[(rec['scenario'], rec['seed'])] = \
                        rec['fitness_evaluations']

    for scenario_name in args.scenarios:
        for seed in args.seeds:
            native_budget = known_budgets.get((scenario_name, seed))
            for arm in args.arms:
                scenario = scenarios_module.ALL[scenario_name]()
                if args.pop_size:
                    scenario['pop_size'] = args.pop_size
                if args.epochs:
                    scenario['epochs'] = args.epochs
                budget = epochs = None
                if arm == 'golem_eq_budget':
                    if native_budget is None:
                        print('    skipped: no native budget recorded', flush=True)
                        continue
                    budget, epochs = native_budget, args.golem_max_epochs
                print(f'>>> {scenario_name} | {arm} | seed={seed}'
                      + (f' | budget={budget}' if budget else ''), flush=True)
                try:
                    record = run_once(arm, scenario, seed,
                                      eval_budget=budget, epochs=epochs,
                                      golem_params=golem_params,
                                      trace=args.trace,
                                      second_objective=args.second_objective)
                    record['golem_scheme'] = args.golem_scheme
                    record['golem_agent'] = args.golem_adaptive_mutation
                    record['second_objective'] = args.second_objective or 'instability'
                    if arm == 'native':
                        native_budget = record['fitness_evaluations']
                except Exception as exc:                      # keep the sweep going
                    traceback.print_exc()
                    record = dict(arm=arm, scenario=scenario_name, seed=seed,
                                  error=f'{type(exc).__name__}: {exc}')
                with open(out_path, 'a', encoding='utf-8') as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + '\n')
                print(json.dumps({k: v for k, v in record.items()
                                  if k not in ('front_equations', 'anytime',
                                               'kept_equations', 'recovery_trace')},
                                 ensure_ascii=False), flush=True)
    print('written to', out_path)


if __name__ == '__main__':
    main()
