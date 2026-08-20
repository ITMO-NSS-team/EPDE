"""When during the search is the true equation on the front -- and does it stay?

The headline benchmark scores only the *final* front. That hides the effect
this domain is actually dominated by: the parsimonious true equation shows up
early and is then displaced by lower-discrepancy, non-parsimonious forms. This
script records, generation by generation, whether the ground truth is on the
non-dominated front, for both engines.

    PYTHONHASHSEED=0 python recovery_trajectory.py --scenarios wave burgers --seeds 0 1 2
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _common import set_seeds, check_hash_seed, RESULTS_DIR  # noqa: E402
import scenarios as scenarios_module  # noqa: E402
import metrics  # noqa: E402
from run_benchmark import build_search  # noqa: E402


def equation_lines(text_form: str) -> list:
    """Per-equation lines of a system's ``text_form``.

    Single-equation systems print as ``<equation>\\n<metaparameters>``;
    systems print each equation on its own line prefixed by ``/``, ``|`` or
    ``\\``. Metaparameter lines carry no ``=`` at the top level and start with
    ``{``.
    """
    lines = []
    for raw in text_form.split('\n'):
        line = raw.strip()
        if not line or line.startswith('{') or '=' not in line:
            continue
        lines.append(line.lstrip('/|\\ ').strip())
    return lines


def snapshot_hits(snapshot, ground_truth):
    """Is the ground truth present on this generation's front?"""
    for entry in snapshot:
        forms = equation_lines(entry['text_form'])
        score = metrics.score_forms(forms, ground_truth)
        if score['structure_match']:
            return True
    return False


def run(engine, scenario, seed):
    set_seeds(seed)
    search = build_search(engine, scenario, {})
    search.set_moeadd_params(population_size=scenario['pop_size'],
                             training_epochs=scenario['epochs'])
    search.fit(data=scenario['data'],
               variable_names=scenario['variable_names'],
               additional_tokens=scenario['tokens'](),
               **scenario['fit_kwargs'])
    optimizer = search.optimizer
    history = getattr(optimizer, '_pareto_history', None)
    if history is None:
        history = getattr(optimizer, 'pareto_history', [])
    hits = [snapshot_hits(snapshot, scenario['ground_truth']) for snapshot in history]
    return dict(engine=engine, scenario=scenario['name'], seed=seed,
                generations=len(hits), hits=hits,
                ever=any(hits), final=bool(hits and hits[-1]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--scenarios', nargs='+', default=['wave'],
                        choices=sorted(scenarios_module.ALL))
    parser.add_argument('--seeds', nargs='+', type=int, default=[0, 1, 2])
    parser.add_argument('--engines', nargs='+', default=['native', 'golem'])
    parser.add_argument('--pop-size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--out', default=os.path.join(RESULTS_DIR,
                                                      'recovery_trajectory.jsonl'))
    args = parser.parse_args()
    check_hash_seed()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    for name in args.scenarios:
        for seed in args.seeds:
            for engine in args.engines:
                scenario = scenarios_module.ALL[name]()
                scenario['pop_size'] = args.pop_size
                scenario['epochs'] = args.epochs
                print(f'>>> {name} | {engine} | seed={seed}', flush=True)
                try:
                    record = run(engine, scenario, seed)
                except Exception as exc:
                    import traceback
                    traceback.print_exc()
                    record = dict(engine=engine, scenario=name, seed=seed,
                                  error=f'{type(exc).__name__}: {exc}')
                with open(args.out, 'a', encoding='utf-8') as handle:
                    handle.write(json.dumps(record, ensure_ascii=False) + '\n')
                print(json.dumps(record, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
