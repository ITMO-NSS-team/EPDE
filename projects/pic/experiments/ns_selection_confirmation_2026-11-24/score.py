"""Paired, equal-fit-budget audit of the complete systems handed over to selection."""
import argparse
from dataclasses import replace
import json
from time import monotonic
from types import SimpleNamespace

from discover import CHECKPOINT, ROOT, SEEDS
from anonymous_exchange import decode, encode, evaluate, freeze, generate
from fixed_exchange import generate_fixed
from compare import read_archive
from epoch_settings import OPTIONS, SEEDS as OLD_SEEDS, directory, epochs, live


def native_points(archive):
    labels = [encode(freeze(t), archive['variables']) for t in archive['labels']]
    supports = [frozenset(labels[j] for j, _ in point[2]) for point in archive['points']]
    return supports, [point[3:5] for point in archive['points']]


def blind_input(archive, state, parents):
    # Handovers sharing the final fit clock can straddle the epoch callback.
    # At that exact clock only the observed parents are guaranteed to predate it.
    rows = {tuple(r) for r, work in zip(archive['systems'], archive['operator']['system_cost'], strict=True)
            if work < state['cost']}
    rows.update(map(tuple, parents))
    assert all(max(r) < state['points'] for r in rows)
    points = archive['points'][:state['points']]
    assert all(j < state['labels'] for point in points for j, _ in point[2])
    return dict(seed=archive['seed'], variables=archive['variables'], plan=archive['plan'],
                labels=archive['labels'][:state['labels']], points=points,
                systems=sorted(rows), native_front=parents)


def compare_arms(problem, archive, retrospective=False, *, placement_counts=False):
    assert archive['input_shape'] == list(problem.shape) == [200, 50, 100]
    assert archive['noise_percent'] == 0 and archive['epochs_completed'] == epochs('ns')
    assert archive['train_hashes'] == {k:live.R.digest(v) for k,v in problem.data.items()}
    data = SimpleNamespace(data=problem.data, grids=problem.grids, shape=problem.shape, named_arrays={})
    cfg = live.R.active_config()
    width = archive['settings']['equation_terms_max_number']
    live.R.set_active_config(replace(cfg, search_space=replace(cfg.search_space,
                             system_rps_mode='plain', equation_terms_max_number=width)))
    operator = archive['operator']
    budget = archive['calls'] + operator['extra_fits']
    assert budget == operator['full_cost'][-1] and len(operator['system_cost']) == len(archive['systems'])
    native_supports, native_scores = native_points(archive)
    native = evaluate(problem, archive, [], native_supports, native_scores, all_handovers=True)
    native.update(search_fits=budget, new_fits=0, total_fits=budget,
                  finalized_native_epoch=archive['epochs_completed'])
    if retrospective:
        prefix = archive
        prefix_cost = budget
        limit = None
    else:
        clock = operator['epoch_clock']
        assert [c['epoch'] for c in clock] == list(range(1, epochs('ns')+1))
        assert clock[-1]['cost'] == budget and clock[-1]['points'] == len(archive['points'])
        state = clock[CHECKPOINT-1]
        prefix_cost = state['cost']
        prefix = blind_input(archive, state, archive['epoch_fronts'][CHECKPOINT-1]['members'])
        limit = budget - prefix_cost
    arms = dict(native=native)
    bank = {}
    methods = [('exchange', generate_fixed, {'retarget_copy': False}),
               ('exchange_local_target', generate_fixed, {'retarget_copy': True}),
               ('reassignment', generate, {'exchange': False}),
               ('exchange_reassignment', generate, {'exchange': True})]
    for arm, propose, options in methods:
        started = monotonic()
        proposals, supports, scores, work = propose(prefix, data, bank, **options, max_new_fits=limit)
        cutoff = budget if retrospective else budget - work['new_fits']
        assert cutoff >= prefix_cost
        last_epoch = epochs('ns') if retrospective else max(c['epoch'] for c in clock if c['cost'] <= cutoff)
        native_cut = dict(variables=archive['variables'])
        native_cut['systems'] = [r for r, cost in zip(archive['systems'], operator['system_cost'], strict=True)
                                 if cost <= cutoff]
        native_cut['native_front'] = (archive['native_front'] if retrospective else
                                      archive['epoch_fronts'][last_epoch-1]['members'])
        n = len(prefix['points'])
        offset = len(native_supports) - n
        proposals = [tuple(i + offset if i >= n else i for i in r) for r in proposals]
        supports = native_supports + supports[n:]
        scores = native_scores + scores[n:]
        measured = evaluate(problem, native_cut, proposals, supports, scores,
                            all_handovers=True, placement_counts=placement_counts)
        example = measured['first_survivor']
        if example:
            measured['surviving_supports'] = [[decode(t, archive['variables']) for t in sorted(supports[i])]
                                               for i in example]
        assert cutoff in operator['full_cost']
        arms[arm] = dict(**measured, **work, search_fits=cutoff, total_fits=cutoff+work['new_fits'],
                         finalized_native_epoch=last_epoch, assembly_and_scoring_seconds=monotonic()-started)
    return dict(seed=archive['seed'], retrospective=retrospective, native_seconds=archive['seconds'],
                checkpoint_epoch=epochs('ns') if retrospective else CHECKPOINT, checkpoint_fits=prefix_cost,
                native_budget=budget, arms=arms)


def score(seed, retrospective):
    dest = directory('ns', seed, 0) if retrospective else ROOT/'results'/str(seed)
    result = compare_arms(live.load_data('ns', **OPTIONS['ns']), read_archive(dest), retrospective)
    output = ROOT/'results'/str(seed)/'comparison.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    pending = output.with_suffix('.pending')
    pending.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf8')
    pending.replace(output)
    for arm, row in result['arms'].items():
        print(seed, arm, 'generated', row['complete_generated'], 'nondominated', row['complete_nondominated'],
              'extra fits', row['new_fits'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('seed', type=int, choices=(*SEEDS, *OLD_SEEDS))
    args = parser.parse_args()
    score(args.seed, args.seed in OLD_SEEDS)
