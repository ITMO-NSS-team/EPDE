"""Compare cross-parent assembly from one front or all fronts up to the same checkpoint."""
from collections import defaultdict
from dataclasses import replace
from itertools import permutations, product
import json
import sys
from time import monotonic
from types import SimpleNamespace

from score import CHECKPOINT, ROOT, SEEDS, OPTIONS, blind_input, live, native_points, read_archive
from discover import DONOR_CONFIRMATION
from anonymous_exchange import archive_terms, decode, exchanges, fit_pairs, swapped, target_slot
from compare import previous
import numpy as np


def proposal_groups(prefix, donor_rows, data, bank, limit, *, point_targets=None):
    labels, supports, vocab = archive_terms(prefix)
    donors = {supports[i] for row in donor_rows for i in row}
    maps = list(product(exchanges(len(data.data)), exchanges(len(data.shape))))
    edges = set()
    for source in donors:
        for fm, am in maps:
            changed = swapped(source, fm, am)
            if changed != source and all(f in vocab for term in changed for f in term):
                edges.add((source, changed))
    available = donors | {b for _,b in edges}
    targets = {s: [[t for t in s if target_slot(t) == k] for k in range(3)] for s in available}
    ordinary_supports = [[s for s in donors if targets[s][k]] for k in range(3)]
    needed = {(s,t) for s in donors for pivots in targets[s] for t in pivots}
    for source, copy in edges:
        for a,b,c in permutations(range(3)):
            # Excluding two supports can eliminate at most two possible third donors.
            if targets[source][a] and any(s != source and s != copy for s in ordinary_supports[c][:3]):
                needed.update((copy,t) for t in targets[copy][b])
    fits, scores, cost = fit_pairs(prefix, data, bank, labels, supports, needed, limit)
    if point_targets is not None:
        point_targets.update((i, t) for (_,t),i in fits.items())
    by_slot = defaultdict(list)
    for (s,t), i in fits.items():
        by_slot[s,target_slot(t)].append(i)
    ordinary = [np.asarray([i for s in donors for i in by_slot.get((s,k), ())], dtype=int) for k in range(3)]
    # Every triple has either three donors, or one donor, its transformed copy,
    # and an independent donor. Discard repeated supports in every placement.
    groups = []
    def add(a, b, c, pairs):
        ids = np.asarray([(i,j) for i,j in pairs if supports[i] != supports[j]], dtype=int).reshape(-1,2)
        if len(ids) and len(ordinary[c]):
            groups.append(((a,b,c), ids, ordinary[c]))
    add(0,1,2,product(ordinary[0], ordinary[1]))
    for a,b,c in permutations(range(3)):
        add(a,b,c, ((i,j) for source,copy in edges for i in by_slot.get((source,a), ())
                    for j in by_slot.get((copy,b), ())))
    return groups, supports, scores, dict(cost, donor_supports=len(donors))


def candidate_batches(groups, structure_ids):
    for (a,b,c), pairs, third in groups:
        step = max(1, 65536 // len(third))
        for start in range(0,len(pairs),step):
            kept = pairs[start:start+step]
            distinct = ((structure_ids[kept[:,0],None] != structure_ids[third])
                        & (structure_ids[kept[:,1],None] != structure_ids[third]))
            i,j = np.nonzero(distinct)
            rows = np.empty((len(i),3), dtype=int)
            rows[:,a], rows[:,b], rows[:,c] = kept[i,0], kept[i,1], third[j]
            yield rows


def matching_systems(problem, fields, supports, native, groups):
    # Only assessment uses the registered laws. The groups already define and
    # score every admissible candidate, independent of this reference.
    forms, freqs = previous.registry(problem.name, problem)
    laws = {law:i for i,law in enumerate(sorted({s for f in forms for s in f}, key=sorted))}
    orders = {order for form in forms for order in permutations([laws[s] for s in form])}
    identities = {s:i for i,s in enumerate(dict.fromkeys(supports))}
    structure_ids = np.asarray([identities[s] for s in supports])
    kinds = [laws.get(frozenset(previous.label_key(decode(t, fields), freqs) for t in s), -1) for s in identities]
    tags = np.asarray(kinds)[structure_ids]
    native = np.asarray(native, dtype=int)
    native_tags = tags[native]
    native_hits = {tuple(r) for order in orders for r in native[np.all(native_tags == order, axis=1)]}
    assembled = set()
    for (a,b,c), pairs, third in groups:
        for order in orders:
            left = pairs[(tags[pairs[:,0]] == order[a]) & (tags[pairs[:,1]] == order[b])]
            right = third[tags[third] == order[c]]
            for (i,j),k in product(left,right):
                row = [None]*3
                row[a], row[b], row[c] = i, j, k
                # The registered complete systems have three distinct laws.
                assert len(set(structure_ids[row])) == 3
                assembled.add(tuple(row))
    return structure_ids, native_hits, assembled


def verify(problem, fields, supports, scores, native, groups, *, later_native=()):
    native = np.asarray(native, dtype=int)
    structure_ids, native_hits, assembled = matching_systems(problem, fields, supports, native, groups)
    scores = np.asarray(scores)
    checks = {}
    for row in (native_hits | assembled):
        idx = np.asarray(row)
        same = structure_ids[:,None] == structure_ids[idx]
        weak = same | np.all(scores[:,None,:] <= scores[idx],axis=2)
        strict = ~same & np.any(scores[:,None,:] < scores[idx],axis=2)
        checks[row] = weak, strict
    def dominated(rows, weak, strict):
        x, y, z = rows.T
        return np.any(weak[x,0] & weak[y,1] & weak[z,2] & (strict[x,0] | strict[y,1] | strict[z,2]))
    survivors = {r for r,flags in checks.items() if not dominated(native, *flags)}
    unbeaten_by_native = len(survivors)
    if survivors:
        for rows in candidate_batches(groups, structure_ids):
            survivors = {r for r in survivors if not dominated(rows, *checks[r])}
            if not survivors:
                break
    later_native = np.asarray(later_native, dtype=int).reshape(-1,3)
    return dict(native_complete=len(native_hits), assembled_complete=len(assembled),
                complete_generated=len(native_hits | assembled), unbeaten_by_native=unbeaten_by_native,
                complete_nondominated=len(survivors),
                surviving_all_native=sum(not dominated(later_native, *checks[r]) for r in survivors))


def score_archive(archive, checkpoint=CHECKPOINT):
    seed = archive['seed']
    problem = live.load_data('ns', **OPTIONS['ns'])
    assert archive['noise_percent'] == 0 and archive['epochs_completed'] == 150
    assert archive['train_hashes'] == {k:live.R.digest(v) for k,v in problem.data.items()}
    data = SimpleNamespace(data=problem.data, grids=problem.grids, shape=problem.shape, named_arrays={})
    config = live.R.active_config()
    live.R.set_active_config(replace(config, search_space=replace(config.search_space,
                             system_rps_mode='plain', equation_terms_max_number=7)))
    assert archive['settings']['equation_terms_max_number'] == 7
    op = archive['operator']
    assert 1 <= checkpoint <= archive['epochs_completed']
    state = op['epoch_clock'][checkpoint-1]
    budget = op['full_cost'][-1]
    fronts = archive['epoch_fronts'][:checkpoint]
    assert state['epoch'] == fronts[-1]['epoch'] == checkpoint
    prefix = blind_input(archive, state, fronts[-1]['members'])
    historical = {tuple(r) for h in fronts for r in h['members']}
    full_supports, full_scores = native_points(archive)
    native_rows = np.asarray(archive['systems'], dtype=int)
    native_costs = np.asarray(op['system_cost'])
    assert len(native_rows) == len(native_costs)
    result = {}
    for name, parents in [('current', prefix['native_front']), ('historical', historical)]:
        started = monotonic()
        groups, supports, scores, work = proposal_groups(prefix, parents, data, {}, budget-state['cost'])
        n = len(prefix['points'])
        offset = len(full_supports) - n
        groups = [(slots, np.where(p>=n, p+offset, p), np.where(t>=n, t+offset, t)) for slots,p,t in groups]
        cutoff = budget - work['new_fits']
        assert cutoff >= state['cost'] and cutoff in op['full_cost']
        admitted = native_costs <= cutoff
        result[name] = dict(**work, native_fits=cutoff, total_fits=budget,
                            **verify(problem, archive['variables'], full_supports+supports[n:],
                                     full_scores+scores[n:], native_rows[admitted], groups,
                                     later_native=native_rows[~admitted]), seconds=monotonic()-started)
        if 'fit_milliseconds' in op:
            elapsed = op['fit_milliseconds'][op['full_cost'].index(cutoff)]/1000
            result[name].update(native_cutoff_seconds=elapsed, native_full_seconds=archive['seconds'])
        print(seed, checkpoint, name, result[name], flush=True)
    return result


def run(seed):
    result = score_archive(read_archive(ROOT/'results'/str(seed)))
    path = ROOT/'results'/str(seed)/'donor_history.json'
    temp = path.with_suffix('.pending')
    temp.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf8')
    temp.replace(path)


if __name__ == '__main__':
    seed = int(sys.argv[1])
    if seed not in SEEDS + DONOR_CONFIRMATION:
        raise ValueError(seed)
    run(seed)
