"""Try pressure-gradient targets on the actual final fronts of the clean NS searches."""
import json
import re
from dataclasses import replace
from itertools import permutations, product

from epoch_settings import OPTIONS, ROOT, SEEDS, directory, live, np
from compare import previous, read_archive
from epde.operators.common.subset_selection import subset_coefficients
from epde.operators.common.survival import dyad_scores


def freeze(term):
    return tuple(sorted((name, tuple(sorted(map(tuple, params)))) for name, params in term))


def mirror(term):
    # (t,y,x) -> (t,x,y), (u,v,p) -> (v,u,p). Keep derivative orders and powers.
    def swap(name):
        name = name.translate(str.maketrans('uv', 'vu'))
        return re.sub(r'/dx([12])', lambda m: '/dx' + str(3-int(m[1])), name)
    return tuple(sorted((swap(name), params) for name, params in term))


def pressure_target(term):
    return (len(term) == 1 and term[0][0] in ('dp/dx1', 'dp/dx2')
            and dict(term[0][1])['power'] == 1)


def polynomial(support):
    return all(n in ('u', 'v', 'p') or re.fullmatch(r'd(?:\^\d+)?[uvp]/dx\d+(?:\^\d+)?', n)
               for t in support for n, _ in t)


def eligible_derivative(t, slot):
    if slot == 'p':
        return pressure_target(t)
    return (len(t) == 1 and re.fullmatch(r'd(?:\^\d+)?' + slot + r'/dx[012](?:\^\d+)?', t[0][0])
            and dict(t[0][1])['power'] == 1)


def make_columns(problem, raw, plan, needed, cache):
    missing = sorted(needed - cache.keys())
    for offset in range(0, len(missing), 8):
        block = missing[offset:offset+8]
        terms, opaque = live.R.symbols(problem, block)
        matrix = live.R.LocalBank(live.R.axes_of(problem), terms, opaque, plan).matrix(raw)
        cache.update(zip(block, matrix.T))
    if () in cache:
        np.testing.assert_allclose(cache[()], 1, atol=1e-12)
        cache[()] = np.ones_like(cache[()])


def refit(support, target, columns, shape):
    predictors = sorted(support - {target, ()})
    if not predictors:
        return None
    x = np.column_stack([columns[t] for t in predictors])
    y = columns[target]
    w = np.ones_like(y)
    intercept = () in support
    beta = subset_coefficients(x, y, w, [True]*len(predictors) + [intercept])
    design = np.column_stack([x, w]) if intercept else x
    scales = np.linalg.norm(design, axis=0)
    if np.any(scales == 0) or np.linalg.matrix_rank(design/scales) != design.shape[1]:
        return None
    active = beta if intercept else beta[:-1]
    if np.any(active == 0):
        return None
    amp = np.abs(active) @ scales / np.linalg.norm(y)
    residual = np.sum(np.abs(y - x@beta[:-1] - beta[-1])) / np.sum(np.abs(y))
    if residual > 1 or amp > live.R.active_config().search_space.rps_amplification_cap:
        return None
    instability = np.sum(dyad_scores(design, y, w, shape, fit_intercept=False))
    if not np.isfinite(instability) or instability >= live.LOSS_NAN_VAL:
        return None
    return (float(residual), float(instability))


def measure(combos, original, objectives, structures, targets, gradient_targets, kind, allowed,
            count_all=False):
    # Find hits only after every proposal was constructed and fitted. Compare a
    # hit to all tested rivals using EPDE's rule that skips equal slot structures.
    combos = np.asarray(sorted(combos), dtype=int).reshape(-1, 3)
    all_rows = np.concatenate([original, combos])
    structure_ids = np.asarray(structures)
    labels = structure_ids[all_rows]
    objectives = np.asarray(objectives)
    scores = objectives[all_rows]
    target_ids = np.asarray(targets)
    pressure = target_ids[all_rows[:, 2]]
    correct = np.array([frozenset(kind[i] for i in row) in allowed for row in combos], dtype=bool)
    in_gradient_group = np.isin(pressure, gradient_targets)
    surviving, same_pressure, gradient_group, original_only, witnesses = [], [], [], [], []
    global_count = gradient_count = 0
    for row, hit in zip(combos, correct):
        if not (hit or count_all):
            continue
        same = labels == structure_ids[row]
        point = objectives[row]
        weak = np.all(same | np.all(scores <= point, axis=2), axis=1)
        strict = np.any(~same & np.any(scores < point, axis=2), axis=1)
        dominates = weak & strict
        global_pass = not np.any(dominates)
        gradient_pass = not np.any(dominates & in_gradient_group)
        global_count += global_pass
        gradient_count += gradient_pass
        if not hit:
            continue
        if global_pass:
            surviving.append(row)
        if gradient_pass:
            gradient_group.append(row)
        if not np.any(dominates[:len(original)]):
            original_only.append(row)
        if not np.any(dominates & (pressure == target_ids[row[2]])):
            same_pressure.append(row)
        if not witnesses and np.any(dominates):
            idx = int(np.flatnonzero(dominates)[0])
            witnesses.append(dict(candidate=row.tolist(), rival=all_rows[idx].tolist(),
                                  rival_is_original=idx < len(original), candidate_scores=point.tolist(),
                                  rival_scores=scores[idx].tolist(), same_slot_structures=same[idx].tolist()))
    example = None
    if surviving:
        row = surviving[0]
        example = dict(point_ids=row.tolist(), pressure_wape=objectives[row[2]][0],
                       pressure_instability=objectives[row[2]][1])
    return dict(proposed_systems=len(combos), complete=int(correct.sum()), unbeaten_by_original=len(original_only),
                complete_nondominated=len(surviving), unbeaten_with_same_pressure_target=len(same_pressure),
                complete_on_pressure_gradient_front=len(gradient_group),
                proposed_on_global_front=global_count if count_all else None,
                proposed_on_pressure_gradient_front=gradient_count if count_all else None,
                example=example, dominance_witness=witnesses)


def replay(seed, problem, column_banks):
    archive = read_archive(directory('ns', seed, 0))
    assert archive['variables'] == ['u', 'v', 'p']
    assert archive['train_hashes'] == {k: live.R.digest(v) for k,v in problem.data.items()}
    plan = {k: tuple(v) for k,v in archive['plan'].items()}
    token = tuple(sorted(plan.items()))
    cache = column_banks.setdefault(token, {})
    labels = list(map(freeze, archive['labels']))
    points = archive['points']
    supports = [frozenset(labels[j] for j,_ in p[2]) for p in points]
    targets = [labels[p[1]] for p in points]
    objectives = [p[3:5] for p in points]
    front = np.asarray(archive['native_front'])
    pairs = sorted(set(map(tuple, front[:, :2])))
    donors = {supports[i] for pair in pairs for i in pair
              if polynomial(supports[i]) and any(pressure_target(t) for t in supports[i])}
    mirrored = {s: frozenset(map(mirror, s)) for s in donors}
    possible = donors | set(mirrored.values())
    triples = set()
    for a, b in pairs:
        original = (supports[a], supports[b])
        if all(map(polynomial, original)):
            for s in original:
                if s in mirrored and len(set((*original, mirrored[s]))) == 3:
                    triples.add(frozenset((*original, mirrored[s])))
    all_supports = possible | {s for triple in triples for s in triple}
    make_columns(problem, problem.data, plan, set().union(*all_supports), cache)
    ids = {}
    for s in sorted(all_supports, key=lambda item: tuple(sorted(item))):
        for t in sorted(s):
            if any(eligible_derivative(t, slot) for slot in ('u', 'v', 'p')):
                score = refit(s, t, cache, plan['shape'])
                if score is not None:
                    ids[s, t] = len(supports)
                    supports.append(s)
                    targets.append(t)
                    objectives.append(score)
    direct, exchanged = set(), set()
    def add(pool, pair, support):
        if support in (supports[pair[0]], supports[pair[1]]):
            return
        for target in support:
            idx = ids.get((support, target))
            if idx is not None and pressure_target(target):
                pool.add((*pair, idx))
    for pair, s in product(pairs, donors):
        add(direct, pair, s)
    for pair in pairs:
        for i in pair:
            if supports[i] in donors:
                add(exchanged, pair, mirrored[supports[i]])
    reassigned = set()
    for triple in triples:
        for ordered in permutations(triple):
            options = [[ids[s,t] for t in s if eligible_derivative(t, slot) and (s,t) in ids]
                       for s,slot in zip(ordered, archive['variables'])]
            reassigned.update(product(*options))

    # Parity check on any measured pressure fit also present in the archive.
    errors = []
    for i,p in enumerate(points):
        if (j := ids.get((supports[i], targets[i]))) is not None:
            errors.append(np.abs(np.asarray(objectives[j])-objectives[i]) / (1+np.abs(objectives[i])))
    assert errors
    max_error = np.max(errors, axis=0)
    assert np.max(max_error) < 1e-6, max_error

    forms, frequencies = previous.registry('ns', problem)
    allowed = {frozenset(system) for system in forms}
    accepted_laws = set.union(*(set(f) for f in forms))
    tags = {}
    registry_ids = {}
    structure_ids = []
    for s in supports:
        if s not in registry_ids:
            registry_ids[s] = len(registry_ids)
            signature = frozenset(previous.label_key(t, frequencies) for t in s)
            tags[s] = signature if signature in accepted_laws else None
        structure_ids.append(registry_ids[s])
    kind = [tags[s] for s in supports]
    original = {tuple(c) for c in front}
    baseline = sum(frozenset(kind[i] for i in c) in allowed for c in original)
    target_codes = {t:i for i,t in enumerate(sorted(set(targets)))}
    target_ids = [target_codes[t] for t in targets]
    gradient_targets = [code for t, code in target_codes.items() if pressure_target(t)]
    recorded = np.asarray(list(dict.fromkeys(map(tuple, [*front, *archive['systems']]))))
    def describe(i):
        return dict(target=previous.label_key(targets[i], ()),
                    terms=sorted(previous.label_key(t, ()) for t in supports[i]),
                    wape=objectives[i][0], instability=objectives[i][1])
    def evaluate(proposals, count_all=False):
        result = measure(proposals, recorded, objectives, structure_ids, target_ids, gradient_targets,
                         kind, allowed, count_all)
        if result['example']:
            result['example']['system'] = [describe(i) for i in result['example']['point_ids']]
        for witness in result['dominance_witness']:
            witness['candidate_pressure'] = describe(witness['candidate'][2])
            witness['rival_pressure'] = describe(witness['rival'][2])
        return result
    return dict(seed=seed, baseline_complete=baseline, baseline_front=len(front),
                original_velocity_pairs=len(pairs), donor_structures=len(donors),
                recorded_systems=len(recorded), distinct_fits=len(ids), archived_parity_checks=len(errors),
                max_scaled_score_error=max_error.tolist(),
                direct=evaluate(direct), exchange=evaluate(exchanged),
                exchange_and_reassign=evaluate(reassigned, count_all=True),
                combined=evaluate(direct | exchanged | reassigned))


if __name__ == '__main__':
    config = live.R.active_config()
    live.R.set_active_config(replace(config, search_space=replace(config.search_space,
                                 system_rps_mode='plain', equation_terms_max_number=7)))
    problem = live.load_data('ns', **OPTIONS['ns'])
    cache = {}
    runs = []
    for seed in SEEDS:
        runs.append(replay(seed, problem, cache))
        r = runs[-1]
        print(seed, r['max_scaled_score_error'], {a: (r[a]['complete'], r[a]['complete_nondominated'],
              r[a]['unbeaten_with_same_pressure_target']) for a in
              ('direct', 'exchange', 'exchange_and_reassign', 'combined')}, flush=True)
    assert any(r['archived_parity_checks'] for r in runs)
    path = ROOT/'ns_pressure_replay.json'
    temp = path.with_suffix('.pending')
    temp.write_text(json.dumps(runs, indent=2, allow_nan=False), encoding='utf8')
    temp.replace(path)
