"""Replay generic field/axis exchanges on clean fronts. Only evaluate() reads the law registry."""
import json
import re
from collections import Counter
from dataclasses import replace
from functools import cache
from itertools import combinations, permutations, product
from time import monotonic
from types import SimpleNamespace

from epoch_settings import OPTIONS, ROOT, SEEDS, directory, live, np
from compare import previous, read_archive
from ns_pressure_replay import freeze, make_columns, refit


DERIVATIVE = re.compile(r'd(?:\^(\d+))?([A-Za-z_]\w*)/dx(\d+)(?:\^\d+)?')


def encode(term, fields):
    factors = []
    for name, params in term:
        p = dict(params)
        dim = int(p.pop('dim', -1))
        var, order = -1, 0
        if name in fields and dim == -1:
            var, name = fields.index(name), 'field'
        elif m := DERIVATIVE.fullmatch(name):
            var = fields.index(m[2])
            order, dim, name = int(m[1] or 1), int(m[3]), 'field'
        factors.append((name, var, dim, order, tuple(sorted(p.items()))))
    return tuple(sorted(factors))


def decode(term, fields):
    raw = []
    for name, var, dim, order, params in term:
        p = dict(params)
        if name == 'field':
            name = fields[var]
            if order:
                suffix = '' if order == 1 else f'^{order}'
                name = f'd{suffix}{name}/dx{dim}{suffix}'
        elif dim != -1:
            p['dim'] = float(dim)
        raw.append((name, tuple(sorted(p.items()))))
    return tuple(sorted(raw))


def rename_factor(factor, field_map, axis_map):
    op, field, axis, order, params = factor
    return (op, field_map[field] if field >= 0 else field,
            axis_map[axis] if axis >= 0 else axis, order, params)


def swapped(support, field_map, axis_map):
    return frozenset(tuple(sorted(rename_factor(f, field_map, axis_map) for f in term))
                     for term in support)


def exchanges(size):
    maps = [tuple(range(size))]
    for i, j in combinations(range(size), 2):
        order = list(range(size))
        order[i], order[j] = order[j], order[i]
        maps.append(tuple(order))
    return maps


def target_slot(term):
    if len(term) == 1:
        op, field, _, order, params = term[0]
        if op == 'field' and order and dict(params)['power'] == 1:
            return field
    return -1


def propose(parents, vocabulary, nfields, naxes, *, exchange=True):
    """Parents, indexed terms, and available primitive tokens are the only search inputs."""
    sets = {frozenset(parent) for parent in parents}
    if exchange:
        seeds = {s for parent in parents for s in parent}
        maps = list(product(exchanges(nfields), exchanges(naxes)))
        variants = {}
        for s in seeds:
            edits = {swapped(s, fm, am) for fm, am in maps}
            variants[s] = [e for e in edits if e != s and all(f in vocabulary for t in e for f in t)]
        for parent in parents:
            for outgoing in range(len(parent)):
                kept = parent[:outgoing] + parent[outgoing+1:]
                for source in kept:
                    sets.update(frozenset((*kept, e)) for e in variants[source] if e not in kept)
    slots = {s: {target_slot(t) for t in s} for group in sets for s in group}
    assignments = set()
    for group in sets:
        if len(group) == nfields:
            for ordered in permutations(group):
                if all(i in slots[s] for i,s in enumerate(ordered)):
                    assignments.add(ordered)
    return assignments


def verify_anonymous(parents, assignments, vocabulary, nfields, naxes, *, exchange=True):
    """Scramble both index spaces, rerun proposals, and compare the complete candidate sets."""
    fm = tuple((v+1) % nfields for v in range(nfields))
    am = tuple((a-1) % naxes for a in range(naxes))
    @cache
    def rename(s):
        return swapped(s, fm, am)
    def reorder(eq):
        return tuple(rename(eq[fm.index(i)]) for i in range(nfields))
    vocab = {rename_factor(factor, fm, am) for factor in vocabulary}
    after = propose([reorder(p) for p in parents], vocab, nfields, naxes, exchange=exchange)
    assert len(assignments) == len(after) and all(reorder(a) in after for a in assignments)


def budgeted_fits(missing, limit):
    """Admit whole, label-independent complexity groups in increasing order."""
    if limit is None or len(missing) <= limit:
        return missing
    groups = {}
    for key in missing:
        s, pivot = key
        complexity = (len(s), sum(map(len, s)), sum(f[3] for t in s for f in t), pivot[0][3])
        groups.setdefault(complexity, set()).add(key)
    selected = set()
    for _, group in sorted(groups.items()):
        if len(group) <= limit - len(selected):
            selected.update(group)
    return selected


def archive_terms(archive):
    fields = archive['variables']
    labels = [encode(freeze(t), fields) for t in archive['labels']]
    assert [decode(t, fields) for t in labels] == list(map(freeze, archive['labels']))
    supports = [frozenset(labels[j] for j,_ in p[2]) for p in archive['points']]
    return labels, supports, {f for t in labels for f in t}


def fit_pairs(archive, data, bank, labels, supports, needed, max_new_fits):
    # Reuse measured scores for an identical support and target on the same weak rows.
    fields, points = archive['variables'], archive['points']
    fits = {}
    handed_over = sorted({i for system in archive['systems'] for i in system})
    for i in handed_over:
        p = points[i]
        key = (supports[i], labels[p[1]])
        if key in needed:
            fits.setdefault(key, i)
    cached = len(fits)
    missing = budgeted_fits(needed - fits.keys(), max_new_fits)
    print(archive['seed'], len(needed), 'support-target pairs,', len(missing), 'new fits', flush=True)
    plan = {k:tuple(v) for k,v in archive['plan'].items()}
    columns = bank.setdefault(tuple(sorted(plan.items())), {})
    terms = {t:decode(t, fields) for s,pivot in missing for t in s}
    make_columns(data, data.data, plan, set(terms.values()), columns)
    scores = [p[3:5] for p in points]
    raw_support = {s:frozenset(terms[t] for t in s) for s,t in missing}
    for s,t in sorted(missing, key=lambda q:(tuple(sorted(q[0])),q[1])):
        score = refit(raw_support[s], terms[t], columns, plan['shape'])
        if score is not None:
            fits[s,t] = len(scores)
            supports.append(s)
            scores.append(score)
    cost = dict(requested_fits=len(needed), archived_fits=cached, new_fits=len(missing),
                accepted_fits=len(fits), over_budget_fits=len(needed)-cached-len(missing))
    return fits, scores, cost


def generate(archive, data, bank, *, exchange=True, max_new_fits=None):
    labels, supports, vocabulary = archive_terms(archive)
    parents = [tuple(supports[i] for i in row) for row in archive['native_front']]
    nfields, naxes = len(archive['variables']), len(data.shape)
    assignments = propose(parents, vocabulary, nfields, naxes, exchange=exchange)
    verify_anonymous(parents, assignments, vocabulary, nfields, naxes, exchange=exchange)
    needed = {(s, t) for row in assignments for slot,s in enumerate(row)
              for t in s if target_slot(t) == slot}
    fits, scores, cost = fit_pairs(archive, data, bank, labels, supports, needed, max_new_fits)
    proposals = set()
    for row in assignments:
        options = [[fits[s,t] for t in s if target_slot(t) == slot and (s,t) in fits]
                   for slot,s in enumerate(row)]
        proposals.update(product(*options))
    cost.update(parent_front=len(parents), assignments=len(assignments),
                relabelled_assignments_verified=len(assignments))
    return sorted(proposals), supports, scores, cost


def evaluate(problem, archive, proposals, supports, scores, *, all_handovers=False, placement_counts=False):
    # Numeric dominance uses the same structure exemption as NSGA-II. Check all
    # archived handovers and all admitted proposals, including the original hits.
    fields = archive['variables']
    forms, frequencies = previous.registry(problem.name, problem)
    allowed = {frozenset(form) for form in forms}
    laws = sorted(set().union(*allowed), key=sorted)
    codes = {law:i for i,law in enumerate(laws)}
    accepted = {tuple(sorted(codes[law] for law in f)) for f in allowed}
    kinds = []
    identities = {}
    for s in supports:
        if s not in identities:
            identities[s] = len(identities)
            law = frozenset(previous.label_key(decode(t, fields), frequencies) for t in s)
            kinds.append(codes.get(law, -1))
    structure_ids = np.array([identities[s] for s in supports])
    tags = np.array(kinds)[structure_ids]
    def correct(rows):
        all_tags = np.sort(tags[rows], axis=1)
        return np.logical_or.reduce([np.all(all_tags == form, axis=1) for form in accepted])
    original = np.asarray(archive['systems'])
    front = np.asarray(archive['native_front'])
    new = np.asarray(proposals, dtype=int).reshape(-1, len(fields))
    front_hits = front[correct(front)]
    old_hits = original[correct(original)] if all_handovers else front_hits
    proposed_hits = new[correct(new)]
    scores = np.asarray(scores)
    all_rows = np.concatenate([original, new])
    survivors = []
    hits = sorted(set(map(tuple, proposed_hits)) | set(map(tuple, old_hits)))
    for row in hits:
        idx = np.asarray(row)
        beaten = False
        for start in range(0, len(all_rows), 16384):
            batch = all_rows[start:start+16384]
            same = structure_ids[batch] == structure_ids[idx]
            rival = scores[batch]
            weak = np.all(same | np.all(rival <= scores[idx], axis=2), axis=1)
            strict = np.any(~same & np.any(rival < scores[idx], axis=2), axis=1)
            if np.any(weak & strict):
                beaten = True
                break
        if not beaten:
            survivors.append(row)
    result = dict(recorded_systems=len(original), proposed_systems=len(new),
                  original_front_complete=len(front_hits), proposed_complete=len(proposed_hits),
                  complete_nondominated=len(survivors))
    if all_handovers:
        result.update(native_complete=len(old_hits), complete_generated=len(hits),
                      first_survivor=[int(i) for i in survivors[0]] if survivors else None)
    if placement_counts:
        # The NS registry lists both momentum laws and continuity in that order,
        # including for the identity-equivalent forms. Classify only after scoring.
        roles = {}
        for form in forms:
            for law, role in zip(form, ('X momentum', 'Y momentum', 'Continuity'), strict=True):
                assert roles.setdefault(codes[law], role) == role
        counts = Counter(tuple(roles[tags[i]] for i in row) for row in hits)
        unbeaten = Counter(tuple(roles[tags[i]] for i in row) for row in survivors)
        result['complete_placements'] = [dict(laws=order, generated=n, nondominated=unbeaten[order])
                                         for order,n in sorted(counts.items())]
    return result


if __name__ == '__main__':
    # These names select a benchmark fixture only. The generator receives opaque field/axis IDs.
    name = 'ns'
    problem = live.load_data(name, **OPTIONS[name])
    data = SimpleNamespace(data=problem.data, grids=problem.grids, shape=problem.shape, named_arrays={})
    column_banks, results = {}, []
    for seed in SEEDS:
        started = monotonic()
        archive = read_archive(directory(name, seed, 0))
        assert archive['train_hashes'] == {k:live.R.digest(v) for k,v in data.data.items()}
        cfg = live.R.active_config()
        width = archive['settings']['equation_terms_max_number']
        live.R.set_active_config(replace(cfg, search_space=replace(cfg.search_space,
                                     system_rps_mode='plain', equation_terms_max_number=width)))
        systems, supports, scores, cost = generate(archive, data, column_banks)
        result = dict(seed=seed, **cost, **evaluate(problem, archive, systems, supports, scores))
        results.append(result)
        print(json.dumps(result), round(monotonic()-started, 2), 'seconds', flush=True)
    path = ROOT/'anonymous_exchange.json'
    temp = path.with_suffix('.pending')
    temp.write_text(json.dumps(results, indent=2), encoding='utf8')
    temp.replace(path)
