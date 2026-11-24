"""Replay archived fits, reconstruct both budget prefixes and audit the observed swaps."""
import argparse
import csv
import gzip
import json
from concurrent.futures import ProcessPoolExecutor
from itertools import permutations

from swap_trial import ARMS, CONFIRMATION, folder, live, np
import assess as previous
from misses import peak_grid


def read_archive(dest):
    with gzip.open(dest / 'archive.json.gz', 'rt', encoding='utf8') as stream:
        return json.load(stream)


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf8') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)


def system_front(values):
    """All nondominated rows, including ties. Unique rows are ordered lexicographically."""
    if len(values) == 0:
        return np.zeros(0, dtype=bool)
    unique, inverse = np.unique(values, axis=0, return_inverse=True)
    keep = np.zeros(len(unique), dtype=bool)
    front = np.empty_like(unique)
    size = 0
    for i, row in enumerate(unique):
        if not np.any(np.all(front[:size] <= row, axis=1)):
            keep[i] = True
            front[size] = row
            size += 1
    return keep[inverse.ravel()]


def swap_audit(events, labels, forms):
    tally = dict(suitable=0, right_victim=0, screened=0, shortlisted=0, selected=0, first_selected=None,
                 best_useful_rank=None, mean_useful_draw_probability=None)
    useful_mass = 0.
    for event in events:
        if event['target'] is None:
            continue
        target = labels[event['target']]
        parent = {target, *(labels[j] for j in event['eligible_parent'])}
        swaps = {(next(iter(parent-form)), next(iter(form-parent))) for form in forms
                 if target in form and len(parent-form) == len(form-parent) == 1}
        if not swaps:
            continue
        tally['suitable'] += 1
        replacements = {new for old, new in swaps if old == labels[event['outgoing']]}
        if not replacements:
            continue
        tally['right_victim'] += 1
        if event['path'] in ('guided', 'empty_shortlist'):
            tally['screened'] += 1
        shortlist = event['shortlist']
        useful = [(rank,gain) for rank,(j,gain) in enumerate(shortlist,1) if labels[j] in replacements]
        if useful:
            tally['shortlisted'] += 1
            tally['best_useful_rank'] = min(useful[0][0], tally['best_useful_rank'] or useful[0][0])
            useful_mass += sum(g for rank,g in useful) / sum(g for j,g in shortlist)
        if event['changed'] and labels[event['selected']] in replacements:
            tally['selected'] += 1
            if tally['first_selected'] is None:
                tally['first_selected'] = event['after_call']
    if tally['screened']:
        tally['mean_useful_draw_probability'] = useful_mass / tally['screened']
    return tally


def audit_fate(fate, audit):
    if fate != 'never scored':
        return fate
    if not audit['suitable']:
        return 'no suitable swap parent'
    if not audit['right_victim']:
        return 'wrong outgoing term'
    if audit['selected']:
        return 'completion proposed but law never scored'
    if audit['shortlisted']:
        return 'shortlisted but not selected'
    if audit['screened']:
        return 'replacement outside shortlist'
    return 'ordinary draw did not complete'


def coefficient_errors(problem, frequencies, labels, points):
    # Orient both sides on the recorded target, so the -1 pivot agrees.
    references = {}
    for system in problem.truth_systems:
        for law, equation in enumerate(system):
            target, terms, weights = live.R.parse(equation)
            vector = {previous.label_key(t, frequencies): w for t,w in zip(terms, weights)}
            vector[previous.label_key(target, frequencies)] = -1.
            references.setdefault((law, frozenset(vector)), []).append(vector)
    def error(law, i):
        _, pivot, pairs, *_ = points[i]
        observed = {labels[j]: w for j,w in pairs}
        target = labels[pivot]
        predictors = sorted(set(observed) - {target})
        errors = []
        for truth in references.get((law, frozenset(observed)), []):
            ref = np.array([-truth[t]/truth[target] for t in predictors])
            errors.append(float(np.linalg.norm([observed[t] for t in predictors] - ref) / np.linalg.norm(ref)))
        return min(errors, default=None)
    return error


def assess_run(job, *, destination=None, moves='swaps', audit_moves=swap_audit, classify=audit_fate,
               comparator='native', timing='swap_seconds'):
    name, seed, arm = job
    dest = destination or folder(*job)
    rec = read_archive(dest)
    problem = live.load_data(name)
    source, held, dyad, valid, error = previous.replay(problem, problem.noisy(5,seed), rec)
    allowed, freq = previous.registry(name, problem)
    labels = [previous.label_key(t, freq) for t in rec['labels']]
    points = rec['points']
    slots = np.array([p[0] for p in points])
    first = np.array([p[5] for p in points])
    supports = [frozenset(labels[j] for j,w in p[2]) for p in points]
    variables = rec['variables']
    combos = np.asarray(rec['systems'], dtype=int).reshape(-1, len(variables))
    legal = {perm for row in allowed for perm in permutations(row)}
    actual = np.array([tuple(supports[i] for i in c) in legal for c in combos], dtype=bool)
    op = rec['operator']
    clock = np.array(op['full_cost'])
    assert (np.diff(clock) >= 1).all() and len(clock) == live.BUDGET
    assert clock[-1] <= rec['calls'] + op['extra_fits']
    handovers = np.array(op['system_cost'])
    assert len(combos) == len(handovers) and (clock[first[combos].max(axis=1)-1] <= handovers).all()
    traces = [previous.label_key(t, freq) for t in op['trace_labels']] if moves else []
    coefficient_error = coefficient_errors(problem, freq, labels, points)
    result = dict(system=name, seed=seed, arm=arm, replay_error=error, seconds=rec['seconds'],
                  extra_fits=op['extra_fits'], gain_columns=op['gain_columns'], **{timing:op[timing]},
                  work=op['work'], hashes=op['hashes'], windows=rec['plan'], train_hashes=rec['train_hashes'],
                  settings=rec['settings'], trig=rec['trig'], peaks=peak_grid((dest/'search.log').read_text()))
    if arm == comparator and seed in live.SEEDS:
        baseline = live.folder(name, seed, arm)
        original = read_archive(baseline)
        for key in ('settings','axis','train_hashes','plan','trig','calls','variables','labels','points','systems','system_events'):
            if original[key] != rec[key]:
                raise ValueError(f'{name}/{seed}: {arm} archive differs in {key}')
        assert result['peaks'] == peak_grid((baseline/'search.log').read_text())
        result[comparator+'_archive_verified'] = True

    audit_records = []
    for budget in ('full', 'total'):
        in_budget = np.ones(len(points), dtype=bool) if budget == 'full' else clock[first-1] <= live.BUDGET
        combined = np.ones(len(combos), dtype=bool) if budget == 'full' else handovers <= live.BUDGET
        calls = live.BUDGET if budget == 'full' else int(np.searchsorted(clock, live.BUDGET, side='right'))
        system = dict(assembled=int(np.any(combined & actual)), candidates=int(combined.sum()),
                      calls=calls, total_fits=rec['calls'] + op['extra_fits'] if budget=='full' else live.BUDGET)
        if moves:
            events = [e for e in op[moves] if e['after_call'] < live.BUDGET and
                      (budget == 'full' or e['after_fit_cost'] <= live.BUDGET)]
            system.update({moves:len(events)}, guided=sum(e['path']=='guided' for e in events))
        masks = {}
        for cohort, scores in [('source',source), ('held',held)]:
            mask = np.zeros(len(points), dtype=bool)
            eligible = valid & in_budget & np.isfinite(scores) & (scores <= 1.)
            for slot in variables:
                ids = np.flatnonzero(eligible & (slots == slot))
                mask[previous.nondominated(ids, scores, dyad)] = True
            masks[cohort] = mask
            ids = np.flatnonzero(combined & eligible[combos].all(axis=1))
            values = np.column_stack((scores[combos[ids]], dyad[combos[ids]]))
            members = ids[system_front(values)]
            system[cohort] = int(actual[members].any())
            system[cohort+'_system_front_size'] = len(members)
            system[cohort+'_law_front_size'] = int(mask.sum())
            if name == 'ode_simple':
                from ode_identities import identity
                def symbolic(i, snap):
                    factors = tuple(tuple((n,tuple(map(tuple,pars))) for n,pars in rec['labels'][j])
                                    for j,w in points[i][2])
                    return identity(factors, snap)
                for snap,tag in ((False,'literal'),(True,'registered_frequency')):
                    system[cohort+'_identity_'+tag] = int(any(symbolic(i,snap) for i in combos[members].ravel()))
                system[cohort] |= system[cohort+'_identity_registered_frequency']
        laws = []
        for law in range(len(allowed[0])):
            forms = {row[law] for row in allowed}
            match = in_budget & np.array([s in forms for s in supports])
            ids = np.flatnonzero(match)
            on_held = np.flatnonzero(match & masks['held'])
            first_id = min(ids, key=first.__getitem__, default=None)
            best_id = min(on_held, key=held.__getitem__, default=None)
            row = dict(law=law, generated=int(len(ids)>0), first=int(first[first_id]) if len(ids) else None,
                       first_total=int(clock[first[first_id]-1]) if len(ids) else None,
                       source=int(np.any(match & masks['source'])), held=int(len(on_held)>0),
                       first_coefficient_error=coefficient_error(law, first_id) if len(ids) else None,
                       held_coefficient_error=coefficient_error(law, best_id) if len(on_held) else None)
            usable = match & valid & np.isfinite(held) & (held <= 1.)
            if not len(ids):
                fate = 'never scored'
            elif not (match & valid).any():
                fate = 'failed native eligibility'
            elif not usable.any():
                fate = 'outside held WAPE cutoff'
            else:
                fate = 'on front' if len(on_held) else 'dominated'
            row['held_fate'] = fate
            if moves:
                audit = audit_moves(events, traces, {form - {'1'} for form in forms})
                row.update(**audit, trace_fate=classify(fate,audit))
            laws.append(row)
            # Record every observed dominator of a lost law on its held-out slot front.
            if fate == 'dominated':
                for i in np.flatnonzero(usable):
                    dom = in_budget & valid & (slots==slots[i]) & np.isfinite(held) & (held<=held[i]) & (dyad<=dyad[i])
                    dom &= (held<held[i]) | (dyad<dyad[i])
                    js = np.flatnonzero(dom)
                    assert len(js)
                    audit_records.append(dict(budget=budget, law=law, point=int(i), dominators=js.tolist()))
        result[budget] = dict(system=system, laws=laws)
    def describe(i):
        slot, target, pairs, wape, stability, first_call = points[i]
        return dict(slot=slot, target=rec['labels'][target], coefficients=[[rec['labels'][j],w] for j,w in pairs],
                    source_wape=wape, held_wape=held[i], dyad=stability, first=first_call)
    used = {j for row in audit_records for j in [row['point'], *row['dominators']]}
    with gzip.open(dest/'dominators.json.gz','wt',encoding='utf8') as stream:
        json.dump(dict(cases=audit_records, points={i:describe(i) for i in sorted(used)}), stream, allow_nan=False)
    with (dest/'assessment.json').open('w',encoding='utf8') as stream:
        json.dump(result,stream,allow_nan=False)
    return job


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workers',type=int,default=3)
    group = p.add_mutually_exclusive_group()
    group.add_argument('--seeds',nargs='+',type=int,default=live.SEEDS)
    group.add_argument('--confirmation',action='store_true')
    p.add_argument('--refresh',action='store_true')
    args = p.parse_args()
    seeds = CONFIRMATION if args.confirmation else args.seeds
    jobs = [(s,n,a) for s in live.DATASETS for n in seeds for a in ARMS]
    todo = [j for j in jobs if (folder(*j)/'archive.json.gz').exists() and
            (args.refresh or not (folder(*j)/'assessment.json').exists())]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for job in pool.map(assess_run,todo):
            print(*job,flush=True)
