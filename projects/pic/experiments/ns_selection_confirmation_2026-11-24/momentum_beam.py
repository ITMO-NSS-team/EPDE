"""Audit native momentum discovery and replay the fill beam on its original weak rows."""
from collections import Counter
from contextlib import ExitStack
import json
from types import SimpleNamespace
from unittest.mock import patch

from discover import ROOT, trial, whole_record
from epoch_settings import OPTIONS, live, np
from compare import previous, read_archive
from epde.operators.common import fill_prune as fp


SEED = 24112402
CALLS = 5000


class PrefixComplete(BaseException):
    pass


def retention(archive, momentum, continuity, occupied_epochs):
    points = archive['points']
    supports = [frozenset(j for j,_ in p[2]) for p in points]
    lookup = {s:i for i,s in enumerate(dict.fromkeys(supports))}
    structures = np.array([lookup[s] for s in supports])
    scores = np.array([p[3:5] for p in points])
    def hits(rows):
        return [r for r in rows if any(i in momentum for i in r)]
    def beaten(row, rivals):
        row = np.asarray(row)
        same = structures[rivals] == structures[row]
        weak = np.all(same | np.all(scores[rivals] <= scores[row], axis=2), axis=1)
        strict = np.any(~same & np.any(scores[rivals] < scores[row], axis=2), axis=1)
        return bool(np.any(weak & strict))
    all_hits = hits(archive['systems'])
    parents = np.asarray(archive['epoch_fronts'][99]['members'])
    result = dict(with_continuity=sum(any(i in continuity for i in r) for r in all_hits),
                  dominated_by_epoch100=sum(beaten(r, parents) for r in all_hits))
    if occupied_epochs and max(occupied_epochs) < len(archive['epoch_fronts']):
        last = max(occupied_epochs)
        last_hits = hits(archive['epoch_fronts'][last-1]['members'])
        rivals = np.asarray(archive['epoch_fronts'][last]['members'])
        result.update(last_epoch_hits=len(last_hits), beaten_by_next_front=sum(beaten(r, rivals) for r in last_hits))
    return result


def run():
    archive = read_archive(ROOT/'results'/str(SEED))
    problem = live.load_data('ns', **OPTIONS['ns'])
    assert archive['train_hashes'] == {k:live.R.digest(v) for k,v in problem.data.items()}
    forms, freq = previous.registry('ns', problem)
    pivots = {'u': 'du/dx0', 'v': 'dv/dx0'}
    alternatives = {slot: sorted({f[i] for f in forms}, key=sorted) for i,slot in enumerate(pivots)}
    labels = [previous.label_key(t, freq) for t in archive['labels']]
    pts = archive['points']
    supports = [frozenset(labels[j] for j,_ in p[2]) for p in pts]
    continuity = {i for i,s in enumerate(supports) if any(s == form[2] for form in forms)}
    history, partners = {}, {}
    for k, slot in enumerate(pivots):
        ids = {i for i,s in enumerate(supports) if s in alternatives[slot]}
        fronts = [h['epoch'] for h in archive['epoch_fronts'] if any(i in ids for row in h['members'] for i in row)]
        by_partner = Counter(tuple(labels[pts[i][1]] for j,i in enumerate(row) if j != k)
                             for row in archive['systems'] if labels[pts[row[k]][1]] == pivots[slot])
        partners[slot], count = by_partner.most_common(1)[0]
        history[slot] = dict(first_call=min((pts[i][5] for i in ids), default=None), fitted_equations=len(ids),
                             containing_systems=sum(any(i in ids for i in r) for r in archive['systems']),
                             front_epochs=fronts, time_target_systems=by_partner.total(),
                             common_partner_targets=partners[slot], common_partner_systems=count,
                             retention=retention(archive, ids, continuity, fronts))

    term_observer = live.Observer(archive['variables'])
    def label(t):
        return previous.label_key(term_observer.labels[term_observer.term(t)], freq)

    stores, names, stats = {}, {}, {v: Counter() for v in pivots}
    missed_shortlists = {v: [] for v in pivots}
    active = None
    missed_request = None
    observer = None
    checked_plans = []
    native_plan = live.training_plan
    def plan(*args, **kwargs):
        p = native_plan(*args, **kwargs)
        signature = {key:list(getattr(p, key)) for key in archive['plan']}
        assert signature == archive['plan']
        checked_plans.append(signature)
        return p

    levels = fp._beam_levels
    native_candidates = fp.candidates
    current_partners = set()
    def watch_candidates(eq, target, partner_labels, gs, t):
        nonlocal current_partners
        current_partners = partner_labels
        return native_candidates(eq, target, partner_labels, gs, t)

    def watch_levels(gs, var, target_label, cand, sig, t, K, width, top, levels_mode):
        nonlocal active, missed_request
        assert (width, top, levels_mode) == (1000, 20, 'all')
        hit = levels(gs, var, target_label, cand, sig, t, K, width, top, levels_mode)
        active = None
        missed_request = None
        if var not in pivots:
            return hit
        if var not in stores:
            stores[var] = gs
            names[var] = [label(term) for term in gs['lib']]
        assert stores[var] is gs
        stats[var]['beam_requests_any_target'] += 1
        if names[var][t] == pivots[var]:
            stats[var]['beam_requests_time_target'] += 1
            active = var
            if hit[0] == 'ok':
                shortlist = hit[1].get(5, [])
                if any((frozenset(names[var][j] for j in s) | {pivots[var]}) in alternatives[var] for s,_ in shortlist):
                    stats[var]['exact_available_at_size_five'] += 1
                else:
                    available = {names[var][j] for j in cand} | {pivots[var]}
                    missed_request = dict(max_predictors=int(K),
                        partner_targets=sorted(names[var][gs['index'][p]] for p in current_partners),
                        missing_from_candidate_library=[sorted(s - available) for s in alternatives[var]])
                    missed_shortlists[var].append(missed_request)
                    if all(s <= available for s in alternatives[var]):
                        # Keep this actual candidate mask, its Gram and target fixed. No RNG calls.
                        gram = fp.Gram(gs['Gn'][np.ix_(cand, cand)], gs['Gn'][cand,t], 1.)
                        local = {names[var][j]:i for i,j in enumerate(cand)}
                        laws = [{local[name] for name in law-{pivots[var]}} for law in alternatives[var]]
                        sweep = {}
                        for w in (width, width*2, width*4):
                            levels_by_size = fp.beam_search(gram, w, min(5, K))
                            sweep[w] = [[next((i for i,(s,_) in enumerate(level, 1) if law.issuperset(s)), None)
                                         for level in levels_by_size.values()] for law in laws]
                        missed_request['best_true_subset_ranks_by_size_and_width'] = sweep
        return hit

    draw = fp.draw_fill
    def watch_draw(top):
        pos = draw(top)
        if active:
            support = frozenset(names[active][j] for j in top[pos][0]) | {pivots[active]}
            stats[active]['time_target_draws'] += 1
            if missed_request is not None:
                missed_request['drawn_predictors'] = sorted(support-{pivots[active]})
            if any(s <= support for s in alternatives[active]):
                stats[active]['draw_containing_law'] += 1
        return pos

    install = fp._install
    def watch_install(eq, target, terms):
        result = install(eq, target, terms)
        slot = eq.main_var_to_explain
        if slot in pivots and label(target) == pivots[slot]:
            if frozenset(label(t) for t in eq.structure) in alternatives[slot]:
                stats[slot]['exact_installs'] += 1
        return result

    evaluated = trial.Trial.evaluated
    def stop_at(meter, eq):
        nonlocal observer
        evaluated(meter, eq)
        if meter.calls == CALLS:
            observer = meter
            raise PrefixComplete

    with ExitStack() as stack:
        for obj, attr, repl in [(live, 'split', whole_record), (live, 'training_plan', plan),
                                (fp, 'candidates', watch_candidates), (fp, '_beam_levels', watch_levels),
                                (fp, 'draw_fill', watch_draw),
                                (fp, '_install', watch_install), (trial.Trial, 'evaluated', stop_at)]:
            stack.enter_context(patch.object(obj, attr, repl))
        try:
            trial.worker('ns', SEED, 'candidate', noise_level=0, epoch_budget=150, loader_options=OPTIONS['ns'])
        except PrefixComplete:
            pass
    assert observer is not None and len(checked_plans) == 1
    ref_points = [p for p in pts if p[5] <= CALLS]
    matched = json.loads(json.dumps(observer.points)) == ref_points
    matched &= json.loads(json.dumps(observer.labels)) == archive['labels'][:len(observer.labels)]
    print('Identical archive prefix:', matched, 'unique points:', len(ref_points), flush=True)

    probes = {}
    for slot, target in pivots.items():
        gs = stores[slot]
        ix = {name:i for i,name in enumerate(names[slot])}
        t = ix[target]
        term = gs['lib'][t]
        eq = SimpleNamespace(main_var_to_explain=slot, _plain_genes={'target': term.factors_labels})
        partner_labels = {gs['labels'][ix[name]] for name in partners[slot]}
        cand, _ = fp.candidates(eq, term, partner_labels, gs, t)
        gram = fp.Gram(gs['Gn'][np.ix_(cand, cand)], gs['Gn'][cand,t], 1.)
        beam = fp.beam_search(gram, 1000, 6)
        level = [(frozenset(names[slot][cand[j]] for j in s) | {target}, rss) for s,rss in beam[5]]
        per_form = []
        for law in alternatives[slot]:
            rank = next((i for i,(s,_) in enumerate(level, 1) if s == law), None)
            per_form.append(dict(terms=sorted(law), rank_at_five=rank,
                                 all_predictors_admissible=all(ix[n] in cand for n in law-{target})))
        probes[slot] = dict(candidates=int(len(cand)), target=target, partner_targets=partners[slot],
                            beam_width=1000, draw_from_best=20, forms=per_form)
        print(slot, 'five-term ranks', [f['rank_at_five'] for f in per_form], flush=True)
    result = dict(seed=SEED, archive_calls=archive['calls'], max_equation_terms=archive['settings']['equation_terms_max_number'],
                  archive_history=history, probe=probes, replay_calls=CALLS,
                  identical_points_and_labels=matched, replay_stats=stats,
                  missed_five_term_shortlists=missed_shortlists)
    out = ROOT/'results'/str(SEED)/'momentum_beam.json'
    pending = out.with_suffix('.pending')
    pending.write_text(json.dumps(result, indent=2), encoding='utf8')
    pending.replace(out)
    print(json.dumps(stats), flush=True)


if __name__ == '__main__':
    run()
