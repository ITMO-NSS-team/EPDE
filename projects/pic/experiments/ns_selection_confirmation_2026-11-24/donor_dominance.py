"""Trace the native Pareto dominators of the twelve archived historical-donor hits."""
from collections import Counter
from dataclasses import replace
import json
import sys
from types import SimpleNamespace

from donor_history import (CHECKPOINT, OPTIONS, ROOT, blind_input, decode, live, matching_systems,
                           native_points, np, previous, proposal_groups, read_archive)
from anonymous_exchange import encode, freeze


def audit():
    dest = ROOT/'results'/'24112401'
    archived = json.loads((dest/'donor_history.json').read_text())['historical']
    archive = read_archive(dest)
    fields = archive['variables']
    problem = live.load_data('ns', **OPTIONS['ns'])
    assert archive['train_hashes'] == {k:live.R.digest(v) for k,v in problem.data.items()}
    cfg = live.R.active_config()
    live.R.set_active_config(replace(cfg, search_space=replace(cfg.search_space,
                             system_rps_mode='plain', equation_terms_max_number=7)))
    fronts = archive['epoch_fronts'][:CHECKPOINT]
    clock = archive['operator']['epoch_clock'][CHECKPOINT-1]
    prefix = blind_input(archive, clock, fronts[-1]['members'])
    parents = {tuple(row) for h in fronts for row in h['members']}
    data = SimpleNamespace(data=problem.data, grids=problem.grids, shape=problem.shape, named_arrays={})
    new_targets = {}
    groups, ps, pm, work = proposal_groups(prefix, parents, data, {}, archived['total_fits']-clock['cost'],
                                           point_targets=new_targets)
    assert work['new_fits'] == archived['new_fits']
    ns, nm = native_points(archive)
    n = len(prefix['points'])
    offset = len(ns)-n
    groups = [(slots, np.where(p>=n,p+offset,p), np.where(t>=n,t+offset,t)) for slots,p,t in groups]
    supports = ns + ps[n:]
    scores = np.asarray(nm + pm[n:])
    labels = [encode(freeze(t), fields) for t in archive['labels']]
    targets = [labels[p[1]] for p in archive['points']] + [new_targets[i] for i in range(n,len(ps))]
    costs = np.asarray(archive['operator']['system_cost'])
    accepted = costs <= archived['native_fits']
    native = np.asarray(archive['systems'])[accepted]
    costs = costs[accepted]
    ids, native_hits, hits = matching_systems(problem, fields, supports, native, groups)
    assert len(native_hits) == 0 and len(hits) == archived['complete_generated'] == 12
    forms, freqs = previous.registry(problem.name, problem)
    roles = {s:role for form in forms for s,role in zip(form, ('X momentum','Y momentum','Continuity'))}
    def label(t):
        return previous.label_key(decode(t, fields), freqs)
    def equation(i):
        terms = sorted(label(t) for t in supports[i])
        return dict(role=roles.get(frozenset(terms)), target=label(targets[i]), terms=terms,
                    wape=float(scores[i,0]), instability=float(scores[i,1]))
    native_scores = scores[native]
    native_ids = ids[native]
    results = []
    for hit in sorted(hits):
        idx = np.asarray(hit)
        changed = native_ids != ids[idx]
        no_worse = np.all(~changed | np.all(native_scores <= scores[idx],axis=2), axis=1)
        improves = np.any(changed & np.any(native_scores < scores[idx],axis=2), axis=1)
        rivals = np.flatnonzero(no_worse & improves)
        assert len(rivals)
        hist = Counter(tuple(field for field,diff in zip(fields,mask) if diff) for mask in changed[rivals])
        witness = min(rivals, key=lambda i:(int(changed[i].sum()), int(costs[i]), int(i)))
        row = native[witness]
        details = []
        for k in np.flatnonzero(changed[witness]):
            details.append(dict(slot=fields[k], candidate=equation(idx[k]), rival=equation(row[k])))
        results.append(dict(candidate_ids=list(map(int,hit)), candidate=[equation(i) for i in hit],
                            native_dominators=len(rivals), changed_slots={','.join(k):v for k,v in sorted(hist.items())},
                            witness_ids=list(map(int,row)), witness_cost=int(costs[witness]), changes=details))
    result = dict(seed=archive['seed'], native_cutoff=archived['native_fits'], native_systems=len(native),
                  witness_rule='fewest changed slot structures, then earliest native handover cost',
                  candidates=results)
    path = dest/'donor_dominance.json'
    pending = path.with_suffix('.pending')
    pending.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf8')
    pending.replace(path)
    print(f'{len(results)} complete candidates audited against {len(native)} native systems.', flush=True)


def summarize():
    dest = ROOT/'results'/'24112401'
    result = json.loads((dest/'donor_dominance.json').read_text())
    archive = read_archive(dest)
    rows = result['candidates']
    assert all([d['slot'] for d in r['changes']] == ['p'] for r in rows)
    witness_ids = sorted({r['witness_ids'][2] for r in rows})
    n_rivals = [r['native_dominators'] for r in rows]
    text = ['# Why the 12 complete systems lost on seed 24112401', '',
            f"Each complete system had {min(n_rivals)}–{max(n_rivals)} dominators among "
            f"{result['native_systems']:,} native handovers at the historical-donor arm’s fit cutoff. "
            'Every candidate had a rival sharing the same two velocity-slot equation structures. '
            'In the pressure slot the rival fitted a pressure–Poisson relation and improved both objectives. '
            'EPDE skips the objectives of structurally identical slots in this comparison.', '',
            '| Candidate fit IDs (u, v, p) | Momentum law in pressure slot | Rival pressure fit ID | Candidate WAPE | Rival WAPE | Candidate instability | Rival instability |',
            '|---|---|---:|---:|---:|---:|---:|']
    for row in rows:
        change, = row['changes']
        c, r = change['candidate'], change['rival']
        assert r['wape'] < c['wape'] and r['instability'] < c['instability']
        text.append(f"| {', '.join(map(str,row['candidate_ids']))} | {c['role']} | {row['witness_ids'][2]} | "
                    f"{c['wape']:.6g} | {r['wape']:.6g} | {c['instability']:.6g} | {r['instability']:.6g} |")
    text.extend(['', 'The table selects the native rival with the fewest changed structures and then the '
                 'earliest handover. Each column below gives the coefficients of a fitted pressure-slot relation '
                 'whose weighted sum of terms approximates zero. Subscripts denote derivatives.', ''])
    named = {}
    for i in witness_ids:
        named[i] = {previous.label_key(archive['labels'][j], set()):w for j,w in archive['points'][i][2]}
    text.append('| Term | ' + ' | '.join(f'Fit {i}' for i in witness_ids)+' |')
    text.append('|---|' + '---:|'*len(witness_ids))
    notation = {'d^2p/dx1^2':'pᵧᵧ', 'd^2p/dx2^2':'pₓₓ', 'du/dx1*dv/dx2':'uᵧ vₓ',
                'du/dx2*dv/dx1':'uₓ vᵧ', 'du/dx2^2':'(uₓ)²'}
    for term in sorted(set().union(*(set(w) for w in named.values()))):
        text.append(f'| {notation.get(term,term)} | ' + ' | '.join(f'{w[term]:.6f}' if term in w else '—' for w in named.values()) + ' |')
    text.extend(['', 'The two supports agree under incompressibility, which gives v_y = −u_x. The pressure–Poisson '
                 'equation follows by taking the spatial divergence of momentum. Replacing one momentum equation '
                 'by this relation can lose information about acceleration. For example, with spatially constant '
                 'pressure and velocity (u(t), 0), continuity, Y momentum and pressure–Poisson all hold, even when '
                 'u changes in time and the X-momentum equation fails. This is an equation-level example without '
                 'additional boundary conditions.', '',
                 '[All candidate and witness scores](results/24112401/donor_dominance.json)'])
    (ROOT/'dominance_24112401.md').write_text('\n'.join(text)+'\n', encoding='utf8')


if __name__ == '__main__':
    if sys.argv[1:] != ['--report-only']:
        audit()
    summarize()
