"""Replay frozen equations and check parent provenance before comparing endpoints."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import re
from statistics import median

from candidate_trial import CHECKPOINT, CONFIRMATION, ROOT, folder, live
from assess_swap import assess_run, previous, read_archive, write_csv

ARMS = ('native', 'candidate')


def assess(job):
    return assess_run(job, destination=folder(*job), moves=None, timing='frequency_seconds')


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(pairs):
    return dict(native=sum(a for a,b in pairs), candidate=sum(b for a,b in pairs),
                gains=pairs.count((0,1)), losses=pairs.count((1,0)), pairs=len(pairs))


def check_pair(native, candidate):
    for key in ('variables', 'axis', 'trig', 'settings', 'train_hashes', 'plan', 'calls'):
        assert native[key] == candidate[key], key
    prefix = [p for p in native['points'] if p[5] <= CHECKPOINT]
    assert prefix == [p for p in candidate['points'] if p[5] <= CHECKPOINT]
    ids = {j for p in prefix for j,_ in p[2]}
    assert all(native['labels'][j] == candidate['labels'][j] for j in ids)
    clocks = [{tuple(c):t for c,t in zip(a['systems'], a['operator']['system_cost'], strict=True)}
              for a in (native, candidate)]
    selected = [{c:t for c,t in costs.items() if t <= CHECKPOINT} for costs in clocks]
    assert selected[0] == selected[1]
    op = candidate['operator']
    assert op['extra_fits'] == sum(r['fits'] for r in op['profiles'])
    assert op['full_cost'][-1] == live.BUDGET + op['extra_fits']
    for r in op['profiles']:
        parent = tuple(r['parent'])
        slot = candidate['variables'].index(r['slot'])
        assert r['point'] == parent[slot] and parent in selected[0]
        assert r['lower'] <= r['proposed'] <= r['upper']
        for child in r['children']:
            assert tuple(child) in clocks[1] and clocks[1][tuple(child)] <= r['end_cost']
            assert all(x == y for i,(x,y) in enumerate(zip(parent, child, strict=True)) if i != slot)
            assert r['best_sse'] < r['initial_sse']
    if not op['profiles']:
        for key in ('labels','points','systems','system_events'):
            assert native[key] == candidate[key], key
    return len(prefix), len(selected[0])


def forcing_audit(name, seed, archive, results, details=None):
    problem = live.load_data(name)
    allowed, freq = previous.registry(name, problem)
    forms = {f for variant in allowed for f in variant}
    pattern = lambda label: re.sub(r'~[^@*]+@', '~*@', label)
    patterns = {frozenset(map(pattern, f)) for f in forms}
    omega, = freq
    labels = [previous.label_key(term, freq) for term in archive['labels']]
    points = archive['points']
    supports = [frozenset(labels[j] for j,_ in p[2]) for p in points]
    matches = {i for i, support in enumerate(supports)
               if frozenset(map(pattern, support)) in patterns}
    op = archive['operator']
    candidates = [(c,t) for c,t in zip(archive['systems'], op['system_cost']) if t <= CHECKPOINT]
    rows = [r for r in op['profiles'] if r['point'] in matches]
    if details is not None and rows:
        _,held,_,_,_ = previous.replay(problem,problem.noisy(5,seed),archive)
        for r in rows:
            parent = r['point']
            for combo in r['children']:
                child, = combo
                if child not in matches or points[child][1] != points[parent][1]:
                    continue
                details.append(dict(dataset=name,seed=seed,parent=parent,child=child,target=labels[points[parent][1]],
                                    initial=r['initial'],refined=r['proposed'],registered=omega,
                                    within_band=int(abs(r['proposed']-omega)<=.01*abs(omega)),
                                    parent_train=points[parent][3],child_train=points[child][3],
                                    parent_held=float(held[parent]),child_held=float(held[child]),
                                    parent_dyad=points[parent][4],child_dyad=points[child][4]))
    return dict(dataset=name, seed=seed, registered=omega,
                pattern_scored_anytime=int(bool(matches)),
                pattern_in_parent_by_checkpoint=int(any(c[0] in matches for c,t in candidates)),
                selected_matching_parents=len(rows),
                frequencies=json.dumps([r['proposed'] for r in rows]),
                selected_in_band=sum(abs(r['proposed']-omega) <= .01*abs(omega) for r in rows),
                admitted_matching_children=sum(supports[c[0]] in forms for r in op['profiles'] for c in r['children']),
                exact_law_scored=results['total']['laws'][0]['generated'],
                exact_law_held=results['total']['laws'][0]['held'],
                system_held=results['total']['system']['held'])


def publish(seeds, out):
    screen = list(seeds) == list(live.SEEDS)
    prior = json.loads((ROOT.parent/'guided_swap_2026-11-23'/'manifest.json').read_text())
    digests = {(r['dataset'], r['seed']):r['archive_sha256']['native'] for r in prior['pairs']}
    systems, laws, work, forced, by_system, manifest, control, details = [], [], [], [], [], [], [], []
    budgets = {b:[] for b in ('full', 'total')}
    for name in live.DATASETS:
        for seed in seeds:
            jobs = [(name,seed,a) for a in ARMS]
            dirs = [folder(*j) for j in jobs]
            a, g = [read_archive(d) for d in dirs]
            pair = [json.loads((d/'assessment.json').read_text()) for d in dirs]
            assert pair[0]['peaks'] == pair[1]['peaks']
            counts = check_pair(a, g)
            hashes = {arm:file_hash(d/'archive.json.gz') for arm,d in zip(ARMS, dirs)}
            if screen:
                assert hashes['native'] == digests[name,seed] and pair[0]['native_archive_verified']
            manifest.append(dict(dataset=name, seed=seed, archive_sha256=hashes,
                                 verified_prefix_points=counts[0], verified_prefix_systems=counts[1],
                                 train_hashes=g['train_hashes'], plan=g['plan']))
            op = g['operator']
            w = op['work']
            rows = op['profiles']
            work.append(dict(dataset=name, seed=seed, checkpoint=op['checkpoint'], parents=len(rows),
                             eligible_parents=w.get('eligible_parent_points',0),
                             moved=sum(r['proposed'] != r['initial'] for r in rows),
                             admitted_children=sum(len(r['children']) for r in rows), extra_fits=op['extra_fits'],
                             weak_columns=w.get('weak_columns_built',0), weak_hits=w.get('weak_cache_hits',0),
                             profile_hits=w.get('profile_cache_hits',0), profile_seconds=op['frequency_seconds'],
                             native_seconds=pair[0]['seconds'], candidate_seconds=pair[1]['seconds'],
                             replay_error=pair[1]['replay_error']))
            if name in ('ode','duffing'):
                forced.append(forcing_audit(name,seed,g,pair[1],details))
            for b in budgets:
                for arm,r in zip(ARMS,pair):
                    s = r[b]['system']
                    ident = dict(dataset=name, seed=seed, arm=arm, budget=b)
                    systems.append(dict(**ident, **{k:s[k] for k in
                        ('calls','total_fits','candidates','assembled','source','held','source_system_front_size',
                         'held_system_front_size','source_law_front_size','held_law_front_size')}))
                    assert s['calls' if b=='full' else 'total_fits'] == live.BUDGET
                    if name == 'ode_simple':
                        control.append(dict(**ident, **{k:v for k,v in s.items() if 'identity' in k or k=='held'}))
                    for law in r[b]['laws']:
                        laws.append(dict(**ident, **{k:law[k] for k in
                            ('law','generated','first','first_total','source','held','held_fate',
                             'first_coefficient_error','held_coefficient_error')}))
                if name != 'ode_simple':
                    budgets[b].append((name,seed,pair[0][b],pair[1][b]))
    summary = {}
    for b, items in budgets.items():
        law_pairs = [p for n,s,a,g in items for p in zip(a['laws'],g['laws'],strict=True)]
        entry = dict(systems={k:compare([(a['system'][k],g['system'][k]) for n,s,a,g in items])
                              for k in ('assembled','source','held')},
                     laws={k:compare([(a[k],g[k]) for a,g in law_pairs]) for k in ('generated','held')})
        for flag,v in [('gains',1),('losses',-1)]:
            entry[flag] = [[n,s] for n,s,a,g in items if g['system']['held']-a['system']['held']==v]
        for name in live.SYSTEMS:
            pairs = [(a['system'],g['system']) for n,s,a,g in items if n==name]
            by_system.append(dict(dataset=name,budget=b,**compare([(a['held'],g['held']) for a,g in pairs]),
                                  **{arm+'_median_front':median(p[i]['held_system_front_size'] for p in pairs)
                                     for i,arm in enumerate(ARMS)}))
        summary[b] = entry
    suite = [r for r in work if r['dataset'] != 'ode_simple']
    summary['work'] = {key:sum(r[key] for r in suite) for key in
                       ('parents','moved','admitted_children','extra_fits','weak_columns','weak_hits','profile_hits')}
    summary['max_replay_error'] = max(r['replay_error'] for r in work)
    summary['control_held'] = {arm:sum(r['held'] for r in control if r['arm']==arm and r['budget']=='total')
                               for arm in ARMS}
    held = summary['total']['systems']['held']
    summary['positive_balance'] = held['gains'] > held['losses']
    out.mkdir(exist_ok=True)
    for name,rows in [('systems',systems),('laws',laws),('work',work),('forcing',forced),
                      ('by_system',by_system),('control',control)]:
        write_csv(out/(name+'.csv'), rows)
    if details:
        write_csv(out/'forcing_profiles.csv',details)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(held), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workers', type=int, default=3)
    p.add_argument('--confirmation', action='store_true')
    args = p.parse_args()
    seeds = CONFIRMATION if args.confirmation else live.SEEDS
    jobs = [(s,n,a) for s in live.DATASETS for n in seeds for a in ARMS]
    todo = [j for j in jobs if (folder(*j)/'archive.json.gz').exists() and not (folder(*j)/'assessment.json').exists()]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for j in pool.map(assess,todo):
            print(*j, flush=True)
    if all((folder(*j)/'assessment.json').exists() for j in jobs):
        publish(seeds, ROOT/'confirmation' if args.confirmation else ROOT)
