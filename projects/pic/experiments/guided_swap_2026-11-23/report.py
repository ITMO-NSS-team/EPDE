"""Publish paired endpoints, provenance, and the opportunity-by-opportunity swap trace."""
import argparse
from collections import Counter
import csv
import gzip
import hashlib
import json
from statistics import median

from assess_swap import read_archive, write_csv
from swap_trial import ARMS, CONFIRMATION, ROOT, folder, live


def comparison(pairs, metric):
    pairs = [(a[metric], b[metric]) for a,b in pairs]
    assert all(a in (0,1) and b in (0,1) for a,b in pairs)
    return dict(cases=len(pairs), native=sum(a for a,b in pairs), guided=sum(b for a,b in pairs),
                gains=pairs.count((0,1)), losses=pairs.count((1,0)))


def historical_misses():
    def rows(file):
        with (live.ROOT/file).open() as f:
            return list(csv.DictReader(f))
    keys = ('system','seed','law')
    missing = {tuple(r[k] for k in keys) for r in rows('laws.csv') if r['arm']=='pvra' and r['generated']=='0'}
    cases = {tuple(r[k] for k in keys) for r in rows('reachability.csv') if r['arm']=='pvra'
             and r['one_swap']=='1' and r['all_terms_scored']=='1'} & missing
    assert len(cases)==7
    return {(s,int(n),int(law)) for s,n,law in cases}


def verify_trace(op, arm):
    for e in op['swaps']:
        parent, child, top = e['parent'], e['child'], e['shortlist']
        assert len(parent) == len(child) and e['outgoing'] in parent
        assert len(set(parent)) == len(parent) and len(set(child)) == len(child)
        assert e['after_fit_cost'] >= e['fit_cost'] >= e['after_call']
        if e['changed']:
            pos = parent.index(e['outgoing'])
            assert child[:pos] + child[pos+1:] == parent[:pos] + parent[pos+1:]
            assert child[pos] == e['selected']
        else:
            assert child == parent and e['selected'] is None
        if arm == 'native':
            assert not top and e['path']=='native'
        if e['path']=='guided':
            labels, gains = zip(*top)
            assert len(top)<=8 and len(set(labels))==len(top)
            assert not set(parent).intersection(labels) and e['selected'] in labels
            assert e['outgoing'] != e['target'] and e['outgoing'] in e['eligible_parent']
            assert all(g>0 for g in gains) and list(gains)==sorted(gains,reverse=True)
            assert abs(sum(e['victim_probabilities'])-1.) < 1e-12


def publish(seeds, dest):
    dest.mkdir(parents=True,exist_ok=True)
    historical = historical_misses()
    systems, laws, runs, by_system, seven = [], [], [], [], []
    paired_systems = {b:[] for b in ('full','total')}
    paired_laws = {b:[] for b in paired_systems}
    controls = []
    provenance = []
    with (gzip.open(dest/'swap_trace.jsonl.gz','wt',encoding='utf8',compresslevel=5) as traces,
          gzip.open(dest/'domination_replay.jsonl.gz','wt',encoding='utf8',compresslevel=5) as replay):
        for name in live.DATASETS:
            for seed in seeds:
                reports = []
                for arm in ARMS:
                    path = folder(name,seed,arm)
                    rec = json.loads((path/'assessment.json').read_text())
                    reports.append(rec)
                    if arm=='native' and seed in live.SEEDS:
                        assert rec['native_archive_verified']
                        original = json.loads((live.folder(name,seed,arm)/'assessment.json').read_text())
                        assert all(rec['full']['system'][k]==original['joint'][k]
                                   for k in ('held','source','assembled','candidates'))
                    ids = dict(dataset=name,seed=seed,arm=arm)
                    with gzip.open(path/'dominators.json.gz','rt',encoding='utf8') as stream:
                        observed = json.load(stream)
                    if observed['cases']:
                        replay.write(json.dumps(dict(**ids,**observed),separators=(',',':'))+'\n')
                    archive = read_archive(path)
                    op = archive['operator']
                    verify_trace(op,arm)
                    info = dict(**ids, type='dictionary', labels=op['trace_labels'], hashes=op['hashes'])
                    traces.write(json.dumps(info,separators=(',',':'))+'\n')
                    for event in op['swaps']:
                        traces.write(json.dumps(dict(**ids, type='swap', **event),separators=(',',':'))+'\n')
                    runs.append(dict(**ids, calls=archive['calls'], extra_fits=rec['extra_fits'],
                                     gain_columns=rec['gain_columns'], seconds=rec['seconds'],
                                     swap_seconds=rec['swap_seconds'], replay_error=rec['replay_error'],
                                     **{k:rec['work'].get(k,0) for k in ('weak_column_requests','weak_column_cache_hits',
                                                        'fitted_cache_hits','version_cache_hits','parent_prefits')}))
                    for budget in paired_systems:
                        item = rec[budget]
                        systems.append(dict(**ids,budget=budget,**{k:v for k,v in item['system'].items() if 'identity' not in k}))
                        for row in item['laws']:
                            laws.append(dict(**ids,budget=budget,**row))
                            if (name,seed,row['law']) in historical:
                                seven.append(laws[-1])
                left,right = reports
                for k in ('train_hashes','settings','windows','trig','peaks','hashes'):
                    assert left[k]==right[k], (name,seed,k)
                paths = {a:folder(name,seed,a)/'archive.json.gz' for a in ARMS}
                if left.get('native_archive_verified'):
                    paths['previous_native'] = live.folder(name,seed,'native')/'archive.json.gz'
                provenance.append(dict(dataset=name,seed=seed, settings=left['settings'], windows=left['windows'],
                                       train_hashes=left['train_hashes'], peaks=left['peaks'],
                                       archive_sha256={a:hashlib.sha256(p.read_bytes()).hexdigest() for a,p in paths.items()}))
                for budget in paired_systems:
                    a,b = left[budget],right[budget]
                    for s in (a['system'],b['system']):
                        assert s['calls' if budget=='full' else 'total_fits']==live.BUDGET
                    pair = (a['system'],b['system'])
                    if name=='ode_simple':
                        controls.append(dict(seed=seed,budget=budget,**{arm+'_'+k:s[k] for arm,s in zip(ARMS,pair)
                                             for k in ('held','held_identity_literal','held_identity_registered_frequency')}))
                    else:
                        paired_systems[budget].append((name,seed,*pair))
                        paired_laws[budget].extend((name,seed,x,y) for x,y in zip(a['laws'],b['laws'],strict=True))
    write_csv(dest/'systems.csv',systems)
    write_csv(dest/'laws.csv',laws)
    write_csv(dest/'work.csv',runs)
    write_csv(dest/'control.csv',controls)
    (dest/'manifest.json').write_text(json.dumps(dict(operator_sha256=left['hashes'], pairs=provenance),
                                               indent=2),encoding='utf8')
    if seven:
        write_csv(dest/'historical_misses.csv',seven)
    summary = {}
    for budget,system_rows in paired_systems.items():
        sys_pairs = [(a,b) for name,n,a,b in system_rows]
        law_pairs = [(a,b) for name,n,a,b in paired_laws[budget]]
        total = dict(systems={k:comparison(sys_pairs,k) for k in ('assembled','source','held')},
                     laws={k:comparison(law_pairs,k) for k in ('generated','source','held')})
        for name in live.SYSTEMS:
            pairs = [(a,b) for dataset,n,a,b in system_rows if dataset==name]
            by_system.append(dict(dataset=name,budget=budget,**comparison(pairs,'held'),
                                  **{arm+'_front_median':median(p[i]['held_system_front_size'] for p in pairs)
                                     for i,arm in enumerate(ARMS)}))
        common = [(a,b) for a,b in law_pairs if a['generated'] and b['generated']]
        times = 'first' if budget=='full' else 'first_total'
        total['common_discovery'] = dict(cases=len(common),earlier=sum(b[times]<a[times] for a,b in common),
                                         later=sum(b[times]>a[times] for a,b in common),
                                         **{arm+'_median':median(p[i][times] for p in common) for i,arm in enumerate(ARMS)})
        errors = [(a['first_coefficient_error'],b['first_coefficient_error']) for a,b in common
                  if a['first_coefficient_error'] is not None and b['first_coefficient_error'] is not None]
        total['coefficient_error_on_common_first_hits'] = dict(cases=len(errors),
                            **{arm+'_median':median(p[i] for p in errors) for i,arm in enumerate(ARMS)})
        total['guided_fates'] = dict(Counter(b['trace_fate'] for a,b in law_pairs))
        total['gains'] = [[name,n] for name,n,a,b in system_rows if b['held']>a['held']]
        total['losses'] = [[name,n] for name,n,a,b in system_rows if b['held']<a['held']]
        summary[budget] = total
    write_csv(dest/'by_system.csv',by_system)
    suite = [r for r in runs if r['dataset']!='ode_simple']
    guided = [r for r in suite if r['arm']=='guided_swap']
    summary['work'] = {k:sum(r[k] for r in guided) for k in
                       ('calls','extra_fits','gain_columns','weak_column_cache_hits','weak_column_requests',
                        'fitted_cache_hits','version_cache_hits','parent_prefits')}
    times = {(r['dataset'],r['seed'],r['arm']):r['seconds'] for r in suite}
    summary['work']['median_runtime_ratio'] = median(r['seconds']/times[r['dataset'],r['seed'],'native'] for r in guided)
    summary['max_replay_error'] = max(r['replay_error'] for r in runs)
    summary['control_held'] = {a:sum(r[a+'_held'] for r in controls if r['budget']=='total') for a in ARMS}
    gate = summary['total']['systems']['held']
    summary['positive_balance'] = gate['gains'] > gate['losses']
    (dest/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    print(json.dumps(gate),flush=True)


if __name__=='__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--confirmation',action='store_true')
    args=p.parse_args()
    publish(CONFIRMATION if args.confirmation else live.SEEDS, ROOT/'confirmation' if args.confirmation else ROOT)
