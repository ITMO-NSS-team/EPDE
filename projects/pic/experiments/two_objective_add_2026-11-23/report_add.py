"""Check the local Pareto draws and report paired whole-system recovery."""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from statistics import median

from add_trial import ARMS, CONFIRMATION, ROOT, folder, live, np
from assess_swap import read_archive, system_front, write_csv


def paired(pairs, key):
    values=[(a[key],b[key]) for a,b in pairs]
    return dict(cases=len(values),pvra=sum(a for a,b in values),pvxa=sum(b for a,b in values),
                gains=values.count((0,1)),losses=values.count((1,0)))


def verify_trace(op, arm):
    fits=0
    positions=0
    for e in op['additions']:
        before, after = e['parent'], e['child']
        assert after == (before+[e['selected']] if e['changed'] else before)
        ids=[j for j,gain in e['shortlist']]
        gains=[gain for j,gain in e['shortlist']]
        assert len(ids)<=8 and len(set(ids))==len(ids) and not set(ids).intersection(before)
        assert all(g>0 for g in gains) and gains==sorted(gains,reverse=True)
        positions += len(ids)
        if arm=='pvra':
            assert not e['scores'] and not e['front']
            expected=np.array(gains)/sum(gains) if gains else []
        else:
            assert ids==[v['term'] for v in e['scores']]
            coords,owners=[],[]
            for i,s in enumerate(e['scores']):
                coords.extend([v['wape'],v['instability']] for v in s['versions'])
                owners.extend([i]*len(s['versions']))
                if s['source']!='fresh':
                    assert s['fits']==0
            local=sorted(set(np.asarray(owners,dtype=int)[system_front(np.asarray(coords).reshape(-1,2))]))
            assert local==e['front']
            expected=np.zeros(len(ids))
            for i in local:
                expected[i]=.25 if e['repeat'][i] else 1.
            if local:
                expected/=expected.sum()
        np.testing.assert_allclose(expected,e['probabilities'],rtol=1e-14,atol=0.)
        if e['path']=='guided':
            assert e['changed'] and expected[ids.index(e['selected'])]>0.
        charge=sum(s['fits'] for s in e['scores'])
        assert e['after_fit_cost']-e['fit_cost']==charge
        fits += charge
    assert fits==op['extra_fits']
    if arm=='pvxa':
        assert positions==sum(op['work'].get(k,0) for k in ('shortlist_fit_attempts','shortlist_cache_hits','discovery_cache_hits'))
    return positions


def publish(seeds, out):
    out.mkdir(parents=True,exist_ok=True)
    systems,laws,metrics,controls,manifest=[],[],[],[],[]
    groups={budget:[] for budget in ('full','total')}
    max_error=0.
    front_sizes=Counter()
    mass_on_front=[]
    first_on_front=0
    with gzip.open(out/'addition_trace.jsonl.gz','wt',encoding='utf8',compresslevel=5) as trace:
        for s in live.DATASETS:
            for n in seeds:
                reports=[]
                archive_hashes={}
                first_lists=[]
                for a in ARMS:
                    dest=folder(s,n,a)
                    r=json.loads((dest/'assessment.json').read_text())
                    reports.append(r)
                    rec=read_archive(dest)
                    op=rec['operator']
                    ids=dict(dataset=s,seed=n,arm=a)
                    first=next(e for e in op['additions'] if e['shortlist'])
                    labels=op['trace_labels']
                    first_lists.append((first['after_call'],first['slot'],labels[first['target']],
                                        [labels[i] for i in first['parent']],
                                        [(labels[i],gain) for i,gain in first['shortlist']]))
                    if a=='pvxa' and s!='ode_simple':
                        front_sizes.update(len(e['front']) for e in op['additions'] if e['shortlist'])
                        for e in op['additions']:
                            if e['front']:
                                gains=[g for _,g in e['shortlist']]
                                mass_on_front.append(sum(gains[i] for i in e['front'])/sum(gains))
                                first_on_front += int(0 in e['front'])
                    archive_hashes[a]=hashlib.sha256((dest/'archive.json.gz').read_bytes()).hexdigest()
                    positions=verify_trace(op,a)
                    trace.write(json.dumps(dict(**ids,type='dictionary',labels=op['trace_labels']),separators=(',',':'))+'\n')
                    for e in op['additions']:
                        trace.write(json.dumps(dict(**ids,type='addition',**e),separators=(',',':'))+'\n')
                    if a=='pvra' and n in live.SEEDS:
                        assert r['pvra_archive_verified']
                        baseline=live.folder(s,n,a)
                        original=json.loads((baseline/'assessment.json').read_text())
                        assert all(original['joint'][k]==r['full']['system'][k] for k in ('held','source','assembled','candidates'))
                        archive_hashes['archived_pvra']=hashlib.sha256((baseline/'archive.json.gz').read_bytes()).hexdigest()
                    max_error=max(max_error,r['replay_error'])
                    metrics.append(dict(**ids,seconds=r['seconds'],add_seconds=r['add_seconds'],full_fits=rec['calls'],
                                    extra_fits=r['extra_fits'],gain_columns=r['gain_columns'],shortlist_positions=positions,
                                    replay_error=r['replay_error'],**{k:r['work'].get(k,0) for k in
                                    ('shortlist_fit_attempts','shortlist_cache_hits','discovery_cache_hits',
                                     'weak_column_requests','weak_column_cache_hits')}))
                    for budget in groups:
                        sysrow=r[budget]['system']
                        assert sysrow['calls' if budget=='full' else 'total_fits']==live.BUDGET
                        systems.append(dict(**ids,budget=budget,**{k:v for k,v in sysrow.items() if 'identity' not in k}))
                        laws.extend(dict(**ids,budget=budget,**law) for law in r[budget]['laws'])
                        if s=='ode_simple':
                            controls.append(dict(**ids,budget=budget,**{k:v for k,v in sysrow.items() if 'identity' in k or k=='held'}))
                left,right=reports
                assert first_lists[0]==first_lists[1], (s,n,'first common shortlist')
                for key in ('train_hashes','settings','windows','trig','peaks','hashes'):
                    assert left[key]==right[key], (s,n,key)
                manifest.append(dict(dataset=s,seed=n,archive_sha256=archive_hashes,
                                     **{k:left[k] for k in ('train_hashes','windows','settings','peaks','trig')}))
                if s!='ode_simple':
                    for budget in groups:
                        groups[budget].append((s,n,left[budget],right[budget]))
    for path,rows in [('systems',systems),('laws',laws),('work',metrics),('control',controls)]:
        write_csv(out/(path+'.csv'),rows)
    (out/'manifest.json').write_text(json.dumps(dict(operator_sha256=left['hashes'],pairs=manifest),indent=2))
    summary={}
    by_system=[]
    for budget,rows in groups.items():
        syspairs=[(a['system'],b['system']) for s,n,a,b in rows]
        lawpairs=[pair for s,n,a,b in rows for pair in zip(a['laws'],b['laws'],strict=True)]
        result=dict(systems={k:paired(syspairs,k) for k in ('assembled','source','held')},
                    laws={k:paired(lawpairs,k) for k in ('generated','source','held')})
        for name in live.SYSTEMS:
            subset=[(a['system'],b['system']) for s,n,a,b in rows if s==name]
            by_system.append(dict(dataset=name,budget=budget,**paired(subset,'held'),
                                  **{a+'_median_front_size':median(p[i]['held_system_front_size'] for p in subset)
                                     for i,a in enumerate(ARMS)}))
        common=[(a,b) for a,b in lawpairs if a['generated'] and b['generated']]
        first='first' if budget=='full' else 'first_total'
        errors=[p for p in common if all(v['first_coefficient_error'] is not None for v in p)]
        result['common_first_discoveries']=dict(cases=len(common),earlier=sum(b[first]<a[first] for a,b in common),
                            later=sum(b[first]>a[first] for a,b in common),
                            **{a+'_median':median(p[i][first] for p in common) for i,a in enumerate(ARMS)})
        result['common_first_coefficient_errors']=dict(cases=len(errors),**{a+'_median':median(p[i]['first_coefficient_error']
                                         for p in errors) for i,a in enumerate(ARMS)})
        result['gains']=[[s,n] for s,n,a,b in rows if b['system']['held']>a['system']['held']]
        result['losses']=[[s,n] for s,n,a,b in rows if b['system']['held']<a['system']['held']]
        result['fates']={a:dict(Counter(p[i]['trace_fate'] for p in lawpairs)) for i,a in enumerate(ARMS)}
        summary[budget]=result
    write_csv(out/'by_system.csv',by_system)
    suite=[r for r in metrics if r['dataset']!='ode_simple']
    summary['work']={a:{k:sum(r[k] for r in suite if r['arm']==a) for k in
                     ('full_fits','extra_fits','gain_columns','shortlist_positions','shortlist_fit_attempts',
                      'shortlist_cache_hits','discovery_cache_hits','weak_column_requests','weak_column_cache_hits')} for a in ARMS}
    times={(r['dataset'],r['seed'],r['arm']):r['seconds'] for r in suite}
    summary['median_paired_runtime_ratio']=median(r['seconds']/times[r['dataset'],r['seed'],'pvra'] for r in suite if r['arm']=='pvxa')
    summary['control_held']={a:sum(r['held'] for r in controls if r['budget']=='total' and r['arm']==a) for a in ARMS}
    summary['max_replay_error']=max_error
    summary['pvxa_local_front_sizes_at_full_budget']=dict(sorted(front_sizes.items()))
    summary['pvxa_nonempty_local_fronts']=dict(count=len(mass_on_front),top_gain_on_front=first_on_front,
                                       mean_pvra_draw_mass_on_front=sum(mass_on_front)/len(mass_on_front))
    gate=summary['total']['systems']['held']
    summary['positive_balance']=gate['gains']>gate['losses']
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(gate),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--confirmation',action='store_true')
    args=p.parse_args()
    publish(CONFIRMATION if args.confirmation else live.SEEDS,ROOT/'confirmation' if args.confirmation else ROOT)
