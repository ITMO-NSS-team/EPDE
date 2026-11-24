"""Replay the frequency arms and compare the actual systems on both budget clocks."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import re
from statistics import median

from frequency_trial import ARMS, CONFIRMATION, ROOT, folder, live
from assess_swap import assess_run, previous, read_archive, write_csv


def assess(job):
    return assess_run(job,destination=folder(*job),moves=None,timing='frequency_seconds')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(pairs,key):
    values = [(a[key],b[key]) for a,b in pairs]
    return dict(cases=len(values),native=sum(a for a,b in values),refined=sum(b for a,b in values),
                gains=values.count((0,1)),losses=values.count((1,0)))


def forced_laws(allowed, keys, archive, budget, extra, peaks):
    """Use labels after discovery to separate frequency coverage from support visitation."""
    def shape(term):
        return re.sub(r'~[^@*]+@','~*@',term)
    rows = []
    for law in range(len(allowed[0])):
        forms = {row[law] for row in allowed}
        def requirements(form):
            return {(float(w),int(d)) for term in form for w,d in re.findall(r'~([^@*]+)@(\d+)',term)}
        if not all(requirements(f) for f in forms):
            continue
        expected = {frozenset(map(shape,form)) for form in forms}
        within = any(all(any(abs(v-w)<=.01*abs(w) for v in peaks.get(str(d),[])) for w,d in requirements(form))
                     for form in forms)
        patterns = [p for p in archive['points'] if (budget=='full' or p[5]+extra<=live.BUDGET)
                    and frozenset(shape(keys[i]) for i,_ in p[2]) in expected]
        rows.append(dict(law=law,grid_covers_law=int(within),pattern_scored=int(bool(patterns)),
                         first_pattern_call=min((p[5] for p in patterns),default=None)))
    return rows


def publish(seeds,out):
    out.mkdir(parents=True,exist_ok=True)
    screen = list(seeds)==list(live.SEEDS)
    archived = {}
    if screen:
        source = json.loads((ROOT.parent/'guided_swap_2026-11-23'/'manifest.json').read_text())
        archived = {(r['dataset'],r['seed']):r['archive_sha256']['native'] for r in source['pairs']}
    systems,laws,grid,forcing,work,control,manifest=[],[],[],[],[],[],[]
    budgets = {b:[] for b in ('full','total')}
    for name in live.DATASETS:
        forms, frequencies = previous.registry(name,live.load_data(name))
        for seed in seeds:
            pair = []
            archives = []
            hashes = {}
            for arm in ARMS:
                dest=folder(name,seed,arm)
                r=json.loads((dest/'assessment.json').read_text())
                a=read_archive(dest)
                op=a['operator']
                pair.append(r)
                archives.append(a)
                hashes[arm]=sha(dest/'archive.json.gz')
                ids=dict(dataset=name,seed=seed,arm=arm)
                labels=[previous.label_key(t,frequencies) for t in a['labels']]
                w=r['work']
                work.append(dict(**ids,seconds=r['seconds'],calibration_seconds=r.get('frequency_seconds',0.),
                                 extra_fits=r['extra_fits'],replay_error=r['replay_error'],
                                 **{k:w.get(k,0) for k in ('background_columns','background_fits','background_projections',
                                                           'harmonic_columns','harmonic_projections','profile_fits','profile_cache_hits')}))
                assert op['full_cost']==list(range(1+r['extra_fits'],live.BUDGET+1+r['extra_fits']))
                for b in budgets:
                    system=r[b]['system']
                    systems.append(dict(**ids,budget=b,**{k:system[k] for k in
                                        ('assembled','source','held','calls','total_fits','candidates','source_system_front_size',
                                         'held_system_front_size','source_law_front_size','held_law_front_size')}))
                    assert system['calls' if b=='full' else 'total_fits']==live.BUDGET
                    if name=='ode_simple':
                        control.append(dict(**ids,budget=b,**{k:v for k,v in system.items() if 'identity' in k or k=='held'}))
                    for law in r[b]['laws']:
                        laws.append(dict(**ids,budget=b,**{k:law[k] for k in ('law','generated','first','first_total','source','held',
                                         'held_fate','first_coefficient_error','held_coefficient_error')}))
                    for row in forced_laws(forms,labels,a,b,r['extra_fits'],r['peaks']):
                        law=r[b]['laws'][row['law']]
                        forcing.append(dict(**ids,budget=b,**row,generated=law['generated'],held=law['held']))
            left,right=pair
            for key in ('settings','windows','train_hashes','trig'):
                assert left[key]==right[key], (name,seed,key)
            if screen:
                assert left['native_archive_verified'] and hashes['native']==archived[name,seed]
            initial,installed={},{}
            for row in archives[1]['operator']['profiles']:
                axis=str(row['axis'])
                initial.setdefault(axis,[]).append(row['initial'])
                installed.setdefault(axis,[]).append(row['installed'])
                assert row['lower']<=row['installed']<=row['upper']
                if row['decision']=='installed':
                    assert row['bic_gain']>0 and all(b<a for a,b in zip(row['initial_blocks'],row['proposed_blocks']))
                else:
                    assert row['installed']==row['initial']
                grid.append(dict(dataset=name,seed=seed,**row))
            assert left['peaks']==initial and right['peaks']==installed,(name,seed,'peak grids')
            if initial==installed:
                for k in ('labels','points','systems','system_events'):
                    assert archives[0][k]==archives[1][k],(name,seed,k)
            manifest.append(dict(dataset=name,seed=seed,archive_sha256=hashes,settings=right['settings'],
                                 train_hashes=right['train_hashes'],windows=right['windows'],
                                 initial=initial,installed=installed))
            if name!='ode_simple':
                for b in budgets:
                    budgets[b].append((name,seed,left[b],right[b]))
    for file,values in [('systems',systems),('laws',laws),('forcing',forcing),('work',work),('control',control)]:
        write_csv(out/(file+'.csv'),values)
    (out/'profiles.json').write_text(json.dumps(grid,indent=2,allow_nan=False))
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    summary={}
    by_dataset=[]
    for b,rows in budgets.items():
        sys_pairs=[(a['system'],g['system']) for name,n,a,g in rows]
        law_pairs=[p for name,n,a,g in rows for p in zip(a['laws'],g['laws'],strict=True)]
        entry=dict(systems={k:compare(sys_pairs,k) for k in ('assembled','source','held')},
                   laws={k:compare(law_pairs,k) for k in ('generated','source','held')})
        entry['gains']=[[s,n] for s,n,a,g in rows if g['system']['held']>a['system']['held']]
        entry['losses']=[[s,n] for s,n,a,g in rows if g['system']['held']<a['system']['held']]
        hits=[p for p in law_pairs if all(v['generated'] for v in p)]
        clock='first_total' if b=='total' else 'first'
        entry['common_first_discoveries']=dict(cases=len(hits),earlier=sum(g[clock]<a[clock] for a,g in hits),
              later=sum(g[clock]>a[clock] for a,g in hits),**{arm:median(p[i][clock] for p in hits) for i,arm in enumerate(ARMS)})
        errors=[p for p in hits if all(v['first_coefficient_error'] is not None for v in p)]
        entry['common_first_coefficient_error']=dict(cases=len(errors),**{arm:median(p[i]['first_coefficient_error']
                                                for p in errors) for i,arm in enumerate(ARMS)})
        for s in live.SYSTEMS:
            pairs=[(a['system'],g['system']) for name,n,a,g in rows if name==s]
            by_dataset.append(dict(dataset=s,budget=b,**compare(pairs,'held'),
                              **{arm+'_median_front_size':median(p[i]['held_system_front_size'] for p in pairs)
                                 for i,arm in enumerate(ARMS)}))
        summary[b]=entry
    write_csv(out/'by_system.csv',by_dataset)
    suite=[r for r in work if r['arm']=='refined' and r['dataset']!='ode_simple']
    profiles=[r for r in grid if r['dataset']!='ode_simple']
    summary['profiled_peaks']=len(profiles)
    summary['installed_moves']=sum(r['decision']=='installed' for r in profiles)
    summary['changed_runs']=len({(r['dataset'],r['seed']) for r in profiles if r['decision']=='installed'})
    summary['extra_fits']=sum(r['extra_fits'] for r in suite)
    summary['median_calibration_seconds']=median(r['calibration_seconds'] for r in suite)
    summary['median_remaining_calls']=median(live.BUDGET-r['extra_fits'] for r in suite)
    summary['max_replay_error']=max(r['replay_error'] for r in work)
    summary['control_held']={arm:sum(r['held'] for r in control if r['arm']==arm and r['budget']=='total') for arm in ARMS}
    gate=summary['total']['systems']['held']
    summary['positive_balance']=gate['gains']>gate['losses']
    (out/'summary.json').write_text(json.dumps(summary,indent=2))
    print(json.dumps(gate),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--confirmation',action='store_true')
    p.add_argument('--workers',type=int,default=3)
    args=p.parse_args()
    seeds=CONFIRMATION if args.confirmation else live.SEEDS
    jobs=[(s,n,a) for n in seeds for s in live.DATASETS for a in ARMS]
    todo=[j for j in jobs if (folder(*j)/'archive.json.gz').is_file() and not (folder(*j)/'assessment.json').is_file()]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for job in pool.map(assess,todo):
            print(*job,flush=True)
    if all((folder(*j)/'assessment.json').is_file() for j in jobs):
        publish(seeds, ROOT/'confirmation' if args.confirmation else ROOT)
