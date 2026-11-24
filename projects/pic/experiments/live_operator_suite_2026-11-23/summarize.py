"""Summarize paired runs without treating the control as a sixteenth suite member."""
import csv
import json
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parent


def read(name):
    with (ROOT/name).open() as stream:
        return list(csv.DictReader(stream))


def matched(rows, keys):
    indexed = {(tuple(r[k] for k in keys),r['arm']):r for r in rows}
    assert len(rows)==len(indexed)
    cases = sorted({k for k,_ in indexed})
    return [(indexed[k,'native'],indexed[k,'pvra']) for k in cases]


def compare(pairs, metric):
    values = [(int(n[metric]),int(g[metric])) for n,g in pairs]
    assert all(a in (0,1) and b in (0,1) for a,b in values)
    return dict(native=sum(a for a,b in values), guided=sum(b for a,b in values),
                gains=values.count((0,1)),losses=values.count((1,0)))


def main():
    laws = read('laws.csv')
    runs = read('systems.csv')
    distances = read('reachability.csv')
    keys = ('system','seed','law')
    all_pairs = matched(laws,keys)
    by_system=[]
    for name in sorted({r['system'] for r in laws}):
        pairs = [(a,b) for a,b in all_pairs if a['system']==name]
        by_system.append(dict(system=name,law_observations=len(pairs),
                         **{k+'_'+m:compare(pairs,k)[m] for k in ('generated','source','held')
                            for m in ('native','guided','gains','losses')}))
    with (ROOT/'by_system.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=by_system[0])
        writer.writeheader()
        writer.writerows(by_system)
    suite = [(a,b) for a,b in all_pairs if a['system']!='ode_simple']
    systems = matched([r for r in runs if r['system']!='ode_simple'],keys[:2])
    assert len(suite)==60 and len(systems)==45 and all(int(a['calls'])==int(b['calls'])==3000 for a,b in systems)
    times = [(int(a['first']),int(b['first'])) for a,b in suite if a['first'] and b['first']]
    reach = {(r['system'],r['seed'],r['law'],r['arm']):r for r in distances}
    state = {}
    for arm,index in [('native',0),('pvra',1)]:
        missing = [p[index] for p in suite if not int(p[index]['generated'])]
        missing_d = [reach[r['system'],r['seed'],r['law'],arm] for r in missing]
        state[arm]=dict(misses=len(missing),one_edit=sum(r['edits']=='1' for r in missing_d),
                        all_terms=sum(int(r['all_terms_scored']) for r in missing_d),
                        missing_frequency=sum(r['frequency_grid_covers_form']=='0' for r in missing_d))
    out = dict(laws={k:compare(suite,k) for k in ('generated','source','held','exact_generated','exact_source','exact_held')},
               systems={k:compare(systems,k) for k in ('assembled','source','held')}, misses=state,
               common_discoveries=len(times), earlier=sum(b<a for a,b in times), later=sum(b>a for a,b in times),
               same=sum(b==a for a,b in times), native_median_calls=median(a for a,b in times),
               guided_median_calls=median(b for a,b in times),median_seconds_ratio=median(float(b['seconds'])/float(a['seconds']) for a,b in systems),
               gains_screened=sum(int(b['gain_columns']) for a,b in systems),guided_additions=sum(int(b['guided']) for a,b in systems),
               max_replay_error=max(float(r['replay_error']) for r in runs))
    (ROOT/'summary.json').write_text(json.dumps(out,indent=2))
    print(json.dumps(out))


if __name__=='__main__':
    main()
