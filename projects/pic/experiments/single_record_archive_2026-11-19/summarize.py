"""Tabulate paired endpoints, failures, and one-edit distances in the actual archives."""
from collections import Counter, defaultdict
import csv
import gzip
import json
from statistics import median

from run_split import PARTS, ROOT, SEEDS, SPATIAL, SYSTEMS, folder
from transfer import LAWS, LOSS_NAN_VAL, labels_of, tokens


def write(name,rows):
    with (ROOT/name).open('w',newline='',encoding='utf8') as f:
        out=csv.DictWriter(f,fieldnames=list(rows[0]))
        out.writeheader()
        out.writerows(rows)


def summarize():
    outcomes, windows, cases = [], [], []
    for system in SYSTEMS:
        truth,physical_target = tokens(LAWS[system])
        for seed in SEEDS:
            hashes = {}
            for part in (*PARTS,*SPATIAL) if system=='wave' else PARTS:
                dest = folder(system,part)
                r = json.loads((dest/f'validation_{seed}.json').read_text())
                with gzip.open(dest/f'archive_{seed}.json.gz','rt') as f:
                    arc = json.load(f)
                assert sum(p['calls'] for p in arc['rows'])==r['calls']
                hashes[part]=r['source_hash'],r['validation_hash']
                names = [labels_of('0 = '+n)[0] for n in arc['labels']]
                truncated = {frozenset(names[j] for j in point['ids']) for point in arc['rows']
                             if max(point['dyad'],point['wape'])<LOSS_NAN_VAL}
                ref = frozenset(truth)
                neighbors = {}
                eligible_rows = [pt for pt in arc['rows'] if max(pt['wape'],pt['dyad'])<LOSS_NAN_VAL]
                observed_terms = {names[j] for pt in eligible_rows for j in pt['ids']}
                for pool in ('returned','terminal','archive'):
                    parents = {frozenset(names[j] for j in pt['ids']) for pt in eligible_rows
                               if (pool=='archive' or pt[pool]) and names[pt['pivot']]==physical_target}
                    neighbors[pool+'_parents'] = sum(len(ref-s)==1 and len(s-ref)<=1 for s in parents) if ref<=observed_terms else 0
                meta = dict(system=system,seed=seed,training=part)
                diag = r['coefficients']
                cases.append(dict(**meta,observed=r['observed'],calls=r['calls'],eligible_fits=r['rows'],
                                  max_replay_error=r['max_replay_error'],half_widths=str(r['geometry']['native']['ms']),
                                  coefficient_1=diag['weights'][1],coefficient_2=diag['weights'][2] if len(ref)==3 else '',
                                  fit_wape=diag['test'],known_wape=diag['test_known'],
                                  one_extra=sum(ref<s and len(s-ref)==1 for s in truncated),
                                  one_missing=sum(s<ref and len(ref-s)==1 for s in truncated),
                                  one_replacement=sum(len(ref-s)==len(s-ref)==1 for s in truncated),**neighbors))
                for result in r['outcomes']:
                    key = ('evaluation','pool','gated','hit','fate','size')
                    outcomes.append(dict(**meta,**{k:result[k] for k in key}))
                if system=='wave' and part in ('early','late'):
                    wider=json.loads((dest/f'window_{seed}.json').read_text())
                    for w in wider:
                        for case in w['cases']:
                            windows.append(dict(**meta,method=w['method'],time_radius=w['ms'][0],time_centers=w['shape'][0],
                                                known_wape=w['physical'],clean_known_wape=w['clean_physical'],fit_wape=w['fitted'],
                                                **{k:case[k] for k in ('pool','gated','hit','eligible_reference','size')}))
            assert hashes['early']==hashes['late'][::-1] and hashes['full'][0]==hashes['full'][1]
            if system=='wave':
                assert hashes['left']==hashes['right'][::-1]
    write('cases.csv',cases)
    write('outcomes.csv',outcomes)
    write('window_check.csv',windows)
    grouped=defaultdict(list)
    for row in outcomes:
        grouped[tuple(row[k] for k in ('system','training','evaluation','pool','gated'))].append(row)
    summary=[]
    for (s,p,e,pool,gate),rows in grouped.items():
        assert len(rows)==len(SEEDS)
        summary.append(dict(system=s,training=p,evaluation=e,pool=pool,gated=gate,hits=sum(r['hit'] for r in rows),
                            median_size=median(r['size'] for r in rows),fates=json.dumps(Counter(r['fate'] for r in rows))))
    write('summary.csv',summary)


if __name__=='__main__':
    summarize()
