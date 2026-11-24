"""Summarize recovery, front size and alternatives using only saved archive indices."""
import csv
import gzip
import json
from statistics import median

from capture import ROOT, SEEDS


def write(name, rows):
    with (ROOT/name).open('w',newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def summarize():
    folder = ROOT/'results'
    seed_rows, competitor_sets = [], {}
    for seed in SEEDS:
        with gzip.open(folder/f'archive_{seed}.json.gz','rt') as f:
            archive = json.load(f)
        with gzip.open(folder/f'validation_{seed}.json.gz','rt') as f:
            val = json.load(f)
        raw = archive['rows']
        names = archive['labels']
        assert sum(r['calls'] for r in raw)==archive['calls']
        assert val['max_wape_replay_error']<1e-9 and len(val['results'])==294
        derivatives = {'d^2u/dx0^2{power: 1.0}', 'd^2u/dx1^2{power: 1.0}'}
        for row in val['results']:
            keys = set()
            for i, score, blocked in row['front']:
                point = raw[i]
                assert score<=1 and (not row['gated'] or blocked<=1)
                assert row['pool']=='archive' or point[row['pool']]
                keys.add((tuple(sorted(names[j] for j in point['ids'])), names[point['pivot']]))
            supports = {x for x,t in keys}
            assert len(keys)==row['size'] and len(supports)==row['supports']
            assert any(set(s)==derivatives for s in supports)==row['exact_hit']
            seed_rows.append(dict(seed=seed, profile=row['profile'], repeat=row['repeat'], pool=row['pool'],
                                  gated=int(row['gated']), eligible_fits=row['eligible'], hit=int(row['exact_hit']),
                                  reference_eligible=int(row['reference_eligible']), size=len(keys), supports=len(supports),
                                  wave_with_extra_terms=sum(derivatives<set(s) for s in supports)))
            if row['gated'] and row['profile']=='independent' and row['pool']=='archive':
                for support in supports:
                    competitor_sets.setdefault(support,[]).append((seed,row['repeat']))
    write('paired_results.csv',seed_rows)
    groups = {}
    for row in seed_rows:
        key = row['profile'],row['pool'],row['gated']
        groups.setdefault(key,[]).append(row)
    summary = []
    for (profile,pool,gated),rows in groups.items():
        summary.append(dict(profile=profile,pool=pool,gated=gated,n=len(rows),hits=sum(r['hit'] for r in rows),
                            reference_eligible=sum(r['reference_eligible'] for r in rows),median_size=median(r['size'] for r in rows),
                            median_supports=median(r['supports'] for r in rows),minimum_size=min(r['size'] for r in rows),
                            maximum_size=max(r['size'] for r in rows),with_superset=sum(r['wave_with_extra_terms']>0 for r in rows)))
    write('summary.csv',summary)
    write('primary_supports.csv',[dict(terms=' + '.join(s),observations=len(cases),seeds=len(set(i for i,j in cases)))
                                  for s,cases in sorted(competitor_sets.items(), key=lambda e:(-len(e[1]),e[0]))])


if __name__=='__main__':
    summarize()
