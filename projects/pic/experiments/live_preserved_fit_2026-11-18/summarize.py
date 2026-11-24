"""Tabulate live-search outcomes separately from pooled-front and analytic diagnostics."""
import csv
import json
from statistics import median
from search import ARMS,ROOT,SEEDS,SYSTEMS


def dominates(a,b):
    return all(x<=y for x,y in zip(a,b)) and any(x<y for x,y in zip(a,b))


if __name__=='__main__':
    transfer={(r['system'],r['seed']):r for r in json.loads((ROOT/'transfer.json').read_text())}
    ode=json.loads((ROOT/'ode_identities.json').read_text())
    rows=[]
    for s in SYSTEMS:
        for seed in SEEDS:
            for arm in ARMS:
                r=json.loads((ROOT/'results'/f'{s}_{seed}_{arm}.json').read_text())
                front=r['front']
                assert all(not any(dominates(b['objectives'],a['objectives']) for b in front) for a in front)
                hit=any(a['hit'] for a in front)
                points=r['reference_points']
                finite=[p for p in points if max(p['objectives'])<1e7]
                if hit:
                    assert finite
                if hit:
                    fate='retained'
                elif not points:
                    fate='never scored'
                elif not finite:
                    fate='stamped'
                elif all(any(dominates(a['objectives'],p['objectives']) for a in front) for p in finite):
                    fate='dominated'
                else:
                    fate='absent with an evaluated point undominated by final front'
                transfer_wins=transfer[s,seed]['survival'][arm]
                identity=[a for a in ode if a['arm']==arm and a['seed']==seed] if s=='ode' else []
                rows.append(dict(system=s,seed=seed,arm=arm,registered=int(hit),exact=int(any(a['hit']=='exact' for a in front)),
                                 reference_evaluated=int(bool(points)),fate=fate,front_size=len(front),
                                 min_distance=min(a['distance'] for a in front),axis=r['geometry']['axis'],
                                 half_shapes=str(r['geometry']['halves']),native_window=str(r['geometry']['native']['ms']),
                                 calls=r['calls'],evaluated_supports=r['evaluated_supports'],seconds=round(r['seconds'],2),
                                 pooled_source=int(transfer_wins[0]),pooled_fresh=sum(transfer_wins[1:]),
                                 ode_literal=int(any(i['literal'] for i in identity)) if identity else None,
                                 ode_unit_frequency=int(any(i['unit_frequency'] for i in identity)) if identity else None))
    with (ROOT/'outcomes.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    for s in (*SYSTEMS,'all'):
        print(s, {a:tuple(sum(z[k] for z in rows if z['arm']==a and (s=='all' or z['system']==s))
                      for k in ('registered','exact','reference_evaluated','pooled_source','pooled_fresh')) for a in ARMS})
    print('median front sizes',{a:median(r['front_size'] for r in rows if r['arm']==a) for a in ARMS})
