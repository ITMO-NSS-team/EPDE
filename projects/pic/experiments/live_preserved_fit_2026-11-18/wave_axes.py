"""Check both equally ranked split axes on the same pooled wave equations."""
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import csv
import json
from math import prod
from search import ROOT
from blocked import block_specs
from transfer import run_pair


if __name__=='__main__':
    previous = [r for file in ('transfer.json','wave_transfer.json') for r in json.loads((ROOT/file).read_text())
                if r['system']=='wave']
    seeds = [r['seed'] for r in previous]
    with ProcessPoolExecutor(max_workers=3) as pool:
        spatial = list(pool.map(partial(run_pair,axis_override=1),[('wave',n) for n in seeds]))
    outcomes = []
    for old,new in zip(previous,spatial):
        seed=old['seed']
        assert seed==new['seed']
        g=json.loads((ROOT/'results'/f'wave_{seed}_wape.json').read_text())['geometry']
        assert g['axis']==0
        levels=[block_specs((81,81),g['native'],d)[0][1]['shape'] for d in (0,1)]
        assert (min(levels[0]),prod(levels[0]))==(min(levels[1]),prod(levels[1]))
        pairs=list(zip(old['variants'],new['variants'],strict=True))
        for a,b in pairs:
            assert json.dumps(a['target'])==json.dumps(b['target'])
            assert all(a[k]==b[k] for k in ('reference','equation','source','eligible_amplification','dyad'))
        for method in ('wls','tls'):
            average=[]
            for i in range(len(old['survival'][method])):
                pts=[((a['scores'][method][i]+b['scores'][method][i])/2,a['dyad'],a['reference']) for a,b in pairs
                     if a['eligible_amplification']]
                pts=[p for p in pts if p[0]<=1.]
                average.append(any(not any(x<=u and y<=v and (x<u or y<v) for x,y,_ in pts) for u,v,ref in pts if ref))
            for name,hits in [('time',old['survival'][method]),('space',new['survival'][method]),('mean',average)]:
                outcomes.append(dict(seed=seed,cohort='screen' if seed<917000 else 'confirmation',arm=method,
                                     split=name,source=int(hits[0]),fresh=sum(hits[1:])))
    with (ROOT/'wave_axes.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(outcomes[0]))
        w.writeheader()
        w.writerows(outcomes)
    for cohort in ('screen','confirmation'):
        for method in ('wls','tls'):
            print(cohort,method,{s:sum(r['fresh'] for r in outcomes if (r['cohort'],r['split'],r['arm'])==(cohort,s,method))
                                 for s in ('time','space','mean')})
