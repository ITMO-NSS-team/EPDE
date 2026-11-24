"""Pair forced directional measurements with the unforced KdV observations."""
from collections import Counter
from fractions import Fraction
import json
import numpy as np
from pair import ROOT, PREVIOUS, Bank, observed_banks, classify, fixed, sha
from forcing_profile import READOUTS, scores


def forced_banks(path, manifest):
    assert sha(path) == manifest[path.name]
    with np.load(path) as z:
        grams = [z[b+'_grams'] for b in READOUTS]
        covs = [g[:32].mean(axis=0) for g in grams]
        moments = [z[b+'_observed'] for b in READOUTS]
        dist = np.maximum.reduce([scores(g[32:],s) for g,s in zip(grams,covs)])
        assert dist.shape == (128,) and np.isfinite(dist).all()
        cutoff = float(np.sort(dist)[125])
    return [(m,s,cutoff,Bank(m,s)) for m,s in zip(moments,covs)], cutoff


def main():
    earlier=json.loads((PREVIOUS/'forcing_profile'/'replay.json').read_text())
    manifest={f"{r['profile']}_{r['seed']}.npz":r['archive'] for r in earlier['rows']}
    off={s:observed_banks('kdv',s)[0] for s in [51,52,53]}
    rows=[]
    paths=sorted((PREVIOUS/'forcing_profile').glob('*_47*.npz'))
    assert len(paths)==12
    for path in paths:
        profile,seed=path.stem.split('_')
        forced,cutoff=forced_banks(path,manifest)
        for anchor,banks in [(None,forced),*((s,v+forced) for s,v in off.items())]:
            fits={name:classify(banks,free) for name,free in
                  [('full',[0,1,2]),('drop_forcing',[0,1]),('dispersion_only',[1])]}
            controls={name:fixed(banks,v) for name,v in [('physical',[6,1,-1,1]),
                      ('physical_intrinsic_without_forcing',[6,1,0,1]),
                      ('rival',[0,Fraction(1,5),0,1])]}
            rows.append(dict(profile=profile,seed=int(seed),off_seed=anchor,cutoff=cutoff,
                             fits=fits,controls=controls,array_sha256=manifest[path.name]))
    for group in [False,True]:
        rr=[r for r in rows if (r['off_seed'] is not None)==group]
        for p in ['iid','positive','negative']:
            sub=[r for r in rr if r['profile']==p]
            print(group,p,{name:dict(Counter(r['fits'][name]['status'] for r in sub)) for name in sub[0]['fits']},
                  {name:sum(r['controls'][name]['passed'] for r in sub) for name in sub[0]['controls']},flush=True)
    source={p.name:sha(p) for p in (ROOT/'forcing.py',ROOT/'pair.py',ROOT/'forcing_protocol.txt',
                                     PREVIOUS/'forcing_profile.py',PREVIOUS/'resolution.py',PREVIOUS/'audit.py')}
    (ROOT/'forcing_results.json').write_text(json.dumps(dict(rows=rows,sources=source),indent=2))


if __name__=='__main__':
    main()
