"""Rebuild support-restricted moment tensors and verify every held-out noise field."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import json
import os
import sys
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import numpy as np
import projected as exp

ROOT=exp.ROOT
OUT=ROOT/'forcing_profile'


def replay(path):
    sys.path.insert(0,str(exp.SOURCE))
    import context as u
    d=json.loads(path.read_text());p,seed=d['profile'],d['seed']
    e,_,_=u.C.primary('kdv_cossin',p,51)
    e.seed=seed;e.raw=e.true_field(0,True);e.model=d['model']
    assert {k:u.R.digest(x) for k,x in e.raw.items()}==d['raw_hash']
    e.pilot,_=u.Pilot.reconstruct(e.raw,e.axes,e.model)
    assert {k:u.R.digest(x) for k,x in e.pilot.items()}==d['pilot_hash']
    design=u.read(exp.SOURCE/'design_screen.json')
    u.D.build(e,{**e.specs,'selected':design['specs'][design['selected']]})
    c=next(x for x in e.cs if x['label']=='reference')
    indices=c['xi']+[c['yi']]
    assert c['exact']==[-6.,-1.,1.] and indices[2] in e.deterministic
    assert all(indices[k] in e.active for k in (0,1,3))
    train=u.matrices(e,e.pilot,correct=False)
    obs=u.matrices(e,e.raw);clean=u.matrices(e,e.p.data,correct=False)
    with np.load(ROOT/'forcing'/'directions.npz') as f:
        dirs={b:f[b] for b in exp.BANKS};fixed=f['vectors'][indices]
    result={};grams={};readouts=[]
    for b in exp.BANKS:
        for projected in (False,True):
            name=('projected_' if projected else 'whole_')+b
            readouts.append((name,b,projected));grams[name]=[]
            for key,mats in [('observed',obs),('clean',clean)]:
                mat=mats[b][:,indices]
                result[name+'_'+key]=dirs[b][None,:]@mat if projected else mat.copy()
    rng=np.random.default_rng(np.random.SeedSequence([992702,e.system_id,e.pid,seed]))
    with np.load(path.with_suffix('.npz')) as prev:
        old={b:prev[b+'_errors'] for b in exp.BANKS}
    for j,model in enumerate(d['bootstrap_models']):
        field,_=e.draw(rng)
        assert {k:u.R.digest(x) for k,x in field.items()}==d['draw_hashes'][j]
        measured=u.matrices(e,field,model)
        for b in exp.BANKS:
            delta=(measured[b]-train[b])[:,indices]
            assert np.all(delta[:,2]==0)
            np.testing.assert_allclose(delta@fixed,old[b][j],rtol=2e-11,atol=1e-12)
            r=dirs[b][None,:]@delta
            grams['whole_'+b].append(delta.T@delta)
            grams['projected_'+b].append(r.T@r)
    for name,_,_ in readouts:
        result[name+'_grams']=np.stack(grams[name])
        assert result[name+'_grams'].shape==(160,4,4)
    np.savez_compressed(OUT/path.with_suffix('.npz').name,**result)
    return dict(profile=p,seed=seed,field_replays=160,source=exp.sha(path),
                archive=exp.sha(OUT/path.with_suffix('.npz').name))


if __name__=='__main__':
    paths=sorted((ROOT/'forcing').glob('*_47*.json'));assert len(paths)==12
    OUT.mkdir(exist_ok=True)
    with ProcessPoolExecutor(max_workers=8) as pool:
        rows=list(pool.map(replay,paths))
    (OUT/'replay.json').write_text(json.dumps(dict(status='PASS',rows=rows),indent=2))
    print('Replayed 12 new observations and 1,920 draws, including stored two-equation residuals.')
