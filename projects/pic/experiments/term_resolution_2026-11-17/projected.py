"""Pairwise residual projection with frozen directions and fresh noise refits."""
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import hashlib
import json
import os
import sys
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import numpy as np

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT.parent/'nuisance_direction_2026-11-17'
OUT=ROOT/'projected'
BANKS=('local','broad','selected')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def vectors(meta):
    cand=next(c for c in meta['candidates'] if c['label']=='reference')
    assert cand['exact']==[-6.,-1.,1.] and 'd^3u/' in str(meta['labels'][cand['xi'][1]])
    v=np.zeros((len(meta['labels']),2))
    v[cand['yi']]=1.
    v[cand['xi'],0]=[6.,1.,-1.]
    v[cand['xi'],1]=[0.,.25,0.]
    return v


def lock():
    path=SOURCE/'results'/'kdv_cossin_iid_seed57.json'
    meta=json.loads(path.read_text())
    assert sha(path.with_suffix('.npz'))==meta['array_sha256']
    v=vectors(meta)
    with np.load(path.with_suffix('.npz')) as z:
        dirs={b:z[b+'_pilot']@(v[:,1]-v[:,0]) for b in BANKS}
    np.savez(OUT/'directions.npz',**{b:x/np.linalg.norm(x) for b,x in dirs.items()},vectors=v)
    return dict(training_json=sha(path),training_array=meta['array_sha256'],
                direction_sha256=sha(OUT/'directions.npz'),protocol=sha(ROOT/'projection_protocol.txt'))


def statistic(residuals, directions):
    # Input contains one or more replicates with shape (..., rows, candidates).
    whole=np.stack([np.sum(residuals[b]**2,axis=-2) for b in BANKS],axis=-2)
    projected=np.stack([np.einsum('r,...rc->...c',directions[b],residuals[b])**2
                        for b in BANKS],axis=-2)
    return np.stack((whole,projected),axis=-3)


def run(job):
    profile,seed=job
    sys.path.insert(0,str(SOURCE))
    import context as engine
    design=SOURCE/'design_screen.json'
    e=engine.Experiment('kdv_cossin',seed,profile,design)
    with np.load(OUT/'directions.npz') as z:
        dirs={b:z[b] for b in BANKS};v=z['vectors']
    np.testing.assert_array_equal(v,vectors({'candidates':e.cs,'labels':e.labels}))
    pilot=e.matrices(e.pilot,correct=False)
    observed=e.matrices(e.raw)
    errors={b:[] for b in BANKS};fits=[];hashes=[];warn=[]
    if any(x['status']!='ok' for x in e.fit_info.values()):
        warn.append(-1)
    rng=np.random.default_rng(np.random.SeedSequence([992702,e.system_id,e.pid,seed]))
    for j in range(160):
        field,_=e.draw(rng)
        model,info=e.estimate(field)
        measured=e.matrices(field,model)
        for b in BANKS:
            errors[b].append((measured[b]-pilot[b])@v)
        fits.append(model)
        hashes.append({name:engine.R.digest(y) for name,y in field.items()})
        if any(x['status']!='ok' for x in info.values()):
            warn.append(j)
        if j in (39,79,119):
            print(profile,seed,j+1,flush=True)
    errors={b:np.stack(rows) for b,rows in errors.items()}
    noise=statistic(errors,dirs)
    assert noise.shape==(160,2,3,2)
    scale=noise[:32].mean(axis=0)
    assert np.all(scale>0)
    scores=(noise[32:]/scale).max(axis=(-2,-1))
    cut=np.sort(scores,axis=0)[122]
    obs={b:m@v for b,m in observed.items()}
    ratios=(statistic(obs,dirs)/scale).max(axis=-2)/cut[:,None]
    prefix=OUT/f'{profile}_{seed}'
    frozen=dict(profile=profile,seed=seed,model=e.model,fit_info=e.fit_info,
                pilot_hash={k:engine.R.digest(y) for k,y in e.pilot.items()},
                raw_hash={k:engine.R.digest(y) for k,y in e.raw.items()},draw_hashes=hashes,
                bootstrap_models=fits,warnings=warn,scale=scale,cutoff=cut,ratios=ratios,
                directions=sha(OUT/'directions.npz'),protocol=sha(ROOT/'projection_protocol.txt'),
                sources={**engine.sources('run.py'),str(Path(__file__)):sha(Path(__file__))})
    engine.dump(prefix.with_suffix('.json'),frozen)
    # Read clean moments for diagnostics only after both test decisions are frozen.
    clean=e.matrices(e.p.data,correct=False)
    arrays={b+'_errors':arr for b,arr in errors.items()}
    target=next(c['yi'] for c in e.cs if c['label']=='reference')
    for b in BANKS:
        arrays[b+'_observed']=obs[b]
        arrays[b+'_clean']=clean[b]@v
        arrays[b+'_target_norm']=np.linalg.norm(clean[b][:,target])
    np.savez_compressed(prefix.with_suffix('.npz'),**arrays)
    print('Complete',profile,seed,'ratios',ratios,flush=True)


if __name__=='__main__':
    OUT.mkdir(exist_ok=True)
    (OUT/'lock.json').write_text(json.dumps(lock(),indent=2))
    jobs=[(p,s) for p in ('iid','positive','negative') for s in (4703,4704)]
    with ProcessPoolExecutor(max_workers=6) as pool:
        list(pool.map(run,jobs))
