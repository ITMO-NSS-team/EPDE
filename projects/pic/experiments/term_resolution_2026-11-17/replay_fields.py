"""Regenerate fresh field streams and replay corrected moments from each cohort."""
import os
import sys
import json
from pathlib import Path
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[key]='1'
import numpy as np
ROOT=Path(__file__).resolve().parent
SOURCE=ROOT.parent/'nuisance_direction_2026-11-17'
sys.path.insert(0,str(SOURCE))
import context as U


def replay(path,projected):
    d=json.loads(path.read_text());profile,seed=d['profile'],d['seed']
    e,_,_=U.C.primary('kdv_cossin',profile,51)
    e.seed=seed
    e.raw=e.true_field(0,True)
    raw=d['raw_hash'] if projected else d['raw_sha256']
    assert {k:U.R.digest(x) for k,x in e.raw.items()}==raw
    e.model=d['model']
    design=U.read(SOURCE/'design_screen.json')
    specs=dict(e.specs);specs['selected']=design['specs'][design['selected']]
    U.D.build(e,specs)
    with np.load(path.with_suffix('.npz')) as z:
        if projected:
            e.pilot,_=U.Pilot.reconstruct(e.raw,e.axes,e.model)
            assert {k:U.R.digest(x) for k,x in e.pilot.items()}==d['pilot_hash']
            vectors=np.load(ROOT/'projected'/'directions.npz')['vectors']
        else:
            e.pilot={k:z['anchor_pilot_'+k] for k in e.raw}
        measured=U.matrices(e,e.raw)
        pilot=U.matrices(e,e.pilot,correct=False)
        for b,m in measured.items():
            np.testing.assert_allclose(m@vectors if projected else m,z[b+'_observed'],rtol=2e-10,atol=1e-12)
        domain=992702 if projected else 984102
        rng=np.random.default_rng(np.random.SeedSequence([domain,e.system_id,e.pid,seed]))
        hashes=d['draw_hashes'] if projected else d['bootstrap_field_sha256']
        for i,h in enumerate(hashes):
            f,_=e.draw(rng)
            assert {k:U.R.digest(x) for k,x in f.items()}==h
            if i not in (0,31,32,len(hashes)-1):
                continue
            model=d['bootstrap_models'][i] if projected else d['refits'][i]['model']
            ms=U.matrices(e,f,model)
            for b,m in ms.items():
                err=m-pilot[b]
                replay=err@vectors if projected else err.T@err
                stored=z[b+('_errors' if projected else '_bootstrap_grams')][i]
                np.testing.assert_allclose(replay,stored,rtol=2e-9,atol=3e-12)
    return dict(seed=seed,profile=profile,raw=raw,draws=len(hashes),matrix_draws=4)


if __name__=='__main__':
    project='--projected' in sys.argv
    paths=(ROOT/'projected').glob('*_*.json') if project else (ROOT/'fresh'/'results').glob('*.json')
    paths=sorted(p for p in paths if p.stem[-1].isdigit())
    assert len(paths)==6
    rows=[replay(p,project) for p in paths]
    name='projected' if project else 'fresh'
    (ROOT/f'{name}_replay.json').write_text(json.dumps(dict(status='PASS',rows=rows),indent=2))
    print('Replayed',len(rows),'observations and',sum(r['draws'] for r in rows),'bootstrap field hashes.')
