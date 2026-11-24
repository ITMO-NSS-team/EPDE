"""Build matched training, validation, and omitted-recording statistics for the expanded search."""
import hashlib
import json
import os
from functools import lru_cache
from pathlib import Path

for key in ('OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'

import numpy as np

from expanded_terms import BOOK, GATES, IDS, TERMS, V, WARMUP, columns, history
from physics import CAP, CUTOFF, DT, LAG, OUT, geometry, input_at, motion_inputs, recordings, robot_parameters, rotate
from run import check_inputs

PROJECTED = os.environ.get('EPDE_COLLISION_GEOMETRY','raw') == 'projected'
DEST = OUT/'preprocessed'/'search' if PROJECTED else OUT/'expanded_factors'
CACHE = DEST/'statistics'


def calibration(held):
    meta = json.loads((OUT/held/'training.json').read_text())
    lookup = {tuple(map(int,k.split('/'))): np.array(v) for k,v in meta['lookup'].items()}
    return lookup, np.array(meta['common']), {int(k):np.array(v) for k,v in meta['arenas'].items()}


def moments(a):
    a = np.asarray(a,float).reshape(-1,a.shape[-1])
    return a.T @ a


def score_block(tr, rows, mem, params, role, record_index, observed=None):
    p = len(TERMS)
    g = np.zeros((p+1,p+1))
    normal, tangential = np.zeros_like(g), np.zeros_like(g)
    sy = sy2 = 0.
    count = pairs = 0
    screen = []
    if role == 'train':
        available = np.flatnonzero(np.repeat(mem[-1][...,None],2,axis=-1))
        rng = np.random.default_rng(22031 + record_index)
        edges = np.linspace(0,len(available),513,dtype=int)
        selected = available[rng.integers(edges[:-1],edges[1:])]
    for start in range(0,len(rows),32):
        slots = np.arange(start,min(start+32,len(rows)))
        origins = rows[slots]
        x, base, y, valid, (f,i,j,n,t), gates = columns(tr,origins,slots,mem,params,observed)
        if observed is not None and role != 'train':
            y = (observed.pos[origins+LAG]-observed.pos[origins])/DT
            if role == 'test':
                distances, axis = geometry(observed.pos[origins])
                f,i,j = np.nonzero(np.triu((distances>0)&(distances<CUTOFF),k=1))
                n = axis[f,i,j]
                t = rotate(n)
        if not start:
            # Independent replay of the original fourteen physical predictors.
            old, old_base, _ = motion_inputs(*input_at(tr,rows[slots]),robot_parameters(tr,params[0]),
                                              params[1],params[2][int(tr.tag[-2:])])
            ix = [1,2,81,82,20,35,48,100,101,15,16,17,18,19]
            scale = np.array([V,V,V,V,1,1,CAP/2,1,1,V,V,V,V,V])
            np.testing.assert_allclose(x[...,np.array(ix)-1], old*scale,rtol=1e-5,atol=.0002)
            np.testing.assert_allclose(base,old_base,rtol=1e-9,atol=1e-8)
        a = np.concatenate((x,(y-base)[...,None]),axis=-1)
        if not np.isfinite(a).all():
            raise ValueError(f'Nonfinite features in {tr.tag} at {rows[start]}')
        g += moments(a[valid])
        yy = y[valid]
        sy += yy.sum()
        sy2 += (yy**2).sum()
        count += yy.size
        if role == 'train':
            stride = len(tr.ids)*2
            idx = selected[(selected>=start*stride)&(selected<(slots[-1]+1)*stride)]-start*stride
            screen.append(np.column_stack([a.reshape(-1,p+1)[idx],
                          *[gates[name].ravel()[idx//2] for name in GATES]]))
        elif role == 'test':
            use = (i<j)&valid[f,i]&valid[f,j]
            relative = a[f[use],i[use]]-a[f[use],j[use]]
            for axis, target in [(n,normal),(t,tangential)]:
                target += moments(np.einsum('nc,nck->nk',axis[use],relative))
            pairs += use.sum()
    result = dict(g=g/count, count=count, variance=sy2/count-(sy/count)**2)
    if role=='train':
        result['sample'] = np.concatenate(screen).astype(np.float32)
        assert len(result['sample'])==512
    if role=='test':
        result.update(normal=normal/pairs,tangent=tangential/pairs,pairs=int(pairs))
    return result


@lru_cache(maxsize=1)
def fingerprint():
    files = [Path(__file__),Path(__file__).with_name('expanded_terms.py'),BOOK,
             Path(__file__).with_name('physics.py'), OUT/'inputs.json', *sorted(OUT.glob('*/training.json'))]
    if PROJECTED:
        files += [Path(__file__).with_name('projected_data.py'), *sorted((OUT/'preprocessed').glob('N*_D*.npz'))]
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.read_bytes())
    return digest.hexdigest()


def prepare():
    check_inputs()
    if PROJECTED:
        from projected_data import projected_tracks, fold_calibrations
        observations, tracks = projected_tracks()
        parameters = fold_calibrations(tracks)
    else:
        observations, tracks = {}, recordings()
        parameters = {tag:calibration(tag) for tag in tracks}
    CACHE.mkdir(exist_ok=True,parents=True)
    digest = fingerprint()
    for idx,(tag,track) in enumerate(tracks.items()):
        cut = len(track.pos)*3//4
        for role, start, stop, folds in [('train',0,cut,set(tracks)-{tag}),
                                         ('valid',cut,None,set(tracks)-{tag}),('test',0,None,[tag])]:
            tr = track.part(start,stop)
            observed = observations[tag].part(start,stop) if PROJECTED else None
            needed = []
            for held in sorted(folds):
                path = CACHE/f'{held}__{tag}__{role}.npz'
                if path.exists():
                    with np.load(path) as z:
                        if str(z['digest']) == digest:
                            continue
                needed.append((held,path))
            if not needed:
                continue
            rows = np.arange(WARMUP,len(tr.pos)-LAG,LAG)
            mem = history(tr,rows,observed)
            for held,path in needed:
                result = score_block(tr,rows,mem,parameters[held],role,idx,observed)
                tmp = path.with_suffix('.tmp.npz')
                np.savez_compressed(tmp, digest=digest, ids=IDS, **result)
                tmp.replace(path)
            print(tag,role,len(rows),'forecast frames;',len(needed),'calibrations',flush=True)


if __name__=='__main__':
    prepare()
