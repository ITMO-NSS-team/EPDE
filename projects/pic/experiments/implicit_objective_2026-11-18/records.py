"""Rebuild the established weak measurements without running a candidate equation."""
import os
for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[name] = '1'
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parent
LEGACY = ROOT.parent / 'uncertainty_refit_2026-11-16'
SYSTEMS = 'ac burgers burgers_inviscid duffing kdv kdv_cossin ks lorenz lv ns ode pde_compound pde_divide vdp wave'.split()
SEEDS = (21, 6101, 6102, 6103)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(job):
    sys.path.insert(0, str(LEGACY))
    import experiment as E
    name, seed = job
    e = E.Experiment(name, seed)
    points = []
    for idx, st in zip(e.indices, e.states):
        mapped = E.O.map_values(st['map'], {v:y[idx] for v,y in e.raw.items()})
        points.append({key:e.basis[key].convert(E.O.matrix(bank,mapped,st['cov']))
                       for key,bank in e.vbanks.items()})
    data = {'old_coefficients':e.anchor_vectors, 'dyad_matrix':e.dbank.matrix(e.raw)}
    clean_scales = []
    for g, (idx, st) in enumerate(zip(e.indices, e.states)):
        obs = {v:y[idx] for v,y in e.raw.items()}
        pilot,_ = E.R.reconstruct(obs,e.axes,idx)
        segment,axes,opaque,plan = E.Q.mapped_plan(e.old,g,e.local['plans'][g]['plan'],e.axes,e.opaque)
        local = E.Q.Bank(axes,e.terms,opaque,plan=plan)
        data[f'train{g}'] = local.matrix({v:y[segment] for v,y in pilot.items()})
        key = e.keys[g]
        valpilot = e.basis[key].convert(e.vbanks[key].matrix({v:y[e.vseg] for v,y in pilot.items()}))
        scales = np.linalg.norm(st['white']@valpilot,axis=0)
        np.testing.assert_allclose(np.square(scales[[c['yi'] for c in e.cs]]),st['scale'],rtol=1e-9)
        data[f'scale{g}'] = scales
        left,right = [h for h in range(3) if h!=g]
        data[f'left{g}'] = st['white']@points[left][key]
        data[f'right{g}'] = st['white']@points[right][key]
        clean_scales.append((local,segment))
    # Only downstream coefficient diagnostics use these columns.
    for g,(bank,seg) in enumerate(clean_scales):
        data[f'clean_scale{g}'] = np.linalg.norm(bank.matrix({v:y[seg] for v,y in e.p.data.items()}),axis=0)
    dest=ROOT/'records'/f'{name}_{seed}.npz'
    np.savez_compressed(dest,**data)
    meta=dict(system=name,seed=seed,candidates=e.cs,labels=e.labels,shape=e.dshape,
              inputs=e.inputs,raw_hash={v:E.R.digest(y) for v,y in e.raw.items()},
              archive=sha(dest),legacy_sources=E.sources(),preparation=sha(Path(__file__)))
    E.R.dump(dest.with_suffix('.json'),meta)
    return dest.stem


if __name__=='__main__':
    (ROOT/'records').mkdir(exist_ok=True)
    with ProcessPoolExecutor(max_workers=8) as pool:
        jobs=[(s,n) for n in SEEDS for s in SYSTEMS]
        futures=[pool.submit(build,j) for j in jobs if not (ROOT/'records'/f'{j[0]}_{j[1]}.json').exists()]
        for f in as_completed(futures):
            print(f.result(),flush=True)
