from records import LEGACY, ROOT, SEEDS, SYSTEMS, sha
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import json
import sys
import numpy as np
from implicit import fit, score, projective_error


def stability(matrix,cols,target,shape,K):
    if len(cols)==0:
        return 0.
    values=K.R.SV.dyad_scores(matrix[:,cols],matrix[:,target],np.ones(len(matrix)),shape,fit_intercept=False)
    return float(np.sum(values))


def run(name_seed):
    name,seed=name_seed
    sys.path.insert(0,str(LEGACY))
    import experiment as E
    path=ROOT/'records'/f'{name}_{seed}.npz'
    meta=json.loads(path.with_suffix('.json').read_text())
    assert sha(path)==meta['archive']
    candidates=meta['candidates']
    E.K.common_width(max(len(c['xi']) for c in candidates))
    checks=0;max_distance=0.;max_score_error=0.
    rows=[];metric_cache={}
    with np.load(path) as archive:
        data={k:archive[k] for k in archive.files}
    for j,c in enumerate(candidates):
        original=[c['yi'],*c['xi']]
        ids=sorted(original)
        assert len(set(ids))==len(ids)
        target=ids.index(c['yi'])
        matrix=data['dyad_matrix']
        original_dyad=stability(matrix,c['xi'],c['yi'],meta['shape'],E.K)
        key=tuple(ids)
        if key not in metric_cache:
            metric_cache[key]=[stability(matrix,[k for k in ids if k!=t],t,meta['shape'],E.K) for t in ids]
        pivot_dyads=metric_cache[key]
        stable=float(np.median(pivot_dyads))
        folds=[]
        for g in range(3):
            left,right=[data[f'{side}{g}'][:,ids] for side in ('left','right')]
            scale=data[f'scale{g}'][ids]
            train=data[f'train{g}'][:,ids]
            old=data['old_coefficients'][g,ids,j]
            assert old[target]==1.
            legacy=float((left@old)@(right@old)/scale[target]**2)
            ablation=score(left,right,scale,old)
            vector,status=fit(train)
            value=None if vector is None else score(left,right,scale,vector)
            fold=dict(legacy=legacy,denominator_only=ablation,implicit=value,status=status,
                      vector=None if vector is None else vector.tolist())
            if vector is not None and value is not None:
                # Refit across every written pivot. Change units and signs at the
                # same time, map back, and independently compare law and score.
                units=np.geomspace(.03,31.,len(ids))*(-1.)**np.arange(len(ids))
                for i in range(len(ids)):
                    order=np.roll(np.arange(len(ids)),-i)
                    oriented,s=fit((train*units)[:,order])
                    assert s==status
                    restored=(oriented[np.argsort(order)]*units)
                    distance=projective_error(vector,restored,np.linalg.norm(train,axis=0))
                    expected=score((left*units)[:,order],(right*units)[:,order],
                                   (scale*abs(units))[order],oriented)
                    delta=abs(value-expected)/max(1.,abs(value),abs(expected))
                    max_distance=max(max_distance,distance);max_score_error=max(max_score_error,delta)
                    checks+=1
            if c['label']=='reference':
                truth=np.zeros(len(ids));truth[target]=1.
                for k,b in zip(c['xi'],c['exact']):
                    truth[ids.index(k)]=-b
                unit=data[f'clean_scale{g}'][ids]
                fold['old_angle']=projective_error(old,truth,unit)
                fold['implicit_angle']=None if vector is None else projective_error(vector,truth,unit)
            folds.append(fold)
        rows.append(dict(id=c['id'],slot=c['slot'],registered=c['registered'],label=c['label'],ids=ids,
                         dyad=original_dyad,symmetric_dyad=stable,pivot_dyads=pivot_dyads,folds=folds))
    replay=None
    if seed==21:
        p=ROOT.parent/'contrast_stability_2026-11-17'/'results'/f'{name}_seed21.npz'
        with np.load(p) as z:
            ours=np.array([np.mean([f['legacy'] for f in r['folds']]) for r in rows])
            saved=z['observed_discrepancy'][0]
            np.testing.assert_allclose(ours,saved,rtol=5e-7,atol=2e-8)
            np.testing.assert_allclose([r['dyad'] for r in rows],z['observed_dyad'],rtol=2e-9,atol=2e-10)
            replay=float(np.max(abs(ours-saved)/np.maximum(1,abs(saved))))
    out=dict(system=name,seed=seed,record_hash=meta['archive'],checks=checks,
             max_direction_difference=max_distance,max_relative_score_difference=max_score_error,
             replay_error=replay,rows=rows)
    (ROOT/'results'/f'{name}_{seed}.json').write_text(json.dumps(out,allow_nan=False))
    return name,seed


if __name__=='__main__':
    (ROOT/'results').mkdir(exist_ok=True)
    with ProcessPoolExecutor(max_workers=8) as pool:
        futures=[pool.submit(run,(name,seed)) for seed in SEEDS for name in SYSTEMS]
        for f in as_completed(futures):
            print(f.result(),flush=True)
