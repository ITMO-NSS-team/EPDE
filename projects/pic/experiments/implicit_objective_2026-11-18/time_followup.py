from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import sys
from records import ROOT, LEGACY, SEEDS, SYSTEMS, build, sha
import numpy as np
from implicit import fit,projective_error,score
import benchmark

CONFIRM=(6111,6112,6113)


def extra_block(system,seed):
    sys.path.insert(0,str(LEGACY))
    import experiment as E
    problem=E.R.datasets.load(system);axes=E.R.axes_of(problem)
    cs,labels=E.R.panel(problem);terms,opaque=E.R.symbols(problem,labels)
    _,old=E.F.baseline(system,0)
    source=E.M.ROOT/'results'/f'{system}_seed0.json'
    config=json.loads(source.read_text())
    raw=problem.noisy(5,seed)
    seg=slice(*config['segment'])
    ax=[axes[0][seg],*axes[1:]]
    extra={k:v[seg] for k,v in opaque.items()}
    result=[]
    for g,group in enumerate(old['groups']):
        key=config['folds'][g]['selection']['q0']['key']
        plan=config['banks'][key]['plan']
        modes=np.r_[np.arange(5),-np.arange(1,5)]
        frequencies=[(modes if j==0 else np.array([0]))*np.pi/(m*(a[1]-a[0]))
                     for j,(a,m) in enumerate(zip(ax,plan['ms']))]
        bank=E.Q.Bank(ax,terms,extra,plan=plan,frequencies=frequencies)
        idx=np.asarray(group['indices'])
        pilot,_=E.R.reconstruct({v:y[idx] for v,y in raw.items()},axes,idx)
        result.append(bank.matrix({v:y[seg] for v,y in pilot.items()}))
    return result


def run(job):
    system,seed=job
    path=ROOT/'records'/f'{system}_{seed}.npz'
    if not path.with_suffix('.json').exists():
        build(job)
    baseline=ROOT/'results'/f'{system}_{seed}.json'
    if not baseline.exists():
        benchmark.run(job)
    before=json.loads(baseline.read_text())
    meta=json.loads(path.with_suffix('.json').read_text())
    extra=extra_block(*job)
    block_path=ROOT/'augmented'/f'{system}_{seed}.npz'
    np.savez_compressed(block_path,**{f'block{g}':a for g,a in enumerate(extra)})
    with np.load(path) as z:
        data={k:z[k] for k in z.files}
    max_err=0.;max_angle=0.;checks=0
    for r,c in zip(before['rows'],meta['candidates']):
        assert r['id']==c['id']
        ids=r['ids']
        for g,f in enumerate(r['folds']):
            a=data[f'train{g}'][:,ids];b=extra[g][:,ids]
            mat=np.vstack((a/np.sqrt(len(a)),b/np.sqrt(len(b))))
            v,status=fit(mat)
            left,right=[data[f'{key}{g}'][:,ids] for key in ('left','right')]
            scale=data[f'scale{g}'][ids]
            f['augmented']=None if v is None else score(left,right,scale,v)
            f['augmented_status']=status
            f['augmented_vector']=None if v is None else v.tolist()
            if v is None:
                continue
            units=np.geomspace(.03,31.,len(ids))*(-1.)**np.arange(len(ids))
            for i in range(len(ids)):
                order=np.roll(np.arange(len(ids)),-i)
                other,s=fit((mat*units)[:,order])
                assert s==status
                err=projective_error(v,other[np.argsort(order)]*units,np.linalg.norm(mat,axis=0))
                val=score((left*units)[:,order],(right*units)[:,order],(scale*abs(units))[order],other)
                max_angle=max(max_angle,err)
                max_err=max(max_err,abs(val-f['augmented'])/max(1.,abs(val),abs(f['augmented'])))
                checks+=1
            if c['label']=='reference':
                truth=np.zeros(len(ids));truth[ids.index(c['yi'])]=1.
                for k,x in zip(c['xi'],c['exact']):
                    truth[ids.index(k)]=-x
                f['augmented_angle']=projective_error(v,truth,data[f'clean_scale{g}'][ids])
    out=dict(system=system,seed=seed,rows=before['rows'],checks=checks,max_angle=max_angle,max_score=max_err,
             baseline=sha(baseline),blocks=sha(block_path))
    block_path.with_suffix('.json').write_text(json.dumps(out,allow_nan=False))
    return system,seed


if __name__=='__main__':
    (ROOT/'augmented').mkdir(exist_ok=True)
    with ProcessPoolExecutor(max_workers=8) as pool:
        futures=[pool.submit(run,(s,n)) for n in (*CONFIRM,*SEEDS) for s in SYSTEMS]
        for f in as_completed(futures):
            print(f.result(),flush=True)
