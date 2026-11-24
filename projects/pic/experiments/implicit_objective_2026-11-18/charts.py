from collections import defaultdict
import json
import numpy as np
from records import ROOT,SEEDS,SYSTEMS,sha
from time_followup import CONFIRM
from report import front
from implicit import score,projective_error


def charts(mat,ids):
    m=np.vstack((mat.real,mat.imag)) if np.iscomplexobj(mat) else mat
    scales=np.linalg.norm(m,axis=0)
    assert min(scales)>0
    x=m/scales; k=len(ids)
    if k==1:
        return [np.ones(1)/scales],0,np.ones(1)/scales
    vectors=[];conditions=[]
    for i in range(k):
        rest=np.arange(k)!=i
        a,_,_,s=np.linalg.lstsq(x[:,rest],x[:,i],rcond=None)
        v=np.ones(k);v[rest]=-a
        vectors.append(v/np.linalg.norm(v))
        conditions.append(float(s[-1]) if x.shape[0]>=k-1 else 0.)
    # Numerically tied condition measures, including the two-term case,
    # use a stable term identity independent of input order and sign.
    tol=64*np.finfo(float).eps*max(m.shape)*max(conditions)
    pivot=min((i for i in range(k) if max(conditions)-conditions[i]<=tol),key=lambda i:ids[i])
    v=np.asarray(vectors)
    w,u=np.linalg.eigh(v.T@v/k)
    consensus=None if w[-1]-w[-2]<=64*np.finfo(float).eps*k else u[:,-1]/scales
    return [z/scales for z in v],pivot,consensus


def main():
    methods=('written_ols','condition_chart','consensus')
    slots=[];orientation_error=0.;count=0
    for seed in (*SEEDS,*CONFIRM):
        for name in SYSTEMS:
            path=ROOT/'augmented'/f'{name}_{seed}.npz'
            d=json.loads(path.with_suffix('.json').read_text()); meta=json.loads((ROOT/'records'/path.with_suffix('.json').name).read_text())
            with np.load(ROOT/'records'/path.name) as z,np.load(path) as a:
                blocks=[np.vstack((z[f'train{g}']/np.sqrt(len(z[f'train{g}'])),a[f'block{g}']/np.sqrt(len(a[f'block{g}'])))) for g in range(3)]
                lr=[(z[f'left{g}'],z[f'right{g}'],z[f'scale{g}']) for g in range(3)]
                for r,c in zip(d['rows'],meta['candidates']):
                    ids=r['ids']
                    for g,f in enumerate(r['folds']):
                        mat=blocks[g][:,ids];l,b,s=lr[g];s=s[ids]
                        fitted,pivot,cons=charts(mat,ids)
                        for key,v in zip(methods,(fitted[ids.index(c['yi'])],fitted[pivot],cons)):
                            f[key]=None if v is None else score(l[:,ids],b[:,ids],s,v)
                        units=np.geomspace(.1,7,len(ids))*(-1.)**np.arange(len(ids))
                        reverse=np.arange(len(ids))[::-1]
                        vv,p2,cc=charts((mat*units)[:,reverse],[ids[i] for i in reverse])
                        assert ids[pivot]==ids[reverse[p2]]
                        for v,v2 in [(fitted[pivot],vv[p2]),(cons,cc)]:
                            assert (v is None)==(v2 is None)
                            if v is not None:
                                orientation_error=max(orientation_error,projective_error(v,v2[reverse]*units,np.linalg.norm(mat,axis=0)))
                                count+=1
                for slot in set(r['slot'] for r in d['rows']):
                    rs=[r for r in d['rows'] if r['slot']==slot]
                    slots.append(dict(system=name,seed=seed,slot=slot,arms={k:front(rs,k,'symmetric_dyad') for k in (*methods,'augmented')}))
    totals={}
    for cohort,ss in [('screen',SEEDS[1:]),('confirmation',CONFIRM)]:
        rr=[r for r in slots if r['seed'] in ss]
        totals[cohort]={k:dict(registered=np.sum([r['arms'][k]['registered'] for r in rr],axis=0).tolist(),
                       exact=np.sum([r['arms'][k]['reference'] for r in rr],axis=0).tolist(),
                       ns_slot0=[r['arms'][k]['registered'] for r in rr if r['system']=='ns' and r['slot']==0]) for k in (*methods,'augmented')}
    out=dict(slots=slots,totals=totals,checks=count,max_orientation_angle=orientation_error)
    (ROOT/'chart_results.json').write_text(json.dumps(out,indent=2))
    print(totals,count,orientation_error)


if __name__=='__main__':
    main()
