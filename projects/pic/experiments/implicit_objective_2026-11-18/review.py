import csv
import json
from collections import Counter
from pathlib import Path
from scipy.linalg import svd
import numpy as np
from records import ROOT,SEEDS,SYSTEMS,sha
from implicit import projective_error
from report import front
from time_followup import CONFIRM

ARMS={'current':('legacy','dyad'),'symmetric_score_only':('denominator_only','dyad'),
      'local_implicit_old_dyad':('implicit','dyad'),'local_implicit':('implicit','symmetric_dyad'),
      'augmented_old_dyad':('augmented','dyad'),'augmented':('augmented','symmetric_dyad')}


def load(p):
    return json.loads(p.read_text())


def audit_fit(m,l,r,s,fold):
    a=np.vstack((m.real,m.imag)) if np.iscomplexobj(m) else m
    col=np.linalg.norm(a,axis=0)
    if min(col)==0:
        assert fold['augmented_status']=='zero training column'
        return
    scaled=a/col
    _,sig,vt=svd(scaled,full_matrices=len(a)<len(col),lapack_driver='gesvd')
    sig=np.pad(sig,(0,len(col)-len(sig)))
    unique=len(col)==1 or sig[-2]-sig[-1]>64*np.finfo(float).eps*max(a.shape)*sig[0]
    assert unique==(fold['augmented_status']=='ok')
    if not unique:
        return
    vector=np.array(fold['augmented_vector'])
    angle=projective_error(vector,vt[-1]/col,col)
    assert angle<3e-7,angle
    v=vector/np.linalg.norm(s*vector)
    via_gram=float(v@(l.T@r)@v)
    np.testing.assert_allclose(via_gram,fold['augmented'],rtol=3e-8,atol=3e-9)


def main():
    slots=[];coeff=[];stats=[];audited=0
    for seed in (*SEEDS,*CONFIRM):
        for name in SYSTEMS:
            path=ROOT/'augmented'/f'{name}_{seed}.json';d=load(path)
            old=ROOT/'results'/path.name
            zp=ROOT/'records'/path.with_suffix('.npz').name
            assert sha(old)==d['baseline'] and sha(path.with_suffix('.npz'))==d['blocks']
            assert sha(zp)==load(old)['record_hash']
            meta=load(zp.with_suffix('.json'))
            stats.append({k:d[k] for k in ('system','seed','checks','max_angle','max_score')})
            with np.load(zp) as archive, np.load(path.with_suffix('.npz')) as blocks:
                mats=[np.vstack((archive[f'train{g}']/np.sqrt(len(archive[f'train{g}'])),
                                 blocks[f'block{g}']/np.sqrt(len(blocks[f'block{g}'])))) for g in range(3)]
                lr=[(archive[f'left{g}'],archive[f'right{g}'],archive[f'scale{g}']) for g in range(3)]
                for r,c in zip(d['rows'],meta['candidates']):
                    ids=r['ids'];assert sorted(c['xi']+[c['yi']])==ids
                    for g,f in enumerate(r['folds']):
                        l,h,s=lr[g]
                        audit_fit(mats[g][:,ids],l[:,ids],h[:,ids],s[ids],f);audited+=1
                        if r['label']=='reference':
                            coeff.append(dict(system=name,seed=seed,slot=r['slot'],fold=g,old=f['old_angle'],
                                              local=f['implicit_angle'],augmented=f.get('augmented_angle')))
            for slot in sorted({r['slot'] for r in d['rows']}):
                rs=[r for r in d['rows'] if r['slot']==slot]
                slots.append(dict(system=name,seed=seed,slot=slot,
                                  arms={arm:front(rs,x,y) for arm,(x,y) in ARMS.items()}))
    results={}
    for tag,seeds in [('replay',[21]),('screen',SEEDS[1:]),('confirmation',CONFIRM)]:
        rows=[r for r in slots if r['seed'] in seeds]
        per={}
        for method in ARMS:
            rr=[r['arms'][method] for r in rows]
            per[method]=dict(registered=np.sum([r['registered'] for r in rr],axis=0).tolist(),
                             reference=np.sum([r['reference'] for r in rr],axis=0).tolist(),
                             unresolved=sum(len(r['unknown']) for r in rr),
                             mean_front=float(np.mean([len(r['known_front']) for r in rr])))
        cc=[r for r in coeff if r['seed'] in seeds]
        per['median_angle_degrees']={k:float(np.degrees(np.median([r[k] for r in cc if r[k] is not None])))
                                     for k in ('old','local','augmented')}
        per['reference_folds_used']={k:sum(r[k] is not None for r in cc) for k in ('old','local','augmented')}
        per['changes']=[dict(system=r['system'],seed=r['seed'],slot=r['slot'],
                             old=r['arms']['current']['registered'],new=r['arms']['augmented']['registered'],
                             old_front=r['arms']['current']['known_front'],new_front=r['arms']['augmented']['known_front']) for r in rows
                        if r['arms']['current']['registered']!=r['arms']['augmented']['registered']]
        results[tag]=per
    out=dict(by_cohort=results,slots=slots,folds_audited=audited,orientations=sum(r['checks'] for r in stats),
             max_angle=max(r['max_angle'] for r in stats),max_score=max(r['max_score'] for r in stats))
    (ROOT/'comparison.json').write_text(json.dumps(out,indent=2))
    with (ROOT/'coefficients_by_fold.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,coeff[0]);w.writeheader();w.writerows(coeff)
    for key,v in results.items():
        print(key,v['current'],v['augmented'],v['median_angle_degrees'])
        print('changes',[(r['system'],r['seed'],r['slot'],r['new']) for r in v['changes']])
    print('audited',audited,'orientation checks',out['orientations'],'errors',out['max_angle'],out['max_score'])


if __name__=='__main__':
    main()
