import csv
import json
from collections import Counter,defaultdict
import numpy as np
from records import ROOT,SEEDS,SYSTEMS

ARMS={
    'current':('legacy','dyad'),
    'symmetric_score_only':('denominator_only','dyad'),
    'symmetric_score_and_dyad_old_fit':('denominator_only','symmetric_dyad'),
    'implicit_fit_and_score':('implicit','dyad'),
    'all_symmetric':('implicit','symmetric_dyad'),
}


def front(rows,score,dyad):
    good=[];unknown=[]
    for r in rows:
        values=[f[score] for f in r['folds']]
        if None in values:
            unknown.append(r)
        else:
            good.append((r,float(np.mean(values)),r[dyad]))
    ff=[r for r,d,i in good if not any(a<=d and b<=i and (a<d or b<i) for _,a,b in good)]
    def bounds(eligible):
        known=[r for r in ff if eligible(r)]
        lower=any(all(eligible(u) or u[dyad]>r[dyad] for u in unknown) for r in known)
        upper=bool(known) or any(eligible(r) for r in unknown)
        return int(lower),int(upper)
    return dict(registered=bounds(lambda r:r['registered']),
                reference=bounds(lambda r:r['label']=='reference'),known_front=[r['id'] for r in ff],
                unknown=[r['id'] for r in unknown])


def main():
    slots=[];angle_rows=[];diagnostics=[]
    for seed in SEEDS:
        for system in SYSTEMS:
            d=json.loads((ROOT/'results'/f'{system}_{seed}.json').read_text())
            diagnostics.append({k:v for k,v in d.items() if k!='rows'})
            for slot in sorted({r['slot'] for r in d['rows']}):
                rs=[r for r in d['rows'] if r['slot']==slot]
                fs={n:front(rs,*m) for n,m in ARMS.items()}
                slots.append(dict(system=system,seed=seed,slot=slot,arms=fs))
                ref=next(r for r in rs if r['label']=='reference')
                for g,f in enumerate(ref['folds']):
                    angle_rows.append(dict(system=system,seed=seed,slot=slot,fold=g,old=f['old_angle'],new=f['implicit_angle']))
    summaries={}
    for cohort in ('replay','fresh'):
        sub=[r for r in slots if (r['seed']==21)==(cohort=='replay')]
        result={}
        for arm in ARMS:
            fs=[r['arms'][arm] for r in sub]
            result[arm]=dict(slots=len(fs),registered=np.sum([f['registered'] for f in fs],axis=0).tolist(),
                reference=np.sum([f['reference'] for f in fs],axis=0).tolist(),
                mean_known_front_size=float(np.mean([len(f['known_front']) for f in fs])),
                unknown=sum(len(f['unknown']) for f in fs))
        angles=[r for r in angle_rows if (r['seed']==21)==(cohort=='replay') and r['new'] is not None]
        result['angles']=dict(paired_folds=len(angles),old_deg=float(np.degrees(np.median([r['old'] for r in angles]))),
                              new_deg=float(np.degrees(np.median([r['new'] for r in angles]))))
        summaries[cohort]=result
    numerics=dict(replays=sum(r['replay_error'] is not None for r in diagnostics),
        largest_replay_error=max(r['replay_error'] or 0. for r in diagnostics),
        orientations=sum(r['checks'] for r in diagnostics),
        largest_angle_difference=max(r['max_direction_difference'] for r in diagnostics),
        largest_score_difference=max(r['max_relative_score_difference'] for r in diagnostics))
    with (ROOT/'fit_angles.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,angle_rows[0]);w.writeheader();w.writerows(angle_rows)
    (ROOT/'summary.json').write_text(json.dumps(dict(summary=summaries,numerics=numerics,slots=slots),indent=2))
    print(json.dumps(summaries,indent=1));print(numerics)
    print('changed fully symmetric vs control:')
    for r in slots:
        if r['seed']!=21 and r['arms']['current']['registered']!=r['arms']['all_symmetric']['registered']:
            print(r['system'],r['seed'],r['slot'],r['arms']['current']['registered'],r['arms']['all_symmetric']['registered'])


if __name__=='__main__':
    main()
