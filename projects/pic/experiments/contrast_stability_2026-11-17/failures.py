"""Exploratory attribution of primary Q losses; no selection rule changes."""
import metric as C
E,R,np,ROOT=C.E,C.R,C.np,C.ROOT
import json,csv
from collections import Counter


def main():
    changes=json.loads((ROOT/'changes.json').read_text());lost=[r for r in changes if r['arm']=='Q_bias' and r['change']=='loss']
    rows=[];cases=[]
    for loss in lost:
        name,seed,slot=loss['system'],loss['seed'],loss['slot'];path=ROOT/'results'/f'{name}_seed{seed}.json'
        meta=json.loads(path.read_text());a=np.load(path.with_suffix('.npz'));f=json.loads((ROOT/'frozen'/path.name).read_text());cs=meta['candidates']
        ref=next(j for j,c in enumerate(cs) if c['slot']==slot and c['label']=='reference');rc=cs[ref]
        d,q=a['observed_discrepancy'][0],a['observed_q_bias'];v=np.stack([d,q],axis=-1)
        dominators=[j for j,c in enumerate(cs) if c['slot']==slot and np.all(v[j]<=v[ref]) and np.any(v[j]<v[ref])]
        assert dominators
        case=dict(system=name,seed=seed,slot=slot,reference=rc['id'],reference_features=len(rc['xi']),dominators=len(dominators),
            any_larger=any(len(cs[j]['xi'])>len(rc['xi']) for j in dominators),
            any_reference_superset=any(cs[j]['yi']==rc['yi'] and set(rc['xi'])<set(cs[j]['xi']) for j in dominators),
            any_deficient=any(a['observed_deficient'][j]>0 for j in dominators))
        cases.append(case)
        for j in dominators:
            c=cs[j];superset=c['yi']==rc['yi'] and set(rc['xi'])<set(c['xi'])
            row=dict(system=name,seed=seed,slot=slot,dominator=c['id'],reference=rc['id'],reference_features=len(rc['xi']),features=len(c['xi']),
                reference_D=float(d[ref]),dominator_D=float(d[j]),reference_Q=float(q[ref]),dominator_Q=float(q[j]),
                reference_dyad=float(a['observed_dyad'][ref]),dominator_dyad=float(a['observed_dyad'][j]),
                reference_superset=superset,deficient_slabs=int(a['observed_deficient'][j]))
            if superset:
                rg=[g for g in f['groups'] if g['candidate']==ref];dg=[g for g in f['groups'] if g['candidate']==j];ratios=[]
                for rgroup,dgroup in zip(rg,dg):
                    assert (rgroup['axis'],rgroup['level'],rgroup['slabs'])==(dgroup['axis'],dgroup['level'],dgroup['slabs'])
                    rv=a['variance'][rgroup['start']:rgroup['stop']].reshape(-1,len(rc['xi']))
                    dv=a['variance'][dgroup['start']:dgroup['stop']].reshape(-1,len(c['xi']))[:,[c['xi'].index(i) for i in rc['xi']]]
                    valid=rv>0;ratios.extend((dv[valid]/rv[valid]).tolist())
                row['shared_feature_median_variance_ratio']=float(np.median(ratios))
            rows.append(row)
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (ROOT/'loss_dominators.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rows)
    supersets=[r for r in rows if r['reference_superset']]
    out=dict(scope='Exploratory attribution on completed fresh-anchor results; no new selection rule.',
        lost_cases=len(cases),gained_cases=sum(r['arm']=='Q_bias' and r['change']=='gain' for r in changes),
        cases_with_larger_dominator=sum(r['any_larger'] for r in cases),cases_with_reference_superset=sum(r['any_reference_superset'] for r in cases),
        cases_with_deficient_dominator=sum(r['any_deficient'] for r in cases),
        superset_dominator_pairs=len(supersets),superset_pairs_with_increased_shared_variance=sum(r['shared_feature_median_variance_ratio']>1 for r in supersets),
        median_shared_feature_variance_ratio=float(np.median([r['shared_feature_median_variance_ratio'] for r in supersets])),
        losses_by_system=dict(Counter(r['system'] for r in cases)),cases=cases,
        source_sha256={__file__:E.S.sha(__file__)},
        input_sha256={str(p):E.S.sha(p) for p in [ROOT/'changes.json',*sorted((ROOT/'results').glob('*.*')),*sorted((ROOT/'frozen').glob('*.json'))]})
    R.dump(ROOT/'failure_summary.json',out);print(json.dumps({k:v for k,v in out.items() if k not in ('cases','source_sha256','input_sha256')},indent=2))


if __name__=='__main__':main()
