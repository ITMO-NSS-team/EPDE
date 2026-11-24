import run_broad as B
from pathlib import Path
import json,hashlib,csv


def main():
    data=[json.loads(p.read_text()) for p in (B.ROOT/'results').glob('*_seed*.json')]
    assert len(data)==12
    assert {(d['system'],d['seed']) for d in data}=={(n,s) for n in ['ac','wave','kdv_cossin','ks'] for s in range(3)}
    rows=[];selections=[]
    for d in data:
        assert d['source_sha256']=={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(B.__file__),Path(B.Q.__file__)]}
        for f in d['folds']:
            for q,m in f['selection'].items():selections.append(dict(system=d['system'],seed=d['seed'],training=f['training'],extra_smoothness=q,feasible=m['feasible'],k=m.get('k')))
        for method,entry in d['aggregates'].items():
            i=next(i for i,c in enumerate(d['candidates']) if c['label']=='reference');fit=method.split('_',1)[1]
            for metric in entry.values():assert B.np.isfinite(metric['scores']).all()
            rows.append(dict(system=d['system'],seed=d['seed'],method=method,
                retained=bool(B.A.frontier(entry['cross']['scores'],d['instability'],d['candidates'],0)),
                frozen_clean_energy=d['aggregates']['legacy_'+fit]['clean_energy']['scores'][i],
                negative_fits=sum(f['arms'][method][i]['fit'].get('negative',0)>0 for f in d['folds'])))
    summary={m:dict(available_cases=sum(r['method']==m for r in rows),retained=sum(r['retained'] for r in rows if r['method']==m)) for m in sorted({r['method'] for r in rows})}
    for name,rr in [('per_case.csv',rows),('selected_banks.csv',selections)]:
        with (B.ROOT/name).open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(rr[0]));w.writeheader();w.writerows(rr)
    B.R.dump(B.ROOT/'summary.json',summary)
    B.R.dump(B.ROOT/'audit.json',dict(cases=12,source_hashes_verified=True,finite_scores=True,
        feasible_fold_arms=sum(s['feasible'] for s in selections),fold_arms=len(selections),
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in B.ROOT.glob('*.py')}))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
