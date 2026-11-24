import run_replica as T
from pathlib import Path
import json,hashlib,csv


def main():
    data=[json.loads(p.read_text()) for p in (T.ROOT/'results').glob('*_seed*.json')]
    assert len(data)==12
    assert {(d['system'],d['seed']) for d in data}=={(n,s) for n in ['ac','wave','kdv_cossin','ks'] for s in range(3)}
    rows=[]
    for d in data:
        assert d['source_sha256']=={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(T.__file__),Path(T.Q.__file__)]}
        files=dict(local_result=T.L.ROOT/'results'/f"{d['system']}_seed{d['seed']}.json",
                   training_plan=T.F.ROOT/'fit_results'/f"{d['system']}_seed{d['seed']}.json")
        assert d['input_sha256']=={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in files.items()}
        for method,entry in d['aggregates'].items():
            i=next(i for i,c in enumerate(d['candidates']) if c['label']=='reference');fit=method.split('_',1)[1]
            for metric in entry.values():assert T.np.isfinite(metric['scores']).all()
            rows.append(dict(system=d['system'],seed=d['seed'],method=method,
                retained=bool(T.A.frontier(entry['cross']['scores'],d['instability'],d['candidates'],0)),
                frozen_clean_energy=d['aggregates']['legacy_'+fit]['clean_energy']['scores'][i],
                clean_replica_energy=float(T.np.mean([f['clean_replica_control'][i]['clean_energy'] for f in d['folds']])),
                negative_fits=sum(f['arms'][method][i]['fit'].get('negative',0)>0 for f in d['folds'])))
    summary={m:sum(r['retained'] for r in rows if r['method']==m) for m in sorted({r['method'] for r in rows})}
    with (T.ROOT/'per_case.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    T.R.dump(T.ROOT/'summary.json',summary)
    T.R.dump(T.ROOT/'audit.json',dict(cases=12,hashes_verified=True,finite_scores=True,
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in T.ROOT.glob('*.py')}))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
