import run_information as I
from pathlib import Path
import json,hashlib,csv


def write_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    data=[json.loads(p.read_text()) for p in (I.ROOT/'results').glob('*_seed*.json')]
    assert len(data)==12
    assert {(d['system'],d['seed']) for d in data}=={(n,s) for n in ['ac','wave','kdv_cossin','ks'] for s in range(3)}
    rows=[];choices=[];changes=0;failures=0
    for d in data:
        assert d['source_sha256']=={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path(I.__file__),Path(I.Q.__file__)]}
        paths=dict(local=I.L.ROOT/'results'/f"{d['system']}_seed{d['seed']}.json",broad=I.B.ROOT/'results'/f"{d['system']}_seed{d['seed']}.json",
            continuous=I.Q.ROOT/'unit_results'/f"{d['system']}_seed{d['seed']}.json",plan=I.F.ROOT/'fit_results'/f"{d['system']}_seed{d['seed']}.json")
        assert d['input_sha256']=={k:hashlib.sha256(p.read_bytes()).hexdigest() for k,p in paths.items()}
        for f in d['folds']:
            failures+=len(f['selection_failures'])
            half={c['candidate']:c for c in f['choices'] if c['draws']==d['draws']//2}
            for c in f['choices']:
                if c['draws']!=d['draws']:continue
                changes+=c['choice']!=half[c['candidate']]['choice']
                choices.append(dict(system=d['system'],seed=d['seed'],training=f['training'],candidate=c['candidate'],
                    selected=c['choice'],half_selected=half[c['candidate']]['choice'],
                    local_score=c['scores'].get('local',{}).get('score'),broad0_score=c['scores'].get('broad0',{}).get('score'),
                    broad1_score=c['scores'].get('broad1',{}).get('score')))
        b=d['draws']
        for method,entry in d['aggregates'].items():
            i=next(i for i,c in enumerate(d['candidates']) if c['label']=='reference');fit=method.split('_',1)[1]
            normalized=method.replace(f'_{b}', '_full').replace(f'_{b//2}', '_half')
            for metric in entry.values():assert I.np.isfinite(metric['scores']).all()
            rows.append(dict(system=d['system'],seed=d['seed'],method=normalized,
                retained=bool(I.A.frontier(entry['cross']['scores'],d['instability'],d['candidates'],0)),
                frozen_clean_energy=d['aggregates']['legacy_'+fit]['clean_energy']['scores'][i]))
    summary={m:dict(cases=sum(r['method']==m for r in rows),retained=sum(r['retained'] for r in rows if r['method']==m)) for m in sorted({r['method'] for r in rows})}
    write_csv(I.ROOT/'per_case.csv',rows);write_csv(I.ROOT/'selected_banks.csv',choices)
    I.R.dump(I.ROOT/'summary.json',summary)
    controls=[json.loads(p.read_text()) for p in (I.ROOT/'error_budget').glob('*.json')]
    assert len(controls)==12
    error=max(r['decomposition_error'] for d in controls for r in d['rows'])
    I.R.dump(I.ROOT/'audit.json',dict(cases=12,source_and_input_hashes_verified=True,finite_scores=True,
        selection_failures=failures,half_full_bank_changes=changes,bank_choices=len(choices),
        error_budget_cases=12,max_decomposition_error=error,
        source_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in I.ROOT.glob('*.py')}))
    print(json.dumps(summary,indent=2));print('Bank changes half/full:',changes,'/',len(choices),'decomposition error',error)


if __name__=='__main__':main()
