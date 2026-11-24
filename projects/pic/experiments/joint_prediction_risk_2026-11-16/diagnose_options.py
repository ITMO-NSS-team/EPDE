"""Post-selection clean controls; never imported by the fitting/selection code."""
import run_joint as J
import json,csv,hashlib,argparse
from pathlib import Path


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--systems',nargs='+',default=['ac','wave','kdv_cossin','ks'])
    args=ap.parse_args();rows=[];inputs={}
    for name in args.systems:
        for seed in range(3):
            path=J.ROOT/'results'/f'{name}_seed{seed}.json';inputs[path.name]=hashlib.sha256(path.read_bytes()).hexdigest()
            d=json.loads(path.read_text());_,old=J.F.baseline(name,seed)
            p=J.R.datasets.load(name);axes=J.R.axes_of(p);cs,labels=J.R.panel(p);terms,opaque=J.R.symbols(p,labels)
            plans=json.loads((J.F.ROOT/'fit_results'/path.name).read_text())
            frozen=J.F.evaluation_grams(p,old,terms,opaque,labels)
            for fold in d['folds']:
                g=fold['training'];segment,ax,opq,plan=J.Q.mapped_plan(old,g,plans['plans'][g]['plan'],axes,opaque)
                bank=J.Q.Bank(ax,terms,opq,plan=plan)
                clean=bank.matrix({v:y[segment] for v,y in p.data.items()});gram=J.grams(clean)
                for i,c in enumerate(cs):
                    if c['label']!='reference':continue
                    cid=c['id'];xi,yi=c['xi'],c['yi']
                    chosen=next(v['selected']['all'] for v in fold['choices'] if v['candidate']==cid and v['draws']==128)
                    for option,e in fold['options'][i]['options'].items():
                        v=J.np.zeros(len(labels));v[yi]=1.;v[xi]-=e['coefficients']
                        local=float(v@gram@v/gram[yi,yi])
                        legacy=float(v@frozen[f'{g}_loo_oracle']@v/frozen[f'{g}_loo_clean_scale'][yi])
                        risk=e['diagnostics']['128']
                        row=dict(system=name,seed=seed,training=g,slot=c['slot'],candidate=cid,option=option,selected=option==chosen,
                            risk=risk['risk'],apparent=risk['apparent'],optimism=risk['optimism'],optimism_mcse=risk['optimism_mcse'],
                            local_clean_energy=local,frozen_clean_energy=legacy,
                            coefficients=json.dumps(e['coefficients']),exact_coefficients=json.dumps(c['exact']))
                        assert J.np.isfinite([local,legacy,risk['risk']]).all();rows.append(row)
    out=J.ROOT/'option_clean_diagnostics.csv'
    with out.open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    J.R.dump(J.ROOT/'option_clean_diagnostics_audit.json',dict(system_runs=len(inputs),option_rows=len(rows),
        input_sha256=inputs,source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),post_selection_only=True))
    for name in args.systems:
        rr=[r for r in rows if r['system']==name and r['seed']==0 and r['training']==0]
        print(name,json.dumps(rr,indent=2))


if __name__=='__main__':main()
