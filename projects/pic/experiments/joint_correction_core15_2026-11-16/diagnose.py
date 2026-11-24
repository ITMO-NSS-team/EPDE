"""Post-hoc selection/outer/clean agreement; no change to choices or fits."""
import run_case as C
R,np,ROOT=C.R,C.np,C.ROOT
import json,csv
from collections import Counter


def main():
    rows=[];details=[]
    for path in sorted((ROOT/'results').glob('*.json')):
        d=json.loads(path.read_text());refs={c['id'] for c in d['candidates'] if c['label']=='reference'}
        for g,fold in enumerate(d['folds']):
            for decision in fold['decisions']:
                if decision['candidate'] not in refs or decision['status']!='ok':continue
                rule=decision['rule'];valid={r['option'] for r in decision['options'] if r['status']=='ok' and r['inner'] is not None}
                options=[r for r in fold['rows'] if r['candidate']==decision['candidate'] and r['option'] in valid]
                selected=next(r for r in options if r['option']==decision['option'])
                best=min(options,key=lambda r:r['frozen_clean_energy']);clean=min(options,key=lambda r:r['clean_'+rule]);outer=min(options,key=lambda r:r['outer_'+rule])
                row=dict(system=d['system'],seed=d['seed'],training=g,candidate=decision['candidate'],rule=rule,selected=selected['option'],
                    available=len(options),chosen_energy=selected['frozen_clean_energy'],best_energy=best['frozen_clean_energy'],
                    ratio_to_best=selected['frozen_clean_energy']/max(best['frozen_clean_energy'],np.finfo(float).tiny),
                    selects_best=selected['option']==best['option'],outer_agrees=outer['option']==selected['option'],
                    clean_metric_selects_best=clean['option']==best['option'],outer_minus_inner=selected['outer_'+rule]-selected['inner_'+rule])
                rows.append(row)
                if rule=='mixed_transfer' and row['ratio_to_best']>10:
                    details.append(dict(**row,options=[{k:v for k,v in r.items() if k in ('option','coefficients','frozen_clean_energy','fit') or k.endswith(rule)} for r in options]))
    summaries=[]
    for name in C.SYSTEMS:
        for rule in C.RULES:
            rr=[r for r in rows if r['system']==name and r['rule']==rule]
            summaries.append(dict(system=name,rule=rule,reference_folds=len(rr),selects_best=sum(r['selects_best'] for r in rr),
                outer_agrees=sum(r['outer_agrees'] for r in rr),clean_metric_selects_best=sum(r['clean_metric_selects_best'] for r in rr),
                median_ratio_to_best=float(np.median([r['ratio_to_best'] for r in rr]))))
    for name,rr in [('selection_diagnostics.csv',rows),('selection_diagnostic_summary.csv',summaries)]:
        with (ROOT/name).open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,list(rr[0]));writer.writeheader();writer.writerows(rr)
    R.dump(ROOT/'large_selection_errors.json',dict(post_hoc_diagnostic=True,threshold_for_listing_only=10,rows=details,
         source_sha256=C.S.sha(__file__)))
    print(json.dumps([r for r in summaries if r['system'] in ('ks','wave','kdv_cossin')],indent=2))


if __name__=='__main__':main()
