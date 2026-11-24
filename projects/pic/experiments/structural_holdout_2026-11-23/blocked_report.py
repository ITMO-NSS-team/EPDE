"""Summarize the true four-fold discovery runs separately from coefficient cross-validation."""
import csv
from pathlib import Path
from statistics import median

ROOT=Path(__file__).resolve().parent


def read(name):
    with (ROOT/name).open() as stream:
        return list(csv.DictReader(stream))


def report():
    runs=read('blocked_systems.csv')
    folds=read('blocked_folds.csv')
    choices=read('blocked_returns.csv')
    assert len(runs)==48 and len(folds)==192
    assert len({(r['dataset'],r['seed'],r['fold']) for r in folds})==192
    out=[]
    for name in ['suite',*sorted({r['dataset'] for r in runs})]:
        keep=lambda r:r['dataset']==name or (name=='suite' and r['dataset']!='ode_simple')
        systems=[r for r in runs if keep(r)]
        fold=[r for r in folds if keep(r)]
        eligible=[r for r in fold if r['status']=='completed']
        returns=[r for r in choices if keep(r)]
        row=dict(dataset=name,runs=len(systems),completed_folds=len(eligible),scheduled_folds=len(fold))
        for key in ('consensus_six','consensus_one','fold_train_hits','fold_held_hits','baseline_six_750',
                    'baseline_six_3000','matched_six_750','matched_six_3000','matched_one_3000'):
            row[key]=sum(int(r[key]) for r in systems)
        row['gains_vs_matched']=sum(r['consensus_six']=='1' and r['matched_six_3000']=='0' for r in systems)
        row['losses_vs_matched']=sum(r['consensus_six']=='0' and r['matched_six_3000']=='1' for r in systems)
        row['hit_in_any_fold_list']=sum(int(r['fold_held_hits'])>0 for r in systems)
        row['hit_in_every_fold_list']=sum(int(r['fold_held_hits'])==4 for r in systems)
        row['fold_hit_lost_in_consensus']=sum(int(r['fold_held_hits'])>0 and r['consensus_six']=='0' for r in systems)
        row['any_fold_assembled']=sum(int(r['folds_assembled'])>0 for r in systems) if name!='ode_simple' else None
        row['all_folds_assembled']=sum(int(r['folds_assembled'])==4 for r in systems) if name!='ode_simple' else None
        row['median_oof_wape_of_training_leader']=median(float(r['out_of_fold_wape']) for r in eligible) if eligible else None
        row['median_coverage_of_consensus_leader']=median(int(r['coverage']) for r in returns if r['rank']=='1') if returns else None
        out.append(row)
    with (ROOT/'blocked_by_dataset.csv').open('w',newline='',encoding='utf8') as stream:
        writer=csv.DictWriter(stream,fieldnames=out[0])
        writer.writeheader()
        writer.writerows(out)
    print(out[0])


if __name__=='__main__':
    report()
