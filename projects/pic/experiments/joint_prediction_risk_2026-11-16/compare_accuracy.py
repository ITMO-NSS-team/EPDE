"""Summarize frozen, post-selection accuracy and unchanged failure cases."""
import run_joint as J
import csv,json
from collections import Counter


def main():
    rows=list(csv.DictReader((J.ROOT/'core_per_equation.csv').open()))
    index={(r['system'],r['seed'],r['slot'],r['method']):r for r in rows}
    cases=[]
    for r in rows:
        if r['method']!='local_joint_all_full':continue
        b=index[r['system'],r['seed'],r['slot'],'local_fixed_local_clsm']
        old=float(b['frozen_clean_energy']);new=float(r['frozen_clean_energy'])
        status='unchanged' if old==new else 'improved' if new<old else 'worsened'
        cases.append(dict(system=r['system'],seed=int(r['seed']),slot=int(r['slot']),baseline_energy=old,
            joint_energy=new,ratio=new/old,change=new-old,status=status,
            baseline_retained=bool(b['cross_valid_front']),joint_retained=bool(r['cross_valid_front'])))
    groups=[]
    for name,slot in sorted({(r['system'],r['slot']) for r in cases}):
        rr=[r for r in cases if r['system']==name and r['slot']==slot]
        groups.append(dict(system=name,slot=slot,baseline_median=float(J.np.median([r['baseline_energy'] for r in rr])),
            joint_median=float(J.np.median([r['joint_energy'] for r in rr])),
            baseline_retained=sum(r['baseline_retained'] for r in rr),joint_retained=sum(r['joint_retained'] for r in rr),
            changes=dict(Counter(r['status'] for r in rr))))
    choices=list(csv.DictReader((J.ROOT/'core_choices.csv').open()))
    refs=[r for r in choices if r['candidate'].endswith(':reference')]
    summary=dict(cases=len(cases),accuracy_changes=dict(Counter(r['status'] for r in cases)),
        retention_changes=[r for r in cases if r['baseline_retained']!=r['joint_retained']],
        failures=[r for r in cases if not r['joint_retained']],
        reference_fold_choices=dict(Counter(r['selected'] for r in refs)),
        reference_half_full_changes=sum(r['selected']!=r['half_selected'] for r in refs),
        worst_ratio_regressions=sorted([r for r in cases if r['status']=='worsened'],key=lambda r:r['ratio'],reverse=True)[:5],
        best_absolute_improvements=sorted(cases,key=lambda r:r['change'])[:5],per_slot=groups)
    for name,data in [('accuracy_comparison.csv',cases),('accuracy_per_slot.csv',groups)]:
        with (J.ROOT/name).open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
    J.R.dump(J.ROOT/'accuracy_summary.json',summary)
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
