from context import *
from collections import Counter

def main():
    old=C.PREVIOUS;paths=list((old/'results').glob('*.json'))+list((PREVIOUS/'repair_results').glob('*.json'));assert len(paths)==72;rows=[]
    for path in sorted(paths):
        d=read(path);repaired=path.parent.name=='repair_results';active=d['active']
        with np.load(path.with_suffix('.npz')) as z:a={k:z[k] for k in z.files}
        scores={};cuts={};oldcuts=d['cuts'] if repaired else d['cuts']['fixed_pilot'];actual=d['actual_scores']
        for b in ('local','broad'):
            cv=a[b+('_covariance' if repaired else '_anchor_covariance')];w,info=S.metric(cv,active);assert w is not None;scores[b]=S.operator_scores(a[b+'_bootstrap_grams'],w,active);cuts[b]=Legacy.cutoff(scores[b])
        cut=Legacy.cutoff(np.maximum(scores['local'],scores['broad']));jointold=d['joint']['cutoff'] if repaired else d['joint']['fixed_pilot']['cutoff'];oldcuts=dict(oldcuts,joint=jointold);cuts['joint']=cut;actual=dict(actual,joint=max(actual.values()))
        row=dict(system=d['system'],profile=d['profile'],seed=d['seed'],cohort='repaired_ode' if repaired else 'historical_default',studentized={b:actual[b]<=oldcuts[b] for b in cuts},fixed_metric={b:actual[b]<=cuts[b] for b in cuts},cutoff_ratios={b:cuts[b]/oldcuts[b] for b in cuts},cuts=cuts,original_cuts=oldcuts,actual_scores=actual)
        rows.append(row)
    summary={}
    for cohort in ('historical_default','repaired_ode'):
        rr=[r for r in rows if r['cohort']==cohort];summary[cohort]={m:{b:sum(r[m][b] for r in rr) for b in ('local','broad','joint')} for m in ('studentized','fixed_metric')}
    dump(ROOT/'retrospective.json',dict(status='COMPLETE',rows=rows,summary=summary,source_sha256=sources('retrospective.py'),input_sha256={str(p):sha(p) for p in paths}));print(summary,flush=True)

if __name__=='__main__':main()
