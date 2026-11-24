"""Audit one-addition opportunities and replay frozen archive fronts at both budgets."""
import argparse
from concurrent.futures import ProcessPoolExecutor

from add_trial import ARMS, CONFIRMATION, folder, live
from assess_swap import assess_run


def audit_additions(events, labels, forms):
    counts = dict(suitable=0, screened=0, shortlisted=0, selectable=0, selected=0, first_selected=None,
                  best_useful_rank=None, mean_useful_probability=None)
    mass = 0.
    for e in events:
        if e['target'] is None:
            continue
        parent = {labels[e['target']],*(labels[i] for i in e['eligible_parent'])}
        missing = {next(iter(form-parent)) for form in forms if parent<form and len(form-parent)==1}
        if not missing:
            continue
        counts['suitable'] += 1
        counts['screened'] += int(e['screened'])
        useful = [i for i,(j,_) in enumerate(e['shortlist']) if labels[j] in missing]
        if useful:
            counts['shortlisted'] += 1
            rank = useful[0]+1
            counts['best_useful_rank'] = min(rank,counts['best_useful_rank'] or rank)
            prob = sum(e['probabilities'][i] for i in useful)
            counts['selectable'] += int(prob>0.)
            mass += prob
        if e['changed'] and labels[e['selected']] in missing:
            counts['selected'] += 1
            if counts['first_selected'] is None:
                counts['first_selected'] = e['after_call']
    if counts['screened']:
        counts['mean_useful_probability'] = mass/counts['screened']
    return counts


def classify(fate, audit):
    if fate!='never scored':
        return fate
    if not audit['suitable']:
        return 'no suitable addition parent'
    if audit['selected']:
        return 'completion proposed but law never scored'
    if audit['selectable']:
        return 'selectable but not drawn'
    if audit['shortlisted']:
        return 'shortlisted but no local front version'
    if audit['screened']:
        return 'completion outside shortlist'
    return 'suitable parent received ordinary draw'


def assess(job):
    return assess_run(job,destination=folder(*job),moves='additions',audit_moves=audit_additions,
                      classify=classify,comparator='pvra',timing='add_seconds')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--workers',type=int,default=3)
    p.add_argument('--confirmation',action='store_true')
    p.add_argument('--refresh',action='store_true')
    args=p.parse_args()
    seeds=CONFIRMATION if args.confirmation else live.SEEDS
    jobs=[(s,n,a) for s in live.DATASETS for n in seeds for a in ARMS]
    todo=[j for j in jobs if (folder(*j)/'archive.json.gz').is_file()
          and (args.refresh or not (folder(*j)/'assessment.json').is_file())]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for job in pool.map(assess,todo):
            print(*job,flush=True)
