"""Identify complementary surviving parents from matched, fully observed native trajectories."""
from itertools import combinations, permutations
import json

from trace_lineage import ROOT, folder, live
from swap_trial import folder as native_folder
from assess_swap import previous, read_archive, write_csv


def audit(name, seed):
    a = read_archive(folder(name,seed))
    native = read_archive(native_folder(name,seed,'native'))
    for key in ('calls','settings','train_hashes','plan','variables','trig','axis','labels','points','systems','system_events'):
        assert a[key] == native[key], (name,seed,key)
    op = a['operator']
    for key in ('full_cost','system_cost'):
        assert op[key] == native['operator'][key]
    forms,freq = previous.registry(name,live.load_data(name))
    labels = [previous.label_key(t,freq) for t in a['labels']]
    supports = [frozenset(labels[i] for i,_ in row[2]) for row in a['points']]
    width = len(a['variables'])
    full = (1<<width)-1
    assignments = {p for law in forms for p in permutations(law)}
    known = {tuple(c) for c in a['systems']}

    def mask(combo, assignment):
        return sum(1<<i for i,(p,f) in enumerate(zip(combo,assignment,strict=True))
                   if p is not None and supports[p] == f)

    cohorts = {}
    for i,gen in enumerate(op['generations']):
        available = set()
        distributed = set()
        for assignment in assignments:
            observed = {mask(c,assignment) for c in gen['population']}
            available.update(m.bit_count() for m in observed)
            for x,y in combinations(observed,2):
                joined = x|y
                if not any((m&joined)==joined for m in observed):
                    distributed.add(joined.bit_count())
        cohorts[i] = dict(dataset=name,seed=seed,generation=i,at=gen['at'],members=len(gen['population']),
                          best_in_one=max(available),best_distributed=max(distributed,default=0))
    pairs = []
    for i,pair in enumerate(op['pairs']):
        children = pair['children']
        parents = pair['parents']
        sets = [[o['points'] for attempt in child['fits'] for o in attempt['outcomes']] for child in children]
        for c in sets[0]+sets[1]:
            if None not in c:
                assert tuple(c) in known
        distributed = 0
        child_kept = [False,False]
        for assignment in assignments:
            x,y = (mask(p,assignment) for p in parents)
            both = x|y
            if both==x or both==y:
                continue
            distributed = max(distributed,both.bit_count())
            for j,items in enumerate(sets):
                child_kept[j] |= any(mask(c,assignment)&both==both for c in items)
        pairs.append(dict(dataset=name,seed=seed,generation=pair['generation'],pair=i,at=pair['at'],
                          complementary_laws=distributed,assembled_fits=len(children[0]['fits']),
                          crossed_fits=len(children[1]['fits']),assembled_outcomes=len(sets[0]),
                          crossed_outcomes=len(sets[1]),assembled_kept=int(child_kept[0]),crossed_kept=int(child_kept[1])))
    available_laws = [any(s in {f[j] for f in forms} for s in supports) for j in range(width)]
    complete = any(mask(c,f)==full for c in known for f in assignments)
    summary = dict(dataset=name,seed=seed,verified=True,pools=len(cohorts),mating_pairs=len(pairs),
                   registered_laws_ever_scored=sum(available_laws),actual_complete_system=int(complete),
                   pools_with_distributed_laws=sum(r['best_distributed']>=2 for r in cohorts.values()),
                   pools_with_distributed_system=sum(r['best_distributed']==width for r in cohorts.values()),
                   complementary_matings=sum(r['complementary_laws']>=2 for r in pairs),
                   matings_spanning_system=sum(r['complementary_laws']==width for r in pairs),
                   assembled_kept=sum(r['assembled_kept'] for r in pairs),crossed_kept=sum(r['crossed_kept'] for r in pairs))
    return summary, list(cohorts.values()), pairs


if __name__ == '__main__':
    results = [audit(s,n) for s in ('lorenz','ns') for n in live.SEEDS]
    summary = [r for r,_,_ in results]
    write_csv(ROOT/'by_run.csv', summary)
    write_csv(ROOT/'populations.csv', [x for _,rows,_ in results for x in rows])
    write_csv(ROOT/'matings.csv', [x for _,_,rows in results for x in rows])
    print(json.dumps(summary), flush=True)
