"""Post-selection coefficient errors for the first correct system in each list."""
import argparse
import csv
from collections import defaultdict
from itertools import permutations
import json
from statistics import median

from experiment import ROOT, directory, full_directory, live, trial
from assess_swap import coefficient_errors, previous, read_archive, write_csv


def report(design, cohort):
    source = ROOT/f'{design}_{cohort}.csv'
    with source.open() as stream:
        groups = defaultdict(list)
        for row in csv.DictReader(stream):
            if row['six_hit']=='1' and row['dataset']!='ode_simple':
                groups[row['dataset'],int(row['seed'])].append(row)
    rows = []
    for (name,seed), results in groups.items():
        path = {'contiguous':trial.folder,'alternate':directory,'full':full_directory,'slabs':full_directory}[design]
        archive = read_archive(path(name,seed,'candidate'))
        problem = live.load_data(name)
        allowed, frequencies = previous.registry(name,problem)
        labels = [previous.label_key(t,frequencies) for t in archive['labels']]
        points = archive['points']
        coefficient_error = coefficient_errors(problem,frequencies,labels,points)
        orders = list(permutations(range(len(problem.variables))))
        for res in results:
            for combo in json.loads(res['selected']):
                support = [frozenset(labels[j] for j,_ in points[i][2]) for i in combo]
                matches = [order for order in orders if any(all(support[k]==system[law] for k,law in enumerate(order))
                                                             for system in allowed)]
                if not matches:
                    continue
                errors = [[coefficient_error(law,i) for law,i in zip(order,combo)] for order in matches]
                resolved = [max(e) for e in errors if all(v is not None for v in e)]
                rows.append(dict(dataset=name,seed=seed,design=design,rule=res['rule'],
                                 max_relative_error=min(resolved,default=None)))
                break
            else:
                raise ValueError(f'Cannot reproduce selected structure for {name}/{seed}')
    write_csv(ROOT/f'coefficients_{design}_{cohort}.csv',rows)
    for rule in sorted({r['rule'] for r in rows}):
        e = [r['max_relative_error'] for r in rows if r['rule']==rule and r['max_relative_error'] is not None]
        print(rule,len(e),median(e))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('design',choices=['contiguous','alternate','full','slabs'])
    parser.add_argument('cohort',choices=['screen','confirmation'])
    args=parser.parse_args()
    report(args.design,args.cohort)
