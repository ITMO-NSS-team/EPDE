"""Paired structural recovery, front sizes and independent residuals by dataset."""
import argparse
import csv
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parent


def read(name, cohort):
    with (ROOT/f'{name}_{cohort}.csv').open() as stream:
        return {(r['dataset'],int(r['seed']),r['rule']):r for r in csv.DictReader(stream)}


def compare(pairs):
    def counts(col):
        return [sum(int(pair[i][col]) for pair in pairs) for i in (0,1)]
    diffs = [int(b['six_hit'])-int(a['six_hit']) for a,b in pairs]
    fresh = [(float(a['fresh_wape']),float(b['fresh_wape'])) for a,b in pairs if a['fresh_wape'] and b['fresh_wape']]
    return dict(pairs=len(pairs), six_base=counts('six_hit')[0],six_test=counts('six_hit')[1],
                gains=diffs.count(1), losses=diffs.count(-1), one_base=counts('one_hit')[0],one_test=counts('one_hit')[1],
                primary_base=counts('primary_six')[0],primary_test=counts('primary_six')[1],
                front_base=counts('front_hit')[0],front_test=counts('front_hit')[1],
                assembled_base=counts('assembled')[0],assembled_test=counts('assembled')[1],
                shorter_base=sum(int(a['six_size'])<6 for a,b in pairs),shorter_test=sum(int(b['six_size'])<6 for a,b in pairs),
                median_front_base=median(int(a['front_size']) for a,b in pairs),
                median_front_test=median(int(b['front_size']) for a,b in pairs),fresh_pairs=len(fresh),
                median_fresh_base=median(a for a,b in fresh) if fresh else None,
                median_fresh_test=median(b for a,b in fresh) if fresh else None)


def summarize(cohort, designs):
    base = read('contiguous',cohort)
    tests = [('contiguous',base,r) for r in ('held','train','mean','worst')]
    if 'alternate' in designs:
        alternative = read('alternate',cohort)
        assert set(base)==set(alternative)
        tests += [('alternate',alternative,r) for r in ('held','train','mean','worst')]
    if 'full' in designs:
        full = read('full',cohort)
        assert set(full)=={(n,s,r) for n,s,r in base if r=='train'}
        tests.append(('full',full,'train'))
    if 'slabs' in designs:
        slabs=read('slabs',cohort)
        assert {(n,s) for n,s,r in slabs}=={(n,s) for n,s,r in base}
        tests += [('slabs',slabs,r) for r in ('slab_frozen','slab_crossfit','slab_cross_scale')]
    groups = sorted({n for n,s,r in base})
    rows = []
    for design,data,rule in tests:
        for group in ['suite',*groups]:
            pairs = [(a,data[n,s,rule]) for (n,s,r),a in base.items() if r=='held' and
                     (group==n or (group=='suite' and n!='ode_simple'))]
            assert len(pairs)==(3 if cohort=='screen' else 12)*(15 if group=='suite' else 1)
            rows.append(dict(cohort=cohort,design=design,rule=rule,group=group,**compare(pairs)))
    dest = ROOT/f'summary_{cohort}.csv'
    with dest.open('w',newline='',encoding='utf8') as out:
        writer = csv.DictWriter(out,fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    for r in rows:
        if r['group']=='suite':
            print(*(r[k] for k in ('design','rule','six_test','gains','losses','one_test','front_test','assembled_test')))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('cohort',choices=['screen','confirmation'])
    parser.add_argument('--alternate',action='store_true')
    parser.add_argument('--full',action='store_true')
    parser.add_argument('--slabs',action='store_true')
    args=parser.parse_args()
    summarize(args.cohort, [design for design in ('alternate','full','slabs') if getattr(args,design)])
