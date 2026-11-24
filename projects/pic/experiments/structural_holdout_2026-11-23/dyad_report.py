"""Paired comparisons of native weak-slab discrepancy, including the halves ablation."""
import json
from statistics import median

from summarize import ROOT, read, compare
import csv


def report():
    rows = []
    for cohort, n in [('screen', 3), ('confirmation', 12)]:
        results = read('dyad_rows', cohort)
        baseline = read('contiguous', cohort)
        full = read('full', cohort)
        rules = {rule for _, _, rule in results}
        cases = sorted((name, seed) for name, seed, rule in baseline if rule == 'held')
        assert len(cases) == 16*n and len(results) == len(cases)*6
        for name, seed in cases:
            ordinary = results[name, seed, 'ordinary_stability']
            assert all(ordinary[k] == full[name, seed, 'train'][k] for k in
                       ('assembled','front_size','front_hit','six_size','six_hit','one_hit','selected'))
            if max(s['slabs'] for s in json.loads(ordinary['slab_geometry'])) == 2:
                for objective in ('only', 'stability'):
                    a, b = [results[name,seed,f'{scale}_{objective}'] for scale in ('halves','multiscale')]
                    assert a['selected'] == b['selected'] and a['front_hit'] == b['front_hit']
        for group in ['suite', *sorted({name for name, _ in cases})]:
            subset = [(name, seed) for name, seed in cases if name == group or (group == 'suite' and name != 'ode_simple')]
            for rule in sorted(rules):
                scored = [results[name, seed, rule] for name, seed in subset]
                reference = 'ordinary_'+rule.split('_')[1]
                pairs = [(results[name,seed,reference], r) for (name,seed),r in zip(subset,scored)]
                matched = compare(pairs)
                def delta(base, key):
                    return [int(r['six_hit']) - int(base[name,seed,key]['six_hit'])
                            for (name,seed), r in zip(subset,scored)]
                vs_held = delta(baseline, 'held')
                vs_halves = delta(results, 'halves_'+rule.split('_')[1])
                counts = [max(s['slabs'] for s in json.loads(r['slab_geometry'])) for r in scored]
                rows.append(dict(cohort=cohort, dataset=group, rule=rule, runs=len(subset),
                                 assembled=matched['assembled_test'], first=matched['one_test'],
                                 six=matched['six_test'], front=matched['front_test'],
                                 median_front=matched['median_front_test'], shorter_than_six=matched['shorter_test'],
                                 gains_vs_ordinary=matched['gains'], losses_vs_ordinary=matched['losses'],
                                 gains_vs_halves=vs_halves.count(1), losses_vs_halves=vs_halves.count(-1),
                                 gains_vs_holdout=vs_held.count(1), losses_vs_holdout=vs_held.count(-1),
                                 runs_with_finer_scale=sum(c>2 for c in counts), max_slabs=max(counts),
                                 validation_refits=sum(int(r['validation_refits']) for r in scored),
                                 median_fresh_wape=median(float(r['fresh_wape']) for r in scored if r['fresh_wape'])))
    with (ROOT/'dyad_discrepancy_by_dataset.csv').open('w', newline='', encoding='utf8') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        if row['dataset'] in ('suite','ode_simple'):
            print(*(row[k] for k in ('cohort','dataset','rule','six','first','front','median_front',
                                     'gains_vs_ordinary','losses_vs_ordinary','gains_vs_halves','losses_vs_halves',
                                     'gains_vs_holdout','losses_vs_holdout','runs_with_finer_scale','validation_refits')))


if __name__ == '__main__':
    report()
