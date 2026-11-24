"""Reconstruct Pareto bounds, paired changes and diffusion controls from the saved scores."""
from collections import Counter, defaultdict
import csv
import json
from statistics import median
from study import METHODS, ROOT, SPLITS, SYSTEMS


def outcome(rows, method, split):
    points = [(r['methods'][method]['scores'][split], r['dyad']) for r in rows]
    unknown = {i for i, (score, _) in enumerate(points) if score is None}
    front = set()
    for i,(value,dyad) in enumerate(points):
        if value is None:
            continue
        if not any(x is not None and x <= value and y <= dyad and (x < value or y < dyad) for x,y in points):
            front.add(i)
    possible = front | unknown
    certain = {i for i in front if all(points[j][1] > points[i][1] for j in unknown)}
    certain.update(i for i in unknown if all(points[j][1] > points[i][1] for j in range(len(rows)) if j != i))
    out = {'unresolved': len(unknown)}
    for kind, ids in [('registered', {i for i,r in enumerate(rows) if r['registered']}),
                      ('exact', {i for i,r in enumerate(rows) if r['label']=='reference'}),
                      ('additions', {i for i,r in enumerate(rows) if r['label'].startswith('add_') and not r['registered']})]:
        lower, upper = len(certain & ids), len(possible & ids)
        if kind != 'additions':
            lower = int(bool(lower) or any(all(j in ids or points[j][1] > points[i][1] for j in unknown) for i in front & ids))
            upper = int(bool(upper))
        out[kind+'_lower'], out[kind+'_upper'] = lower, upper
    return out


def main():
    all_rows, controls, angles = [], [], defaultdict(list)
    differences = []
    for system in [*SYSTEMS, 'ac_zero']:
        saved = json.loads((ROOT/'results'/f'{system}.json').read_text())
        slots = defaultdict(list)
        for row in saved['rows']:
            slots[row['anchor'],row['slot']].append(row)
        for (anchor,slot), rows in slots.items():
            cohort = 'screen' if anchor < 6 else 'confirmation'
            for method in METHODS:
                ref, = [r for r in rows if r['label'] == 'reference']
                val = ref['methods'][method]['angle']
                if system != 'ac_zero' and val is not None:
                    angles[cohort,method].append(val)
            for split in SPLITS:
                if system in ('ac','ac_zero'):
                    reduced, = [r for r in rows if r['label'] == 'drop_0']
                    for method in METHODS:
                        d = reduced['methods'][method]['scores'][split] - ref['methods'][method]['scores'][split]
                        controls.append(dict(system=system,anchor=anchor,cohort=cohort,split=split,method=method,margin=d))
                if system == 'ac_zero':
                    continue
                pair = {}
                for method in METHODS:
                    vals = outcome(rows, method, split)
                    all_rows.append(dict(system=system, anchor=anchor, slot=slot, cohort=cohort, split=split, method=method, **vals))
                    pair[method] = vals
                if pair['wls']['registered_lower'] != pair['tls']['registered_lower']:
                    differences.append(dict(system=system, anchor=anchor, slot=slot, cohort=cohort, split=split,
                                            wls=pair['wls']['registered_lower'],tls=pair['tls']['registered_lower']))
    for stem, rows in [('slots', all_rows),('ac_controls', controls),('changes', differences)]:
        with (ROOT/f'{stem}.csv').open('w',newline='') as f:
            writer = csv.DictWriter(f,fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    counts = defaultdict(Counter)
    for row in all_rows:
        key = (row['cohort'], row['split'], row['method'])
        counts[key].update({k:v for k,v in row.items() if k.endswith(('_upper','_lower')) or k=='unresolved'})
        counts[key]['slots'] += 1
    summary = {}
    for (cohort, split, method), vals in counts.items():
        assert vals['slots'] == 120
        out = summary.setdefault(cohort, {}).setdefault(split,{})
        out[method] = dict(vals)
        out[method]['median_fullfit_angle_degrees'] = median(angles[cohort,method])
        out[method]['retains_diffusion'] = {}
        for system in ('ac','ac_zero'):
            margins = [r['margin'] for r in controls if (r['cohort'],r['split'],r['method'],r['system'])
                       == (cohort,split,method,system)]
            assert len(margins) == 6
            out[method]['retains_diffusion'][system] = sum(m > 0 for m in margins)
    (ROOT/'summary.json').write_text(json.dumps(summary,indent=1,allow_nan=False))
    for cohort, splits in summary.items():
        for split, vals in splits.items():
            print(cohort,split,{k:(v['registered_lower'],v['registered_upper'],v['exact_lower'],v['additions_lower'],
                                   v['unresolved'], v['retains_diffusion']) for k,v in vals.items()})


if __name__ == '__main__':
    main()
