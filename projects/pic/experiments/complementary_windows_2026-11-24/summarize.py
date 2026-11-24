"""Verify paired counts and summarize support recovery and coefficient repeatability."""
from collections import defaultdict
from itertools import combinations
import json
import math
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parent
ARMS = ('native', 'complementary')


def spread(refs):
    distances = []
    for a, b in combinations(refs, 2):
        distances.append(math.sqrt(sum(((u-v)*(s+t)/2)**2 for u, v, s, t in
                         zip(a['weights'][1:], b['weights'][1:], a['scales'], b['scales'], strict=True))))
    return median(distances) if distances else None


def summarize(cases):
    result = dict(recordings=len(cases), already_full=0, first_proposals=0, selected=0,
                  rows=0, replaced=0, slots=0, gains=0, losses=0)
    refs = defaultdict(lambda: [[], []])
    for arm in ARMS:
        result[arm] = dict(hits=0, exact=0, additions=0, cpu_seconds=0.)
    for c in cases:
        audit = c['selector']
        result['already_full'] += audit['original'] == audit['probe_count']
        result['first_proposals'] += audit['proposed'] > audit['original']
        result['selected'] += c['second'] is not None
        result['rows'] += c['rows']
        result['replaced'] += c['changed_rows']
        assert c['rows'] == math.prod(c['plan']['shape'])
        if c['second'] is None:
            assert not c['changed_rows'] and c['native'] == c['complementary']
        else:
            assert 0 < c['changed_rows'] <= (c['rows']+1)//2
            assert audit['proposed'] > audit['original'] and audit['check_proposed'] > audit['check_original']
        for arm in ARMS:
            result[arm]['cpu_seconds'] += c['cpu_seconds'][arm]
        for slot, (a, b) in enumerate(zip(c['native'], c['complementary'], strict=True)):
            result['slots'] += 1
            result['gains'] += b['hit'] and not a['hit']
            result['losses'] += a['hit'] and not b['hit']
            for arm, row in zip(ARMS, (a, b)):
                result[arm]['hits'] += row['hit']
                result[arm]['exact'] += row['exact']
                result[arm]['additions'] += row['additions']
            if a['reference'] and b['reference']:
                for i, row in enumerate((a, b)):
                    refs[c['system'], slot][i].append(row['reference'])
    changes = {}
    for (system, slot), (a, b) in refs.items():
        if len(a) >= 2:
            changes[f'{system}/{slot}'] = [spread(a), spread(b)]
    pairs = list(changes.values())
    result['coefficient_spreads'] = changes
    result['stability_slots'] = len(pairs)
    result['more_stable'] = sum(b < a and not math.isclose(a, b, rel_tol=1e-8, abs_tol=1e-12) for a,b in pairs)
    result['less_stable'] = sum(b > a and not math.isclose(a, b, rel_tol=1e-8, abs_tol=1e-12) for a,b in pairs)
    for i, arm in enumerate(ARMS):
        result[arm]['median_slot_spread'] = median(x[i] for x in pairs) if pairs else None
    return result


def main():
    cases = [json.loads(p.read_text()) for p in sorted((ROOT/'results').glob('*.json'))]
    systems = sorted({c['system'] for c in cases})
    assert len(systems) == 15
    for system in systems:
        assert sorted(c['seed'] for c in cases if c['system'] == system) == list(range(923211,923223))
    out = {}
    for group, fields in [('transfer', [c for c in cases if c['system'] != 'kdv_cossin']),
                          ('motivating_kdv', [c for c in cases if c['system'] == 'kdv_cossin'])]:
        out[group] = {key: summarize(subset) for key, subset in (
            ('screen', [c for c in fields if c['seed'] < 923217]),
            ('confirmation', [c for c in fields if c['seed'] >= 923217]), ('all', fields))}
    out['systems'] = {s: summarize([c for c in cases if c['system'] == s]) for s in systems}
    (ROOT/'summary.json').write_text(json.dumps(out, indent=2, allow_nan=False), encoding='utf8')
    for group in ('transfer', 'motivating_kdv'):
        for key, res in out[group].items():
            print(group, key, {k:v for k,v in res.items() if k != 'coefficient_spreads'})


if __name__ == '__main__':
    main()
