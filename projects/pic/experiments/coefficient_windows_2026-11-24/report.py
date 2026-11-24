"""Validate the paired observations and compare the three window rules."""
from collections import defaultdict
import json
import math
from pathlib import Path
from statistics import median
import sys

ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT.parent/'complementary_windows_2026-11-24'
sys.path.insert(0, str(PREVIOUS))
from summarize import spread

REPLAY = set(range(923211, 923217))
FRESH = set(range(926301, 926307))


def read(path):
    return json.loads(path.read_text(encoding='utf8'))


def validate(case):
    selected = case['second'] is not None
    assert case['rows'] == math.prod(case['plan']['shape'])
    assert max(case['cpu_seconds'].values()) <= 30.
    if selected:
        assert 0 < case['changed_rows'] <= (case['rows']+1)//2
        audit = case['selector']
        assert audit['proposed'] < audit['original'] and audit['check_proposed'] < audit['check_original']
    else:
        assert case['changed_rows'] == 0 and case['native'] == case['complementary']
    if case['seed'] in REPLAY:
        old = read(PREVIOUS/'results'/f"{case['system']}_{case['seed']}.json")
        for attr in ('rows', 'candidate_count', 'plan'):
            assert old[attr] == case[attr]
        for row, previous in zip(case['native'], old['native'], strict=True):
            assert row['front'] == previous['front']
            if row['reference']:
                assert all(math.isclose(a, b, rel_tol=1e-7, abs_tol=1e-10)
                           for a, b in zip(row['reference']['weights'], previous['reference']['weights'], strict=True))
        case['rank'] = old['complementary']
        case['rank_selected'] = old['second'] is not None
    else:
        assert case['native_plan_cpu_seconds'] is not None


def weighted_error(reference, truth):
    return math.sqrt(sum(((b-t)*s)**2 for b, t, s in
                         zip(reference['weights'][1:], truth, reference['scales'], strict=True)))


def compare(cases):
    arms = ['native', 'complementary']
    if all(c['seed'] in REPLAY for c in cases):
        arms.append('rank')
    metrics = {arm: dict(hits=0, exact=0, additions=0, gains=0, losses=0, valid_refs=0) for arm in arms}
    refs = defaultdict(lambda: defaultdict(list))
    errors = defaultdict(lambda: defaultdict(list))
    dispersion = defaultdict(list)
    total_slots = 0
    for case in cases:
        total_slots += len(case['native'])
        for i, base in enumerate(case['native']):
            key = f"{case['system']}/{i}"
            rows = {arm: case[arm][i] for arm in arms}
            for arm, row in rows.items():
                m = metrics[arm]
                m['hits'] += row['hit']
                m['exact'] += row['exact']
                m['additions'] += row['additions']
                m['gains'] += row['hit'] and not base['hit']
                m['losses'] += base['hit'] and not row['hit']
                m['valid_refs'] += row['reference'] is not None
            if any(row['reference'] is None for row in rows.values()):
                continue
            truth = base['reference']['truth']
            for arm, row in rows.items():
                ref = row['reference']
                refs[key][arm].append(ref)
                errors[key][arm].append(weighted_error(ref, truth))
                if case['system'] == 'kdv_cossin':
                    j, = [j for j, t in enumerate(base['reference']['terms']) if 'd^3u/dx1^3' in t]
                    assert truth[j] == -1.
                    dispersion[arm].append(ref['weights'][j+1])
    slot_metrics = {k: {arm: dict(spread=spread(by_arm[arm]), error=median(errors[k][arm]),
                                   n=len(by_arm[arm])) for arm in arms} for k, by_arm in refs.items()}
    for arm in arms:
        for name in ('spread', 'error'):
            pairs = [(s['native'][name], s[arm][name]) for s in slot_metrics.values() if s[arm][name] is not None]
            changes = [b-a for a, b in pairs if not math.isclose(a, b, rel_tol=1e-8, abs_tol=1e-12)]
            metrics[arm][name] = dict(median=median(b for _, b in pairs), better=sum(v < 0 for v in changes),
                                      worse=sum(v > 0 for v in changes), slots=len(pairs))
        if dispersion[arm]:
            vals = dispersion[arm]
            metrics[arm]['dispersion'] = dict(median=median(vals), median_abs_error=median(abs(x+1) for x in vals),
                                              min=min(vals), max=max(vals))
    out = dict(recordings=len(cases), slots=total_slots, selector_proposals=sum('check_original' in c['selector'] for c in cases),
               selected=sum(c['second'] is not None for c in cases), rows=sum(c['rows'] for c in cases),
               replaced=sum(c['changed_rows'] for c in cases), cpu={arm: sum(c['cpu_seconds'][arm] for c in cases)
               for arm in ('native', 'complementary')}, arms=metrics, by_slot=slot_metrics)
    if 'rank' in arms:
        out['rank_selected'] = sum(c['rank_selected'] for c in cases)
    return out


def main():
    cases = [read(p) for p in sorted((ROOT/'results').glob('*.json'))]
    systems = sorted({c['system'] for c in cases})
    assert len(cases) == 180 and len(systems) == 15
    for system in systems:
        current = [c for c in cases if c['system'] == system]
        assert {c['seed'] for c in current} == REPLAY | FRESH
        assert len({tuple(sorted(c['train_hashes'].items())) for c in current}) == len(current)
    for case in cases:
        validate(case)
    out = {}
    for label, motivating in [('transfer', False), ('forced_kdv', True)]:
        subset = [c for c in cases if (c['system'] == 'kdv_cossin') == motivating]
        out[label] = {name: compare([c for c in subset if c['seed'] in seeds])
                      for name, seeds in [('replay', REPLAY), ('fresh', FRESH)]}
    out['max_arm_cpu_seconds'] = max(max(c['cpu_seconds'].values()) for c in cases)
    out['fresh_shared_plan_cpu_seconds'] = sum(c['native_plan_cpu_seconds'] or 0. for c in cases)
    (ROOT/'summary.json').write_text(json.dumps(out, indent=2, allow_nan=False), encoding='utf8')
    for group in ('transfer', 'forced_kdv'):
        for cohort, res in out[group].items():
            print(group, cohort, res['selected'], '/', res['recordings'], res['arms'])


if __name__ == '__main__':
    main()
