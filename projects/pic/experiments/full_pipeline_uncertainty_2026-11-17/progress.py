"""Read-only progress summary; does not write or change experiment decisions."""
import json
from pathlib import Path
from collections import Counter
from statistics import median

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))

rows = [read(p) for p in sorted((ROOT / 'results').glob('*.json'))]
audits = [read(p) for p in sorted((ROOT / 'audit_records').glob('*.json'))]
out = dict(completed=len(rows), expected=69, audit_status=dict(Counter(d['status'] for d in audits)),
           warnings=sum(len(d['warnings']) for d in rows), failures=[], profiles={},
           reference_failures=[], width_ratio={})
widths = []
geometries = Counter()
for d in rows:
    ids = {c['id'] for c in d['candidates'] if c['label'] == 'reference'}
    for new in d['joint']['refit_pilot']['certificates']:
        if new['candidate'] not in ids:
            continue
        geometries[new['geometry']] += 1
        old = next(c for c in d['joint']['fixed_pilot']['certificates'] if c['candidate'] == new['candidate'])
        if new['geometry'] == old['geometry'] == 'bounded_outer':
            ow = [b - a for a, b in zip(old['outer_region']['lower'], old['outer_region']['upper'])]
            nw = [b - a for a, b in zip(new['outer_region']['lower'], new['outer_region']['upper'])]
            if all(w > 0 for w in ow):
                widths.append(median([n / o for n, o in zip(nw, ow)]))
    for bank, covered in d['coverage']['refit_pilot'].items():
        if not covered:
            actual = max(d['actual_scores'].values()) if bank == 'both' else d['actual_scores'][bank]
            cut = d['joint']['refit_pilot']['cutoff'] if bank == 'both' else d['cuts']['refit_pilot'][bank]
            out['failures'].append(dict(system=d['system'], profile=d['profile'], seed=d['seed'], bank=bank,
                                        observed_over_cutoff=actual / cut))
    for r in d['references']:
        if r['mode'] == 'refit_pilot' and not (r['noise_covered'] and r['zero_accepted']):
            out['reference_failures'].append(dict(system=d['system'], profile=d['profile'], seed=d['seed'], **r))
for profile in ('iid', 'positive', 'negative'):
    selected = [d for d in rows if d['profile'] == profile]
    out['profiles'][profile] = dict(datasets=len(selected), **{
        m: {b: sum(d['coverage'][m][b] for d in selected) for b in ('local', 'broad', 'both')}
        for m in ('fixed_pilot', 'refit_pilot')})
out['reference_geometry'] = dict(geometries)
if widths:
    out['width_ratio'] = dict(pairs=len(widths), minimum=min(widths), median=median(widths), maximum=max(widths))
print(json.dumps(out, indent=2))
