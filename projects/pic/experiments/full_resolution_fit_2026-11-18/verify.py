"""Independent numerical replay of saved scores, fronts, and full-resolution controls."""
import csv
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
from study import METHODS, ROOT, SPLITS, SYSTEMS, Panel, R, fit, np, wls


def limits(values, dyad, mask):
    # For membership of any element in the group, jointly favor or disfavor all
    # missing scores in that group. Known objectives take finite values.
    result = []
    missing = np.isnan(values)
    for favorable in (False, True):
        scores = values.copy()
        scores[missing] = np.where(mask[missing] == favorable, -np.inf, np.inf)
        dominates = (scores[:,None]<=scores) & (dyad[:,None]<=dyad) & ((scores[:,None]<scores) | (dyad[:,None]<dyad))
        result.append(int(np.sum(mask & ~dominates.any(axis=0))))
    return result


def verify(system):
    panel = Panel(system)
    saved = json.loads((ROOT/'results'/f'{system}.json').read_text())
    samples = defaultdict(list)
    for r in saved['rows']:
        samples[r['anchor']].append(r)
    table = {}
    with (ROOT/'slots.csv').open() as f:
        for r in csv.DictReader(f):
            if r['system'] == system:
                table[int(r['anchor']),int(r['slot']),r['method'],r['split']] = r
    count = 0
    max_error = 0.
    alias = Counter()
    for anchor, rows in samples.items():
        recordings = [panel.recording(anchor,i) for i in range(3)]
        assert saved['raw_hashes'][anchor] == [{k:R.digest(v) for k,v in f.items()} for f in recordings]
        m, left, right = [panel.matrices(f) for f in recordings]
        for candidate,row in zip(panel.cs, rows, strict=True):
            assert candidate['id'] == row['id']
            ids = row['ids']
            for method, res in row['methods'].items():
                batch = defaultdict(list)
                for g, v in enumerate(res['vectors']):
                    if v is None:
                        continue
                    v = np.asarray(v)
                    train = m[g][:,ids]
                    scales = np.sqrt(np.mean(train**2,axis=0))
                    scale2 = np.dot(v*v, scales*scales)
                    assert scale2 > 0 and np.isfinite(scale2)
                    if method == 'tls' and g and candidate['label'] == 'first_order_fit':
                        if abs(v[ids.index(candidate['yi'])] * scales[ids.index(candidate['yi'])]) < 1e-10*np.sqrt(scale2):
                            alias['negligible_written_target_in_short_training_fold'] += 1
                    if g:
                        h = 3-g
                        pairs = [('blocked_energy',m[h],m[h]),('blocked_cross',left[h],right[h])]
                    else:
                        pairs = [('record_cross',left[0],right[0])]
                    for key,a,b in pairs:
                        cross = a[:,ids].T @ b[:,ids] / len(a)
                        batch[key].append(float(v @ cross @ v / scale2))
                for split, expected in res['scores'].items():
                    if expected is None:
                        assert len(batch[split]) < (1 if split=='record_cross' else 2)
                        continue
                    actual = float(np.mean(batch[split]))
                    delta = abs(actual-expected)/max(1,abs(actual),abs(expected))
                    assert delta < 2e-8, (system, anchor, row['id'], delta)
                    max_error = max(max_error, delta)
                    count += 1
        if system == 'ac_zero':
            continue
        for slot in set(r['slot'] for r in rows):
            group = [r for r in rows if r['slot']==slot]
            dyad = np.array([r['dyad'] for r in group])
            for method in METHODS:
                for split in SPLITS:
                    vals = np.array([r['methods'][method]['scores'][split] for r in group], dtype=float)
                    summary = table[anchor,slot,method,split]
                    for name,mask in [('registered',[r['registered'] for r in group]),
                                      ('exact',[r['label']=='reference' for r in group]),
                                      ('additions',[r['label'].startswith('add_') and not r['registered'] for r in group])]:
                        bounds = limits(vals, dyad, np.array(mask))
                        if name != 'additions':
                            bounds = [int(k>0) for k in bounds]
                        assert bounds == [int(summary[name+s]) for s in ('_lower','_upper')]
    return dict(scores=count, max_scaled_error=max_error, **alias)


def identities():
    t = np.linspace(0,4*np.pi,200)
    u, ut = np.sin(t)+1.3*np.cos(t), np.cos(t)-1.3*np.sin(t)
    checks = 0
    for a,b,constant in [(ut*np.cos(t),u*np.sin(t),1.),(u*np.cos(t),-ut*np.sin(t),1.3),(ut**2,u**2,2.69)]:
        fields = np.column_stack([a,b,np.ones(len(t))])
        half = [np.mean(part.reshape(20,5,3),axis=1) for part in np.split(fields,2)]
        for train,held in [half,half[::-1]]:
            for pivot in range(3):
                for v in (wls(train,pivot)[0], fit(train)[0]):
                    np.testing.assert_allclose(v/v[0],[1.,1.,-constant],atol=3e-12)
                    np.testing.assert_allclose(held @ (v/v[0]),0,atol=5e-12)
                    checks += 1
    return checks


if __name__ == '__main__':
    names = [*SYSTEMS, 'ac_zero']
    with ProcessPoolExecutor(max_workers=4) as pool:
        audited = dict(zip(names, pool.map(verify,names)))
    verified = dict(by_system=audited, identity_checks=identities(),
                    pareto_slot_method_splits=12*20*len(SPLITS)*len(METHODS),
                    score_checks=sum(r['scores'] for r in audited.values()))
    R.dump(ROOT/'verification.json',verified)
    print(verified['score_checks'], verified['pareto_slot_method_splits'], verified['identity_checks'],
          max(r['max_scaled_error'] for r in audited.values()))
