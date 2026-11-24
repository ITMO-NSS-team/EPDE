"""Evaluate the saved wave fronts on paired changes of initial displacement."""
from concurrent.futures import ProcessPoolExecutor
import csv
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
PRIOR = HERE.parent / 'live_preserved_fit_2026-11-18'
sys.path.insert(0, str(PRIOR))
from transfer import ARMS, N_FRESH, U, np, run_pair
from wave_confirmation import SEEDS


def profile_changes(axes, field):
    t, x = np.meshgrid(*axes, indexing='ij')
    changes = {'original': np.zeros_like(field)}
    for order in (3, 4):
        k = order * np.pi
        mode = np.sin(k*x) * np.cos(0.2*k*t)
        amplitude = 0.5 * field.std() / np.sqrt(np.mean(mode**2))
        for tag, speed in [('wave', 0.2), ('mixed_speed', 0.4)]:
            delta = amplitude * np.sin(k*x) * np.cos(speed*k*t)
            np.testing.assert_allclose(delta[:, [0,-1]], 0., atol=1.e-12)
            changes[f'{tag}_{order}'] = delta
        np.testing.assert_allclose(changes[f'wave_{order}'][0], changes[f'mixed_speed_{order}'][0])
    return changes


def reference_on_front(candidates, arm, i, check_arm=None):
    eligible = [r for r in candidates if r['eligible_amplification'] and r['scores'][arm][i] <= 1.
                and (check_arm is None or r['scores'][check_arm][i] <= 1.)]
    for ref in eligible:
        if ref['reference'] and not any(
            rival['dyad'] <= ref['dyad'] and rival['scores'][arm][i] <= ref['scores'][arm][i]
            and (rival['dyad'] < ref['dyad'] or rival['scores'][arm][i] < ref['scores'][arm][i])
            for rival in eligible
        ):
            return True
    return False


def evaluate(seed):
    axes, data = U.load('wave')
    original = data['u']
    changes = profile_changes(axes, original)
    index = list(U.TRUTH).index('wave')
    noisy_fields = [original + .05*original.std()*np.random.default_rng([984102,seed,index,j]).standard_normal(original.shape)
                    for j in range(N_FRESH)]
    checks = ({'u': field+delta} for delta in changes.values() for field in noisy_fields)
    run = run_pair(('wave',seed), validation_fields=checks)
    previous = next(r for r in json.loads((PRIOR/'wave_transfer.json').read_text()) if r['seed']==seed)
    candidates = run['variants']
    assert len(candidates)==len(previous['variants'])
    for row, old in zip(candidates, previous['variants']):
        assert row['equation']==old['equation'] and json.dumps(row['target'])==json.dumps(old['target'])
        for arm in ARMS:
            np.testing.assert_allclose(row['scores'][arm][:1+N_FRESH], old['scores'][arm], atol=1e-10, rtol=1e-9)
    reference = [row for row in candidates if row['reference'] and row['eligible_amplification']]
    physical, = [row for row in reference if row['target']==run['reference_coefficients']['pivot']]
    output = []
    for group, profile in enumerate(changes):
        for arm in ARMS:
            own = [r for r in candidates if arm in r['source']]
            for j in range(N_FRESH):
                i = 1 + group*N_FRESH + j
                pooled = reference_on_front(candidates, arm, i)
                assert pooled==run['survival'][arm][i]
                output.append(dict(seed=seed, profile=profile, repeat=j, arm=arm, pool_size=len(candidates),
                                   own_pool_size=len(own), pooled_hit=int(pooled),
                                   own_hit=int(reference_on_front(own, arm, i)),
                                   own_has_reference=int(any(r['reference'] for r in own)),
                                   pooled_wape_with_block_check=(int(reference_on_front(candidates, arm, i, 'wls'))
                                                                 if arm=='wape' else None),
                                   own_wape_with_block_check=(int(reference_on_front(own, arm, i, 'wls'))
                                                               if arm=='wape' else None),
                                   physical_score=physical['scores'][arm][i],
                                   best_reference_score=min(r['scores'][arm][i] for r in reference)))
    return output


if __name__=='__main__':
    with ProcessPoolExecutor(max_workers=3) as executor:
        rows = [row for batch in executor.map(evaluate, SEEDS) for row in batch]
    assert len(rows)==len(SEEDS)*5*N_FRESH*len(ARMS)
    with (HERE/'profile_results.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for profile in dict.fromkeys(r['profile'] for r in rows):
        for arm in ARMS:
            selected = [r for r in rows if r['arm']==arm and r['profile']==profile]
            print(profile, arm, sum(r['pooled_hit'] for r in selected), sum(r['own_hit'] for r in selected),
                  round(float(np.median([r['physical_score'] for r in selected])), 5))
