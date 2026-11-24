"""Verify the completed paired runs and summarize exact structural recovery."""
from pathlib import Path
from statistics import median
import csv
import json
import math

ROOT = Path(__file__).resolve().parent
SYSTEMS = ('ode', 'vdp', 'wave', 'ac')
SEEDS = (73001, 73002, 73003)


def summarize():
    import run
    pairs = []
    for system in SYSTEMS:
        term_sets = [frozenset([t, *xs]) - {'1'} for _,t,xs in run.U.truth_forms(system)['u']]
        for seed in SEEDS:
            def read(arm):
                record = json.loads((ROOT/'plain_results'/f'{system}_{seed}_{arm}.json').read_text())
                cfg = record['settings']
                assert record['arm'] == arm and record['system'] == system and record['seed'] == seed
                assert cfg['system_rps_mode'] == 'plain' and cfg['engine'] == 'nsga2' and cfg['evaluation_form'] == 'weak'
                assert cfg['instability_metric'] == 'dyad' and cfg['preprocessor_kwargs']['accuracy'] == 4
                assert record['epochs_recorded'] == cfg['training_epochs'] == 50
                assert record['calls'] > 0 and record['vwsr_calls'] == 0
                assert min(row['distance'] for row in record['front']) == record['min_distance']
                assert all(bool(row['hits']) == (row['distance'] == 0) for row in record['front'])
                return record
            wape, weak = read('wape'), read('weak')
            assert wape['settings'] == weak['settings'] and wape['raw_sha256'] == weak['raw_sha256']
            row = dict(system=system, seed=seed)
            old = {x['equation']:x['objectives'][1] for x in wape['front']}
            shared = [x for x in weak['front'] if x['equation'] in old]
            assert all(math.isclose(old[x['equation']], x['objectives'][1], rel_tol=1.e-8, abs_tol=1.e-10) for x in shared)
            row['identical_equations_dyad_checked'] = len(shared)
            for record in (wape, weak):
                arm = record['arm']
                row.update({f'{arm}_{key}': record[key] for key in ('min_distance', 'selected_distance', 'search_seconds')})
                row[f'{arm}_evaluated_law'] = bool(record['evaluated_laws'])
                row[f'{arm}_front_size'] = len(record['front'])
                row[f'{arm}_native_hit'] = any(frozenset(item['terms']) - {'1'} in term_sets
                                               for item in record['front'])
            row['weak_min_discrepancy'] = min(r['objectives'][0] for r in weak['front'])
            row['weak_negative_supports'] = weak['weak']['negative']
            row['weak_unique_supports'] = weak['weak']['supports']
            row['weak_unresolved_supports'] = weak['weak']['unresolved']
            row['weak_unweighted_folds'] = len(weak['weak'].get('unweighted_folds', []))
            row['weak_total_seconds'] = weak['search_seconds'] + weak['preparation_seconds']
            pairs.append(row)
    with (ROOT/'pairs.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(pairs[0]))
        writer.writeheader()
        writer.writerows(pairs)
    for arm in ('wape', 'weak'):
        print(arm, 'front hits', sum(r[f'{arm}_min_distance']==0 for r in pairs),
              'chosen hits', sum(r[f'{arm}_selected_distance']==0 for r in pairs),
              'law evaluated', sum(r[f'{arm}_evaluated_law'] for r in pairs),
              'median distance', median(r[f'{arm}_min_distance'] for r in pairs))
    for system in SYSTEMS:
        print(system, [(r['wape_min_distance'], r['weak_min_distance'], round(r['weak_min_discrepancy'], 4))
                       for r in pairs if r['system'] == system])
    lines = ['Domain-refactor U, live first-objective comparison',
             'Four scalar systems, three paired seeds, 5% Gaussian field noise and 50 NSGA-II epochs per search.',
             'Every pair used identical observations, native weak rows, WLS, dyad, and intercept handling.',
             'The VWSR selector guard recorded zero calls in all 24 searches.',
             'Exact recovery credits registered alternatives and counts an intercept as part of the support.',
             '', 'System            WAPE hits   Weak hits   WAPE min distances   Weak min distances']
    for s in SYSTEMS:
        rows = [r for r in pairs if r['system'] == s]
        def distances(a):
            return [r[f'{a}_min_distance'] for r in rows]
        a, b = distances('wape'), distances('weak')
        lines.append(f'{s:16s}  {a.count(0)}/3         {b.count(0)}/3         {str(a):20s} {b}')
    for arm in ('wape', 'weak'):
        def n(field):
            return sum(r[f'{arm}_{field}'] for r in pairs)
        hit = sum(r[f'{arm}_min_distance'] == 0 for r in pairs)
        selected = sum(r[f'{arm}_selected_distance'] == 0 for r in pairs)
        lines.extend(['', f'{arm.upper()} retained a law in {hit}/12 searches and evaluated one in {n("evaluated_law")}/12.',
                      f'The normalized compromise picked one in {selected}/12. Ignoring intercept presence gave {n("native_hit")}/12 front hits.',
                      f'Median final front size {median(r[f"{arm}_front_size"] for r in pairs):g}.'])
    changes = [r['wape_min_distance']-r['weak_min_distance'] for r in pairs]
    lines.extend(['', f'Closest structure under the weak score improved / tied / worsened in '
                  f'{sum(c>0 for c in changes)} / {changes.count(0)} / {sum(c<0 for c in changes)} paired cases.',
                  f'Median paired total-time ratio (weak / WAPE) {median(r["weak_total_seconds"]/r["wape_search_seconds"] for r in pairs):.2f}. '
                  'Wall time includes shared-machine contention.',
                  f'Unweighted zero-estimated-noise folds {sum(r["weak_unweighted_folds"] for r in pairs)}/36.'])
    checks = [json.loads((ROOT/'rechecks'/f'{r["system"]}_{r["seed"]}_weak.json').read_text()) for r in pairs]
    negative = sum(c['original'] < 0 for c in checks)
    crossed = sum(min(c['fresh']) < 0 < max(c['fresh']) for c in checks)
    false = sum(c['distance'] > 0 for c in checks)
    lines.extend(['', f'The minimum-discrepancy support was outside the registered laws in {false}/12 searches '
                  f'and had a negative signed score in {negative}/12.',
                  f'The score of the same fixed relation took both signs over eight fresh noise realizations in {crossed}/12 cases.',
                  'Each recheck keeps its learned frequencies, coefficients, scales, noise correction and whitening fixed.',
                  'These are repeat observations of each benchmark trajectory. Extrapolation to other trajectories and systems remains open.'])
    (ROOT/'findings.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    return pairs


if __name__ == '__main__':
    summarize()
