"""Summarize completed paired runs. One complete system counts once per seed."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
COHORTS = (
    ('Fresh seeds at equal fit budget', (24112401, 24112402, 24112403), False),
    ('Original seeds with extra fits after epoch 150', (923701, 923702, 923703), True),
)
ARMS = {'native': 'Native', 'reassignment': 'Joint reassignment',
        'exchange': 'Exchange only (inherited target)',
        'exchange_local_target': 'Exchange with target choice for the copy',
        'exchange_reassignment': 'Exchange and joint reassignment'}


def parent_coverage(seeds):
    from discover import CHECKPOINT
    from compare import previous, read_archive
    from epoch_settings import OPTIONS, live, np

    forms, freq = previous.registry('ns', live.load_data('ns', **OPTIONS['ns']))
    kinds = [{form[k] for form in forms} for k in range(3)]
    lines = ['### Equations available at the assembly checkpoint', '',
             'This diagnostic uses benchmark labels after selection. Each count is the number of complete '
             'parent candidates containing the indicated law or combination, including registered identities.', '',
             '| Seed | Source | Candidates | X momentum | Y momentum | Continuity | Continuity with either momentum |',
             '|---:|---|---:|---:|---:|---:|---:|']
    lost_donors = []
    for seed in seeds:
        a = read_archive(ROOT/'results'/str(seed))
        labels = [previous.label_key(t, freq) for t in a['labels']]
        supports = [frozenset(labels[j] for j,_ in p[2]) for p in a['points']]
        tags = np.asarray([[s in group for group in kinds] for s in supports])
        checkpoint = a['operator']['epoch_clock'][CHECKPOINT-1]['cost']
        front = a['epoch_fronts'][CHECKPOINT-1]['members']
        earlier = {tuple(row) for row, cost in zip(a['systems'], a['operator']['system_cost'], strict=True)
                   if cost < checkpoint}
        earlier.update(map(tuple, front))
        for label, rows in [('Eligible native handovers', list(earlier)), ('Assembly parents', front)]:
            found = tags[np.asarray(rows)].any(axis=1)
            x, y, continuity = found.sum(axis=0)
            together = (found[:,2] & found[:,:2].any(axis=1)).sum()
            lines.append(f'| {seed} | {label} | {len(rows):,} | {x:,} | {y:,} | {continuity:,} | {together:,} |')
        handover_points = np.unique(list(earlier))
        front_points = np.unique(front)
        for law, name in enumerate(('X momentum', 'Y momentum', 'Continuity')):
            hits = handover_points[tags[handover_points, law]]
            if len(hits) == 0 or np.any(tags[front_points, law]):
                continue
            structures = [frozenset(j for j,_ in p[2]) for p in a['points']]
            scores = np.asarray([p[3:5] for p in a['points']])
            counts = []
            for hit in hits:
                slot = a['points'][hit][0]
                rivals = np.asarray([i for i in handover_points
                                     if a['points'][i][0] == slot and structures[i] != structures[hit]], dtype=int)
                beats = np.all(scores[rivals] <= scores[hit], axis=1) & np.any(scores[rivals] < scores[hit], axis=1)
                counts.append(len({structures[i] for i in rivals[beats]}))
            lost_donors.append(f'| {seed} | {name} | {len(hits)} | {min(counts)} |')
    if lost_donors:
        lines.extend(['', 'For correct equations seen during search and absent from the assembly parents, '
                      'the table counts the fewest distinct competing supports that dominated a matching '
                      'equation on its own slot’s two objectives.', '',
                      '| Seed | Missing law | Distinct matching fitted equations | Minimum dominating supports |',
                      '|---:|---|---:|---:|', *lost_donors])
    return lines + ['']


def build():
    lines = ['# Complete-system selection on clean Navier–Stokes', '',
             'Each cell counts search seeds on the same clean flow. A hit requires both momentum equations '
             'and continuity together. The global front checks all admitted candidates against one another '
             'on WAPE and dyadic coefficient instability.', '']
    for title, seeds, retrospective in COHORTS:
        rows = [json.loads((ROOT/'results'/str(s)/'comparison.json').read_text()) for s in seeds]
        lines.extend([f'## {title}', '', '| Method | Complete system assembled | On global first front |',
                      '|---|---:|---:|'])
        for arm, label in ARMS.items():
            generated = sum(r['arms'][arm]['complete_generated'] > 0 for r in rows)
            front = sum(r['arms'][arm]['complete_nondominated'] > 0 for r in rows)
            lines.append(f'| {label} | {generated}/{len(rows)} | {front}/{len(rows)} |')
        lines.extend(['', '| Seed | Method | Native fits | Assembly refits | Total fits | Complete candidates | Nondominated complete candidates |',
                      '|---:|---|---:|---:|---:|---:|---:|'])
        for seed, row in zip(seeds, rows, strict=True):
            assert row['seed'] == seed and row['retrospective'] == retrospective
            for arm, label in ARMS.items():
                a = row['arms'][arm]
                assert a['total_fits'] == a['search_fits'] + a['new_fits']
                if not retrospective:
                    assert a['total_fits'] == row['native_budget']
                assert a['complete_nondominated'] <= a['complete_generated']
                lines.append(f"| [{seed}](results/{seed}/comparison.json) | {label} | {a['search_fits']:,} | "
                             f"{a['new_fits']:,} | {a['total_fits']:,} | {a['complete_generated']} | "
                             f"{a['complete_nondominated']} |")
        native_online = sum(r['arms']['native']['original_front_complete'] > 0 for r in rows)
        lines.extend(['', f'EPDE’s native end-of-run archive contained a complete system on '
                      f'{native_online}/{len(rows)} seeds.', ''])
        gained = [r['seed'] for r in rows if r['arms']['exchange_reassignment']['complete_nondominated']
                  and not r['arms']['exchange_local_target']['complete_nondominated']]
        if gained:
            lines.extend(['### Equation placement on seeds gained by joint reassignment', '',
                          'The second method in the table can move any equation to another field slot and '
                          'choose its target. Within each method, all placements face the same global pool of rivals.', '',
                          '| Seed | Method | u slot | v slot | p slot | Complete | Nondominated |',
                          '|---:|---|---|---|---|---:|---:|'])
            for seed in gained:
                placements = json.loads((ROOT/'results'/str(seed)/'placement.json').read_text())
                for arm in ('exchange_local_target', 'exchange_reassignment'):
                    for item in placements[arm]:
                        u, v, p = item['laws']
                        lines.append(f"| {seed} | {ARMS[arm]} | {u} | {v} | {p} | "
                                     f"{item['generated']} | {item['nondominated']} |")
            lines.append('')
        if not retrospective:
            lines.extend(parent_coverage(seeds))
            paths = [ROOT/'results'/str(seed)/'donor_history.json' for seed in seeds]
            if all(path.is_file() for path in paths):
                lines.extend(['### Reusing equations across parents and epochs', '',
                              'All rows reuse the same 150-epoch search budget per seed. The two cross-parent rules '
                              'differ only in which epoch-end fronts can supply equations. The counts below are '
                              'complete candidates, with registered identities accepted.', '',
                              '| Seed | Cross-parent donor pool | Distinct donor structures | Refits | Total fits | Complete | Unbeaten by native candidates | On global first front |',
                              '|---:|---|---:|---:|---:|---:|---:|---:|'])
                for seed, path, base in zip(seeds, paths, rows, strict=True):
                    data = json.loads(path.read_text())
                    for key, name in [('current', 'Epoch 100'), ('historical', 'Epochs 1–100')]:
                        r = data[key]
                        assert r['native_fits'] + r['new_fits'] == r['total_fits'] == base['native_budget']
                        assert r['complete_nondominated'] <= r['unbeaten_by_native'] <= r['complete_generated']
                        lines.append(f"| [{seed}](results/{seed}/donor_history.json) | {name} | {r['donor_supports']:,} | "
                                     f"{r['new_fits']:,} | {r['total_fits']:,} | {r['complete_generated']} | "
                                     f"{r['unbeaten_by_native']} | {r['complete_nondominated']} |")
                lines.append('')
    if (ROOT/'retention_confirmation.md').is_file():
        lines.extend(['[Independent three-seed confirmation of donor retention](retention_confirmation.md)', ''])
    lines.append('[Experiment protocol](protocol.md)')
    (ROOT/'findings.md').write_text('\n'.join(lines)+'\n', encoding='utf8')


if __name__ == '__main__':
    build()
