"""Check availability of complete Navier–Stokes equations in existing actual handovers."""
import csv
from io import StringIO
from itertools import permutations
from settings import ROOT, limit_threads

if __name__ == '__main__':
    limit_threads()

from noise_suite import OPTIONS, SEEDS, atomic_text, baseline_directory, directory, live
from compare import previous, read_archive


def audit(archive, forms, frequencies):
    labels = [previous.label_key(x, frequencies) for x in archive['labels']]
    supports = [frozenset(labels[j] for j, _ in p[2]) for p in archive['points']]
    alternatives = [{f[i] for f in forms} for i in range(3)]
    used = [{supports[c[i]] for c in archive['systems']} for i in range(3)]
    union = set.union(*used)
    exact = {tuple(supports[i] for i in combo) for combo in archive['systems']}
    assembled = any(p in exact for f in forms for p in permutations(f))
    same_slots = any(all(eq in used[i] for i, eq in enumerate(p)) for f in forms for p in permutations(f))
    return dict(x_momentum=int(bool(union & alternatives[0])), y_momentum=int(bool(union & alternatives[1])),
                continuity=int(bool(union & alternatives[2])), assembled=int(assembled),
                recombinable_in_original_slots=int(same_slots),
                all_donors_any_slot=int(any(f <= union for f in map(set, forms))))


if __name__ == '__main__':
    p = live.load_data('ns', **OPTIONS['ns'])
    forms, freq = previous.registry('ns', p)
    rows = []
    for seed in SEEDS:
        for family in ('gaussian','correlated'):
            dest = directory('ns', seed, family) if family == 'correlated' else baseline_directory('ns', seed, 5)
            rows.append(dict(seed=seed, family=family, **audit(read_archive(dest), forms, freq)))
    out = StringIO(newline='')
    writer = csv.DictWriter(out, fieldnames=rows[0], lineterminator='\n')
    writer.writeheader()
    writer.writerows(rows)
    atomic_text(ROOT / 'ns_donors.csv', out.getvalue())
    print(out.getvalue())
