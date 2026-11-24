"""Replay the common radial law with and without the signed tangential spin term."""
import csv
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from discovery import solve
from physics import CAP, CUTOFF, DT, LAG, NAMES, OUT, design, recordings, rotate


RADIAL = ['overlap', 'front', 'approach', 'drive_in']


def score(track):
    folder = OUT / track.tag
    train = json.loads((folder / 'training.json').read_text())
    assert train['held'] == track.tag and track.tag not in train['train']
    lookup = {tuple(map(int, k.split('/'))): np.array(v) for k, v in train['lookup'].items()}
    arenas = {int(k): np.array(v) for k, v in train['arenas'].items()}
    frames, x, base, y, _ = design(track, lookup, np.array(train['common']), arenas, LAG)
    with np.load(folder / 'prepared.npz') as data:
        g = data['strong_gram'].mean(0)
    beta = solve(g, RADIAL + ['gear'])
    pred = {'radial': base + x @ solve(g, RADIAL), 'with_spin': base + x @ beta}
    omega = (track.phi[frames] - track.phi[frames - LAG]) / DT
    left, right = np.triu_indices(len(track.ids), 1)
    delta = track.pos[frames][:, right] - track.pos[frames][:, left]
    distance = np.linalg.norm(delta, axis=-1)
    t, k = np.nonzero((distance < CUTOFF) & (distance > 0))
    i, j = left[k], right[k]
    normal = delta[t, k] / distance[t, k, None]
    tangent = rotate(normal)
    def components(v):
        rel = v[t, j] - v[t, i]
        return np.column_stack(((rel * normal).sum(1), (rel * tangent).sum(1)))
    observed = components(y)
    errors = {name: components(v) - observed for name, v in pred.items()}
    signs = np.sign(omega[t, i]), np.sign(omega[t, j])
    groups = {'all': np.ones(len(t), bool), 'both_positive': (signs[0] == 1) & (signs[1] == 1),
              'both_negative': (signs[0] == -1) & (signs[1] == -1), 'opposite': signs[0] * signs[1] < 0,
              'zero_rate': signs[0] * signs[1] == 0}
    output = []
    for label, cutoff in (('interaction', CUTOFF), ('cap', CAP)):
        for group, mask in groups.items():
            keep = mask & (distance[t, k] < cutoff)
            row = dict(recording=track.tag, distance=label, spin_group=group, pair_frames=int(keep.sum()),
                       gear_px_per_rad=beta[NAMES.index('gear')])
            for name, v in pred.items():
                error = errors[name][keep]
                row[name+'_whole_recording_r2'] = 1 - np.mean((v-y)**2) / np.var(y, dtype=float)
                for axis, component in enumerate(('normal', 'tangent')):
                    row[name+'_'+component+'_rmse_px_s'] = np.mean(error[:, axis]**2)**.5 if len(error) else None
            row['observed_tangent_mean_px_s'] = observed[keep, 1].mean() if keep.any() else None
            output.append(row)
    return output


if __name__ == '__main__':
    from run import check_inputs
    check_inputs()
    rows = [row for tr in recordings().values() for row in score(tr)]
    path = OUT / 'spin_direction.csv'
    with NamedTemporaryFile('w', dir=OUT, newline='', encoding='utf-8', delete=False) as f:
        writer = csv.DictWriter(f, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    Path(f.name).replace(path)
    print(path)
