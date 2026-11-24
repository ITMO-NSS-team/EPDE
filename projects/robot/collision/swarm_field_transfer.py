"""Leave one recording out for 0.4 s forecasts of swarm density and polarization.

Measure Gaussian fields in the interior as in swarm_fields_binned. Use raw centers
and the held fold's archived training-only arena, propulsion and spin calibration.
Fit each fixed equation on the first three quarters of the four training tracks.
Score increments at common origins in the held recording. All spatial derivatives
are analytic derivatives of the Gaussian at the start of each forecast.
"""
import csv
import json
from pathlib import Path

import numpy as np

from physics import (CAP, CUTOFF, DT, LAG, OUT, Track, geometry, heading,
                     robot_parameters, rotate, wall_vectors)
from swarm_fields import BOUNDARY, NGRID
from swarm_fields_binned import groups


DEST = Path(__file__).resolve().parents[1] / 'results' / 'swarm_field_transfer.csv'
DENSITY = {
    'unchanged': (),
    'drive': ('drive',),
    'drive + diffusion': ('drive', 'diffusion'),
    'drive + diffusion + contact': ('drive', 'diffusion', 'contact'),
    'drive + diffusion + wall': ('drive', 'diffusion', 'wall'),
    'drive + diffusion + contact + wall': ('drive', 'diffusion', 'contact', 'wall'),
    'previous velocity': ('previous',),
    'drive + previous velocity': ('drive', 'previous'),
    'drive + previous velocity + diffusion': ('drive', 'previous', 'diffusion'),
    'drive + previous velocity + diffusion + contact + wall':
        ('drive', 'previous', 'diffusion', 'contact', 'wall'),
}


def moments(target, columns, offset=0):
    """Real second moments also handle complex x/y vectors via their dot product."""
    values = np.stack([v.ravel() for v in (*columns, target - offset)], axis=1)
    return (values.conj().T @ values).real / target.size


def field_moments(track, calibration, arena):
    params = robot_parameters(track, calibration)
    pos, phi = track.pos[::LAG], track.phi[::LAG]
    direction = heading(phi)
    drive = direction * params[:, 0, None] + rotate(direction) * params[:, 1, None]
    # Robot diameters are the distance unit, and the Gaussian width is one diameter.
    half = arena[2] / (np.sqrt(2) * CAP)
    grid = np.linspace(-half, half, NGRID)[BOUNDARY:-BOUNDARY]
    gx, gy = np.meshgrid(grid, grid, indexing='ij')
    dx = gx.ravel() - (pos[:, :, 0, None] - arena[0]) / CAP
    dy = gy.ravel() - (pos[:, :, 1, None] - arena[1]) / CAP
    r2 = dx**2 + dy**2
    kernel = np.exp(-0.5 * r2) / 8  # area of a cap times the normalized Gaussian
    rho = kernel.sum(axis=1)
    now = slice(1, -1)

    def flux(velocity):
        # -div(sum_i K_i v_i) over one step
        return DT * (kernel[now] * (dx[now] * velocity[:, :, 0, None]
                                    + dy[now] * velocity[:, :, 1, None])).sum(axis=1)

    dist, unit = geometry(pos[now])
    push = (unit * (dist < CUTOFF)[..., None]).sum(axis=2)
    atoms = dict(drive=flux(drive[now] / CAP), previous=flux((pos[now] - pos[:-2]) / (CAP * DT)),
                 diffusion=DT * (kernel[now] * (r2[now] - 2)).sum(axis=1),
                 contact=flux(push), wall=flux(wall_vectors(pos[now], arena)))
    delta = rho[2:] - rho[now]
    stats = {('density', name): (moments(delta, [atoms[key] for key in keys]), delta.var())
             for name, keys in DENSITY.items()}

    rates = params[:, 2]
    labels = groups(rates, 2)
    orientation = np.exp(1j * (phi + np.arctan2(params[:, 1], params[:, 0])))

    def polarization(group, phase):
        return np.einsum('tiq,ti->tq', kernel[:, group], phase[:, group])

    fields, group_rotated, individual_rotated = [], [], []
    future_phase = orientation * np.exp(1j * DT * rates)
    for b in range(4):
        group = labels == b
        field = polarization(group, orientation)
        fields.append(field)
        group_rotated.append(field[now] * np.exp(1j * DT * rates[group].mean()))
        individual_rotated.append(polarization(group, future_phase)[now])
    field = np.stack(fields, axis=-1)
    present = field[now]
    delta = field[2:] - present
    # Center each group's x and y component independently for the vector R-squared.
    variance = delta.var(axis=(0, 1)).mean()
    for name, advanced in [('unchanged', None), ('amplitude only', present),
                           ('group spin', np.stack(group_rotated, axis=-1)),
                           ('individual spin', np.stack(individual_rotated, axis=-1))]:
        m = moments(delta, []) if advanced is None else moments(delta, [advanced], -present)
        stats['polarization', name] = m, variance
    return stats, len(delta)


def score(training, held_stats, n_steps, tag, missing):
    rows = []
    for (target, model), (test, variance) in held_stats.items():
        train = np.mean([s[target, model][0] for s in training], axis=0)
        coef = np.linalg.lstsq(train[:-1, :-1], train[:-1, -1], rcond=1e-12)[0]
        weights = np.r_[coef, -1.]
        mse = weights @ test @ weights
        names = DENSITY[model] if target == 'density' else ['amplitude']
        rows.append(dict(recording=tag, target=target, model=model, origins=n_steps,
                         fallback_robots=missing, r2=1 - mse / variance, rmse=np.sqrt(mse),
                         coefficients=json.dumps(dict(zip(names, coef.tolist())))))
    return rows


def main():
    tracks = {}
    for path in sorted((OUT / 'preprocessed').glob('N*.npz')):
        with np.load(path) as z:
            tracks[path.stem] = Track(path.stem, int(z['pwm']), z['ids'], z['raw_pos'], z['phi'])
    results = []
    for held, test_track in tracks.items():
        fold = json.loads((OUT / held / 'training.json').read_text())
        assert set(fold['train']) == set(tracks) - {held} and fold['held'] == held
        lookup = {tuple(map(int, k.split('/'))): np.array(v) for k, v in fold['lookup'].items()}

        def summarize(track):
            return field_moments(track, lookup, np.array(fold['arenas'][track.tag.rsplit('D', 1)[1]]))

        training = [summarize(tracks[tag].part(0, fold['split_frames'][tag]))[0] for tag in fold['train']]
        stats, steps = summarize(test_track)
        missing = sum((test_track.pwm, int(i)) not in lookup for i in test_track.ids)
        results.extend(score(training, stats, steps, held, missing))
    with DEST.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    for target, name in stats:
        selected = [r for r in results if (r['target'], r['model']) == (target, name)]
        scores = [r['r2'] for r in selected]
        print(f'{target:13} {name:56} mean R2 {np.mean(scores):.4f} '
              f'[{min(scores):.4f}, {max(scores):.4f}] mean RMSE {np.mean([r["rmse"] for r in selected]):.6f}')


if __name__ == '__main__':
    main()
