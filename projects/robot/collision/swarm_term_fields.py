"""Build causal field-increment libraries and recording-balanced folds for swarm_term_search."""
import argparse
import json
import os
from pathlib import Path

for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '1'

import numpy as np

from physics import CAP, CUTOFF, DT, LAG, OUT, Track, geometry, robot_parameters, wall_vectors
from swarm_fields import BOUNDARY, NGRID
from swarm_fields_binned import groups

DEST = Path(__file__).resolve().parents[1] / 'results' / 'swarm_terms_epde'
LIB = {
    'density': ('Drive', 'Diffusion', 'Contact', 'Wall', 'Memory', 'CrowdedDrive',
                'NonlinearDiffusion', 'BlockedPair', 'BlockedWall', 'SpinContact', 'SpinWall', 'TurningDrive'),
    'polarization': ('IndividualTurn', 'Damping', 'GroupTurn', 'SpinMemory', 'DriveTransport',
                     'ContactTransport', 'WallTransport', 'BlockedTransport', 'DensityGradient',
                     'NematicTransport', 'PolarDiffusion', 'CrowdedDamping'),
}
SCREEN_ROWS = 512
COMBINED = {'polarization': (*LIB['polarization'],'DelayedTurn','SlowTurn')}


def complex_xy(x):
    return x[..., 0] + 1j * x[..., 1]


def dot(a, b):
    return (a.conj() * b).real


def field_block(pos, phi, rows, params, arena, grid, labels, combined=False):
    current = pos[rows]
    rates = params[:, 2]
    phase_offset = np.angle(complex_xy(params[:, :2]))
    p = np.exp(1j * (phi[rows] + phase_offset))
    advanced = p * np.exp(1j * DT * rates)
    drive = p * (np.linalg.norm(params[:, :2], axis=1) / CAP)
    wall = complex_xy(wall_vectors(current, arena))
    dist, normal = geometry(current)
    n = complex_xy(normal) * (dist < CUTOFF)
    push = n.sum(axis=2)
    relative = drive[:, :, None] - drive[:, None, :]
    blocked = .5 * (n * np.maximum(-dot(relative, n), 0)).sum(axis=2)
    wall_block = wall * np.maximum(-dot(drive, wall), 0)
    gear = .5j * (n * (rates[:, None] + rates[None, :])).sum(axis=2)
    d = grid - (complex_xy(current) - complex_xy(arena[:2]))[:, :, None] / CAP
    next_d = grid - (complex_xy(pos[rows + 1]) - complex_xy(arena[:2]))[:, :, None] / CAP
    k = np.exp(-.5 * abs(d)**2) / 8
    next_k = np.exp(-.5 * abs(next_d)**2) / 8
    kd = k * d
    lap = k * (abs(d)**2 - 2)

    def kernel_flux(velocity):
        return DT * dot(kd, velocity[:, :, None])

    def weighted(kernel, values):
        return np.einsum('tnq,tn->tq', kernel, values)

    drive_flux = kernel_flux(drive)
    push_flux = kernel_flux(push)
    wall_flux = kernel_flux(wall)
    block_flux = kernel_flux(blocked + wall_block)
    rho = k.sum(axis=1)
    neg_grad = kd.sum(axis=1)
    diffusion = DT * lap.sum(axis=1)
    drive_current = weighted(k, drive)
    memory = complex_xy(current - pos[rows - 1]) / (CAP * DT)
    density = dict(Drive=drive_flux.sum(axis=1), Diffusion=diffusion, Contact=push_flux.sum(axis=1),
                   Wall=wall_flux.sum(axis=1), Memory=kernel_flux(memory).sum(axis=1),
                   CrowdedDrive=rho * drive_flux.sum(axis=1) + DT * dot(neg_grad, drive_current),
                   NonlinearDiffusion=rho * diffusion + DT * abs(neg_grad)**2,
                   BlockedPair=kernel_flux(blocked).sum(axis=1), BlockedWall=kernel_flux(wall_block).sum(axis=1),
                   SpinContact=kernel_flux(gear).sum(axis=1), SpinWall=kernel_flux(1j * rates * wall).sum(axis=1),
                   TurningDrive=kernel_flux(1j * DT * rates * drive).sum(axis=1))
    if not combined:
        yield 'density', np.stack([density[n] for n in LIB['density']] + [next_k.sum(axis=1) - rho], axis=-1)

    next_p = np.exp(1j * (phi[rows + 1] + phase_offset))
    previous_deviation = phi[rows] - phi[rows - 1] - DT * rates
    outputs = []
    for b in range(4):
        m = labels == b
        current_p = weighted(k[:, m], p[:, m])
        rotated = weighted(k[:, m], advanced[:, m])
        delta = weighted(next_k[:, m], next_p[:, m]) - current_p
        def transported(flux):
            return weighted(flux[:, m], advanced[:, m])

        # (p p^T - I/2) d = p^2 conj(d)/2 in two dimensions.
        polar = dict(IndividualTurn=rotated-current_p, Damping=-rotated,
                     GroupTurn=current_p * np.expm1(1j * DT * rates[m].mean()),
                     SpinMemory=weighted(k[:, m], 1j * previous_deviation[:, m] * advanced[:, m]),
                     DriveTransport=transported(drive_flux), ContactTransport=transported(push_flux),
                     WallTransport=transported(wall_flux), BlockedTransport=transported(block_flux),
                     DensityGradient=DT * kd[:, m].sum(axis=1),
                     NematicTransport=.5 * DT * weighted(kd[:, m].conj(), p[:, m]**2),
                     PolarDiffusion=DT * weighted(lap[:, m], p[:, m]), CrowdedDamping=-rho * rotated)
        if combined:
            delayed = phi[rows-1][:,m] - phi[rows-2][:,m] - DT*rates[m]
            slow = (phi[rows-1][:,m] - phi[rows-31][:,m])/30 - DT*rates[m]
            polar['DelayedTurn'] = weighted(k[:,m], 1j*delayed*advanced[:,m])
            polar['SlowTurn'] = weighted(k[:,m], 1j*slow*advanced[:,m])
        names = (COMBINED if combined else LIB)['polarization']
        outputs.append(np.stack([polar[n] for n in names] + [delta], axis=-1))
    yield 'polarization', np.stack(outputs, axis=2)  # time, space, group, column (complex)


def summarize(track, params, arena, rows, seed, sample, combined=False):
    half = arena[2] / (np.sqrt(2) * CAP)
    g = np.linspace(-half, half, NGRID)[BOUNDARY:-BOUNDARY]
    grid = (g[:, None] + 1j * g[None, :]).ravel()
    pos, phi = track.pos[::LAG], track.phi[::LAG]
    labels = groups(params[:, 2], 2)
    out = {}
    for target, names in (COMBINED if combined else LIB).items():
        scalar_rows = len(rows) * len(grid) * (8 if target == 'polarization' else 1)
        draws = np.random.default_rng(seed).choice(scalar_rows, SCREEN_ROWS, replace=False) if sample else []
        out[target] = dict(gram=np.zeros((len(names)+1, len(names)+1)), ysum=0, n=0,
                           screen=[], draws=np.sort(draws), scalar_offset=0)
    for start in range(0, len(rows), 32):
        for target, data in field_block(pos, phi, rows[start:start+32], params, arena, grid, labels, combined):
            s = out[target]
            y = data[..., -1]
            x = data.reshape(-1, data.shape[-1])
            s['gram'] += (x.conj().T @ x).real
            s['n'] += len(x)
            s['ysum'] += y.sum(axis=(0, 1))
            if sample:
                # Sort rows by recording and time for the native stability objective.
                m = 2 if np.iscomplexobj(x) else 1
                offset = s['scalar_offset']
                idx = s['draws'][(s['draws'] >= offset) & (s['draws'] < offset+m*len(x))] - offset
                sampled = x[idx // m]
                if m == 2:
                    sampled = np.where((idx % 2)[:, None], sampled.imag, sampled.real)
                s['screen'].append(sampled)
                s['scalar_offset'] += m * len(x)
    packed = {}
    for target, s in out.items():
        gram = s['gram'] / s['n']
        component_mean = s['ysum'] * np.size(s['ysum']) / s['n']
        packed[target+'_gram'] = gram
        packed[target+'_variance'] = gram[-1, -1] - np.mean(abs(component_mean)**2)
        if sample:
            packed[target+'_screen'] = np.concatenate(s['screen'])
            assert len(packed[target+'_screen']) == SCREEN_ROWS
    return packed


def prepare(combined=False):
    destination = DEST/'combined' if combined else DEST
    destination.mkdir(exist_ok=True, parents=True)
    history_steps = 31 if combined else 1
    tracks = {}
    for path in sorted((OUT/'preprocessed').glob('N*.npz')):
        with np.load(path) as z:
            tracks[path.stem] = Track(path.stem, int(z['pwm']), z['ids'], z['raw_pos'], z['phi'])
    for held in tracks:
        fold = json.loads((OUT/held/'training.json').read_text())
        assert set(fold['train']) == set(tracks) - {held}
        lookup = {tuple(map(int, k.split('/'))): np.array(v) for k, v in fold['lookup'].items()}
        parts = {}
        for index, (tag, tr) in enumerate(tracks.items()):
            params = robot_parameters(tr, lookup)
            arena = np.array(fold['arenas'][tag[-2:]])
            rows = np.arange(history_steps, len(tr.pos[::LAG])-1)
            if tag == held:
                roles = [('test', rows)]
            else:
                cut = fold['split_frames'][tag]
                roles = [('train', rows[(rows+1)*LAG < cut]), ('valid', rows[(rows-history_steps)*LAG >= cut])]
            for role, origins in roles:
                item = summarize(tr, params, arena, origins, 4301+index, role == 'train',combined)
                for k, value in item.items():
                    parts.setdefault(role+'_'+k, []).append(value)
        np.savez_compressed(destination/f'{held}_moments.npz', **{k: np.stack(v) for k,v in parts.items()})
        print('Prepared', held, flush=True)


if __name__ == '__main__':
    cli = argparse.ArgumentParser()
    cli.add_argument('--combined',action='store_true')
    prepare(cli.parse_args().combined)
