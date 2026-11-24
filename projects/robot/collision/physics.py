"""Kinematics for the transfer study. Length is in camera pixels and time is in seconds."""
from dataclasses import dataclass
import os
from pathlib import Path
import sys

import numpy as np

ROBOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROBOT))
from swarm_data import FPS, ROBOT_DIAMETER_PX as SIGMA, arena_envelope, load_swarm, swarm_files, swarm_meta

LAG = 10
DT = LAG / FPS
CAP = 4.87 * 45.08198974249223  # fixed instrument calibration from the earlier study
CUTOFF = 1.1 * SIGMA
EDGES = np.array([0., .93, .97, 1., 1.05, 1.1])
NAMES = ['push', 'overlap', 'front', 'neighbor_front', 'approach', 'slide', 'gear',
         'drive', 'drive_in'] + [f'band{i}' for i in range(5)]
SEEDS = (923701, 923702, 923703)
OUT = Path(os.environ.get('EPDE_COLLISION_OUTPUT', ROBOT / 'results' / 'collision_transfer')).resolve()


@dataclass
class Track:
    tag: str
    pwm: int
    ids: np.ndarray
    pos: np.ndarray
    phi: np.ndarray

    def part(self, start, stop):
        return Track(self.tag, self.pwm, self.ids, self.pos[start:stop], self.phi[start:stop])


def recordings():
    tracks = {}
    for path in swarm_files():
        meta = swarm_meta(path)
        _, x, y, phi, ids = load_swarm(path)
        tracks[meta['tag']] = Track(meta['tag'], meta['pwm'], ids, np.stack((x, y), -1), phi)
    return tracks


def partition(tracks, held):
    train, valid = [], []
    for tag, track in tracks.items():
        if tag != held:
            cut = len(track.pos) * 3 // 4
            train.append(track.part(0, cut))
            valid.append(track.part(cut, None))
    return train, valid, tracks[held]


def rotate(a):
    return np.stack((-a[..., 1], a[..., 0]), axis=-1)


def heading(phi):
    return np.stack((np.cos(phi), np.sin(phi)), axis=-1)


def geometry(pos):
    delta = pos[..., :, None, :] - pos[..., None, :, :]
    dist = np.linalg.norm(delta, axis=-1)
    idx = np.arange(pos.shape[-2])
    dist[..., idx, idx] = np.inf
    unit = np.divide(delta, dist[..., None], out=np.zeros_like(delta), where=dist[..., None] > 0)
    return dist, unit


def wall_vectors(pos, arena):
    offset = pos - arena[:2]
    rad = np.linalg.norm(offset, axis=-1)
    wall = rad > arena[2] - .25 * SIGMA
    return -offset / np.maximum(rad[..., None], 1e-12) * wall[..., None]


def arena_for(train):
    # Only the training prefixes supply wall geometry, even for a previously seen arena size.
    arenas = {}
    for track in train:
        size = int(track.tag.rsplit('D', 1)[1])
        arenas.setdefault(size, []).append(arena_envelope(*track.pos[::5].transpose(2, 0, 1)))
    return {size: np.median(values, axis=0) for size, values in arenas.items()}


def features(pos, e, prev, omega, prop):
    """One vector per physical mechanism, summed over neighbors before fitting either component."""
    d, u = geometry(pos)
    touch = d < CUTOFF
    active = touch[..., None] * u
    tang = rotate(active)
    vi = prev[..., :, None, :] - prev[..., None, :, :]
    normal_v = np.sum(vi * u, axis=-1)
    front_i = -np.sum(e[..., :, None, :] * u, axis=-1)
    front_j = np.sum(e[..., None, :, :] * u, axis=-1)
    drive_in = np.maximum(np.sum(prop[..., None, :, :] * u, axis=-1), 0)
    slip = np.sum(vi * tang, axis=-1)
    def summed(v, weight):
        return (v * weight[..., None]).sum(axis=-2)
    vectors = [active.sum(-2), summed(active, np.maximum(0., (CUTOFF-d)/SIGMA)),
               summed(active, front_i), summed(active, front_j), summed(active, np.maximum(-normal_v, 0)),
               summed(tang, slip), summed(tang, (omega[..., :, None] + omega[..., None, :])),
               (touch[..., None] * prop[..., None, :, :]).sum(-2), summed(active, drive_in)]
    vectors += [summed(active, (d >= lo*SIGMA) & (d < hi*SIGMA)) for lo, hi in zip(EDGES[:-1], EDGES[1:])]
    return np.stack(vectors, axis=-1), touch.sum(-1)


def input_at(track, rows):
    p = track.pos[rows]
    e = heading(track.phi[rows])
    v = (p - track.pos[rows-LAG]) / DT
    omega = (track.phi[rows] - track.phi[rows-LAG]) / DT
    return p, e, v, omega


def fit_self(train, arenas):
    """Joint fit on contact-free training rows, with individual propulsion and common wall and memory."""
    keys = sorted({(tr.pwm, int(i)) for tr in train for i in tr.ids})
    k = len(keys)
    gram = np.zeros((2*k+2, 2*k+2))
    cross = np.zeros(2*k+2)
    spin = {}
    for tr in train:
        arena = arenas[int(tr.tag[-2:])]
        rows = np.arange(LAG, len(tr.pos)-LAG, LAG)
        p, e, prev, _ = input_at(tr, rows)
        y = (tr.pos[rows+LAG]-p)/DT
        w = wall_vectors(p, arena)
        free = (geometry(p)[0] < CUTOFF).sum(-1) == 0
        rates = np.diff(tr.phi[rows], axis=0)
        for j, rid in enumerate(tr.ids):
            key = tr.pwm, int(rid)
            spin.setdefault(key, []).append(rates[:, j])
            m = free[:, j]
            A = np.stack((e[m,j], rotate(e[m,j]), prev[m,j], w[m,j]), -1).reshape(-1,4)
            ix = [2*keys.index(key), 2*keys.index(key)+1, 2*k, 2*k+1]
            gram[np.ix_(ix, ix)] += A.T @ A
            cross[ix] += A.T @ y[m,j].ravel()
    coef = np.linalg.lstsq(gram, cross, rcond=1e-12)[0]
    lookup = {}
    for j, key in enumerate(keys):
        increments = np.concatenate(spin[key])
        lookup[key] = np.r_[coef[2*j:2*j+2], increments.mean()/DT, increments.std()]
    for pwm in {x[0] for x in keys}:
        lookup[pwm, -1] = np.mean([lookup[key] for key in keys if key[0]==pwm], axis=0)
    return lookup, coef[-2:]


def robot_parameters(track, lookup):
    return np.stack([lookup.get((track.pwm, int(rid)), lookup[track.pwm, -1]) for rid in track.ids])


def motion_inputs(pos, e, prev, omega, params, common, arena):
    prop = params[:, 0, None]*e + params[:, 1, None]*rotate(e)
    base = prop + common[0]*prev + common[1]*wall_vectors(pos, arena)
    X, contacts = features(pos, e, prev, omega, prop)
    return X, base, contacts


def design(track, lookup, common, arenas, stride):
    rows = np.arange(LAG, len(track.pos)-LAG, stride)
    params, arena = robot_parameters(track, lookup), arenas[int(track.tag[-2:])]
    shape = (len(rows), len(params), 2)
    X = np.empty((*shape, len(NAMES)), np.float32)
    base, target = np.empty(shape, np.float32), np.empty(shape, np.float32)
    n = np.empty(shape[:2], np.int16)
    for start in range(0, len(rows), 48):
        sl = slice(start, start+48)
        r = rows[sl]
        pos, e, prev, omega = input_at(track, r)
        X[sl], base[sl], n[sl] = motion_inputs(pos, e, prev, omega, params, common, arena)
        target[sl] = (track.pos[r+LAG]-pos)/DT
    return rows, X, base, target, n
