"""Loader for the multi-robot recordings ``data_00_330_[<N>_bots_PWM_<p>_<a>cw_<b>ccw_D_<d>cm].MP4.pickle``.

Each pickle is a list of frames (25 fps); a frame lists one ``[id, heading_deg, (x_px, y_px)]``
entry per robot. Every robot is present in every frame and IDs are the physical robots
(the same ID keeps its spin direction and rate across recordings).
"""
import glob
import os
import pickle
import re

import numpy as np

DATA_DIR = os.path.dirname(os.path.abspath(__file__))
FPS = 25.0


def swarm_files():
    return sorted(glob.glob(os.path.join(DATA_DIR, "data_00_330_*.pickle")))


def swarm_meta(path):
    m = re.search(r"\[(\d+)_bots_PWM_(\d+)_(\d+)cw_(\d+)ccw_D_(\d+)cm\]", os.path.basename(path))
    n, pwm, ncw, nccw, dcm = map(int, m.groups())
    return dict(n=n, pwm=pwm, ncw=ncw, nccw=nccw, arena_cm=dcm, tag=f"N{n}_PWM{pwm}_D{dcm}")


def load_swarm(path):
    """Return t (T,), x, y (T, N) in pixels, phi (T, N) unwrapped radians, ids (N,)."""
    with open(path, "rb") as fh:
        frames = pickle.load(fh)
    ids = sorted(int(e[0]) for e in frames[0])
    col = {i: k for k, i in enumerate(ids)}
    T, N = len(frames), len(ids)
    x = np.empty((T, N)); y = np.empty((T, N)); ang = np.empty((T, N))
    for t, frame in enumerate(frames):
        for rid, a, (px, py) in frame:
            k = col[int(rid)]
            x[t, k], y[t, k], ang[t, k] = px, py, a
    phi = np.unwrap(np.deg2rad(ang), axis=0)
    return np.arange(T) / FPS, x, y, phi, np.array(ids)


ROBOT_DIAMETER_PX = 230.0  # first peak of the pair-distance distribution (~5 cm)


def spin_rates(t, phi):
    """Each robot's mean spin [rad/s] (phase slope); spin is a property of the physical robot."""
    return np.array([np.polyfit(t, phi[:, k], 1)[0] for k in range(phi.shape[1])])


def contact_features(x, y, w, xc, yc, r_env, sigma=ROBOT_DIAMETER_PX, contact=1.1, wall_band=0.25):
    """Per-robot interaction features from positions at the same instant, each (T, N):

    push_x/y  sum over touching neighbours (d < contact*sigma) of the unit vector pointing AWAY
              from the neighbour -- the direction excluded volume pushes a robot.
    wall_x/y  inward unit vector while the robot touches the wall (r > r_env - wall_band*sigma).
    gear_x/y  sum over touching neighbours of (w_i + w_j) z x r_hat_ij: contact-friction rolling.
    n_contact, n_same, n_opp   number of touching neighbours (all / same spin sign / opposite).

    push and wall are gradient-type (visible only to forward increments of stochastic data),
    gear is transverse (visible to any derivative)."""
    T, N = x.shape
    out = {k: np.zeros((T, N)) for k in ("push_x", "push_y", "gear_x", "gear_y",
                                          "n_contact", "n_same", "n_opp")}
    same_spin = (np.sign(w)[:, None] == np.sign(w)[None, :])[None]
    for t0 in range(0, T, 1000):
        sl = slice(t0, min(T, t0 + 1000))
        dX = x[sl][:, None, :] - x[sl][:, :, None]          # [t, i, j] = x_j - x_i
        dY = y[sl][:, None, :] - y[sl][:, :, None]
        R = np.hypot(dX, dY); R[:, np.arange(N), np.arange(N)] = np.inf
        C = R < contact * sigma
        ux, uy = np.where(C, -dX / R, 0.0), np.where(C, -dY / R, 0.0)   # unit j -> i
        S = (w[:, None] + w[None, :])[None]
        out["push_x"][sl], out["push_y"][sl] = ux.sum(2), uy.sum(2)
        out["gear_x"][sl], out["gear_y"][sl] = (S * uy).sum(2), (S * -ux).sum(2)
        out["n_contact"][sl] = C.sum(2)
        out["n_same"][sl], out["n_opp"][sl] = (C & same_spin).sum(2), (C & ~same_spin).sum(2)
    r = np.hypot(x - xc, y - yc)
    on_wall = r > r_env - wall_band * sigma
    out["wall_x"] = np.where(on_wall, -(x - xc) / r, 0.0)
    out["wall_y"] = np.where(on_wall, -(y - yc) / r, 0.0)
    return out


def _fit_circle(x, y):
    A = np.column_stack([x, y, np.ones_like(x)])
    c, *_ = np.linalg.lstsq(A, x ** 2 + y ** 2, rcond=None)
    xc, yc = c[0] / 2, c[1] / 2
    return xc, yc, np.sqrt(c[2] + xc ** 2 + yc ** 2)


def arena_envelope(x, y, q=99.5):
    """Centre and radius of the envelope of robot CENTRES (the wall sits one robot radius further out).
    Polar-binned q-th radius percentile around a rough centre, circle-fitted, iterated."""
    xs, ys = x.ravel(), y.ravel()
    xc, yc = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2
    for _ in range(3):
        th = np.arctan2(ys - yc, xs - xc); r = np.hypot(xs - xc, ys - yc)
        bins = np.linspace(-np.pi, np.pi, 73)
        ex, ey = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (th >= lo) & (th < hi)
            if m.sum() < 50:
                continue
            rr, tt = np.percentile(r[m], q), (lo + hi) / 2
            ex.append(xc + rr * np.cos(tt)); ey.append(yc + rr * np.sin(tt))
        xc, yc, R = _fit_circle(np.array(ex), np.array(ey))
    return xc, yc, R
