"""Where does the heading propulsion live: in free space or in contact?

Forward-increment drift (lag 0.4 s), all robots pooled per recording:
    dr/dt = mu_p P + mu_w W + a v_prev + sum_k  v0[k] * e(phi) * 1[robot in class k]
Classes: number of touching neighbours (0, 1, 2, 3+) and, separately, the free gap to the
nearest neighbour (d_nn - sigma, in robot diameters).
"""
import sys
import numpy as np

sys.path.insert(0, r"C:\Users\NSSLab\PycharmProjects\EPDE\projects\robot")
from swarm_data import (swarm_files, swarm_meta, load_swarm, arena_envelope, spin_rates,
                        contact_features, ROBOT_DIAMETER_PX as SIGMA)

LAG = 10


def nearest_gap(x, y):
    T, N = x.shape
    out = np.empty((T, N))
    for t0 in range(0, T, 1000):
        sl = slice(t0, min(T, t0 + 1000))
        R = np.hypot(x[sl][:, None, :] - x[sl][:, :, None], y[sl][:, None, :] - y[sl][:, :, None])
        R[:, np.arange(N), np.arange(N)] = np.inf
        out[sl] = R.min(axis=2)
    return (out - SIGMA) / SIGMA


def fit(path, classes_of, labels):
    t, x, y, phi, ids = load_swarm(path)
    dt = t[1] - t[0]
    xc, yc, r_env = arena_envelope(x, y)
    F = contact_features(x, y, spin_rates(t, phi), xc, yc, r_env)
    cls = classes_of(F, x, y)
    rows = np.arange(LAG, len(t) - LAG)
    res = {}
    for comp, pos, e in (("x", x, np.cos(phi)), ("y", y, np.sin(phi))):
        inc = ((pos[rows + LAG] - pos[rows]) / (LAG * dt)).ravel()
        vb = ((pos[rows] - pos[rows - LAG]) / (LAG * dt)).ravel()
        cols = [F[f"push_{comp}"][rows].ravel(), F[f"wall_{comp}"][rows].ravel(), vb]
        c = cls[rows].ravel()
        present = [k for k in range(len(labels)) if (c == k).sum() >= 200]   # skip (near-)empty classes
        cols += [e[rows].ravel() * (c == k) for k in present]
        A = np.column_stack(cols + [np.ones(len(inc))])
        coef = np.linalg.lstsq(A, inc, rcond=None)[0]
        resid = inc - A @ coef
        cov = np.linalg.pinv(A.T @ A) * resid.var()
        se = np.sqrt(np.diag(cov)) * np.sqrt(2 * LAG)      # rough: inflated for the overlapping lag windows
        res[comp] = [(coef[3 + present.index(k)], se[3 + present.index(k)], int((c == k).sum()))
                     if k in present else (np.nan, np.nan, int((c == k).sum())) for k in range(len(labels))]
    return res


def by_contacts(F, x, y):
    return np.minimum(F["n_contact"], 3).astype(int)


def by_gap(F, x, y):
    g = nearest_gap(x, y)
    return np.digitize(g, [0.1, 0.5, 1.0])          # 0: touching, 1: <0.5 d, 2: 0.5-1 d, 3: >1 d


for name, fn, labels in (("contacts", by_contacts, ["0", "1", "2", "3+"]),
                         ("nearest gap", by_gap, ["touch", "<0.5d", "0.5-1d", ">1d"])):
    print(f"\n### propulsion v0 [px/s] by {name} (x | y), share of samples")
    for f in swarm_files():
        tag = swarm_meta(f)["tag"]
        r = fit(f, fn, labels)
        n_tot = sum(n for _, _, n in r["x"])
        cells = []
        for k, lab in enumerate(labels):
            (vx, sx, n), (vy, sy, _) = r["x"][k], r["y"][k]
            cells.append(f"{lab}: {vx:5.1f}|{vy:5.1f} (+-{max(sx, sy):.1f}, {100 * n / n_tot:4.1f}%)")
        print(f"  {tag:15s} " + "   ".join(cells))
