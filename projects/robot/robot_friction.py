r"""Velocity-level (underdamped) laws: friction, with the wide best-subset pipeline.

Velocity  v(t) = (r(t+W) - r(t)) / (W dt)        -- a window of W frames, starting at t
Target    a(t) = (v(t+W) - v(t)) / (W dt)        -- adjacent, non-overlapping windows
Atoms     velocity v_x, v_y, speed (nonlinear friction), and the state at t: contact push,
          wall, gear, heading cos/sin phi, X, Y, n_contact (swarm); X, Y, c, s + t atoms (single)
Pipeline  robot_wide: products of <= 4 atoms, exact best support per complexity 1..5, OLS on the
          lattice, WAPE + window-refit instability, exact front (stable end = least unstable).

Calibration. For an Ornstein-Uhlenbeck velocity (autocovariance s2 exp(-|s|/tau)) the regression of
a(t) on v(t) does not return -1/tau: window averaging and the shared endpoint (tracking noise sn2)
bias it. With window w = W dt:
    Cov(V_next, V) = s2 tau^2/w^2 (1 - e)^2 - sn2/w^2,      e = exp(-w/tau)
    Var(V)         = 2 s2 tau/w (1 - tau/w (1 - e)) + 2 sn2/w^2
    slope b on v   = (Cov/Var - 1) / w
Given the measured Var(V) and the fitted b, tau is solved by bisection. The OU picture is
validated if tau does not depend on W (refits of the selected support at W = 2, 3, 4, 6 frames).

python robot_friction.py [--file all|<tag>|single]
"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import argparse
import json
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import robot_wide as RW                                                    # noqa: E402
from robot_discovery import load_track                                     # noqa: E402
from swarm_data import (swarm_files, swarm_meta, load_swarm, arena_envelope,  # noqa: E402
                        spin_rates, contact_features)

SN2 = 0.4          # tracking-noise variance per coordinate [px^2] (single-robot MSD intercept)
W_MAIN, W_CHECK = 2, (2, 3, 4, 6)


def windows(pos, W, dt):
    """v(t) for t = 0..T-W-1 and a(t) for t = 0..T-2W-1 (as arrays over the leading axis)."""
    v = (pos[W:] - pos[:-W]) / (W * dt)
    a = (v[W:] - v[:-W]) / (W * dt)
    return v, a


def ou_tau(b, var_v, w, sn2=SN2):
    """Relaxation time from the regression slope b of a on v (see module docstring)."""
    target = 1.0 + b * w

    def ratio(tau):
        e = np.exp(-w / tau)
        s2 = (var_v - 2 * sn2 / w ** 2) / (2 * tau / w * (1 - tau / w * (1 - e)))
        cov = s2 * tau ** 2 / w ** 2 * (1 - e) ** 2 - sn2 / w ** 2
        return cov / var_v

    lo, hi = 1e-3, 50.0
    if not (ratio(lo) - target) * (ratio(hi) - target) < 0:
        return np.nan
    for _ in range(100):
        mid = np.sqrt(lo * hi)
        if (ratio(lo) - target) * (ratio(mid) - target) <= 0:
            hi = mid
        else:
            lo = mid
    return float(np.sqrt(lo * hi))


# ------------------------------------------------------------------ data sets
def single_set(W):
    t, x, y, phi = load_track()
    dt = t[1] - t[0]
    pos = np.stack([x - x.mean(), y - y.mean()], -1)
    v, a = windows(pos, W, dt)
    rows = np.arange(len(a))
    atoms = [("v_x", v[rows, 0]), ("v_y", v[rows, 1]), ("speed", np.hypot(*v[rows].T)),
             ("X", pos[rows, 0]), ("Y", pos[rows, 1]), ("c", np.cos(phi[rows])), ("s", np.sin(phi[rows]))]
    atoms += RW.t_atoms(t[rows] + 0.5 * W * dt)
    return dict(atoms=atoms, a={"x": a[:, 0], "y": a[:, 1]}, v={"x": v[rows, 0], "y": v[rows, 1]},
                grid=(len(rows),), lattice=np.arange(0, len(rows), max(1, len(rows) // RW.LATTICE)), dt=dt)


def swarm_set(path, W):
    t, x, y, phi, ids = load_swarm(path)
    dt = t[1] - t[0]
    xc, yc, r_env = arena_envelope(x, y)
    F = contact_features(x, y, spin_rates(t, phi), xc, yc, r_env)
    N = x.shape[1]
    vx, ax = windows(x, W, dt)
    vy, ay = windows(y, W, dt)
    T_in = ax.shape[0]
    flat = lambda arr: np.ascontiguousarray(arr[:T_in].T).ravel()      # noqa: E731  robot-major
    atoms = [("v_x", flat(vx)), ("v_y", flat(vy)), ("speed", flat(np.hypot(vx, vy)))]
    atoms += [(lab, flat(F[lab])) for lab in ("push_x", "push_y", "wall_x", "wall_y", "gear_x", "gear_y",
                                              "n_contact")]
    atoms += [("cos_phi", flat(np.cos(phi))), ("sin_phi", flat(np.sin(phi))), ("X", flat(x - xc)),
              ("Y", flat(y - yc))]
    st = max(1, T_in // RW.LATTICE)
    lat = (np.arange(N)[:, None] * T_in + np.arange(0, T_in, st)[None, :]).ravel()
    return dict(atoms=atoms, a={"x": flat(ax), "y": flat(ay)}, v={"x": flat(vx), "y": flat(vy)},
                grid=(N, T_in), lattice=lat, dt=dt)


# ------------------------------------------------------------------ run
def calibrate(builder, support_by_comp):
    """Refit each component's selected support at several W; tau from its v coefficient."""
    rows = []
    for W in W_CHECK:
        ds = builder(W)
        names = [a[0] for a in ds["atoms"]]
        arr = dict(ds["atoms"])
        for comp, terms in support_by_comp.items():
            cols = [np.prod([arr[f.split("^")[0]] ** (int(f.split("^")[1]) if "^" in f else 1)
                             for f in term.split("*")], axis=0) for term in terms]
            A = np.column_stack(cols)
            coef = np.linalg.lstsq(A, ds["a"][comp], rcond=None)[0]
            fit = dict(zip(terms, coef))
            b = fit.get(f"v_{comp}", np.nan)
            w = W * ds["dt"]
            tau = ou_tau(b, float(np.var(ds["v"][comp])), w) if np.isfinite(b) else np.nan
            rows.append(dict(W=W, comp=comp, b=float(b), tau=tau,
                             forces={k: float(v) for k, v in fit.items() if k != f"v_{comp}"}))
    return rows


def run(label, builder):
    ds = builder(W_MAIN)
    lib = RW.Library(ds["atoms"], ds["lattice"])
    out, support = {}, {}
    for comp in ("x", "y"):
        eq = dict(target=f"a_{comp}", lib=lib, y=ds["a"][comp], grid=ds["grid"], lattice=ds["lattice"])
        cands = RW.candidates(eq)
        RW.show(eq, cands, f"{label} (W={W_MAIN})")
        fr, se = RW.front(cands)
        # the calibrated law: the stable end if it holds v_comp, else the cheapest front member that does
        with_v = [i for i in sorted(fr, key=lambda i: cands[i]["c"]) if f"v_{comp}" in cands[i]["terms"]]
        pick = se if f"v_{comp}" in cands[se]["terms"] else (with_v[0] if with_v else se)
        support[comp] = list(cands[pick]["terms"])
        out[comp] = dict(candidates=cands, front=fr, stable_end=se, calibrated=pick)
    cal = calibrate(builder, support)
    print(f"   calibration of {support}:")
    for r in cal:
        forces = ", ".join(f"{k} {v:+.3g}" for k, v in r["forces"].items())
        print(f"     W={r['W']} ({r['W'] * ds['dt']:.2f}s) a_{r['comp']}: slope on v {r['b']:+.3f}/s -> "
              f"tau {r['tau']:.3f}s (gamma {1 / r['tau'] if r['tau'] else np.nan:.2f}/s) | {forces}")
    return dict(results=out, calibration=cal, support=support)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="all", help="'single', a recording tag, or 'all' (single + swarm)")
    a = ap.parse_args()
    res = {}
    if a.file in ("all", "single"):
        res["single"] = run("single", single_set)
    for f in swarm_files():
        tag = swarm_meta(f)["tag"]
        if a.file in ("all", tag):
            res[tag] = run(tag, lambda W, f=f: swarm_set(f, W))
        json.dump(res, open(os.path.join(RW.RESULTS, "wide_friction.json"), "w"), indent=1, default=float)
    json.dump(res, open(os.path.join(RW.RESULTS, "wide_friction.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
