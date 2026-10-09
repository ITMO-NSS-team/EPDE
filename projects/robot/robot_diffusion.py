r"""Stochastic compensation: the noise half of the SDE  dr = f(state) dt + sqrt(2 D(state)) dW.

1. Drift.  Per recording, the drift law of robot_wide (push + wall + heading + persistence) is refit by
   OLS on forward increments (r(t+L) - r(t)) / (L dt), all robots pooled.
2. Quadratic-variation ("stochastic") token.  With the drift residual e = increment/(L dt) - f_hat,
   q = (e_x^2 + e_y^2)/4 * (L dt)  is a local estimate of the diffusion coefficient D [px^2/s]
   (plus a constant tracking-noise term). The wide best-subset pipeline (products of <= 4 atoms,
   exact best support per complexity, window-refit instability front) finds D(state) from state atoms:
   n_contact, n_same, wall, |push|, r/R, |vb| (previous speed), |w| (own spin rate).
3. Noise-compensated fit quality.  Pure diffusion makes the residual variance fall as 1/(L dt); a
   deterministic part the drift law misses does not. Fitting  E[e^2] = 2 D/(L dt) + U  over lags
   L = 5, 10, 20, 40 frames separates the noise floor 2D/(L dt) from the unexplained drift U, so
   R2_explainable = Var(f_hat) / (Var(f_hat) + U)  is the share of the DETERMINISTIC motion captured.
   CAVEAT: this assumes white noise at these lags. The swarm's velocity memory (persistence, spin-locked
   circling, caging; robot_vmemory.py) violates it, so the decomposition drifts with lag -- indicative only.
4. Heteroscedastic refit.  Weights 1/D_hat(state) from the diffusion law's stable end; drift
   coefficients compared with the unweighted fit.

python robot_diffusion.py [--file all|<tag>]
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
from swarm_data import (swarm_files, swarm_meta, load_swarm, arena_envelope,  # noqa: E402
                        spin_rates, contact_features)

LAG, LAGS = 10, (5, 10, 20, 40)
DRIFT = {"x": ["push_x", "wall_x", "cos_phi", "vb_x"], "y": ["push_y", "wall_y", "sin_phi", "vb_y"]}


def state(path):
    t, x, y, phi, ids = load_swarm(path)
    dt = t[1] - t[0]
    xc, yc, r_env = arena_envelope(x, y)
    w = spin_rates(t, phi)
    F = contact_features(x, y, w, xc, yc, r_env)
    F["cos_phi"], F["sin_phi"] = np.cos(phi), np.sin(phi)
    F["wall"] = (np.abs(F["wall_x"]) + np.abs(F["wall_y"]) > 0).astype(float)
    F["push_mag"] = np.hypot(F["push_x"], F["push_y"])
    F["rR"] = np.hypot(x - xc, y - yc) / r_env
    F["w_abs"] = np.broadcast_to(np.abs(w), x.shape)
    return t, x, y, dt, F


def drift_fit(x, y, dt, F, L, rows, weights=None):
    """OLS (or WLS) of forward increments on the drift support; returns fits, residuals, f_hat."""
    vb = {"x": np.zeros_like(x), "y": np.zeros_like(y)}
    vb["x"][L:] = (x[L:] - x[:-L]) / (L * dt)
    vb["y"][L:] = (y[L:] - y[:-L]) / (L * dt)
    out = {}
    for comp, pos in (("x", x), ("y", y)):
        inc = ((pos[L:] - pos[:-L]) / (L * dt))[rows].ravel()
        cols = [(vb[comp] if lab.startswith("vb") else F[lab])[rows].ravel() for lab in DRIFT[comp]]
        A = np.column_stack(cols + [np.ones(len(inc))])
        if weights is None:
            coef = np.linalg.lstsq(A, inc, rcond=None)[0]
        else:
            sw = np.sqrt(weights)
            coef = np.linalg.lstsq(A * sw[:, None], inc * sw, rcond=None)[0]
        fhat = A @ coef
        out[comp] = dict(coef=dict(zip(DRIFT[comp] + ["const"], coef)), inc=inc, fhat=fhat, e=inc - fhat)
    return out


def analyse(path):
    tag = swarm_meta(path)["tag"]
    t, x, y, dt, F = state(path)
    T, N = x.shape
    Lmax = max(LAGS)
    rows = np.arange(Lmax, T - Lmax)                      # same rows for every lag
    print(f"\n######## {tag}: {N} robots, rows {len(rows)} per robot")

    # 3. noise floor vs unexplained drift, over lags
    pts = []
    for L in LAGS:
        d = drift_fit(x, y, dt, F, L, rows)
        e2 = 0.5 * (np.mean(d["x"]["e"] ** 2) + np.mean(d["y"]["e"] ** 2))
        vf = 0.5 * (np.var(d["x"]["fhat"]) + np.var(d["y"]["fhat"]))
        vt = 0.5 * (np.var(d["x"]["inc"]) + np.var(d["y"]["inc"]))
        pts.append((L * dt, e2, vf, vt))
    s = np.array([p[0] for p in pts]); e2 = np.array([p[1] for p in pts])
    A = np.column_stack([1 / s, np.ones_like(s)])
    (a, U), *_ = np.linalg.lstsq(A, e2, rcond=None)
    D_avg = a / 2
    print(f"  residual variance vs lag: " + ", ".join(f"{si:.2f}s {ei:.0f}" for si, ei in zip(s, e2))
          + f"  ->  E[e^2] = 2*{D_avg:.0f}/s + {U:.1f}   (D = {D_avg:.0f} px^2/s, unexplained drift U = {U:.1f} px^2/s^2)")
    for (si, ei, vf, vt) in pts:
        noise = 2 * D_avg / si
        expl = vf / (vf + max(U, 0.0))
        print(f"    lag {si:.2f}s: total var {vt:7.0f} = drift {vf:6.0f} + residual {ei:7.0f} "
              f"(noise floor {noise:7.0f}); raw R2 {vf / vt:.3f}, R2 of the explainable part {expl:.3f}")

    # 2. diffusion law from the quadratic-variation token at the main lag
    d = drift_fit(x, y, dt, F, LAG, rows)
    # per component Var(e) = 2 D / (L dt), so D_hat = e^2 (L dt) / 2, averaged over x and y
    q = 0.25 * (d["x"]["e"] ** 2 + d["y"]["e"] ** 2) * (LAG * dt)
    vbx = np.zeros_like(x); vbx[LAG:] = (x[LAG:] - x[:-LAG]) / (LAG * dt)
    vby = np.zeros_like(y); vby[LAG:] = (y[LAG:] - y[:-LAG]) / (LAG * dt)
    flat = lambda arr: np.ascontiguousarray(arr[rows].T).ravel()          # noqa: E731  robot-major
    q_rm = np.ascontiguousarray(q.reshape(len(rows), N).T).ravel()
    atoms = [(lab, flat(F[lab])) for lab in ("n_contact", "n_same", "wall", "push_mag", "rR", "w_abs")]
    atoms.append(("vb_speed", flat(np.hypot(vbx, vby))))
    st = max(1, len(rows) // RW.LATTICE)
    lat = (np.arange(N)[:, None] * len(rows) + np.arange(0, len(rows), st)[None, :]).ravel()
    eq = dict(target="D_hat", lib=RW.Library(atoms, lat), y=q_rm, grid=(N, len(rows)), lattice=lat)
    cands = RW.candidates(eq)
    RW.show(eq, cands, tag)
    fr, se = RW.front(cands)
    law = cands[se]
    print(f"  mean D_hat {q.mean():.0f} px^2/s (lag-fit D {D_avg:.0f})")

    # 4. heteroscedastic refit with weights 1 / D_hat(state)
    lib = eq["lib"]
    names = list(law["terms"])
    Dpred = sum(v * lib.column(lib.names.index(k)) for k, v in law["terms"].items())
    Dpred = Dpred + (q_rm - sum(v * lib.column(lib.names.index(k)) for k, v in law["terms"].items())).mean()
    Dpred_tm = np.clip(Dpred.reshape(N, len(rows)).T.ravel(), 0.1 * q.mean(), None)   # back to time-major
    dw = drift_fit(x, y, dt, F, LAG, rows, weights=1 / Dpred_tm)
    for comp in ("x", "y"):
        u, w_ = d[comp]["coef"], dw[comp]["coef"]
        print(f"  drift {comp}: OLS " + ", ".join(f"{k} {v:+.2f}" for k, v in u.items())
              + "\n           WLS " + ", ".join(f"{k} {v:+.2f}" for k, v in w_.items()))
    return dict(tag=tag, lag_fit=dict(D=float(D_avg), U=float(U), points=pts),
                diffusion_front=[cands[i] for i in fr], diffusion_law=law,
                drift_ols={c: d[c]["coef"] for c in "xy"}, drift_wls={c: dw[c]["coef"] for c in "xy"})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="all")
    a = ap.parse_args()
    res = []
    for f in swarm_files():
        if a.file in ("all", swarm_meta(f)["tag"]):
            res.append(analyse(f))
            json.dump(res, open(os.path.join(RW.RESULTS, "diffusion.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
