r"""One local interaction law, 1 -> 2 -> many robots (proxy protocol on the swarm recordings).

Per-robot, time-explicit state equations (each trajectory a braid strand):
    dr_i/dt   = SELF(i, t) + sum_{j touching i} PAIR(i, j, t) + noise          (forward increments, 0.4 s)
    dphi_i/dt = w_i        + sum_{j touching i} TORQUE(i, j, t) + noise
Pair frame of (i, j) at time t: d = |r_j - r_i|, u = unit vector from j to i (away from j),
t = z x (r_j - r_i)/d (sideways), a_i = phi_i - bond angle (cos a_i > 0: j is in FRONT of i),
a_j = phi_j - bond angle (cos a_j < 0: j heads TOWARD i), s_i, s_j = spin signs.

Vector features (lab frame; x and y rows stacked, so one isotropic coefficient per feature):
  self   e_i = e(phi_i) [propulsion], v_prev [persistence], W [wall push]
  pair   push      sum u                        excluded-volume push
         push_ov   sum u (1.1 sigma - d)/sigma  overlap-weighted push
         block     sum e_i                      each contact changes own propulsion
         block_fr  sum e_i cos a_i              ... depending on j in front / behind
         push_fr   sum u cos a_i                push depending on j in front / behind
         drag_j    sum e_j                      neighbour's heading carries i
         push_j    sum u cos a_j                push depending on j heading toward i
         side      sum t                        sideways, spin-independent
         gear      sum t (s_i + s_j)            sideways, contact rolling
         side_i    sum t s_i                    sideways, own spin only
Torque features (spin deviation dphi_i/dt - w_i): n, n_same, sum cos a_i, sum sin a_i, sum s_i s_j cos(a_i-a_j).

Protocol (fit on one recording, default N30):
  1. self law on robots with 0 contacts (all features available, pair ones are 0 there);
  2. pair law = correction on robots with exactly 1 contact, self law held fixed;
  3. superposition: robots with 2, 3, 4+ contacts predicted by self + summed pair law, NO refit,
     compared with a refit of the same features on those robots (the ceiling);
  4. size transfer: the N30 law applied unchanged to the N40 recordings.
Selection: exhaustive subsets (<= 6 features), 5-fold CV grouped by robot, the smallest subset whose
CV error is within one standard error of the best, the SE taken over PAIRED fold differences.

python local_law.py [--fit N30_PWM10_D41]
"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import argparse
import itertools
import json
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from swarm_data import (swarm_files, swarm_meta, load_swarm, arena_envelope, spin_rates,  # noqa: E402
                        ROBOT_DIAMETER_PX as SIGMA)
from scipy.signal import savgol_filter                                                     # noqa: E402

LAG, CONTACT, MAXK, FOLDS = 10, 1.1, 6, 5
SELF = ["e_i", "v_prev", "W"]
PAIR = ["push", "push_ov", "block", "block_fr", "push_fr", "drag_j", "push_j", "side", "gear", "side_i"]
TORQUE = ["n", "n_same", "cos_ai", "sin_ai", "align"]


def build(path):
    """Features at rows t = LAG .. T-LAG-1: dict name -> (T_in, N, 2) vectors / (T_in, N) scalars."""
    t, x, y, phi, ids = load_swarm(path)
    dt = t[1] - t[0]
    T, N = x.shape
    xc, yc, r_env = arena_envelope(x, y)
    w = spin_rates(t, phi)
    s = np.sign(w)
    rows = np.arange(LAG, T - LAG)
    Ti = len(rows)
    V = {k: np.zeros((Ti, N, 2)) for k in SELF + PAIR}
    S = {k: np.zeros((Ti, N)) for k in TORQUE}
    ncon = np.zeros((Ti, N), dtype=int)
    X, Y, PH = x[rows], y[rows], phi[rows]
    E = np.stack([np.cos(PH), np.sin(PH)], -1)                       # (Ti, N, 2)
    V["e_i"] = E
    V["v_prev"] = np.stack([(X - x[rows - LAG]), (Y - y[rows - LAG])], -1) / (LAG * dt)
    r = np.hypot(X - xc, Y - yc)
    wall = r > r_env - 0.25 * SIGMA
    V["W"] = np.stack([np.where(wall, -(X - xc) / r, 0), np.where(wall, -(Y - yc) / r, 0)], -1)
    for t0 in range(0, Ti, 1000):
        sl = slice(t0, min(Ti, t0 + 1000))
        dX = X[sl][:, None, :] - X[sl][:, :, None]                   # [t, i, j] = x_j - x_i
        dY = Y[sl][:, None, :] - Y[sl][:, :, None]
        D = np.hypot(dX, dY); D[:, np.arange(N), np.arange(N)] = np.inf
        C = D < CONTACT * SIGMA
        bx, by = np.where(C, dX / D, 0), np.where(C, dY / D, 0)       # unit i -> j (bond)
        ux, uy = -bx, -by                                             # away from j
        tx, ty = -by, bx                                              # z x bond
        ei = E[sl]                                                    # (t, N, 2) own heading
        ej = E[sl][:, None, :, :]                                     # (t, 1, N, 2) neighbour heading (j axis)
        cos_ai = ei[:, :, None, 0] * bx + ei[:, :, None, 1] * by      # (t, i, j)
        sin_ai = ei[:, :, None, 0] * by - ei[:, :, None, 1] * bx
        cos_aj = ej[..., 0] * bx + ej[..., 1] * by
        sin_aj = ej[..., 0] * by - ej[..., 1] * bx
        ov = np.where(C, (CONTACT * SIGMA - D) / SIGMA, 0)
        si, sj = s[None, :, None], s[None, None, :]
        Cf = C.astype(float)

        def vsum(wgt, vx, vy):
            return np.stack([(wgt * vx).sum(2), (wgt * vy).sum(2)], -1)

        V["push"][sl] = vsum(Cf, ux, uy)
        V["push_ov"][sl] = vsum(ov, ux, uy)
        n = Cf.sum(2)
        V["block"][sl] = ei * n[..., None]
        V["block_fr"][sl] = ei * (Cf * cos_ai).sum(2)[..., None]
        V["push_fr"][sl] = vsum(Cf * cos_ai, ux, uy)
        V["drag_j"][sl] = vsum(Cf, np.broadcast_to(ej[..., 0], C.shape), np.broadcast_to(ej[..., 1], C.shape))
        V["push_j"][sl] = vsum(Cf * cos_aj, ux, uy)
        V["side"][sl] = vsum(Cf, tx, ty)
        V["gear"][sl] = vsum(Cf * (si + sj), tx, ty)
        V["side_i"][sl] = vsum(Cf * si, tx, ty)
        S["n"][sl] = n
        S["n_same"][sl] = (Cf * (si == sj)).sum(2)
        S["cos_ai"][sl] = (Cf * cos_ai).sum(2)
        S["sin_ai"][sl] = (Cf * sin_ai).sum(2)
        S["align"][sl] = (Cf * si * sj * (cos_ai * cos_aj + sin_ai * sin_aj)).sum(2)
        ncon[sl] = n.astype(int)
    target = np.stack([x[rows + LAG] - X, y[rows + LAG] - Y], -1) / (LAG * dt)
    phid = savgol_filter(phi, 9, 3, deriv=1, delta=dt, axis=0)[rows]
    spin_dev = phid - w[None, :]
    return dict(V=V, S=S, n=ncon, target=target, spin_dev=spin_dev, N=N, w=w, tag=swarm_meta(path)["tag"])


def stack(D, names, mask, target=None):
    """Rows (t, i) where mask, x and y components stacked; returns A (rows, k), y, robot id per row."""
    ti, ii = np.nonzero(mask)
    A = np.concatenate([np.column_stack([D["V"][k][ti, ii, c] for k in names]) if names
                        else np.zeros((len(ti), 0)) for c in (0, 1)])
    yv = np.concatenate([(D["target"] if target is None else target)[ti, ii, c] for c in (0, 1)])
    grp = np.concatenate([ii, ii])
    return A, yv, grp


def cv_select(A, y, grp, names, maxk=MAXK, base=None):
    """Exhaustive subsets, 5-fold CV grouped by robot (Gram per fold), 1-SE rule. base: always-on cols."""
    rng = np.random.default_rng(0)
    robots = np.unique(grp)
    fold_of = dict(zip(robots, rng.permutation(len(robots)) % FOLDS))
    f = np.array([fold_of[g] for g in grp])
    A1 = np.column_stack([A, np.ones(len(y))])                       # intercept always in
    p = A.shape[1]
    Gs = [(A1[f != k].T @ A1[f != k], A1[f != k].T @ y[f != k]) for k in range(FOLDS)]
    tests = [(A1[f == k], y[f == k]) for k in range(FOLDS)]
    base = list(base or [])
    free = [j for j in range(p) if j not in base]
    results = []
    for k in range(0, maxk + 1):
        for comb in itertools.combinations(free, k):
            S = sorted(base + list(comb)) + [p]
            errs = []
            for (G, b), (At, yt) in zip(Gs, tests):
                c = np.linalg.solve(G[np.ix_(S, S)] + 1e-9 * np.eye(len(S)), b[S])
                errs.append(np.mean((yt - At[:, S] @ c) ** 2))
            results.append((np.mean(errs), np.array(errs), S))
    best = min(results, key=lambda r: r[0])
    # 1-SE rule on PAIRED fold differences: robots differ in noise level, so the raw fold spread
    # swamps model differences; a subset qualifies when it is not significantly worse than the best
    ok = []
    for r in results:
        diff = r[1] - best[1]
        if diff.mean() <= diff.std(ddof=1) / np.sqrt(FOLDS) + 1e-15:
            ok.append(r)
    pick = min(ok, key=lambda r: (len(r[2]), r[0]))
    S = pick[2]
    coef = np.linalg.lstsq(A1[:, S], y, rcond=None)[0]
    terms = {(names + ["const"])[j]: float(c) for j, c in zip(S, coef)}
    return terms, pick[0]


def predict(D, terms, mask, target=None):
    names = [k for k in terms if k != "const"]
    A, y, _ = stack(D, names, mask, target)
    pred = A @ np.array([terms[k] for k in names]) + terms.get("const", 0.0)
    return pred, y


def r2(pred, y):
    return 1 - np.mean((y - pred) ** 2) / np.var(y)


def fmt(terms):
    return " + ".join(f"{v:+.2f}*{k}" for k, v in terms.items())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", default="N30_PWM10_D41")
    a = ap.parse_args()
    data = {swarm_meta(f)["tag"]: build(f) for f in swarm_files()}
    D = data[a.fit]
    print(f"fit on {a.fit}: contact classes " +
          ", ".join(f"{k}: {100 * np.mean(np.minimum(D['n'], 4) == k):.1f}%" for k in range(5)))

    # 1. self law (0 contacts)
    m0 = D["n"] == 0
    A, y, g = stack(D, SELF, m0)
    self_law, cv = cv_select(A, y, g, SELF)
    p0, y0 = predict(D, self_law, m0)
    print(f"\n1. SELF (0 contacts, {m0.sum()} robot-moments): v = {fmt(self_law)}   R2 {r2(p0, y0):.3f}")

    # 2. pair correction (exactly 1 contact), self law fixed
    m1 = D["n"] == 1
    ps, _ = predict(D, self_law, m1)
    resid_target = D["target"].copy()
    ti, ii = np.nonzero(m1)
    A, y, g = stack(D, PAIR, m1)
    y = y - ps
    pair_law, cv = cv_select(A, y, g, PAIR)
    pair_law.pop("const", None)
    full = {**self_law, **pair_law}
    p1, y1 = predict(D, full, m1)
    ps1, _ = predict(D, self_law, m1)
    print(f"2. PAIR correction (1 contact, {m1.sum()} robot-moments): + {fmt(pair_law)}")
    print(f"   R2 on 1-contact robots: self only {r2(ps1, y1):.3f} -> self + pair {r2(p1, y1):.3f}")

    # 3. superposition (2, 3, 4+ contacts), no refit
    print("3. SUPERPOSITION (no refit) vs refit ceiling, on robots with more contacts:")
    for k in (2, 3, 4):
        mk = (D["n"] == k) if k < 4 else (D["n"] >= 4)
        if mk.sum() < 500:
            continue
        p, yk = predict(D, full, mk)
        names = [n for n in full if n != "const"]
        A, yy, _ = stack(D, names, mk)
        A1 = np.column_stack([A, np.ones(len(yy))])
        ceil = r2(A1 @ np.linalg.lstsq(A1, yy, rcond=None)[0], yy)
        print(f"   {k}{'+' if k == 4 else ''} contacts ({mk.sum()}): summed law R2 {r2(p, yk):.3f}  "
              f"| refit same terms {ceil:.3f}  | transfer efficiency {r2(p, yk) / ceil:.2f}")

    # 3b. crowding correction, learned ONLY on robots with exactly 3 contacts (the "5 robots" step):
    #     terms scaling with the contacts beyond two -- propulsion blocking saturates, the push is caged
    for Dd in data.values():
        ex = np.maximum(Dd["n"] - 2, 0)[..., None].astype(float)
        Dd["V"]["e_crowd"] = Dd["V"]["e_i"] * ex
        Dd["V"]["push_crowd"] = Dd["V"]["push"] * ex
        Dd["V"]["pushfr_crowd"] = Dd["V"]["push_fr"] * ex
    m3 = D["n"] == 3
    base3, _ = predict(D, full, m3)
    CROWD = ["e_crowd", "push_crowd", "pushfr_crowd"]
    A, y, g = stack(D, CROWD, m3)
    crowd, _ = cv_select(A, y - base3, g, CROWD, maxk=3)
    crowd.pop("const", None)
    law = {**full, **crowd}
    print(f"3b. CROWDING correction (learned on 3 contacts only): + {fmt(crowd)}")
    for k in (3, 4, 5):
        mk = (D["n"] == k) if k < 5 else (D["n"] >= 5)
        if mk.sum() < 300:
            continue
        p, yk = predict(D, law, mk)
        names = [n for n in law if n != "const"]
        A, yy, _ = stack(D, names, mk)
        A1 = np.column_stack([A, np.ones(len(yy))])
        ceil = r2(A1 @ np.linalg.lstsq(A1, yy, rcond=None)[0], yy)
        print(f"   {k}{'+' if k == 5 else ''} contacts ({mk.sum()}): law R2 {r2(p, yk):.3f} | refit ceiling {ceil:.3f}"
              f" | efficiency {r2(p, yk) / ceil:.2f}")

    # 4. size transfer, no refit -- pair-only law and pair + crowding law
    for label, lw in (("self + pair", full), ("self + pair + crowding", law)):
        print(f"4. TRANSFER of the {a.fit} {label} law, no refit (all robots, all moments):")
        for tag, Dt in data.items():
            mall = np.ones_like(Dt["n"], dtype=bool)
            p, yt = predict(Dt, lw, mall)
            names = [n for n in lw if n != "const"]
            A, yy, _ = stack(Dt, names, mall)
            A1 = np.column_stack([A, np.ones(len(yy))])
            own = r2(A1 @ np.linalg.lstsq(A1, yy, rcond=None)[0], yy)
            print(f"   {tag:15s} R2 transferred {r2(p, yt):.3f} | own refit {own:.3f} | efficiency {r2(p, yt) / own:.2f}")
    full = law

    # torque: spin deviation on 1-contact robots
    ti, ii = np.nonzero(m1)
    At = np.column_stack([D["S"][k][ti, ii] for k in TORQUE])
    yt = D["spin_dev"][ti, ii]
    torque, _ = cv_select(At, yt, ii, TORQUE, maxk=3)
    print(f"\nTORQUE (spin deviation, 1 contact): dphi/dt - w_i = {fmt(torque)}")

    json.dump(dict(fit=a.fit, self=self_law, pair=pair_law, crowd=crowd, torque=torque),
              open(os.path.join(HERE, "results", f"local_law_{a.fit}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
