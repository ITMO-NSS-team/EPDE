r"""Every law per data file, printed the way EPDE prints a system (search.equations()).

Per robot i of a recording (each robot its own system, its own spin and propulsion tokens):
    dc/dx0 = -w_i s,   ds/dx0 = w_i c                                 (central derivatives)
    dx/dx0 = a_i c - b_i s + mu_P push_x + mu_W wall_x + a vb_x + c_x   (forward increment, 0.4 s)
    dy/dx0 = b_i c + a_i s + mu_P push_y + mu_W wall_y + a vb_y + c_y
a_i + i b_i = v0_i e^{i theta_i} is the robot's OWN propulsion (speed v0_i, direction theta_i from the
marker heading); mu_P, mu_W, a, c_x, c_y are shared by the robots of the recording (joint least squares).
Held out (first half -> second half) this per-robot propulsion raises R2 0.17-0.27 -> 0.31-0.63; per-robot
push / wall / persistence / constants add < 0.03 (scratchpad per_robot_ablation.py).
Velocity level (0.08 s windows, d^2x/dx0^2 = dv_x/dt) and the noise law (D = squared increment / 2 dt)
are per recording, read from results/wide_friction.json and results/diffusion.json (exact
(WAPE, win14) fronts, stable end = least window-unstable member).
Objective function values: [WAPE of every equation ..., window instability win14 of every equation ...].

python epde_style.py      -> results/epde_style_per_file.txt (+ printed)
"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import json
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import subset_search as SUB             # noqa: E402  central-difference crop, window-refit instability
from robot_discovery import load_track                                     # noqa: E402
from swarm_data import (swarm_files, swarm_meta, load_swarm, arena_envelope,  # noqa: E402
                        spin_rates, contact_features)

LAG, BW, LATTICE = 10, 10, 260
RESULTS = os.path.join(HERE, "results")
SHARED = ["mu_P", "mu_W", "a", "c_x", "c_y"]
NAME = {"cos_phi": "c", "sin_phi": "s", "v_x": "dx/dx0", "v_y": "dy/dx0", "X": "x", "Y": "y",
        "w_abs": "|w|", "vb_speed": "|vb|", "c_t": "dc/dx0", "s_t": "ds/dx0", "c_tt": "d^2c/dx0^2",
        "s_tt": "d^2s/dx0^2", "rR": "r/R", "a_x": "d^2x/dx0^2", "a_y": "d^2y/dx0^2", "D_hat": "D",
        "fwd_x": "dx/dx0", "fwd_y": "dy/dx0"}


# ------------------------------------------------------------------ EPDE text form
def num(v):
    return str(float(f"{v:.4g}"))


def factor(f):
    base, _, p = f.partition("^")
    return f"{NAME.get(base, base)}{{power: {float(p or 1):.1f}}}"


def term(name):
    return " * ".join(factor(f) for f in name.split("*"))


def eq_text(terms, const, target):
    body = "".join(f"{num(v)} * {term(k)} + " for k, v in terms.items())
    return body + f"{num(const)} = {factor(target)}"


def system_text(eqs):
    if len(eqs) == 1:
        return eqs[0] + "\n"
    marks = [" / "] + [" | "] * (len(eqs) - 2) + [" \\ "]
    return "".join(m + e + "\n" for m, e in zip(marks, eqs))


def objectives(wapes, instabs):
    return f" , with objective function values of [{' '.join(f'{v:.4g}' for v in list(wapes) + list(instabs))}]"


# ------------------------------------------------------------------ scores
def wape(y, pred):
    return float(np.sum(np.abs(y - pred)) / np.sum(np.abs(y)))


def win14(A, y):
    keep = np.linalg.norm(A, axis=0) > 0
    return float(SUB.instab(A[:, keep], y, (len(y),), **SUB.STATS["win14"])[0])


def spin_law(t, phi):
    """Rotation law per robot on the central-difference crop, OLS on the pipeline lattice."""
    vals, _, grid = SUB._ode(t, {"c": np.cos(phi), "s": np.sin(phi)}, BW)
    lat = np.arange(0, grid[0], max(1, grid[0] // LATTICE))
    out = []
    for tgt, atom in (("c_t", "s"), ("s_t", "c")):
        A, y = vals[atom][:, None], np.asarray(vals[tgt])
        k = float(np.linalg.lstsq(A[lat], y[lat], rcond=None)[0][0])
        out.append((tgt, atom, k, wape(y, A[:, 0] * k), win14(A, y)))
    return out


# ------------------------------------------------------------------ drift with per-robot propulsion
def drift_design(c, s, px, py, wx, wy, vbx, vby):
    """Rows: x equation then y equation. Columns: own (a_i, b_i), shared (mu_P, mu_W, a, c_x, c_y)."""
    n = len(c)
    one, zero = np.ones(n), np.zeros(n)
    Ax = np.column_stack([c, -s, px, wx, vbx, one, zero])
    Ay = np.column_stack([s, c, py, wy, vby, zero, one])
    return np.vstack([Ax, Ay])


def joint_drift(robots):
    """robots: list of dicts with the drift columns and z = (fwd_x, fwd_y). One least squares, Gram
    accumulated robot by robot. Returns shared coefficients and per-robot (a_i, b_i)."""
    R = len(robots)
    p = len(SHARED) + 2 * R
    G, g = np.zeros((p, p)), np.zeros(p)
    for r, d in enumerate(robots):
        A = drift_design(*(d[k] for k in ("c", "s", "px", "py", "wx", "wy", "vbx", "vby")))
        y = np.concatenate([d["fx"], d["fy"]])
        idx = [len(SHARED) + 2 * r, len(SHARED) + 2 * r + 1] + list(range(len(SHARED)))
        G[np.ix_(idx, idx)] += A.T @ A
        g[idx] += A.T @ y
    G += 1e-9 * np.trace(G) / p * np.eye(p)                  # lone robot: push / wall columns are zero
    coef = np.linalg.solve(G, g)
    shared = dict(zip(SHARED, coef[:len(SHARED)]))
    own = coef[len(SHARED):].reshape(R, 2)
    return shared, own


def robot_drift_eqs(d, ai, bi, sh):
    """EPDE text + scores of one robot's x and y drift equations."""
    lines, wps, ins = [], [], []
    for comp, (cc, sc), P, W, vb, cst, y in (
            ("x", (ai, -bi), d["px"], d["wx"], d["vbx"], sh["c_x"], d["fx"]),
            ("y", (bi, ai), d["py"], d["wy"], d["vby"], sh["c_y"], d["fy"])):
        terms = {"c": cc, "s": sc}
        cols = [d["c"], d["s"]]
        if np.any(P):
            terms[f"push_{comp}"] = sh["mu_P"]; cols.append(P)
        if np.any(W):
            terms[f"wall_{comp}"] = sh["mu_W"]; cols.append(W)
        terms[f"vb_{comp}"] = sh["a"]; cols.append(vb)
        A = np.column_stack(cols)
        pred = A @ np.array(list(terms.values())) + cst
        lines.append(eq_text(terms, cst, f"fwd_{comp}"))
        wps.append(wape(y, pred))
        ins.append(win14(np.column_stack([A, np.ones(len(y))]), y))
    return lines, wps, ins


def swarm_robots(path):
    t, x, y, phi, ids = load_swarm(path)
    dt = t[1] - t[0]
    xc, yc, r_env = arena_envelope(x, y)
    w = spin_rates(t, phi)
    F = contact_features(x, y, w, xc, yc, r_env)
    rows = np.arange(LAG, len(t) - LAG)
    out = []
    for k, rid in enumerate(ids):
        out.append(dict(
            id=int(rid), t=t, phi=phi[:, k], w=float(w[k]),
            c=np.cos(phi[rows, k]), s=np.sin(phi[rows, k]),
            px=F["push_x"][rows, k], py=F["push_y"][rows, k], wx=F["wall_x"][rows, k], wy=F["wall_y"][rows, k],
            vbx=(x[rows, k] - x[rows - LAG, k]) / (LAG * dt), vby=(y[rows, k] - y[rows - LAG, k]) / (LAG * dt),
            fx=(x[rows + LAG, k] - x[rows, k]) / (LAG * dt), fy=(y[rows + LAG, k] - y[rows, k]) / (LAG * dt)))
    return out


def single_robot():
    t, x, y, phi = load_track()
    dt = t[1] - t[0]
    rows = np.arange(LAG, len(t) - LAG)
    z = np.zeros(len(rows))
    return [dict(id=1, t=t, phi=phi, w=float(np.polyfit(t, phi, 1)[0]),
                 c=np.cos(phi[rows]), s=np.sin(phi[rows]), px=z, py=z, wx=z, wy=z,
                 vbx=(x[rows] - x[rows - LAG]) / (LAG * dt), vby=(y[rows] - y[rows - LAG]) / (LAG * dt),
                 fx=(x[rows + LAG] - x[rows]) / (LAG * dt), fy=(y[rows + LAG] - y[rows]) / (LAG * dt))]


# ------------------------------------------------------------------ per-recording fronts from results/
def front(cands, stat="win14"):
    idx = [i for i, c in enumerate(cands) if np.isfinite(c["wape"]) and np.isfinite(c[stat])]
    fr = [i for i in idx if not any(cands[j]["wape"] <= cands[i]["wape"] and cands[j][stat] <= cands[i][stat]
                                    and (cands[j]["wape"] < cands[i]["wape"] or cands[j][stat] < cands[i][stat])
                                    for j in idx)]
    return sorted(fr, key=lambda i: cands[i]["c"]), min(fr, key=lambda i: cands[i][stat])


def front_block(title, cands):
    fr, se = front(cands)
    s = f"  {title}: 0-th non-dominated level ({len(fr)} equations; * = stable end)\n"
    for i in fr:
        c = cands[i]
        s += (f"   {'*' if i == se else ' '} {eq_text(c['terms'], 0.0, c['target'])}"
              f"{objectives([c['wape']], [c['win14']])}\n")
    return s


# ------------------------------------------------------------------ main
def recording_block(tag, robots, fric, diff):
    sh, own = joint_drift(robots)
    v0 = np.hypot(own[:, 0], own[:, 1])
    th = np.degrees(np.arctan2(own[:, 1], own[:, 0]))
    s = (f"\n{'=' * 100}\n{tag}: {len(robots)} robot(s); shared drift coefficients "
         + ", ".join(f"{k} {num(v)}" for k, v in sh.items()) + "\n"
         + f"  own propulsion v0_i median {np.median(v0):.1f} px/s [{v0.min():.1f}, {v0.max():.1f}], "
           f"theta_i median {np.median(th):+.1f} deg [{th.min():+.1f}, {th.max():+.1f}]\n")
    for r, d in enumerate(robots):
        spin = spin_law(d["t"], d["phi"])
        dl, dw, di = robot_drift_eqs(d, own[r, 0], own[r, 1], sh)
        eqs = [eq_text({atom: k}, 0.0, tgt) for tgt, atom, k, _, _ in spin] + dl
        s += (f"\n robot {d['id']}: w = {d['w']:+.3f} rad/s, v0 = {v0[r]:.1f} px/s, theta = {th[r]:+.1f} deg\n"
              + system_text(eqs).rstrip("\n")
              + objectives([x[3] for x in spin] + dw, [x[4] for x in spin] + di) + "\n")
    key = "single" if tag.startswith("N1") else tag
    s += f"\n velocity level (0.08 s windows), recording-wide:\n"
    for comp in ("x", "y"):
        s += front_block(f"d^2{comp}/dx0^2", fric[key]["results"][comp]["candidates"])
    if key in diff:
        s += " noise (D per axis, px^2/s), recording-wide:\n" + front_block("D", diff[key]["diffusion_front"])
    return s


def main():
    fric = json.load(open(os.path.join(RESULTS, "wide_friction.json")))
    diff = {e["tag"]: e for e in json.load(open(os.path.join(RESULTS, "diffusion.json")))}
    out = os.path.join(RESULTS, "epde_style_per_file.txt")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(__doc__ + "\n")
        sets = [("N1_PWM10 (lone robot)", single_robot)] + \
               [(swarm_meta(f)["tag"], lambda f=f: swarm_robots(f)) for f in swarm_files()]
        for tag, get in sets:
            block = recording_block(tag, get(), fric, diff)
            fh.write(block)
            fh.flush()
            print(block, flush=True)
    print(f"\nwritten {out}")


if __name__ == "__main__":
    main()
