r"""The best-subset pipeline with WIDE libraries: columns are products of up to 4 atoms.

Same pipeline as robot_bestsubset.py (subset_search.py: exact best support per symbolic
complexity c = 1..5, OLS on a strided fitting lattice, WAPE on the full crop, window-refit
instability win14 / harness via subset_search.instab, the exact (WAPE, instability) front via
subset_search.front), with two changes:

  * columns are products of up to MAX_ATOMS = 4 atoms (subset_search.search stops at 2), so the
    atoms are primitives -- powers come from products (c^3 = c*c*c costs 3, not 1);
  * the exhaustive level search is generalised to 1..4-atom columns (cost = atom count, levels
    capped at c = 5, so a 4-atom column can only pair with one 1-atom column). Only the lattice
    rows of every column are materialised; full-length columns are built for each level's winner.

Atoms (no truth is assumed anywhere):
  single  N=1 track. Central targets c_t, s_t, x_t, y_t: fields x, y, c, s and their t / tt
          derivatives (target excluded) + t atoms (t, 1/t, sin t, cos t, sin 2t, cos 2t).
          Forward targets fwd_x, fwd_y (lag 10): CAUSAL atoms only -- X, Y, c, s at t, the backward
          velocities vb_x, vb_y, t atoms (a central derivative would share the increment's future).
  spin    every robot of every recording, targets c_t, s_t: c, s, their derivatives (target
          excluded), t atoms, and context n_same, n_opp, wall, x_t, y_t of that robot.
  drift   per recording, all robots pooled, targets fwd_x, fwd_y (lag 10): push_x/y, wall_x/y,
          gear_x/y, X, Y, cos phi, sin phi, n_contact, vb_x, vb_y, r.

python robot_wide.py --problem single|spin|drift [--file all|<tag>]
"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")      # the study's machine rule: one process, BLAS <= 4

import argparse
import csv
import itertools
import json
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import subset_search as SUB             # noqa: E402  batched RSS, Gram, window-refit instability, front
from robot_discovery import load_track                                    # noqa: E402
from swarm_data import (swarm_files, swarm_meta, load_swarm, arena_envelope,  # noqa: E402
                        spin_rates, contact_features)

MAX_ATOMS, CMAX, BW, LATTICE, LAG = 4, 5, 10, 260, 10
RESULTS = os.path.join(HERE, "results")


# ------------------------------------------------------------------ library
class Library:
    """Columns = products of 1..max_atoms atoms (with repetition), named canonically; exact
    duplicates on the lattice (e.g. powers of a 0/1 indicator) keep the cheapest."""

    def __init__(self, atoms, lattice, max_atoms=MAX_ATOMS):
        self.atom_names = [a[0] for a in atoms]
        self.atoms = [np.asarray(a[1], float) for a in atoms]
        lat_atoms = [a[lattice] for a in self.atoms]
        combos, names, ks, cols, seen = [], [], [], [], set()
        for k in range(1, max_atoms + 1):
            for combo in itertools.combinations_with_replacement(range(len(atoms)), k):
                col = np.prod([lat_atoms[i] for i in combo], axis=0)
                nrm = np.linalg.norm(col)
                if not np.all(np.isfinite(col)) or nrm == 0:
                    continue
                key = np.round(col / nrm, 12).tobytes()
                if key in seen:
                    continue
                seen.add(key)
                combos.append(combo); names.append(self._name(combo)); ks.append(k); cols.append(col)
        self.combos, self.names, self.ks = combos, names, np.array(ks)
        self.Xl = np.column_stack(cols)

    def _name(self, combo):
        cnt = {}
        for i in combo:
            cnt[i] = cnt.get(i, 0) + 1
        return "*".join(self.atom_names[i] + (f"^{e}" if e > 1 else "") for i, e in sorted(cnt.items()))

    def column(self, j):
        return np.prod([self.atoms[i] for i in self.combos[j]], axis=0)

    def level_sizes(self, cmax=CMAX):
        from math import comb
        n = {k: int((self.ks == k).sum()) for k in range(1, MAX_ATOMS + 1)}
        return [sum(np.prod([comb(n[k], m) for k, m in comp.items()]) for comp in compositions(c))
                for c in range(1, cmax + 1)]


def compositions(c, maxk=MAX_ATOMS):
    """Ways to spend complexity c on columns of 1..maxk atoms: {atoms per column: how many}."""
    out = []
    for counts in itertools.product(*[range(c // k + 1) for k in range(1, maxk + 1)]):
        if sum(k * n for k, n in zip(range(1, maxk + 1), counts)) == c:
            out.append({k: n for k, n in zip(range(1, maxk + 1), counts) if n})
    return out


def level_supports(classes, comp, block=200000):
    """Every support of one composition, in blocks (vectorised over the largest factor)."""
    parts = [SUB._combos(classes[k], n) for k, n in comp.items()]
    if any(len(p) == 0 for p in parts):
        return
    parts.sort(key=len)
    big, others = parts[-1], parts[:-1]
    buf, size = [], 0
    for rows in itertools.product(*[range(len(p)) for p in others]):
        prefix = (np.concatenate([others[i][r] for i, r in enumerate(rows)]) if others
                  else np.empty(0, dtype=int))
        buf.append(np.hstack([np.repeat(prefix[None, :], len(big), axis=0), big]))
        size += len(big)
        if size >= block:
            yield np.vstack(buf)
            buf, size = [], 0
    if buf:
        yield np.vstack(buf)


def search(gm, ks, cmax=CMAX, B=50):
    classes = {k: np.flatnonzero(ks == k) for k in range(1, MAX_ATOMS + 1)}
    out = {}
    for c in range(1, cmax + 1):
        top, n = [], 0
        for comp in compositions(c):
            for sup in level_supports(classes, comp):
                top = SUB._merge_top(top, sup, SUB.batched_rss(gm, sup), B)
                n += len(sup)
        if top:
            out[c] = dict(best=list(top[0][1]), rss=top[0][0], top=top, n=n)
    return out


def candidates(eq):
    """subset_search.candidates with a lattice-only library and the generalised search."""
    lib, y, lat, grid = eq["lib"], eq["y"], eq["lattice"], eq["grid"]
    gm = SUB.Gram(lib.Xl, y[lat])
    out = []
    for c, r in search(gm, lib.ks).items():
        S = r["best"]
        coef = np.linalg.lstsq(lib.Xl[:, S], y[lat], rcond=None)[0]
        A = np.column_stack([lib.column(j) for j in S])
        cand = dict(target=eq["target"], c=int(c), n_supports=int(r["n"]),
                    terms={lib.names[j]: float(v) for j, v in zip(S, coef)}, J=0.0,
                    wape=float(np.sum(np.abs(y - A @ coef)) / np.sum(np.abs(y))))
        for key, kw in SUB.STATS.items():
            cand[key] = SUB.instab(A, y, grid, **kw)[0]
        out.append(cand)
    return out


def front(cands, stat="win14"):
    fr = SUB.front(cands, list(range(len(cands))), "wape", stat)
    return fr, min(fr, key=lambda i: cands[i][stat])


# ------------------------------------------------------------------ atoms
def t_atoms(tt):
    inv = np.divide(1.0, tt, out=np.zeros_like(tt), where=np.abs(tt) > 1e-12)
    return [("t", tt), ("1/t", inv), ("sin(t)", np.sin(tt)), ("cos(t)", np.cos(tt)),
            ("sin(2t)", np.sin(2 * tt)), ("cos(2t)", np.cos(2 * tt))]


def single_equations():
    t, x, y, phi = load_track()
    dt = t[1] - t[0]
    fields = {"x": x - x.mean(), "y": y - y.mean(), "c": np.cos(phi), "s": np.sin(phi)}
    vals, coords, grid = SUB._ode(t, fields, BW)
    lat = np.arange(0, grid[0], max(1, grid[0] // LATTICE))
    for target in ("c_t", "s_t", "x_t", "y_t"):
        atoms = [(v, vals[v]) for v in fields]
        atoms += [(d, vals[d]) for v in fields for d in (v + "_t", v + "_tt") if d != target]
        atoms += t_atoms(coords["t"])
        yield dict(target=target, lib=Library(atoms, lat), y=np.asarray(vals[target]),
                   grid=grid, lattice=lat)
    rows = np.arange(LAG, len(t) - LAG)                   # causal atoms, forward target
    lat_f = np.arange(0, len(rows), max(1, len(rows) // LATTICE))
    atoms = [(v, f[rows]) for v, f in fields.items()]
    atoms += [("vb_x", (x[rows] - x[rows - LAG]) / (LAG * dt)),
              ("vb_y", (y[rows] - y[rows - LAG]) / (LAG * dt))] + t_atoms(t[rows])
    lib = Library(atoms, lat_f)
    for comp, pos in (("x", x), ("y", y)):
        yield dict(target=f"fwd_{comp}", lib=lib, y=(pos[rows + LAG] - pos[rows]) / (LAG * dt),
                   grid=(len(rows),), lattice=lat_f)


def spin_equations(path):
    t, x, y, phi, ids = load_swarm(path)
    dt = t[1] - t[0]
    xc, yc, r_env = arena_envelope(x, y)
    F = contact_features(x, y, spin_rates(t, phi), xc, yc, r_env)
    wall = (np.abs(F["wall_x"]) + np.abs(F["wall_y"]) > 0).astype(float)
    xv, yv = np.gradient(x, dt, axis=0, edge_order=2), np.gradient(y, dt, axis=0, edge_order=2)
    crop = slice(BW, len(t) - BW)
    for k, rid in enumerate(ids):
        fields = {"c": np.cos(phi[:, k]), "s": np.sin(phi[:, k])}
        vals, coords, grid = SUB._ode(t, fields, BW)
        lat = np.arange(0, grid[0], max(1, grid[0] // LATTICE))
        context = [("n_same", F["n_same"][crop, k]), ("n_opp", F["n_opp"][crop, k]),
                   ("wall", wall[crop, k]), ("x_t", xv[crop, k]), ("y_t", yv[crop, k])]
        for target in ("c_t", "s_t"):
            atoms = [(v, vals[v]) for v in fields]
            atoms += [(d, vals[d]) for v in fields for d in (v + "_t", v + "_tt") if d != target]
            atoms += t_atoms(coords["t"]) + context
            yield dict(target=target, lib=Library(atoms, lat), y=np.asarray(vals[target]),
                       grid=grid, lattice=lat, robot_id=int(rid),
                       omega_phase=float(np.polyfit(t, phi[:, k], 1)[0]))


def drift_equations(path):
    t, x, y, phi, ids = load_swarm(path)
    dt = t[1] - t[0]
    xc, yc, r_env = arena_envelope(x, y)
    F = contact_features(x, y, spin_rates(t, phi), xc, yc, r_env)
    N = x.shape[1]
    rows = np.arange(LAG, len(t) - LAG)
    flat = lambda a: np.ascontiguousarray(a[rows].T).ravel()    # noqa: E731  robot-major
    X, Y = x - xc, y - yc
    vbx = np.zeros_like(x); vbx[LAG:] = (x[LAG:] - x[:-LAG]) / (LAG * dt)
    vby = np.zeros_like(y); vby[LAG:] = (y[LAG:] - y[:-LAG]) / (LAG * dt)
    atoms = [(lab, flat(F[lab])) for lab in ("push_x", "push_y", "wall_x", "wall_y", "gear_x", "gear_y",
                                             "n_contact")]
    atoms += [("X", flat(X)), ("Y", flat(Y)), ("cos_phi", flat(np.cos(phi))), ("sin_phi", flat(np.sin(phi))),
              ("vb_x", flat(vbx)), ("vb_y", flat(vby)), ("r", flat(np.hypot(X, Y)))]
    st = max(1, len(rows) // LATTICE)
    lat = (np.arange(N)[:, None] * len(rows) + np.arange(0, len(rows), st)[None, :]).ravel()
    lib = Library(atoms, lat)
    for comp, pos in (("x", x), ("y", y)):
        inc = np.zeros_like(pos); inc[:-LAG] = (pos[LAG:] - pos[:-LAG]) / (LAG * dt)
        yield dict(target=f"fwd_{comp}", lib=lib, y=flat(inc), grid=(N, len(rows)), lattice=lat)


# ------------------------------------------------------------------ runs
def show(eq, cands, label):
    lib = eq["lib"]
    print(f"\n=== {label} {eq['target']}: {len(lib.atom_names)} atoms -> {len(lib.names)} columns "
          f"(per atom count {[int((lib.ks == k).sum()) for k in range(1, MAX_ATOMS + 1)]}), "
          f"supports per c {lib.level_sizes()}", flush=True)
    for c in cands:
        terms = " + ".join(f"{v:+.4g}*{k}" for k, v in c["terms"].items())
        print(f"   c={c['c']} WAPE {c['wape']:.4f} win14 {c['win14']:.3e} harness {c['harness']:.3e} | {terms}")
    fr, se = front(cands)
    print("   win14 front: " + "; ".join(" + ".join(cands[i]["terms"]) for i in fr)
          + f"   -> stable end: {' + '.join(cands[se]['terms'])}", flush=True)


def strip(cands):
    return [{k: v for k, v in c.items()} for c in cands]


def run_single():
    out = {}
    for eq in single_equations():
        cands = candidates(eq)
        show(eq, cands, "single")
        out[eq["target"]] = strip(cands)
    json.dump(out, open(os.path.join(RESULTS, "wide_single.json"), "w"), indent=1, default=float)


def run_drift(files):
    out = {}
    for f in files:
        tag = swarm_meta(f)["tag"]
        for eq in drift_equations(f):
            t0 = time.time()
            cands = candidates(eq)
            show(eq, cands, f"{tag} ({time.time() - t0:.0f}s)")
            out[f"{tag}|{eq['target']}"] = strip(cands)
            json.dump(out, open(os.path.join(RESULTS, "wide_drift.json"), "w"), indent=1, default=float)


def run_spin(files, verbose=False):
    rows = []
    path = os.path.join(RESULTS, "wide_spin.csv")
    fields = ["tag", "robot_id", "target", "omega_phase", "stable_terms", "stable_c", "stable_wape",
              "stable_win14", "rotation_law", "omega", "front", "seconds"]
    for f in files:
        tag = swarm_meta(f)["tag"]
        for eq in spin_equations(f):
            t0 = time.time()
            cands = candidates(eq)
            if verbose:
                show(eq, cands, f"{tag} id {eq['robot_id']}")
            fr, se = front(cands)
            c = cands[se]
            partner = "s" if eq["target"] == "c_t" else "c"
            rot = list(c["terms"]) == [partner]
            rows.append(dict(tag=tag, robot_id=eq["robot_id"], target=eq["target"],
                             omega_phase=eq["omega_phase"], stable_terms=" + ".join(c["terms"]),
                             stable_c=c["c"], stable_wape=c["wape"], stable_win14=c["win14"],
                             rotation_law=rot,
                             omega=((-1 if eq["target"] == "c_t" else 1) * c["terms"][partner]) if rot else np.nan,
                             front=" | ".join(" + ".join(cands[j]["terms"]) for j in fr),
                             seconds=round(time.time() - t0, 1)))
            r = rows[-1]
            print(f"{tag} id {r['robot_id']:2d} {r['target']}: stable end '{r['stable_terms']}' (c={r['stable_c']}, "
                  f"WAPE {r['stable_wape']:.3f}) omega {r['omega']:+.3f} vs phase {r['omega_phase']:+.3f} "
                  f"[{r['seconds']}s]", flush=True)
            with open(path, "w", newline="", encoding="utf-8") as fh:
                wr = csv.DictWriter(fh, fieldnames=fields)
                wr.writeheader(); wr.writerows(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", choices=["single", "spin", "drift"], default="single")
    ap.add_argument("--file", default="all", help="recording tag, or 'all'")
    ap.add_argument("--limit", type=int, default=0, help="spin: only the first N robots of each file")
    ap.add_argument("--verbose", action="store_true")
    a = ap.parse_args()
    os.makedirs(RESULTS, exist_ok=True)
    files = (swarm_files() if a.file == "all"
             else [p for p in swarm_files() if swarm_meta(p)["tag"] == a.file])
    if a.problem == "single":
        run_single()
    elif a.problem == "drift":
        run_drift(files)
    else:
        if a.limit:
            global spin_equations
            base = spin_equations
            spin_equations = lambda p: itertools.islice(base(p), 2 * a.limit)   # noqa: E731
        run_spin(files, a.verbose)


if __name__ == "__main__":
    main()
