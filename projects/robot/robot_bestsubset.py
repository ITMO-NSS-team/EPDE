r"""The general-library best-subset pipeline (the SINDy-comparison "new pipeline") on the robots.

Pipeline, unchanged, from subset_search.py (candidates / front):
  candidates  the exact best support per symbolic complexity c = 1..5 (atom count; search,
              exhaustive via the unit-norm Gram while a level has <= 5e7 supports), OLS on a
              strided fitting lattice
  scoring     WAPE on the full inner crop; window-refit instability (instab) at
              win14 (1/4 of each axis, 10 positions, relative ridge) and harness (1/2, abs ridge)
  selection   the exact (WAPE, instability) front

Robot problems (no truth is assumed; the front is reported):
  single   the N=1 track: fields x, y (centred px), c = cos(phi), s = sin(phi); central differences
           (np.gradient, as the protocol), targets x_t, y_t, c_t, s_t; library = subset_search.library
           (field powers 1-3, field time derivatives, t atoms, products of <= 2 atoms)
  drift    swarm forward increments (x(t+L)-x(t))/(L dt) of 10 robots pooled (grid robots x time);
           atoms push_*, wall_*, gear_* (swarm_data.contact_features) and centred x, y; products
           of <= 2 atoms. Central differences cannot see these gradient-type drifts (see
           swarm_drift_epde.py), hence the forward target.

python robot_bestsubset.py --problem single|drift [--file N40_PWM20_D41 --robots 10 --lag 10]
"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")      # machine rule of the study: one process, BLAS <= 4

import argparse
import itertools
import json
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import subset_search as SUB             # noqa: E402  library, candidates, exact front
from robot_discovery import load_track                                    # noqa: E402
from swarm_data import (swarm_files, swarm_meta, load_swarm, arena_envelope,  # noqa: E402
                        spin_rates, contact_features)

BW = 10           # boundary crop of the central-difference protocol (subset_search._ode)
LATTICE = 260     # fitting rows on an ODE, as the protocol


def products(atoms):
    """subset_search.library's product step for an explicit atom list [(key, exponent, array)]."""
    cols = {}
    for k in (1, 2):
        for combo in itertools.combinations_with_replacement(range(len(atoms)), k):
            expo = {}
            for i in combo:
                key, e, _a = atoms[i]
                expo[key] = expo.get(key, 0) + e
            name = SUB.name_of(expo)
            if name == "" or name in cols:
                continue
            arr = np.ones_like(atoms[0][2])
            for i in combo:
                arr = arr * atoms[i][2]
            if np.all(np.isfinite(arr)) and np.linalg.norm(arr) > 0:
                cols[name] = (k, arr)
    names = sorted(cols, key=lambda n: (cols[n][0], n))
    return names, np.column_stack([cols[n][1] for n in names]), np.array([cols[n][0] for n in names])


def single_equations():
    t, x, y, phi = load_track()
    fields = {"x": x - x.mean(), "y": y - y.mean(), "c": np.cos(phi), "s": np.sin(phi)}
    vals, coords, grid = SUB._ode(t, fields, BW)
    lat = np.arange(0, grid[0], max(1, grid[0] // LATTICE))
    for target in ("c_t", "s_t", "x_t", "y_t"):
        names, X, ks = SUB.library(vals, coords, list(fields), target, False)
        yield dict(target=target, truth={}, names=names, X=X, ks=ks,
                   y=np.asarray(vals[target], float), grid=grid, lattice=lat)


def drift_equations(file_key, robots, lag):
    path = next(p for p in swarm_files() if swarm_meta(p)["tag"] == file_key)
    t, x, y, phi, ids = load_swarm(path)
    dt = t[1] - t[0]
    xc, yc, r_env = arena_envelope(x, y)
    F = contact_features(x, y, spin_rates(t, phi), xc, yc, r_env)
    if not robots:
        robots = x.shape[1]                              # 0 = every robot of the recording
    chosen = np.linspace(0, x.shape[1] - 1, robots).astype(int)
    T_in = x.shape[0] - lag                             # rows with a forward increment
    grid = (robots, T_in)                               # robot-major, like a (sample, time) grid
    flat = lambda a: np.ascontiguousarray(a[:T_in][:, chosen].T).ravel()   # noqa: E731
    atoms = [(("pow", lab), 1, flat(F[lab])) for lab in
             ("push_x", "push_y", "wall_x", "wall_y", "gear_x", "gear_y")]
    atoms += [(("pow", "X"), 1, flat(x - xc)), (("pow", "Y"), 1, flat(y - yc))]
    names, X, ks = products(atoms)
    # fitting lattice: every robot, strided in time (~LATTICE rows per robot)
    st = max(1, T_in // LATTICE)
    lat = (np.arange(robots)[:, None] * T_in + np.arange(0, T_in, st)[None, :]).ravel()
    for comp, pos in (("x", x), ("y", y)):
        inc = (pos[lag:] - pos[:-lag]) / (lag * dt)
        yield dict(target=f"fwd_{comp}", truth={}, names=names, X=X, ks=ks,
                   y=np.ascontiguousarray(inc[:, chosen].T).ravel(), grid=grid, lattice=lat)


def spin_equations(path):
    """Per robot of one recording: fields c, s (no x, y), targets c_t, s_t."""
    t, x, y, phi, ids = load_swarm(path)
    for k, rid in enumerate(ids):
        fields = {"c": np.cos(phi[:, k]), "s": np.sin(phi[:, k])}
        vals, coords, grid = SUB._ode(t, fields, BW)
        lat = np.arange(0, grid[0], max(1, grid[0] // LATTICE))
        w_phase = float(np.polyfit(t, phi[:, k], 1)[0])
        for target in ("c_t", "s_t"):
            names, X, ks = SUB.library(vals, coords, list(fields), target, False)
            yield dict(target=target, truth={}, names=names, X=X, ks=ks, robot_id=int(rid),
                       omega_phase=w_phase, y=np.asarray(vals[target], float), grid=grid,
                       lattice=lat)


def stable_end(cands, stat):
    """The front's least-unstable member (the front is sorted by neither axis)."""
    fr = SUB.front(cands, list(range(len(cands))), "wape", stat)
    return min(fr, key=lambda i: cands[i][stat]), fr


def run_spin(files, stat="win14"):
    import csv
    path = os.path.join(HERE, "results", "bestsubset_spin.csv")
    fields = ["tag", "robot_id", "target", "omega_phase", "stable_terms", "stable_c", "stable_wape",
              "stable_instab", "rotation_law", "omega", "front"]
    rows = []
    for f in files:
        tag = swarm_meta(f)["tag"]
        for eq in spin_equations(f):
            cands, _ = SUB.candidates(eq, "open")
            i, fr = stable_end(cands, stat)
            c = cands[i]
            partner = "s" if eq["target"] == "c_t" else "c"
            rot = list(c["terms"]) == [partner]
            w = (-1 if eq["target"] == "c_t" else 1) * c["terms"][partner] if rot else np.nan
            rows.append(dict(tag=tag, robot_id=eq["robot_id"], target=eq["target"],
                             omega_phase=eq["omega_phase"], stable_terms=" + ".join(c["terms"]),
                             stable_c=c["c"], stable_wape=c["wape"], stable_instab=c[stat],
                             rotation_law=rot, omega=w,
                             front=" | ".join(" + ".join(cands[j]["terms"]) for j in fr)))
            r = rows[-1]
            print(f"{tag} id {r['robot_id']:2d} {r['target']}: stable end '{r['stable_terms']}' "
                  f"(c={r['stable_c']}, WAPE {r['stable_wape']:.3f})  omega {r['omega']:+.3f} "
                  f"vs phase {r['omega_phase']:+.3f}", flush=True)
        with open(path, "w", newline="", encoding="utf-8") as fh:
            wr = csv.DictWriter(fh, fieldnames=fields)
            wr.writeheader()
            wr.writerows(rows)
    import pandas as pd
    df = pd.DataFrame(rows)
    print(f"\n{len(df)} equations ({df.robot_id.nunique()} robot ids, {df.tag.nunique()} recordings); "
          f"stable end = 1-term rotation law in {df.rotation_law.sum()}/{len(df)}")
    ok = df[df.rotation_law]
    rel = (ok.omega - ok.omega_phase) / ok.omega_phase.abs()
    print(f"omega vs phase slope (rotation-law rows): median {rel.median():+.4f}, "
          f"IQR [{rel.quantile(.25):+.4f}, {rel.quantile(.75):+.4f}]")
    bad = df[~df.rotation_law]
    if len(bad):
        print("stable end NOT the rotation law:")
        print(bad[["tag", "robot_id", "target", "omega_phase", "stable_terms", "stable_wape"]]
              .to_string(index=False))
    print("saved", path)


def run_drift(files, robots, lag):
    out = {}
    path = os.path.join(HERE, "results", "bestsubset_drift_all.json")
    for f in files:
        tag = swarm_meta(f)["tag"]
        for eq in drift_equations(tag, robots, lag):
            cands, info = SUB.candidates(eq, "open")
            report(eq, cands)
            out[f"{tag}|{eq['target']}"] = dict(info=info, grid=eq["grid"], candidates=cands)
            json.dump(out, open(path, "w", encoding="utf-8"), indent=1, default=float)
    print("\nsummary (stable end of the win14 front | the c=1 best for comparison):")
    for key, rec in out.items():
        cands = rec["candidates"]
        i, fr = stable_end(cands, "win14")
        c, one = cands[i], next(cc for cc in cands if cc["c"] == 1)
        terms = " + ".join(f"{v:.1f}*{k}" for k, v in c["terms"].items())
        print(f"  {key:22s} {terms:40s} WAPE {c['wape']:.4f} win14 {c['win14']:.3e} | "
              f"c=1 {' + '.join(f'{v:.1f}*{k}' for k, v in one['terms'].items())} "
              f"WAPE {one['wape']:.4f} win14 {one['win14']:.3e}; front size {len(fr)}")
    print("saved", path)


def report(eq, cands):
    print(f"\n=== target {eq['target']}: {len(eq['names'])} library columns, rows {len(eq['y'])} "
          f"(lattice {len(eq['lattice'])}), grid {eq['grid']}")
    for c in cands:
        terms = " + ".join(f"{v:+.4g}*{k}" for k, v in c["terms"].items())
        print(f"   c={c['c']} [{c['mode'][0]}] WAPE {c['wape']:.4f}  win14 {c['win14']:.3e}  "
              f"harness {c['harness']:.3e}  | {terms}")
    idx = list(range(len(cands)))
    for stat in SUB.STATS:
        fr = SUB.front(cands, idx, "wape", stat)
        print(f"   front ({stat}): " + "; ".join(f"c={cands[i]['c']} " + " + ".join(cands[i]["terms"])
                                               for i in fr))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", choices=["single", "drift", "spin"], default="single")
    ap.add_argument("--file", default="N40_PWM20_D41", help="recording tag, or 'all'")
    ap.add_argument("--robots", type=int, default=10, help="robots pooled for drift; 0 = all")
    ap.add_argument("--lag", type=int, default=10)
    a = ap.parse_args()
    files = (swarm_files() if a.file == "all"
             else [p for p in swarm_files() if swarm_meta(p)["tag"] == a.file])
    if a.problem == "spin":
        return run_spin(files)
    if a.problem == "drift" and (a.file == "all" or not a.robots):
        return run_drift(files, a.robots, a.lag)
    eqs = single_equations() if a.problem == "single" else drift_equations(a.file, a.robots, a.lag)
    out = {}
    for eq in eqs:
        cands, info = SUB.candidates(eq, "open")
        report(eq, cands)
        out[eq["target"]] = dict(info=info, candidates=cands)
    path = os.path.join(HERE, "results", f"bestsubset_{a.problem}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    json.dump(out, open(path, "w", encoding="utf-8"), indent=1, default=float)
    print("\nsaved", path)


if __name__ == "__main__":
    main()
