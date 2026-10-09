r"""The best-subset pipeline per robot, with each robot's own tokens (the per-robot EPDE runs' set
plus slow velocity memory), x and y as separate equations.

Pipeline (robot_wide.py / subset_search.py): exact best support per complexity c = 1..CMAX (OLS), WAPE on
all rows, window-refit instability win14 / harness, the exact (WAPE, win14) front and its stable
end (least unstable member). Library: single tokens (the per-robot EPDE runs use factors_num [1]).
Per robot the drift is noisy (R2 ~0.2-0.6), so every row is a fitting row (no 260-point lattice).

Target (forward increment, lag 10 = 0.4 s):  fwd_<v>(t) = (<v>(t + 0.4 s) - <v>(t)) / 0.4 s
Tokens at time t (causal):
    c, s                          own heading
    push_x, push_y, wall_x, wall_y (the ones the robot ever touches)
    vb_x, vb_y                    velocity over the last 0.4 s
    m1..m4 _x/_y                  velocity over EARLIER windows (slow memory): 0.4-0.8 s, 0.8-1.6 s,
                                  1.6-3.6 s, 3.6-9.6 s ago
    x or y                        own position, centred on the arena

    python robot_pipeline_per_robot.py            -> results/robot_pipeline.jsonl
    python robot_pipeline_per_robot.py --summary
"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "4")      # the study's machine rule: one process, BLAS <= 4

import argparse
import json
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import robot_wide as RW                                                    # noqa: E402
import subset_search as SUB                                                # noqa: E402
import swarm_robot_epde as SRE                                             # noqa: E402
from swarm_data import swarm_files, swarm_meta, load_swarm                # noqa: E402

LAG, CMAX = 10, 9
MEM = [(1, 2), (2, 4), (4, 9), (9, 24)]          # memory windows in 0.4 s steps back from t
HIST = MEM[-1][1]
RESULTS = os.path.join(HERE, "results", "robot_pipeline.jsonl")


def memory_tokens(pos, dt):
    """m_j(t) = (pos(t - a lag) - pos(t - b lag)) / ((b - a) lag dt), zero where undefined."""
    out = []
    for a, b in MEM:
        m = np.zeros_like(pos)
        idx = np.arange(b * LAG, len(pos))
        m[idx] = (pos[idx - a * LAG] - pos[idx - b * LAG]) / ((b - a) * LAG * dt)
        out.append(m)
    return out


def time_library(atoms, lat, partners):
    """Single tokens plus t x token for the physical tokens in ``partners`` (cost 2 each): a law whose
    coefficients drift linearly over the recording. ``t`` must be the last atom."""
    lib = RW.Library(atoms, lat, max_atoms=2)
    ti = lib.atom_names.index("t")
    keep = [j for j, cb in enumerate(lib.combos)
            if len(cb) == 1 or (len(cb) == 2 and ti in cb and cb[0] != cb[1]
                                and lib.atom_names[cb[0] if cb[1] == ti else cb[1]] in partners)]
    lib.combos = [lib.combos[j] for j in keep]
    lib.names = [lib.names[j] for j in keep]
    lib.ks, lib.Xl = lib.ks[keep], lib.Xl[:, keep]
    return lib


_DRIVE = {}


def neighbour_drive_of(tag):
    """Per recording, once: robot ids and the neighbour-drive tokens (rows aligned with ``equations``),
    the neighbours' propulsion read from the own-propulsion law fitted on that recording."""
    if tag not in _DRIVE:
        import swarm_sim as SS
        R = SS.recording(next(p for p in swarm_files() if swarm_meta(p)["tag"] == tag))
        assert SS.HIST == HIST and SS.LAG == LAG
        _DRIVE[tag] = (R["ids"], SS.neighbour_drive(R, SS.fit_joint(R, True, False)[0]), SS.NP_TOK)
    return _DRIVE[tag]


def equations(tag, rid, time_tokens=False, neighbour=False):
    t, x, y, phi, tokens = SRE.robot_data(tag, rid)
    dt = t[1] - t[0]
    rows = np.arange(HIST * LAG, len(t) - LAG)
    atoms = [(k, v[rows]) for fam in ("heading", "interaction", "memory") for k, v in tokens[fam].items()]
    for j, (mx, my) in enumerate(zip(memory_tokens(x, dt), memory_tokens(y, dt)), 1):
        atoms += [(f"m{j}_x", mx[rows]), (f"m{j}_y", my[rows])]
    if neighbour:
        ids, NP, names = neighbour_drive_of(tag)
        atoms += [(n, NP[:, ids.index(rid), k]) for k, n in enumerate(names)]
    lat = np.arange(len(rows))
    tn = (t[rows] - t[rows].mean()) / (t[rows][-1] - t[rows][0])          # -0.5 .. 0.5 over the recording
    for var, pos in (("x", x), ("y", y)):
        own = atoms + [(var, pos[rows])]
        if time_tokens:
            partners = {"c", "s", var} | {k for k, _ in own if not k.startswith("m")}
            lib = time_library(own + [("t", tn)], lat, partners)
        else:
            lib = RW.Library(own, lat, max_atoms=1)
        yield var, dict(target=f"fwd_{var}", lib=lib, y=(pos[rows + LAG] - pos[rows]) / (LAG * dt),
                        grid=(len(rows),), lattice=lat)


def interleaved_r2(A, y, n_blocks=10):
    """Fit on alternate contiguous blocks, test on the others (and swap): no extrapolation in time."""
    blk = np.arange(len(y)) * n_blocks // len(y) % 2
    sse = 0.0
    for k in (0, 1):
        tr, te = blk == k, blk != k
        coef = np.linalg.lstsq(A[tr], y[tr], rcond=None)[0]
        sse += np.sum((y[te] - A[te] @ coef) ** 2)
    return float(1 - sse / np.sum((y - y.mean()) ** 2))


def is_special(name, time_tokens, neighbour):
    """The tokens a --time / --neighbour run is testing."""
    return (time_tokens and "t" in name.split("*")) or (neighbour and name.startswith("np_"))


def run(tag, rid, time_tokens=False, neighbour=False):
    out = []
    for var, eq in equations(tag, rid, time_tokens, neighbour):
        t0 = time.time()
        RW.CMAX = CMAX
        cands = []
        gm = SUB.Gram(eq["lib"].Xl, eq["y"])
        for c, r in RW.search(gm, eq["lib"].ks, cmax=CMAX).items():
            S = r["best"]
            A = eq["lib"].Xl[:, S]
            coef = np.linalg.lstsq(A, eq["y"], rcond=None)[0]
            cand = dict(c=int(c), terms={eq["lib"].names[j]: float(v) for j, v in zip(S, coef)},
                        wape=float(np.sum(np.abs(eq["y"] - A @ coef)) / np.sum(np.abs(eq["y"]))))
            for key, kw in SUB.STATS.items():
                cand[key] = float(SUB.instab(A, eq["y"], eq["grid"], **kw)[0])
            cands.append(cand)
        fr, se = RW.front(cands)
        row = dict(tag=tag, robot_id=rid, var=var, cands=cands, front=[int(i) for i in fr], stable=int(se),
                   seconds=round(time.time() - t0, 1))
        if time_tokens or neighbour:
            # does the most complete law need its tested terms? same structure with and without them
            full = cands[min(fr, key=lambda j: cands[j]["wape"])]["terms"]
            names = eq["lib"].names
            cols = [names.index(k) for k in full]
            no_t = [j for j in cols if not is_special(names[j], time_tokens, neighbour)]
            row["heldout"] = dict(full=interleaved_r2(eq["lib"].Xl[:, cols], eq["y"]),
                                  no_t=interleaved_r2(eq["lib"].Xl[:, no_t], eq["y"]) if no_t else float("nan"))
        out.append(row)
    return out


# ------------------------------------------------------------------ summary
def propulsion(tx, ty):
    return SRE.propulsion(tx, ty)


def summary():
    rows = list(map(json.loads, open(RESULTS)))
    ols = {(r["tag"], r["robot_id"]): r["ols"] for r in map(json.loads, open(SRE.RESULTS)) if r["status"] == "ok"}
    by = {}
    for r in rows:
        by.setdefault(r["tag"], {}).setdefault(r["robot_id"], {})[r["var"]] = r
    for tag, robots in by.items():
        print(f"\n{tag}: {len(robots)} robots, median {np.median([r['seconds'] for v in robots.values() for r in v.values()]):.1f}s/eq")
        for pick in ("stable", "full"):
            got, ve, vo, te, to, cnt, sizes = 0, [], [], [], [], {}, []
            for rid, eqs in robots.items():
                terms = {}
                for var, r in eqs.items():
                    i = r["stable"] if pick == "stable" else min(r["front"], key=lambda j: r["cands"][j]["wape"])
                    terms[var] = r["cands"][i]["terms"]
                    sizes.append(len(terms[var]))
                    for k in terms[var]:
                        kk = k.replace(f"_{var}", "_own") if k.endswith(f"_{var}") else k
                        cnt[kk] = cnt.get(kk, 0) + 1
                p = propulsion(terms.get("x"), terms.get("y"))
                if p and (tag, rid) in ols:
                    got += 1
                    ve.append(p[0]); te.append(p[1]); vo.append(ols[(tag, rid)]["v0"]); to.append(ols[(tag, rid)]["theta"])
            msg = (f"  {pick:6s} (median {np.median(sizes):.0f} terms): own propulsion with angle for {got}/{len(robots)}")
            if got > 2:
                dth = (np.array(te) - np.array(to) + 180) % 360 - 180
                msg += (f"; v0 vs OLS corr {np.corrcoef(ve, vo)[0, 1]:+.2f} ratio {np.median(np.array(ve) / np.array(vo)):.2f}; "
                        f"theta median |diff| {np.median(np.abs(dth)):.1f} deg")
            print(msg)
            n_eq = sum(len(e) for e in robots.values())
            print("     share of equations with: " + ", ".join(
                f"{k} {100 * v / n_eq:.0f}%" for k, v in sorted(cnt.items(), key=lambda kv: -kv[1])))


def time_summary(path):
    """--time runs: how often time terms enter, how much they move propulsion, and whether they
    predict held-out (interleaved) data better than the same law without them."""
    rows = list(map(json.loads, open(path)))
    by = {}
    for r in rows:
        by.setdefault(r["tag"], []).append(r)
    for tag, rs in by.items():
        n = len(rs)
        has_t = lambda terms: any("t" in k.split("*") for k in terms)                     # noqa: E731
        full = [r["cands"][min(r["front"], key=lambda j: r["cands"][j]["wape"])]["terms"] for r in rs]
        stab = [r["cands"][r["stable"]]["terms"] for r in rs]
        gain = np.array([r["heldout"]["full"] - r["heldout"]["no_t"] for r in rs
                         if has_t(r["cands"][min(r["front"], key=lambda j: r["cands"][j]["wape"])]["terms"])])
        drift = []                                   # relative change of propulsion over the recording
        for r, f in zip(rs, full):
            v = r["var"]
            if all(k in f for k in ("c", "s")) and any(k in f for k in ("c*t", "s*t")):
                z0 = abs(f["c"] - 1j * f["s"]) if v == "x" else abs(f["s"] + 1j * f["c"])
                zt = (abs(f.get("c*t", 0) - 1j * f.get("s*t", 0)) if v == "x"
                      else abs(f.get("s*t", 0) + 1j * f.get("c*t", 0)))
                drift.append(zt / max(z0, 1e-9))
        cnt = {}
        for f in full:
            for k in f:
                if "t" in k.split("*"):
                    kk = k.replace("_x", "_own").replace("_y", "_own") if k[0] in "pwv" else k
                    cnt[kk] = cnt.get(kk, 0) + 1
        print(f"\n{tag}: {n} equations")
        print(f"  time terms in the stable end {100 * np.mean([has_t(s) for s in stab]):.0f}%, "
              f"in the most complete law {100 * np.mean([has_t(f) for f in full]):.0f}%  ("
              + ", ".join(f"{k} {100 * v / n:.0f}%" for k, v in sorted(cnt.items(), key=lambda kv: -kv[1])) + ")")
        if len(gain):
            print(f"  held-out R2 (interleaved 33 s blocks), law with vs without its time terms: median gain "
                  f"{np.median(gain):+.4f}, better in {100 * np.mean(gain > 0):.0f}% of {len(gain)} equations")
        if drift:
            print(f"  propulsion drift over the recording |d(v0 e^i theta)| / v0: median {np.median(drift):.2f} "
                  f"[{np.percentile(drift, 25):.2f}, {np.percentile(drift, 75):.2f}] ({len(drift)} equations)")


def neighbour_summary(path):
    """--neighbour runs: how often the neighbour-drive tokens enter, their coefficients, held-out gain."""
    rows = list(map(json.loads, open(path)))
    by = {}
    for r in rows:
        by.setdefault(r["tag"], []).append(r)
    for tag, rs in by.items():
        full = [r["cands"][min(r["front"], key=lambda j: r["cands"][j]["wape"])]["terms"] for r in rs]
        stab = [r["cands"][r["stable"]]["terms"] for r in rs]
        own = {"np": [], "np_in": []}
        for r, f in zip(rs, full):
            for k in own:
                c = f.get(f"{k}_{r['var']}")
                if c is not None:
                    own[k].append(c)
        has = lambda terms: any(k.startswith("np_") for k in terms)                       # noqa: E731
        gain = np.array([r["heldout"]["full"] - r["heldout"]["no_t"] for r, f in zip(rs, full) if has(f)])
        print(f"\n{tag}: {len(rs)} equations; neighbour drive in the stable end {100 * np.mean([has(s) for s in stab]):.0f}%, "
              f"in the most complete law {100 * np.mean([has(f) for f in full]):.0f}%")
        for k, v in own.items():
            if v:
                print(f"  {k} (own axis) in {100 * len(v) / len(rs):.0f}% of equations, coefficient median {np.median(v):+.3f} "
                      f"[{np.percentile(v, 25):+.3f}, {np.percentile(v, 75):+.3f}]")
        if len(gain):
            print(f"  held-out R2 (interleaved blocks), law with vs without its neighbour-drive terms: median gain "
                  f"{np.median(gain):+.4f}, mean {gain.mean():+.4f}, better in {100 * np.mean(gain > 0):.0f}% of {len(gain)}")


def main():
    global RESULTS
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--time", action="store_true", help="add t and t x token columns (results/robot_pipeline_t.jsonl)")
    ap.add_argument("--neighbour", action="store_true",
                    help="add the neighbour-drive tokens np, np_in (results/robot_pipeline_np.jsonl)")
    ap.add_argument("--one", nargs=2, metavar=("TAG", "ROBOT_ID"))
    a = ap.parse_args()
    if a.time and a.neighbour:
        ap.error("--time and --neighbour are separate tests")
    if a.time:
        RESULTS = os.path.join(HERE, "results", "robot_pipeline_t.jsonl")
    if a.neighbour:
        RESULTS = os.path.join(HERE, "results", "robot_pipeline_np.jsonl")
    if a.summary:
        return time_summary(RESULTS) if a.time else neighbour_summary(RESULTS) if a.neighbour else summary()
    if a.one:
        for r in run(a.one[0], int(a.one[1]), a.time, a.neighbour):
            print(f"\n{r['var']} ({r['seconds']}s)" + (f"  held-out R2 full {r['heldout']['full']:.4f}, "
                                                       f"without the tested terms {r['heldout']['no_t']:.4f}"
                                                       if "heldout" in r else ""))
            for i, c in enumerate(r["cands"]):
                mark = "*" if i == r["stable"] else ("f" if i in r["front"] else " ")
                print(f"  {mark} c={c['c']} WAPE {c['wape']:.4f} win14 {c['win14']:.3g} | "
                      + " + ".join(f"{v:.4g}*{k}" for k, v in c["terms"].items()))
        return
    jobs = [(swarm_meta(f)["tag"], int(rid)) for f in swarm_files() for rid in load_swarm(f)[4]]
    done = set()
    if os.path.exists(RESULTS):
        done = {(r["tag"], r["robot_id"]) for r in map(json.loads, open(RESULTS))}
    todo = [j for j in jobs if j not in done]
    print(f"{len(jobs)} robots, {len(done)} done, {len(todo)} to go", flush=True)
    with open(RESULTS, "a") as fh:
        for i, (tag, rid) in enumerate(todo, 1):
            for r in run(tag, rid, a.time, a.neighbour):
                fh.write(json.dumps(r) + "\n")
                se = r["cands"][r["stable"]]
                print(f"[{i}/{len(todo)}] {tag} id {rid} {r['var']}: stable end "
                      f"{' + '.join(se['terms'])} (WAPE {se['wape']:.3f}) [{r['seconds']}s]", flush=True)
            fh.flush()


if __name__ == "__main__":
    main()
