r"""Per-robot drift laws with EPDE: every robot its own search, with its own tokens.

Each coordinate of each robot of each recording (and the lone robot) is one EPDE run in
forward-increment mode (lag 10 frames = 0.4 s; central derivatives cannot see gradient drifts of
stochastic data). x and y are SEPARATE single-variable searches (centred on the arena, target
dx/dx0 resp. dy/dx0): in one system EPDE regresses dx/dx0 on dy/dx0, and the two forward increments
share the same future step -- a leak, not a law. The robot's own tokens, cached per trajectory:
    heading      c = cos phi, s = sin phi        own propulsion v0 e^{i theta} -> c and s in both equations
    interaction  push_x, push_y, wall_x, wall_y  (only the ones the robot ever touches)
    memory       vb_x, vb_y                      velocity over the previous 0.4 s
A shared-coefficient OLS of the same robot is stored as the reference (v0, theta, push, wall, a).
Every robot runs in a fresh interpreter (EPDE keeps process-global caches) under a hard timeout.

    python swarm_robot_epde.py                          # all robots, results/swarm_robot_epde.jsonl
    python swarm_robot_epde.py --one N30_PWM10_D41 12 x  # one robot, one coordinate, printed
    python swarm_robot_epde.py --summary
"""
import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")          # one BLAS thread per run: several run in parallel

HERE = os.path.dirname(os.path.abspath(__file__))
# insert, not append: a stale pip-installed epde would otherwise shadow the repo
sys.path.insert(0, os.path.abspath(os.path.join(HERE, '../..')))
sys.path.insert(0, HERE)

import numpy as np

from swarm_data import swarm_files, swarm_meta, load_swarm, arena_envelope, spin_rates, contact_features

RESULTS = os.path.join(HERE, "results", "swarm_robot_epde.jsonl")
LAG = 10
LONE = "N1_PWM10"


# ------------------------------------------------------------------ data of one robot
def robot_data(tag, robot_id):
    """Full-length series of one robot: t, x, y (centred), tokens {family: {label: array}}."""
    if tag == LONE:
        from robot_discovery import load_track
        t, x, y, phi = load_track()
        x, y = x - x.mean(), y - y.mean()
        inter = {}
    else:
        path = next(p for p in swarm_files() if swarm_meta(p)["tag"] == tag)
        t, X, Y, PHI, ids = load_swarm(path)
        xc, yc, r_env = arena_envelope(X, Y)
        F = contact_features(X, Y, spin_rates(t, PHI), xc, yc, r_env)
        k = list(ids).index(robot_id)
        x, y, phi = X[:, k] - xc, Y[:, k] - yc, PHI[:, k]
        inter = {lab: F[lab][:, k] for lab in ("push_x", "push_y", "wall_x", "wall_y")
                 if np.ptp(F[lab][:, k]) > 0}                  # a token the robot never feels is constant
    dt = t[1] - t[0]
    vb = {}
    for lab, pos in (("vb_x", x), ("vb_y", y)):
        v = np.zeros_like(pos)
        v[LAG:] = (pos[LAG:] - pos[:-LAG]) / (LAG * dt)
        vb[lab] = v
    tokens = {"heading": {"c": np.cos(phi), "s": np.sin(phi)}, "interaction": inter, "memory": vb}
    return t, x, y, phi, tokens


def ols_reference(t, x, y, phi, tokens):
    """Same robot, isotropic complex model z = v0 e^{i theta} e^{i phi} + mu_P P + mu_W W + a vb + const."""
    dt = t[1] - t[0]
    rows = np.arange(LAG + 1, len(t) - LAG - 1)
    z = ((x[rows + LAG] - x[rows]) + 1j * (y[rows + LAG] - y[rows])) / (LAG * dt)
    it = tokens["interaction"]
    P = it.get("push_x", 0 * x)[rows] + 1j * it.get("push_y", 0 * x)[rows]
    W = it.get("wall_x", 0 * x)[rows] + 1j * it.get("wall_y", 0 * x)[rows]
    zb = tokens["memory"]["vb_x"][rows] + 1j * tokens["memory"]["vb_y"][rows]
    E = np.exp(1j * phi[rows])
    n = len(rows)
    one, zero = np.ones(n), np.zeros(n)
    A = np.vstack([np.column_stack([E.real, -E.imag, P.real, W.real, zb.real, one, zero]),
                   np.column_stack([E.imag, E.real, P.imag, W.imag, zb.imag, zero, one])])
    keep = np.linalg.norm(A, axis=0) > 0
    c = np.zeros(A.shape[1])
    c[keep] = np.linalg.lstsq(A[:, keep], np.concatenate([z.real, z.imag]), rcond=None)[0]
    v0 = c[0] + 1j * c[1]
    return dict(v0=float(abs(v0)), theta=float(np.degrees(np.angle(v0))), push=float(c[2]),
                wall=float(c[3]), a=float(c[4]))


# ------------------------------------------------------------------ one EPDE run
def run_robot(tag, robot_id, var, popsize=10, epochs=15, terms=7, sparsity="knee"):
    from epde.interface.interface import EpdeSearch
    from epde import CacheStoredTokens
    t, x, y, phi, tokens = robot_data(tag, robot_id)
    t0 = time.time()
    # 'budget': best subset within each equation's evolved ('complexity', var) budget. The default knee
    # rule keeps an extension only if its RSS drop matches the weakest accepted one, and one dominant
    # propulsion term then shuts out push / wall / the heading offset for good.
    sp = (dict(sparsity_cls="knee", sparsity_kwargs={"realization": "complexity"}) if sparsity == "budget"
          else dict(sparsity_cls=sparsity))
    search = EpdeSearch(multiobjective_mode=True, use_solver=False, **sp,
                        boundary_width=LAG + 1,                 # forward tail and the vb head
                        default_preprocessor_type="forward", preprocessor_kwargs={"lag": LAG},
                        population_size=popsize, training_epochs=epochs,
                        verbose_params={"show_iter_idx": False})
    _, domain = search.createDomain((t,), ID=0)
    families, cached = [], {}
    for name, labs in tokens.items():
        if not labs:
            continue
        fam = CacheStoredTokens(token_type=name, token_labels=list(labs), token_tensors=None,
                                params_ranges={"power": (1, 1)}, params_equality_ranges=None, meaningful=True)
        families.append(fam)
        cached[fam] = labs
    _, traj = search.createTrajectory({var: x if var == "x" else y}, domain, cache_id=0,
                                      cached_token_tensors=cached)
    search.fit(data=[traj], max_deriv_order=(1,), data_fun_pow=1, equation_terms_max_number=terms,
               additional_tokens=families, equation_factors_max_number={"factors_num": [1], "probas": [1.0]})
    level0 = search.equations(only_print=False, num=1)[0]
    front = [dict(text=s.text_form.split("\n{")[0].replace("{power: 1.0}", "").strip(),
                  obj=[float(v) for v in np.ravel(s.obj_fun)]) for s in level0]
    return dict(tag=tag, robot_id=robot_id, var=var, status="ok", seconds=round(time.time() - t0, 1),
                omega=float(np.polyfit(t, phi, 1)[0]), tokens=[l for labs in tokens.values() for l in labs],
                ols=ols_reference(t, x, y, phi, tokens), front=front)


def _run_subprocess(tag, robot_id, var, timeout, extra):
    t0 = time.time()
    try:
        out = subprocess.run([sys.executable, "-u", os.path.abspath(__file__), "--one", tag, str(robot_id), var,
                              "--json"] + extra, capture_output=True, text=True, timeout=timeout, cwd=HERE)
        line = [l for l in out.stdout.splitlines() if l.startswith('{"tag"')]
        if line:
            return json.loads(line[-1])
        status = f"error: {out.stderr.strip().splitlines()[-1] if out.stderr.strip() else 'no output'}"
    except subprocess.TimeoutExpired:
        status = f"timeout {timeout}s"
    return dict(tag=tag, robot_id=robot_id, var=var, status=status, seconds=round(time.time() - t0, 1), front=[])


def run_all(workers, timeout, extra):
    jobs = [(LONE, 1, v) for v in ("x", "y")]
    for path in swarm_files():
        tag = swarm_meta(path)["tag"]
        jobs += [(tag, int(rid), v) for rid in load_swarm(path)[4] for v in ("x", "y")]
    done = set()
    if os.path.exists(RESULTS):
        done = {(r["tag"], r["robot_id"], r["var"]) for r in map(json.loads, open(RESULTS)) if r["status"] == "ok"}
    todo = [j for j in jobs if j not in done]
    print(f"{len(jobs)} runs (robot x coordinate), {len(done)} done, {len(todo)} to go", flush=True)
    with ThreadPoolExecutor(workers) as pool, open(RESULTS, "a") as fh:
        futures = [pool.submit(_run_subprocess, tag, rid, v, timeout, extra) for tag, rid, v in todo]
        for i, fut in enumerate(as_completed(futures), 1):
            row = fut.result()
            fh.write(json.dumps(row) + "\n"); fh.flush()
            print(f"[{i}/{len(todo)}] {row['tag']} id {row['robot_id']} {row['var']}: {row['status']} "
                  f"{row['seconds']}s, {len(row['front'])} equations", flush=True)


# ------------------------------------------------------------------ parsing / summary
def parse_system(text):
    """EPDE system text -> {target: ({term: coef}, intercept)} (keyed by target, never by slot)."""
    out = {}
    for eq in text.replace("\\", "|").split("\n"):
        eq = eq.strip(" /|")
        if " = " not in eq:
            continue
        lhs, tgt = eq.rsplit(" = ", 1)
        terms, icpt = {}, 0.0
        for item in lhs.split(" + "):
            if " * " in item:
                k, name = item.split(" * ", 1)
                terms[name.strip()] = float(k)
            else:
                icpt = float(item)
        out[tgt.strip()] = (terms, icpt)
    return out


def normal_form(text, var):
    """One EPDE equation -> ({term: coef}, intercept) solved for d<var>/dx0, or None if it does not
    contain the derivative. Inverted forms (target c, derivative on the right) are solved back."""
    d = f"d{var}/dx0"
    (tgt, (terms, icpt)), = parse_system(text).items()
    if tgt == d:
        return terms, icpt, "normal"
    if d not in terms:
        return None
    k = terms.pop(d)
    out = {tgt: 1.0 / k, **{n: -c / k for n, c in terms.items()}}
    return out, -icpt / k, "inverted"


def most_complete(row):
    """Lowest-discrepancy member of the front that describes the derivative (EPDE obj = [disc, instab])."""
    eqs = [(normal_form(s["text"], row["var"]), s["obj"]) for s in row["front"]]
    eqs = [(e, o) for e, o in eqs if e is not None]
    return min(eqs, key=lambda eo: eo[1][0]) if eqs else (None, None)


def propulsion(ex, ey):
    """v0 e^{i theta} from x eq  v0 cos(th) c - v0 sin(th) s  and  y eq  v0 sin(th) c + v0 cos(th) s;
    each equation alone determines it when it holds both c and s."""
    zs = []
    if ex and "c" in ex and "s" in ex:
        zs.append(ex["c"] - 1j * ex["s"])
    if ey and "c" in ey and "s" in ey:
        zs.append(ey["s"] + 1j * ey["c"])
    if not zs:
        return None
    z = np.mean(zs)
    return float(abs(z)), float(np.degrees(np.angle(z)))


def summary():
    rows = list(map(json.loads, open(RESULTS)))
    ok = [r for r in rows if r["status"] == "ok"]
    print(f"{len(ok)} ok runs, {len(rows) - len(ok)} failed "
          f"({sorted({r['status'] for r in rows if r['status'] != 'ok'})}); "
          f"median {np.median([r['seconds'] for r in ok]):.0f}s")
    runs = {(r["tag"], r["robot_id"], r["var"]): r for r in ok}
    for tag in dict.fromkeys(r["tag"] for r in ok):
        ids = sorted({rid for (tg, rid, _) in runs if tg == tag})
        got, forms, terms, v_e, v_o, th_e, th_o = 0, {}, {}, [], [], [], []
        for rid in ids:
            best = {}
            for var in ("x", "y"):
                r = runs.get((tag, rid, var))
                e, _ = most_complete(r) if r else (None, None)
                best[var] = e
                if e:
                    forms[e[2]] = forms.get(e[2], 0) + 1
                    for n in e[0]:
                        terms[n] = terms.get(n, 0) + 1
            p = propulsion(best["x"] and best["x"][0], best["y"] and best["y"][0])
            ref = (runs.get((tag, rid, "x")) or runs.get((tag, rid, "y")))["ols"]
            if p:
                got += 1
                v_e.append(p[0]); th_e.append(p[1]); v_o.append(ref["v0"]); th_o.append(ref["theta"])
        msg = f"{tag:15s} {len(ids):3d} robots: own propulsion (c and s in an equation) for {got}"
        if got > 2:
            dth = (np.array(th_e) - np.array(th_o) + 180) % 360 - 180
            msg += (f"; v0 EPDE vs OLS corr {np.corrcoef(v_e, v_o)[0, 1]:+.2f}, median ratio "
                    f"{np.median(np.array(v_e) / np.array(v_o)):.2f}; theta corr "
                    f"{np.corrcoef(th_e, th_o)[0, 1]:+.2f}, median |diff| {np.median(np.abs(dth)):.1f} deg")
        print(msg + f"; forms {forms}")
        print("   terms in the most complete equations: " +
              ", ".join(f"{k} {v}" for k, v in sorted(terms.items(), key=lambda kv: -kv[1])))


def epde_text(text):
    """Put EPDE's factor parameters back on the stored (stripped) equation text."""
    lhs, tgt = text.rsplit(" = ", 1)
    items = []
    for item in lhs.split(" + "):
        if " * " in item:
            k, name = item.split(" * ", 1)
            items.append(f"{k} * {name.strip()}{{power: 1.0}}")
        else:
            items.append(item)
    return " + ".join(items) + f" = {tgt.strip()}{{power: 1.0}}"


def report():
    """EPDE-style text of every robot's fronts, per recording -> results/swarm_robot_epde.txt."""
    rows = [r for r in map(json.loads, open(RESULTS)) if r["status"] == "ok"]
    runs = {(r["tag"], r["robot_id"], r["var"]): r for r in rows}
    out = os.path.join(HERE, "results", "swarm_robot_epde.txt")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(__doc__ + "\nObjective function values: [discrepancy, instability] (EPDE defaults).\n"
                 "Propulsion from the lowest-discrepancy equation: x eq  v0 cos(th) c - v0 sin(th) s,"
                 "  y eq  v0 sin(th) c + v0 cos(th) s.\n")
        for tag in dict.fromkeys(r["tag"] for r in rows):
            fh.write(f"\n{'=' * 100}\n{tag}\n")
            for rid in sorted({rid for (tg, rid, _) in runs if tg == tag}):
                rx, ry = runs.get((tag, rid, "x")), runs.get((tag, rid, "y"))
                ref = (rx or ry)["ols"]
                ex = most_complete(rx)[0] if rx else None
                ey = most_complete(ry)[0] if ry else None
                p = propulsion(ex and ex[0], ey and ey[0])
                fh.write(f"\n robot {rid}: w = {(rx or ry)['omega']:+.3f} rad/s | EPDE propulsion "
                         + (f"v0 {p[0]:.1f} px/s, theta {p[1]:+.1f} deg" if p else "not found")
                         + f" | OLS reference v0 {ref['v0']:.1f} px/s, theta {ref['theta']:+.1f} deg, "
                           f"push {ref['push']:.1f}, wall {ref['wall']:.1f}, a {ref['a']:.2f}\n")
                for var, r in (("x", rx), ("y", ry)):
                    if r is None:
                        fh.write(f"  d{var}/dx0: run missing\n")
                        continue
                    fh.write(f"  d{var}/dx0, 0-th non-dominated level:\n")
                    for s in sorted(r["front"], key=lambda s: s["obj"][0]):
                        fh.write(f"   {epde_text(s['text'])} , with objective function values of "
                                 f"[{' '.join(f'{v:.4g}' for v in s['obj'])}]\n")
    print(f"written {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--timeout", type=int, default=900, help="seconds per robot")
    ap.add_argument("--popsize", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--terms", type=int, default=7)
    ap.add_argument("--sparsity", default="knee", choices=["knee", "budget", "lasso", "vwsr"])
    ap.add_argument("--one", nargs=3, metavar=("TAG", "ROBOT_ID", "VAR"))
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--report", action="store_true", help="EPDE-style text of all fronts")
    a = ap.parse_args()
    extra = ["--popsize", str(a.popsize), "--epochs", str(a.epochs), "--terms", str(a.terms),
             "--sparsity", a.sparsity]
    if a.summary:
        summary()
    elif a.report:
        report()
    elif a.one:
        row = run_robot(a.one[0], int(a.one[1]), a.one[2], a.popsize, a.epochs, a.terms, a.sparsity)
        if a.json:
            print(json.dumps(row, default=float))
        else:
            print(json.dumps({k: v for k, v in row.items() if k != "front"}, default=float, indent=1))
            for s in row["front"]:
                print(s["text"], "  obj", np.round(s["obj"], 4).tolist())
    else:
        run_all(a.workers, a.timeout, extra)
