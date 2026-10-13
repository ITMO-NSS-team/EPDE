r"""Mesoscale: field equations of the swarm's coarse-grained density and polarization, per recording.

Fields on the square inscribed in the arena (away from the wall), Gaussian kernel K of width one cap
diameter (4.87 cm), every 2 frames (0.08 s), positions in cm (common camera scale), time in s:
    rho(t, r) = sum_i A K(r - r_i(t))               area fraction, A = cap area pi (2.435 cm)^2
    P(t, r)   = sum_i A p_i K(r - r_i)              polarization of the self-propulsion direction
                                                     p_i = (cos(phi_i + theta_i), sin(phi_i + theta_i)),
                                                     theta_i = the robot's own offset (common-system law)
    W(t, r)   = sum_i A w_i p_i K(r - r_i)          spin-weighted polarization [rad/s], w_i own spin rate
Targets: forward increments over 0.08 s of rho, Px, Py, Wx, Wy (d<f>/dx0; x0 = t, x1 = x, x2 = y).
Library per target: the target field and its spatial derivatives (central differences), the other
fields, their first derivatives (<f>_x, <f>_y) and Laplacians (lap_<f>); every product of <= 2 atoms
(cost = atom count). Pipeline (subset_search.py, cxfast.py): the exact best support per complexity
c = 1..CMAX, OLS on a lattice (every 8th time step, all grid points), WAPE on the whole inner domain,
window-refit instability win14, the exact (WAPE, win14) front; each member also refit on the first
half and scored on the second (held-out R2). Reported law: the BALANCED member, both precise and
stable (WAPE and log10 win14 rescaled to [0, 1] over the front, closest to the ideal corner). EPDE itself picks its own left-hand side here (a spatial
second derivative) and has no setting that pins the time derivative, so the pipeline is used.
For independent spinning self-propelled discs the expectation is
    rho_t = -v0 div P + D lap rho,   P_t = z x W - D_r P + ...,   W_t = <w^2> z x P + ...
(scored the same way as 'expected form').

python swarm_fields.py [--workers 7 --cmax 6]   -> results/swarm_fields.json, results/swarm_fields.txt
"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import swarm_sim as SS                                           # noqa: E402
import subset_search as SUB                                      # noqa: E402
import robot_wide as RW                                          # noqa: E402
import cxfast                                                    # noqa: E402
from swarm_data import swarm_files, swarm_meta, load_swarm, arena_envelope, spin_rates  # noqa: E402

STEP, NGRID, BOUNDARY, LAG, LAT_STRIDE = 2, 14, 2, 1, 8
VARS = ["rho", "Px", "Py", "Wx", "Wy"]
FIELD_DIR = os.path.join(HERE, "results", "fields")
EXPECTED = {"rho": ["Px_x", "Py_y", "d^2rho/dx1^2", "d^2rho/dx2^2"],
            "Px": ["Wy", "Px", "d^2Px/dx1^2", "d^2Px/dx2^2"], "Py": ["Wx", "Py", "d^2Py/dx1^2", "d^2Py/dx2^2"],
            "Wx": ["Py", "Wx", "d^2Wx/dx1^2", "d^2Wx/dx2^2"], "Wy": ["Px", "Wy", "d^2Wy/dx1^2", "d^2Wy/dx2^2"]}


# ------------------------------------------------------------------ fields
def build_fields(path):
    tag = swarm_meta(path)["tag"]
    out = os.path.join(FIELD_DIR, f"{tag}.npz")
    if os.path.exists(out):
        return dict(np.load(out))
    t, x, y, phi, ids = load_swarm(path)
    xc, yc, r_env = arena_envelope(x, y)
    s = SS.px_per_cm()
    R = SS.recording(path)
    K = SS.fit_joint(R, True, False)[0]
    a, b = K[:, 0, SS.TI["c"]], K[:, 1, SS.TI["c"]]
    theta, v0 = np.arctan2(b, a), np.hypot(a, b) / s
    w = spin_rates(t, phi)
    sl = slice(None, None, STEP)
    X, Y, ph = (x[sl] - xc) / s, (y[sl] - yc) / s, phi[sl]
    half = (r_env / s) / np.sqrt(2)
    g = np.linspace(-half, half, NGRID)
    ell = SS.CAP_CM
    norm = np.pi * (SS.CAP_CM / 2) ** 2 / (2 * np.pi * ell ** 2)
    gx = np.exp(-(g[None, None, :] - X[:, :, None]) ** 2 / (2 * ell ** 2))
    gy = np.exp(-(g[None, None, :] - Y[:, :, None]) ** 2 / (2 * ell ** 2))
    px, py = np.cos(ph + theta), np.sin(ph + theta)
    field = lambda wt: norm * np.einsum("ti,tix,tiy->txy", wt, gx, gy)     # noqa: E731
    F = dict(t=t[sl], g=g, rho=field(np.ones_like(px)), Px=field(px), Py=field(py), Wx=field(w * px),
             Wy=field(w * py), v0=v0, theta=theta, w=w)
    os.makedirs(FIELD_DIR, exist_ok=True)
    np.savez_compressed(out, **F)
    return F


def spatial(F, a, axis):
    return np.gradient(a, F["g"], axis=axis, edge_order=2)


def atoms_for(F, var, fields=VARS):
    """[(name, field)] for target `var`: own field and spatial derivatives, the other fields, their
    first derivatives and Laplacians."""
    u = F[var]
    ux, uy = spatial(F, u, 1), spatial(F, u, 2)
    out = [(var, u), (f"d{var}/dx1", ux), (f"d{var}/dx2", uy),
           (f"d^2{var}/dx1^2", spatial(F, ux, 1)), (f"d^2{var}/dx2^2", spatial(F, uy, 2))]
    for v in fields:
        if v == var:
            continue
        gx, gy = spatial(F, F[v], 1), spatial(F, F[v], 2)
        out += [(v, F[v]), (f"{v}_x", gx), (f"{v}_y", gy), (f"lap_{v}", spatial(F, gx, 1) + spatial(F, gy, 2))]
    return out


def atom(F, var, name, fields=VARS):
    """One atom of atoms_for(F, var, fields) by name, computed the same way; None if not an atom."""
    u = F[var]
    own = {var: lambda: u, f"d{var}/dx1": lambda: spatial(F, u, 1), f"d{var}/dx2": lambda: spatial(F, u, 2),
           f"d^2{var}/dx1^2": lambda: spatial(F, spatial(F, u, 1), 1),
           f"d^2{var}/dx2^2": lambda: spatial(F, spatial(F, u, 2), 2)}
    if name in own:
        return own[name]()
    for v in fields:
        if v == var:
            continue
        if name == v:
            return F[v]
        if name == f"{v}_x":
            return spatial(F, F[v], 1)
        if name == f"{v}_y":
            return spatial(F, F[v], 2)
        if name == f"lap_{v}":
            return spatial(F, spatial(F, F[v], 1), 1) + spatial(F, spatial(F, F[v], 2), 2)
    return None


def inner(a):
    b = BOUNDARY
    return a[: -max(b, LAG)][:, b:-b, b:-b] if a.ndim == 3 else a


def problem(F, var, fields=VARS):
    """Target, atoms and lattice on the inner domain, rows ordered (t, x, y)."""
    u = F[var]
    y = (u[LAG:] - u[:-LAG]) / (F["t"][LAG:] - F["t"][:-LAG])[:, None, None]
    y = np.concatenate([y, np.full((LAG,) + u.shape[1:], np.nan)])
    yi = inner(y)
    grid = yi.shape
    atoms = [(n, inner(a).ravel()) for n, a in atoms_for(F, var, fields)]
    lat = (np.arange(0, grid[0], LAT_STRIDE)[:, None] * grid[1] * grid[2]
           + np.arange(grid[1] * grid[2])[None, :]).ravel()
    return dict(target=f"d{var}/dx0", y=yi.ravel(), atoms=atoms, grid=grid, lattice=lat)


# ------------------------------------------------------------------ pipeline
def heldout_r2(A, y, grid):
    T = grid[0]
    rows = np.arange(len(y)).reshape(grid)
    tr, te = rows[: T // 2 - 2].ravel(), rows[T // 2 + 2:].ravel()
    coef = np.linalg.lstsq(A[tr], y[tr], rcond=None)[0]
    e = y[te] - A[te] @ coef
    return float(1 - e @ e / np.sum((y[te] - y[te].mean()) ** 2))


def run(F, var, cmax, workers, fields=VARS, expected=None):
    P = problem(F, var, fields)
    lib = RW.Library(P["atoms"], P["lattice"], max_atoms=2)
    y, lat, grid = P["y"], P["lattice"], P["grid"]
    gm = SUB.Gram(lib.Xl, y[lat])
    t0 = time.time()
    res = cxfast.search(gm, lib.ks, cmax, B=200, workers=workers)
    cands = []
    for c, r in res.items():
        S = r["best"]
        coef = np.linalg.lstsq(lib.Xl[:, S], y[lat], rcond=None)[0]
        A = np.column_stack([lib.column(j) for j in S])
        cands.append(dict(c=int(c), terms={lib.names[j]: float(v) for j, v in zip(S, coef)},
                          wape=float(np.sum(np.abs(y - A @ coef)) / np.sum(np.abs(y))),
                          win14=float(SUB.instab(A, y, grid, **SUB.STATS["win14"])[0]),
                          r2=heldout_r2(A, y, grid)))
    fr = SUB.front(cands, list(range(len(cands))), "wape", "win14")
    named = {n: a for n, a in P["atoms"]}
    enames = expected or EXPECTED[var]
    E = np.column_stack([named[n] for n in enames])
    ecoef = np.linalg.lstsq(E[lat], y[lat], rcond=None)[0]
    expected = dict(terms=dict(zip(enames, ecoef.tolist())),
                    wape=float(np.sum(np.abs(y - E @ ecoef)) / np.sum(np.abs(y))),
                    win14=float(SUB.instab(E, y, grid, **SUB.STATS["win14"])[0]), r2=heldout_r2(E, y, grid))
    return dict(var=var, n_columns=len(lib.names), atoms=[n for n, _ in P["atoms"]], grid=list(grid),
                seconds=round(time.time() - t0, 1),
                cands=cands, front=[int(i) for i in fr], stable=int(min(fr, key=lambda i: cands[i]["win14"])),
                expected=expected)


def balanced(cands, fr):
    """The front member both precise and stable: WAPE and log10(win14), each rescaled to [0, 1] over
    the front, and the member closest to the ideal corner (0, 0)."""
    w = np.array([cands[i]["wape"] for i in fr], float)
    s = np.log10(np.maximum([cands[i]["win14"] for i in fr], 1e-300))
    unit = lambda a: (a - a.min()) / (a.max() - a.min()) if a.max() > a.min() else np.zeros_like(a)   # noqa: E731
    return fr[int(np.argmin(np.hypot(unit(w), unit(s))))]


# ------------------------------------------------------------------ EPDE text form
def epde_factor(f, atoms):
    """Library factor -> EPDE token; a repeated atom is written 'atom^k' by the library."""
    if f in atoms or "^" not in f or f.rsplit("^", 1)[0] not in atoms:
        return f"{f}{{power: 1.0}}"
    base, p = f.rsplit("^", 1)
    return f"{base}{{power: {float(p)}}}"


def epde_text(terms, target, atoms):
    body = "".join(f"{float(f'{v:.4g}')} * " + " * ".join(epde_factor(f, atoms) for f in k.split("*")) + " + "
                   for k, v in terms.items())
    return body + f"0.0 = {target}{{power: 1.0}}"


def report(out):
    lines = [__doc__.split("python swarm_fields")[0]]
    for tag, rec in out.items():
        lines.append(f"\n{'=' * 100}\n{tag}: <rho> {rec['rho_mean']:.3f} (area fraction), <|P|> {rec['P_mean']:.3f}, "
                     f"mean v0 {rec['v0_mean']:.2f} cm/s, <w^2> {rec['w2_mean']:.2f} rad^2/s^2")
        lines.append(" balanced laws (precise and stable; held-out R2):")
        for var in VARS:
            r = rec["eqs"][var]
            c = r["cands"][balanced(r["cands"], r["front"])]
            lines.append(f"   {epde_text(c['terms'], f'd{var}/dx0', set(r['atoms']))}   R2 {c['r2']:.3f}")
        for var in VARS:
            r = rec["eqs"][var]
            e = r["expected"]
            atoms = set(r["atoms"])
            bal = balanced(r["cands"], r["front"])
            lines.append(f"\n d{var}/dx0: {r['n_columns']} columns, front of {len(r['front'])} (> = balanced, "
                         f"precise and stable); [WAPE, win14] and held-out R2")
            for i in r["front"]:
                c = r["cands"][i]
                lines.append(f"  {'>' if i == bal else ' '} {epde_text(c['terms'], f'd{var}/dx0', atoms)} , "
                             f"with objective function values of [{c['wape']:.4g} {c['win14']:.4g}]   R2 {c['r2']:.3f}")
            lines.append(f"    expected: {epde_text(e['terms'], f'd{var}/dx0', atoms)} , [{e['wape']:.4g} "
                         f"{e['win14']:.4g}]   R2 {e['r2']:.3f}")
    text = "\n".join(lines) + "\n"
    open(os.path.join(HERE, "results", "swarm_fields.txt"), "w", encoding="utf-8").write(text)
    print(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--cmax", type=int, default=6)
    ap.add_argument("--report", action="store_true", help="only rewrite the text from results/swarm_fields.json")
    a = ap.parse_args()
    path_json = os.path.join(HERE, "results", "swarm_fields.json")
    out = json.load(open(path_json)) if os.path.exists(path_json) else {}
    if not a.report:
        for f in swarm_files():
            tag = swarm_meta(f)["tag"]
            F = build_fields(f)
            rec = dict(rho_mean=float(F["rho"].mean()), P_mean=float(np.hypot(F["Px"], F["Py"]).mean()),
                       v0_mean=float(F["v0"].mean()), w2_mean=float(np.mean(F["w"] ** 2)), eqs={})
            for var in VARS:
                rec["eqs"][var] = run(F, var, a.cmax, a.workers)
                st = rec["eqs"][var]
                bal = st["cands"][balanced(st["cands"], st["front"])]
                print(f"{tag} d{var}/dx0: {st['seconds']}s, front {len(st['front'])}, balanced "
                      f"{bal['terms']} R2 {bal['r2']:.3f}", flush=True)
            out[tag] = rec
            json.dump(out, open(path_json, "w"), indent=1)
    report(out)


if __name__ == "__main__":
    main()
