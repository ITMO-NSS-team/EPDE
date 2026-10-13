r"""Polar fields about the arena centre: the swarm's density and polarization on rings and angles over the
whole arena (out to the wall), their laws, and their simulation.

Grid: NR rings r = R_IN .. R (R = the radius robot centres reach, arena_envelope) and NT angles. Gaussian
kernel of one cap diameter (as swarm_fields.py), fields every 0.08 s, then every 0.4 s (TAU):
    rho          area fraction
    P_r, P_t     the polarization P resolved along r_hat (outward) and t_hat (the direction of increasing
                 angle in the image frame, x right / y down) at each grid point
    W_r, W_t     the same for the spin-weighted polarization
A law with fixed coefficients in these components is the same in every direction about the centre, so no
x / y pairing is needed; the radial and the tangential laws may differ.
Library per target: own field f and d f/dr, (1/r) d f/dtheta, d^2 f/dr^2, (1/r^2) d^2 f/dtheta^2; the other
fields, their d/dr, (1/r) d/dtheta and polar Laplacian; static maps rho_bar (each point's time-mean density),
r, 1/r and 1 (so a law can change with the distance from the centre); products of <= 2 atoms. Fit on 0.4 s
increments over every angle and every ring but the first and the last, pipeline and front exactly as
swarm_fields.run (subset_search + cxfast, exact (WAPE, win14) front, held-out R2).
Simulation: explicit 0.4 s steps, periodic in angle, the fit's own one-sided radial differences at the first
and the last ring (mirroring the fields there instead sets the radial gradient at the wall to zero and erases
the density's drop at the wall, and with it any edge-driven circulation), the laws'
recorded residual fields (one random recorded instant per step) with their time mean removed as noise (the
mean is the law's systematic misfit; left in, it alone reproduces e.g. the circulation W_t(r) rising toward
the wall, which the laws lack); a law using rho_bar relaxes toward the recorded mean density map, so its
density profile is matched by construction. No law is selected: every front law
of every equation is simulated one at a time with the other equations at their balanced laws (as
swarm_field_sim --fronts). Statistics area-weighted (by r), plus radial profiles (time and angle means).

python swarm_fields_polar.py --fit [--workers 7 --cmax 6]  -> results/swarm_fields_polar.json, .txt
python swarm_fields_polar.py --sim [--tags ...] [--seeds 4] -> results/swarm_field_sim_polar_<tag>.json
python swarm_fields_polar.py --merge                        -> results/swarm_field_sim_polar.json
python swarm_fields_polar.py --hypotheses [--tags ...]       -> results/swarm_field_hypotheses_polar.json
    Several hypothesis systems (HYPOTHESES): H1 transport + rotating polarization, H2 + density relaxation,
    H3 + circulation driven by the density's shape or the distance from the centre. Every law is a member of
    its equation's non-dominated front (the most stable member with the hypothesis's property); a
    hypothesis with no such front member is not supported in that recording.
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
import swarm_fields as SF                                        # noqa: E402
import swarm_field_sim as FS                                     # noqa: E402
import subset_search as SUB                                      # noqa: E402
import robot_wide as RW                                          # noqa: E402
import cxfast                                                    # noqa: E402
from swarm_data import swarm_files, swarm_meta, load_swarm, arena_envelope  # noqa: E402

NR, NT, R_IN = 10, 24, 1.0         # rings, angles, innermost ring radius (cm)
BOUND, TAU, PAD = 1, 5, 2          # rings left out of the fit at each end; fit step (x 0.08 s); padding rings
VARS = ["rho", "Pr", "Pt", "Wr", "Wt"]
STATIC = ["rho_bar", "inv_r", "r", "one"]


# ------------------------------------------------------------------ fields
def coarse_polar(X, Y, phi, theta, w, r, th, chunk=200):
    """Kernel fields of robot centres (X, Y in cm about the arena centre; (T, N)) on the polar grid."""
    ell = SS.CAP_CM
    norm = np.pi * (SS.CAP_CM / 2) ** 2 / (2 * np.pi * ell ** 2)
    gx, gy = r[:, None] * np.cos(th)[None, :], r[:, None] * np.sin(th)[None, :]
    px, py = np.cos(phi + theta), np.sin(phi + theta)
    parts = {k: [] for k in ("rho", "Px", "Py", "Wx", "Wy")}
    for a in range(0, len(X), chunk):
        s = slice(a, a + chunk)
        K = norm * np.exp(-((X[s, :, None, None] - gx) ** 2 + (Y[s, :, None, None] - gy) ** 2) / (2 * ell ** 2))
        parts["rho"].append(K.sum(1))
        for name, wt in (("Px", px[s]), ("Py", py[s]), ("Wx", w * px[s]), ("Wy", w * py[s])):
            parts[name].append(np.einsum("tn,tnrk->trk", wt, K))
    f = {k: np.concatenate(v) for k, v in parts.items()}
    c, sn = np.cos(th)[None, None, :], np.sin(th)[None, None, :]
    return dict(rho=f["rho"], Pr=f["Px"] * c + f["Py"] * sn, Pt=-f["Px"] * sn + f["Py"] * c,
                Wr=f["Wx"] * c + f["Wy"] * sn, Wt=-f["Wx"] * sn + f["Wy"] * c)


def grid_for(path):
    t, x, y, phi, ids = load_swarm(path)
    xc, yc, r_env = arena_envelope(x, y)
    s = SS.px_per_cm()
    return (t, x, y, phi, xc, yc, s), np.linspace(R_IN, r_env / s, NR), np.arange(NT) * 2 * np.pi / NT


def build_polar(path):
    """Polar fields every 0.08 s (cached), robots' drive offsets and spin rates from swarm_fields."""
    tag = swarm_meta(path)["tag"]
    out = os.path.join(SF.FIELD_DIR, f"{tag}_polar.npz")
    if os.path.exists(out):
        return dict(np.load(out))
    F = SF.build_fields(path)
    (t, x, y, phi, xc, yc, s), r, th = grid_for(path)
    sl = slice(None, None, SF.STEP)
    P = coarse_polar((x[sl] - xc) / s, (y[sl] - yc) / s, phi[sl], F["theta"], F["w"], r, th)
    P.update(t=t[sl], r=r, th=th, theta=F["theta"], w=F["w"])
    np.savez_compressed(out, **P)
    return P


def fields_polar(path, tau=TAU):
    """The fields every tau-th step (0.4 s) with rho_bar, each point's time-mean density."""
    P = build_polar(path)
    G = {k: (v[::tau] if isinstance(v, np.ndarray) and v.ndim == 3 else v) for k, v in P.items()}
    G["t"] = P["t"][::tau]
    G["rho_bar"] = P["rho"].mean(0)
    return G


# ------------------------------------------------------------------ polar atoms
def d_r(F, a):
    return np.gradient(a, F["r"], axis=1, edge_order=2)


def d_t(F, a):
    h = 2 * np.pi / NT
    return (np.roll(a, -1, axis=2) - np.roll(a, 1, axis=2)) / (2 * h) / F["r"][None, :, None]


def d_tt(F, a):
    h = 2 * np.pi / NT
    return (np.roll(a, -1, axis=2) - 2 * a + np.roll(a, 1, axis=2)) / h ** 2 / F["r"][None, :, None] ** 2


class PolarAtoms:
    """Library atoms of every law at one state, each (operation, field) computed once and shared."""

    def __init__(self, state, fields):
        self.s, self.fields, self.cache = state, fields, {}

    def value(self, op, v):
        if (op, v) not in self.cache:
            s = self.s
            shape = s["rho"].shape
            if op == "id":
                val = s[v]
            elif op == "dr":
                val = d_r(s, s[v])
            elif op == "dt":
                val = d_t(s, s[v])
            elif op == "drr":
                val = d_r(s, self.value("dr", v))
            elif op == "dtt":
                val = d_tt(s, s[v])
            elif op == "lap":
                val = self.value("drr", v) + self.value("dr", v) / s["r"][None, :, None] + self.value("dtt", v)
            elif op == "rho_bar":
                val = np.broadcast_to(s["rho_bar"], shape)
            elif op == "inv_r":
                val = np.broadcast_to(1.0 / s["r"][None, :, None], shape)
            elif op == "r":
                val = np.broadcast_to(s["r"][None, :, None], shape)
            else:                                                              # "one"
                val = np.ones(shape)
            self.cache[(op, v)] = val
        return self.cache[(op, v)]

    def names(self, var):
        """Atom name -> (operation, field), as var's library names them, in library order."""
        out = {var: ("id", var), f"d{var}/dr": ("dr", var), f"d{var}/dt": ("dt", var),
               f"d^2{var}/dr^2": ("drr", var), f"d^2{var}/dt^2": ("dtt", var)}
        for v in self.fields:
            if v != var:
                out.update({v: ("id", v), f"{v}_r": ("dr", v), f"{v}_t": ("dt", v), f"lap_{v}": ("lap", v)})
        out.update({n: (n, None) for n in STATIC})
        return out

    def view(self, var):
        names = self.names(var)
        return type("View", (), {"get": lambda _, n: None if n not in names else self.value(*names[n])})()


def atoms_for(F, var, fields=VARS):
    A = PolarAtoms(F, fields)
    return [(n, A.value(*key)) for n, key in A.names(var).items()]


def law_rhs(state, var, terms, fields=VARS, atoms=None):
    view = (atoms or PolarAtoms(state, fields)).view(var)
    return sum(c * FS.term_array(view, n) for n, c in terms.items())


# ------------------------------------------------------------------ pipeline
def inner(a):
    return a[:-1, BOUND:NR - BOUND, :]


def problem(F, var, fields=VARS, lat_stride=2):
    u = F[var]
    y = (u[1:] - u[:-1]) / np.diff(F["t"])[:, None, None]
    y = np.concatenate([y, np.full((1,) + u.shape[1:], np.nan)])
    yi = inner(y)
    grid = yi.shape
    atoms = [(n, inner(np.asarray(a)).ravel()) for n, a in atoms_for(F, var, fields)]
    lat = (np.arange(0, grid[0], lat_stride)[:, None] * grid[1] * grid[2] + np.arange(grid[1] * grid[2])[None, :]).ravel()
    return dict(y=yi.ravel(), atoms=atoms, grid=grid, lattice=lat)


def run(F, var, cmax, workers, fields=VARS):
    """As swarm_fields.run, on the polar library."""
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
                          r2=SF.heldout_r2(A, y, grid)))
    fr = SUB.front(cands, list(range(len(cands))), "wape", "win14")
    return dict(var=var, n_columns=len(lib.names), grid=list(grid), seconds=round(time.time() - t0, 1),
                cands=cands, front=[int(i) for i in fr])


def fit_main(a):
    path_json = os.path.join(HERE, "results", "swarm_fields_polar.json")
    out = json.load(open(path_json)) if os.path.exists(path_json) else {}
    for path in swarm_files():
        tag = swarm_meta(path)["tag"]
        F = fields_polar(path)
        rec = dict(r=F["r"].tolist(), eqs={})
        for var in VARS:
            st = run(F, var, a.cmax, a.workers)
            rec["eqs"][var] = st
            bal = st["cands"][SF.balanced(st["cands"], st["front"])]
            print(f"{tag} d{var}/dx0: {st['n_columns']} cols, {st['seconds']}s, front {len(st['front'])}, balanced "
                  f"{ {k: round(v, 3) for k, v in bal['terms'].items()} } R2 {bal['r2']:.3f}", flush=True)
        out[tag] = rec
        json.dump(out, open(path_json, "w"), indent=1)
    lines = [__doc__.split("python swarm_fields_polar")[0]]
    for tag, rec in out.items():
        lines.append(f"\n{'=' * 100}\n{tag}")
        for var, st in rec["eqs"].items():
            lines.append(f"  d{var}/dx0 front ({st['n_columns']} columns), most stable first: [WAPE, win14] held-out R2")
            for i in sorted(st["front"], key=lambda i: st["cands"][i]["win14"]):
                c = st["cands"][i]
                lines.append("    " + " ".join(f"{v:+.4g} {n}" for n, v in c["terms"].items())
                             + f"   [{c['wape']:.4f} {c['win14']:.3g}] R2 {c['r2']:.3f}")
    open(os.path.join(HERE, "results", "swarm_fields_polar.txt"), "w", encoding="utf-8").write("\n".join(lines) + "\n")


# ------------------------------------------------------------------ simulation
def mirrored(state, fields):
    """Fields mirrored beyond the inner and the outer ring (zero radial gradient); angle is periodic."""
    r = state["r"]
    h = r[1] - r[0]
    out = {"r": np.concatenate([r[0] - h * np.arange(PAD, 0, -1), r, r[-1] + h * np.arange(1, PAD + 1)]),
           **{v: np.pad(state[v], ((0, 0), (PAD, PAD), (0, 0)), mode="reflect") for v in fields}}
    for key in ("rho_bar", "rho_mean"):
        if key in state:
            out[key] = np.pad(state[key], ((PAD, PAD), (0, 0)), mode="reflect")
    return out


def residuals(F, laws, fields=VARS):
    dt = np.diff(F["t"])[:, None, None]
    atoms = PolarAtoms(F, fields)
    return {v: (F[v][1:] - F[v][:-1]) / dt - law_rhs(F, v, laws[v], fields, atoms)[:-1] for v in fields}


def simulate(F, laws, E, rng, fields=VARS, wall="onesided"):
    """Stepped fields from the recorded start, or None once a field exceeds FS.DIVERGED. wall="onesided":
    the fit's own operator, one-sided radial differences at the first and the last ring (the density's steep
    drop at the wall is kept); wall="mirrored": fields mirrored beyond those rings (zero radial gradient
    there, which erases the gradient at the wall)."""
    dt = float(np.median(np.diff(F["t"])))
    state = {"r": F["r"], "rho_bar": F["rho_bar"], **{v: F[v][:1].copy() for v in fields}}
    out = {v: [state[v][0].copy()] for v in fields}
    nE = len(E[fields[0]])
    for step in range(len(F["t"]) - 1):
        k = rng.integers(nE)
        if wall == "mirrored":
            padded = mirrored(state, fields)
            atoms = PolarAtoms(padded, fields)
            rhs = {v: law_rhs(padded, v, laws[v], fields, atoms)[:, PAD:-PAD, :] for v in fields}
        else:
            atoms = PolarAtoms(state, fields)
            rhs = {v: law_rhs(state, v, laws[v], fields, atoms) for v in fields}
        for v in fields:
            state[v] = state[v] + dt * (rhs[v] + E[v][k])
            out[v].append(state[v][0].copy())
        if step % 10 == 0 and not all(np.all(np.abs(state[v]) < FS.DIVERGED) for v in fields):
            return None
    return {v: np.array(a) for v, a in out.items()}


def stats(fl, r):
    """Area-weighted (by r) autocorrelations, density structure function and amplitudes, as
    swarm_field_sim.stats, plus radial profiles (means over time and angle)."""
    wgt = (r / r.mean())[None, :, None]
    P, W, rho = (fl["Pr"], fl["Pt"]), (fl["Wr"], fl["Wt"]), fl["rho"]
    T = len(rho)

    def acf(V):
        den = np.mean(wgt * (V[0] ** 2 + V[1] ** 2))
        return [float(np.mean(wgt[:, :, :] * (V[0][L:] * V[0][:T - L] + V[1][L:] * V[1][:T - L])) / den) for L in FS.CP_LAGS]

    d = rho - rho.mean(0)
    return dict(C_P=acf(P), C_W=acf(W),
                S_rho=[float(np.mean(wgt * (rho[L:] - rho[:-L]) ** 2)) for L in FS.S_LAGS if L < T],
                rms_rho=float(np.sqrt(np.mean(wgt * d ** 2))), rms_P=float(np.sqrt(np.mean(wgt * (P[0] ** 2 + P[1] ** 2)))),
                rms_W=float(np.sqrt(np.mean(wgt * (W[0] ** 2 + W[1] ** 2)))),
                rho_r=rho.mean((0, 2)).tolist(), Pr_r=P[0].mean((0, 2)).tolist(), Pt_r=P[1].mean((0, 2)).tolist(),
                Wt_r=W[1].mean((0, 2)).tolist())


def experiment_04(path, P):
    """The experiment's polar fields on the agent simulations' clock (0.4 s, from 9.6 s)."""
    (t, x, y, phi, xc, yc, s), r, th = grid_for(path)
    samp = np.arange(0, len(t), SS.LAG)[SS.HIST:]
    return coarse_polar((x[samp] - xc) / s, (y[samp] - yc) / s, phi[samp], P["theta"], P["w"], P["r"], P["th"])


def agents_04(path, P, suffix, name):
    z = np.load(os.path.join(HERE, "results", "viz", f"{swarm_meta(path)['tag']}{suffix}.npz"))
    i = [str(n) for n in z["names"]].index(name)
    pos, phi = z[f"sim{i}_pos"], z[f"sim{i}_phi"]
    s = SS.px_per_cm()
    return coarse_polar((pos[..., 0] - z["meta"][0]) / s, (pos[..., 1] - z["meta"][1]) / s, phi,
                        P["theta"], P["w"], P["r"], P["th"])


def sim_main(a):
    fits = json.load(open(os.path.join(HERE, "results", "swarm_fields_polar.json")))
    tags = [t for t in a.tags.split(",") if t]
    for path in swarm_files():
        tag = swarm_meta(path)["tag"]
        if tags and tag not in tags:
            continue
        G = fields_polar(path)
        eqs = fits[tag]["eqs"]
        r = G["r"]
        res = dict(r=r.tolist(), sources={"experiment": stats(experiment_04(path, G), r)})
        for label, (suffix, name) in FS.AGENT_MODELS.items():
            res["sources"][f"agents: {label}"] = stats(agents_04(path, G, suffix, name), r)
        bal = {v: SF.balanced(eqs[v]["cands"], eqs[v]["front"]) for v in VARS}
        base = {v: eqs[v]["cands"][bal[v]]["terms"] for v in VARS}

        def run_laws(laws):
            # zero-mean noise: each residual field's time mean (the law's systematic misfit, e.g. the
            # circulation W_t(r) that rises toward the wall) is removed, so every steady structure the
            # simulation shows has to come from the law itself
            E = {v: e - e.mean(0, keepdims=True) for v, e in residuals(G, laws).items()}
            runs = [simulate(G, laws, E, np.random.default_rng(s)) for s in range(a.seeds)]
            ok = [stats(x, r) for x in runs if x is not None]
            return (FS.average(ok) if len(ok) == a.seeds else None), len(ok) < a.seeds

        t0 = time.time()
        base_stats, base_div = run_laws(base)
        res["base"] = dict(laws=base, diverged=base_div, stats=base_stats)
        res["fronts"] = {}
        for v in VARS:
            res["fronts"][v] = []
            for i in sorted(eqs[v]["front"], key=lambda i: eqs[v]["cands"][i]["win14"]):
                c = eqs[v]["cands"][i]
                st, div = (base_stats, base_div) if i == bal[v] else run_laws(dict(base, **{v: c["terms"]}))
                res["fronts"][v].append(dict(terms=c["terms"], wape=c["wape"], win14=c["win14"], r2=c["r2"],
                                             balanced=bool(i == bal[v]), diverged=div, stats=st))
                print(f"{tag} {v}: {sorted(c['terms'])} -> {'diverges' if div else 'ok'}", flush=True)
        print(f"{tag}: simulated in {time.time() - t0:.0f}s", flush=True)
        json.dump({tag: res}, open(os.path.join(HERE, "results", f"swarm_field_sim_polar_{tag}.json"), "w"), indent=1)


# ------------------------------------------------------------------ hypothesis systems
# Every law of a hypothesis is a member of its equation's non-dominated (WAPE, win14) front, as the search
# found it: a hypothesis names a property, and the law is the most stable front member that has it. A
# hypothesis whose property no front member has is not supported in that recording (and not simulated).
DRIVE = {"lap_rho", "rho_r", "rho_t", "r", "inv_r", "one", "rho_bar"}   # density shape / position, additive
CORE = {"Pr": dict(need={"Pr", "Wt"}), "Pt": dict(need={"Pt", "Wr"}),    # decay + rotation coupling
        "Wr": dict(need={"Wr", "Pt"}), "Wt": dict(need={"Wt", "Pr"})}
HYPOTHESES = {
    "H1": ("transport + rotating polarization",
           dict(rho=dict(need={"Pr_r"}, forbid={"rho", "rho_bar"}), **CORE)),
    "H2": ("H1 + density relaxation",
           dict(rho=dict(need={"Pr_r", "rho", "rho_bar"}), **CORE)),
    "H3": ("H2 + circulation driven by the density's shape or the distance from the centre",
           dict(rho=dict(need={"Pr_r", "rho", "rho_bar"}), **dict(CORE, Wt=dict(need={"Wt", "Pr"}, any=DRIVE)))),
}


def front_pick(eq, need=(), forbid=(), any=None):
    """The most stable front member whose standalone terms include every `need`, none of `forbid`, and
    (if given) at least one of `any`; None if no front member qualifies."""
    for i in sorted(eq["front"], key=lambda i: eq["cands"][i]["win14"]):
        terms = set(eq["cands"][i]["terms"])
        if set(need) <= terms and not terms & set(forbid) and (any is None or terms & set(any)):
            return i
    return None


def hypotheses_main(a):
    """Several hypothesis systems (HYPOTHESES), every law a front member, simulated with zero-mean noise."""
    fits = json.load(open(os.path.join(HERE, "results", "swarm_fields_polar.json")))
    path_json = os.path.join(HERE, "results", "swarm_field_hypotheses_polar.json")
    out = {}
    tags = [t for t in a.tags.split(",") if t]
    for path in swarm_files():
        tag = swarm_meta(path)["tag"]
        if tags and tag not in tags:
            continue
        G = fields_polar(path)
        r = G["r"]
        eqs = fits[tag]["eqs"]
        rec = dict(r=r.tolist(), sources={"experiment": stats(experiment_04(path, G), r)}, hypotheses={})
        for label, (suffix, name) in FS.AGENT_MODELS.items():
            rec["sources"][f"agents: {label}"] = stats(agents_04(path, G, suffix, name), r)
        for key, (desc, rule) in HYPOTHESES.items():
            picks = {v: front_pick(eqs[v], **rule[v]) for v in VARS}
            missing = [v for v, i in picks.items() if i is None]
            h = dict(description=desc, supported=not missing, missing=missing, laws={}, scores={})
            for v, i in picks.items():
                if i is not None:
                    c = eqs[v]["cands"][i]
                    order = sorted(eqs[v]["front"], key=lambda j: eqs[v]["cands"][j]["win14"])
                    h["laws"][v] = c["terms"]
                    h["scores"][v] = dict(wape=c["wape"], win14=c["win14"], r2=c["r2"], position=order.index(i) + 1,
                                          front=len(order))
            if not missing:
                E = {v: e - e.mean(0, keepdims=True) for v, e in residuals(G, h["laws"]).items()}
                runs = [simulate(G, h["laws"], E, np.random.default_rng(s)) for s in range(a.seeds)]
                ok = [stats(x, r) for x in runs if x is not None]
                h.update(diverged=len(ok) < a.seeds, stats=FS.average(ok) if len(ok) == a.seeds else None)
            rec["hypotheses"][key] = h
            st = h.get("stats")
            print(f"{tag} {key}: " + (f"not supported (no front law for {missing})" if missing else
                  "diverges" if st is None else
                  f"W_t at wall {st['Wt_r'][-1]:+.3f} (exp {rec['sources']['experiment']['Wt_r'][-1]:+.3f}), "
                  f"rms rho {st['rms_rho']:.4f} (exp {rec['sources']['experiment']['rms_rho']:.4f})")
                  + (f" | W_t law {h['laws']['Wt']}" if key == "H3" and not missing else ""), flush=True)
        out[tag] = rec
    json.dump(out, open(path_json, "w"), indent=1)


def merge():
    out = {}
    for path in swarm_files():
        p = os.path.join(HERE, "results", f"swarm_field_sim_polar_{swarm_meta(path)['tag']}.json")
        if os.path.exists(p):
            out.update(json.load(open(p)))
    json.dump(out, open(os.path.join(HERE, "results", "swarm_field_sim_polar.json"), "w"), indent=1)
    print(f"merged {len(out)} recordings")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", action="store_true")
    ap.add_argument("--sim", action="store_true")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--hypotheses", action="store_true", help="simulate the hypothesis systems (HYPOTHESES)")
    ap.add_argument("--tags", default="")
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--cmax", type=int, default=6)
    ap.add_argument("--seeds", type=int, default=4)
    a = ap.parse_args()
    if a.fit:
        fit_main(a)
    if a.sim:
        sim_main(a)
    if a.merge:
        merge()
    if a.hypotheses:
        hypotheses_main(a)


if __name__ == "__main__":
    main()
