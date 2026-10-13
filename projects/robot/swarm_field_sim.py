r"""The swarm simulated with its field equations, compared with the experiment and the agent simulations.

Field model (swarm_fields.py): per recording the balanced law of each field (rho, Px, Py, Wx, Wy),
stepped explicitly at the fitted step dt = 0.08 s on the field grid,
    f(t + dt) = f(t) + dt (law_f(t) + e_f),
from the recorded fields at the start. e = the laws' residual fields of one recorded instant drawn at
random each step (all five fields of the same instant: their spatial and cross-field correlations
kept, white in time). Spatial derivatives of the stepped fields are taken with the fields mirrored
across the grid's edges (zero normal gradient). Variants:
  field laws                        the balanced laws
  field laws + density relaxation   the density law refit as -v0x Px_x - v0y Py_y + D lap rho
                                    - kappa (rho - <rho>(r)): isotropic D, and relaxation toward each
                                    point's mean density (confinement keeps the packing bounded)
Compared with the recorded fields and with the fields of the agent simulations (seed 0 of
swarm_spin_contact.py: per-robot EPDE laws and the common system, contact law, cap core), all
coarse-grained the same way (swarm_fields.build_fields) and sampled every 0.4 s:
  C_P(tau), C_W(tau)    autocorrelation of the polarization / spin-weighted polarization, 0..10 s
  S_rho(tau)            density structure function <(rho(t + tau) - rho(t))^2>, 0.4..100 s
  rms                   of the density fluctuation (about each point's time mean), |P|, |W|

--tau 5: the laws of swarm_fields.py --tau 5 (fitted on 0.4 s increments), stepped every 0.4 s with the
residual fields of the 0.4 s increments.
--fronts: no law is selected: every member of every equation's front is simulated, one equation at a time,
with the other equations at their balanced members (-> results/swarm_field_sim[_tau5]_fronts[_<tag>].json;
a law whose stepped map grows is reported as diverged).
--combos [--chunk i/n]: every combination of one front law per equation, each distinct system once
(-> results/combos/..._combos_<tag>_<i>of<n>.json; --combos --merge -> results/swarm_field_sim[_tau5]_combos.json).
--isotropic: the isotropy rule instead of every combination. The arena has no preferred direction, so a
vector field's y law must be its x law turned by 90 degrees (P_x <-> P_y, W_x <-> W_y, d/dx <-> d/dy, same
terms) and the density law must be unchanged by the turn; every system built from such laws is simulated
(-> results/swarm_field_sim[_tau5]_iso_<tag>.json; --isotropic --merge -> ..._iso.json).
--sample N [--chunk i/n]: the check of that rule: N random combinations that break it (fixed seed)
(-> results/combos/..._sample_<tag>_<i>of<n>.json; --sample N --merge -> ..._sample.json).

python swarm_field_sim.py [--seeds 4] [--tags ...] [--tau 5]
        -> results/swarm_field_sim[_tau5]_<tag>.json (merged into results/swarm_field_sim[_tau5].json;
           --merge merges only)
"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import swarm_sim as SS                                           # noqa: E402
import swarm_fields as SF                                        # noqa: E402
from swarm_data import swarm_files, swarm_meta, arena_envelope, load_swarm  # noqa: E402

SUB_STEP = 5                       # 0.08 s fields -> 0.4 s, the agent simulations' clock
CP_LAGS = list(range(26))          # 0 .. 10 s in 0.4 s steps
S_LAGS = [1, 2, 5, 10, 25, 50, 100, 250]
AGENT_MODELS = {"per-robot EPDE laws": ("", "contact law, cap core"),
                "common system": ("_own_prop", "contact law, cap core")}


# ------------------------------------------------------------------ laws
class StepAtoms:
    """The library atoms of every law at one state, each (operation, field) computed on first use and
    shared by all the laws (the same arithmetic as swarm_fields.atoms_for). view(var) names them the way
    var's library does, plus lap_self (the isotropic Laplacian of the target) and delta_self (its
    deviation from the local time mean)."""

    def __init__(self, state, fields):
        self.s, self.fields, self.cache = state, fields, {}

    def value(self, op, v):
        if (op, v) not in self.cache:
            s = self.s
            if op == "id":
                val = s[v]
            elif op == "dx":
                val = SF.spatial(s, s[v], 1)
            elif op == "dy":
                val = SF.spatial(s, s[v], 2)
            elif op == "dxx":
                val = SF.spatial(s, self.value("dx", v), 1)
            elif op == "dyy":
                val = SF.spatial(s, self.value("dy", v), 2)
            elif op == "lap":
                val = self.value("dxx", v) + self.value("dyy", v)
            elif op == "delta":
                val = s[v] - s[f"{v}_mean"] if f"{v}_mean" in s else None
            else:                                                              # "bar"
                val = np.broadcast_to(s["rho_bar"], s["rho"].shape) if "rho_bar" in s else None
            self.cache[(op, v)] = val
        return self.cache[(op, v)]

    def view(self, var):
        return AtomView(self, var)


class AtomView:
    """StepAtoms seen from one target var: atom name -> array (None if the name is not an atom)."""

    def __init__(self, atoms, var):
        self.atoms = atoms
        self.names = {var: ("id", var), f"d{var}/dx1": ("dx", var), f"d{var}/dx2": ("dy", var),
                      f"d^2{var}/dx1^2": ("dxx", var), f"d^2{var}/dx2^2": ("dyy", var),
                      "lap_self": ("lap", var), "delta_self": ("delta", var), "rho_bar": ("bar", None)}
        for v in atoms.fields:
            if v != var:
                self.names.update({v: ("id", v), f"{v}_x": ("dx", v), f"{v}_y": ("dy", v), f"lap_{v}": ("lap", v)})

    def get(self, name):
        key = self.names.get(name)
        return None if key is None else self.atoms.value(*key)


def term_array(atoms, name):
    out = 1.0
    for f in name.split("*"):
        a = atoms.get(f)
        if a is not None:
            out = out * a
        else:
            base, p = f.rsplit("^", 1)
            out = out * atoms.get(base) ** int(p)
    return out


def law_rhs(state, var, terms, fields=SF.VARS, atoms=None):
    view = (atoms or StepAtoms(state, fields)).view(var)
    return sum(c * term_array(view, n) for n, c in terms.items())


def fit_relaxation(F, fields=SF.VARS):
    """rho_t = -v0x Px_x - v0y Py_y + D lap rho - kappa (rho - <rho>(r)): OLS on the forward increments
    of the inner domain. With several polarization fields (P<b>x, P<b>y) each gets its own v0."""
    names = [f"{v}_{v[-1]}" for v in fields if v.startswith("P")] + ["lap_self", "delta_self"]
    G = dict(F, rho_mean=F["rho"].mean(0))
    atoms = dict(SF.atoms_for(G, "rho", fields))
    atoms["lap_self"] = atoms["d^2rho/dx1^2"] + atoms["d^2rho/dx2^2"]
    atoms["delta_self"] = G["rho"] - G["rho_mean"]
    y = (F["rho"][1:] - F["rho"][:-1]) / np.diff(F["t"])[:, None, None]
    y = np.concatenate([y, np.full((1,) + y.shape[1:], np.nan)])
    A = np.column_stack([SF.inner(atoms[n]).ravel() for n in names])
    coef = np.linalg.lstsq(A, SF.inner(y).ravel(), rcond=None)[0]
    return dict(zip(names, coef.tolist()))


PAD = 2


def mirrored(state, fields):
    """The state with the fields mirrored across the grid's edges (zero normal gradient there), so the
    one-sided edge differences cannot amplify; the stepped model's boundary condition."""
    g, h = state["g"], state["g"][1] - state["g"][0]
    gp = np.concatenate([g[0] - h * np.arange(PAD, 0, -1), g, g[-1] + h * np.arange(1, PAD + 1)])
    padded = {"g": gp, **{v: np.pad(state[v], ((0, 0), (PAD, PAD), (PAD, PAD)), mode="reflect") for v in fields}}
    for key in ("rho_mean", "rho_bar"):                                        # static maps
        if key in state:
            padded[key] = np.pad(state[key], ((PAD, PAD), (PAD, PAD)), mode="reflect")
    return padded


def residuals(F, laws, fields=SF.VARS):
    """e_f(t) = (f(t + dt) - f(t)) / dt - law_f(t), every field, full grid, t = 0 .. T-2."""
    dt = F["t"][1:] - F["t"][:-1]
    G = dict(F, rho_mean=F["rho"].mean(0))
    atoms = StepAtoms(G, fields)
    return {v: (F[v][1:] - F[v][:-1]) / dt[:, None, None] - law_rhs(G, v, laws[v], fields, atoms)[:-1]
            for v in fields}


DIVERGED = 100.0                   # |field| beyond this (fields are O(0.1-1)): the stepped law has blown up


def simulate(F, laws, E, rng, fields=SF.VARS):
    """Stepped fields, or None once any field exceeds DIVERGED (a law whose stepped map grows)."""
    dt = float(np.median(np.diff(F["t"])))
    state = {"g": F["g"], "rho_mean": F["rho"].mean(0),
             **{v: F[v][:1].copy() for v in fields}}                          # (1, x, y): the atoms' axis layout
    if "rho_bar" in F:
        state["rho_bar"] = F["rho_bar"]
    out = {v: [state[v][0].copy()] for v in fields}
    nE = len(E["rho"])
    for step in range(len(F["t"]) - 1):
        k = rng.integers(nE)
        padded = mirrored(state, fields)
        atoms = StepAtoms(padded, fields)
        rhs = {v: law_rhs(padded, v, laws[v], fields, atoms)[:, PAD:-PAD, PAD:-PAD] for v in fields}
        for v in fields:
            state[v] = state[v] + dt * (rhs[v] + E[v][k])
            out[v].append(state[v][0].copy())
        if step % 10 == 0 and not all(np.all(np.abs(state[v]) < DIVERGED) for v in fields):
            return None
    return {v: np.array(a) for v, a in out.items()}


def front_combinations(eqs, fields):
    """Every combination of one front law per equation ({field: candidate index}), each distinct system
    once: combinations with identical laws in every equation are duplicates and are dropped."""
    import itertools
    seen, out = set(), []
    for combo in itertools.product(*[sorted(eqs[v]["front"]) for v in fields]):
        key = json.dumps([sorted(eqs[v]["cands"][i]["terms"].items()) for v, i in zip(fields, combo)])
        if key not in seen:
            seen.add(key)
            out.append(dict(zip(fields, combo)))
    return out


# ------------------------------------------------------------------ isotropy
_SWAP = {"x": "y", "y": "x", "1": "2", "2": "1"}


def mirror_atom(a):
    """An atom turned by 90 degrees, structure only (signs aside): vector components and derivative axes
    swapped (P_x <-> P_y, W_x <-> W_y, d/dx <-> d/dy); scalars (rho, rho_bar, Laplacians) unchanged."""
    if a in ("rho", "rho_bar", "lap_self", "delta_self") or a.startswith("lap_"):
        return re.sub(r"(P\d?|W)([xy])", lambda m: m.group(1) + _SWAP[m.group(2)], a)
    a = re.sub(r"(P\d?|W)([xy])", lambda m: m.group(1) + _SWAP[m.group(2)], a)
    a = re.sub(r"dx([12])", lambda m: "dx" + _SWAP[m.group(1)], a)
    return re.sub(r"_([xy])$", lambda m: "_" + _SWAP[m.group(1)], a)


def term_set(terms, turn=False):
    return frozenset(frozenset((mirror_atom(f) if turn else f) for f in t.split("*")) for t in terms)


def isotropic_combinations(eqs, pairs):
    """The isotropy rule: the arena has no preferred direction, so a vector field's y law must be its x
    law turned by 90 degrees, and a scalar field's law must be unchanged by the turn. For each pair
    (field_x, field_y) every front law of field_x whose turned form is on field_y's front; for each other
    field its front laws equal to their turned form. Returns ([{field: candidate index}], per pair the
    matched (x index, y index) list)."""
    import itertools
    in_pair = {v for p in pairs for v in p}
    matched = {}
    for vx, vy in pairs:
        ykeys = {term_set(eqs[vy]["cands"][j]["terms"]): j for j in eqs[vy]["front"]}
        matched[(vx, vy)] = [(i, ykeys[term_set(eqs[vx]["cands"][i]["terms"], turn=True)])
                             for i in sorted(eqs[vx]["front"])
                             if term_set(eqs[vx]["cands"][i]["terms"], turn=True) in ykeys]
    scalars = {v: [i for i in sorted(e["front"]) if term_set(e["cands"][i]["terms"]) ==
                   term_set(e["cands"][i]["terms"], turn=True)] for v, e in eqs.items() if v not in in_pair}
    choices = [[{p[0]: i, p[1]: j} for i, j in matched[p]] for p in pairs] + \
              [[{v: i} for i in idx] for v, idx in scalars.items()]
    combos = [dict(kv for part in parts for kv in part.items()) for parts in itertools.product(*choices)]
    return combos, {f"{vx}/{vy}": m for (vx, vy), m in matched.items()}


def simulate_combinations(G, eqs, fields, seeds, sub, chunk=(0, 1), totals=lambda fl: fl, combos=None):
    """Every combination of front laws (front_combinations), or the given combos; this process's share
    chunk = (i, n): every n-th combination from the i-th. Each equation's residual field depends only on
    its own law, so it is computed once per law and reused.
    Returns [{members: {field: candidate index}, diverged, stats}]."""
    combos = (front_combinations(eqs, fields) if combos is None else combos)[chunk[0]::chunk[1]]
    dt = G["t"][1:] - G["t"][:-1]
    Gm = dict(G, rho_mean=G["rho"].mean(0))
    atoms = StepAtoms(Gm, fields)
    cache = {}

    def resid(v, i):
        if (v, i) not in cache:
            cache[(v, i)] = (G[v][1:] - G[v][:-1]) / dt[:, None, None] - \
                law_rhs(Gm, v, eqs[v]["cands"][i]["terms"], fields, atoms)[:-1]
        return cache[(v, i)]

    out = []
    for n, combo in enumerate(combos):
        laws = {v: eqs[v]["cands"][i]["terms"] for v, i in combo.items()}
        E = {v: resid(v, i) for v, i in combo.items()}
        runs = [simulate(G, laws, E, np.random.default_rng(s), fields) for s in range(seeds)]
        ok = [stats(totals(sub(r))) for r in runs if r is not None]
        st = average(ok) if len(ok) == seeds else None
        out.append(dict(members=combo, diverged=st is None,
                        stats=None if st is None else {k: st[k] for k in ("C_P", "C_W", "S_rho", "rms_rho", "rms_P", "rms_W")}))
        if n % 200 == 0:
            print(f"   combination {n + 1}/{len(combos)}", flush=True)
    return out


def simulate_fronts(G, eqs, fields, seeds, sub, totals=lambda fl: fl):
    """No law selected: every member of every equation's front, one equation at a time, simulated with
    the other equations at their balanced members (the base run, all balanced, is run once and reused
    for each equation's balanced member). Members are ordered from most stable to most precise."""
    bal = {v: SF.balanced(eqs[v]["cands"], eqs[v]["front"]) for v in fields}
    base = {v: eqs[v]["cands"][bal[v]]["terms"] for v in fields}

    def run(laws):
        E = residuals(G, laws, fields)
        runs = [simulate(G, laws, E, np.random.default_rng(s), fields) for s in range(seeds)]
        ok = [stats(totals(sub(r))) for r in runs if r is not None]
        return (average(ok) if len(ok) == seeds else None), len(ok) < seeds

    base_stats, base_div = run(base)
    fronts = {}
    for v in fields:
        fronts[v] = []
        for i in sorted(eqs[v]["front"], key=lambda i: eqs[v]["cands"][i]["win14"]):
            c = eqs[v]["cands"][i]
            st, div = (base_stats, base_div) if i == bal[v] else run(dict(base, **{v: c["terms"]}))
            fronts[v].append(dict(terms=c["terms"], wape=c["wape"], win14=c["win14"], r2=c["r2"],
                                  balanced=bool(i == bal[v]), diverged=div, stats=st))
            print(f"   {v}: {sorted(c['terms'])} -> {'diverges' if div else 'ok'}", flush=True)
    return dict(base=dict(laws=base, diverged=base_div, stats=base_stats), fronts=fronts)


# ------------------------------------------------------------------ agent simulations as fields
def agent_fields(path, F, suffix, name):
    """Fields of a saved agent simulation (positions every 0.4 s), the experiment's kernel and grid."""
    z = np.load(os.path.join(HERE, "results", "viz", f"{swarm_meta(path)['tag']}{suffix}.npz"))
    i = [str(n) for n in z["names"]].index(name)
    pos, phi = z[f"sim{i}_pos"], z[f"sim{i}_phi"]
    xc, yc = z["meta"][0], z["meta"][1]
    s = SS.px_per_cm()
    X, Y = (pos[..., 0] - xc) / s, (pos[..., 1] - yc) / s
    return coarse(F, X, Y, phi)


def coarse(F, X, Y, phi):
    g, ell = F["g"], SS.CAP_CM
    norm = np.pi * (SS.CAP_CM / 2) ** 2 / (2 * np.pi * ell ** 2)
    gx = np.exp(-(g[None, None, :] - X[:, :, None]) ** 2 / (2 * ell ** 2))
    gy = np.exp(-(g[None, None, :] - Y[:, :, None]) ** 2 / (2 * ell ** 2))
    px, py = np.cos(phi + F["theta"]), np.sin(phi + F["theta"])
    field = lambda wt: norm * np.einsum("ti,tix,tiy->txy", wt, gx, gy)     # noqa: E731
    w = F["w"]
    return dict(rho=field(np.ones_like(px)), Px=field(px), Py=field(py), Wx=field(w * px), Wy=field(w * py))


def experiment_fields_04(path, F):
    """The experiment's fields on the agent simulations' clock (0.4 s), from the same start (9.6 s)."""
    t, x, y, phi, ids = load_swarm(path)
    xc, yc, r_env = arena_envelope(x, y)
    s = SS.px_per_cm()
    samp = np.arange(0, len(t), SS.LAG)[SS.HIST:]
    return coarse(F, (x[samp] - xc) / s, (y[samp] - yc) / s, phi[samp])


# ------------------------------------------------------------------ statistics
def stats(fl):
    P = np.stack([fl["Px"], fl["Py"]], -1)
    W = np.stack([fl["Wx"], fl["Wy"]], -1)
    T = len(P)

    def acf(V):
        den = np.mean(np.sum(V * V, -1))
        return [float(np.mean(np.sum(V[L:] * V[:T - L], -1)) / den) for L in CP_LAGS]

    rho = fl["rho"]
    d = rho - rho.mean(0)
    return dict(C_P=acf(P), C_W=acf(W),
                S_rho=[float(np.mean((rho[L:] - rho[:-L]) ** 2)) for L in S_LAGS if L < T],
                rms_rho=float(np.sqrt(np.mean(d ** 2))), rms_P=float(np.sqrt(np.mean(np.sum(P * P, -1)))),
                rms_W=float(np.sqrt(np.mean(np.sum(W * W, -1)))), mean_rho=float(rho.mean()))


def average(runs):
    keys = runs[0].keys()
    return {k: (np.mean([r[k] for r in runs], 0).tolist() if isinstance(runs[0][k], list)
                else float(np.mean([r[k] for r in runs]))) for k in keys}


def part_path(tag, tau=1, extra=""):
    return os.path.join(HERE, "results", f"swarm_field_sim{SF.suffix(tau)}{extra}_{tag}.json")


def merge(tau=1, extra=""):
    out = {}
    for f in swarm_files():
        p = part_path(swarm_meta(f)["tag"], tau, extra)
        if os.path.exists(p):
            out.update(json.load(open(p)))
    json.dump(out, open(os.path.join(HERE, "results", f"swarm_field_sim{SF.suffix(tau)}{extra}.json"), "w"), indent=1)
    print(f"merged {len(out)} recordings")


def combos_path(tag, tau, chunk):
    i, n = chunk
    return os.path.join(HERE, "results", "combos", f"swarm_field_sim{SF.suffix(tau)}_combos_{tag}_{i}of{n}.json")


def front_table(eqs, fields):
    """Each equation's front (most stable first): candidate index, law, WAPE, instability, held-out R2."""
    return {v: [dict(i=int(i), terms=eqs[v]["cands"][i]["terms"], wape=eqs[v]["cands"][i]["wape"],
                     win14=eqs[v]["cands"][i]["win14"], r2=eqs[v]["cands"][i]["r2"])
                for i in sorted(eqs[v]["front"], key=lambda i: eqs[v]["cands"][i]["win14"])] for v in fields}


def merge_combos(path_of, out_path):
    """Join the chunk files of every recording: {tag: {fronts, experiment, combos}}."""
    import glob
    out = {}
    for p in sorted(glob.glob(path_of)):
        for tag, part in json.load(open(p)).items():
            rec = out.setdefault(tag, dict({k: v for k, v in part.items() if k != "combos"}, combos=[]))
            rec["combos"] += part["combos"]
    json.dump(out, open(out_path, "w"))
    print(f"merged combinations: " + ", ".join(f"{t} {len(r['combos'])}" for t, r in out.items()))


PAIRS = [("Px", "Py"), ("Wx", "Wy")]


def iso_path(tag, tau):
    return os.path.join(HERE, "results", f"swarm_field_sim{SF.suffix(tau)}_iso_{tag}.json")


def iso_main(a):
    """Every isotropic system (isotropic_combinations) of the two-field laws, simulated as found."""
    tau = a.tau
    if a.merge:
        return merge_combos(os.path.join(HERE, "results", f"swarm_field_sim{SF.suffix(tau)}_iso_N*.json"),
                            os.path.join(HERE, "results", f"swarm_field_sim{SF.suffix(tau)}_iso.json"))
    tags = [t for t in a.tags.split(",") if t]
    fits = json.load(open(os.path.join(HERE, "results", f"swarm_fields{SF.suffix(tau)}.json")))
    step = max(1, SUB_STEP // tau)
    sub = lambda fl: {v: fl[v][::step] for v in SF.VARS}                         # noqa: E731
    for path in swarm_files():
        tag = swarm_meta(path)["tag"]
        if tags and tag not in tags:
            continue
        F = SF.build_fields(path)
        G = SF.coarse_time(F, tau) if tau > 1 else F
        eqs = fits[tag]["eqs"]
        combos, matched = isotropic_combinations(eqs, PAIRS)
        print(f"### {tag}: {len(combos)} isotropic systems", flush=True)
        res = dict(fronts=front_table(eqs, SF.VARS), experiment=stats(experiment_fields_04(path, F)), pairs=matched,
                   combos=simulate_combinations(G, eqs, SF.VARS, a.seeds, sub, combos=combos))
        json.dump({tag: res}, open(iso_path(tag, tau), "w"))


def sample_main(a):
    """The check of the isotropy rule: a uniform random sample (fixed seed) of the combinations that break
    it (every combination minus the isotropic ones), simulated the same way; this process's share --chunk."""
    tau, chunk = a.tau, tuple(int(x) for x in a.chunk.split("/"))
    if a.merge:
        return merge_combos(os.path.join(HERE, "results", "combos", f"swarm_field_sim{SF.suffix(tau)}_sample_N*.json"),
                            os.path.join(HERE, "results", f"swarm_field_sim{SF.suffix(tau)}_sample.json"))
    os.makedirs(os.path.join(HERE, "results", "combos"), exist_ok=True)
    tags = [t for t in a.tags.split(",") if t]
    fits = json.load(open(os.path.join(HERE, "results", f"swarm_fields{SF.suffix(tau)}.json")))
    step = max(1, SUB_STEP // tau)
    sub = lambda fl: {v: fl[v][::step] for v in SF.VARS}                         # noqa: E731
    for path in swarm_files():
        tag = swarm_meta(path)["tag"]
        if tags and tag not in tags:
            continue
        eqs = fits[tag]["eqs"]
        iso = {json.dumps(c, sort_keys=True) for c in isotropic_combinations(eqs, PAIRS)[0]}
        others = [c for c in front_combinations(eqs, SF.VARS) if json.dumps(c, sort_keys=True) not in iso]
        pick = np.random.default_rng(0).choice(len(others), min(a.sample, len(others)), replace=False)
        combos = [others[i] for i in sorted(pick)]
        F = SF.build_fields(path)
        G = SF.coarse_time(F, tau) if tau > 1 else F
        print(f"### {tag}: {len(combos)} of {len(others)} non-isotropic combinations, chunk {chunk}", flush=True)
        res = dict(fronts=front_table(eqs, SF.VARS), experiment=stats(experiment_fields_04(path, F)),
                   n_population=len(others), combos=simulate_combinations(G, eqs, SF.VARS, a.seeds, sub, chunk, combos=combos))
        path_out = os.path.join(HERE, "results", "combos",
                                f"swarm_field_sim{SF.suffix(tau)}_sample_{tag}_{chunk[0]}of{chunk[1]}.json")
        json.dump({tag: res}, open(path_out, "w"))


def combos_main(a):
    tau, chunk = a.tau, tuple(int(x) for x in a.chunk.split("/"))
    if a.merge:
        return merge_combos(os.path.join(HERE, "results", "combos", f"swarm_field_sim{SF.suffix(tau)}_combos_N*.json"),
                            os.path.join(HERE, "results", f"swarm_field_sim{SF.suffix(tau)}_combos.json"))
    os.makedirs(os.path.join(HERE, "results", "combos"), exist_ok=True)
    tags = [t for t in a.tags.split(",") if t]
    fits = json.load(open(os.path.join(HERE, "results", f"swarm_fields{SF.suffix(tau)}.json")))
    step = max(1, SUB_STEP // tau)
    sub = lambda fl: {v: fl[v][::step] for v in SF.VARS}                         # noqa: E731
    for path in swarm_files():
        tag = swarm_meta(path)["tag"]
        if tags and tag not in tags:
            continue
        F = SF.build_fields(path)
        G = SF.coarse_time(F, tau) if tau > 1 else F
        eqs = fits[tag]["eqs"]
        print(f"### {tag} chunk {chunk[0]}/{chunk[1]}", flush=True)
        res = dict(fronts=front_table(eqs, SF.VARS), experiment=stats(experiment_fields_04(path, F)),
                   combos=simulate_combinations(G, eqs, SF.VARS, a.seeds, sub, chunk))
        json.dump({tag: res}, open(combos_path(tag, tau, chunk), "w"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--tags", default="")
    ap.add_argument("--tau", type=int, default=1, help="laws fitted and stepped every tau field steps (5: 0.4 s)")
    ap.add_argument("--fronts", action="store_true",
                    help="no law selected: every front member of every equation (simulate_fronts)")
    ap.add_argument("--combos", action="store_true",
                    help="every combination of front laws, duplicates dropped (simulate_combinations)")
    ap.add_argument("--chunk", default="0/1", help="with --combos: this process's share i/n")
    ap.add_argument("--isotropic", action="store_true",
                    help="every isotropic system (x law = y law turned by 90 degrees, scalar laws unchanged)")
    ap.add_argument("--sample", type=int, default=0,
                    help="check of the isotropy rule: this many random non-isotropic combinations (with --chunk)")
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    extra = "_fronts" if a.fronts else ""
    if a.sample:
        return sample_main(a)
    if a.isotropic:
        return iso_main(a)
    if a.combos:
        return combos_main(a)
    if a.merge:
        return merge(a.tau, extra)
    tags = [t for t in a.tags.split(",") if t]
    fits = json.load(open(os.path.join(HERE, "results", f"swarm_fields{SF.suffix(a.tau)}.json")))
    step = max(1, SUB_STEP // a.tau)                                  # simulated fields -> the 0.4 s clock
    for path in swarm_files():
        tag = swarm_meta(path)["tag"]
        if tags and tag not in tags:
            continue
        F = SF.build_fields(path)
        G = SF.coarse_time(F, a.tau) if a.tau > 1 else F               # the fields the laws were fitted on
        eqs = fits[tag]["eqs"]
        laws = {v: eqs[v]["cands"][SF.balanced(eqs[v]["cands"], eqs[v]["front"])]["terms"] for v in SF.VARS}
        laws_D = dict(laws, rho=fit_relaxation(G))
        res = dict(tau=a.tau, laws=laws, density_relaxation=laws_D["rho"], sources={})
        sub = lambda fl: {v: fl[v][::step] for v in SF.VARS}                     # noqa: E731
        res["sources"][f"experiment ({0.08 * a.tau:.2f} s fields)"] = stats(sub(G))
        res["sources"]["experiment"] = stats(experiment_fields_04(path, F))
        for label, (suffix, name) in AGENT_MODELS.items():
            res["sources"][f"agents: {label}"] = stats(agent_fields(path, F, suffix, name))
        if a.fronts:
            res = dict(tau=a.tau, sources=res["sources"], **simulate_fronts(G, eqs, SF.VARS, a.seeds, sub))
        else:
            for label, L in (("field laws", laws), ("field laws + density relaxation", laws_D)):
                E = residuals(G, L)
                runs = [stats(sub(simulate(G, L, E, np.random.default_rng(s)))) for s in range(a.seeds)]
                res["sources"][label] = average(runs)
        print(f"\n### {tag}: density relaxation law " + ", ".join(f"{k} {v:+.3g}" for k, v in laws_D["rho"].items()))
        for k, s in res["sources"].items():
            print(f"  {k:32s} rms rho {s['rms_rho']:.4f} |P| {s['rms_P']:.4f} |W| {s['rms_W']:.4f} | C_P 0.4/0.8/2/4 s "
                  + "/".join(f"{s['C_P'][L]:+.2f}" for L in (1, 2, 5, 10)) + " | C_W "
                  + "/".join(f"{s['C_W'][L]:+.2f}" for L in (1, 2, 5, 10)) + " | S_rho 0.4/4/40/100 s "
                  + "/".join(f"{v:.2e}" for v in (s["S_rho"][0], s["S_rho"][3], s["S_rho"][6], s["S_rho"][-1])),
                  flush=True)
        json.dump({tag: res}, open(part_path(tag, a.tau, extra), "w"), indent=1)
    if not tags:
        merge(a.tau, extra)


if __name__ == "__main__":
    main()
