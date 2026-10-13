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

python swarm_field_sim.py [--seeds 4] [--tags ...]  -> results/swarm_field_sim_<tag>.json (merged into
                                                       results/swarm_field_sim.json; --merge merges only)
"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import json
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
class LazyAtoms:
    """The atoms of a law's terms, each computed on first use (swarm_fields.atom), plus lap_self (the
    isotropic Laplacian of the target) and delta_self (its deviation from the local time mean)."""

    def __init__(self, state, var, fields):
        self.state, self.var, self.fields, self.cache = state, var, fields, {}

    def get(self, name):
        if name not in self.cache:
            s, v = self.state, self.var
            if name == "lap_self":
                val = self.get(f"d^2{v}/dx1^2") + self.get(f"d^2{v}/dx2^2")
            elif name == "delta_self":
                val = s[v] - s[f"{v}_mean"] if f"{v}_mean" in s else None
            else:
                val = SF.atom(s, v, name, self.fields)
            self.cache[name] = val
        return self.cache[name]


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


def law_rhs(state, var, terms, fields=SF.VARS):
    atoms = LazyAtoms(state, var, fields)
    return sum(c * term_array(atoms, n) for n, c in terms.items())


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


def law_rhs_mirrored(state, var, terms, fields=SF.VARS):
    """law_rhs with the fields mirrored across the grid's edges (zero normal gradient there), so the
    one-sided edge differences cannot amplify; the stepped model's boundary condition."""
    g, h = state["g"], state["g"][1] - state["g"][0]
    gp = np.concatenate([g[0] - h * np.arange(PAD, 0, -1), g, g[-1] + h * np.arange(1, PAD + 1)])
    padded = {"g": gp, **{v: np.pad(state[v], ((0, 0), (PAD, PAD), (PAD, PAD)), mode="reflect") for v in fields}}
    if "rho_mean" in state:
        padded["rho_mean"] = np.pad(state["rho_mean"], ((PAD, PAD), (PAD, PAD)), mode="reflect")
    return law_rhs(padded, var, terms, fields)[:, PAD:-PAD, PAD:-PAD]


def residuals(F, laws, fields=SF.VARS):
    """e_f(t) = (f(t + dt) - f(t)) / dt - law_f(t), every field, full grid, t = 0 .. T-2."""
    dt = F["t"][1:] - F["t"][:-1]
    G = dict(F, rho_mean=F["rho"].mean(0))
    return {v: (F[v][1:] - F[v][:-1]) / dt[:, None, None] - law_rhs(G, v, laws[v], fields)[:-1] for v in fields}


def simulate(F, laws, E, rng, fields=SF.VARS):
    dt = float(np.median(np.diff(F["t"])))
    state = {"g": F["g"], "rho_mean": F["rho"].mean(0),
             **{v: F[v][:1].copy() for v in fields}}                          # (1, x, y): the atoms' axis layout
    out = {v: [state[v][0].copy()] for v in fields}
    nE = len(E["rho"])
    for _ in range(len(F["t"]) - 1):
        k = rng.integers(nE)
        rhs = {v: law_rhs_mirrored(state, v, laws[v], fields) for v in fields}
        for v in fields:
            state[v] = state[v] + dt * (rhs[v] + E[v][k])
            out[v].append(state[v][0].copy())
    return {v: np.array(a) for v, a in out.items()}


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


def part_path(tag):
    return os.path.join(HERE, "results", f"swarm_field_sim_{tag}.json")


def merge():
    out = {}
    for f in swarm_files():
        p = part_path(swarm_meta(f)["tag"])
        if os.path.exists(p):
            out.update(json.load(open(p)))
    json.dump(out, open(os.path.join(HERE, "results", "swarm_field_sim.json"), "w"), indent=1)
    print(f"merged {len(out)} recordings")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--tags", default="")
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    if a.merge:
        return merge()
    tags = [t for t in a.tags.split(",") if t]
    fits = json.load(open(os.path.join(HERE, "results", "swarm_fields.json")))
    for path in swarm_files():
        tag = swarm_meta(path)["tag"]
        if tags and tag not in tags:
            continue
        F = SF.build_fields(path)
        eqs = fits[tag]["eqs"]
        laws = {v: eqs[v]["cands"][SF.balanced(eqs[v]["cands"], eqs[v]["front"])]["terms"] for v in SF.VARS}
        laws_D = dict(laws, rho=fit_relaxation(F))
        res = dict(laws=laws, density_relaxation=laws_D["rho"], sources={})
        sub = lambda fl: {v: fl[v][::SUB_STEP] for v in SF.VARS}                 # noqa: E731
        res["sources"]["experiment (0.08 s fields)"] = stats(sub(F))
        res["sources"]["experiment"] = stats(experiment_fields_04(path, F))
        for label, (suffix, name) in AGENT_MODELS.items():
            res["sources"][f"agents: {label}"] = stats(agent_fields(path, F, suffix, name))
        for label, L in (("field laws", laws), ("field laws + density relaxation", laws_D)):
            E = residuals(F, L)
            runs = [stats(sub(simulate(F, L, E, np.random.default_rng(s)))) for s in range(a.seeds)]
            res["sources"][label] = average(runs)
        print(f"\n### {tag}: density relaxation law " + ", ".join(f"{k} {v:+.3g}" for k, v in laws_D["rho"].items()))
        for k, s in res["sources"].items():
            print(f"  {k:32s} rms rho {s['rms_rho']:.4f} |P| {s['rms_P']:.4f} |W| {s['rms_W']:.4f} | C_P 0.4/0.8/2/4 s "
                  + "/".join(f"{s['C_P'][L]:+.2f}" for L in (1, 2, 5, 10)) + " | C_W "
                  + "/".join(f"{s['C_W'][L]:+.2f}" for L in (1, 2, 5, 10)) + " | S_rho 0.4/4/40/100 s "
                  + "/".join(f"{v:.2e}" for v in (s["S_rho"][0], s["S_rho"][3], s["S_rho"][6], s["S_rho"][-1])),
                  flush=True)
        json.dump({tag: res}, open(part_path(tag), "w"), indent=1)
    if not tags:
        merge()


if __name__ == "__main__":
    main()
