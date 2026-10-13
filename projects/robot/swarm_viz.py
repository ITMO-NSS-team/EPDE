r"""Experiment vs simulation, side by side: one HTML page from swarm_spin_contact.py (both laws).

Per recording: the recorded swarm and each simulated variant (seed 0; every robot its own EPDE laws,
or one common system with each robot's own propulsion tokens) replayed in two arenas (robots
drawn at the cap's true size, coloured by spin direction, a tick along the marker heading), plus the
swarm statistics of all variants against the experiment: MSD, nearest-neighbour distance, contact
numbers, radial layering, speed by contact number, and the summary errors.

python swarm_viz.py [--out results/swarm_viz.html]
"""
import argparse
import base64
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import swarm_sim as SS                                           # noqa: E402
import swarm_contact_law as SCL                                  # noqa: E402
from swarm_data import swarm_files, swarm_meta, ROBOT_DIAMETER_PX as SIGMA  # noqa: E402

TEMPLATE = os.path.join(HERE, "swarm_viz_template.html")


#: law -> (file suffix, label on the page)
LAWS = {"epde": ("", "per-robot EPDE laws"), "own_prop": ("_own_prop", "common system")}
GROUPS_SHOWN = ("2",)              # spin-rate groups per direction on the page (3 adds little)


def recording_payload(tag, md, allres):
    res = allres["epde"][tag]
    zs = {law: np.load(os.path.join(HERE, "results", "viz", f"{tag}{suf}.npz")) for law, (suf, _) in LAWS.items()}
    z = zs["epde"]
    xc, yc, r_env, cap, dt = z["meta"]
    names, trajs, sources = [], [(z["data_pos"], z["data_phi"])], []
    for law, (suf, label) in LAWS.items():
        for i, n in enumerate(zs[law]["names"]):
            names.append(f"{label} · {n}")
            trajs.append((zs[law][f"sim{i}_pos"], zs[law][f"sim{i}_phi"]))
            sources.append(allres[law][tag]["test"][str(n)])
    T = min(len(p) for p, _ in trajs)
    buf = np.empty((len(trajs), T, z["data_pos"].shape[1], 3), dtype=np.int16)
    for k, (p, ph) in enumerate(trajs):
        buf[k, ..., 0] = np.round(p[:T, :, 0] - xc)
        buf[k, ..., 1] = np.round(p[:T, :, 1] - yc)
        buf[k, ..., 2] = np.round(np.mod(ph[:T], 2 * np.pi) * 1000)
    scale = cap / SS.CAP_CM                                       # px per cm
    data = res["data"]

    def series(c):
        return dict(msd=[c["msd"][str(L)] / scale ** 2 for L in SS.MSD_LAGS],
                    contacts=c["contacts"], gap=c["gap"], rhist=c["rhist"],
                    speed=[(v or 0) * 10 / scale for v in c["speed_by_n"]])

    models = []
    for name, c in zip(names, sources):
        ratios = list(c["msd_ratio"].values())
        sp = [abs(np.log(a / b)) for a, b in zip(c["speed_by_n"], data["speed_by_n"]) if a and b]
        models.append(dict(name=name, **series(c), summary=dict(
            msd_err=float(np.mean(np.abs(np.log(ratios)))),
            contact_tv=float(0.5 * np.abs(np.array(c["contacts"]) - data["contacts"]).sum()),
            gap_tv=c["gap_tv"], layer_tv=c["rhist_tv"], speed_err=float(np.mean(sp)),
            r_msd10=c["corr"]["msd10"], r_radius=c["corr"]["radius"], r_tang=c["corr"]["tang"])))
    exp = series(dict(msd={str(L): data["msd"][str(L)] for L in SS.MSD_LAGS}, contacts=data["contacts"],
                      gap=res["gap"]["hist"], rhist=data["rhist"], speed_by_n=data["speed_by_n"]))
    gap_mm = [e * SIGMA / scale * 10 for e in SCL.GAP_EDGES[1:-1]]
    gpath = os.path.join(HERE, "results", "fields", f"{tag}_bins{GROUPS_SHOWN[0]}.npz")
    group = np.load(gpath)["labels"].astype(int).tolist() if os.path.exists(gpath) else None
    return dict(tag=tag, n=int(md["n"]), pwm=int(md["pwm"]), arena_cm=float(md["arena_cm"]),
                ncw=int(md["ncw"]), nccw=int(md["nccw"]), scale=float(scale), r_env=float(r_env),
                cap=float(cap), dt=float(dt), t0=float(SS.HIST * dt), frames=int(T), robots=int(buf.shape[2]),
                spin=np.sign(z["w"]).astype(int).tolist(), wrate=[round(float(v), 3) for v in z["w"]],
                group=group, names=names,
                coef=res["spin_coef"], heldout=res["heldout"],
                fit={label: dict(law=allres[law][tag]["heldout"]["law"],
                                 contact=allres[law][tag]["heldout"]["contact law"])
                     for law, (_, label) in LAWS.items()},
                gap_mm=[round(v, 1) for v in gap_mm], rbins=SS.RBINS,
                lags=[L * float(dt) for L in SS.MSD_LAGS], exp=exp, models=models,
                b64=base64.b64encode(buf.tobytes()).decode("ascii"))


FIELD_NAMES = {"rho": "ρ", "Px": "Pₓ", "Py": "Pᵧ", "Wx": "Wₓ", "Wy": "Wᵧ",
               "Pr": "P_r", "Pt": "P_θ", "Wr": "W_r", "Wt": "W_θ",
               **{f"P{b}{c}": f"P{b}{'ₓ' if c == 'x' else 'ᵧ'}" for b in range(1, 7) for c in "xy"}}
STATIC_NAMES = {"inv_r": "1/r", "one": "1"}


def pretty(name):
    """Field-library token -> math text: Px_x -> ∂ₓPₓ, d^2rho/dx1^2 -> ∂ₓ²ρ, lap_W -> ∇²W, products a·b."""
    def one(f):
        if f == "lap_self":
            return "∇²ρ"
        if f == "delta_self":
            return "(ρ − ρ̄)"
        if f == "rho_bar":
            return "ρ̄"
        if f in STATIC_NAMES:
            return STATIC_NAMES[f]
        p = ""
        if "^" in f and not f.startswith("d^2"):
            f, e = f.rsplit("^", 1)
            p = "²" if e == "2" else f"^{e}"
        for v, s in FIELD_NAMES.items():
            if f == v:
                return s + p
            if f == f"{v}_x":
                return f"∂ₓ{s}" + p
            if f == f"{v}_y":
                return f"∂ᵧ{s}" + p
            if f == f"lap_{v}":
                return f"∇²{s}" + p
            if f == f"d{v}/dx1":
                return f"∂ₓ{s}" + p
            if f == f"d{v}/dx2":
                return f"∂ᵧ{s}" + p
            if f == f"d^2{v}/dx1^2":
                return f"∂ₓ²{s}" + p
            if f == f"d^2{v}/dx2^2":
                return f"∂ᵧ²{s}" + p
            if f in (f"{v}_r", f"d{v}/dr"):                                    # polar components
                return f"∂_r{s}" + p
            if f in (f"{v}_t", f"d{v}/dt"):
                return f"(1/r)∂_θ{s}" + p
            if f == f"d^2{v}/dr^2":
                return f"∂_r²{s}" + p
            if f == f"d^2{v}/dt^2":
                return f"(1/r²)∂_θ²{s}" + p
        return f + p
    return "·".join(one(f) for f in name.split("*"))


def law_text(var, terms):
    parts = []
    for i, (n, c) in enumerate(terms.items()):
        mag = f"{abs(c):.3g} {pretty(n)}"
        parts.append((("−" if c < 0 else "") if i == 0 else ("− " if c < 0 else "+ ")) + mag)
    return f"∂ₜ{FIELD_NAMES[var]} = " + " ".join(parts)


STAT_KEYS = ("C_P", "C_W", "S_rho", "rms_rho", "rms_P", "rms_W")


PROFILE_KEYS = ("rho_r", "Pr_r", "Pt_r", "Wt_r")                 # radial profiles (polar fields)


def stats_payload(st):
    return None if st is None else {k: st[k] for k in STAT_KEYS + PROFILE_KEYS if k in st}


def summary_of(s, exp):
    """[C_P error, C_W error, <drho^2>(100 s) ratio, drho ratio, |P| ratio, |W| ratio] against the experiment."""
    err = lambda k: float(np.mean(np.abs(np.array(s[k][1:]) - np.array(exp[k][1:]))))         # noqa: E731
    return [round(err("C_P"), 4), round(err("C_W"), 4), round(s["S_rho"][-1] / exp["S_rho"][-1], 3),
            round(s["rms_rho"] / exp["rms_rho"], 3), round(s["rms_P"] / exp["rms_P"], 3), round(s["rms_W"] / exp["rms_W"], 3)]


def hypotheses_payload(rec):
    """Hypothesis systems on the polar fields (swarm_fields_polar --hypotheses): per hypothesis whether the
    fronts support it, each law as math text with its place on its non-dominated front and its scores, and
    the simulated statistics against the experiment (None when not supported or diverging)."""
    exp = rec["sources"]["experiment"]
    hyps = []
    for key, h in rec["hypotheses"].items():
        st = h.get("stats")
        hyps.append(dict(key=key, description=h["description"], supported=h["supported"],
                         missing=[FIELD_NAMES[v] for v in h["missing"]], diverged=bool(h.get("diverged")),
                         laws=[dict(law=law_text(v, t), **h["scores"][v]) for v, t in h["laws"].items()],
                         stats=stats_payload(st), summary=None if st is None else summary_of(st, exp)))
    return dict(r=rec["r"], ref=stats_payload(exp), hypotheses=hyps,
                agents={k: stats_payload(v) for k, v in rec["sources"].items() if k.startswith("agents")})


def group_rates(rec, dt=0.4):
    """Per spin-rate group: the measured mean spin rate and the range of rotation rates over every law on
    the fronts of its two polarization equations (each with the other equations at their balanced laws)."""
    import swarm_fields_binned as SB
    base = rec["base"]["laws"]
    lo, hi = [], []
    for b in range(len(rec["wbar"])):
        rates = [SB.group_map(dict(base, **{v: m["terms"]}), b, dt)["rate"]
                 for v in (f"P{b + 1}x", f"P{b + 1}y") for m in rec["fronts"][v]]
        lo.append(min(rates))
        hi.append(max(rates))
    return dict(k=int(rec["k"]), sizes=rec["sizes"], measured=rec["wbar"], rate_lo=lo, rate_hi=hi)


def field_payload(hyp, fg):
    """Field section: the hypothesis systems with the experiment and the agent simulations (polar fields);
    the spin-rate groups' measured rates for the replay's colouring."""
    g = fg.get(GROUPS_SHOWN[0])
    return dict(hyp=hypotheses_payload(hyp), groups=group_rates(g) if g else None,
                cp_lags=[0.4 * L for L in range(26)], s_lags=[0.4 * L for L in (1, 2, 5, 10, 25, 50, 100, 250)])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "results", "swarm_viz.html"))
    a = ap.parse_args()
    allres = {law: json.load(open(os.path.join(HERE, "results", f"swarm_spin_contact{suf}.json")))
              for law, (suf, _) in LAWS.items()}
    load = lambda name: json.load(open(p)) if os.path.exists(p := os.path.join(HERE, "results", name)) else {}  # noqa: E731
    hyp, fg = load("swarm_field_hypotheses_polar.json"), load("swarm_fields_binned_tau5_fronts.json")
    recs = []
    for f in swarm_files():
        md = swarm_meta(f)
        if all(md["tag"] in r for r in allres.values()):
            recs.append(recording_payload(md["tag"], md, allres))
            if md["tag"] in hyp:
                recs[-1]["fields"] = field_payload(hyp[md["tag"]], fg.get(md["tag"], {}))
    html = open(TEMPLATE, encoding="utf-8").read().replace("/*PAYLOAD*/null", json.dumps(recs))
    with open(a.out, "w", encoding="utf-8") as fh:
        fh.write(html)
    print(f"written {a.out} ({os.path.getsize(a.out) / 1e6:.1f} MB, {len(recs)} recordings)")


if __name__ == "__main__":
    main()
