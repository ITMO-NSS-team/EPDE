r"""Mesoscale with the spin spread kept: polarization fields per spin-rate group, their laws, and the
field simulation.

The (P, W) closure of swarm_fields.py behaves like two spin rates, w+ and w-; inside each spin
direction the robots' rates spread by sd 0.8-1.3 rad/s, which dephases a group's polarization within
~2 s. Here each spin direction is split into K groups of (nearly) equal size by spin rate; groups
b = 1 .. 2K are ordered from the fastest clockwise to the fastest counter-clockwise, and each group
gets its own polarization field next to the total density:
    rho(t, r)  = sum_i A K(r - r_i)                   area fraction
    P_b(t, r)  = sum_{i in b} A p_i K(r - r_i)         polarization of group b
(kernel, grid, time step and p_i as in swarm_fields.py). Pipeline per field as in swarm_fields.py:
library = own field and its derivatives, the other fields, their gradients and Laplacians, products
of <= 2 atoms; exact (WAPE, win14) front; reported law = the balanced member. Expected for
independent spinners:
    rho_t = -sum_b v_b div P_b,     P_b,t = w_b z x P_b - D_r P_b.
Field simulation as in swarm_field_sim.py (bootstrapped residual fields, mirrored edges). Group laws
in the simulation: the rotation-damping form P_bc,t = a P_bc + b P_b(other c), refit by OLS -- the
balanced member's form for most group fields, but where the balanced member keeps the rotation and
drops the damping the stepped map grows (|lambda| > 1) and the simulation diverges. Rate and damping
are read from the stepped map's eigenvalue lambda = exp((-D + i w) dt), which removes the w^2 dt / 2 a
forward-increment fit adds to the own-field coefficient. Density: the balanced law, or ("+ density
relaxation") -sum_b (v_bx d_x P_bx + v_by d_y P_by) + D lap rho - kappa (rho - <rho>(r)). Statistics
on the totals P = sum_b P_b and W = sum_b w_b P_b, w_b = the group's mean spin rate (the experiment's
W with group means is reported beside the exact one).

--tau 5: fields every 0.4 s (with rho_bar), laws fitted on 0.4 s increments and stepped every 0.4 s; the
density laws then come from swarm_fields.py --tau 5.

python swarm_fields_binned.py --k 2 [--tags ...] [--stage fit|sim|all] [--tau 5] [--workers 7 --cmax 6 --seeds 4]
                                       -> results/swarm_fields_bins<K>[_tau5]_<tag>.json
python swarm_fields_binned.py --merge [--tau 5]
                                       -> results/swarm_fields_binned[_tau5].json, .txt
python swarm_fields_binned.py --isotropic --k 2 --tau 5 [--tags ...] [--merge]
        every isotropic system of the group laws (each group's y law its x law turned by 90 degrees, the
        density law unchanged by the turn) -> results/swarm_fields_bins2_tau5_iso_<tag>.json, merged
        results/swarm_fields_binned_tau5_iso.json
python swarm_fields_binned.py --fronts --k 2 --tau 5 [--tags ...] [--merge]
        no law selected: every front member of every group-system equation, one equation at a time with the
        others at their balanced members -> results/swarm_fields_bins2_tau5_fronts_<tag>.json,
        merged results/swarm_fields_binned_tau5_fronts.json
"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse
import glob
import json
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import swarm_sim as SS                                           # noqa: E402
import swarm_fields as SF                                        # noqa: E402
import swarm_field_sim as FS                                     # noqa: E402
from swarm_data import swarm_files, swarm_meta, load_swarm, arena_envelope  # noqa: E402


# ------------------------------------------------------------------ fields
def groups(w, k):
    """Group label per robot: each spin direction split into k groups by rate, 0 = fastest clockwise
    .. 2k-1 = fastest counter-clockwise."""
    lab = np.empty(len(w), int)
    cw, ccw = np.where(w < 0)[0], np.where(w > 0)[0]
    for b, part in enumerate(np.array_split(cw[np.argsort(w[cw])], k)):
        lab[part] = b
    for b, part in enumerate(np.array_split(ccw[np.argsort(w[ccw])], k)):
        lab[part] = k + b
    return lab


def field_names(nb):
    return ["rho"] + [f"P{b + 1}{c}" for b in range(nb) for c in "xy"]


def coarse_groups(F, X, Y, phi, lab, nb):
    g, ell = F["g"], SS.CAP_CM
    norm = np.pi * (SS.CAP_CM / 2) ** 2 / (2 * np.pi * ell ** 2)
    gx = np.exp(-(g[None, None, :] - X[:, :, None]) ** 2 / (2 * ell ** 2))
    gy = np.exp(-(g[None, None, :] - Y[:, :, None]) ** 2 / (2 * ell ** 2))
    px, py = np.cos(phi + F["theta"]), np.sin(phi + F["theta"])
    field = lambda wt: norm * np.einsum("ti,tix,tiy->txy", wt, gx, gy)     # noqa: E731
    out = dict(rho=field(np.ones_like(px)))
    for b in range(nb):
        m = (lab == b).astype(float)
        out[f"P{b + 1}x"], out[f"P{b + 1}y"] = field(px * m), field(py * m)
    return out


def build_groups(path, k):
    tag = swarm_meta(path)["tag"]
    out = os.path.join(SF.FIELD_DIR, f"{tag}_bins{k}.npz")
    if os.path.exists(out):
        return dict(np.load(out))
    F = SF.build_fields(path)
    t, x, y, phi, ids = load_swarm(path)
    xc, yc, r_env = arena_envelope(x, y)
    s = SS.px_per_cm()
    sl = slice(None, None, SF.STEP)
    lab = groups(F["w"], k)
    B = coarse_groups(F, (x[sl] - xc) / s, (y[sl] - yc) / s, phi[sl], lab, 2 * k)
    B.update(t=F["t"], g=F["g"], theta=F["theta"], w=F["w"], v0=F["v0"], labels=lab,
             wbar=np.array([F["w"][lab == b].mean() for b in range(2 * k)]))
    np.savez_compressed(out, **B)
    return B


def totals(fl, wbar):
    """rho, P = sum_b P_b, W = sum_b w_b P_b (the statistics' fields)."""
    nb = len(wbar)
    P = [sum(fl[f"P{b + 1}{c}"] for b in range(nb)) for c in "xy"]
    W = [sum(wbar[b] * fl[f"P{b + 1}{c}"] for b in range(nb)) for c in "xy"]
    return dict(rho=fl["rho"], Px=P[0], Py=P[1], Wx=W[0], Wy=W[1])


def expected(var, nb):
    if var == "rho":
        return [f"P{b + 1}{c}_{c}" for b in range(nb) for c in "xy"] + ["d^2rho/dx1^2", "d^2rho/dx2^2"]
    other = var[:-1] + ("y" if var[-1] == "x" else "x")
    return [other, var, f"d^2{var}/dx1^2", f"d^2{var}/dx2^2"]


def perpendicular(var):
    return var[:-1] + ("y" if var[-1] == "x" else "x")


def rotation_law(B, var, names, lat_stride=SF.LAT_STRIDE):
    """The group law in its rotation-damping form, P_bc,t = a P_bc + b P_b(other c): OLS on the
    pipeline's lattice (as every pipeline candidate)."""
    P = SF.problem(B, var, names, lat_stride)
    named = dict(P["atoms"])
    terms = [var, perpendicular(var)]
    A = np.column_stack([named[n] for n in terms])[P["lattice"]]
    coef = np.linalg.lstsq(A, P["y"][P["lattice"]], rcond=None)[0]
    return dict(zip(terms, coef.tolist()))


def group_map(laws, b, dt):
    """Rotation rate and damping of group b from its stepped map M = I + dt A (A from the P_bx, P_by
    laws' own-group terms): eigenvalue lambda = exp((-D + i w) dt). Removes the forward step's
    w^2 dt / 2 that the fitted own-field coefficient carries."""
    x, y = f"P{b + 1}x", f"P{b + 1}y"
    A = np.array([[laws[x].get(x, 0.0), laws[x].get(y, 0.0)], [laws[y].get(x, 0.0), laws[y].get(y, 0.0)]])
    lam = np.linalg.eigvals(np.eye(2) + dt * A)
    lam = lam[np.argmax(np.abs(lam))]                                  # the slower-decaying mode
    sense = np.sign(A[1, 0] - A[0, 1])                                 # +1: counter-clockwise rotation
    return dict(rate=float(sense * abs(np.angle(lam)) / dt), damping=float(-np.log(np.abs(lam)) / dt),
                radius=float(np.abs(lam)))


# ------------------------------------------------------------------ one recording
def part_path(k, tag, tau=1):
    return os.path.join(HERE, "results", f"swarm_fields_bins{k}{SF.suffix(tau)}_{tag}.json")


def lat_stride_for(tau):
    return max(1, round(SF.LAT_STRIDE / tau))


def fields_at(path, k, tau):
    """Group fields, every tau-th field step (with rho_bar) when tau > 1."""
    B = build_groups(path, k)
    return SF.coarse_time(B, tau) if tau > 1 else B


def fit_recording(path, k, cmax, workers, tau=1):
    tag = swarm_meta(path)["tag"]
    SF.build_fields(path)
    B = fields_at(path, k, tau)
    nb, names = 2 * k, field_names(2 * k)
    rec = dict(k=k, tau=tau, wbar=B["wbar"].tolist(), sizes=np.bincount(B["labels"], minlength=nb).tolist(),
               sd_in_group=[float(B["w"][B["labels"] == b].std()) for b in range(nb)], eqs={})
    for var in names:
        st = SF.run(B, var, cmax, workers, fields=names, expected=expected(var, nb), lat_stride=lat_stride_for(tau))
        rec["eqs"][var] = st
        bal = st["cands"][SF.balanced(st["cands"], st["front"])]
        print(f"{tag} K={k} d{var}/dx0: {st['n_columns']} cols, {st['seconds']}s, balanced {bal['terms']} "
              f"R2 {bal['r2']:.3f}", flush=True)
    json.dump({tag: rec}, open(part_path(k, tag, tau), "w"), indent=1)
    return rec


RHO_ATOMS = {"rho", "rho_bar", "drho/dx1", "drho/dx2", "d^2rho/dx1^2", "d^2rho/dx2^2", "lap_self", "delta_self"}


def density_on_groups(terms, nb):
    """A density law of swarm_fields.py (terms Px_x, Py_y, and terms of rho and rho_bar alone) written
    on the group fields: d_x P_x = sum_b d_x P_bx, the same law since rho is the same field and
    P = sum_b P_b."""
    out = {}
    for n, c in terms.items():
        factors = [f if f in RHO_ATOMS else f.rsplit("^", 1)[0] for f in n.split("*")]
        if n in ("Px_x", "Py_y"):
            for b in range(nb):
                out[f"P{b + 1}{n[1]}_{n[-1]}"] = c
        elif all(f in RHO_ATOMS for f in factors):
            out[n] = c
        else:
            raise ValueError(f"density term {n} has no group form")
    return out


def sim_recording(path, k, seeds, tau=1):
    """Simulate from the saved fit. Groups: the rotation-damping form (the balanced member's form for
    most group fields; a balanced member that keeps the rotation but drops the damping makes the
    stepped map grow, |lambda| > 1). Density: the same laws as the two-field model (its balanced law,
    or its relaxation law), so only the polarization dynamics differ; the per-group relaxation fit
    is kept for the report (its kappa is ~0 at N30 and can come out negative)."""
    tag = swarm_meta(path)["tag"]
    rec = json.load(open(part_path(k, tag, tau)))[tag]
    F = SF.build_fields(path)
    G = SF.coarse_time(F, tau) if tau > 1 else F
    B = fields_at(path, k, tau)
    nb, names = 2 * k, field_names(2 * k)
    dt = float(np.median(np.diff(B["t"])))
    balanced = {v: rec["eqs"][v]["cands"][SF.balanced(rec["eqs"][v]["cands"], rec["eqs"][v]["front"])]["terms"]
                for v in names}
    eqs5 = json.load(open(os.path.join(HERE, "results", f"swarm_fields{SF.suffix(tau)}.json")))[tag]["eqs"]
    rho5 = eqs5["rho"]["cands"][SF.balanced(eqs5["rho"]["cands"], eqs5["rho"]["front"])]["terms"]
    laws = {"rho": density_on_groups(rho5, nb),
            **{v: rotation_law(B, v, names, lat_stride_for(tau)) for v in names[1:]}}
    laws_D = dict(laws, rho=density_on_groups(FS.fit_relaxation(G), nb))
    rec.update(balanced=balanced, laws=laws, density_relaxation=laws_D["rho"],
               density_relaxation_per_group=FS.fit_relaxation(B, names),
               balanced_is_rotation_form={v: sorted(balanced[v]) == sorted(laws[v]) for v in names[1:]},
               balanced_maps=[group_map({**laws, **{v: balanced[v] for v in names[1:]}}, b, dt) for b in range(nb)],
               maps=[group_map(laws, b, dt) for b in range(nb)], sources={})
    step = max(1, FS.SUB_STEP // tau)                                          # -> the 0.4 s clock
    sub = lambda fl: {v: fl[v][::step] for v in fl if v in names}             # noqa: E731
    rec["sources"]["experiment (W from group means)"] = FS.stats(totals(sub(B), B["wbar"]))
    for label, L in ((f"field laws, {k} rate groups per spin direction", laws),
                     (f"field laws, {k} rate groups per spin direction + density relaxation", laws_D)):
        t0 = time.time()
        E = FS.residuals(B, L, names)
        runs = [FS.stats(totals(sub(FS.simulate(B, L, E, np.random.default_rng(s), names)), B["wbar"]))
                for s in range(seeds)]
        rec["sources"][label] = FS.average(runs)
        print(f"{tag} K={k} simulated '{label}' x{seeds} in {time.time() - t0:.0f}s", flush=True)
    json.dump({tag: rec}, open(part_path(k, tag, tau), "w"), indent=1)
    return rec


def fronts_path(k, tag, tau=1):
    return os.path.join(HERE, "results", f"swarm_fields_bins{k}{SF.suffix(tau)}_fronts_{tag}.json")


def sim_fronts(path, k, seeds, tau=1):
    """No law selected: every front member of every equation of the group system (density from this
    fit's own front), one equation at a time with the others at their balanced members, simulated as
    found (swarm_field_sim.simulate_fronts); each group's rotation rate and decay from the base laws."""
    tag = swarm_meta(path)["tag"]
    rec = json.load(open(part_path(k, tag, tau)))[tag]
    B = fields_at(path, k, tau)
    nb, names = 2 * k, field_names(2 * k)
    dt = float(np.median(np.diff(B["t"])))
    step = max(1, FS.SUB_STEP // tau)
    sub = lambda fl: {v: fl[v][::step] for v in fl if v in names}             # noqa: E731
    print(f"{tag} K={k}: fronts", flush=True)
    res = FS.simulate_fronts(B, rec["eqs"], names, seeds, sub, lambda fl: totals(fl, B["wbar"]))
    res["base"]["maps"] = [group_map(res["base"]["laws"], b, dt) for b in range(nb)]
    out = dict(k=k, tau=tau, wbar=rec["wbar"], sizes=rec["sizes"], **res)
    json.dump({tag: out}, open(fronts_path(k, tag, tau), "w"), indent=1)
    return out


def iso_path(k, tag, tau=1):
    return os.path.join(HERE, "results", f"swarm_fields_bins{k}{SF.suffix(tau)}_iso_{tag}.json")


def sim_iso(path, k, seeds, tau=1):
    """Every isotropic system of the group laws (swarm_field_sim.isotropic_combinations: each group's y law
    is its x law turned by 90 degrees, the density law unchanged by the turn), simulated as found."""
    tag = swarm_meta(path)["tag"]
    rec = json.load(open(part_path(k, tag, tau)))[tag]
    F = SF.build_fields(path)
    B = fields_at(path, k, tau)
    nb, names = 2 * k, field_names(2 * k)
    step = max(1, FS.SUB_STEP // tau)
    sub = lambda fl: {v: fl[v][::step] for v in fl if v in names}             # noqa: E731
    combos, matched = FS.isotropic_combinations(rec["eqs"], [(f"P{b + 1}x", f"P{b + 1}y") for b in range(nb)])
    print(f"{tag} K={k}: {len(combos)} isotropic systems", flush=True)
    res = dict(k=k, tau=tau, wbar=rec["wbar"], sizes=rec["sizes"], fronts=FS.front_table(rec["eqs"], names),
               experiment=FS.stats(FS.experiment_fields_04(path, F)), pairs=matched,
               combos=FS.simulate_combinations(B, rec["eqs"], names, seeds, sub,
                                               totals=lambda fl: totals(fl, B["wbar"]), combos=combos))
    json.dump({tag: res}, open(iso_path(k, tag, tau), "w"))
    return res


def merge_fronts(tau=1):
    out = {}
    for p in sorted(glob.glob(os.path.join(HERE, "results", f"swarm_fields_bins?{SF.suffix(tau)}_fronts_N*.json"))):
        for tag, rec in json.load(open(p)).items():
            out.setdefault(tag, {})[str(rec["k"])] = rec
    json.dump(out, open(os.path.join(HERE, "results", f"swarm_fields_binned{SF.suffix(tau)}_fronts.json"), "w"),
              indent=1)
    print(f"merged fronts of {len(out)} recordings")


# ------------------------------------------------------------------ report
def corr_error(s, ref, key):
    """Mean |C(tau) - C_experiment(tau)| over the lags 0.4 .. 10 s."""
    return float(np.mean(np.abs(np.array(s[key][1:]) - np.array(ref[key][1:]))))


def law_line(var, terms):
    return f"d{var}/dx0 = " + " ".join(f"{v:+.4g} {n}" for n, v in terms.items())


def merge(tau=1):
    sfx = SF.suffix(tau)
    out = {}
    for p in sorted(glob.glob(os.path.join(HERE, "results", f"swarm_fields_bins?{sfx}_N*.json"))):
        for tag, rec in json.load(open(p)).items():
            out.setdefault(tag, {})[str(rec["k"])] = rec
    json.dump(out, open(os.path.join(HERE, "results", f"swarm_fields_binned{sfx}.json"), "w"), indent=1)
    base = json.load(open(os.path.join(HERE, "results", f"swarm_field_sim{sfx}.json")))
    lines = [__doc__.split("python swarm_fields_binned")[0]]
    for tag, byk in out.items():
        lines.append(f"\n{'=' * 100}\n{tag}")
        for k, rec in sorted(byk.items()):
            nb = 2 * int(k)
            lines.append(f"\n K = {k} groups per spin direction: sizes {rec['sizes']}, mean rates "
                         + ", ".join(f"{w:+.2f}" for w in rec["wbar"]) + " rad/s, sd inside groups "
                         + ", ".join(f"{s:.2f}" for s in rec["sd_in_group"]))
            lines.append("  group  measured w | stepped map of the simulated law: rate  damping  |lambda| | balanced "
                         "member: rotation form? (x / y), |lambda|, held-out R2 (x / y)")
            for b in range(nb):
                vx, vy = f"P{b + 1}x", f"P{b + 1}y"
                m, mb = rec["maps"][b], rec["balanced_maps"][b]
                form = rec["balanced_is_rotation_form"]
                r2 = [rec["eqs"][v]["cands"][SF.balanced(rec["eqs"][v]["cands"], rec["eqs"][v]["front"])]["r2"]
                      for v in (vx, vy)]
                lines.append(f"  {b + 1:5d}  {rec['wbar'][b]:+9.2f}  | {m['rate']:+18.2f} {m['damping']:8.3f} "
                             f"{m['radius']:8.4f} | {'yes' if form[vx] else 'no':>3s} / {'yes' if form[vy] else 'no':3s}"
                             f"  {mb['radius']:.4f}  {r2[0]:.3f} / {r2[1]:.3f}")
            lines.append("  balanced laws (pipeline):")
            for v, terms in rec["balanced"].items():
                lines.append("   " + law_line(v, terms))
            lines.append("  simulated group laws (rotation-damping form):")
            for v, terms in rec["laws"].items():
                if v != "rho":
                    lines.append("   " + law_line(v, terms))
            lines.append("   density (simulation, as the two-field model): " + law_line("rho", rec["laws"]["rho"]))
            lines.append("   density with relaxation (as the two-field model): " + law_line("rho", rec["density_relaxation"]))
            if "density_relaxation_per_group" in rec:
                lines.append("   per-group relaxation fit (report only): "
                             + law_line("rho", rec["density_relaxation_per_group"]))
        srcs = dict(base.get(tag, {}).get("sources", {}))
        for k, rec in sorted(byk.items()):
            srcs.update(rec["sources"])
        ref = srcs["experiment"]
        lines.append("\n  source                                                              rms rho  |P|     |W|    "
                     "C_P 0.4/0.8/2/4 s        C_W 0.4/0.8/2/4 s        err C_P  err C_W   S_rho 4/40/100 s")
        for n, s in srcs.items():
            if n.startswith("experiment (") and n.endswith(" s fields)"):
                continue
            lines.append(f"  {n:68s} {s['rms_rho']:.4f}  {s['rms_P']:.4f}  {s['rms_W']:.3f}  "
                         + " ".join(f"{s['C_P'][L]:+.2f}" for L in (1, 2, 5, 10)) + "  "
                         + " ".join(f"{s['C_W'][L]:+.2f}" for L in (1, 2, 5, 10)) + "  "
                         + f"{corr_error(s, ref, 'C_P'):7.3f}  {corr_error(s, ref, 'C_W'):7.3f}   "
                         + " ".join(f"{s['S_rho'][i]:.2e}" for i in (3, 6, len(s["S_rho"]) - 1)))
    text = "\n".join(lines) + "\n"
    open(os.path.join(HERE, "results", f"swarm_fields_binned{sfx}.txt"), "w", encoding="utf-8").write(text)
    print(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--tags", default="")
    ap.add_argument("--workers", type=int, default=7)
    ap.add_argument("--cmax", type=int, default=6)
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--tau", type=int, default=1, help="fit and step every tau field steps (5: 0.4 s)")
    ap.add_argument("--stage", choices=["fit", "sim", "all"], default="all")
    ap.add_argument("--fronts", action="store_true",
                    help="no law selected: every front member of every equation (needs the fit)")
    ap.add_argument("--isotropic", action="store_true",
                    help="every isotropic system of the group laws (needs the fit)")
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    if a.isotropic:
        if a.merge:
            return FS.merge_combos(os.path.join(HERE, "results", f"swarm_fields_bins{a.k}{SF.suffix(a.tau)}_iso_N*.json"),
                                   os.path.join(HERE, "results", f"swarm_fields_binned{SF.suffix(a.tau)}_iso.json"))
        for path in swarm_files():
            if not a.tags or swarm_meta(path)["tag"] in a.tags.split(","):
                sim_iso(path, a.k, a.seeds, a.tau)
        return
    if a.merge:
        return merge_fronts(a.tau) if a.fronts else merge(a.tau)
    if a.fronts:
        for path in swarm_files():
            if not a.tags or swarm_meta(path)["tag"] in a.tags.split(","):
                sim_fronts(path, a.k, a.seeds, a.tau)
        return
    tags = [t for t in a.tags.split(",") if t]
    for path in swarm_files():
        if tags and swarm_meta(path)["tag"] not in tags:
            continue
        if a.stage in ("fit", "all"):
            fit_recording(path, a.k, a.cmax, a.workers, a.tau)
        if a.stage in ("sim", "all"):
            sim_recording(path, a.k, a.seeds, a.tau)


if __name__ == "__main__":
    main()
