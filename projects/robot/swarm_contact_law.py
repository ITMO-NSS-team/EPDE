r"""The learned contact law in place of the hard-disk projection.

Contact law (shared, isotropic, local, fitted on the residuals of each law):
  push(D)     the push resolved by the pair distance D (robot diameters): the law's own push plus a
              correction per band [0, .93) [.93, .97) [.97, 1) [1, 1.05) [1.05, 1.1)
  np, np_in   the neighbours' drive and its part driving into the robot (swarm_neighbour_drive.py)
Held-out (first half -> second half): base law, + neighbour drive, + contact law.
Simulations (4 seeds) per law; the arena wall stays the container (centres inside r_max):
  <law>                         hard-disk projection
  <law> + neighbour drive       hard-disk projection
  <law> + drive, no projection  constant push + np + np_in, no pair projection
  <law> + contact law           distance-resolved push + np + np_in, no pair projection
  <law> + contact law, guard    the same, pairs only kept apart below the closest distance the
                                recording ever shows (its 0.01 % nearest-neighbour distance)
Extra statistics: the nearest-neighbour distance distribution (robot diameters) and its total
variation distance to the recording's; the 0.1 % NN distance (how deep robots overlap); the
bond-normal relative velocity of touching pairs per distance band, data vs law.

python swarm_contact_law.py [--tags N30_PWM10_D41,...]  -> results/swarm_contact_law_<tag>.json, merged
                                                          into results/swarm_contact_law.json
python swarm_contact_law.py --merge
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import swarm_sim as SS                                           # noqa: E402
import swarm_robot_epde as SRE                                   # noqa: E402
from swarm_data import swarm_files, swarm_meta, ROBOT_DIAMETER_PX as SIGMA  # noqa: E402

GAP_EDGES = np.array([0.0, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.25, np.inf])


def nn_distances(pos):
    """Nearest-neighbour distances (robot diameters) of positions (T, N, 2), every 2nd step."""
    out = []
    for p in pos[::2]:
        d = np.linalg.norm(p[None] - p[:, None], axis=-1)
        d[np.diag_indices_from(d)] = np.inf
        out.append(d.min(1))
    return np.concatenate(out) / SIGMA


def gaps(pos):
    nn = nn_distances(pos)
    return dict(hist=(np.histogram(nn, GAP_EDGES)[0] / nn.size).tolist(), p01=float(np.percentile(nn, 0.1)))


def bond_profile(R, px, py, sub=slice(None, None, 5)):
    """Touching pairs: mean bond-normal relative velocity (>0 separating) per push band, data and law."""
    pos = np.stack([R["tok"][sub][..., SS.TI["x"]], R["tok"][sub][..., SS.TI["y"]]], -1)
    d = pos[:, None] - pos[:, :, None]
    D = np.linalg.norm(d, axis=-1)
    iu = np.triu_indices(D.shape[1], 1)
    u = d[:, iu[0], iu[1]] / D[:, iu[0], iu[1], None]
    band = np.digitize(D[:, iu[0], iu[1]] / SIGMA, SS.PUSH_EDGES) - 1
    out = {}
    for name, (vx, vy) in (("data", (R["fx"][sub], R["fy"][sub])), ("law", (px[sub], py[sub]))):
        v = np.stack([vx, vy], -1)
        rel = np.einsum("tpk,tpk->tp", v[:, iu[1]] - v[:, iu[0]], u)
        out[name] = [float(rel[band == b].mean()) for b in range(len(SS.PUSH_EDGES) - 1)]
    return out


def r2(R, px, py, rows):
    y = np.concatenate([R["fx"][rows].ravel(), R["fy"][rows].ravel()])
    e = y - np.concatenate([px.ravel(), py.ravel()])
    return float(1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2))


def heldout(R, K_of_rows, BP):
    T = R["tok"].shape[0]
    tr, te = slice(0, T // 2 - SS.LAG), slice(T // 2 + SS.LAG, T)
    K = K_of_rows(tr)
    NP = SS.neighbour_drive(R, K)
    Knp = SS.fit_drive(R, K, NP, (0, 1), tr)
    Kb, Knp_c = SS.fit_contact(R, K, NP, BP, rows=tr)
    return dict(base=r2(R, *SS.predict(R, K, te), te), drive=r2(R, *SS.predict_np(R, K, Knp, NP, te), te),
                contact=r2(R, *SS.predict_contact(R, K, Knp_c, NP, Kb, BP, te), te))


def sim(R, data, gap_data, K, sd, seeds=4, **kw):
    pos = [SS.simulate(R, K, sd, np.random.default_rng(s), **kw) for s in range(seeds)]
    c = SS.compare([SS.stats(p, R) for p in pos], data)
    g = [gaps(p) for p in pos]
    c["gap"] = np.mean([x["hist"] for x in g], 0).tolist()
    c["gap_tv"] = float(0.5 * np.abs(np.array(c["gap"]) - gap_data["hist"]).sum())
    c["gap_p01"] = float(np.mean([x["p01"] for x in g]))
    return c


def line(name, c):
    mr = c["msd_ratio"]
    return (f"  {name:32s} MSD " + " / ".join(f"{mr[L]:.2f}" for L in (1, 5, 25, 100, 250))
            + " | contacts " + "/".join(f"{v:.2f}" for v in c["contacts"]) + f" | layer TV {c['rhist_tv']:.3f}"
            + f" | NN-gap TV {c['gap_tv']:.3f}, NN p0.1 {c['gap_p01']:.3f}"
            + " | speed by contacts " + "/".join(f"{v:.0f}" for v in c["speed_by_n"])
            + f" | per robot MSD10 {c['corr']['msd10']:+.2f} radius {c['corr']['radius']:+.2f} tang {c['corr']['tang']:+.2f}")


def part_path(tag):
    return os.path.join(HERE, "results", f"swarm_contact_law_{tag}.json")


def merge():
    out = {}
    for f in swarm_files():
        tag = swarm_meta(f)["tag"]
        if os.path.exists(part_path(tag)):
            out.update(json.load(open(part_path(tag))))
    json.dump(out, open(os.path.join(HERE, "results", "swarm_contact_law.json"), "w"), indent=1, default=float)
    print(f"merged {len(out)} recordings")


def main(tags=None):
    runs = {(r["tag"], r["robot_id"], r["var"]): r for r in map(json.loads, open(SRE.RESULTS))
            if r["status"] == "ok"}
    load = lambda name: {(r["tag"], r["robot_id"], r["var"]): r                          # noqa: E731
                         for r in map(json.loads, open(os.path.join(HERE, "results", name)))}
    pipe, pipe_np = load("robot_pipeline.jsonl"), load("robot_pipeline_np.jsonl")
    for f in swarm_files():
        if tags and swarm_meta(f)["tag"] not in tags:
            continue
        R = SS.recording(f)
        BP = SS.recording_band_push(R)
        data = SS.stats(R["pos"][SS.HIST:], R)
        gap_data = gaps(R["pos"][SS.HIST:])
        guard = float(np.percentile(nn_distances(R["pos"]), 0.01) * SIGMA)
        res = dict(data={k: v for k, v in data.items() if k != "robot"}, gap=gap_data, guard=guard / SIGMA,
                   heldout={}, coef={}, profile={}, test={})
        print(f"\n### {R['tag']}  (hard-disk core {R['core'] / SIGMA:.3f}, guard {guard / SIGMA:.3f} robot diameters)")
        print(f"  {'data':32s} contacts " + "/".join(f"{v:.2f}" for v in data["contacts"])
              + " | NN-gap " + "/".join(f"{v:.3f}" for v in gap_data["hist"]) + f", NN p0.1 {gap_data['p01']:.3f}"
              + " | speed by contacts " + "/".join(f"{v:.0f}" for v in data["speed_by_n"]))
        K_own, K_epde = SS.fit_joint(R, True, False)[0], SS.epde_K(R, runs)
        Kp = SS.pipeline_K(R, pipe)
        Kq, Kq_np = SS.pipeline_K_np(R, pipe_np)
        laws = {"own_prop": (K_own, K_own, None, lambda rows: SS.fit_joint(R, True, False, rows)[0]),
                "epde": (K_epde, K_epde, None, lambda rows: K_epde),
                "pipeline": (Kp, Kq, Kq_np, None)}
        for name, (K0, K, Knp_own, K_of_rows) in laws.items():
            Ksrc = K if Knp_own is None else K_own                    # pipeline laws: p_j from own_prop
            NP = SS.neighbour_drive(R, Ksrc)
            if Knp_own is None:
                Knp = SS.fit_drive(R, K, NP, (0, 1))
                Kb, Knp_c = SS.fit_contact(R, K, NP, BP)
                res["heldout"][name] = h = heldout(R, K_of_rows, BP)
                print(f"  {name}: held-out R2 base {h['base']:.4f} | + neighbour drive {h['drive'] - h['base']:+.4f}"
                      f" | + contact law {h['contact'] - h['base']:+.4f}")
            else:
                Knp = Knp_c = Knp_own
                Kb = SS.fit_contact(R, K, NP, BP, Knp=Knp_own)[0]
            push = float(K[:, 0, SS.TI["push_x"]].mean())
            res["coef"][name] = dict(push=push, push_by_band=(push + Kb).tolist(),
                                     np=float(Knp_c[:, 0, 0].mean()), np_in=float(Knp_c[:, 0, 2].mean()))
            prof = bond_profile(R, *SS.predict_contact(R, K, Knp_c, NP, Kb, BP))
            prof["law_const_push"] = bond_profile(R, *SS.predict_np(R, K, Knp, NP))["law"]
            res["profile"][name] = prof
            bands = [f"[{a:.2f},{b:.2f})" for a, b in zip(SS.PUSH_EDGES[:-1], SS.PUSH_EDGES[1:])]
            print(f"  {name}: push by band (px/s) " + " ".join(f"{b} {v:+.1f}" for b, v in zip(bands, push + Kb))
                  + f" (constant push {push:+.1f}) | np {res['coef'][name]['np']:+.3f}, np_in "
                  f"{res['coef'][name]['np_in']:+.3f}")
            print(f"  {name}: touching pairs, bond-normal relative velocity (px/s) data / constant push / contact law: "
                  + "  ".join(f"{b} {x:+.0f}/{z:+.0f}/{y:+.0f}" for b, x, y, z in
                              zip(bands, prof["data"], prof["law"], prof["law_const_push"])))
            variants = {
                name: dict(K=K0, sd=SS.residual_sd(R, K0)),
                f"{name} + neighbour drive": dict(K=K, sd=SS.residual_sd_np(R, K, Knp, NP), Knp=Knp, Ksrc=Ksrc),
                f"{name} + drive, no projection": dict(K=K, sd=SS.residual_sd_np(R, K, Knp, NP), Knp=Knp, Ksrc=Ksrc,
                                                       core=0.0),
                f"{name} + contact law": dict(K=K, sd=SS.residual_sd_contact(R, K, Knp_c, NP, Kb, BP), Knp=Knp_c,
                                              Ksrc=Ksrc, Kb=Kb, core=0.0),
                f"{name} + contact law, guard": dict(K=K, sd=SS.residual_sd_contact(R, K, Knp_c, NP, Kb, BP),
                                                     Knp=Knp_c, Ksrc=Ksrc, Kb=Kb, core=guard)}
            for vname, kw in variants.items():
                res["test"][vname] = sim(R, data, gap_data, kw.pop("K"), kw.pop("sd"), **kw)
                print(line(vname, res["test"][vname]), flush=True)
        json.dump({R["tag"]: res}, open(part_path(R["tag"]), "w"), indent=1, default=float)
    if not tags:
        merge()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="", help="comma-separated recording tags (default: all)")
    ap.add_argument("--merge", action="store_true", help="only merge the per-recording files")
    a = ap.parse_args()
    merge() if a.merge else main([t for t in a.tags.split(",") if t])
