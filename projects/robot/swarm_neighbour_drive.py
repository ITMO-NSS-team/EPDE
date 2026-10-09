r"""Neighbour-drive tokens in the laws and in the swarm simulation.

Robot j's own propulsion p_j = v0_j e(phi_j + theta_j) is read off j's own law (its c, s terms):
    np_i    = sum over robots touching i of p_j                       (neighbours' drive)
    np_in_i = sum over touching j of max(0, p_j . u_ji) u_ji          (the part driving INTO robot i)
Local and identity-free: the same token for any number of robots.
1. Laws. Shared isotropic coefficients on top of the own-propulsion law and of every robot's EPDE
   equations; held-out first half -> second half (np alone, np_in alone, both). Every robot's own
   pipeline law with the tokens offered to subset selection (results/robot_pipeline_np.jsonl).
2. Simulation (4 seeds, independent noise): each law with and without the tokens, recomputed from
   the simulated state every step; the swarm statistics of swarm_sim.py.

python swarm_neighbour_drive.py      -> results/swarm_neighbour_drive.json
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import swarm_sim as SS                                           # noqa: E402
import swarm_robot_epde as SRE                                   # noqa: E402
from swarm_data import swarm_files                               # noqa: E402

VARIANTS = {"np": (0,), "np_in": (1,), "both": (0, 1)}


def r2(R, K, Knp, NP, rows):
    px, py = SS.predict_np(R, K, Knp, NP, rows) if Knp is not None else SS.predict(R, K, rows)
    y = np.concatenate([R["fx"][rows].ravel(), R["fy"][rows].ravel()])
    e = y - np.concatenate([px.ravel(), py.ravel()])
    return float(1 - np.sum(e ** 2) / np.sum((y - y.mean()) ** 2))


def heldout(R, K_of_rows):
    """K_of_rows(rows) -> law fitted on those rows (a fixed law ignores them). Returns R2 on the
    second half for the base law and each variant, coefficients fitted on the first half."""
    T = R["tok"].shape[0]
    tr, te = slice(0, T // 2 - SS.LAG), slice(T // 2 + SS.LAG, T)
    K = K_of_rows(tr)
    NP = SS.neighbour_drive(R, K)
    out = {"base": r2(R, K, None, NP, te)}
    for name, which in VARIANTS.items():
        out[name] = r2(R, K, SS.fit_drive(R, K, NP, which, tr), NP, te)
    return out


def sim(R, data, K, sd, Knp=None, Ksrc=None, seeds=4):
    sims = [SS.stats(SS.simulate(R, K, sd, np.random.default_rng(s), Knp=Knp, Ksrc=Ksrc), R) for s in range(seeds)]
    return SS.compare(sims, data)


def line(name, c):
    mr = c["msd_ratio"]
    return (f"  {name:24s} MSD " + " / ".join(f"{mr[L]:.2f}" for L in (1, 5, 25, 100, 250))
            + " | contacts " + "/".join(f"{v:.2f}" for v in c["contacts"]) + f" | layer TV {c['rhist_tv']:.3f}"
            + " | speed by contacts " + "/".join(f"{v:.0f}" for v in c["speed_by_n"])
            + f" | per robot MSD10 {c['corr']['msd10']:+.2f} radius {c['corr']['radius']:+.2f} tang {c['corr']['tang']:+.2f}")


def main():
    runs = {(r["tag"], r["robot_id"], r["var"]): r for r in map(json.loads, open(SRE.RESULTS))
            if r["status"] == "ok"}
    load = lambda name: {(r["tag"], r["robot_id"], r["var"]): r                          # noqa: E731
                         for r in map(json.loads, open(os.path.join(HERE, "results", name)))}
    pipe, pipe_np = load("robot_pipeline.jsonl"), load("robot_pipeline_np.jsonl")
    out = {}
    for f in swarm_files():
        R = SS.recording(f)
        data = SS.stats(R["pos"][SS.HIST:], R)
        res = dict(data={k: v for k, v in data.items() if k != "robot"}, heldout={}, coef={}, test={})
        print(f"\n### {R['tag']}")
        print(f"  {'data':24s} contacts " + "/".join(f"{v:.2f}" for v in data["contacts"])
              + " | speed by contacts " + "/".join(f"{v:.0f}" for v in data["speed_by_n"]))
        K_own, K_epde = SS.fit_joint(R, True, False)[0], SS.epde_K(R, runs)
        laws = {"own_prop": (K_own, lambda rows: SS.fit_joint(R, True, False, rows)[0]),
                "epde": (K_epde, lambda rows: K_epde)}                 # EPDE laws are fixed: only Knp is fitted
        for name, (K, K_of_rows) in laws.items():
            h = heldout(R, K_of_rows)
            res["heldout"][name] = h
            NP = SS.neighbour_drive(R, K)
            Knp = SS.fit_drive(R, K, NP, (0, 1))
            res["coef"][name] = dict(np=float(Knp[0, 0, 0]), np_in=float(Knp[0, 0, 2]))
            print(f"  {name}: held-out R2 base {h['base']:.4f} | + np {h['np'] - h['base']:+.4f}, + np_in "
                  f"{h['np_in'] - h['base']:+.4f}, + both {h['both'] - h['base']:+.4f} | coefficients np "
                  f"{Knp[0, 0, 0]:+.3f}, np_in {Knp[0, 0, 2]:+.3f}")
            res["test"][name] = sim(R, data, K, SS.residual_sd(R, K))
            res["test"][f"{name} + neighbour drive"] = sim(R, data, K, SS.residual_sd_np(R, K, Knp, NP), Knp, K)
            print(line(name, res["test"][name]))
            print(line(f"{name} + neighbour drive", res["test"][f"{name} + neighbour drive"]), flush=True)
        # every robot's own pipeline law, with and without the tokens available to subset selection
        Kp = SS.pipeline_K(R, pipe)
        Kq, Kq_np = SS.pipeline_K_np(R, pipe_np)
        NP = SS.neighbour_drive(R, K_own)
        res["test"]["pipeline"] = sim(R, data, Kp, SS.residual_sd(R, Kp))
        res["test"]["pipeline + neighbour drive"] = sim(R, data, Kq, SS.residual_sd_np(R, Kq, Kq_np, NP), Kq_np, K_own)
        print(line("pipeline", res["test"]["pipeline"]))
        print(line("pipeline + neighbour drive", res["test"]["pipeline + neighbour drive"]), flush=True)
        out[R["tag"]] = res
        json.dump(out, open(os.path.join(HERE, "results", "swarm_neighbour_drive.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
