r"""Is the HEADING the missing piece of the swarm's long-time mobility?

swarm_sim.py turns every robot at its own constant spin w_i plus independent phase noise, so its
propulsion v0_i e(phi_i + theta_i) just rotates: the robot circles in place. Real spin may slow,
stall or speed up for seconds (e.g. in contact), and a stalled heading drives a straight run.
1. Data: per robot, the spin-rate deviation  dphi/dt - w_i  over 0.4 s steps -- its spread, its
   autocorrelation at 0.4 .. 10 s, and the fraction of steps with |dphi/dt| < |w_i| / 2 (stalls).
2. Simulation (4 seeds, independent noise): the same laws with the constant-spin heading vs the
   RECORDED heading replayed -- MSD ratios sim/data, 4+ contacts, per-robot correlations.

python swarm_heading_replay.py      -> results/swarm_heading_replay.json
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

ACF_LAGS = [1, 2, 5, 10, 25]                                     # steps of 0.4 s


def spin_stats(R):
    rate = np.diff(R["phi"], axis=0) / R["dt"]                   # (steps, N)
    dev = rate - R["w"][None]
    dev_c = dev - dev.mean(0)
    acf = [float(np.mean(dev_c[L:] * dev_c[:-L]) / np.mean(dev_c ** 2)) for L in ACF_LAGS]
    stall = float(np.mean(np.abs(rate) < 0.5 * np.abs(R["w"][None])))
    rel_sd = float(np.median(dev.std(0) / np.abs(R["w"])))
    return dict(acf=acf, stall=stall, rel_sd=rel_sd)


def main():
    runs = {(r["tag"], r["robot_id"], r["var"]): r for r in map(json.loads, open(SRE.RESULTS))
            if r["status"] == "ok"}
    pipe = {(r["tag"], r["robot_id"], r["var"]): r
            for r in map(json.loads, open(os.path.join(HERE, "results", "robot_pipeline.jsonl")))}
    out = {}
    for f in swarm_files():
        R = SS.recording(f)
        data = SS.stats(R["pos"][SS.HIST:], R)
        sp = spin_stats(R)
        res = dict(spin=sp, test={})
        print(f"\n### {R['tag']}: spin-rate deviation sd {100 * sp['rel_sd']:.0f}% of |w| (median robot); "
              f"its autocorrelation at 0.4/0.8/2/4/10 s " + " / ".join(f"{v:+.2f}" for v in sp["acf"])
              + f"; stalled (|dphi/dt| < |w|/2) {100 * sp['stall']:.1f}% of steps")
        laws = {"epde": SS.epde_K(R, runs), "own_prop": SS.fit_joint(R, True, False)[0],
                "pipeline": SS.pipeline_K(R, pipe)}
        for name, K in laws.items():
            sd = SS.residual_sd(R, K)
            for label, replay in (("constant spin", False), ("recorded heading", True)):
                sims = [SS.stats(SS.simulate(R, K, sd, np.random.default_rng(s), phi_replay=replay), R)
                        for s in range(4)]
                c = SS.compare(sims, data)
                res["test"][f"{name}|{label}"] = c
                mr = c["msd_ratio"]
                print(f"  {name:9s} {label:17s} MSD " + " / ".join(f"{mr[L]:.2f}" for L in (1, 5, 25, 100, 250))
                      + f" | 4+ {c['contacts'][4]:.2f} (data {data['contacts'][4]:.2f})"
                      + f" | per robot MSD10 {c['corr']['msd10']:+.2f} radius {c['corr']['radius']:+.2f}"
                      + f" tang {c['corr']['tang']:+.2f}", flush=True)
        out[R["tag"]] = res
        json.dump(out, open(os.path.join(HERE, "results", "swarm_heading_replay.json"), "w"), indent=1,
                  default=float)


if __name__ == "__main__":
    main()
