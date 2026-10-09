r"""Does neighbour-shared noise restore the swarm's long-time mobility?

In the crowded recordings the laws' residuals of touching robots are correlated (+0.10..+0.18),
but the simulation (swarm_sim.py) gives every robot independent noise. Hard-disk projection already
couples touching robots, so the correlation is measured the same way in the simulated swarm: the
residual of the same law, by pair distance. Shared noise = a smooth random field per step:
    corr(xi_i, xi_j) = lam exp(-d_ij^2 / 2 ell^2),  unit variance per robot (total noise unchanged).
1. Calibration (EPDE per-robot laws, 2 seeds): grid over (lam, ell); pick the setting whose
   simulated residual-correlation curve matches the recording's.
2. Test (4 seeds): independent vs calibrated shared noise for the EPDE, own-propulsion and pipeline
   laws: MSD ratios sim/data, 4+ contacts, per-robot correlations.

python swarm_shared_noise.py        -> results/swarm_shared_noise.json
"""
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import swarm_sim as SS                                           # noqa: E402
import swarm_robot_epde as SRE                                   # noqa: E402
from swarm_data import swarm_files, ROBOT_DIAMETER_PX as SIGMA   # noqa: E402

BINS = [0, 1.1, 1.5, 2.0, 3.0, np.inf]                           # pair distance in robot diameters
LABELS = ["contact", "1.1-1.5", "1.5-2", "2-3", ">3"]
GRID = [(0.0, 1.0)] + [(lam, ell) for lam in (0.25, 0.5, 0.75) for ell in (150.0, 250.0, 400.0)]


def corr_by_distance(pos, e):
    """<e_i . e_j> / <|e|^2> over pairs in each distance bin; pos (S, N, 2), e (S, N, 2)."""
    den = np.mean(np.sum(e ** 2, -1))
    sums, cnts = np.zeros(len(LABELS)), np.zeros(len(LABELS))
    iu = np.triu_indices(pos.shape[1], 1)
    for p, ee in zip(pos, e):
        d = np.linalg.norm(p[:, None] - p[None], axis=-1)[iu] / SIGMA
        dot = (ee @ ee.T)[iu]
        b = np.digitize(d, BINS) - 1
        sums += np.bincount(b, dot, len(LABELS))
        cnts += np.bincount(b, minlength=len(LABELS))
    return (sums / np.maximum(cnts, 1) / den).tolist()


def data_corr(R, K):
    px, py = SS.predict(R, K)
    e = np.stack([R["fx"] - px, R["fy"] - py], -1)[::SS.LAG]
    pos = np.stack([R["tok"][..., SS.TI["x"]], R["tok"][..., SS.TI["y"]]], -1)[::SS.LAG]
    return corr_by_distance(pos, e)


def run(R, K, sd, shared, seeds, data):
    sims, curves = [], []
    for s in range(seeds):
        pos, res = SS.simulate(R, K, sd, np.random.default_rng(s), shared=shared, track=True)
        sims.append(SS.stats(pos, R))
        curves.append(corr_by_distance(pos[:-1], res))
    c = SS.compare(sims, data)
    c["corr_curve"] = np.mean(curves, 0).tolist()
    return c


def line(name, c):
    mr = c["msd_ratio"]
    return (f"  {name:28s} corr " + " ".join(f"{v:+.3f}" for v in c["corr_curve"])
            + " | MSD " + " / ".join(f"{mr[L]:.2f}" for L in (1, 5, 25, 100, 250) if L in mr)
            + f" | 4+ {c['contacts'][4]:.2f} | per robot MSD10 {c['corr']['msd10']:+.2f} radius {c['corr']['radius']:+.2f}")


def main():
    runs = {(r["tag"], r["robot_id"], r["var"]): r for r in map(json.loads, open(SRE.RESULTS))
            if r["status"] == "ok"}
    pipe = {(r["tag"], r["robot_id"], r["var"]): r
            for r in map(json.loads, open(os.path.join(HERE, "results", "robot_pipeline.jsonl")))}
    out = {}
    for f in swarm_files():
        t0 = time.time()
        R = SS.recording(f)
        data = SS.stats(R["pos"][SS.HIST:], R)
        laws = {"epde": SS.epde_K(R, runs), "own_prop": SS.fit_joint(R, True, False)[0],
                "pipeline": SS.pipeline_K(R, pipe)}
        sds = {k: SS.residual_sd(R, K) for k, K in laws.items()}
        dc = data_corr(R, laws["epde"])
        res = dict(data_corr=dc, data_contacts4=data["contacts"][4], grid={}, test={})
        print(f"\n### {R['tag']}   pair distance bins (diameters): " + " ".join(f"{b:>7s}" for b in LABELS))
        print(f"  {'data (EPDE-law residuals)':28s} corr " + " ".join(f"{v:+.3f}" for v in dc)
              + f" | 4+ contacts {data['contacts'][4]:.2f}")
        # 1. calibration on the EPDE laws
        for lam, ell in GRID:
            c = run(R, laws["epde"], sds["epde"], None if lam == 0 else (lam, ell), 2, data)
            res["grid"][f"{lam},{ell}"] = c
            print(line(f"epde lam {lam:.2f} ell {ell:.0f}px" if lam else "epde independent", c), flush=True)
        miss = {k: float(np.sum((np.array(c["corr_curve"][:4]) - np.array(dc[:4])) ** 2))
                for k, c in res["grid"].items()}
        best = min(miss, key=miss.get)
        lam, ell = map(float, best.split(","))
        res["calibrated"] = dict(lam=lam, ell=ell)
        print(f"  calibrated: lam {lam:.2f}, ell {ell:.0f} px (closest correlation curve)")
        # 2. test, 4 seeds
        for name, K in laws.items():
            for label, shared in (("independent", None), ("shared", None if lam == 0 else (lam, ell))):
                c = run(R, K, sds[name], shared, 4, data)
                res["test"][f"{name}|{label}"] = c
                print(line(f"{name} {label} noise", c), flush=True)
        out[R["tag"]] = res
        json.dump(out, open(os.path.join(HERE, "results", "swarm_shared_noise.json"), "w"), indent=1, default=float)
        print(f"  ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
