r"""Spinning caps: the cap's hard core and its contact friction, in the laws and the swarm simulation.

The robots are spinning cylinders with a cap 48.7 mm across and 19.2 mm high. Two consequences:
  hard core      the arenas (diameter D = 40 / 41 cm, centres reach D/2 - cap/2) give one camera scale,
                 45.08 px/cm (swarm_sim.px_per_cm), so the cap is 219.5 px in every recording;
                 the hard-disk step keeps centres one cap apart ('cap core') instead of below the
                 recording's closest distance only ('guard', swarm_contact_law.py)
  spin friction  sliding friction at a contact acts along the contact tangent (swarm_sim.SPIN_TOK):
                 gear / gear_sign between touching caps, roll / roll_sign at the wall
Laws: every robot's EPDE law + the contact law (distance-resolved push, np, np_in) + shared isotropic
spin-friction coefficients on its residuals; held-out first half -> second half per token set.
Simulation (4 seeds):
  --law epde      every robot its own EPDE equations: contact law, guard | contact law, cap core |
                  + spin friction, guard | + spin friction, cap core
  --law own_prop  one common system per recording (shared push, wall, persistence, constants; each
                  robot its own propulsion tokens): contact law, guard | contact law, cap core
Seed 0's trajectories of every variant and the recording are kept for the visual comparison
(results/viz/<tag>[_own_prop].npz, swarm_viz.py).

python swarm_spin_contact.py [--law epde|own_prop] [--tags N30_PWM10_D41,...]
    -> results/swarm_spin_contact[_own_prop]_<tag>.json
python swarm_spin_contact.py [--law ...] --merge   -> results/swarm_spin_contact[_own_prop].json
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
import swarm_contact_law as SCL                                  # noqa: E402
from swarm_data import swarm_files, swarm_meta, ROBOT_DIAMETER_PX as SIGMA  # noqa: E402

SETS = {"gear": (0,), "gear_sign": (1,), "roll": (2,), "roll_sign": (3,), "gear + roll": (0, 2), "all": (0, 1, 2, 3)}
VIZ = os.path.join(HERE, "results", "viz")


#: law -> file suffix. epde: every robot its own EPDE equations; own_prop: one common system per
#: recording (shared push, wall, persistence, constants) with each robot's own propulsion tokens
SUFFIX = {"epde": "", "own_prop": "_own_prop"}


def part_path(tag, law="epde"):
    return os.path.join(HERE, "results", f"swarm_spin_contact{SUFFIX[law]}_{tag}.json")


def contact_residuals(R, K, Knp, NP, Kb, BP, rows=slice(None)):
    px, py = SS.predict_contact(R, K, Knp, NP, Kb, BP, rows)
    return R["fx"][rows] - px, R["fy"][rows] - py


def spin_pred(ST, Ks, rows=slice(None)):
    return ST[rows][..., 0] @ Ks, ST[rows][..., 1] @ Ks


def heldout(R, K_of_rows, BP, ST):
    """K_of_rows(rows) -> the law fitted on those rows (a fixed per-robot EPDE law ignores them)."""
    T = R["tok"].shape[0]
    tr, te = slice(0, T // 2 - SS.LAG), slice(T // 2 + SS.LAG, T)
    K = K_of_rows(tr)
    NP = SS.neighbour_drive(R, K)
    Kb, Knp = SS.fit_contact(R, K, NP, BP, rows=tr)
    ex, ey = contact_residuals(R, K, Knp, NP, Kb, BP, tr)
    px, py = SS.predict_contact(R, K, Knp, NP, Kb, BP, te)
    out = {"law": SCL.r2(R, *SS.predict(R, K, te), te), "contact law": SCL.r2(R, px, py, te)}
    for name, which in SETS.items():
        Ks = SS.fit_spin(R, ex, ey, ST, which, tr)
        sx, sy = spin_pred(ST, Ks, te)
        out[name] = SCL.r2(R, px + sx, py + sy, te)
    return out


def sim(R, data, gap_data, K, sd, seeds=4, **kw):
    runs = [SS.simulate(R, K, sd, np.random.default_rng(s), return_phi=True, **kw) for s in range(seeds)]
    st = [SS.stats(p, R) for p, _ in runs]
    c = SS.compare(st, data)
    g = [SCL.gaps(p) for p, _ in runs]
    c["gap"] = np.mean([x["hist"] for x in g], 0).tolist()
    c["gap_tv"] = float(0.5 * np.abs(np.array(c["gap"]) - gap_data["hist"]).sum())
    c["gap_p01"] = float(np.mean([x["p01"] for x in g]))
    c["msd"] = {L: float(np.mean([s["msd"][L] for s in st])) for L in data["msd"]}
    c["rhist"] = np.mean([s["rhist"] for s in st], 0).tolist()
    return c, runs[0]


def run(f, law="epde"):
    R = SS.recording(f)
    BP, ST = SS.recording_band_push(R), SS.recording_spin_tokens(R)
    data = SS.stats(R["pos"][SS.HIST:], R)
    gap_data = SCL.gaps(R["pos"][SS.HIST:])
    guard = float(np.percentile(SCL.nn_distances(R["pos"]), 0.01) * SIGMA)
    cap = float(SS.cap_px(R))
    if law == "epde":
        runs = {(r["tag"], r["robot_id"], r["var"]): r for r in map(json.loads, open(SRE.RESULTS))
                if r["status"] == "ok"}
        K = SS.epde_K(R, runs)
        K_of_rows = lambda rows: K                                       # noqa: E731
    else:
        K = SS.fit_joint(R, True, False)[0]
        K_of_rows = lambda rows: SS.fit_joint(R, True, False, rows)[0]   # noqa: E731
    NP = SS.neighbour_drive(R, K)
    Kb, Knp = SS.fit_contact(R, K, NP, BP)
    ex, ey = contact_residuals(R, K, Knp, NP, Kb, BP)
    Ks = SS.fit_spin(R, ex, ey, ST, SETS["all"])
    sx, sy = spin_pred(ST, Ks)
    sd_c = np.stack([ex.std(0), ey.std(0)], -1)
    sd_s = np.stack([(ex - sx).std(0), (ey - sy).std(0)], -1)
    h = heldout(R, K_of_rows, BP, ST)
    res = dict(tag=R["tag"], law=law, cap_px=cap, cap_sigma=cap / SIGMA, guard_sigma=guard / SIGMA,
               px_per_cm=cap / SS.CAP_CM, heldout=h, spin_coef=dict(zip(SS.SPIN_TOK, Ks.tolist())),
               data={k: v for k, v in data.items() if k != "robot"}, gap=gap_data, test={})
    print(f"\n### {R['tag']} ({law}): cap {cap:.0f} px = {cap / SIGMA:.3f} sigma ({cap / SS.CAP_CM:.2f} px/cm), "
          f"guard {guard / SIGMA:.3f} sigma")
    print("  held-out R2 law {:.4f}, contact law {:.4f} | ".format(h["law"], h["contact law"])
          + " | ".join(f"+ {k} {h[k] - h['contact law']:+.4f}" for k in SETS))
    print("  spin-friction coefficients (all four): " + ", ".join(f"{k} {v:+.2f}" for k, v in zip(SS.SPIN_TOK, Ks)))
    variants = {"contact law, guard": dict(sd=sd_c, core=guard),
                "contact law, cap core": dict(sd=sd_c, core=cap)}
    if law == "epde":
        variants.update({"contact law + spin friction, guard": dict(sd=sd_s, core=guard, Ks=Ks),
                         "contact law + spin friction, cap core": dict(sd=sd_s, core=cap, Ks=Ks)})
    os.makedirs(VIZ, exist_ok=True)
    keep = dict(data_pos=R["pos"][SS.HIST:].astype(np.float32), data_phi=R["phi"][SS.HIST:].astype(np.float32),
                w=R["w"].astype(np.float32), meta=np.array([R["xc"], R["yc"], R["r_env"], cap, R["dt"]]))
    for i, (name, kw) in enumerate(variants.items()):
        c, (pos, phi) = sim(R, data, gap_data, K, kw.pop("sd"), Knp=Knp, Ksrc=K, Kb=Kb, **kw)
        res["test"][name] = c
        keep[f"sim{i}_pos"], keep[f"sim{i}_phi"] = pos.astype(np.float32), phi.astype(np.float32)
        print(SCL.line(name, c), flush=True)
    keep["names"] = np.array(list(variants))
    np.savez_compressed(os.path.join(VIZ, f"{R['tag']}{SUFFIX[law]}.npz"), **keep)
    json.dump({R["tag"]: res}, open(part_path(R["tag"], law), "w"), indent=1, default=float)


def merge(law="epde"):
    out = {}
    for f in swarm_files():
        p = part_path(swarm_meta(f)["tag"], law)
        if os.path.exists(p):
            out.update(json.load(open(p)))
    json.dump(out, open(os.path.join(HERE, "results", f"swarm_spin_contact{SUFFIX[law]}.json"), "w"), indent=1,
              default=float)
    print(f"merged {len(out)} recordings ({law})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", default="")
    ap.add_argument("--law", default="epde", choices=list(SUFFIX))
    ap.add_argument("--merge", action="store_true")
    a = ap.parse_args()
    if a.merge:
        merge(a.law)
    else:
        tags = [t for t in a.tags.split(",") if t]
        for f in swarm_files():
            if not tags or swarm_meta(f)["tag"] in tags:
                run(f, a.law)
        if not tags:
            merge(a.law)
