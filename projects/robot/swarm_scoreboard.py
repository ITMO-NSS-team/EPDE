r"""Which simulated swarm dynamics matches the experiment best? One scoreboard over every run.

Reads results/swarm_sim.json (laws +/- slow memory), results/swarm_shared_noise.json (shared noise),
results/swarm_heading_replay.json (recorded heading, a diagnostic: the heading is not simulated),
results/swarm_neighbour_drive.json (neighbour-drive tokens), results/swarm_contact_law.json (the
learned contact law in place of the hard-disk projection) and results/swarm_spin_contact.json (the
cap's hard core, spin friction), when present.
Per law, averaged over the 5 recordings:
  per-robot correlation sim vs experiment   MSD over 10 s, mean radius, mean tangential velocity
  MSD error            mean |ln(sim / exp)| over lags 0.4 .. 100 s
  contacts TV          total variation distance of the 0/1/2/3/4+ contact fractions
  layering TV          total variation distance of the radial (r/R) histogram
  speed error          mean |ln(sim / exp)| of the speed in each contact class
Rank = mean rank over the 7 measures (1 = best).

python swarm_scoreboard.py
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
METRICS = ["r_msd10", "r_radius", "r_tang", "msd_err", "contact_tv", "layer_tv", "speed_err"]
HIGHER_BETTER = {"r_msd10", "r_radius", "r_tang"}


def measures(c, data):
    ratios = [v for v in c["msd_ratio"].values()]
    sp = [abs(np.log(s / d)) for s, d in zip(c["speed_by_n"], data["speed_by_n"]) if s and d and np.isfinite(s * d)]
    return dict(r_msd10=c["corr"]["msd10"], r_radius=c["corr"]["radius"], r_tang=c["corr"]["tang"],
                msd_err=float(np.mean(np.abs(np.log(ratios)))),
                contact_tv=float(0.5 * np.sum(np.abs(np.array(c["contacts"]) - np.array(data["contacts"])))),
                layer_tv=c["rhist_tv"], speed_err=float(np.mean(sp)))


def main():
    sim = json.load(open(os.path.join(RES, "swarm_sim.json")))
    shared = json.load(open(os.path.join(RES, "swarm_shared_noise.json")))
    replay = json.load(open(os.path.join(RES, "swarm_heading_replay.json")))
    drive_path = os.path.join(RES, "swarm_neighbour_drive.json")
    drive = json.load(open(drive_path)) if os.path.exists(drive_path) else {}
    contact_path = os.path.join(RES, "swarm_contact_law.json")
    contact = json.load(open(contact_path)) if os.path.exists(contact_path) else {}
    spin_path = os.path.join(RES, "swarm_spin_contact.json")
    spin = json.load(open(spin_path)) if os.path.exists(spin_path) else {}
    own_path = os.path.join(RES, "swarm_spin_contact_own_prop.json")
    own = json.load(open(own_path)) if os.path.exists(own_path) else {}
    rows = {}
    for tag, r in sim.items():
        for law, c in spin.get(tag, {}).get("test", {}).items():
            if "cap core" in law or "spin friction" in law:
                rows.setdefault(f"epde + {law}", []).append(measures(c, r["data"]))
        for law, c in own.get(tag, {}).get("test", {}).items():
            if "cap core" in law:
                rows.setdefault(f"own_prop + {law}", []).append(measures(c, r["data"]))
    for tag, r in sim.items():
        data = r["data"]
        for law, c in drive.get(tag, {}).get("test", {}).items():
            if law.endswith("neighbour drive"):
                rows.setdefault(law, []).append(measures(c, data))
        for law, c in contact.get(tag, {}).get("test", {}).items():
            if "no projection" in law or "contact law" in law:
                rows.setdefault(law, []).append(measures(c, data))
        for law, c in r.items():
            if isinstance(c, dict) and "corr" in c:
                rows.setdefault(law, []).append(measures(c, data))
        for key, c in shared.get(tag, {}).get("test", {}).items():
            law, noise = key.split("|")
            if noise == "shared":
                rows.setdefault(f"{law} + shared noise", []).append(measures(c, data))
        for key, c in replay.get(tag, {}).get("test", {}).items():
            law, head = key.split("|")
            if head == "recorded heading":
                rows.setdefault(f"{law} + recorded heading*", []).append(measures(c, data))
    mean = {law: {m: float(np.mean([x[m] for x in v])) for m in METRICS} for law, v in rows.items() if len(v) == 5}
    ranks = {law: [] for law in mean}
    for m in METRICS:
        order = sorted(mean, key=lambda k: -mean[k][m] if m in HIGHER_BETTER else mean[k][m])
        for i, law in enumerate(order, 1):
            ranks[law].append(i)
    score = {law: float(np.mean(r)) for law, r in ranks.items()}
    print(f"{'dynamics':30s} {'rank':>5s} | per-robot corr: {'MSD10':>6s} {'radius':>6s} {'tang':>6s} | "
          f"{'MSD err':>7s} {'contactTV':>9s} {'layerTV':>7s} {'speed err':>9s}")
    for law in sorted(score, key=score.get):
        m = mean[law]
        print(f"{law:30s} {score[law]:5.1f} |                 {m['r_msd10']:+6.2f} {m['r_radius']:+6.2f} "
              f"{m['r_tang']:+6.2f} | {m['msd_err']:7.3f} {m['contact_tv']:9.3f} {m['layer_tv']:7.3f} "
              f"{m['speed_err']:9.3f}")
    print("\n* recorded heading = the experiment's own headings replayed (a diagnostic, not a model)")
    print("best per measure: " + "; ".join(
        f"{m} {max(mean, key=lambda k: mean[k][m]) if m in HIGHER_BETTER else min(mean, key=lambda k: mean[k][m])}"
        for m in METRICS))


if __name__ == "__main__":
    main()
