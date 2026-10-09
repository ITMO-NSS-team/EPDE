r"""EPDE-format systems, one file per swarm recording, one system per robot.

Each robot's system is assembled from its own EPDE searches:
  dc/dx0, ds/dx0   heading rotation, from swarm_spin_epde.py (per robot, central derivatives; the
                   first system of the robot's front, exactly as EPDE printed it)
  dx/dx0, dy/dx0   drift, from swarm_robot_epde.py (per robot and coordinate, forward increments over
                   0.4 s, own tokens: c, s, push, wall, vb, own position; the lowest-discrepancy member
                   of each front that describes the derivative)
Objective function values: those of the two drift equations, [discrepancy dx, dy, instability dx, dy]
(EPDE defaults; the instability is large and noisy on these stochastic targets). The spin runs stored
their front texts only.

python epde_systems.py      -> results/epde_systems_<recording>.txt
"""
import csv
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import swarm_robot_epde as SRE                                   # noqa: E402
from swarm_data import swarm_files, swarm_meta                   # noqa: E402


def spin_equations(front):
    """The first system of a spin front ('/ eq  \\ eq || ...') -> its two equations."""
    first = front.split(" || ")[0]
    return [e.strip(" /") for e in first.split("\\") if " = " in e]


def drift_equation(row):
    """Lowest-discrepancy front member that describes the derivative: (text, objectives)."""
    eqs = [s for s in row["front"] if SRE.normal_form(s["text"], row["var"]) is not None]
    best = min(eqs, key=lambda s: s["obj"][0])
    return best["text"], best["obj"]


def main():
    spin = {(r["tag"], int(r["robot_id"])): r for r in csv.DictReader(open(os.path.join(HERE, "results",
                                                                                         "swarm_spin_epde.csv")))}
    drift = {(r["tag"], r["robot_id"], r["var"]): r for r in map(json.loads, open(SRE.RESULTS))
             if r["status"] == "ok"}
    for path in swarm_files():
        md = swarm_meta(path)
        tag = md["tag"]
        ids = sorted({rid for (tg, rid) in spin if tg == tag})
        out = os.path.join(HERE, "results", f"epde_systems_{tag}.txt")
        with open(out, "w", encoding="utf-8") as fh:
            fh.write(f"{os.path.basename(path)}\n{md['n']} robots, PWM {md['pwm']}, arena {md['arena_cm']} cm, "
                     f"{md['ncw']} cw / {md['nccw']} ccw\n" + __doc__.split("python")[0] + "\n")
            for rid in ids:
                s, rx, ry = spin[(tag, rid)], drift.get((tag, rid, "x")), drift.get((tag, rid, "y"))
                eqs = spin_equations(s["front"])
                objs = []
                for r in (rx, ry):
                    text, obj = drift_equation(r)
                    eqs.append(text)
                    objs.append(obj)
                ex = SRE.most_complete(rx)[0]
                ey = SRE.most_complete(ry)[0]
                p = SRE.propulsion(ex and ex[0], ey and ey[0])
                head = (f"robot {rid}: spin (phase slope) {float(s['omega_phase']):+.3f} rad/s, EPDE "
                        f"{float(s['omega_c']):+.3f} / {float(s['omega_s']):+.3f} rad/s; own propulsion "
                        + (f"v0 {p[0]:.1f} px/s, theta {p[1]:+.1f} deg" if p else "axis term only (offset not resolved)"))
                marks = [" / "] + [" | "] * (len(eqs) - 2) + [" \\ "]
                body = "".join(m + SRE.epde_text(e) + "\n" for m, e in zip(marks, eqs))
                vals = [objs[0][0], objs[1][0], objs[0][1], objs[1][1]]
                fh.write(f"\n{head}\n{body.rstrip()} , with objective function values of "
                         f"[{' '.join(f'{v:.4g}' for v in vals)}]\n")
        print(f"written {out} ({len(ids)} robots)")


if __name__ == "__main__":
    main()
