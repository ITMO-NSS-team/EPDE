r"""The EPDE-format systems with every token written as its mathematical formula.

Reads the EPDE-format texts (epde_systems.py, epde_style.py) and replaces each token
'name{power: p}' by its definition, coefficients unchanged. dx/dx0 means a different difference in
each part of epde_style_per_file.txt: the 0.4 s forward increment in the robot systems, the 0.08 s
window velocity in the velocity-level fronts.

python epde_formulas.py      -> results/epde_systems_<recording>_formulas.txt,
                                results/epde_style_per_file_formulas.txt
"""
import glob
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

PUSH = "Σ_{j: r_ij<1.1σ} (V_i−V_j)/r_ij"
WALL = "H(ρ_i−R+σ/4)·(−V_i/ρ_i)"
COMMON = {
    "c": "cos φ_i", "s": "sin φ_i",
    "dc/dx0": "d(cos φ_i)/dt", "ds/dx0": "d(sin φ_i)/dt",
    "push_x": PUSH.replace("V", "x"), "push_y": PUSH.replace("V", "y"),
    "wall_x": WALL.replace("V", "x"), "wall_y": WALL.replace("V", "y"),
    "vb_x": "(x_i(t)−x_i(t−τ))/τ", "vb_y": "(y_i(t)−y_i(t−τ))/τ",
    "x": "x_i", "y": "y_i", "t": "t",
    "n_contact": "Σ_j H(1.1σ−r_ij)", "wall": "H(ρ_i−R+σ/4)", "r/R": "ρ_i/R",
    "|vb|": "|r_i(t)−r_i(t−τ)|/τ", "|w|": "|ω_i|",
    "D": "(τ/4)·|(r_i(t+τ)−r_i(t))/τ − f_i|²",
}
CONTEXT = {
    "drift": {"dx/dx0": "(x_i(t+τ)−x_i(t))/τ", "dy/dx0": "(y_i(t+τ)−y_i(t))/τ"},
    "velocity": {"dx/dx0": "(x_i(t+w)−x_i(t))/w", "dy/dx0": "(y_i(t+w)−y_i(t))/w",
                 "d^2x/dx0^2": "(x_i(t+2w)−2x_i(t+w)+x_i(t))/w²",
                 "d^2y/dx0^2": "(y_i(t+2w)−2y_i(t+w)+y_i(t))/w²"},
}
CONTEXT["noise"] = CONTEXT["drift"]
SIMPLE = re.compile(r"^(cos φ_i|sin φ_i|[xy]_i|t|\|ω_i\|)$")
TOKEN = re.compile(r"(\S+)\{power: ([0-9.]+)\}")

LEGEND = """Tokens written as formulas (robot i at time t):
  cos φ_i, sin φ_i     heading of the robot's marker
  x_i, y_i             centre position relative to the arena centre (lone robot: to its track mean);
                       r_i = (x_i, y_i), ρ_i = |r_i|, R = arena radius of robot centres
  r_ij = |r_i − r_j|   centre distance to robot j;  σ = 230 px, the robot diameter (≈ 5 cm)
  H(z)                 step: 1 for z > 0, else 0
  τ = 0.4 s (10 frames), w = 0.08 s (2 frames); ω_i = the robot's own mean spin rate
  f_i                  the drift law's prediction of (r_i(t+τ)−r_i(t))/τ (noise law only)
Token -> formula:
  c, s                 cos φ_i, sin φ_i
  push_x, push_y       {push_x},  {push_y}
  wall_x, wall_y       {wall_x},  {wall_y}
  vb_x, vb_y           {vb_x},  {vb_y}
  x, y                 x_i, y_i
  dc/dx0, ds/dx0       d(cos φ_i)/dt, d(sin φ_i)/dt   (central derivatives)
  dx/dx0, dy/dx0       robot systems: {fx},  {fy}   (forward increments)
                       velocity level: {vx},  {vy}
  d^2x/dx0^2           {ax}   (and the same in y)
  n_contact            {nc}   (number of touching neighbours)
  wall, r/R            {wall},  ρ_i/R
  |vb|, |w|            {vb},  |ω_i|
  D                    {D}   (diffusion, px^2/s)
Units: positions px, velocities px/s, rates rad/s. Coefficients are EPDE's, unchanged.
""".format(fx=CONTEXT["drift"]["dx/dx0"], fy=CONTEXT["drift"]["dy/dx0"], vx=CONTEXT["velocity"]["dx/dx0"],
           vy=CONTEXT["velocity"]["dy/dx0"], ax=CONTEXT["velocity"]["d^2x/dx0^2"], nc=COMMON["n_contact"],
           wall=COMMON["wall"], vb=COMMON["|vb|"], D=COMMON["D"],
           **{k: COMMON[k] for k in ("push_x", "push_y", "wall_x", "wall_y", "vb_x", "vb_y")})


def formula(name, power, ctx):
    f = CONTEXT[ctx].get(name, COMMON.get(name))
    if f is None:
        raise KeyError(f"no formula for token {name!r} ({ctx})")
    p = float(power)
    if p != 1.0:
        return f"({f})" + ("²" if p == 2.0 else f"^{p:g}")
    return f if SIMPLE.match(f) else f"[{f}]"


def convert(text, sectioned):
    out, ctx = [], "drift"
    for line in text.splitlines():
        if sectioned:
            if line.startswith(" velocity level"):
                ctx = "velocity"
            elif line.startswith(" noise"):
                ctx = "noise"
            elif line.startswith("====") or line.startswith(" robot "):
                ctx = "drift"
        out.append(TOKEN.sub(lambda m: formula(m.group(1), m.group(2), ctx), line))
    return "\n".join(out) + "\n"


def main():
    jobs = [(p, False) for p in sorted(glob.glob(os.path.join(RESULTS, "epde_systems_*.txt")))
            if not p.endswith("_formulas.txt")]
    jobs.append((os.path.join(RESULTS, "epde_style_per_file.txt"), True))
    for path, sectioned in jobs:
        text = convert(open(path, encoding="utf-8").read(), sectioned)
        assert "{power:" not in text, path
        out = path[:-4] + "_formulas.txt"
        with open(out, "w", encoding="utf-8") as fh:
            fh.write(LEGEND + "\n" + text)
        print("written", out)


if __name__ == "__main__":
    main()
