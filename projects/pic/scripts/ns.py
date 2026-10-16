"""Navier-Stokes, flow past a cylinder.

Data set in this folder (all of them: projects/pic/DATASETS.md):

  ns: Navier-Stokes, cylinder wake (Re = 100)
      truth:
          -1.0 * u{power: 1.0} * du/dx2{power: 1.0} + -1.0 * v{power: 1.0} * du/dx1{power: 1.0} + -1.0 * dp/dx2{power: 1.0} + 0.01 * d^2u/dx2^2{power: 1.0} + 0.01 * d^2u/dx1^2{power: 1.0} = du/dx0{power: 1.0}
          -1.0 * u{power: 1.0} * dv/dx2{power: 1.0} + -1.0 * v{power: 1.0} * dv/dx1{power: 1.0} + -1.0 * dp/dx1{power: 1.0} + 0.01 * d^2v/dx2^2{power: 1.0} + 0.01 * d^2v/dx1^2{power: 1.0} = dv/dx0{power: 1.0}
          -1.0 * dv/dx1{power: 1.0} = du/dx2{power: 1.0}
      notes:  Axes (t, y, x): dx1 = d/dy, dx2 = d/dx. Two momentum equations (nu = 0.01)
      and continuity. Default subset = gate.py's (36k points); 'full50' is the old ns.py
      window (250k points per variable).
      shape:  (50, 20, 36), axes t, y, x
      data:   projects/pic/data/ns/cylinder_nektar_wake.mat
      config: projects/pic/configs/ns.yaml (on top of configs/_base.yaml)
      suite:  core

Usage -- the same on Windows, Linux and macOS, from any directory:

    python projects/pic/scripts/ns.py                    # one noise-free run, seed 0
    python projects/pic/scripts/ns.py --noise 5 --seed 3
    python projects/pic/scripts/ns.py --variant poly     # method variants: configs/variants.yaml
    python projects/pic/scripts/ns.py --check            # does the known law hold on the data?
    python projects/pic/scripts/ns.py --plot             # look at the data
    python projects/pic/scripts/ns.py --help

The same from Python or a notebook (projects/pic/notebooks/):

    from epde_bench import datasets, load_config, resolve_for_problem, discover
    problem = datasets.load('ns')
    cfg = load_config('ns')
    search, front, objectives, seconds = discover(problem, resolve_for_problem(cfg, problem))

Settings are not in this file: the search is configured by the YAML files
named above, data loading by epde_bench/datasets.py. The original research
script of this folder, projects/pic/data/ns/ns.py, is kept unchanged.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))     # projects/pic

from epde_bench.script import main  # noqa: E402

if __name__ == '__main__':
    sys.exit(main(['ns']))
