"""Burgers equation (viscous and inviscid).

Data sets in this folder (all of them: projects/pic/DATASETS.md):

  burgers: Viscous Burgers equation (PDE-FIND data)
      truth:
          -1.0 * u{power: 1.0} * du/dx1{power: 1.0} + 0.1 * d^2u/dx1^2{power: 1.0} = du/dx0{power: 1.0}
      notes:  u_t = -u u_x + 0.1 u_xx, periodic in x. (burgers_test in the old
      burgers.py held the Allen-Cahn equation by mistake.)
      shape:  (101, 256), axes t, x
      data:   projects/pic/data/burgers/burgers.mat
      config: projects/pic/configs/burgers.yaml (on top of configs/_base.yaml)
      suite:  core

  burgers_inviscid: Inviscid Burgers equation
      truth:
          -1.0 * u{power: 1.0} * du/dx1{power: 1.0} = du/dx0{power: 1.0}
          (+ 2 equivalent form(s) accepted)
      notes:  The record is the similarity solution u = x / (t + c), so two identities
      hold as well as the PDE and count as correct.
      shape:  (101, 101), axes t, x
      data:   projects/pic/data/burgers/burgers_sln_100.csv
      config: projects/pic/configs/burgers_inviscid.yaml (on top of configs/_base.yaml)
      suite:  core

Usage -- the same on Windows, Linux and macOS, from any directory:

    python projects/pic/scripts/burgers.py --dataset burgers_inviscid                    # one noise-free run, seed 0
    python projects/pic/scripts/burgers.py --dataset burgers_inviscid --noise 5 --seed 3
    python projects/pic/scripts/burgers.py --dataset burgers_inviscid --variant poly     # method variants: configs/variants.yaml
    python projects/pic/scripts/burgers.py --dataset burgers_inviscid --check            # does the known law hold on the data?
    python projects/pic/scripts/burgers.py --dataset burgers_inviscid --plot             # look at the data
    python projects/pic/scripts/burgers.py --help

The same from Python or a notebook (projects/pic/notebooks/):

    from epde_bench import datasets, load_config, resolve_for_problem, discover
    problem = datasets.load('burgers')
    cfg = load_config('burgers')
    search, front, objectives, seconds = discover(problem, resolve_for_problem(cfg, problem))

Settings are not in this file: the search is configured by the YAML files
named above, data loading by epde_bench/datasets.py. The original research
script of this folder, projects/pic/data/burgers/burgers.py, is kept unchanged.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))     # projects/pic

from epde_bench.script import main  # noqa: E402

if __name__ == '__main__':
    sys.exit(main(['burgers', 'burgers_inviscid']))
