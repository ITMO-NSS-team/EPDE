"""Nonlinear diffusion u_t = (u u_x)_x.

Data set in this folder (all of them: projects/pic/DATASETS.md):

  pde_compound: Nonlinear diffusion u_t = (u u_x)_x
      truth:
          1.0 * du/dx1{power: 2.0} + 1.0 * d^2u/dx1^2{power: 1.0} * u{power: 1.0} = du/dx0{power: 1.0}
      notes:  u_t = u_x^2 + u u_xx on t in [0, 0.5], x in [1, 2].
      shape:  (251, 100), axes t, x
      data:   projects/pic/data/pde_compound/PDE_compound.npy
      config: projects/pic/configs/pde_compound.yaml (on top of configs/_base.yaml)
      suite:  core

Usage -- the same on Windows, Linux and macOS, from any directory:

    python projects/pic/scripts/pde_compound.py                    # one noise-free run, seed 0
    python projects/pic/scripts/pde_compound.py --noise 5 --seed 3
    python projects/pic/scripts/pde_compound.py --variant poly     # method variants: configs/variants.yaml
    python projects/pic/scripts/pde_compound.py --check            # does the known law hold on the data?
    python projects/pic/scripts/pde_compound.py --plot             # look at the data
    python projects/pic/scripts/pde_compound.py --help

The same from Python or a notebook (projects/pic/notebooks/):

    from epde_bench import datasets, load_config, resolve_for_problem, discover
    problem = datasets.load('pde_compound')
    cfg = load_config('pde_compound')
    search, front, objectives, seconds = discover(problem, resolve_for_problem(cfg, problem))

Settings are not in this file: the search is configured by the YAML files
named above, data loading by epde_bench/datasets.py. The original research
script of this folder, projects/pic/data/pde_compound/pde_compound.py, is kept unchanged.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))     # projects/pic

from epde_bench.script import main  # noqa: E402

if __name__ == '__main__':
    sys.exit(main(['pde_compound']))
