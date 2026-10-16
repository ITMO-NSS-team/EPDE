"""PDE with a 1/x coefficient.

Data set in this folder (all of them: projects/pic/DATASETS.md):

  pde_divide: PDE with a 1/x coefficient
      truth:
          -2.0 * du/dx1{power: 1.0} + 0.5 * d^2u/dx1^2{power: 1.0} * x{power: 1.0, dim: 1.0} = du/dx0{power: 1.0} * x{power: 1.0, dim: 1.0}
      notes:  x u_t = -2 u_x + 0.5 x u_xx, i.e. u_t = -2 u_x / x + 0.5 u_xx (needs the x
      token).
      shape:  (251, 100), axes t, x
      data:   projects/pic/data/pde_divide/PDE_divide.npy
      config: projects/pic/configs/pde_divide.yaml (on top of configs/_base.yaml)
      suite:  core

Usage -- the same on Windows, Linux and macOS, from any directory:

    python projects/pic/scripts/pde_divide.py                    # one noise-free run, seed 0
    python projects/pic/scripts/pde_divide.py --noise 5 --seed 3
    python projects/pic/scripts/pde_divide.py --variant poly     # method variants: configs/variants.yaml
    python projects/pic/scripts/pde_divide.py --check            # does the known law hold on the data?
    python projects/pic/scripts/pde_divide.py --plot             # look at the data
    python projects/pic/scripts/pde_divide.py --help

The same from Python or a notebook (projects/pic/notebooks/):

    from epde_bench import datasets, load_config, resolve_for_problem, discover
    problem = datasets.load('pde_divide')
    cfg = load_config('pde_divide')
    search, front, objectives, seconds = discover(problem, resolve_for_problem(cfg, problem))

Settings are not in this file: the search is configured by the YAML files
named above, data loading by epde_bench/datasets.py. The original research
script of this folder, projects/pic/data/pde_divide/pde_divide.py, is kept unchanged.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))     # projects/pic

from epde_bench.script import main  # noqa: E402

if __name__ == '__main__':
    sys.exit(main(['pde_divide']))
