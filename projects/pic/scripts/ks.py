"""Kuramoto-Sivashinsky equation.

Data set in this folder (all of them: projects/pic/DATASETS.md):

  ks: Kuramoto-Sivashinsky equation
      truth:
          -1.0 * u{power: 1.0} * du/dx1{power: 1.0} + -1.0 * d^2u/dx1^2{power: 1.0} + -1.0 * d^4u/dx1^4{power: 1.0} = du/dx0{power: 1.0}
      notes:  u_t = -u u_x - u_xx - u_xxxx; chaotic, needs a 4th derivative. The old
      ks.py opened the file relative to the working directory.
      shape:  (251, 1024), axes t, x
      data:   projects/pic/data/ks/kuramoto_sivishinky.mat
      config: projects/pic/configs/ks.yaml (on top of configs/_base.yaml)
      suite:  core

Usage -- the same on Windows, Linux and macOS, from any directory:

    python projects/pic/scripts/ks.py                    # one noise-free run, seed 0
    python projects/pic/scripts/ks.py --noise 5 --seed 3
    python projects/pic/scripts/ks.py --variant poly     # method variants: configs/variants.yaml
    python projects/pic/scripts/ks.py --check            # does the known law hold on the data?
    python projects/pic/scripts/ks.py --plot             # look at the data
    python projects/pic/scripts/ks.py --help

The same from Python or a notebook (projects/pic/notebooks/):

    from epde_bench import datasets, load_config, resolve_for_problem, discover
    problem = datasets.load('ks')
    cfg = load_config('ks')
    search, front, objectives, seconds = discover(problem, resolve_for_problem(cfg, problem))

Settings are not in this file: the search is configured by the YAML files
named above, data loading by epde_bench/datasets.py. The original research
script of this folder, projects/pic/data/ks/ks.py, is kept unchanged.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))     # projects/pic

from epde_bench.script import main  # noqa: E402

if __name__ == '__main__':
    sys.exit(main(['ks']))
