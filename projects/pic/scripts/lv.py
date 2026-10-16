"""Lotka-Volterra predator-prey system.

Data set in this folder (all of them: projects/pic/DATASETS.md):

  lv: Lotka-Volterra predator-prey
      truth:
          20.0 * u{power: 1.0} + -20.0 * u{power: 1.0} * v{power: 1.0} = du/dx0{power: 1.0}
          20.0 * u{power: 1.0} * v{power: 1.0} + -20.0 * v{power: 1.0} = dv/dx0{power: 1.0}
          (+ 1 equivalent form(s) accepted)
      notes:  alpha = beta = gamma = delta = 20, all 301 samples (lv.py used the first
      150; gate.py explains why the full record is preferable).
      shape:  (301,), axes t
      data:   projects/pic/data/lv/t_20.npy, lv/data_20.npy
      config: projects/pic/configs/lv.yaml (on top of configs/_base.yaml)
      suite:  core

Usage -- the same on Windows, Linux and macOS, from any directory:

    python projects/pic/scripts/lv.py                    # one noise-free run, seed 0
    python projects/pic/scripts/lv.py --noise 5 --seed 3
    python projects/pic/scripts/lv.py --variant poly     # method variants: configs/variants.yaml
    python projects/pic/scripts/lv.py --check            # does the known law hold on the data?
    python projects/pic/scripts/lv.py --plot             # look at the data
    python projects/pic/scripts/lv.py --help

The same from Python or a notebook (projects/pic/notebooks/):

    from epde_bench import datasets, load_config, resolve_for_problem, discover
    problem = datasets.load('lv')
    cfg = load_config('lv')
    search, front, objectives, seconds = discover(problem, resolve_for_problem(cfg, problem))

Settings are not in this file: the search is configured by the YAML files
named above, data loading by epde_bench/datasets.py. The original research
script of this folder, projects/pic/data/lv/lv.py, is kept unchanged.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))     # projects/pic

from epde_bench.script import main  # noqa: E402

if __name__ == '__main__':
    sys.exit(main(['lv']))
