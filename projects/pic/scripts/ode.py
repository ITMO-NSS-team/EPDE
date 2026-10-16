"""Forced oscillator with time-dependent damping.

Data set in this folder (all of them: projects/pic/DATASETS.md):

  ode: Forced oscillator with time-dependent damping
      truth:
          -4.0 * u{power: 1.0} + -1.0 * du/dx0{power: 1.0} * sin{power: 1.0, freq: 2.0, dim: 0.0} + 1.5 * x{power: 1.0, dim: 0.0} = d^2u/dx0^2{power: 1.0}
      notes:  u'' + sin(2t) u' + 4u = 1.5t on t in [0, 16), dt = 0.05.
      shape:  (320,), axes t
      data:   projects/pic/data/ode/ode_data.npy
      config: projects/pic/configs/ode.yaml (on top of configs/_base.yaml)
      suite:  core

Usage -- the same on Windows, Linux and macOS, from any directory:

    python projects/pic/scripts/ode.py                    # one noise-free run, seed 0
    python projects/pic/scripts/ode.py --noise 5 --seed 3
    python projects/pic/scripts/ode.py --variant poly     # method variants: configs/variants.yaml
    python projects/pic/scripts/ode.py --check            # does the known law hold on the data?
    python projects/pic/scripts/ode.py --plot             # look at the data
    python projects/pic/scripts/ode.py --help

The same from Python or a notebook (projects/pic/notebooks/):

    from epde_bench import datasets, load_config, resolve_for_problem, discover
    problem = datasets.load('ode')
    cfg = load_config('ode')
    search, front, objectives, seconds = discover(problem, resolve_for_problem(cfg, problem))

Settings are not in this file: the search is configured by the YAML files
named above, data loading by epde_bench/datasets.py. The original research
script of this folder, projects/pic/data/ode/ode.py, is kept unchanged.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))     # projects/pic

from epde_bench.script import main  # noqa: E402

if __name__ == '__main__':
    sys.exit(main(['ode']))
