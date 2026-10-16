"""Forced Duffing oscillator.

Data set in this folder (all of them: projects/pic/DATASETS.md):

  duffing: Forced Duffing oscillator
      truth:
          -0.20000000298023224 * du/dx0{power: 1.0} + -1.0 * u{power: 1.0} + -1.0 * u{power: 3.0} + 0.30000001192092896 * cos{power: 1.0, freq: 1.0, dim: 0.0} = d^2u/dx0^2{power: 1.0}
      notes:  u'' + delta u' + alpha u + beta u^3 = gamma cos(omega t); parameters
      stored in the file (as in projects/pinn/gate.py).
      shape:  (1001,), axes t
      data:   projects/pic/data/duffing/duffing.npz
      config: projects/pic/configs/duffing.yaml (on top of configs/_base.yaml)
      suite:  core

Usage -- the same on Windows, Linux and macOS, from any directory:

    python projects/pic/scripts/duffing.py                    # one noise-free run, seed 0
    python projects/pic/scripts/duffing.py --noise 5 --seed 3
    python projects/pic/scripts/duffing.py --variant poly     # method variants: configs/variants.yaml
    python projects/pic/scripts/duffing.py --check            # does the known law hold on the data?
    python projects/pic/scripts/duffing.py --plot             # look at the data
    python projects/pic/scripts/duffing.py --help

The same from Python or a notebook (projects/pic/notebooks/):

    from epde_bench import datasets, load_config, resolve_for_problem, discover
    problem = datasets.load('duffing')
    cfg = load_config('duffing')
    search, front, objectives, seconds = discover(problem, resolve_for_problem(cfg, problem))

Settings are not in this file: the search is configured by the YAML files
named above, data loading by epde_bench/datasets.py. The original research
script of this folder is kept unchanged.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))     # projects/pic

from epde_bench.script import main  # noqa: E402

if __name__ == '__main__':
    sys.exit(main(['duffing']))
