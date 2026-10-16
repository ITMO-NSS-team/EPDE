"""Van der Pol oscillator.

Data set in this folder (all of them: projects/pic/DATASETS.md):

  vdp: Van der Pol oscillator (mu = 0.2)
      truth:
          -0.2 * u{power: 2.0} * du/dx0{power: 1.0} + 0.2 * du/dx0{power: 1.0} + -1.0 * u{power: 1.0} = d^2u/dx0^2{power: 1.0}
      notes:  u'' = 0.2 (1 - u^2) u' - u on t in [0, 16), dt = 0.05.
      shape:  (320,), axes t
      data:   projects/pic/data/vdp/vdp_data.npy
      config: projects/pic/configs/vdp.yaml (on top of configs/_base.yaml)
      suite:  core

Usage -- the same on Windows, Linux and macOS, from any directory:

    python projects/pic/scripts/vdp.py                    # one noise-free run, seed 0
    python projects/pic/scripts/vdp.py --noise 5 --seed 3
    python projects/pic/scripts/vdp.py --variant poly     # method variants: configs/variants.yaml
    python projects/pic/scripts/vdp.py --check            # does the known law hold on the data?
    python projects/pic/scripts/vdp.py --plot             # look at the data
    python projects/pic/scripts/vdp.py --help

The same from Python or a notebook (projects/pic/notebooks/):

    from epde_bench import datasets, load_config, resolve_for_problem, discover
    problem = datasets.load('vdp')
    cfg = load_config('vdp')
    search, front, objectives, seconds = discover(problem, resolve_for_problem(cfg, problem))

Settings are not in this file: the search is configured by the YAML files
named above, data loading by epde_bench/datasets.py. The original research
script of this folder, projects/pic/data/vdp/vdp.py, is kept unchanged.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))     # projects/pic

from epde_bench.script import main  # noqa: E402

if __name__ == '__main__':
    sys.exit(main(['vdp']))
