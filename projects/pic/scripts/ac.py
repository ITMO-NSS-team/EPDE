"""Allen-Cahn equation.

Data set in this folder (all of them: projects/pic/DATASETS.md):

  ac: Allen-Cahn equation
      truth:
          0.0001 * d^2u/dx1^2{power: 1.0} + -5.0 * u{power: 3.0} + 5.0 * u{power: 1.0} = du/dx0{power: 1.0}
      notes:  u_t = 1e-4 u_xx + 5u - 5u^3. The diffusion term is tiny, which makes it
      hard to separate from noise.
      shape:  (51, 128), axes t, x
      data:   projects/pic/data/ac/ac_data.npy
      config: projects/pic/configs/ac.yaml (on top of configs/_base.yaml)
      suite:  core

Usage -- the same on Windows, Linux and macOS, from any directory:

    python projects/pic/scripts/ac.py                    # one noise-free run, seed 0
    python projects/pic/scripts/ac.py --noise 5 --seed 3
    python projects/pic/scripts/ac.py --variant poly     # method variants: configs/variants.yaml
    python projects/pic/scripts/ac.py --check            # does the known law hold on the data?
    python projects/pic/scripts/ac.py --plot             # look at the data
    python projects/pic/scripts/ac.py --help

The same from Python or a notebook (projects/pic/notebooks/):

    from epde_bench import datasets, load_config, resolve_for_problem, discover
    problem = datasets.load('ac')
    cfg = load_config('ac')
    search, front, objectives, seconds = discover(problem, resolve_for_problem(cfg, problem))

Settings are not in this file: the search is configured by the YAML files
named above, data loading by epde_bench/datasets.py. The original research
script of this folder, projects/pic/data/ac/ac.py, is kept unchanged.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))     # projects/pic

from epde_bench.script import main  # noqa: E402

if __name__ == '__main__':
    sys.exit(main(['ac']))
