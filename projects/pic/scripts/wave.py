"""Wave equation.

Data set in this folder (all of them: projects/pic/DATASETS.md):

  wave: Wave equation
      truth:
          0.04 * d^2u/dx1^2{power: 1.0} = d^2u/dx0^2{power: 1.0}
          (+ 4 equivalent form(s) accepted)
      notes:  u_tt = 0.04 u_xx. The alternatives are the wave equation multiplied by
      another factor; they are accepted as in the group's former benchmark.
      shape:  (81, 81), axes t, x
      data:   projects/pic/data/wave/wave_sln_80.csv
      config: projects/pic/configs/wave.yaml (on top of configs/_base.yaml)
      suite:  core

Usage -- the same on Windows, Linux and macOS, from any directory:

    python projects/pic/scripts/wave.py                    # one noise-free run, seed 0
    python projects/pic/scripts/wave.py --noise 5 --seed 3
    python projects/pic/scripts/wave.py --variant poly     # method variants: configs/variants.yaml
    python projects/pic/scripts/wave.py --check            # does the known law hold on the data?
    python projects/pic/scripts/wave.py --plot             # look at the data
    python projects/pic/scripts/wave.py --help

The same from Python or a notebook (projects/pic/notebooks/):

    from epde_bench import datasets, load_config, resolve_for_problem, discover
    problem = datasets.load('wave')
    cfg = load_config('wave')
    search, front, objectives, seconds = discover(problem, resolve_for_problem(cfg, problem))

Settings are not in this file: the search is configured by the YAML files
named above, data loading by epde_bench/datasets.py. The original research
script of this folder, projects/pic/data/wave/wave.py, is kept unchanged.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))     # projects/pic

from epde_bench.script import main  # noqa: E402

if __name__ == '__main__':
    sys.exit(main(['wave']))
