"""Lorenz-63 system.

Data set in this folder (all of them: projects/pic/DATASETS.md):

  lorenz: Lorenz-63 system
      truth:
          10.0 * v{power: 1.0} + -10.0 * u{power: 1.0} = du/dx0{power: 1.0}
          28.0 * u{power: 1.0} + -1.0 * u{power: 1.0} * w{power: 1.0} + -1.0 * v{power: 1.0} = dv/dx0{power: 1.0}
          1.0 * u{power: 1.0} * v{power: 1.0} + -2.6666666666666665 * w{power: 1.0} = dw/dx0{power: 1.0}
      notes:  sigma = 10, rho = 28, beta = 8/3. Window t in [20.0, 25.2] of the stored
      run, every 5th sample (gate.py): on the attractor. lorenz.py and the old benchmark
      used t[:1000], an off-attractor transient.
      shape:  (1041,), axes t
      data:   projects/pic/data/lorenz/t.npy, lorenz/lorenz.npy
      config: projects/pic/configs/lorenz.yaml (on top of configs/_base.yaml)
      suite:  core

Usage -- the same on Windows, Linux and macOS, from any directory:

    python projects/pic/scripts/lorenz.py                    # one noise-free run, seed 0
    python projects/pic/scripts/lorenz.py --noise 5 --seed 3
    python projects/pic/scripts/lorenz.py --variant poly     # method variants: configs/variants.yaml
    python projects/pic/scripts/lorenz.py --check            # does the known law hold on the data?
    python projects/pic/scripts/lorenz.py --plot             # look at the data
    python projects/pic/scripts/lorenz.py --help

The same from Python or a notebook (projects/pic/notebooks/):

    from epde_bench import datasets, load_config, resolve_for_problem, discover
    problem = datasets.load('lorenz')
    cfg = load_config('lorenz')
    search, front, objectives, seconds = discover(problem, resolve_for_problem(cfg, problem))

Settings are not in this file: the search is configured by the YAML files
named above, data loading by epde_bench/datasets.py. The original research
script of this folder, projects/pic/data/lorenz/lorenz.py, is kept unchanged.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))     # projects/pic

from epde_bench.script import main  # noqa: E402

if __name__ == '__main__':
    sys.exit(main(['lorenz']))
