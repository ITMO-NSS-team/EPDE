"""Darcy flow.

Data set in this folder (all of them: projects/pic/DATASETS.md):

  darcy: Darcy flow -div(nu grad u) = 1 (data files missing)
      DATA FILES MISSING -- the loader is ready for when they are added.
      data:   projects/pic/data/darcy/darcy_1.0.npy, darcy/darcy_nu_1.0.npy (not in the repository)
      config: projects/pic/configs/darcy.yaml (on top of configs/_base.yaml)
      suite:  other

Usage -- the same on Windows, Linux and macOS, from any directory:

    python projects/pic/scripts/darcy.py                    # one noise-free run, seed 0
    python projects/pic/scripts/darcy.py --noise 5 --seed 3
    python projects/pic/scripts/darcy.py --variant poly     # method variants: configs/variants.yaml
    python projects/pic/scripts/darcy.py --check            # does the known law hold on the data?
    python projects/pic/scripts/darcy.py --plot             # look at the data
    python projects/pic/scripts/darcy.py --help

The same from Python or a notebook (projects/pic/notebooks/):

    from epde_bench import datasets, load_config, resolve_for_problem, discover
    problem = datasets.load('darcy')
    cfg = load_config('darcy')
    search, front, objectives, seconds = discover(problem, resolve_for_problem(cfg, problem))

Settings are not in this file: the search is configured by the YAML files
named above, data loading by epde_bench/datasets.py. The original research
script of this folder, projects/pic/data/darcy/darcy.py, is kept unchanged.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))     # projects/pic

from epde_bench.script import main  # noqa: E402

if __name__ == '__main__':
    sys.exit(main(['darcy']))
