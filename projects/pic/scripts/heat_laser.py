"""Heat equation with a moving laser source.

Data set in this folder (all of them: projects/pic/DATASETS.md):

  heat_laser: Heat equation with a moving laser source, 3-D
      truth:  unknown (not scored)
      notes:  Only 3 points along z and 20 in t, so z- and t-derivatives are crude. The
      source L is laser.npy (t, x, y), assumed uniform in z -- the old script rebuilt it
      from a formula with mismatched axes. No truth is scored.
      shape:  (20, 51, 51, 3), axes t, x, y, z
      data:   projects/pic/data/heat_laser/heat_laser.npz
      config: projects/pic/configs/heat_laser.yaml (on top of configs/_base.yaml)
      suite:  other

Usage -- the same on Windows, Linux and macOS, from any directory:

    python projects/pic/scripts/heat_laser.py                    # one noise-free run, seed 0
    python projects/pic/scripts/heat_laser.py --noise 5 --seed 3
    python projects/pic/scripts/heat_laser.py --variant poly     # method variants: configs/variants.yaml
    python projects/pic/scripts/heat_laser.py --check            # does the known law hold on the data?
    python projects/pic/scripts/heat_laser.py --plot             # look at the data
    python projects/pic/scripts/heat_laser.py --help

The same from Python or a notebook (projects/pic/notebooks/):

    from epde_bench import datasets, load_config, resolve_for_problem, discover
    problem = datasets.load('heat_laser')
    cfg = load_config('heat_laser')
    search, front, objectives, seconds = discover(problem, resolve_for_problem(cfg, problem))

Settings are not in this file: the search is configured by the YAML files
named above, data loading by epde_bench/datasets.py. The original research
script of this folder, projects/pic/data/heat_laser/heat_laser.py, is kept unchanged.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))     # projects/pic

from epde_bench.script import main  # noqa: E402

if __name__ == '__main__':
    sys.exit(main(['heat_laser']))
