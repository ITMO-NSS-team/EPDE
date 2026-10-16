"""Heat in soil under solar forcing.

Data sets in this folder (all of them: projects/pic/DATASETS.md):

  heat_solar_1d: Heat in soil under solar forcing, 1-D
      truth:  unknown (not scored)
      notes:  Simulated soil temperature with a periodic surface flux. Expected law: the
      heat equation u_t = a u_xx in the interior (coefficient not stored), so no truth
      is scored. The file also stores du.
      shape:  (576, 51), axes t, x
      data:   projects/pic/data/heat_solar/heat_soil_uniform_1d_p1.npz
      config: projects/pic/configs/heat_solar_1d.yaml (on top of configs/_base.yaml)
      suite:  other

  heat_solar_2d: Heat in soil under solar forcing, 2-D
      truth:  unknown (not scored)
      notes:  2-D version of heat_solar_1d (576 x 51 x 51 before striding in t).
      shape:  (144, 51, 51), axes t, x, y
      data:   projects/pic/data/heat_solar/heat_soil_uniform_2d_p1.npz
      config: projects/pic/configs/heat_solar_2d.yaml (on top of configs/_base.yaml)
      suite:  other

Usage -- the same on Windows, Linux and macOS, from any directory:

    python projects/pic/scripts/heat_solar.py --dataset heat_solar_2d                    # one noise-free run, seed 0
    python projects/pic/scripts/heat_solar.py --dataset heat_solar_2d --noise 5 --seed 3
    python projects/pic/scripts/heat_solar.py --dataset heat_solar_2d --variant poly     # method variants: configs/variants.yaml
    python projects/pic/scripts/heat_solar.py --dataset heat_solar_2d --check            # does the known law hold on the data?
    python projects/pic/scripts/heat_solar.py --dataset heat_solar_2d --plot             # look at the data
    python projects/pic/scripts/heat_solar.py --help

The same from Python or a notebook (projects/pic/notebooks/):

    from epde_bench import datasets, load_config, resolve_for_problem, discover
    problem = datasets.load('heat_solar_1d')
    cfg = load_config('heat_solar_1d')
    search, front, objectives, seconds = discover(problem, resolve_for_problem(cfg, problem))

Settings are not in this file: the search is configured by the YAML files
named above, data loading by epde_bench/datasets.py. The original research
script of this folder, projects/pic/data/heat_solar/heat_solar.py, is kept unchanged.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))     # projects/pic

from epde_bench.script import main  # noqa: E402

if __name__ == '__main__':
    sys.exit(main(['heat_solar_1d', 'heat_solar_2d']))
