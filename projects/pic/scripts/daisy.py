"""DaISy system-identification records.

Data sets in this folder (all of them: projects/pic/DATASETS.md):

  robot_arm: DaISy flexible robot arm (input torque -> acceleration)
      truth:  unknown (not scored)
      notes:  Unknown truth (a ~5th-order flexible structure). dt is not distributed
      with the file; 0.01 s is assumed (coefficients scale with it, structure does not).
      shape:  (1024,), axes t
      data:   projects/pic/data/daisy/robot_arm.dat.gz
      config: projects/pic/configs/robot_arm.yaml (on top of configs/_base.yaml)
      suite:  real

  ballbeam: DaISy ball and beam (beam angle -> ball position)
      truth:  unknown (not scored)
      notes:  Unknown truth; idealised physics is y'' proportional to the beam angle.
      Sampling period 0.1 s per the DaISy description.
      shape:  (1000,), axes t
      data:   projects/pic/data/daisy/ballbeam.dat.gz
      config: projects/pic/configs/ballbeam.yaml (on top of configs/_base.yaml)
      suite:  real

Usage -- the same on Windows, Linux and macOS, from any directory:

    python projects/pic/scripts/daisy.py --dataset ballbeam                    # one noise-free run, seed 0
    python projects/pic/scripts/daisy.py --dataset ballbeam --noise 5 --seed 3
    python projects/pic/scripts/daisy.py --dataset ballbeam --variant poly     # method variants: configs/variants.yaml
    python projects/pic/scripts/daisy.py --dataset ballbeam --check            # does the known law hold on the data?
    python projects/pic/scripts/daisy.py --dataset ballbeam --plot             # look at the data
    python projects/pic/scripts/daisy.py --help

The same from Python or a notebook (projects/pic/notebooks/):

    from epde_bench import datasets, load_config, resolve_for_problem, discover
    problem = datasets.load('robot_arm')
    cfg = load_config('robot_arm')
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
    sys.exit(main(['robot_arm', 'ballbeam']))
