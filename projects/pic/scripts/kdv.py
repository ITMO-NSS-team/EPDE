"""Korteweg-de Vries equation, four records.

Data sets in this folder (all of them: projects/pic/DATASETS.md):

  kdv: Korteweg-de Vries equation (PDE-FIND data)
      truth:
          -6.0 * du/dx1{power: 1.0} * u{power: 1.0} + -1.0 * d^3u/dx1^3{power: 1.0} = du/dx0{power: 1.0}
          (+ 3 equivalent form(s) accepted)
      notes:  u_t = -6 u u_x - u_xxx. The record is a soliton family, so three
      identities of it are also exact and accepted. (kdv_sindy/kdv.mat is a byte-
      identical copy.)
      shape:  (201, 512), axes t, x
      data:   projects/pic/data/kdv/kdv_sindy.mat
      config: projects/pic/configs/kdv.yaml (on top of configs/_base.yaml)
      suite:  core

  kdv_cossin: KdV with a cos(t)sin(x) source
      truth:
          -6.0 * du/dx1{power: 1.0} * u{power: 1.0} + -1.0 * d^3u/dx1^3{power: 1.0} + 1.0 * cos(t)sin(x){power: 1.0} = du/dx0{power: 1.0}
      notes:  The source enters as one product token cos(t)sin(x).
      shape:  (81, 81), axes t, x
      data:   projects/pic/data/kdv/data.csv
      config: projects/pic/configs/kdv_cossin.yaml (on top of configs/_base.yaml)
      suite:  core

  kdv_homogen: KdV, homogeneous, x in [-3, 3]
      truth:
          -6.0 * du/dx1{power: 1.0} * u{power: 1.0} + -1.0 * d^3u/dx1^3{power: 1.0} = du/dx0{power: 1.0}
      notes:  Truth from KdV_h_test in the old kdv.py.
      shape:  (120, 480), axes t, x
      data:   projects/pic/data/kdv/data_kdv_homogen.npy
      config: projects/pic/configs/kdv_homogen.yaml (on top of configs/_base.yaml)
      suite:  extended

  kdv_sga: KdV, SGA-PDE record (u_t = -u u_x - 0.0025 u_xxx)
      truth:
          -1.0 * du/dx1{power: 1.0} * u{power: 1.0} + -0.0025 * d^3u/dx1^3{power: 1.0} = du/dx0{power: 1.0}
      notes:  Truth from KdV_sga_test in the old kdv.py.
      shape:  (201, 512), axes t, x
      data:   projects/pic/data/kdv/Kdv.mat
      config: projects/pic/configs/kdv_sga.yaml (on top of configs/_base.yaml)
      suite:  extended

Usage -- the same on Windows, Linux and macOS, from any directory:

    python projects/pic/scripts/kdv.py --dataset kdv_cossin                    # one noise-free run, seed 0
    python projects/pic/scripts/kdv.py --dataset kdv_cossin --noise 5 --seed 3
    python projects/pic/scripts/kdv.py --dataset kdv_cossin --variant poly     # method variants: configs/variants.yaml
    python projects/pic/scripts/kdv.py --dataset kdv_cossin --check            # does the known law hold on the data?
    python projects/pic/scripts/kdv.py --dataset kdv_cossin --plot             # look at the data
    python projects/pic/scripts/kdv.py --help

The same from Python or a notebook (projects/pic/notebooks/):

    from epde_bench import datasets, load_config, resolve_for_problem, discover
    problem = datasets.load('kdv')
    cfg = load_config('kdv')
    search, front, objectives, seconds = discover(problem, resolve_for_problem(cfg, problem))

Settings are not in this file: the search is configured by the YAML files
named above, data loading by epde_bench/datasets.py. The original research
script of this folder, projects/pic/data/kdv/kdv.py, is kept unchanged.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))     # projects/pic

from epde_bench.script import main  # noqa: E402

if __name__ == '__main__':
    sys.exit(main(['kdv', 'kdv_cossin', 'kdv_homogen', 'kdv_sga']))
