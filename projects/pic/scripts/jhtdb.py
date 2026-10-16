"""Isotropic turbulence slice (Johns Hopkins Turbulence Database).

Data set in this folder (all of them: projects/pic/DATASETS.md):

  jhtdb_plane: Isotropic turbulence, 2-D slice of JHTDB
      truth:
          -1.0 * u{power: 1.0} * du/dx2{power: 1.0} + -1.0 * v{power: 1.0} * du/dx1{power: 1.0} + -1.0 * w{power: 1.0} * u_z{power: 1.0} + -1.0 * p_x{power: 1.0} + 1.0 * nu_lap_u{power: 1.0} = du/dx0{power: 1.0}
          -1.0 * u{power: 1.0} * dv/dx2{power: 1.0} + -1.0 * v{power: 1.0} * dv/dx1{power: 1.0} + -1.0 * w{power: 1.0} * v_z{power: 1.0} + -1.0 * p_y{power: 1.0} + 1.0 * nu_lap_v{power: 1.0} = dv/dx0{power: 1.0}
      notes:  48 x 48 slice (stride 8 of the DNS grid), 40 frames. Too coarse for
      numerical derivatives (aliased), so the exact server-side gradients are passed
      instead; out-of-plane and pressure terms enter as exact tokens. Artificial noise
      is unsupported with these supplied derivatives; use noise=0.
      shape:  (40, 48, 48), axes t, y, x
      data:   projects/pic/data/jhtdb/jhtdb_pilot_plane.npz
      config: projects/pic/configs/jhtdb_plane.yaml (on top of configs/_base.yaml)
      suite:  extended

Usage -- the same on Windows, Linux and macOS, from any directory:

    python projects/pic/scripts/jhtdb.py                    # one noise-free run, seed 0
    python projects/pic/scripts/jhtdb.py --noise 5 --seed 3
    python projects/pic/scripts/jhtdb.py --variant poly     # method variants: configs/variants.yaml
    python projects/pic/scripts/jhtdb.py --check            # does the known law hold on the data?
    python projects/pic/scripts/jhtdb.py --plot             # look at the data
    python projects/pic/scripts/jhtdb.py --help

The same from Python or a notebook (projects/pic/notebooks/):

    from epde_bench import datasets, load_config, resolve_for_problem, discover
    problem = datasets.load('jhtdb_plane')
    cfg = load_config('jhtdb_plane')
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
    sys.exit(main(['jhtdb_plane']))
