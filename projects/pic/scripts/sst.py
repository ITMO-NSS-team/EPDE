"""Sea surface temperature (ESA CCI L4).

Data set in this folder (all of them: projects/pic/DATASETS.md):

  sst: Sea surface temperature, ESA CCI L4 (Jan-Mar 2025)
      truth:  unknown (not scored)
      notes:  Daily analysed SST (K), 90 days. By default the largest rectangular ocean
      region finite at every frame is used; crop_ocean=False loads the original NaN-
      masked box for plotting only. sst/sst_l4.nc is a ZIP archive of the same daily
      files despite its extension.
      shape:  (90, 268, 384), axes t, lat, lon
      data:   projects/pic/data/sst/sst_l4_files/*.nc (90 daily files)
      config: projects/pic/configs/sst.yaml (on top of configs/_base.yaml)
      suite:  other

Usage -- the same on Windows, Linux and macOS, from any directory:

    python projects/pic/scripts/sst.py                    # one noise-free run, seed 0
    python projects/pic/scripts/sst.py --noise 5 --seed 3
    python projects/pic/scripts/sst.py --variant poly     # method variants: configs/variants.yaml
    python projects/pic/scripts/sst.py --check            # does the known law hold on the data?
    python projects/pic/scripts/sst.py --plot             # look at the data
    python projects/pic/scripts/sst.py --help

The same from Python or a notebook (projects/pic/notebooks/):

    from epde_bench import datasets, load_config, resolve_for_problem, discover
    problem = datasets.load('sst')
    cfg = load_config('sst')
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
    sys.exit(main(['sst']))
