"""Pendulums: real rig and simulation.

Data sets in this folder (all of them: projects/pic/DATASETS.md):

  dp_encoder: Real double pendulum (encoder, HardwareX rig)
      truth:  unknown (not scored)
      notes:  Coupled Lagrangian equations with cos/sin of the angle difference; their
      exact coefficients are not known, so the run is not scored. sin(theta) and the
      coupling factors enter as tokens.
      shape:  (6000,), axes t
      data:   projects/pic/data/dp/MultiArm_Pendulum/Double_FreeSwing_1.mat
      config: projects/pic/configs/dp_encoder.yaml (on top of configs/_base.yaml)
      suite:  real

  pend_single: Real single pendulum (encoder, HardwareX rig)
      truth:
          -64.8 * theta{power: 1.0} + -0.65 * dtheta/dx0{power: 1.0} = d^2theta/dx0^2{power: 1.0}
          (+ 1 equivalent form(s) accepted)
      notes:  Small swing about the hanging position, angle centred (theta - pi): a
      damped linear oscillator, -64.8 = -g/l. The undamped form also counts (damping is
      weak).
      shape:  (5000,), axes t
      data:   projects/pic/data/dp/MultiArm_Pendulum/Single_FreeSwing_1.mat
      config: projects/pic/configs/pend_single.yaml (on top of configs/_base.yaml)
      suite:  real

  dp_sim: Simulated double pendulum
      truth:  unknown (not scored)
      notes:  Used by the PINN studies in dp/. The equations of motion need coupling
      tokens and are not written in token form here.
      shape:  (1001,), axes t
      data:   projects/pic/data/dp/dp.npz
      config: projects/pic/configs/dp_sim.yaml (on top of configs/_base.yaml)
      suite:  other

  dp_video: Real double pendulum (video tracking)
      truth:  unknown (not scored)
      notes:  Mean link angles from video markers (row 0 = time, row 1 = angle). Noisier
      than the encoder record; dp_encoder is the preferred source.
      shape:  (38827,), axes t
      data:   projects/pic/data/dp/Video_Tracking_Data/Video_Tracking_Data/Trial*/DPmean_data_RB*.npy
      config: projects/pic/configs/dp_video.yaml (on top of configs/_base.yaml)
      suite:  other

Usage -- the same on Windows, Linux and macOS, from any directory:

    python projects/pic/scripts/dp.py --dataset pend_single                    # one noise-free run, seed 0
    python projects/pic/scripts/dp.py --dataset pend_single --noise 5 --seed 3
    python projects/pic/scripts/dp.py --dataset pend_single --variant poly     # method variants: configs/variants.yaml
    python projects/pic/scripts/dp.py --dataset pend_single --check            # does the known law hold on the data?
    python projects/pic/scripts/dp.py --dataset pend_single --plot             # look at the data
    python projects/pic/scripts/dp.py --help

The same from Python or a notebook (projects/pic/notebooks/):

    from epde_bench import datasets, load_config, resolve_for_problem, discover
    problem = datasets.load('dp_encoder')
    cfg = load_config('dp_encoder')
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
    sys.exit(main(['dp_encoder', 'pend_single', 'dp_sim', 'dp_video']))
