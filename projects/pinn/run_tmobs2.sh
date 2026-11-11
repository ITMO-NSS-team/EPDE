#!/bin/bash
# PI, Oct 7 ("go ahead with 1 and 2"):
#  1. TM-obs-3k (lbfgs_maxiter 3000): Burgers first (its TM-obs solves ran over
#     T_B), then every other system R-route sends to TM-obs, one budget;
#  2. Lorenz seeds 3-5: TM-obs and B (B for the paired contrast and T_B).
# New tag (gate.py gained an arm); the start control B-ctl seed 0 on burgers
# must equal the stage2 B rows bitwise.
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export GATE_DUMP_DIR=results/fields
set -x
python gate.py drive --system burgers --arms B-ctl --seeds 0 --tag tmobs2_start || exit 1
python gate.py drive --system burgers --arms TM-obs-3k --seeds 0-2 --tag tmobs2 || exit 1
echo BURGERS_3K_DONE
for s in duffing lv ac; do
  python gate.py drive --system $s --arms TM-obs-3k --seeds 0-2 --tag tmobs2 || exit 1
done
python gate.py drive --system lorenz --arms TM-obs-3k --seeds 0-5 --tag tmobs2 || exit 1
python gate.py drive --system lorenz --arms TM-obs --seeds 3-5 --tag tmobs2 || exit 1
python gate.py drive --system lorenz --arms B --seeds 3-5 --tag tmobs2 || exit 1
echo TMOBS2_DONE
