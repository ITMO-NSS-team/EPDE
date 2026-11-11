#!/bin/bash
# TM-obs promoted to a candidate (PI, Oct 6). Own budget, solve mode, seeds
# 0-2 on every system. T_B comes from the stage2 B rows (copied here); the
# start controls B-ctl (seed 0) must equal those B rows bitwise on each
# system, which checks this code state reproduces the shipped recipe.
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export GATE_DUMP_DIR=results/fields
set -x
for s in lv duffing ac burgers lorenz; do
  python gate.py drive --system $s --arms B-ctl --seeds 0 --tag tmobs_start || exit 1
done
for s in lorenz lv duffing ac burgers; do
  python gate.py drive --system $s --arms TM-obs --seeds 0-2 --tag tmobs || exit 1
done
echo TMOBS_GRID_DONE
python gate.py drive --system ns --arms TM-obs --seeds 0-2 --tag tmobs
echo TMOBS_ALL_DONE
