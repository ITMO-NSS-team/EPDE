#!/bin/bash
# Lorenz Round 1 (PI, Oct 6: time-marching PINN route, own budget first,
# full-window marching as the base). Seed 0, both forms, solve mode, own
# budget. Controls: B and S1 (G-0000) must reproduce the stage2 rows bitwise;
# B-ctl at the end equals B. Fields are dumped for the kill rules.
cd "$(dirname "$0")"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export GATE_DUMP_DIR=results/fields
set -x
python gate.py drive --system lorenz --arms B --seeds 0 --tag lorenz_m1_start || exit 1
python gate.py drive --system lorenz --arms G-0000 --seeds 0 --mode trace --tag lorenz_m1_start || exit 1
python gate.py drive --system lorenz --arms TM-obs,TM,DE,CW --seeds 0 --tag lorenz_m1
python gate.py drive --system lorenz --arms B-ctl --seeds 0 --tag lorenz_m1_end
echo LORENZ_M1_DONE
