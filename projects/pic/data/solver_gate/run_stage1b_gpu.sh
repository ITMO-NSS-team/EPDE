#!/bin/bash
# Stage 1b/1c, network arms (GPU), one code state, then the basis cells (CPU)
# only once the GPU is done (no timed work shares the CPU).
cd "$(dirname "$0")"
set -x
for s in ac duffing burgers; do
  python gate.py prefit --system $s --arm S10 --seeds 0-2
done
for s in ac duffing burgers; do
  python gate.py drive --system $s --arms B-ctl --seeds 0-2 --tag stage1b_start
  # a REAL capped solve: its wall-clock is the point, so no trace overhead
  python gate.py drive --system $s --arms S1-cap --seeds 0-2 --tag stage1b
  python gate.py drive --system $s --arms S10,S7f,S8f,P1,P2,D1,D2,S0-32 --seeds 0-2 --mode trace --tag stage1b
done
for s in ac duffing burgers; do
  # Levenberg-Marquardt has no L-BFGS trace: solve mode
  python gate.py drive --system $s --arms S5-32 --seeds 0-2 --tag stage1b
done
for s in ac duffing burgers; do
  python gate.py drive --system $s --arms B-ctl --seeds 0-2 --tag stage1b_end
done
echo STAGE1B_GPU_DONE
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=""
for s in duffing burgers ac; do
  python gate.py drive --system $s --arms BL-pred,BL-sinv-live-f,BL-sinv-data-f --seeds 0 --tag stage1b_basis
done
echo STAGE1B_ALL_DONE
