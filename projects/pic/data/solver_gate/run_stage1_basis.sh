#!/bin/bash
# Stage-1 sweep, fixed-basis cells (CPU, deterministic, one seed). Runs
# alongside the GPU batch, pinned to 4 threads so both have room; the
# concurrency is recorded in every row's env block.
cd "$(dirname "$0")"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export CUDA_VISIBLE_DEVICES=""
set -x
ARMS=BL,BL-n1,BL-n2,BL-n3,BL-n4,BL-r5,BL-data,BL-noprec,BL-t4,BL-c3,BL-sinv-live,BL-sinv-data
for s in duffing burgers ac; do
  python gate.py drive --system $s --arms $ARMS --seeds 0 --tag stage1_basis
done
echo STAGE1_BASIS_DONE
