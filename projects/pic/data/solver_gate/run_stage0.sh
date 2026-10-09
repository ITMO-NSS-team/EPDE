#!/bin/bash
# Phase 0 baseline + Stage-1 controls, GPU, sequential (one solve process at a time).
cd "$(dirname "$0")"
set -x
for s in ac duffing burgers; do
  python gate.py drive --system $s --arms B,B-ctl --seeds 0-2 --tag baseline
done
for s in ac duffing burgers; do
  python gate.py drive --system $s --arms B-4k,S0 --seeds 0-2 --mode trace --tag stage1
done
echo STAGE0_DONE
