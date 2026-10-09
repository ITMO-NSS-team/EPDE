#!/bin/bash
# Stage-1 sweep, network arms (GPU), one code state. Order per the plan:
# shared fits -> B-ctl (start) -> B-4k, S0 -> null checks -> S arms -> B-ctl (end).
cd "$(dirname "$0")"
set -x
for s in ac duffing burgers; do
  python gate.py prefit --system $s --arm S9 --seeds 0-2
done
for s in ac duffing burgers; do
  python gate.py drive --system $s --arms B-ctl --seeds 0-2 --tag stage1_start
  python gate.py drive --system $s --arms B-4k,S0 --seeds 0-2 --mode trace --tag stage1
  python gate.py drive --system $s --arms S0-nolr,S3-null --seeds 0 --mode trace --tag stage1
  python gate.py drive --system $s --arms S1,S2,S3,S4,S6,S7,S8,S9,B9 --seeds 0-2 --mode trace --tag stage1
done
for s in ac duffing burgers; do
  python gate.py drive --system $s --arms B-ctl --seeds 0-2 --tag stage1_end
done
echo STAGE1_GPU_DONE
