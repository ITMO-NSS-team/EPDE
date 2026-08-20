#!/bin/sh
# Second round of the GOLEM-configuration study.
#
# Round one (10% of the native budget, 30 runs per configuration) separated
# nothing: every configuration landed in 67-77% recovery, inside the +-8pp
# standard error, and every one of them reached the same median discrepancy.
# Two follow-ups:
#
#   * a harsher budget (3%), in case 10% was already past the point where the
#     search algorithm matters;
#   * a budget ladder on lotka_volterra -- the one scenario where budget, not
#     search quality, is the binding constraint (0% at 10%, 100% at 100%) --
#     to see which configuration reaches recovery soonest.

set -e
cd "$(dirname "$0")"
PY="${PYTHON:-python}"
export PYTHONHASHSEED=0

BUDGET_SRC=results/benchmark_complexity.jsonl
COMMON="--seeds 0 1 2 3 4 --second-objective complexity --budget-from $BUDGET_SRC"

$PY improve.py --scenarios wave burgers kdv allen_cahn van_der_pol lotka_volterra \
    --configs base nsga2 cache restart3 $COMMON --budget-scale 0.03 --tag b03

for scale in 0.2 0.4 0.6; do
    $PY improve.py --scenarios lotka_volterra \
        --configs base nsga2 restart3 $COMMON \
        --budget-scale $scale --tag "lv_$scale"
done

echo "IMPROVE DONE"
