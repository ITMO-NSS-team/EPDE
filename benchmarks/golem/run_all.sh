#!/bin/sh
# Full experiment matrix. Runs strictly sequentially: wall-clock time is one of
# the reported metrics, so nothing else heavy may share the CPU.
#
#   sh experiments/run_all.sh
#
# PYTHONHASHSEED is mandatory -- EPDE's search path depends on string-hash
# order, so without it repeated runs of the same seed diverge.

set -e
cd "$(dirname "$0")"
PY="${PYTHON:-python}"
export PYTHONHASHSEED=0

SCENARIOS="wave burgers kdv allen_cahn van_der_pol lotka_volterra"
SEEDS="0 1 2 3 4"
COMMON="--scenarios $SCENARIOS --seeds $SEEDS --pop-size 16 --epochs 2"

# A. The configuration EPDE's own test scenarios use: second Pareto axis is
#    coefficient instability, so nothing in the objective penalises complexity.
$PY run_benchmark.py $COMMON --tag instability --trace

# B. Second Pareto axis is structural complexity -- the classic
#    accuracy-vs-parsimony front.
$PY run_benchmark.py $COMMON --tag complexity --trace \
    --second-objective complexity

# C. Ablation: GOLEM's generational scheme, which in multi-objective mode
#    applies no survival selection at all (see README finding 2).
$PY run_benchmark.py --scenarios wave kdv van_der_pol --seeds 0 1 2 \
    --pop-size 16 --epochs 2 --arms golem_eq_gen golem_eq_budget \
    --tag scheme_generational --golem-scheme generational \
    --budget-from results/benchmark_instability.jsonl

# D. Ablation: GOLEM's multi-armed-bandit mutation agent instead of uniform
#    random choice between the two mutation actions.
$PY run_benchmark.py --scenarios wave kdv van_der_pol --seeds 0 1 2 \
    --pop-size 16 --epochs 2 --arms golem_eq_gen golem_eq_budget \
    --tag agent_bandit --golem-adaptive-mutation bandit \
    --budget-from results/benchmark_instability.jsonl

echo "ALL DONE"
