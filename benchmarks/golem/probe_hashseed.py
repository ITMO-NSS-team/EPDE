"""Locate the code whose behaviour depends on Python's string-hash salt.

Runs are reproducible under a fixed ``PYTHONHASHSEED`` and diverge without one,
so something in the search iterates a container whose order the salt decides.
This runs one scenario and reports both the run's outcome and a census of the
suspect sites, so two invocations under different salts can be diffed.

    PYTHONHASHSEED=1 python probe_hashseed.py > a.txt
    PYTHONHASHSEED=2 python probe_hashseed.py > b.txt
    diff a.txt b.txt
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import scenarios as scenarios_module  # noqa: E402
from run_benchmark import run_once  # noqa: E402

def main():
    print('PYTHONHASHSEED =', os.environ.get('PYTHONHASHSEED'))

    scenario = scenarios_module.ALL[os.environ.get('PROBE_SCENARIO', 'wave')]()
    scenario['pop_size'] = 16
    scenario['epochs'] = 2
    record = run_once('native', scenario, 0)

    print('evaluations      :', record['fitness_evaluations'])
    print('front size       :', record['front_size'])
    print('best discrepancy :', record['best_discrepancy'])
    print('structure match  :', record['structure_match'])
    print('front equations  :')
    for forms in record['front_equations']:
        for line in forms:
            print('   ', line)


if __name__ == '__main__':
    main()
