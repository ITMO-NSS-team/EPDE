"""Profile a single run to see where the wall clock goes.

    python profile_golem.py <arm> <scenario> [pop_size] [epochs] [budget]
"""

import cProfile
import io
import os
import pstats
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import scenarios as scenarios_module  # noqa: E402
from run_benchmark import run_once  # noqa: E402


def main():
    arm = sys.argv[1] if len(sys.argv) > 1 else 'golem_eq_gen'
    name = sys.argv[2] if len(sys.argv) > 2 else 'wave'
    pop = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    epochs = int(sys.argv[4]) if len(sys.argv) > 4 else 2
    budget = int(sys.argv[5]) if len(sys.argv) > 5 else None

    scenario = scenarios_module.ALL[name]()
    scenario['pop_size'] = pop
    scenario['epochs'] = epochs

    profiler = cProfile.Profile()
    profiler.enable()
    run_once(arm, scenario, 0, eval_budget=budget,
             epochs=500 if budget else epochs)
    profiler.disable()

    stream = io.StringIO()
    pstats.Stats(profiler, stream=stream).sort_stats('tottime').print_stats(30)
    print(stream.getvalue())


if __name__ == '__main__':
    main()
