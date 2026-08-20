"""End-to-end check of the GOLEM backend in EPDE's single-objective mode.

Runs the same scenario through EPDE's ``SimpleOptimizer`` and through GOLEM,
and prints what each returned.
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

from _common import DATA_DIR, set_seeds, check_hash_seed  # noqa: E402
import metrics  # noqa: E402

GROUND_TRUTH = ['0.04 * d^2u/dx1^2{power: 1.0} + 0.0 = d^2u/dx0^2{power: 1.0}']


def wave():
    shape = 80
    data = np.loadtxt(os.path.join(DATA_DIR, 'wave', 'wave_sln_80.csv'), delimiter=',').T
    axis = np.linspace(0, 1, shape + 1)
    return np.meshgrid(axis, axis, indexing='ij'), data


def run(engine, grids, data, pop_size=12, epochs=3, seed=0):
    set_seeds(seed)
    if engine == 'native':
        from epde.interface.interface import EpdeSearch as Cls
        extra = {}
    else:
        from epde_golem import EpdeGolemSearch as Cls
        extra = {'golem_params': {}}
    search = Cls(use_solver=False, multiobjective_mode=False, boundary=20,
                 coordinate_tensors=grids, verbose_params={'show_iter_idx': False},
                 device='cpu', **extra)
    search.set_preprocessor(default_preprocessor_type='FD', preprocessor_kwargs={})
    search.set_singleobjective_params(population_size=pop_size, training_epochs=epochs)

    t0 = time.perf_counter()
    search.fit(data=data, variable_names=['u'], max_deriv_order=(2, 2),
               equation_terms_max_number=5, data_fun_pow=1, additional_tokens=[],
               equation_factors_max_number=1, eq_sparsity_interval=(1e-6, 1e-6))
    elapsed = time.perf_counter() - t0

    population = search.optimizer.population.population
    best = metrics.best_match(population, GROUND_TRUTH)
    print(f'--- {engine}: {elapsed:.2f}s, population {len(population)}, '
          f'match={best["structure_match"]}')
    for system in population[:3]:
        print('   ', system.text_form.split('\n')[0][:140])
    return elapsed, best


def main():
    check_hash_seed()
    grids, data = wave()
    for engine in ('native', 'golem'):
        run(engine, grids, data)


if __name__ == '__main__':
    main()
