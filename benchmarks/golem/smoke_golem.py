"""Minimal end-to-end check of the GOLEM-backed EPDE search on the wave data."""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import DATA_DIR, set_seeds  # noqa: E402

import numpy as np  # noqa: E402
from epde_golem import EpdeGolemSearch  # noqa: E402


def main():
    set_seeds(0)
    folder = os.path.join(DATA_DIR, 'wave')
    shape = 80
    data = np.loadtxt(os.path.join(folder, 'wave_sln_80.csv'), delimiter=',').T
    t = np.linspace(0, 1, shape + 1)
    x = np.linspace(0, 1, shape + 1)
    grids = np.meshgrid(t, x, indexing='ij')

    search = EpdeGolemSearch(
        use_solver=False, multiobjective_mode=True, use_pic=True, boundary=20,
        coordinate_tensors=grids, verbose_params={'show_iter_idx': False},
        device='cpu', golem_params={'show_progress': True})
    search.set_preprocessor(default_preprocessor_type='FD', preprocessor_kwargs={})
    search.set_moeadd_params(population_size=8, training_epochs=2)

    t0 = time.perf_counter()
    search.fit(data=data, variable_names=['u'], max_deriv_order=(2, 2),
               equation_terms_max_number=5, data_fun_pow=1, additional_tokens=[],
               equation_factors_max_number={'factors_num': [1, 2],
                                            'probas': [0.65, 0.35]},
               eq_sparsity_interval=(1e-6, 1e-4))
    print('ELAPSED', time.perf_counter() - t0)
    print('evaluations:', search.optimizer.evaluator.n_evaluations,
          'failures:', search.optimizer.evaluator.n_failures)
    search.equations(only_print=True, num=1)


if __name__ == '__main__':
    main()
