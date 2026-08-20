"""Minimal usage demo: EPDE equation discovery with GOLEM as the engine.

    PYTHONHASHSEED=0 python demo.py

The only difference from a stock EPDE script is the import on the next line.
Everything after it -- preprocessing, the token pool, ``fit``, ``equations()``
-- is unchanged EPDE API.
"""

import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from epde_golem import EpdeGolemSearch as EpdeSearch   # noqa: E402
# from epde.interface.interface import EpdeSearch      # ...the stock engine

DATA = os.path.join(REPO_ROOT, 'tests', 'functional', 'scenarios', 'wave')


def main():
    np.random.seed(0)
    shape = 80
    data = np.loadtxt(os.path.join(DATA, 'wave_sln_80.csv'), delimiter=',').T
    axis = np.linspace(0, 1, shape + 1)
    grids = np.meshgrid(axis, axis, indexing='ij')

    search = EpdeSearch(use_solver=False, multiobjective_mode=True, use_pic=False,
                        boundary=20, coordinate_tensors=grids,
                        verbose_params={'show_iter_idx': False}, device='cpu')
    search.set_preprocessor(default_preprocessor_type='FD', preprocessor_kwargs={})
    search.set_moeadd_params(population_size=16, training_epochs=4)

    search.fit(data=data, variable_names=['u'], max_deriv_order=(2, 3),
               equation_terms_max_number=5, data_fun_pow=3, additional_tokens=[],
               equation_factors_max_number={'factors_num': [1, 2],
                                            'probas': [0.65, 0.35]},
               eq_sparsity_interval=(1e-6, 1e-4))

    print('\nNon-dominated front (ground truth: u_tt = 0.04 u_xx):\n')
    search.equations(only_print=True, num=1)
    print(f'\nequation evaluations: {search.optimizer.evaluator.n_evaluations}')


if __name__ == '__main__':
    main()
