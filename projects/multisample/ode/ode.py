import os
import numpy as np
import os
import sys
from typing import Union, Callable

import matplotlib.pyplot as plt
import matplotlib as mpl
import epde
from epde.structure.domain import VariableEntry, Trajectory, Domain

def ODE_discovery():
    C1 = 1.3
    t1 = np.linspace(0, 4 * np.pi, 200)
    x1 = np.sin(t1) + C1 * np.cos(t1)
    x1_dot = np.cos(t1) - C1 * np.sin(t1)
    x1_dot_dot = -np.sin(t1) - C1 * np.cos(t1)
    x1_dot_dot_dot = -np.cos(t1) + C1 * np.sin(t1)

    C2 = 0.7
    x2 = np.sin(t1) + C2 * np.cos(t1)
    x2_dot = np.cos(t1) - C2 * np.sin(t1)
    x2_dot_dot = -np.sin(t1) - C2 * np.cos(t1)
    x2_dot_dot_dot = -np.cos(t1) + C2 * np.sin(t1)
    max_axis_idx = x2.ndim - 1

    t2 = np.linspace(0.1, 2 * np.pi, 152)
    C3 = 1.7
    x3 = np.sin(t2) + C3 * np.cos(t2)
    x3_dot = np.cos(t2) - C3 * np.sin(t2)
    x3_dot_dot = -np.sin(t2) - C3 * np.cos(t2)
    x3_dot_dot_dot = -np.cos(t2) + C3 * np.sin(t2)

    plt.plot(t1, x1, color='k', label='x_1(t)')
    plt.plot(t1, x1_dot, '--', color='k', label="x_1'(t)")

    plt.plot(t1, x2, color='r', label='x_2(t)')
    plt.plot(t1, x2_dot, '--', color='r', label="x_2'(t)")

    plt.plot(t2, x3, color='b', label='x_3(t)')
    plt.plot(t2, x3_dot, '--', color='b', label="x_3'(t)")

    plt.legend(loc='upper right')
    plt.xlabel('t')
    plt.ylabel('x')
    plt.show()

    n_epochs = 15
    popsize = 8

    epde_search_obj = epde.EpdeSearch(use_solver=False, multiobjective_mode=True,  # boundary = bnd,
                                      verbose_params={'show_iter_idx': True},
                                      device='cpu')  # False for brevity coordinate_tensors = [t,],
    epde_search_obj.set_preprocessor(default_preprocessor_type='poly',
                                     preprocessor_kwargs={'use_smoothing': False, 'sigma': 1,
                                                          'polynomial_window': 3, 'poly_order': 3})
    trig_tokens = epde.TrigonometricTokens(freq=(0.999, 1.001), dimensionality=max_axis_idx)
    grid_tokens = epde.GridTokens(['x_0', ], dimensionality=max_axis_idx, max_power=2)

    epde_search_obj.set_moeadd_params(population_size=popsize, training_epochs=n_epochs)

    factors_max_number = {'factors_num': [1, 2], 'probas': [0.5, 0.5]}  # 1 factor with P = 0.65, 2 with P = 0.35

    bnd = 10
    domain_id_1, domain_1 = epde_search_obj.createDomain(t1, boundary_width=bnd, ID=0)
    domain_id_2, domain_2 = epde_search_obj.createDomain(t2, boundary_width=bnd, ID=1)

    print(domain_2.ID)

    # sample_id_1, sample_1 = epde_search_obj.createTrajectory({'u': x1, }, domain_1, cache_id=0)
    # sample_id_2, sample_2 = epde_search_obj.createTrajectory({'u': x2, }, domain_1, cache_id=1)
    # sample_id_3, sample_3 = epde_search_obj.createTrajectory({'u': x3, }, domain_2, cache_id=2)

    sample_id_1, sample_1 = epde_search_obj.createTrajectory({'u': x1, }, domain_1, cache_id=0)
    sample_id_2, sample_2 = epde_search_obj.createTrajectory({'u': x2, }, domain_1, cache_id=1)
    sample_id_3, sample_3 = epde_search_obj.createTrajectory({'u': x3, }, domain_2, cache_id=2)

    u = [sample_1, sample_2, sample_3]

    # One pool, three samples: max_deriv_order and data_fun_pow describe the
    # pool, so they are stated once here rather than repeated per trajectory.
    epde_search_obj.fit(data=u, max_deriv_order=(1,), data_fun_pow=3,
                        equation_terms_max_number=5,
                        additional_tokens=[trig_tokens, grid_tokens],
                        equation_factors_max_number=factors_max_number)

    epde_search_obj.equations()

    epde_search_obj.visualize_solutions()

if __name__ == "__main__":
    ODE_discovery()
