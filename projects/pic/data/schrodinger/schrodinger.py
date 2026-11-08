"""Equation discovery on the linear Schrodinger equation with a harmonic potential.

    i psi_t = -1/2 psi_xx + 1/2 x^2 psi,    psi = u + i v

Truth, as the real two-equation system EPDE searches for (axes (t, x): du/dx0 = u_t, d^2u/dx1^2 = u_xx):

    u_t = -0.5 v_xx + 0.5 x^2 v
    v_t =  0.5 u_xx - 0.5 x^2 u

The potential term needs a grid factor (x^2) times the other field, so the pool carries grid tokens up to power 2
and terms of up to two factors. Data: schrodinger_qho.npz from schrodinger_gen.py (exact, no time-stepping error).
"""
import os
import sys

# The repo itself, ahead of any installed epde (site-packages holds an old 1.2.17).
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..')))

import numpy as np

from epde import GridTokens
from epde.interface.interface import EpdeSearch

FOLDER = os.path.dirname(os.path.abspath(__file__))
TRUE_SYSTEM = ('u_t = -0.5 v_xx + 0.5 x^2 v\n'
               'v_t =  0.5 u_xx - 0.5 x^2 u')


def noise_data(data, noise_level):
    # noise_level in percent of the field's standard deviation, as the other pic/data scripts
    return noise_level * 0.01 * np.std(data) * np.random.normal(size=data.shape) + data


def schrodinger_data(filename: str = os.path.join(FOLDER, 'schrodinger_qho.npz')):
    if not os.path.exists(filename):
        raise FileNotFoundError(f'{filename} is missing: run schrodinger_gen.py first')
    z = np.load(filename)
    grids = np.meshgrid(z['t'], z['x'], indexing='ij')
    return grids, {'u': z['u'], 'v': z['v']}


def schrodinger_discovery(noise_level: float = 0):
    grid, data = schrodinger_data()
    data = {var: noise_data(field, noise_level) for var, field in data.items()}
    print('Shapes:', {var: field.shape for var, field in data.items()}, 'grid', grid[0].shape)

    epde_search_obj = EpdeSearch(use_solver=False, multiobjective_mode=True, device='cpu')
    _, domain = epde_search_obj.createDomain(grid, ID=0)
    epde_search_obj.set_preprocessor(default_preprocessor_type='FD', preprocessor_kwargs={})

    grid_tokens = GridTokens(['t', 'x'], max_power=2)
    factors_max_number = {'factors_num': [1, 2], 'probas': [0.65, 0.35]}

    _, trajectory = epde_search_obj.createTrajectory(data, domain, cache_id=0)
    epde_search_obj.fit(data=[trajectory], max_deriv_order=(1, 2), data_fun_pow=1,
                        equation_terms_max_number=5,
                        additional_tokens=[grid_tokens],
                        equation_factors_max_number=factors_max_number)

    print('\nTrue system:\n' + TRUE_SYSTEM + '\n')
    epde_search_obj.equations(only_print=True, num=1)
    return epde_search_obj


if __name__ == '__main__':
    schrodinger_discovery(noise_level=0)
