import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../..')))

import pickle
from typing import Tuple, List
import numpy as np

from epde.interface.prepared_tokens import CustomTokens, PhasedSine1DTokens, ConstantToken, CustomEvaluator
from epde.interface.equation_translator import translate_equation
from epde.interface.interface import EpdeSearch

from epde import TrigonometricTokens, GridTokens, CacheStoredTokens
import epde.globals as global_var

import scipy.io as scio

def load_pretrained_PINN(ann_filename):
    try:
        with open(ann_filename, 'rb') as data_input_file:
            data_nn = pickle.load(data_input_file)
    except FileNotFoundError:
        print('No model located, proceeding with ann approx. retraining.')
        data_nn = None
    return data_nn


def noise_data(data, noise_level):
    # add noise level to the input data
    return noise_level * 0.01 * np.std(data) * np.random.normal(size=data.shape) + data


def harelynx_discovery(noise_level):
    import torch
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Leigh (1968) hare-lynx pelt counts: columns year,hare,lynx; 57 yearly rows 1847..1903.
    data_file = os.path.join(os.path.dirname(__file__), 'Leigh1968_harelynx.csv')
    raw = np.genfromtxt(data_file, delimiter=',', skip_header=1)

    # Time in years, starting at 0 (unit step).
    t = raw[:, 0].astype(np.float64)
    t = t - t[0]

    # Populations in thousands so u, v are O(1..150) and the u*v vs u coefficient
    # range stays conditioned. hare (prey) -> u, lynx (predator) -> v.
    u = raw[:, 1].astype(np.float64) / 1000.0
    v = raw[:, 2].astype(np.float64) / 1000.0

    u = noise_data(u, noise_level)
    v = noise_data(v, noise_level)

    dimensionality = u.ndim - 1

    # Pure-polynomial Lotka-Volterra RHS: keep only the grid token (absorbs any
    # secular trend). The synthetic LV script's freq~2 trig basis is meaningless
    # on real yearly counts, so it is dropped.
    grid_tokens = GridTokens(['x_0', ], dimensionality=dimensionality, max_power=2)

    # Only 57 points: keep the boundary cut small (a few points), not 15-32.
    epde_search_obj = EpdeSearch(use_solver=False, multiobjective_mode=True, use_pic=True, boundary=1,
                                 coordinate_tensors=(t,), verbose_params={'show_iter_idx': True},
                                 device=device)

    # Real data is noisy: Savitsky-Golay smoothing before differentiation
    # instead of raw finite differences.
    epde_search_obj.set_preprocessor(default_preprocessor_type='poly',
                                     preprocessor_kwargs={})

    popsize = 32
    epde_search_obj.set_moeadd_params(population_size=popsize, training_epochs=5)

    factors_max_number = {'factors_num': [1, 2], 'probas' : [0.8, 0.2]}

    # First-order ODE system (matches lv.yaml max_deriv_order=[1]); factors up to 2 so the u*v cross-term is representable.
    # grid_tokens
    epde_search_obj.fit(data=[u, v], variable_names=['u', 'v'], max_deriv_order=(1,),
                        equation_terms_max_number=7, data_fun_pow=3, additional_tokens=[],
                        equation_factors_max_number=factors_max_number,
                        eq_sparsity_interval=(1e-8, 1e-0))  #

    epde_search_obj.equations(only_print=True, num=1)
    epde_search_obj.visualize_solutions()

    return epde_search_obj


if __name__ == "__main__":
    import torch
    from epde.operators.utils.default_parameter_loader import EvolutionaryParams
    print(torch.cuda.is_available())

    harelynx_discovery(0)
