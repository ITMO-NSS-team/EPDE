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


def aizawa_discovery(noise_level):
    data_file = os.path.join(os.path.dirname(__file__), 'aizawa.npz')
    data = np.load(data_file)
    t = data['t']
    u = data['u']

    x = u[..., 0]
    y = u[..., 1]
    z = u[..., 2]
    dimensionality = x.ndim - 1

    trig_tokens = TrigonometricTokens(freq=(2 - 1e-8, 2 + 1e-8),
                                      dimensionality=dimensionality)
    grid_tokens = GridTokens(['x_0', ], dimensionality=dimensionality, max_power=2)

    epde_search_obj = EpdeSearch(use_solver=False, multiobjective_mode=True, use_pic=True, boundary=15,
                                 coordinate_tensors=(t,), verbose_params={'show_iter_idx': True},
                                 device='cuda')

    epde_search_obj.set_preprocessor(default_preprocessor_type='FD',
                                     preprocessor_kwargs={})

    popsize = 16
    epde_search_obj.set_moeadd_params(population_size=popsize, training_epochs=50)

    factors_max_number = {'factors_num': [1, 2], 'probas' : [0.8, 0.2]}

    epde_search_obj.fit(data=[x, y, z], variable_names=['x', 'y', 'z'], max_deriv_order=(1,),
                        equation_terms_max_number=7, data_fun_pow=3, additional_tokens=[],
                        equation_factors_max_number=factors_max_number,
                        eq_sparsity_interval=(1e-8, 1e-0))  #

    epde_search_obj.equations(only_print=True, num=1)
    epde_search_obj.visualize_solutions()

    return epde_search_obj


if __name__ == "__main__":
    import torch
    print(torch.cuda.is_available())

    aizawa_discovery(0)
