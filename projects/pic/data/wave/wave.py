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


def wave_data(filename):
    shape = 80

    # print(os.path.dirname( __file__ ))
    data = np.loadtxt(filename, delimiter=',').T
    t = np.linspace(0, 1, shape + 1);
    x = np.linspace(0, 1, shape + 1)
    grids = np.stack(np.meshgrid(t, x, indexing='ij'), axis=2)
    return grids, data


def wave_discovery(foldername, noise_level):
    grid, data = wave_data(os.path.join(foldername, 'wave_sln_80.csv'))
    noised_data = noise_data(data, noise_level)
    data_nn = load_pretrained_PINN(os.path.join(foldername, 'ann_pretrained.pickle'))

    dimensionality = data.ndim - 1

    epde_search_obj = EpdeSearch(use_solver=False, use_pic=True,
                                      boundary=20,
                                      coordinate_tensors=(grid[..., 0], grid[..., 1]), device='cuda')

    # epde_search_obj.set_preprocessor(default_preprocessor_type='ANN',
    #                                     preprocessor_kwargs={'epochs_max' : 1e4})
    # epde_search_obj.set_preprocessor(default_preprocessor_type='spectral',
    #                                  preprocessor_kwargs={"n": 80})
    epde_search_obj.set_preprocessor(default_preprocessor_type='FD',
                                     preprocessor_kwargs={})
    # epde_search_obj.set_preprocessor(default_preprocessor_type='poly',
    #                                  preprocessor_kwargs={'use_smoothing': True})
    popsize = 16

    epde_search_obj.set_moeadd_params(population_size=popsize,
                                      training_epochs=1)


    custom_grid_tokens = CacheStoredTokens(token_type='grid',
                                                token_labels=['t', 'x'],
                                                token_tensors={'t': grid[..., 0], 'x': grid[..., 1]},
                                                params_ranges={'power': (1, 1)},
                                                params_equality_ranges=None)

    trig_params_ranges = {'power': (1, 1)}
    trig_params_equal_ranges = {}

    trig_tokens = TrigonometricTokens(dimensionality=dimensionality, freq = (0.999, 1.001))

    factors_max_number = {'factors_num': [1, 2], 'probas': [0.65, 0.35]}

    bounds = (1e-6, 1e-4)
    epde_search_obj.fit(data=noised_data, variable_names=['u', ], max_deriv_order=(2, 3), derivs=None,
                        equation_terms_max_number=5, data_fun_pow=3,
                        additional_tokens=[],
                        equation_factors_max_number=factors_max_number,
                        eq_sparsity_interval=bounds, fourier_layers=False) # , data_nn=data_nn

    epde_search_obj.equations(only_print=True, num=1)
    epde_search_obj.visualize_solutions()

    return epde_search_obj


if __name__ == "__main__":
    import torch
    print(torch.cuda.is_available())

    # Paths
    directory = os.path.dirname(os.path.realpath(__file__))
    wave_folder_name = os.path.join(directory)

    wave_discovery(wave_folder_name, 0)