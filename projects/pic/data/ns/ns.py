import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../..')))

import pickle
from typing import Tuple, List
import numpy as np

import copy
from epde.interface.token_family import TFPool
from epde.structure.main_structures import SoEq, Chromosome
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


def ns_data(filename: str):
    data = scio.loadmat('cylinder_nektar_wake.mat')
    U_star = data['U_star']  # N x 2 x T
    P_star = data['p_star']  # N x T
    t_star = data['t']  # T x 1
    X_star = data['X_star']  # N x 2

    N = X_star.shape[0]
    T = t_star.shape[0]

    t_train = 50

    x = np.unique(X_star[:, 0:1].flatten())  # N x T
    y = np.unique(X_star[:, 1:2].flatten()) # N x T
    t = t_star.flatten()  # N x T

    u = U_star[:, 0, :].T.reshape(*t.shape, *y.shape, *x.shape)[:t_train] # N x T
    v = U_star[:, 1, :].T.reshape(*t.shape, *y.shape, *x.shape)[:t_train] # N x T
    p = P_star.T.reshape(*t.shape, *y.shape, *x.shape)[:t_train]   # N x T

    grids = np.meshgrid(t[:t_train], y, x, indexing = 'ij')  # np.stack(, axis = 2) , axis = 2)
    data = [u, v, p]
    return grids, data


def ns_discovery(foldername, noise_level):
    grid, data = ns_data(os.path.join(foldername, 'cylinder_nektar_wake.mat'))
    # noised_data = noise_data(data, noise_level)
    data_nn = load_pretrained_PINN(os.path.join(foldername, f'kdv_{noise_level}_ann.pickle'))

    # dimensionality = data.ndim - 1

    epde_search_obj = EpdeSearch(use_solver=False, multiobjective_mode=True,
                                      use_pic=True, boundary=[21, 21, 46],
                                      coordinate_tensors=grid, device='cuda')

    # epde_search_obj.set_preprocessor(default_preprocessor_type='ANN',
    #                                     preprocessor_kwargs={'epochs_max' : 1e3})
    epde_search_obj.set_preprocessor(default_preprocessor_type='FD',
                                     preprocessor_kwargs={})
    popsize = 64

    epde_search_obj.set_moeadd_params(population_size=popsize,
                                      training_epochs=30)

    custom_grid_tokens = CacheStoredTokens(token_type='grid',
                                                token_labels=['t', 'x'],
                                                token_tensors={'t': grid[0], 'x': grid[1]},
                                                params_ranges={'power': (1, 1)},
                                                params_equality_ranges=None)

    trig_params_ranges = {'power': (1, 1)}
    trig_params_equal_ranges = {}

    # trig_tokens = TrigonometricTokens(dimensionality=dimensionality, freq = (0.999, 1.001))

    factors_max_number = {'factors_num': [1, 2], 'probas': [0.8, 0.2]}

    bounds = (1e-12, 1e-0)
    epde_search_obj.fit(data=data, variable_names=["u", "v", "p"], max_deriv_order=(1, 2, 2), derivs=None,
                        equation_terms_max_number=20, data_fun_pow=1,
                        additional_tokens=[],
                        equation_factors_max_number=factors_max_number,
                        eq_sparsity_interval=bounds, fourier_layers=False) # , data_nn=data_nn

    epde_search_obj.equations(only_print=True, num=1)
    epde_search_obj.visualize_solutions()

    return epde_search_obj


if __name__ == "__main__":
    import torch
    from epde.operators.utils.default_parameter_loader import EvolutionaryParams
    print(torch.cuda.is_available())

    # Paths
    directory = os.path.dirname(os.path.realpath(__file__))
    ns_folder_name = os.path.join(directory)

    ns_discovery(ns_folder_name, 0)
