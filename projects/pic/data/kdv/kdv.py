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
        import torch
        data_nn = torch.load(ann_filename, map_location=torch.device('cpu'))
    except FileNotFoundError:
        print('No model located, proceeding with ann approx. retraining.')
        data_nn = None
    return data_nn


def noise_data(data, noise_level):
    # add noise level to the input data
    return noise_level * np.std(data) * np.random.normal(size=data.shape) + data


def kdv_data(filename, shape=80):
    shape = 80

    print(os.path.dirname(__file__))
    data = np.loadtxt(filename, delimiter=',').T
    t = np.linspace(0, 1, shape + 1)
    x = np.linspace(0, 1, shape + 1)
    grids = np.meshgrid(t, x, indexing='ij')  # np.stack(, axis = 2)
    return grids, data


def kdv_data_h(filename, shape=80):
    shape = 119

    print(os.path.dirname(__file__))
    data = np.load(filename)

    t = np.linspace(0, 1, 120)
    x = np.linspace(-3, 3, 480)
    grids = np.meshgrid(t, x, indexing='ij')  # np.stack(, axis = 2)
    return grids, data


def kdv_data_sga(filename):
    data = scio.loadmat(filename)
    u = data.get("uu").T
    n, m = u.shape
    x = np.squeeze(data.get("x")).reshape(-1, 1)
    t = np.squeeze(data.get("tt").reshape(-1, 1))
    grids = np.meshgrid(t, x, indexing='ij')  # np.stack(, axis = 2)
    return grids, u

def kdv_sindy_data(filename):
    data = scio.loadmat(filename)
    t = np.ravel(data['t'])
    x = np.ravel(data['x'])
    u = np.real(data['usol'])
    u = np.transpose(u)
    grids = np.meshgrid(t, x, indexing='ij')  # np.stack(, axis = 2)
    return grids, u

def kdv_discovery(foldername, noise_level):
    grid, data = kdv_data(os.path.join(foldername, 'data.csv'))
    noised_data = noise_data(data, noise_level)
    data_nn = load_pretrained_PINN(os.path.join(foldername, f'kdv_{noise_level}_ann.pickle'))

    dimensionality = data.ndim - 1

    epde_search_obj = EpdeSearch(use_solver=False, use_pic=True,
                                      boundary=5,
                                      coordinate_tensors=grid, device='cuda' if torch.cuda.is_available() else 'cpu')

    # epde_search_obj.set_preprocessor(default_preprocessor_type='ANN',
    #                                     preprocessor_kwargs={'epochs_max' : 1e3})
    epde_search_obj.set_preprocessor(default_preprocessor_type='FD',
                                     preprocessor_kwargs={})
    popsize = 16

    epde_search_obj.set_moeadd_params(population_size=popsize,
                                      training_epochs=5)

    custom_trigonometric_eval_fun = {
        'cos(t)sin(x)': lambda *grids, **kwargs: (np.cos(grids[0]) * np.sin(grids[1])) ** kwargs['power']}
    custom_trig_evaluator = CustomEvaluator(custom_trigonometric_eval_fun, eval_fun_params_labels=['power'])
    trig_params_ranges = {'power': (1, 1)}
    trig_params_equal_ranges = {}

    custom_trig_tokens = CustomTokens(token_type='trigonometric',
                                         token_labels=['cos(t)sin(x)'],
                                         evaluator=custom_trig_evaluator,
                                         params_ranges=trig_params_ranges,
                                         params_equality_ranges=trig_params_equal_ranges,
                                         meaningful=True, unique_token_type=False)

    trig_tokens = TrigonometricTokens(dimensionality=dimensionality, freq = (0.999, 1.001))
    trig_tokens._token_family.set_status(unique_specific_token=True, unique_token_type=False,
                                  meaningful=True)

    factors_max_number = {'factors_num': [1, 2], 'probas': [0.65, 0.35]}

    bounds = (1e-5, 1e-2)
    epde_search_obj.fit(data=noised_data, variable_names=['u', ], max_deriv_order=(2, 3), derivs=None,
                        equation_terms_max_number=10, data_fun_pow=3,
                        additional_tokens=[custom_trig_tokens], #custom_trig_tokens
                        equation_factors_max_number=factors_max_number,
                        eq_sparsity_interval=bounds, fourier_layers=False) # , data_nn=data_nn

    epde_search_obj.equations(only_print=True, num=1)
    res = epde_search_obj.equations(only_print=False, num=1)
    epde_search_obj.visualize_solutions()

    return epde_search_obj


def kdv_h_discovery(foldername, noise_level):
    grid, data = kdv_data_h(os.path.join(foldername, 'data_kdv_homogen.npy'))
    noised_data = noise_data(data, noise_level)
    data_nn = load_pretrained_PINN(os.path.join(foldername, f'kdv_{noise_level}_ann.pickle'))

    dimensionality = data.ndim - 1

    epde_search_obj = EpdeSearch(use_solver=False, use_pic=True,
                                      boundary=20,
                                      coordinate_tensors=grid, device='cuda' if torch.cuda.is_available() else 'cpu')

    # epde_search_obj.set_preprocessor(default_preprocessor_type='ANN',
    #                                     preprocessor_kwargs={'epochs_max' : 1e3})
    epde_search_obj.set_preprocessor(default_preprocessor_type='FD',
                                     preprocessor_kwargs={})
    popsize = 8

    epde_search_obj.set_moeadd_params(population_size=popsize,
                                      training_epochs=1)


    custom_grid_tokens = CacheStoredTokens(token_type='grid',
                                                token_labels=['t', 'x'],
                                                token_tensors={'t': grid[0], 'x': grid[1]},
                                                params_ranges={'power': (1, 1)},
                                                params_equality_ranges=None)

    custom_trigonometric_eval_fun = {
        'cos(t)sin(x)': lambda *grids, **kwargs: (np.cos(grids[0]) * np.sin(grids[1])) ** kwargs['power']}
    custom_trig_evaluator = CustomEvaluator(custom_trigonometric_eval_fun, eval_fun_params_labels=['power'])
    trig_params_ranges = {'power': (1, 1)}
    trig_params_equal_ranges = {}

    custom_trig_tokens = CustomTokens(token_type='trigonometric',
                                         token_labels=['cos(t)sin(x)'],
                                         evaluator=custom_trig_evaluator,
                                         params_ranges=trig_params_ranges,
                                         params_equality_ranges=trig_params_equal_ranges,
                                         meaningful=True, unique_token_type=True)

    trig_tokens = TrigonometricTokens(dimensionality=dimensionality, freq = (0.999, 1.001))

    factors_max_number = {'factors_num': [1, 2], 'probas': [0.65, 0.35]}

    bounds = (1e-5, 1e-0)
    epde_search_obj.fit(data=noised_data, variable_names=['u', ], max_deriv_order=(2, 3), derivs=None,
                        equation_terms_max_number=5, data_fun_pow=1,
                        additional_tokens=[],
                        equation_factors_max_number=factors_max_number,
                        eq_sparsity_interval=bounds, fourier_layers=False) # , data_nn=data_nn

    epde_search_obj.equations(only_print=True, num=1)
    epde_search_obj.visualize_solutions()

    return epde_search_obj

def kdv_sga_discovery(foldername, noise_level):
    grid, data = kdv_data_sga(os.path.join(foldername, 'Kdv.mat'))
    noised_data = noise_data(data, noise_level)
    data_nn = load_pretrained_PINN(os.path.join(foldername, f'kdv_{noise_level}_ann.pickle'))

    dimensionality = data.ndim - 1

    epde_search_obj = EpdeSearch(use_solver=False, use_pic=True,
                                      boundary=20,
                                      coordinate_tensors=grid, device='cuda' if torch.cuda.is_available() else 'cpu')

    epde_search_obj.set_preprocessor(default_preprocessor_type='ANN',
                                        preprocessor_kwargs={'epochs_max' : 1e3})
    # epde_search_obj.set_preprocessor(default_preprocessor_type='poly',
    #                                  preprocessor_kwargs={})
    popsize = 8

    epde_search_obj.set_moeadd_params(population_size=popsize,
                                      training_epochs=5)


    custom_grid_tokens = CacheStoredTokens(token_type='grid',
                                                token_labels=['t', 'x'],
                                                token_tensors={'t': grid[0], 'x': grid[1]},
                                                params_ranges={'power': (1, 1)},
                                                params_equality_ranges=None)

    custom_trigonometric_eval_fun = {
        'cos(t)sin(x)': lambda *grids, **kwargs: (np.cos(grids[0]) * np.sin(grids[1])) ** kwargs['power']}
    custom_trig_evaluator = CustomEvaluator(custom_trigonometric_eval_fun, eval_fun_params_labels=['power'])
    trig_params_ranges = {'power': (1, 1)}
    trig_params_equal_ranges = {}

    custom_trig_tokens = CustomTokens(token_type='trigonometric',
                                         token_labels=['cos(t)sin(x)'],
                                         evaluator=custom_trig_evaluator,
                                         params_ranges=trig_params_ranges,
                                         params_equality_ranges=trig_params_equal_ranges,
                                         meaningful=True, unique_token_type=True)

    trig_tokens = TrigonometricTokens(dimensionality=dimensionality, freq = (0.999, 1.001))

    factors_max_number = {'factors_num': [1, 2], 'probas': [0.65, 0.35]}

    bounds = (1e-3, 1e-0)
    epde_search_obj.fit(data=noised_data, variable_names=['u', ], max_deriv_order=(2, 3), derivs=None,
                        equation_terms_max_number=5, data_fun_pow=1,
                        additional_tokens=[],
                        equation_factors_max_number=factors_max_number,
                        eq_sparsity_interval=bounds, fourier_layers=False) # , data_nn=data_nn

    epde_search_obj.equations(only_print=True, num=1)
    epde_search_obj.equations(only_print=False, num=1)
    epde_search_obj.visualize_solutions()

    return epde_search_obj


def kdv_sindy_discovery(foldername, noise_level):
    grid, data = kdv_sindy_data(os.path.join(foldername, 'kdv_sindy.mat'))
    noised_data = noise_data(data, noise_level)
    # data_nn = load_pretrained_PINN(os.path.join(foldername, f'kdv_{noise_level}_ann.pickle'))

    dimensionality = data.ndim - 1

    epde_search_obj = EpdeSearch(use_solver=False, use_pic=True,
                                      boundary=(40, 100),
                                      coordinate_tensors=grid, device='cuda' if torch.cuda.is_available() else 'cpu')

    # epde_search_obj.set_preprocessor(default_preprocessor_type='ANN',
    #                                     preprocessor_kwargs={'epochs_max' : 1e3})
    epde_search_obj.set_preprocessor(default_preprocessor_type='poly',
                                     preprocessor_kwargs={}) #'use_smoothing': True
    popsize = 16

    epde_search_obj.set_moeadd_params(population_size=popsize,
                                      training_epochs=3)

    custom_trigonometric_eval_fun = {
        'cos(t)sin(x)': lambda *grids, **kwargs: (np.cos(grids[0]) * np.sin(grids[1])) ** kwargs['power']}
    custom_trig_evaluator = CustomEvaluator(custom_trigonometric_eval_fun, eval_fun_params_labels=['power'])
    trig_params_ranges = {'power': (1, 1)}
    trig_params_equal_ranges = {}

    custom_trig_tokens = CustomTokens(token_type='trigonometric',
                                         token_labels=['cos(t)sin(x)'],
                                         evaluator=custom_trig_evaluator,
                                         params_ranges=trig_params_ranges,
                                         params_equality_ranges=trig_params_equal_ranges,
                                         meaningful=True, unique_token_type=False)

    trig_tokens = TrigonometricTokens(dimensionality=dimensionality, freq = (0.999, 1.001))
    trig_tokens._token_family.set_status(unique_specific_token=True, unique_token_type=False,
                                  meaningful=True)

    factors_max_number = {'factors_num': [1, 2], 'probas': [0.65, 0.35]}

    bounds = (1e-12, 1e-0)
    epde_search_obj.fit(data=noised_data, variable_names=['u', ], max_deriv_order=(2, 3), derivs=None,
                        equation_terms_max_number=5, data_fun_pow=3,
                        additional_tokens=[],
                        equation_factors_max_number=factors_max_number,
                        eq_sparsity_interval=bounds, fourier_layers=False) # , data_nn=data_nn

    epde_search_obj.equations(only_print=True, num=1)
    res = epde_search_obj.equations(only_print=False, num=1)
    epde_search_obj.visualize_solutions()

    return epde_search_obj


if __name__ == "__main__":
    import torch
    print("CUDA available:", torch.cuda.is_available())

    # Paths
    directory = os.path.dirname(os.path.realpath(__file__))
    kdv_folder_name = os.path.join(directory)

    # kdv_discovery(kdv_folder_name, 0)
    # kdv_h_discovery(kdv_folder_name, 0)
    # kdv_sga_discovery(kdv_folder_name, 5)
    kdv_sindy_discovery(kdv_folder_name, 0)