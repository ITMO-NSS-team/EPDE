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


def vdp_discovery(foldername, noise_level):
    step = 0.05;
    steps_num = 320
    t = np.arange(start=0., stop=step * steps_num, step=step)
    data = np.load(os.path.join(foldername, 'vdp_data.npy'))
    noised_data = noise_data(data, noise_level)
    data_nn = load_pretrained_PINN(os.path.join(foldername, 'vdp_ann_pretrained.pickle'))

    dimensionality = 0

    trig_tokens = TrigonometricTokens(freq=(2 - 1e-8, 2 + 1e-8),
                                      dimensionality=dimensionality)
    grid_tokens = GridTokens(['x_0', ], dimensionality=dimensionality, max_power=2)

    epde_search_obj = EpdeSearch(use_solver=False, use_pic=True, boundary=32,
                                 coordinate_tensors=(t,), verbose_params={'show_iter_idx': True},
                                 device='cuda')

    epde_search_obj.set_preprocessor(default_preprocessor_type='FD',
                                     preprocessor_kwargs={})

    popsize = 16
    epde_search_obj.set_moeadd_params(population_size=popsize, training_epochs=5)

    factors_max_number = {'factors_num': [1, 2], 'probas': [0.65, 0.35]}

    epde_search_obj.fit(data=[noised_data, ], variable_names=['u', ], max_deriv_order=(2, 3),
                        equation_terms_max_number=5, data_fun_pow=3,
                        additional_tokens=[trig_tokens, grid_tokens],
                        equation_factors_max_number=factors_max_number,
                        eq_sparsity_interval=(1e-5, 1e-0)) #

    epde_search_obj.equations(only_print=True, num=1)

    # import pickle
    # fname = os.path.join(r'C:\Users\user\EPDE_tests\models', 'ode_0_ann.pickle')
    # with open(fname, 'wb') as output_file:
    #     pickle.dump(global_var.solution_guess_nn, output_file)
    epde_search_obj.visualize_solutions()
    return epde_search_obj


if __name__ == "__main__":
    import torch
    print(torch.cuda.is_available())

    # Paths
    directory = os.path.dirname(os.path.realpath(__file__))

    vdp_folder_name = os.path.join(directory)

    vdp_discovery(vdp_folder_name, 0)