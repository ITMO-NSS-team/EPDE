import math
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
import matplotlib.pyplot as plt


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


def plot_all_projections(data, coords, dim_names=['t', 'x', 'y', 'z']):
    """
    Plots all 6 2D projections of 4D data.
    data: 4D numpy array (t, x, y, z)
    coords: list of 1D arrays [t_vals, x_vals, y_vals, z_vals]
    """
    # Pairs of dimensions to plot (0=t, 1=x, 2=y, 3=z)
    pairs = [
        (1, 2),  # X-Y (Beam Profile)
        (0, 1),  # T-X (Evolution along X)
        (0, 2),  # T-Y (Evolution along Y)
        (1, 3),  # X-Z (Side Profile)
        (2, 3),  # Y-Z (Front Profile)
        (0, 3)  # T-Z (Evolution along Z)
    ]

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for i, (d1, d2) in enumerate(pairs):
        ax = axes[i]

        # 1. Identify axes to collapse (the ones NOT in d1, d2)
        all_dims = {0, 1, 2, 3}
        collapse_dims = tuple(all_dims - {d1, d2})

        # 2. Create Projection (Max Intensity)
        # We use np.max to see the brightest spots ("hotspots")
        # Use np.mean() or np.sum() if you want total energy
        proj_2d = np.max(data, axis=collapse_dims)

        # 3. Handle Transpose for Plotting
        # pcolormesh expects (x, y) but numpy arrays are (row, col) -> (y, x)
        # So we pass coords[d2] as x-axis, coords[d1] as y-axis
        # and project_2d needs to align with that.
        X_grid, Y_grid = np.meshgrid(coords[d2], coords[d1])

        # Plot
        im = ax.pcolormesh(X_grid, Y_grid, proj_2d, cmap='inferno', shading='auto')

        # Labels
        ax.set_title(f"{dim_names[d1]} vs {dim_names[d2]} Projection")
        ax.set_xlabel(dim_names[d2])
        ax.set_ylabel(dim_names[d1])
        fig.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.show()


def hs_data(filename: str):
    data = np.load(filename)
    t = data['t']
    x = data['x']
    # y = data['y']
    # z = data['z']
    data = data['u'].squeeze().T
    # data = np.transpose(data['u'].squeeze(), axes=(3, 0, 1, 2))
    # grids = np.meshgrid(x, y, z, t, indexing = 'ij')
    grids = np.meshgrid(t, x, indexing='ij')

    # plot_all_projections(data, [x, y, z, t], dim_names=["x", "y", "z", "t"])
    # plot_all_projections(data, [t, x, y, z], dim_names=["t", "x", "y", "z"])

    return grids, data

def hs_data_2d(filename: str):
    data = np.load(filename)
    t = data['t']
    x = data['x']
    y = data['y']
    # z = data['z']
    # data = data['u'].squeeze().T
    # t_size = 20
    # t = t[:t_size]
    data = np.transpose(data['u'].squeeze(), axes=(2, 0, 1))
    # grids = np.meshgrid(x, y, z, t, indexing = 'ij')
    grids = np.meshgrid(t, x, y, indexing='ij')

    # plot_all_projections(data, [x, y, z, t], dim_names=["x", "y", "z", "t"])
    # plot_all_projections(data, [t, x, y, z], dim_names=["t", "x", "y", "z"])

    return grids, data

def hs_data_3d(filename: str):
    data = np.load(filename)
    t = data['t']
    x = data['x']
    y = data['y']
    z = data['z']
    # data = data['u'].squeeze().T
    # t_size = 20
    # t = t[:t_size]
    data = np.transpose(data['u'].squeeze(), axes=(3, 0, 1, 2))
    # grids = np.meshgrid(x, y, z, t, indexing = 'ij')
    grids = np.meshgrid(t, x, y, z, indexing='ij')

    # plot_all_projections(data, [x, y, z, t], dim_names=["x", "y", "z", "t"])
    # plot_all_projections(data, [t, x, y, z], dim_names=["t", "x", "y", "z"])

    return grids, data


def hs_discovery(foldername, noise_level):
    grid, data = hs_data(os.path.join(foldername, 'heat_soil_uniform_1d_p1.npz'))
    noised_data = noise_data(data, noise_level)
    data_nn = load_pretrained_PINN(os.path.join(foldername, f'kdv_{noise_level}_ann.pickle'))

    dimensionality = data.ndim - 1

    epde_search_obj = EpdeSearch(use_solver=False, multiobjective_mode=True,
                                      use_pic=True, boundary=(1,1),
                                      coordinate_tensors=grid, device='cuda')

    # epde_search_obj.set_preprocessor(default_preprocessor_type='ANN',
    #                                     preprocessor_kwargs={'epochs_max' : 1e3})
    epde_search_obj.set_preprocessor(default_preprocessor_type='FD',
                                     preprocessor_kwargs={})
    popsize = 16

    epde_search_obj.set_moeadd_params(population_size=popsize,
                                      training_epochs=1)

    trig_params_ranges = {'power': (1, 1)}
    trig_params_equal_ranges = {}

    trig_tokens = TrigonometricTokens(dimensionality=dimensionality, freq = (0.999, 1.001))

    factors_max_number = {'factors_num': [1, 2], 'probas': [0.8, 0.2]}

    bounds = (1e-12, 1e-0)
    epde_search_obj.fit(data=noised_data, variable_names=['u', ], max_deriv_order=(2, 2), derivs=None,
                        equation_terms_max_number=10, data_fun_pow=1,
                        additional_tokens=[],
                        equation_factors_max_number=factors_max_number,
                        eq_sparsity_interval=bounds, fourier_layers=False) #, data_nn=data_nn

    epde_search_obj.equations(only_print=True, num=1)
    epde_search_obj.visualize_solutions()

    return epde_search_obj

def hs_2d_discovery(foldername, noise_level):
    grid, data = hs_data_2d(os.path.join(foldername, 'heat_soil_uniform_2d_p1.npz'))
    noised_data = noise_data(data, noise_level)
    data_nn = load_pretrained_PINN(os.path.join(foldername, f'kdv_{noise_level}_ann.pickle'))

    dimensionality = data.ndim - 1

    epde_search_obj = EpdeSearch(use_solver=False, multiobjective_mode=True,
                                      use_pic=True, boundary=(100,5,5),
                                      coordinate_tensors=grid, device='cuda')

    # epde_search_obj.set_preprocessor(default_preprocessor_type='ANN',
    #                                     preprocessor_kwargs={'epochs_max' : 1e3})
    epde_search_obj.set_preprocessor(default_preprocessor_type='FD',
                                     preprocessor_kwargs={})
    popsize = 16

    epde_search_obj.set_moeadd_params(population_size=popsize,
                                      training_epochs=5)

    # def laser_f(t, x, y):
    #     return 3e6 * np.exp(-50000 * (np.pow(x - 0.5 * 0.1 * (1 + 0.5 * np.sin(2 * math.pi * t / 5)), 2) + np.pow(y - 0.02 * t, 2)))
    #
    # laser = laser_f(grid[-1], grid[0], grid[1])
    #
    # custom_laser_tokens = CacheStoredTokens(token_type='laser',
    #                                             token_labels=['L'],
    #                                             token_tensors={'L': laser},
    #                                             params_ranges={'power': (1, 1)},
    #                                             params_equality_ranges=None, meaningful=True)

    trig_params_ranges = {'power': (1, 1)}
    trig_params_equal_ranges = {}

    trig_tokens = TrigonometricTokens(dimensionality=dimensionality, freq = (0.999, 1.001))

    factors_max_number = {'factors_num': [1, 2], 'probas': [0.8, 0.2]}

    bounds = (1e-12, 1e-0)
    epde_search_obj.fit(data=noised_data, variable_names=['u', ], max_deriv_order=(2, 2, 2), derivs=None,
                        equation_terms_max_number=10, data_fun_pow=1,
                        additional_tokens=[],
                        equation_factors_max_number=factors_max_number,
                        eq_sparsity_interval=bounds, fourier_layers=False) #, data_nn=data_nn

    epde_search_obj.equations(only_print=True, num=1)
    epde_search_obj.visualize_solutions()

    return epde_search_obj

def hs_3d_discovery(foldername, noise_level):
    grid, data = hs_data_3d(os.path.join(foldername, 'heat_soil_uniform_3d_p1.npz'))
    noised_data = noise_data(data, noise_level)
    data_nn = load_pretrained_PINN(os.path.join(foldername, f'kdv_{noise_level}_ann.pickle'))

    dimensionality = data.ndim - 1

    epde_search_obj = EpdeSearch(use_solver=False, multiobjective_mode=True,
                                      use_pic=True, boundary=(2,5,5,1),
                                      coordinate_tensors=grid, device='cuda')

    # epde_search_obj.set_preprocessor(default_preprocessor_type='ANN',
    #                                     preprocessor_kwargs={'epochs_max' : 1e3})
    epde_search_obj.set_preprocessor(default_preprocessor_type='FD',
                                     preprocessor_kwargs={})
    popsize = 16

    epde_search_obj.set_moeadd_params(population_size=popsize,
                                      training_epochs=1)

    trig_params_ranges = {'power': (1, 1)}
    trig_params_equal_ranges = {}

    trig_tokens = TrigonometricTokens(dimensionality=dimensionality, freq = (0.999, 1.001))

    factors_max_number = {'factors_num': [1, 2], 'probas': [0.8, 0.2]}

    bounds = (1e-12, 1e-0)
    epde_search_obj.fit(data=noised_data, variable_names=['u', ], max_deriv_order=(2, 2, 2, 2), derivs=None,
                        equation_terms_max_number=10, data_fun_pow=1,
                        additional_tokens=[],
                        equation_factors_max_number=factors_max_number,
                        eq_sparsity_interval=bounds, fourier_layers=False) #, data_nn=data_nn

    epde_search_obj.equations(only_print=True, num=1)
    epde_search_obj.visualize_solutions()

    return epde_search_obj


if __name__ == "__main__":
    import torch
    print(torch.cuda.is_available())

    # Paths
    directory = os.path.dirname(os.path.realpath(__file__))
    ac_folder_name = os.path.join(directory)

    # hs_discovery(ac_folder_name, 0)
    hs_2d_discovery(ac_folder_name, 0)
    # hs_3d_discovery(ac_folder_name, 0)

