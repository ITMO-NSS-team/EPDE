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

from epde.operators.common.coeff_calculation import LinRegBasedCoeffsEquation
from epde.operators.common.sparsity import LASSOSparsity

from epde.operators.utils.operator_mappers import map_operator_between_levels
import epde.operators.common.fitness as fitness
from epde.operators.utils.template import CompoundOperator

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


def compare_equations(correct_symbolic: str, eq_incorrect_symbolic: str,
                      search_obj: EpdeSearch, all_vars: List[str] = ['u', ]) -> bool:
    metaparams = {('sparsity', var): {'optimizable': False, 'value': 1E-6} for var in all_vars}

    correct_eq = translate_equation(correct_symbolic, search_obj.pool, all_vars=all_vars)
    for var in all_vars:
        correct_eq.vals[var].main_var_to_explain = var
        correct_eq.vals[var].metaparameters = metaparams
        correct_eq.vals[var].weights_internal = np.ones(len(correct_eq.vals[var].structure) - 1)
        correct_eq.vals[var].weights_internal_evald = True
    print(correct_eq.text_form)

    incorrect_eq = translate_equation(eq_incorrect_symbolic, search_obj.pool,
                                      all_vars=all_vars)  # , all_vars = ['u', 'v'])
    for var in all_vars:
        incorrect_eq.vals[var].main_var_to_explain = var
        incorrect_eq.vals[var].metaparameters = metaparams
        incorrect_eq.vals[var].weights_internal = np.ones(len(incorrect_eq.vals[var].structure) - 1)
        incorrect_eq.vals[var].weights_internal_evald = True
    print(incorrect_eq.text_form)

    fit_operator.apply(correct_eq, {})
    fit_operator.apply(incorrect_eq, {})
    print([[correct_eq.vals[var].fitness_value, incorrect_eq.vals[var].fitness_value] for var in all_vars])
    print([[correct_eq.vals[var].coefficients_stability, incorrect_eq.vals[var].coefficients_stability] for var in
           all_vars])
    print([[correct_eq.vals[var].aic, incorrect_eq.vals[var].aic] for var in all_vars])

    # print([correct_eq.vals[var].coefficients_stability < incorrect_eq.vals[var].coefficients_stability for var in all_vars])
    return all([correct_eq.vals[var].coefficients_stability < incorrect_eq.vals[var].coefficients_stability for var in
                all_vars])


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


def prepare_suboperators(fitness_operator: CompoundOperator, operator_params: dict) -> CompoundOperator:
    sparsity = LASSOSparsity()
    coeff_calc = LinRegBasedCoeffsEquation()

    # sparsity = map_operator_between_levels(sparsity, 'gene level', 'chromosome level')
    # coeff_calc = map_operator_between_levels(coeff_calc, 'gene level', 'chromosome level')

    fitness_operator.set_suboperators({'sparsity': sparsity,
                                       'coeff_calc': coeff_calc})
    fitness_cond = lambda x: not getattr(x, 'fitness_calculated')
    fitness_operator.params = operator_params
    fitness_operator = map_operator_between_levels(fitness_operator, 'gene level', 'chromosome level',
                                                   objective_condition=fitness_cond)
    return fitness_operator

def hl_data(filename: str):
    data = np.load(filename)
    t = data['t']
    x = data['x']
    y = data['y']
    z = data['z']
    # data = data['u'].squeeze()
    data = np.transpose(data['u'].squeeze(), axes=(3, 0, 1, 2))
    # grids = np.meshgrid(x, y, z, t, indexing = 'ij')
    grids = np.meshgrid(t, x, y, z, indexing='ij')

    # plot_all_projections(data, [x, y, z, t], dim_names=["x", "y", "z", "t"])
    # plot_all_projections(data, [t, x, y, z], dim_names=["t", "x", "y", "z"])

    return grids, data


def hl_test(operator: CompoundOperator, foldername: str, noise_level: int = 0):
    # Test scenario to evaluate performance on Allen-Cahn equation
    eq_ac_symbolic = '0.0001 * d^2u/dx0^2{power: 1.0} + -5.0 * d^2u/dx1^2{power: 1.0} + 5.0 * d^2u/dx2^2{power: 1.0} + 5.0 * L{power: 1.0} + 0.0 = du/dx3{power: 1.0}'
    eq_ac_incorrect = '0.0001 * d^2u/dx1^2{power: 1.0} + -5.0 * d^2u/dx2^2{power: 1.0} + 5.0 * d^2u/dx3^2{power: 1.0} + 5.0 * L{power: 1.0} + 0.0 = du/dx0{power: 1.0}'

    grid, data = hl_data(os.path.join(foldername, 'heat_laser.npz'))
    noised_data = noise_data(data, noise_level)
    data_nn = load_pretrained_PINN(os.path.join(foldername, 'ac_ann_pretrained.pickle'))

    print('Shapes:', data.shape, grid[0].shape)
    dimensionality = 1

    epde_search_obj = EpdeSearch(use_solver=False, use_pic=True, boundary=(0, 0, 0, 0),
                                 coordinate_tensors=grid, verbose_params={'show_iter_idx': True},
                                 device='cuda')

    epde_search_obj.set_preprocessor(default_preprocessor_type='FD',
                                     preprocessor_kwargs={})

    def laser_f(t, x, y):
        return 3e6 * np.exp(-50000 * (np.pow(x - 0.5 * 0.1 * (1 + 0.5 * np.sin(2 * math.pi * t / 5)), 2) + np.pow(y - 0.02 * t, 2)))

    # laser = laser_f(grid[-1], grid[0], grid[1])
    laser = laser_f(grid[0], grid[1], grid[2])
    # plot_all_projections(laser, [np.unique(grid[0]), np.unique(grid[1]), np.unique(grid[2]), np.unique(grid[3])], dim_names=["x", "y", "z", "t"])
    plot_all_projections(laser, [np.unique(grid[0]), np.unique(grid[1]), np.unique(grid[2]), np.unique(grid[3])], dim_names=["t", "x", "y", "z"])


    custom_laser_tokens = CacheStoredTokens(token_type='laser',
                                                token_labels=['L'],
                                                token_tensors={'L': laser},
                                                params_ranges={'power': (1, 1)},
                                                params_equality_ranges=None, meaningful=True)

    epde_search_obj.create_pool(data=noised_data, variable_names=['u', ], max_deriv_order=(2, 2, 2, 2),
                                additional_tokens=[custom_laser_tokens]) #, data_nn=data_nn

    assert compare_equations(eq_ac_symbolic, eq_ac_incorrect, epde_search_obj)


def hl_discovery(foldername, noise_level):
    grid, data = hl_data(os.path.join(foldername, 'heat_laser.npz'))
    noised_data = noise_data(data, noise_level)
    data_nn = load_pretrained_PINN(os.path.join(foldername, f'kdv_{noise_level}_ann.pickle'))

    dimensionality = data.ndim - 1

    epde_search_obj = EpdeSearch(use_solver=False, multiobjective_mode=True,
                                      use_pic=True, boundary=(0,0,0,0),
                                      coordinate_tensors=grid, device='cuda')

    # epde_search_obj.set_preprocessor(default_preprocessor_type='ANN',
    #                                     preprocessor_kwargs={'epochs_max' : 1e3})
    epde_search_obj.set_preprocessor(default_preprocessor_type='FD',
                                     preprocessor_kwargs={})
    popsize = 16

    epde_search_obj.set_moeadd_params(population_size=popsize,
                                      training_epochs=5)

    def laser_f(t, x, y):
        return 3e6 * np.exp(-50000 * (np.pow(x - 0.5 * 0.1 * (1 + 0.5 * np.sin(2 * math.pi * t / 5)), 2) + np.pow(y - 0.02 * t, 2)))

    laser = laser_f(grid[-1], grid[0], grid[1])

    custom_laser_tokens = CacheStoredTokens(token_type='laser',
                                                token_labels=['L'],
                                                token_tensors={'L': laser},
                                                params_ranges={'power': (1, 1)},
                                                params_equality_ranges=None, meaningful=True)

    trig_params_ranges = {'power': (1, 1)}
    trig_params_equal_ranges = {}

    trig_tokens = TrigonometricTokens(dimensionality=dimensionality, freq = (0.999, 1.001))

    factors_max_number = {'factors_num': [1, 2], 'probas': [0.8, 0.2]}

    bounds = (1e-12, 1e-0)
    epde_search_obj.fit(data=noised_data, variable_names=['u', ], max_deriv_order=(2, 2, 2, 2), derivs=None,
                        equation_terms_max_number=10, data_fun_pow=1,
                        additional_tokens=[custom_laser_tokens],
                        equation_factors_max_number=factors_max_number,
                        eq_sparsity_interval=bounds, fourier_layers=False) #, data_nn=data_nn

    epde_search_obj.equations(only_print=True, num=1)
    epde_search_obj.visualize_solutions()

    return epde_search_obj


if __name__ == "__main__":
    import torch
    from epde.operators.utils.default_parameter_loader import EvolutionaryParams
    print(torch.cuda.is_available())
    # Operator = fitness.SolverBasedFitness # Replace by the developed PIC-based operator.
    # Operator = fitness.PIC
    Operator = fitness.L2LRFitness
    params = EvolutionaryParams()
    operator_params = params.get_default_params_for_operator('DiscrepancyBasedFitnessWithCV') #{"penalty_coeff": 0.2, "pinn_loss_mult": 1e4}
    print('operator_params ', operator_params)
    fit_operator = prepare_suboperators(Operator(list(operator_params.keys())), operator_params)

    # Paths
    directory = os.path.dirname(os.path.realpath(__file__))
    ac_folder_name = os.path.join(directory)

    hl_test(fit_operator, ac_folder_name, 0)
    # hl_discovery(ac_folder_name, 0)
