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
    print(correct_eq.text_form)

    incorrect_eq = translate_equation(eq_incorrect_symbolic, search_obj.pool,
                                      all_vars=all_vars)  # , all_vars = ['u', 'v'])
    for var in all_vars:
        incorrect_eq.vals[var].main_var_to_explain = var
        incorrect_eq.vals[var].metaparameters = metaparams
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


def darcy_data(filename: str):
    t = np.linspace(0., 1., 2)
    x = np.linspace(0., 1., 128)
    y = np.linspace(0., 1., 128)
    grids = np.meshgrid(t, x, y, indexing='ij')  # np.stack(, axis = 2)
    data = np.load(filename)
    data = data[0]
    data = np.stack([data]*2, axis=0)
    print(data.shape)
    return grids, data

def darcy_test(operator: CompoundOperator, foldername: str, noise_level: int = 0):
    # Test scenario to evaluate performance on darcy equation
    # eq_darcy_symbolic = '-1.0 * du/dx1{power: 1.0} * dnu/dy{power: 1.0} + -1.0 * nu{power: 1.0} * d^2u/dx1^2{power: 1} + \
    #                       -2.0 * nu{power: 1.0} * d^2u/dxdy{power: 1} + \
    #                       -1.0 * du/dx2{power: 1.0} * dnu/dx{power: 1.0} + -1.0 * du/dx2{power: 1.0} * dnu/dy{power: 1.0} + \
    #                       -1.0 * nu{power: 1.0} * d^2u/dx2^2{power: 1} + -1.0 = du/dx1{power: 1.0} * dnu/dx{power: 1.0}'

    eq_darcy_symbolic = '-1.0 * du/dx2{power: 1.0} * dnu/dy{power: 1.0} + -1.0 * nu{power: 1.0} * d^2u/dx1^2{power: 1} + \
                              -1.0 * nu{power: 1.0} * d^2u/dx2^2{power: 1} + -1.0 = du/dx1{power: 1.0} * dnu/dx{power: 1.0}'

    eq_darcy_incorrect = '0.04 * d^2u/dx1^2{power: 1} + 0. = d^2u/dx2^2{power: 1}'

    grid, data = darcy_data(os.path.join(foldername, 'darcy_1.0.npy'))
    nu = np.load('darcy_nu_1.0.npy')
    noised_data = noise_data(data, noise_level)
    x = np.linspace(0., 1., 128)
    y = np.linspace(0., 1., 128)
    dx = x[1] - x[0]
    dy = y[1] - y[0]

    dimensionality = 1

    epde_search_obj = EpdeSearch(use_solver=False, multiobjective_mode=True, use_pic=True, boundary=0,
                                 coordinate_tensors=grid, device='cuda')

    # epde_search_obj.set_preprocessor(default_preprocessor_type='ANN',
    #                                     preprocessor_kwargs={'epochs_max' : 1e3})
    epde_search_obj.set_preprocessor(default_preprocessor_type='FD',
                                     preprocessor_kwargs={})

    darcy_gradient_nux = np.gradient(nu[0], dx, axis=0, edge_order=2)
    darcy_gradient_nuy = np.gradient(nu[0], dy, axis=1, edge_order=2)
    darcy_gradient_xy = np.gradient(np.gradient(data, dx, axis=1, edge_order=2), dy, axis=2, edge_order=2)

    custom_grid_tokens_nu = CacheStoredTokens(token_type='nu-tensors',
                                              token_labels=['nu', 'dnu/dx', 'dnu/dy'],
                                              token_tensors={'nu': nu[0], 'dnu/dx': darcy_gradient_nux,
                                                             'dnu/dy': darcy_gradient_nuy},
                                              params_ranges={'power': (1, 1)},
                                              params_equality_ranges=None, )

    custom_grid_tokens_xy = CacheStoredTokens(token_type='xy-tensor',
                                              token_labels=['d^2u/dxdy'],
                                              token_tensors={'d^2u/dxdy': darcy_gradient_xy},
                                              params_ranges={'power': (1, 1)},
                                              params_equality_ranges=None,
                                              meaningful=True
                                              )

    epde_search_obj.create_pool(data=noised_data, variable_names=['u',], max_deriv_order=(0, 2, 2),
                                additional_tokens = [custom_grid_tokens_nu, custom_grid_tokens_xy]) # , data_nn = data_nn

    # np.save(os.path.join(foldername, 'kdv_0_derivs.npy'), epde_search_obj.derivatives)

    assert compare_equations(eq_darcy_symbolic, eq_darcy_incorrect, epde_search_obj)


def darcy_discovery(foldername, noise_level):
    grid, data = darcy_data(os.path.join(foldername, 'darcy_1.0.npy'))
    nu = np.load('darcy_nu_1.0.npy')
    noised_data = noise_data(data, noise_level)
    x = np.linspace(0., 1., 128)
    y = np.linspace(0., 1., 128)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    # data_nn = load_pretrained_PINN(os.path.join(foldername, 'ann_pretrained.pickle'))

    dimensionality = data.ndim - 1

    epde_search_obj = EpdeSearch(use_solver=False, use_pic=True, boundary=0,
                                      coordinate_tensors=grid, device='cuda')

    # epde_search_obj.set_preprocessor(default_preprocessor_type='ANN',
    #                                     preprocessor_kwargs={'epochs_max' : 1e3})
    epde_search_obj.set_preprocessor(default_preprocessor_type='FD',
                                     preprocessor_kwargs={})
    popsize = 8

    epde_search_obj.set_moeadd_params(population_size=popsize,
                                      training_epochs=30)

    darcy_gradient_nux = np.gradient(nu[0], dx, axis=0, edge_order=2)
    darcy_gradient_nuy = np.gradient(nu[0], dy, axis=1, edge_order=2)
    darcy_gradient_xy = np.gradient(np.gradient(data, dx, axis=1, edge_order=2), dy, axis=2, edge_order=2)

    custom_grid_tokens_nu = CacheStoredTokens(token_type='nu-tensors',
                                                token_labels=['nu', 'dnu/dx', 'dnu/dy'],
                                                token_tensors={'nu': nu[0], 'dnu/dx': darcy_gradient_nux, 'dnu/dy': darcy_gradient_nuy},
                                                params_ranges={'power': (1, 1)},
                                                params_equality_ranges=None,)

    custom_grid_tokens_xy = CacheStoredTokens(token_type='xy-tensor',
                                               token_labels=['d^2u/dxdy'],
                                               token_tensors={'d^2u/dxdy': darcy_gradient_xy},
                                               params_ranges={'power': (1, 1)},
                                               params_equality_ranges=None,
                                               meaningful=True
                                              )

    trig_params_ranges = {'power': (1, 1)}
    trig_params_equal_ranges = {}

    trig_tokens = TrigonometricTokens(dimensionality=dimensionality, freq = (0.999, 1.001))

    factors_max_number = {'factors_num': [1, 2], 'probas': [0.5, 0.5]}

    bounds = (1e-12, 1e-2)
    epde_search_obj.fit(data=noised_data, variable_names=['u', ], max_deriv_order=(0, 2, 2), derivs=None,
                        equation_terms_max_number=5, data_fun_pow=1,
                        additional_tokens=[custom_grid_tokens_nu, custom_grid_tokens_xy],
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
    darcy_folder_name = os.path.join(directory)

    # Pair-wise tests
    darcy_test(fit_operator, darcy_folder_name, 0)

    # Full_scale test
    darcy_discovery(darcy_folder_name, 0)