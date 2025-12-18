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


def ns_test(operator: CompoundOperator, foldername: str, noise_level: int = 0):
    # Test scenario to evaluate performance on Allen-Cahn equation
    eq_ac_symbolic = '0.0001 * d^2u/dx1^2{power: 1.0} + -5.0 * u{power: 3.0} + 5.0 * u{power: 1.0} + 0.0 = du/dx0{power: 1.0}'
    eq_ac_incorrect = '4.976781518840499 * u{power: 1.0} + 0.0001 * d^2u/dx1^2{power: 1.0} + -4.974425220166616 * u{power: 3.0} + 0.0 * du/dx1{power: 1.0} * d^2u/dx0^2{power: 1.0} + 0.002262543822130977 = du/dx0{power: 1.0}'

    grid, data = ns_data(os.path.join(foldername, 'cylinder_nektar_wake.mat'))
    # noised_data = noise_data(data, noise_level)
    # data_nn = load_pretrained_PINN(os.path.join(foldername, 'ac_ann_pretrained.pickle'))

    # print('Shapes:', data.shape, grid[0].shape)
    dimensionality = 1

    epde_search_obj = EpdeSearch(use_solver=False, use_pic=True, boundary=10,
                                 coordinate_tensors=grid, verbose_params={'show_iter_idx': True},
                                 device='cpu')

    epde_search_obj.set_preprocessor(default_preprocessor_type='FD',
                                     preprocessor_kwargs={})

    epde_search_obj.create_pool(data=data, variable_names=["u", "v", "p"], max_deriv_order=(1, 2, 2),
                                additional_tokens=[])#, data_nn=data_nn

    assert compare_equations([eq_ac_symbolic] * 3, [eq_ac_incorrect] * 3, epde_search_obj)


def ns_discovery(foldername, noise_level):
    grid, data = ns_data(os.path.join(foldername, 'cylinder_nektar_wake.mat'))
    # noised_data = noise_data(data, noise_level)
    data_nn = load_pretrained_PINN(os.path.join(foldername, f'kdv_{noise_level}_ann.pickle'))

    # dimensionality = data.ndim - 1

    epde_search_obj = EpdeSearch(use_solver=False, multiobjective_mode=True,
                                      use_pic=True, boundary=5,
                                      coordinate_tensors=grid, device='cuda')

    # epde_search_obj.set_preprocessor(default_preprocessor_type='ANN',
    #                                     preprocessor_kwargs={'epochs_max' : 1e3})
    epde_search_obj.set_preprocessor(default_preprocessor_type='FD',
                                     preprocessor_kwargs={})
    popsize = 16

    epde_search_obj.set_moeadd_params(population_size=popsize,
                                      training_epochs=50)

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
                        equation_terms_max_number=10, data_fun_pow=1,
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
    # Operator = fitness.SolverBasedFitness # Replace by the developed PIC-based operator.
    # Operator = fitness.PIC
    Operator = fitness.L2LRFitness
    params = EvolutionaryParams()
    operator_params = params.get_default_params_for_operator('DiscrepancyBasedFitnessWithCV') #{"penalty_coeff": 0.2, "pinn_loss_mult": 1e4}
    print('operator_params ', operator_params)
    fit_operator = prepare_suboperators(Operator(list(operator_params.keys())), operator_params)

    # Paths
    directory = os.path.dirname(os.path.realpath(__file__))
    ns_folder_name = os.path.join(directory)

    # ns_test(fit_operator, ns_folder_name, 0)
    ns_discovery(ns_folder_name, 0)
