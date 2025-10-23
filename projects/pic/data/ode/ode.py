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
        correct_eq.vals[var].simplified = True
    print(correct_eq.text_form)

    incorrect_eq = translate_equation(eq_incorrect_symbolic, search_obj.pool,
                                      all_vars=all_vars)  # , all_vars = ['u', 'v'])
    for var in all_vars:
        incorrect_eq.vals[var].main_var_to_explain = var
        incorrect_eq.vals[var].metaparameters = metaparams
        incorrect_eq.vals[var].simplified = True
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


def ODE_test(operator: CompoundOperator, foldername: str, noise_level: int = 0):
    # Test scenario to evaluate performance on simple 2nd order ODE
    # x'' + sin(2t) x' + 4 x = 1.5 t,  written as $g_{1} x'' + g_{2} x' + g_{3} x = g_{4}
    # g1 = lambda x: 1.
    # g2 = lambda x: np.sin(2*x)
    # g3 = lambda x: 4.
    # g4 = lambda x: 1.5*x

    eq_ode_symbolic = '-3.9920083373920305 * u{power: 1.0} + -0.986615483876419 * du/dx0{power: 1.0} * sin{power: 1.0, freq: 2.0000000038280255, dim: 0.0} + 1.497626641591792 * x_0{power: 1.0, dim: 0.0} + -0.03054107154984344 = d^2u/dx0^2{power: 1.0}'
    eq_ode_incorrect = '1.0 * du/dx0{power: 1.0} + 3.5 * x_0{power: 1.0, dim: 0.0} * u{power: 1.0} + -1.2 \
                        = du/dx0{power: 1.0} * sin{power: 1.0, freq: 2.0, dim: 0.0}'

    step = 0.05;
    steps_num = 320
    t = np.arange(start=0., stop=step * steps_num, step=step)
    data = np.load(os.path.join(foldername, 'ode_data.npy'))
    noised_data = noise_data(data, noise_level)
    data_nn = load_pretrained_PINN(os.path.join(foldername, 'ode_0_ann.pickle')).cuda()

    dimensionality = 0

    trig_tokens = TrigonometricTokens(freq=(2 - 1e-8, 2 + 1e-8),
                                      dimensionality=dimensionality)
    grid_tokens = GridTokens(['x_0', ], dimensionality=dimensionality, max_power=2)

    epde_search_obj = EpdeSearch(use_solver=False, use_pic=True, boundary=10,
                                 coordinate_tensors=[t,], verbose_params={'show_iter_idx': True},
                                 device='cuda')

    epde_search_obj.set_preprocessor(default_preprocessor_type='FD',
                                     preprocessor_kwargs={})

    epde_search_obj.create_pool(data=noised_data, variable_names=['u', ], max_deriv_order=(2,),
                                additional_tokens=[grid_tokens, trig_tokens], data_nn=data_nn)

    assert compare_equations(eq_ode_symbolic, eq_ode_incorrect, epde_search_obj)


def ODE_discovery(foldername, noise_level):
    step = 0.05
    steps_num = 320
    t = np.arange(start=0., stop=step * steps_num, step=step)
    data = np.load(os.path.join(foldername, 'ode_data.npy'))
    noised_data = noise_data(data, noise_level)
    data_nn = load_pretrained_PINN(os.path.join(foldername, 'ode_0_ann.pickle')).cpu()

    dimensionality = 0

    trig_tokens = TrigonometricTokens(freq=(2 - 1e-8, 2 + 1e-8),
                                      dimensionality=dimensionality)
    grid_tokens = GridTokens(['x_0', ], dimensionality=dimensionality, max_power=2)

    epde_search_obj = EpdeSearch(use_solver=False, use_pic=True, boundary=20,
                                 coordinate_tensors=[t,], verbose_params={'show_iter_idx': True},
                                 device='cuda')

    epde_search_obj.set_preprocessor(default_preprocessor_type='FD',
                                     preprocessor_kwargs={})

    popsize = 16
    epde_search_obj.set_moeadd_params(population_size=popsize, training_epochs=15)

    factors_max_number = {'factors_num': [1, 2], 'probas': [0.65, 0.35]}

    epde_search_obj.fit(data=[noised_data, ], variable_names=['u', ], max_deriv_order=(2, 3),
                        equation_terms_max_number=5, data_fun_pow=3,
                        additional_tokens=[trig_tokens, grid_tokens],
                        equation_factors_max_number=factors_max_number,
                        eq_sparsity_interval=(1e-8, 1e-0)) # , data_nn=data_nn

    epde_search_obj.equations(only_print=True, num=1)

    # import pickle
    # fname = os.path.join(r'C:\Users\user\EPDE_tests\models', 'ode_0_ann.pickle')
    # with open(fname, 'wb') as output_file:
    #     pickle.dump(global_var.solution_guess_nn, output_file)
    epde_search_obj.visualize_solutions()
    return epde_search_obj

def ODE_simple_discovery(foldername, noise_level):
    C = 1.3
    t = np.linspace(0, 4 * np.pi, 200)
    x = np.sin(t) + C * np.cos(t)
    x_dot = np.cos(t) - C * np.sin(t)
    x_dot_dot = -np.sin(t) - C * np.cos(t)
    x_dot_dot_dot = -np.cos(t) + C * np.sin(t)

    dimensionality = x.ndim - 1

    trig_tokens = TrigonometricTokens(freq = (0.999, 1.001),
                                      dimensionality=dimensionality)
    grid_tokens = GridTokens(['x_0', ], dimensionality=dimensionality, max_power=2)

    epde_search_obj = EpdeSearch(use_solver=False, use_pic=True, boundary=10,
                                 coordinate_tensors=[t,], verbose_params={'show_iter_idx': True},
                                 device='cuda')

    epde_search_obj.set_preprocessor(default_preprocessor_type='FD',
                                     preprocessor_kwargs={})

    popsize = 8
    epde_search_obj.set_moeadd_params(population_size=popsize, training_epochs=10)

    factors_max_number = {'factors_num': [1, 2], 'probas': [0.65, 0.35]}

    epde_search_obj.fit(data=[x, ], variable_names=['u', ], max_deriv_order=(2, 3),
                        equation_terms_max_number=5, data_fun_pow=3,
                        additional_tokens=[trig_tokens, grid_tokens],
                        equation_factors_max_number=factors_max_number,
                        eq_sparsity_interval=(1e-4, 1e-0)) #

    epde_search_obj.equations(only_print=True, num=1)

    # import pickle
    # fname = os.path.join(r'C:\Users\user\EPDE_tests\models', 'ode_0_ann.pickle')
    # with open(fname, 'wb') as output_file:
    #     pickle.dump(global_var.solution_guess_nn, output_file)
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
    ode_folder_name = os.path.join(directory)

    # ODE_test(fit_operator, ode_folder_name, 0)
    ODE_discovery(ode_folder_name, 0)
    # ODE_simple_discovery(ode_folder_name, 0)

