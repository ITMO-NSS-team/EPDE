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

# Introduce noise levels, test with complex setups
# np.random.seed(0)

def load_pretrained_PINN(ann_filename):
    try:
        with open(ann_filename, 'rb') as data_input_file:  
            data_nn = pickle.load(data_input_file)
    except FileNotFoundError:
        print('No model located, proceeding with ann approx. retraining.')
        data_nn = None
    return data_nn


def compare_equations(correct_symbolic: str, eq_incorrect_symbolic: str, 
                      search_obj: EpdeSearch, all_vars: List[str] = ['u',]) -> bool:
    metaparams = {('sparsity', var): {'optimizable': False, 'value': 1E-6} for var in all_vars}

    correct_eq = translate_equation(correct_symbolic, search_obj.pool, all_vars = all_vars)
    for var in all_vars:
        correct_eq.vals[var].main_var_to_explain = var
        correct_eq.vals[var].metaparameters = metaparams
    print(correct_eq.text_form)
 
    incorrect_eq = translate_equation(eq_incorrect_symbolic, search_obj.pool, all_vars = all_vars)   #  , all_vars = ['u', 'v'])
    for var in all_vars:
        incorrect_eq.vals[var].main_var_to_explain = var
        incorrect_eq.vals[var].metaparameters = metaparams
    print(incorrect_eq.text_form)

    fit_operator.apply(correct_eq, {})
    fit_operator.apply(incorrect_eq, {})
    print([[correct_eq.vals[var].fitness_value, incorrect_eq.vals[var].fitness_value] for var in all_vars])
    print([[correct_eq.vals[var].coefficients_stability, incorrect_eq.vals[var].coefficients_stability] for var in all_vars])
    # print([correct_eq.vals[var].coefficients_stability < incorrect_eq.vals[var].coefficients_stability for var in all_vars])
    return all([correct_eq.vals[var].coefficients_stability < incorrect_eq.vals[var].coefficients_stability for var in all_vars])


def prepare_suboperators(fitness_operator: CompoundOperator, operator_params: dict) -> CompoundOperator:
    sparsity = LASSOSparsity()
    coeff_calc = LinRegBasedCoeffsEquation()

    # sparsity = map_operator_between_levels(sparsity, 'gene level', 'chromosome level')
    # coeff_calc = map_operator_between_levels(coeff_calc, 'gene level', 'chromosome level')

    fitness_operator.set_suboperators({'sparsity' : sparsity,
                                       'coeff_calc' : coeff_calc})
    fitness_cond = lambda x: not getattr(x, 'fitness_calculated')
    fitness_operator.params = operator_params
    fitness_operator = map_operator_between_levels(fitness_operator, 'gene level', 'chromosome level',
                                                  objective_condition=fitness_cond)
    return fitness_operator


def noise_data(data, noise_level):
    # add noise level to the input data
    return noise_level * 0.01 * np.std(data) * np.random.normal(size=data.shape) + data


def ODE_test(operator: CompoundOperator, foldername: str, noise_level: int = 0):
    # Test scenario to evaluate performance on simple 2nd order ODE
    # x'' + sin(2t) x' + 4 x = 1.5 t,  written as $g_{1} x'' + g_{2} x' + g_{3} x = g_{4}
    # g1 = lambda x: 1.
    # g2 = lambda x: np.sin(2*x)
    # g3 = lambda x: 4.
    # g4 = lambda x: 1.5*x
    
    eq_ode_symbolic = '-1.0 * d^2u/dx0^2{power: 1.0} + 1.5 * x_0{power: 1.0, dim: 0.0} + -4.0 * u{power: 1.0} + -0.0 \
                       = du/dx0{power: 1.0} * sin{power: 1.0, freq: 2.0, dim: 0.0}'
    eq_ode_incorrect = '1.0 * du/dx0{power: 1.0} + 3.5 * x_0{power: 1.0, dim: 0.0} * u{power: 1.0} + -1.2 \
                        = du/dx0{power: 1.0} * sin{power: 1.0, freq: 2.0, dim: 0.0}'

    step = 0.05; steps_num = 320
    t = np.arange(start = 0., stop = step * steps_num, step = step)
    data = np.load(os.path.join(foldername, 'ode_data.npy'))
    noised_data = noise_data(data, noise_level)
    data_nn = load_pretrained_PINN(os.path.join(foldername, 'ode_0_ann.pickle')).cuda()

    dimensionality = 0

    trig_tokens = TrigonometricTokens(freq = (2 - 1e-8, 2 + 1e-8), 
                                      dimensionality = dimensionality)
    grid_tokens = GridTokens(['x_0',], dimensionality = dimensionality, max_power = 2)
    
    epde_search_obj = EpdeSearch(use_solver = False, dimensionality = dimensionality, boundary = 10,
                                 coordinate_tensors = (t,), verbose_params = {'show_iter_idx' : True},
                                 device = 'cuda')

    epde_search_obj.set_preprocessor(default_preprocessor_type='FD',
                                     preprocessor_kwargs={})

    epde_search_obj.create_pool(data=noised_data, variable_names=['u',], max_deriv_order=(2,),
                                additional_tokens = [grid_tokens, trig_tokens], data_nn = data_nn)

    assert compare_equations(eq_ode_symbolic, eq_ode_incorrect, epde_search_obj)


def VdP_test(operator: CompoundOperator, foldername: str, noise_level: int = 0):
    # u'' + E (u^2 - 1)u' + u = 0, where $\mathcal{E}$ is a positive constant (in the example we will use $\mathcal{E} = 0.2$)
    # Test scenario to evaluate performance on Van-der-Pol oscillator
    eq_vdp_symbolic = '-0.2 * u{power: 2.0} * du/dx0{power: 1.0} + 0.2 * du/dx0{power: 1.0} + -1.0 * u{power: 1.0} + -0.0 \
                           = d^2u/dx0^2{power: 1.0}'
    eq_vdp_incorrect = '-1.0 * d^2u/dx0^2{power: 1.0} + 1.5 * x_0{power: 1.0, dim: 0.0} + -4.0 * u{power: 1.0} + -0.0 \
                        = du/dx0{power: 1.0} * sin{power: 1.0, freq: 2.0, dim: 0.0}'

    # grid, data = load_data(os.path.join(foldername, 'data.npy'))

    step = 0.05; steps_num = 320
    t = np.arange(start = 0., stop = step * steps_num, step = step)
    data = np.load(os.path.join(foldername, 'vdp_data.npy'))
    noised_data = noise_data(data, noise_level)
    data_nn = load_pretrained_PINN(os.path.join(foldername, 'vdp_ann_pretrained.pickle'))

    dimensionality = 0

    trig_tokens = TrigonometricTokens(freq = (2 - 1e-8, 2 + 1e-8), 
                                      dimensionality = dimensionality)
    grid_tokens = GridTokens(['x_0',], dimensionality = dimensionality, max_power = 2)

    epde_search_obj = EpdeSearch(use_solver = True, dimensionality = dimensionality, boundary = 10,
                                 coordinate_tensors = (t,), verbose_params = {'show_iter_idx' : True},
                                 device = 'cpu')

    epde_search_obj.set_preprocessor(default_preprocessor_type='FD',
                                     preprocessor_kwargs={})

    epde_search_obj.create_pool(data=noised_data, variable_names=['u',], max_deriv_order=(2,),
                                additional_tokens = [grid_tokens, trig_tokens], data_nn = data_nn)

    assert compare_equations(eq_vdp_symbolic, eq_vdp_incorrect, epde_search_obj)

def ac_data(filename: str):
    t = np.linspace(0., 1., 51)
    x = np.linspace(-1., 0.984375, 128)
    data = np.load(filename)
    # t = np.linspace(0, 1, shape+1); x = np.linspace(0, 1, shape+1)
    grids = np.meshgrid(t, x, indexing = 'ij')  # np.stack(, axis = 2) , axis = 2)
    return grids, data    


def AC_test(operator: CompoundOperator, foldername: str, noise_level: int = 0):
    # Test scenario to evaluate performance on Allen-Cahn equation
    eq_ac_symbolic = '0.0001 * d^2u/dx1^2{power: 1.0} + -5.0 * u{power: 3.0} + 5.0 * u{power: 1.0} + 0.0 = du/dx0{power: 1.0}'
    eq_ac_incorrect = '-1.0 * d^2u/dx0^2{power: 1.0} + 1.5 * u{power: 1.0} + -0.0 = du/dx0{power: 1.0}'
    
    grid, data = ac_data(os.path.join(foldername, 'ac_data.npy'))
    noised_data = noise_data(data, noise_level)
    data_nn = load_pretrained_PINN(os.path.join(foldername, 'ac_ann_pretrained.pickle'))

    print('Shapes:', data.shape, grid[0].shape)
    dimensionality = 1

    epde_search_obj = EpdeSearch(use_solver = True, dimensionality = dimensionality, boundary = 10,
                                 coordinate_tensors = ((grid[0], grid[1])), verbose_params = {'show_iter_idx' : True},
                                 device = 'cpu')

    epde_search_obj.set_preprocessor(default_preprocessor_type='FD',
                                     preprocessor_kwargs={})

    epde_search_obj.create_pool(data=noised_data, variable_names=['u',], max_deriv_order=(2, 2),
                                additional_tokens = [], data_nn = data_nn)

    assert compare_equations(eq_ac_symbolic, eq_ac_incorrect, epde_search_obj)


def wave_data(filename):
    shape = 80

    # print(os.path.dirname( __file__ ))
    data = np.loadtxt(filename, delimiter=',').T
    t = np.linspace(0, 1, shape + 1);
    x = np.linspace(0, 1, shape + 1)
    grids = np.stack(np.meshgrid(t, x, indexing='ij'), axis=2)
    return grids, data

def wave_test(operator: CompoundOperator, foldername: str, noise_level: int = 0):
    # eq_wave_symbolic = '1. * d^2u/dx1^2{power: 1} + 0. = d^2u/dx0^2{power: 1}'
    eq_wave_symbolic = '0.04 * d^2u/dx1^2{power: 1} + 0. = d^2u/dx0^2{power: 1}'
    eq_wave_incorrect = '1. * d^2u/dx1^2{power: 1} * du/dx1{power: 1} + 2.3 * d^2u/dx0^2{power: 1} + 0. = du/dx0{power: 1}'

    grid, data = wave_data(os.path.join(foldername, 'wave_sln_80.csv'))
    noised_data = noise_data(data, noise_level)
    data_nn = load_pretrained_PINN(os.path.join(foldername, 'ann_pretrained.pickle'))

    dimensionality = data.ndim - 1

    epde_search_obj = EpdeSearch(use_solver=True, dimensionality=dimensionality, boundary=10,
                                 coordinate_tensors=(grid[..., 0], grid[..., 1]),
                                 verbose_params={'show_iter_idx': True},
                                 device='cpu')

    epde_search_obj.set_preprocessor(default_preprocessor_type='FD',
                                     preprocessor_kwargs={})

    epde_search_obj.create_pool(data=noised_data, variable_names=['u', ], max_deriv_order=(2, 2),
                                additional_tokens=[], data_nn=data_nn)

    assert compare_equations(eq_wave_symbolic, eq_wave_incorrect, epde_search_obj)


def kdv_data(filename, shape = 80):
    shape = 80
    
    print(os.path.dirname( __file__ ))
    data = np.loadtxt(filename, delimiter = ',').T

    t = np.linspace(0, 1, shape+1); x = np.linspace(0, 1, shape+1)
    grids = np.meshgrid(t, x, indexing = 'ij') # np.stack(, axis = 2)
    return grids, data


def KdV_test(operator: CompoundOperator, foldername: str, noise_level: int = 0):
    # Test scenario to evaluate performance on Korteweg-de Vries equation
    eq_kdv_symbolic = '-6.0 * du/dx1{power: 1.0} * u{power: 1.0} + -1.0 * d^3u/dx1^3{power: 1.0} + \
                           1.0 * sin{power: 1, freq: 1.0, dim: 1} * cos{power: 1, freq: 1.0, dim: 0} + \
                           0.0 = du/dx0{power: 1.0}'

    eq_kdv_incorrect = '0.04 * d^2u/dx1^2{power: 1} + 0. = d^2u/dx0^2{power: 1}'

    grid, data = kdv_data(os.path.join(foldername, 'data.csv'))
    noised_data = noise_data(data, noise_level)
    data_nn = load_pretrained_PINN(os.path.join(foldername, 'kdv_0_ann.pickle'))

    print('Shapes:', data.shape, grid[0].shape)
    dimensionality = 1

    trig_tokens = TrigonometricTokens(freq=(1 - 1e-8, 1 + 1e-8),
                                      dimensionality=dimensionality)

    epde_search_obj = EpdeSearch(use_solver = True, dimensionality = dimensionality, boundary = 10,
                                 coordinate_tensors = (grid[0], grid[1]), verbose_params = {'show_iter_idx' : True},
                                 device = 'cuda')

    epde_search_obj.set_preprocessor(default_preprocessor_type='FD',
                                     preprocessor_kwargs={}) #'epochs_max': 5e4

    epde_search_obj.create_pool(data=noised_data, variable_names=['u',], max_deriv_order=(2, 3),
                                additional_tokens = [trig_tokens,], data_nn = data_nn)

    # np.save(os.path.join(foldername, 'kdv_0_derivs.npy'), epde_search_obj.derivatives)

    assert compare_equations(eq_kdv_symbolic, eq_kdv_incorrect, epde_search_obj)


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

    epde_search_obj = EpdeSearch(use_solver=False, multiobjective_mode=True, dimensionality=dimensionality, boundary=10,
                                 coordinate_tensors=(t,), verbose_params={'show_iter_idx': True},
                                 device='cuda')

    epde_search_obj.set_preprocessor(default_preprocessor_type='FD',
                                     preprocessor_kwargs={})

    popsize = 8
    epde_search_obj.set_moeadd_params(population_size=popsize, training_epochs=5)

    factors_max_number = {'factors_num': [1, 2], 'probas': [0.65, 0.35]}

    epde_search_obj.fit(data=[noised_data, ], variable_names=['u', ], max_deriv_order=(2, 2),
                        equation_terms_max_number=5, data_fun_pow=1,
                        additional_tokens=[trig_tokens, grid_tokens],
                        equation_factors_max_number=factors_max_number,
                        eq_sparsity_interval=(1e-12, 1e-4), data_nn=data_nn) #

    epde_search_obj.equations(only_print=True, num=1)

    # import pickle
    # fname = os.path.join(r'C:\Users\user\EPDE_tests\models', 'ode_0_ann.pickle')
    # with open(fname, 'wb') as output_file:
    #     pickle.dump(global_var.solution_guess_nn, output_file)
    # epde_search_obj.visualize_solutions()
    return epde_search_obj

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

    epde_search_obj = EpdeSearch(use_solver=False, dimensionality=dimensionality, boundary=10,
                                 coordinate_tensors=(t,), verbose_params={'show_iter_idx': True},
                                 device='cuda')

    epde_search_obj.set_preprocessor(default_preprocessor_type='FD',
                                     preprocessor_kwargs={})

    popsize = 8
    epde_search_obj.set_moeadd_params(population_size=popsize, training_epochs=5)

    factors_max_number = {'factors_num': [1, 2], 'probas': [0.65, 0.35]}

    epde_search_obj.fit(data=[noised_data, ], variable_names=['u', ], max_deriv_order=(2, 2),
                        equation_terms_max_number=5, data_fun_pow=2,
                        additional_tokens=[trig_tokens, grid_tokens],
                        equation_factors_max_number=factors_max_number,
                        eq_sparsity_interval=(1e-12, 1e-4), data_nn=data_nn) #

    epde_search_obj.equations(only_print=True, num=1)

    # import pickle
    # fname = os.path.join(r'C:\Users\user\EPDE_tests\models', 'ode_0_ann.pickle')
    # with open(fname, 'wb') as output_file:
    #     pickle.dump(global_var.solution_guess_nn, output_file)
    # epde_search_obj.visualize_solutions()
    return epde_search_obj


def kdv_discovery(foldername, noise_level):
    grid, data = kdv_data(os.path.join(foldername, 'data.csv'))
    noised_data = noise_data(data, noise_level)
    data_nn = load_pretrained_PINN(os.path.join(foldername, f'kdv_{noise_level}_ann.pickle'))

    dimensionality = data.ndim - 1

    epde_search_obj = EpdeSearch(use_solver=False, multiobjective_mode=True,
                                      dimensionality=dimensionality, boundary=5,
                                      coordinate_tensors=grid, device='cuda')

    # epde_search_obj.set_preprocessor(default_preprocessor_type='ANN',
    #                                     preprocessor_kwargs={'epochs_max' : 1e3})
    epde_search_obj.set_preprocessor(default_preprocessor_type='FD',
                                     preprocessor_kwargs={})
    popsize = 8

    epde_search_obj.set_moeadd_params(population_size=popsize,
                                      training_epochs=10)


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

    bounds = (1e-12, 1e-4)
    epde_search_obj.fit(data=noised_data, variable_names=['u', ], max_deriv_order=(2, 3), derivs=None,
                        equation_terms_max_number=5, data_fun_pow=1,
                        additional_tokens=[custom_trig_tokens],
                        equation_factors_max_number=factors_max_number,
                        eq_sparsity_interval=bounds, fourier_layers=False, data_nn=data_nn)

    epde_search_obj.equations(only_print=True, num=1)
    epde_search_obj.visualize_solutions()

    return epde_search_obj

def wave_discovery(foldername, noise_level):
    grid, data = wave_data(os.path.join(foldername, 'wave_sln_80.csv'))
    noised_data = noise_data(data, noise_level)
    data_nn = load_pretrained_PINN(os.path.join(foldername, 'ann_pretrained.pickle'))

    dimensionality = data.ndim - 1

    epde_search_obj = EpdeSearch(use_solver=False, multiobjective_mode=True,
                                      dimensionality=dimensionality, boundary=20,
                                      coordinate_tensors=(grid[..., 0], grid[..., 1]), device='cuda')

    # epde_search_obj.set_preprocessor(default_preprocessor_type='ANN',
    #                                     preprocessor_kwargs={'epochs_max' : 1e3})
    epde_search_obj.set_preprocessor(default_preprocessor_type='FD',
                                     preprocessor_kwargs={})
    popsize = 8

    epde_search_obj.set_moeadd_params(population_size=popsize,
                                      training_epochs=10)


    custom_grid_tokens = CacheStoredTokens(token_type='grid',
                                                token_labels=['t', 'x'],
                                                token_tensors={'t': grid[0], 'x': grid[1]},
                                                params_ranges={'power': (1, 1)},
                                                params_equality_ranges=None)

    trig_params_ranges = {'power': (1, 1)}
    trig_params_equal_ranges = {}

    trig_tokens = TrigonometricTokens(dimensionality=dimensionality, freq = (0.999, 1.001))

    factors_max_number = {'factors_num': [1, 2], 'probas': [0.65, 0.35]}

    bounds = (1e-12, 1e-4)
    epde_search_obj.fit(data=noised_data, variable_names=['u', ], max_deriv_order=(2, 3), derivs=None,
                        equation_terms_max_number=5, data_fun_pow=3,
                        additional_tokens=[],
                        equation_factors_max_number=factors_max_number,
                        eq_sparsity_interval=bounds, fourier_layers=False, data_nn=data_nn) #

    epde_search_obj.equations(only_print=True, num=1)
    epde_search_obj.visualize_solutions()

    return epde_search_obj

def ac_discovery(foldername, noise_level):
    grid, data = ac_data(os.path.join(foldername, 'ac_data.npy'))
    noised_data = noise_data(data, noise_level)
    data_nn = load_pretrained_PINN(os.path.join(foldername, f'kdv_{noise_level}_ann.pickle'))

    dimensionality = data.ndim - 1

    epde_search_obj = EpdeSearch(use_solver=False, multiobjective_mode=True,
                                      dimensionality=dimensionality, boundary=20,
                                      coordinate_tensors=grid, device='cuda')

    # epde_search_obj.set_preprocessor(default_preprocessor_type='ANN',
    #                                     preprocessor_kwargs={'epochs_max' : 1e3})
    epde_search_obj.set_preprocessor(default_preprocessor_type='FD',
                                     preprocessor_kwargs={})
    popsize = 8

    epde_search_obj.set_moeadd_params(population_size=popsize,
                                      training_epochs=10)


    custom_grid_tokens = CacheStoredTokens(token_type='grid',
                                                token_labels=['t', 'x'],
                                                token_tensors={'t': grid[0], 'x': grid[1]},
                                                params_ranges={'power': (1, 1)},
                                                params_equality_ranges=None)

    trig_params_ranges = {'power': (1, 1)}
    trig_params_equal_ranges = {}

    trig_tokens = TrigonometricTokens(dimensionality=dimensionality, freq = (0.999, 1.001))

    factors_max_number = {'factors_num': [1, 2], 'probas': [0.65, 0.35]}

    bounds = (1e-12, 1e-4)
    epde_search_obj.fit(data=noised_data, variable_names=['u', ], max_deriv_order=(2, 3), derivs=None,
                        equation_terms_max_number=5, data_fun_pow=3,
                        additional_tokens=[],
                        equation_factors_max_number=factors_max_number,
                        eq_sparsity_interval=bounds, fourier_layers=False, data_nn=data_nn) #

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
    ode_folder_name = os.path.join(directory, 'data\\ode')
    vdp_folder_name = os.path.join(directory, 'data\\vdp')
    ac_folder_name = os.path.join(directory, 'data\\ac')
    wave_folder_name = os.path.join(directory, 'data\\wave')
    kdv_folder_name = os.path.join(directory, 'data\\kdv')

    # Pair-wise tests
    # ODE_test(fit_operator, ode_folder_name, 0)
    # VdP_test(fit_operator, vdp_folder_name, 0)
    # AC_test(fit_operator, ac_folder_name, 0)
    # wave_test(fit_operator, wave_folder_name, 0)
    # KdV_test(fit_operator, kdv_folder_name, 0)

    # Full_scale test
    # eso = ODE_discovery(ode_folder_name, 0)
    # eso = vdp_discovery(vdp_folder_name, 0)
    # eso = ac_discovery(ac_folder_name, 0)
    # eso = wave_discovery(wave_folder_name, 0)
    eso = kdv_discovery(kdv_folder_name, 0)

