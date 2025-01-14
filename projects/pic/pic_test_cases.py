import sys
import os 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../..')))

import pickle
from typing import Tuple, List
import numpy as np

from epde.interface.prepared_tokens import CustomTokens, PhasedSine1DTokens, ConstantToken
from epde.interface.equation_translator import translate_equation
from epde.interface.interface import EpdeSearch

from epde.operators.common.coeff_calculation import LinRegBasedCoeffsEquation
from epde.operators.common.sparsity import LASSOSparsity

from epde.operators.utils.operator_mappers import map_operator_between_levels
import epde.operators.common.fitness as fitness
from epde.operators.utils.template import CompoundOperator

# Introduce noise levels, test with complex setups


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

    print([correct_eq.vals[var].fitness_value < incorrect_eq.vals[var].fitness_value for var in all_vars])
    return all([correct_eq.vals[var].fitness_value < incorrect_eq.vals[var].fitness_value for var in all_vars])


def prepare_suboperators(fitness_operator: CompoundOperator) -> CompoundOperator:
    sparsity = LASSOSparsity()
    coeff_calc = LinRegBasedCoeffsEquation()

    sparsity = map_operator_between_levels(sparsity, 'gene level', 'chromosome level')
    coeff_calc = map_operator_between_levels(coeff_calc, 'gene level', 'chromosome level')

    fitness_operator.set_suboperators({'sparsity' : sparsity,
                                       'coeff_calc' : coeff_calc})
    return fitness_operator


def ODE_test(operator: CompoundOperator, foldername: str):
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
    data_nn = load_pretrained_PINN(os.path.join(foldername, 'ode_ann_pretrained.pickle'))

    dimensionality = 0

    from epde import TrigonometricTokens, GridTokens
    trig_tokens = TrigonometricTokens(freq = (2 - 1e-8, 2 + 1e-8), 
                                      dimensionality = dimensionality)
    grid_tokens = GridTokens(['x_0',], dimensionality = dimensionality, max_power = 2)
    
    epde_search_obj = EpdeSearch(use_solver = True, dimensionality = dimensionality, boundary = 10,
                                 coordinate_tensors = (t,), verbose_params = {'show_iter_idx' : True},
                                 device = 'cpu')

    epde_search_obj.set_preprocessor(default_preprocessor_type='FD',
                                     preprocessor_kwargs={})

    epde_search_obj.create_pool(data=data, variable_names=['u',], max_deriv_order=(2,),
                                additional_tokens = [grid_tokens, trig_tokens], data_nn = data_nn)

    assert compare_equations(eq_ode_symbolic, eq_ode_incorrect, epde_search_obj)


def VdP_test(operator: CompoundOperator, foldername: str):
    # u'' + E (u^2 - 1)u' + u = 0, where $\mathcal{E}$ is a positive constant (in the example we will use $\mathcal{E} = 0.2$)
    # Test scenario to evaluate performance on Van-der-Pol oscillator
    eq_vdp_symbolic = '-0.2 * u{power: 2.0} * d^2u/dx0^2{power: 1.0} + 0.2 * d^2u/dx0^2{power: 1.0} + -1.0 * u{power: 1.0} + -0.0 \
                       = d^2u/dx0^2{power: 1.0}'
    eq_vdp_incorrect = '-1.0 * d^2u/dx0^2{power: 1.0} + 1.5 * x_0{power: 1.0, dim: 0.0} + -4.0 * u{power: 1.0} + -0.0 \
                        = du/dx0{power: 1.0} * sin{power: 1.0, freq: 2.0, dim: 0.0}'

    # grid, data = load_data(os.path.join(foldername, 'data.npy'))

    step = 0.05; steps_num = 320
    t = np.arange(start = 0., stop = step * steps_num, step = step)
    data = np.load(os.path.join(foldername, 'vdp_data.npy'))    
    data_nn = load_pretrained_PINN(os.path.join(foldername, 'vdp_ann_pretrained.pickle'))

    dimensionality = 0
    
    from epde import TrigonometricTokens, GridTokens
    trig_tokens = TrigonometricTokens(freq = (2 - 1e-8, 2 + 1e-8), 
                                      dimensionality = dimensionality)
    grid_tokens = GridTokens(['x_0',], dimensionality = dimensionality, max_power = 2)

    epde_search_obj = EpdeSearch(use_solver = True, dimensionality = dimensionality, boundary = 10,
                                 coordinate_tensors = (t,), verbose_params = {'show_iter_idx' : True},
                                 device = 'cpu')

    epde_search_obj.set_preprocessor(default_preprocessor_type='FD',
                                     preprocessor_kwargs={})

    epde_search_obj.create_pool(data=data, variable_names=['u',], max_deriv_order=(2,),
                                additional_tokens = [grid_tokens, trig_tokens], data_nn = data_nn)

    assert compare_equations(eq_vdp_symbolic, eq_vdp_incorrect, epde_search_obj)

def ac_data(filename: str):
    t = np.linspace(0., 1., 51)
    x = np.linspace(-1., 0.984375, 128)
    data = np.load(filename).T
    # t = np.linspace(0, 1, shape+1); x = np.linspace(0, 1, shape+1)
    grids = np.meshgrid(t, x, indexing = 'ij')  # np.stack(, axis = 2) , axis = 2)
    return grids, data    


def AC_test(operator: CompoundOperator, foldername: str):
    # Test scenario to evaluate performance on Allen-Cahn equation
    eq_ac_symbolic = '0.0001 * d^2u/dx1^2{power: 1.0} + -5.0 * u{power: 1.0} + 5.0 * u{power: 1.0} + 0.0 = du/dx0{power: 1.0}'
    eq_ac_incorrect = '-1.0 * d^2u/dx0^2{power: 1.0} + 1.5 * u{power: 1.0} -0.0 = du/dx0{power: 1.0}'
    
    grid, data = ac_data(os.path.join(foldername, 'ac_data.npy'))
    data_nn = load_pretrained_PINN(os.path.join(foldername, 'ac_ann_pretrained.pickle'))

    print('Shapes:', data.shape, grid[0].shape)
    dimensionality = 1

    epde_search_obj = EpdeSearch(use_solver = True, dimensionality = dimensionality, boundary = 10,
                                 coordinate_tensors = ((grid[0], grid[1])), verbose_params = {'show_iter_idx' : True},
                                 device = 'cpu')

    epde_search_obj.set_preprocessor(default_preprocessor_type='FD',
                                     preprocessor_kwargs={})

    epde_search_obj.create_pool(data=data, variable_names=['u',], max_deriv_order=(2, 2),
                                additional_tokens = [], data_nn = data_nn)

    assert compare_equations(eq_ac_symbolic, eq_ac_incorrect, epde_search_obj) 


def wave_test(operator: CompoundOperator, foldername: str):
    # Test scenario to evaluate performance on wave equation
    # Can be implemented from the proof-of-concept scenario
    pass


def kdv_data(filename, shape = 80):
    shape = 80
    
    print(os.path.dirname( __file__ ))
    data = np.loadtxt(filename, delimiter = ',').T

    t = np.linspace(0, 1, shape+1); x = np.linspace(0, 1, shape+1)
    grids = np.meshgrid(t, x, indexing = 'ij') # np.stack(, axis = 2)
    return grids, data


def KdV_test(operator: CompoundOperator, foldername: str):
    # Test scenario to evaluate performance on Korteweg-de Vries equation
    eq_kdv_symbolic = '-6.0 * du/dx1{power: 1.0} * u{power: 1.0} + -1.0 * d^3u/dx1^3{power: 1.0} + \
                       1.0 * sin{power: 1, freq: 2.0, dim: 1} * cos{power: 1, freq: 2.0, dim: 1} + \
                       -1.0 * u{power: 1.0} + 0.0 = du/dx0{power: 1.0}'
    eq_kdv_incorrect = '0.04 * d^2u/dx1^2{power: 1} + 0. = d^2u/dx0^2{power: 1}'
    
    grid, data = kdv_data(os.path.join(foldername, 'data.csv'))
    data_nn = load_pretrained_PINN(os.path.join(foldername, 'kdv_ann_pretrained.pickle'))
    
    print('Shapes:', data.shape, grid[0].shape)
    dimensionality = 1

    from epde import TrigonometricTokens
    trig_tokens = TrigonometricTokens(freq = (2 - 1e-8, 2 + 1e-8), 
                                      dimensionality = dimensionality)

    epde_search_obj = EpdeSearch(use_solver = True, dimensionality = dimensionality, boundary = 10,
                                 coordinate_tensors = (grid[0], grid[1]), verbose_params = {'show_iter_idx' : True},
                                 device = 'cpu')

    epde_search_obj.set_preprocessor(default_preprocessor_type='FD',
                                     preprocessor_kwargs={})

    epde_search_obj.create_pool(data=data, variable_names=['u',], max_deriv_order=(2, 3),
                                additional_tokens = [trig_tokens,], data_nn = data_nn)

    assert compare_equations(eq_kdv_symbolic, eq_kdv_incorrect, epde_search_obj) 

# TODO: implement tests on noised data, corrupted by additive Gaussian noise & preform full-scale equation search experiments.

if __name__ == "__main__":
    # Operator = fitness.SolverBasedFitness # Replace by the developed PIC-based operator.
    Operator = fitness.PIC
    operator_params = {"penalty_coeff" : 0.2, "pinn_loss_mult" : 1e4}
    fit_operator = prepare_suboperators(Operator(list(operator_params.keys())))
    fit_operator.params = operator_params

    ode_folder_name = r"C:\Users\timur\PycharmProjects\EPDE\EPDE\projects\pic\data\ode"
    # ODE_test(fit_operator, ode_folder_name)

    vdp_folder_name = r"C:\Users\timur\PycharmProjects\EPDE\EPDE\projects\pic\data\vdp"
    # VdP_test(fit_operator, vdp_folder_name)

    ac_folder_name = r"C:\Users\timur\PycharmProjects\EPDE\EPDE\projects\pic\data\ac"
    # AC_test(fit_operator, ac_folder_name)

    wave_folder_name = r"C:\Users\timur\PycharmProjects\EPDE\EPDE\projects\pic\data\wave"
    # wave_test(fit_operator, wave_folder_name)

    kdv_folder_name = r"C:\Users\timur\PycharmProjects\EPDE\EPDE\projects\pic\data\kdv"
    KdV_test(fit_operator, kdv_folder_name)