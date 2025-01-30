import sys
import os 

sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '../..')))
import pandas as pd
# import pickle
# from typing import Tuple
import numpy as np
from epde.evaluators import CustomEvaluator
from epde.interface.prepared_tokens import CustomTokens
# from epde.interface.prepared_tokens import CustomTokens, PhasedSine1DTokens, ConstantToken
from epde.interface.equation_translator import translate_equation
from epde.interface.interface import EpdeSearch

from epde.operators.common.coeff_calculation import LinRegBasedCoeffsEquation
from epde.operators.common.sparsity import LASSOSparsity

from epde.operators.utils.operator_mappers import map_operator_between_levels
import epde.operators.common.fitness as fitness
from epde.operators.utils.template import CompoundOperator

# def load_data(data_filename: str, grid_filename : str = None) -> Tuple[np.ndarray]:
#     if '.npy' in data_filename:
#         data = np.load(data_filename)
#     elif ('.csv' in data_filename) or ('.txt' in data_filename):
#         data = np.loadtxt(data_filename)

#     if grid_filename is None:
#         if '.npy' in grid_filename:
#             grid = np.load(grid_filename)
#         elif ('.csv' in grid_filename) or ('.txt' in grid_filename):
#             grid = np.loadtxt(grid_filename)
#     else:


#     return (data, grid)

def load_data(filename):
    shape = 80
    
    # print(os.path.dirname( __file__ ))
    # data = np.loadtxt(filename, delimiter = ',').T
    t = np.linspace(0, 1, shape+1); x = np.linspace(0, 10, 2*shape+1)
    grids = np.stack(np.meshgrid(t, x, indexing = 'ij'), axis = 2)
    # g1 = np.meshgrid(t, x, indexing = 'ij')
    # gggg = grids[-1, -1, 0]
    # gggg1 = grids[-1, -1, 1]
    return grids#, data


# def load_pretrained_PINN(ann_filename):
#     try:
#         with open(ann_filename, 'rb') as data_input_file:
#             data_nn = pickle.load(data_input_file)
#     except FileNotFoundError:
#         print('No model located, proceeding with ann approx. retraining.')
#         data_nn = None
#     return data_nn

def prepare_suboperators(fitness_operator: CompoundOperator, use_solver = True) -> CompoundOperator:
    sparsity = LASSOSparsity()
    coeff_calc = LinRegBasedCoeffsEquation()

    if use_solver:
        sparsity = map_operator_between_levels(sparsity, 'gene level', 'chromosome level')
        coeff_calc = map_operator_between_levels(coeff_calc, 'gene level', 'chromosome level')

    fitness_operator.set_suboperators({'sparsity' : sparsity,
                                       'coeff_calc' : coeff_calc})
    
    if not use_solver:
        fitness_operator = map_operator_between_levels(fitness_operator, 'gene level', 'chromosome level')
    return fitness_operator


def load_data_kdv():
    path_full = os.path.join("/home/mikemaslyaev/Documents/EPDE_merge/EPDE_merge/projects/kdv/data_kdv", "KdV_sln_100.csv")
    df = pd.read_csv(path_full, header=None)

    dddx = pd.read_csv(os.path.join("/home/mikemaslyaev/Documents/EPDE_merge/EPDE_merge/projects/kdv/data_kdv", "ddd_x_100.csv"), header=None)
    ddx = pd.read_csv(os.path.join("/home/mikemaslyaev/Documents/EPDE_merge/EPDE_merge/projects/kdv/data_kdv", "dd_x_100.csv"), header=None)
    dx = pd.read_csv(os.path.join("/home/mikemaslyaev/Documents/EPDE_merge/EPDE_merge/projects/kdv/data_kdv", "d_x_100.csv"), header=None)
    dt = pd.read_csv(os.path.join("/home/mikemaslyaev/Documents/EPDE_merge/EPDE_merge/projects/kdv/data_kdv", "d_t_100.csv"), header=None)

    u_init = df.values
    u = np.transpose(u_init)

    ddd_x = dddx.values
    ddd_x = np.transpose(ddd_x)
    dd_x = ddx.values
    dd_x = np.transpose(dd_x)
    d_x = dx.values
    d_x = np.transpose(d_x)
    d_t = dt.values
    d_t = np.transpose(d_t)

    # derivs = [d_t, d_x, dd_x, ddd_x]
    derivs = np.zeros(shape=(u.shape[0]*u.shape[1], 4))
    derivs[:, 0] = d_t.ravel()
    derivs[:, 1] = d_x.ravel()
    derivs[:, 2] = dd_x.ravel()
    derivs[:, 3] = ddd_x.ravel()

    t = np.linspace(0, 1, u_init.shape[0])
    x = np.linspace(0, 1, u_init.shape[1])
    grids = np.stack(np.meshgrid(t, x, indexing='ij'), axis=2)
    return u, grids, derivs


if __name__ == "__main__":
    # Operator = fitness.SolverBasedFitness # Replace by the developed PIC-based operator.
    u, grids, derivs = load_data_kdv()
    # directory = os.path.dirname(os.path.realpath(__file__))
    dimensionality = u.ndim - 1

    print('u.shape', u.shape)
    print('grids.shape', grids[..., 0].shape, grids[..., 1].shape)
    print('grids 0', grids[..., 0])

    print('grids 1', grids[..., 1])

    print('derivs.shape', derivs.shape)

    custom_trigonometric_eval_fun = {
        'cos(t)sin(x)': lambda *grids, **kwargs: (np.cos(grids[0]) * np.sin(grids[1])) ** kwargs['power']}
    custom_trig_evaluator = CustomEvaluator(custom_trigonometric_eval_fun,
                                            eval_fun_params_labels=['power'])
    trig_params_ranges = {'power': (1, 1)}
    trig_params_equal_ranges = {}

    custom_trig_tokens = CustomTokens(token_type='trigonometric',
                                      token_labels=['cos(t)sin(x)'],
                                      evaluator=custom_trig_evaluator,
                                      params_ranges=trig_params_ranges,
                                      params_equality_ranges=trig_params_equal_ranges,
                                      meaningful=True, unique_token_type=False)


    epde_search_obj = EpdeSearch(use_solver = True, dimensionality = dimensionality, boundary = 10,
                                      coordinate_tensors = (grids[..., 0], grids[..., 1]), verbose_params = {'show_iter_idx' : True},
                                      device = 'cpu')

    epde_search_obj.set_preprocessor(default_preprocessor_type='poly',
                                     preprocessor_kwargs={})

    epde_search_obj.create_pool(data=u, variable_names=['u',], max_deriv_order=(1, 3),
                                additional_tokens = [], derivs=[derivs,], fourier_layers=False)#, data_nn = data_nn) custom_trig_tokens

    # eq_wave_symbolic = '1. * d^2u/dx1^2{power: 1} + 0. = d^2u/dx0^2{power: 1}'
    eq_kdv = '-6. * u{power: 1} * du/dx1{power: 1} + -1. * d^3u/dx1^3{power: 1} + 0. = du/dx0{power: 1}' # cos(t)sin(x){power: 1}'
    # eq_wave_symbolic = '0.04 * d^2u/dx1^2{power: 1} + 0. = d^2u/dx0^2{power: 1}'
    wave_eq = translate_equation(eq_kdv, epde_search_obj.pool, all_vars = ['u',])
    wave_eq.vals['u'].main_var_to_explain = 'u'
    wave_eq.vals['u'].metaparameters = {('sparsity', 'u'): {'optimizable': False, 'value': 0.5}}
    print(wave_eq.text_form)

    # eq_incorrect_symbolic = '1. * d^2u/dx1^2{power: 1} * du/dx1{power: 1} + 2.3 * du/dx0{power: 1} + 0. = du/dx1{power: 1}'
    # incorrect_eq = translate_equation(eq_incorrect_symbolic, epde_search_obj.pool, all_vars = ['u',])
    # incorrect_eq.vals['u'].main_var_to_explain = 'u'
    # incorrect_eq.vals['u'].metaparameters = {('sparsity', 'u'): {'optimizable': False, 'value': 0.5}}
    # print(incorrect_eq.text_form)

    Operator = fitness.SolverBasedFitness  # Replace by the developed PIC-based operator.
    operator_params = {"penalty_coeff": 0.2, }  # "pinn_loss_mult" : 1e4}
    fit_operator = prepare_suboperators(Operator(list(operator_params.keys())))
    fit_operator.params = operator_params
    fit_operator.apply(wave_eq, {})
    # fit_operator.apply(incorrect_eq, {})

    print(wave_eq.vals['u'].fitness_value)
    # print(incorrect_eq.vals['u'].fitness_value)
    # assert wave_eq.vals['u'].fitness_value < incorrect_eq.vals['u'].fitness_value