#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 20:55:49 2023

@author: maslyaev
"""

import numpy as np
import pandas as pd

import epde.interface.interface as epde_alg

from epde.interface.prepared_tokens import TrigonometricTokens, CacheStoredTokens, CustomEvaluator, CustomTokens

import os
import sys

sys.path.append('../')

sys.path.pop()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..')))

# from epde.evaluators import CustomEvaluator, simple_function_evaluator, inverse_function_evaluator
# TODO^ caching of the pre-calculated derivatives
    
def run_KdV_eq_search(multiobjective_mode, derivs):
    epde_search_obj = epde_alg.EpdeSearch(multiobjective_mode=multiobjective_mode, use_solver=False, 
                                           dimensionality=dimensionality, boundary=boundary, 
                                           coordinate_tensors = grids, 
                                           verbose_params = {'show_moeadd_epochs' : True})    
    epde_search_obj.set_preprocessor(default_preprocessor_type='poly', # use_smoothing = True
                                     preprocessor_kwargs={'use_smoothing' : False})
    popsize = 7
    if multiobjective_mode:
        epde_search_obj.set_moeadd_params(population_size = popsize, 
                                          training_epochs=40)
    else:
        epde_search_obj.set_singleobjective_params(population_size = popsize, 
                                                   training_epochs=40)


    custom_grid_tokens = CacheStoredTokens(token_type = 'grid',
                                           # boundary = boundary,
                                           token_labels = ['t', 'x'],
                                           token_tensors={'t' : grids[0], 'x' : grids[1]},
                                           params_ranges = {'power' : (1, 1)},
                                           params_equality_ranges = None)

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

    trig_tokens = TrigonometricTokens(dimensionality = dimensionality)
    factors_max_number = {'factors_num' : [1, 2], 'probas' : [0.7, 0.3]}
    
    opt_val = 1e-7
    bounds = (1e-8, 1e-2) if multiobjective_mode else (opt_val, opt_val)
    epde_search_obj.fit(data=[u, ], variable_names=['u',], max_deriv_order=(1, 3), derivs = [derivs,],
                        equation_terms_max_number=5, data_fun_pow = 2, 
                        additional_tokens=[trig_tokens, custom_grid_tokens, custom_trig_tokens], #custom_grid_tokens 
                        equation_factors_max_number = factors_max_number, 
                        eq_sparsity_interval = bounds)
    epde_search_obj.equation_search_results(only_print = True, num = 1)        
    if multiobjective_mode:    
        try:
            metric = epde_search_obj.get_equations_by_complexity(complexity = 5)[0].obj_fun[0]
        except IndexError:
            metric = 999.
    else:
        metric = epde_search_obj.equation_search_results(only_print = False, num = 1)[0].obj_fun[0]
    print(f'Obtained metric is {metric}')
    
    return epde_search_obj.equation_search_results(only_print = False, num = 1), metric 
    
if __name__ == "__main__":
    
    # multiobjective_mode = True
    
    path = '/home/maslyaev/epde/EPDE_main/projects/benchmarking/KdV/Data/'
    df = pd.read_csv(f'{path}KdV_sln_100.csv', header=None)
    dddx = pd.read_csv(f'{path}ddd_x_100.csv', header=None)
    ddx = pd.read_csv(f'{path}dd_x_100.csv', header=None)
    dx = pd.read_csv(f'{path}d_x_100.csv', header=None)
    dt = pd.read_csv(f'{path}d_t_100.csv', header=None)

    u = df.values
    u = np.transpose(u)

    ddd_x = dddx.values
    ddd_x = np.transpose(ddd_x)
    dd_x = ddx.values
    dd_x = np.transpose(dd_x)
    d_x = dx.values
    d_x = np.transpose(d_x)
    d_t = dt.values
    d_t = np.transpose(d_t)

    derivs = np.zeros(shape=(u.shape[0],u.shape[1],4))
    derivs[:, :, 0] = d_t
    derivs[:, :, 1] = d_x
    derivs[:, :, 2] = dd_x
    derivs[:, :, 3] = ddd_x
    derivs = derivs.reshape((-1, 4))

    t = np.linspace(0, 1, u.shape[0])
    x = np.linspace(0, 1, u.shape[1])
    grids = np.meshgrid(t, x, indexing = 'ij')
    
    dimensionality = u.ndim - 1; boundary = 20

    paretos = []
    exp_num = 20
    for exp_run in range(exp_num):
        # paretos.append(run_KdV_eq_search(multiobjective_mode = exp_run >= exp_num/2, derivs=derivs))
        paretos.append(run_KdV_eq_search(multiobjective_mode = True, derivs=derivs))
        
