#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 20:55:49 2023

@author: maslyaev
"""

import numpy as np
import pandas as pd

'''

You can install EPDE directly from our github repo:
    pip install git+https://github.com/ITMO-NSS-team/EPDE@main    

'''

import epde.interface.interface as epde_alg
from epde.interface.prepared_tokens import TrigonometricTokens, CacheStoredTokens, CustomEvaluator, CustomTokens

import os
import sys

sys.path.append('../')

sys.path.pop()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..')))

def run_KdV_eq_search(multiobjective_mode, derivs):
    """
    Runs the KdV equation search using the EPDE algorithm.
        
        This method orchestrates the equation discovery process for the KdV equation.
        It configures the search space, evolutionary algorithm parameters, and custom tokens
        to effectively explore potential equation candidates. The goal is to identify the
        equation that best describes the relationship between the provided data and its derivatives.
        
        Args:
            multiobjective_mode: A boolean indicating whether to use multiobjective optimization,
                                 allowing for trade-offs between different objectives such as accuracy
                                 and equation complexity.
            derivs: The derivatives of the data, precomputed and provided as input for the equation search.
        
        Returns:
            tuple: A tuple containing the equation search results and the metric value.
                   The equation search results are a list of found equations, ranked by their
                   ability to fit the data and satisfy the search criteria.
                   The metric is a float representing the objective function value
                   for the best equation, quantifying its fitness based on the chosen optimization strategy.
    """
    epde_search_obj = epde_alg.EpdeSearch(multiobjective_mode=multiobjective_mode, use_solver=False, 
                                           dimensionality=dimensionality, boundary=boundary, 
                                           coordinate_tensors = grids)    
    epde_search_obj.set_preprocessor(default_preprocessor_type='poly', # use_smoothing = True
                                     preprocessor_kwargs={'use_smoothing' : False})
    popsize = 7
    if multiobjective_mode:
        epde_search_obj.set_moeadd_params(population_size = popsize, 
                                          training_epochs=80)
    else:
        epde_search_obj.set_singleobjective_params(population_size = popsize, 
                                                   training_epochs=80)


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
    '''
    Ensure the correctness of the paths!
    '''
    
    path = '/home/maslyaev/epde/EPDE_main/projects/benchmarking/KdV/data/'
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

    paretos_mo = []
    paretos_so = []
    
    exp_num = 10
    for exp_run in range(exp_num):
        paretos_mo.append(run_KdV_eq_search(multiobjective_mode = True, derivs=derivs))
        paretos_so.append(run_KdV_eq_search(multiobjective_mode = False, derivs=derivs))
            
    obj_funs_mo = [elem[1] for elem in paretos_mo]
    obj_funs_so = [elem[1] for elem in paretos_so]
    
        
    '''
    In the experiment with multi-objective optimization: 
    [elem[1] for elem in paretos] yields:
        
    obj_funs_mo = [0.19872577110025424,
                   0.010044168726883887,
                   0.010044168726881037,
                   0.010044168726882132,
                   0.010044168726883887,
                   0.010053873940359866,
                   0.024887542837425773,
                   0.03168872586194803,
                   0.037691446421954516,
                   0.029037133966699005]
 
    -- || -- with single-objective optimization: 
    [elem[1] for elem in paretos] yields:
        
    obj_funs_so = [0.02612207554109057,
                   0.2617447823390866,
                   0.02287012183806553,
                   0.0419274773296572,
                   0.018129337042157977,
                   0.22147963853403785,
                   0.15777454100663743,
                   0.05992896372668442,
                   0.09788517017857494,
                   0.09524688508509466]
    
    '''
    
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['ps.fonttype'] = 42    
    
    plt.rcParams["figure.figsize"] = (3.0, 3.5)
    my_dict = {'Single Objective': obj_funs_so, 'Multi-Objective': obj_funs_mo}
    
    fig, ax = plt.subplots()
    ax.set_yscale('log')
    ax.grid(alpha = 0.5)
    ax.boxplot(my_dict.values(), whis=[5, 95])
    ax.set_xticklabels(my_dict.keys())
    plt.savefig('boxplot_KdV.png', dpi = 300, format = 'png', bbox_inches = 'tight')       