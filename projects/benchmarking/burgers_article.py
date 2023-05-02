#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 19 15:51:40 2022

@author: maslyaev
"""

import numpy as np
import pandas as pd
import time

'''

You can install EPDE directly from our github repo:
    pip install git+https://github.com/ITMO-NSS-team/EPDE@main    

'''

import epde.interface.interface as epde_alg

from epde.interface.prepared_tokens import TrigonometricTokens, CacheStoredTokens

import os
import sys

sys.path.append('../')

sys.path.pop()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..')))

# from epde.evaluators import CustomEvaluator, simple_function_evaluator, inverse_function_evaluator
# TODO^ caching of the pre-calculated derivatives
    
def run_burg_eq_search(multiobjective_mode, derivs):
    print(u.shape, grids[0].shape, grids[1].shape)
    epde_search_obj = epde_alg.EpdeSearch(multiobjective_mode=multiobjective_mode, use_solver=False, 
                                           dimensionality=dimensionality, boundary=boundary, 
                                           coordinate_tensors = grids)    
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

    trig_tokens = TrigonometricTokens(dimensionality = dimensionality)
    factors_max_number = {'factors_num' : [1, 2], 'probas' : [0.7, 0.3]}
    
    opt_val = 1e-1
    bounds = (1e-8, 1e0) if multiobjective_mode else (opt_val, opt_val)
    epde_search_obj.fit(data=[u, ], variable_names=['u',], max_deriv_order=(2, 1), derivs = [derivs,],
                        equation_terms_max_number=6, data_fun_pow = 2, 
                        additional_tokens=[trig_tokens, custom_grid_tokens], #custom_grid_tokens 
                        equation_factors_max_number = factors_max_number, 
                        eq_sparsity_interval = bounds)
    epde_search_obj.equation_search_results(only_print = True, num = 1)        
    if multiobjective_mode:    
        metric = epde_search_obj.get_equations_by_complexity(complexity = 3)[0].obj_fun[0]
        print(f'Obtained metric is {metric}')
        print(f'Equation with d2u/dx2: {epde_search_obj.get_equations_by_complexity(complexity = 4)[0].text_form}')
        print(f'Equation without d2u/dx2: {epde_search_obj.get_equations_by_complexity(complexity = 3)[0].text_form}')
        time.sleep(10)
    else:
        metric = epde_search_obj.equation_search_results(only_print = False, num = 1)[0].obj_fun[0]
    return epde_search_obj, metric 
    # .equation_search_results(only_print = False, num = 1)    

if __name__ == "__main__":
    '''
    Ensure the correctness of the paths!
    '''
    
    path = '/home/maslyaev/epde/EPDE_main/projects/benchmarking/Burgers/Data/'

    try:
        u_file = os.path.join(os.path.dirname( __file__ ), 'data/burgers_sln_256.csv')
        u = np.loadtxt(u_file, delimiter=',').T
    except (FileNotFoundError, OSError):
        u_file = '/home/maslyaev/epde/EPDE_main/projects/benchmarking/Burgers/Data/burgers_sln_256.csv'
        u = np.loadtxt(u_file, delimiter=',').T
    
    u = u[:128, ...]
    derives = None
    dx = pd.read_csv(f'{path}burgers_sln_dx_256.csv', header=None)
    d_x = dx.values
    d_x = np.transpose(d_x)

    dt = pd.read_csv(f'{path}burgers_sln_dt_256.csv', header=None)
    d_t = dt.values
    d_t = np.transpose(d_t)

    dtt = pd.read_csv(f'{path}burgers_sln_dtt_256.csv', header=None)
    d_tt = dtt.values
    d_tt = np.transpose(d_tt)

    # derives = np.zeros(shape=(data.shape[0], data.shape[1], 3))
    # derives[:, :, 0] = d_t
    # derives[:, :, 1] = d_tt
    # derives[:, :, 2] = d_x

    derives = np.zeros(shape=(u.shape[0], u.shape[1], 3))
    derives[:, :, 0] = d_t[:128, ...]
    derives[:, :, 1] = d_tt[:128, ...]
    derives[:, :, 2] = d_x[:128, ...]
    
    derives = derives.reshape((-1, 3))
        
    # u = np.moveaxis(u, 1, 0)

    t = np.linspace(0, 4, u.shape[0])
    x = np.linspace(-4000, 4000, u.shape[1])
    grids = np.meshgrid(t, x, indexing = 'ij')
    
    dimensionality = u.ndim - 1; boundary = 20

    paretos_mo = []
    paretos_so = []
    
    exp_num = 3
    for exp_run in range(exp_num):
        paretos_mo.append(run_burg_eq_search(multiobjective_mode = True, derivs=derives))
        paretos_so.append(run_burg_eq_search(multiobjective_mode = False, derivs=derives))

    obj_funs_mo = [elem[1] for elem in paretos_mo]
    obj_funs_so = [elem[1] for elem in paretos_so]

    '''      
    obj_funs_mo = [1.8602792969132718e-07,
                   15.151537975821995,
                   1.8602792969132718e-07,
                   6.045698413857898e-12,
                   6.045877564414611e-12,
                   6.045698413857898e-12,
                   6.045698413857898e-12,
                   1.8602792969132718e-07,
                   6.045698413857898e-12,
                   6.045877564414611e-12]
    
    obj_funs_so = [52.55367840511059,
                   50.92569285913047,
                   4.751400052693273e-10,
                   68.07398197293008,
                   22168.84900209751,
                   68.07398197293008,
                   6.045877564414611e-12,
                   1.00755257337613e-09,
                   55.32605105695978,
                   1.0707313482856096e-09]

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
    plt.savefig('boxplot_burgers.png', dpi = 300, format = 'png', bbox_inches = 'tight')    
