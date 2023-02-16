#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 15:47:47 2022

@author: maslyaev
"""

import numpy as np

import epde.interface.interface as epde_alg
from epde.interface.prepared_tokens import TrigonometricTokens, CacheStoredTokens

import os
import sys

sys.path.append('../')

sys.path.pop()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..')))

def run_wave_eq_search(multiobjective_mode, derivs = None):
    epde_search_obj = epde_alg.EpdeSearch(multiobjective_mode=multiobjective_mode, use_solver = False, 
                                           dimensionality = dimensionality, boundary = 10,
                                           coordinate_tensors = grids, verbose_params = {'show_moeadd_epochs' : True})    
    # epde_search_obj.set_preprocessor(default_preprocessor_type='spectral', # use_smoothing = True
    #                                  preprocessor_kwargs={})
    popsize = 7
    if multiobjective_mode:
        epde_search_obj.set_moeadd_params(population_size = popsize, 
                                          training_epochs=40)
    else:
        epde_search_obj.set_singleobjective_params(population_size = popsize, 
                                                   training_epochs=40)
    # epde_search_obj.set_moeadd_params(population_size = popsize, training_epochs=50)
    
    custom_grid_tokens = CacheStoredTokens(token_type = 'grid',
                                           token_labels = ['t', 'x'],
                                           token_tensors={'t' : grids[0], 'x' : grids[1]},
                                           params_ranges = {'power' : (1, 1)},
                                           params_equality_ranges = None)        
    trig_tokens = TrigonometricTokens(dimensionality = dimensionality)
    
    factors_max_number = {'factors_num' : [1, 2], 'probas' : [0.8, 0.2]}
    
    opt_val = 1e-1
    bounds = (1e-8, 1e0) if multiobjective_mode else (opt_val, opt_val)    
    epde_search_obj.fit(data=data, variable_names=['u',], max_deriv_order=(2, 2),
                        equation_terms_max_number=5, data_fun_pow = 1, additional_tokens=[trig_tokens, custom_grid_tokens], 
                        equation_factors_max_number=factors_max_number,
                        eq_sparsity_interval=bounds)

    if multiobjective_mode:    
        metric = epde_search_obj.get_equations_by_complexity(complexity = 3)[0].obj_fun[0]
    else:
        metric = epde_search_obj.equation_search_results(only_print = False, num = 1)[0].obj_fun[0]
    print(f'Obtained metric is {metric}')
    
    return epde_search_obj.equation_search_results(only_print = False, num = 1), metric 
    

if __name__ == '__main__':
    results = []
    shapes = [30, 40, 50, 60, 70, 80, 90, 100]
    '''
    Подгружаем данные, содержащие временные ряды динамики "вида-охотника" и "вида-жертвы"
    '''
    shape = 80
        # shape = 50
        
    try:
        print(os.path.dirname( __file__ ))
        data_file = os.path.join(os.path.dirname( __file__ ), f'wave/wave_sln_{shape}.csv')
        data = np.loadtxt(data_file, delimiter = ',').T
    except FileNotFoundError:
        data_file = '/home/maslyaev/epde/EPDE_rework/projects/benchmarking/wave/wave_sln_{shape}.csv'
        data = np.loadtxt(data_file, delimiter = ',').T
        
    t = np.linspace(0, 1, shape+1); x = np.linspace(0, 1, shape+1)
    # x += np.random.normal(0, err_factor*np.min(x), size = x.size)
    # y += np.random.normal(0, err_factor*np.min(y), size = y.size)

    grids = np.meshgrid(t, x, indexing = 'ij')
    dimensionality = data.ndim - 1
    
    paretos = []
    exp_num = 20
    for exp_run in range(exp_num):
        paretos.append(run_wave_eq_search(multiobjective_mode = exp_run >= exp_num/2))
        
    obj_funs_so = [2.030488112155742,
                   2.0304881121557465,
                   2.0304881121557465,
                   1.5521921247458994,
                   11.272247683973458,
                   2.030488112155742,
                   5.2740026827245305,
                   6.146619449139421,
                   10.963368782936891,
                   13.393974342048184]

    obj_funs_mo = [2.030488112155742,
                   2.030488112155742,
                   2.030488112155742,
                   2.030488112155742,
                   2.030488112155742,
                   2.030488112155742,
                   2.030488112155742,
                   2.030488112155742,
                   2.030488112155742,
                   2.030488112155742]       