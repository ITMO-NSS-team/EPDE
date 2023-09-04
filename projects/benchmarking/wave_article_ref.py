#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 15:47:47 2022

@author: maslyaev
"""

import numpy as np

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

def run_wave_eq_search(multiobjective_mode, derivs = None):
    epde_search_obj = epde_alg.EpdeSearch(multiobjective_mode=multiobjective_mode, use_solver = False, 
                                          dimensionality = dimensionality, boundary = 10,
                                          coordinate_tensors = grids)    
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
        metric = epde_search_obj.get_equations_by_complexity(complexity = 2)[0].obj_fun[0]
    else:
        metric = epde_search_obj.equation_search_results(only_print = False, num = 1)[0].obj_fun[0]
    print(f'Obtained metric is {metric}')
    
    return epde_search_obj.equation_search_results(only_print = False, num = 1), metric, epde_search_obj.saved_derivaties 
    

if __name__ == '__main__':
    '''
    Ensure the correctness of the paths!
    '''    
    
    results = []
        
    try:
        print(os.path.dirname( __file__ ))
        data_file = os.path.join(os.path.dirname( __file__ ), 'data/wave_sln_80.csv')
        data = np.loadtxt(data_file, delimiter = ',').T
    except FileNotFoundError:
        data_file = '/home/maslyaev/epde/EPDE_rework/projects/benchmarking/wave/wave_sln_80.csv'
        data = np.loadtxt(data_file, delimiter = ',').T
        
    t = np.linspace(0, 1, 81); x = np.linspace(0, 1, 81)

    grids = np.meshgrid(t, x, indexing = 'ij')
    dimensionality = data.ndim - 1
    
    res = run_wave_eq_search(multiobjective_mode = False, derivs = None)
    derivs = res[2]
    
    paretos_mo = []
    paretos_so = []
    
    exp_num = 10
    for exp_run in range(exp_num):
        paretos_mo.append(run_wave_eq_search(multiobjective_mode = True, derivs=derivs)[:2])
        paretos_so.append(run_wave_eq_search(multiobjective_mode = False, derivs=derivs)[:2])
            
    obj_funs_mo = [elem[1] for elem in paretos_mo]
    obj_funs_so = [elem[1] for elem in paretos_so]    
        
    obj_funs_mo = [elem[1] for elem in paretos_mo]
    obj_funs_so = [elem[1] for elem in paretos_so]        
        
    '''    
    obj_funs_so = [2.030488112155742,
                   2.0304881121557465,
                   2.0304881121557465,
                   2.0304881121557465,
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
    '''