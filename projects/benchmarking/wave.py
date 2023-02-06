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


if __name__ == '__main__':
    results = []
    shapes = [30, 40, 50, 60, 70, 80, 90, 100]
    '''
    Подгружаем данные, содержащие временные ряды динамики "вида-охотника" и "вида-жертвы"
    '''
    shape = 80
    for iter_idx in range(1):
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
        
        '''
        Подбираем Парето-множество систем дифф. уравнений.
        '''
        epde_search_obj = epde_alg.epde_search(use_solver = False, dimensionality = dimensionality, boundary = 10,
                                               coordinate_tensors = grids, verbose_params = {'show_moeadd_epochs' : True})    
        # epde_search_obj.set_preprocessor(default_preprocessor_type='spectral', # use_smoothing = True
        #                                  preprocessor_kwargs={})
        popsize = 5
        epde_search_obj.set_moeadd_params(population_size = popsize, training_epochs=150)
        
        custom_grid_tokens = CacheStoredTokens(token_type = 'grid',
                                               token_labels = ['t', 'x'],
                                               token_tensors={'t' : grids[0], 'x' : grids[1]},
                                               params_ranges = {'power' : (1, 1)},
                                               params_equality_ranges = None)        
        trig_tokens = TrigonometricTokens(dimensionality = dimensionality)
        
        factors_max_number = {'factors_num' : [1, 2], 'probas' : [0.8, 0.2]}
        
        epde_search_obj.fit(data=data, variable_names=['u',], max_deriv_order=(2, 2),
                            equation_terms_max_number=6, data_fun_pow = 1, additional_tokens=[trig_tokens, custom_grid_tokens], 
                            equation_factors_max_number=factors_max_number,
                            eq_sparsity_interval=(1e-10, 1e-2), coordinate_tensors=[t, ])
        '''
        Смотрим на найденное Парето-множество, 
        
        "идеальное уравнение" имеет вид:
         / -1.3048888807580532 * u{power: 1.0} * v{power: 1.0} + 0.3922851274813135 * u{power: 1.0} + -0.0003917278536547386 = du/dx1{power: 1.0}
         \ -0.9740492564964498 * v{power: 1.0} + 0.9717873909925121 * u{power: 1.0} * v{power: 1.0} + 0.0003500773115704403 = dv/dx1{power: 1.0}
        {'terms_number': {'optimizable': False, 'value': 3}, 'max_factors_in_term': {'optimizable': False, 'value': {'factors_num': [1, 2], 'probas': [0.8, 0.2]}}, 
         ('sparsity', 'u'): {'optimizable': True, 'value': 0.00027172388370453704}, ('sparsity', 'v'): {'optimizable': True, 'value': 0.00019292375116125682}} , with objective function values of [0.2800438  0.18041074 4.         4.        ]         
        '''
        epde_search_obj.equation_search_results(only_print = True, level_num = 1)
        
        res = epde_search_obj.equation_search_results(only_print = False, level_num = 1)
        results.append(res)
        # sys = res[0][1]
        # sys.text_form
        # pareto_frontiers[(err_factor, sigma)].append(res)
    '''
    Решаем уравнение (систему уравнений) при помощи метода .predict(...)
    '''
    # predictions = epde_search_obj.predict(system = sys, boundary_conditions=None)