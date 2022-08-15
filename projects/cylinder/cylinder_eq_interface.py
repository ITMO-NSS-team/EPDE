#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 12 15:11:52 2021

@author: mike_ubuntu
"""
import os
import pandas as pd

import numpy as np
import epde.interface.interface as epde_alg
import pickle

from epde.interface.prepared_tokens import Custom_tokens, Cache_stored_tokens
from epde.evaluators import CustomEvaluator, simple_function_evaluator, inverse_function_evaluator

def load_data(directory, data_file_format = 'dat', loader_kwargs = {}):
    temp_initial = {7 : 24.86,
                    12 : 25.0307,
                    17 : 25.15106,
                    22 : 26.0285,
                    27 : 26.05125}
    data = []
    for filename in os.listdir(directory):
        print(filename)
        if filename.endswith(".dat"):
            temp = pd.read_csv(directory + '/' + filename, **loader_kwargs)
            dist = int(filename.replace('06V_', '').replace('mm_u.dat', ''))
            
            data.append((dist, temp.loc[temp.tc >= 0].Tr.to_numpy()))
    len_min = min([series.size for _, series in data])
    data = sorted([(dist, series[:len_min]) for dist, series in data], 
                  key = lambda x: x[0])
    
    return np.stack([series - temp_initial[dist] for dist, series in data])
    

if __name__ == "__main__":
    test_iter_limit = 15
    for test_idx in np.arange(test_iter_limit):
        t_max = 400
        
        file = np.loadtxt('/home/maslyaev/epde/EPDE/tests/cylinder/data/Data_32_points_.dat', 
                          delimiter=' ', usecols=range(33))
    
        x = np.linspace(0.5, 16, 32)
        t = file[:t_max, 0]
        grids = np.meshgrid(t, x, indexing = 'ij')
        u = file[:t_max, 1:]
    
        boundary = [10, 1]
    
        dimensionality = u.ndim
        
        epde_search_obj = epde_alg.epde_search(use_solver=False, eq_search_iter = 30, dimensionality=2)
        epde_search_obj.set_memory_properties(u, mem_for_cache_frac = 20)
    
        custom_inverse_eval_fun = lambda *grids, **kwargs: np.power(grids[int(kwargs['dim'])], - kwargs['power']) 
        custom_inv_fun_evaluator = CustomEvaluator(custom_inverse_eval_fun, 
                                                    eval_fun_params_labels = ['dim', 'power'], use_factors_grids = True)    
    
        inv_fun_params_ranges = {'power' : (1, 2), 'dim' : (0, dimensionality - 1)}
    
        custom_inv_fun_tokens = Custom_tokens(token_type = 'inverse', # Выбираем название для семейства токенов - обратных функций.
                                              token_labels = ['1/x_[dim]',], # Задаём названия токенов семейства в формате python-list'a.
                                                                         # Т.к. у нас всего один токен такого типа, задаём лист из 1 элемента
                                              evaluator = custom_inv_fun_evaluator, # Используем заранее заданный инициализированный объект для функции оценки токенов.
                                              params_ranges = inv_fun_params_ranges, # Используем заявленные диапазоны параметров
                                              params_equality_ranges = None) # Используем None, т.к. значения по умолчанию 
                                                                          # (равенство при лишь полном совпадении дискретных параметров)
                                                                          # нас устраивает.
                                                                         
        custom_grid_tokens = Cache_stored_tokens(token_type = 'grid',
                                           boundary = boundary,
                                           token_labels = ['t', 'r'],
                                           token_tensors={'t' : grids[0], 'r' : grids[1]},
                                           params_ranges = {'power' : (1, 1)},
                                           params_equality_ranges = None)
        
        epde_search_obj.set_moeadd_params(population_size=3, training_epochs = 5)
        
        epde_search_obj.fit(data = u, max_deriv_order=(1, 2), boundary=boundary, equation_terms_max_number = 3,
                            equation_factors_max_number = 2, deriv_method='poly', eq_sparsity_interval = (1e-6, 3.0),
                            deriv_method_kwargs = {'smooth': True, 'grid': grids, 'sigma' : 1}, coordinate_tensors = grids,
                            additional_tokens = [custom_grid_tokens, custom_inv_fun_tokens],
                            memory_for_cache=25, prune_domain = False)
    
        res = epde_search_obj.equation_search_results(level_num = 1)
        # solver_forms = []
        # for result in res:
            
