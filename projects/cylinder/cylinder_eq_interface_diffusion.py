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

from epde.prep.interpolation_oversampling import BesselInterpolator

def load_data(directory, data_file_format = 'dat', loader_kwargs = {}):
    temp_initial = {7 : 24.86, 
                    12 : 25.0307, 
                    17 : 25.15106, 
                    22 : 26.0285,
                    27 : 26.05125}
    data = []
    t = None
    x = 1e-4 * np.array(list(temp_initial.keys()))
    
    for filename in os.listdir(directory):
        print(filename)
        if filename.endswith(".dat"):
            temp = pd.read_csv(directory + '/' + filename, **loader_kwargs)
            if t is None:
                t = temp.loc[temp.tc >= 0].ts.to_numpy()
            dist = int(filename.replace('06V_', '').replace('mm_u.dat', ''))
                  
            data.append((dist, temp.loc[temp.tc >= 0].Tr.to_numpy()))
    len_min = min([series.size for _, series in data])
    data = sorted([(dist, series[:len_min]) for dist, series in data], 
                  key = lambda x: x[0])
    
    return t, x, np.stack([series - temp_initial[dist] for dist, series in data])
    

if __name__ == "__main__":
    test_iter_limit = 1
    for test_idx in np.arange(test_iter_limit):
        
        t_min = 10000
        t_max = 20000
        directory = 'tests/cylinder/data/Diffusion'
        
        t, x, u_smol = load_data(os.getcwd() + '/' + directory, 
                               loader_kwargs={'skiprows':2, 
                                              'sep' : '\t', 
                                              'error_bad_lines' : False, 
                                              'names' : ["ts","tc","Tr","Tm"]})
        u_smol = u_smol.T[t_min:t_max, :]; t = t[t_min:t_max]
        
        oversampling_size = 30
        oversampling_x = np.linspace(x[0], x[-1], oversampling_size)  
        
        def oversampling_approx(x_init, x_new, row, order = 4):
            BI = BesselInterpolator(x_init, row, max_order = order)
            return BI.approximate(x_new)
        
        u = np.vstack([oversampling_approx(x, oversampling_x, u_smol[idx, :]) 
                       for idx in range(u_smol.shape[0])])
        # u = u.T
        
        
        grids = np.meshgrid(t, oversampling_x, indexing = 'ij')
    
        boundary = [10, 4]
    
        dimensionality = u.ndim
        
        epde_search_obj = epde_alg.epde_search(use_solver=False, eq_search_iter = 100, dimensionality=2)
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
        
        epde_search_obj.fit(data = u, max_deriv_order=(1, 2), boundary=boundary, equation_terms_max_number = 4,
                            equation_factors_max_number = 2, deriv_method='ANN', eq_sparsity_interval = (1e-6, 3.0),
                            deriv_method_kwargs = {'epochs_max':300}, coordinate_tensors = grids,
                            additional_tokens = [custom_grid_tokens, custom_inv_fun_tokens],
                            memory_for_cache=25, prune_domain = True,
                            division_fractions = (int(u.shape[0] / 10.), int(u.shape[0] / 4.)))
    
        res = epde_search_obj.equation_search_results(level_num = 1)

            