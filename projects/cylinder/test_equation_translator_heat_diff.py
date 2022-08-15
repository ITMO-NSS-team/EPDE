#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  2 13:10:30 2021

@author: maslyaev
"""

import os
import pandas as pd

import numpy as np
import epde.interface.interface as epde_alg
import pickle

from epde.interface.equation_translator import Coeff_less_equation
from epde.interface.prepared_tokens import Custom_tokens, Cache_stored_tokens
from epde.evaluators import CustomEvaluator, simple_function_evaluator, inverse_function_evaluator

from epde.prep.interpolation_oversampling import BesselInterpolator

def load_data(directory, data_file_format = 'dat', loader_kwargs = {}, 
              temp_initial = {}, voltage_prefix = '06'):
    data = []
    t = None
    x = 1e-4 * np.array(list(temp_initial.keys()))
    
    for filename in os.listdir(directory):
        print(filename)
        if filename.endswith(".dat"):
            temp = pd.read_csv(directory + '/' + filename, **loader_kwargs)
            if t is None:
                t = temp.loc[temp.tc >= 0].ts.to_numpy()
            dist = int(filename.replace(voltage_prefix + 'V_', '').replace('mm_u.dat', ''))
                  
            data.append((dist, temp.loc[temp.tc >= 0].Tr.to_numpy()))
    len_min = min([series.size for _, series in data])
    data = sorted([(dist, series[:len_min]) for dist, series in data], 
                  key = lambda x: x[0])
    
    return t, x, np.stack([series - temp_initial[dist] for dist, series in data])

if __name__ == '__main__':   
    temp_initial = {7 : 24.86, 
                    12 : 25.0307, 
                    17 : 25.15106, 
                    22 : 26.0285,
                    27 : 26.05125}
     
    t_min = 15000
    t_max = 25000
    directory = 'tests/cylinder/data/Diffusion'
    
    t, x, u_smol = load_data(os.getcwd() + '/' + directory, 
                           loader_kwargs={'skiprows':2, 
                                          'sep' : '\t', 
                                          'error_bad_lines' : False, 
                                          'names' : ["ts","tc","Tr","Tm"]},
                           temp_initial = temp_initial)
    u_smol = u_smol.T[t_min:t_max, :]; t = t[t_min:t_max]
    # file = np.loadtxt('/home/maslyaev/epde/EPDE_stable/tests/cylinder/data/Data_32_points_.dat', 
    #                   delimiter=' ', usecols=range(33))
    
    oversampling_size = 30
    # oversampling_x = np.linspace(x[0], x[-1], oversampling_size)  
    
    def oversampling_approx(x_init, x_new, row, order = 4):
        BI = BesselInterpolator(x_init, row, max_order = order)
        return BI.approximate(x_new)
    
    # u = np.vstack([oversampling_approx(x, oversampling_x, u_smol[idx, :]) 
    #                for idx in range(u_smol.shape[0])])
    u = u_smol
    oversampling_x = x
    # u = u.T
    
    
    # x = np.linspace(0.5, 16, 32)
    # t = file[:t_max, 0]
    grids = np.meshgrid(t, oversampling_x, indexing = 'ij')
    # u = file[:t_max, 1:]

    boundary = [10, 1]

    dimensionality = u.ndim
        
    epde_search_obj = epde_alg.epde_search()
    epde_search_obj.set_memory_properties(u, mem_for_cache_frac = 10)

    custom_inverse_eval_fun = lambda *grids, **kwargs: np.power(grids[int(kwargs['dim'])], - kwargs['power']) 
    custom_inv_fun_evaluator = CustomEvaluator(custom_inverse_eval_fun, eval_fun_params_labels = ['dim', 'power'], use_factors_grids = True)    

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


    epde_search_obj.create_pool(data = u, max_deriv_order=(1, 2), boundary=boundary, coordinate_tensors = grids, 
                        additional_tokens = [custom_grid_tokens, custom_inv_fun_tokens], 
                        method='ANN', method_kwargs = {'epochs_max' : 1600}, #{'smooth': False, 'grid': grids, 'sigma' : 2}, # #
                        memory_for_cache=5, prune_domain = True,
                        division_fractions = (int(u.shape[0] / 10.), int(u.shape[0] / 4.)))
    
    lp_terms = [['1/x_[dim]{power: 1, dim: 1}', 'du/dx2{power: 1}'], ['d^2u/dx2^2{power: 1}']]
    rp_term = ['du/dx1{power: 1}',]
    test = Coeff_less_equation(lp_terms, rp_term, epde_search_obj.pool)
    
    def map_to_equation(equation, function, function_kwargs = dict()):
        _, target, features = equation.evaluate(normalize = False, return_val = False)
        return function(np.abs(np.dot(features, equation.weights_final[:-1]) + 
                              np.full(target.shape, equation.weights_final[-1]) - 
                              target), **function_kwargs)    

    print(test.equation.text_form)
    print(map_to_equation(test.equation, np.mean))
    
    # np.save('/home/maslyaev/epde/EPDE_stable/tests/diffusion.npy', )