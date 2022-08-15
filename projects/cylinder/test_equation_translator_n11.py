#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct  7 16:40:29 2021

@author: maslyaev
"""

import numpy as np
import epde.interface.interface as epde_alg
import pandas as pd

from epde.interface.equation_translator import Coeff_less_equation
from epde.interface.prepared_tokens import Custom_tokens, Cache_stored_tokens
from epde.evaluators import CustomEvaluator, simple_function_evaluator, inverse_function_evaluator

if __name__ == '__main__':
    t_max = 400
    
    # temperature_experimental = pd.read_excel('tests/cylinder/data/Temperature_filtered.xlsx')
    # # file = np.loadtxt('/home/maslyaev/epde/EPDE/tests/cylinder/data/Data_32_points_.dat', 
    # #                   delimiter=' ', usecols=range(33))
    
    # temp_np = temperature_experimental.to_numpy()
    # x = np.linspace(1., 10., 5)
    # t = temp_np[:t_max, 0]
    # grids = np.meshgrid(t, x, indexing = 'ij')
    # u = temp_np[:t_max, 1::2]

    file = np.loadtxt('/home/maslyaev/epde/EPDE_stable/tests/cylinder/data/Noise_1/Data_20_points_0.dat', 
                  delimiter=' ', skiprows = 3, usecols=range(21))

    x = np.linspace(0.5, 10, 20)
    t = file[:t_max, 0]
    grids = np.meshgrid(t, x, indexing = 'ij')
    u = file[:t_max, 1:]

    boundary = 1
    
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
                        method='ANN', method_kwargs = {'epochs_max' : 20000}, 
                        memory_for_cache=5, prune_domain = True,
                        division_fractions = (int(u.shape[0] / 10.), int(u.shape[0] / 4.)))
    
    lp_terms = [['1/x_[dim]{power: 1, dim: 1}', 'du/dx2{power: 1}'], ['d^2u/dx2^2{power: 1}']]
    rp_term = ['du/dx1{power: 1}',]
    test = Coeff_less_equation(lp_terms, rp_term, epde_search_obj.pool)
    
    def map_to_equation(equation, function, function_kwargs = dict()):
        _, target, features = equation.evaluate(normalize = False, return_val = False)
        return function(np.dot(features, equation.weights_final[:-1]) + 
                              np.full(target.shape, equation.weights_final[-1]) - 
                              target, **function_kwargs)    
    
    print(test.equation.text_form)
    print(map_to_equation(test.equation, np.mean))
    
    