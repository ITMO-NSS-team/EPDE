#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  1 12:04:19 2021

@author: maslyaev
"""

import numpy as np
from collections import OrderedDict
from sklearn.linear_model import LinearRegression

import epde.interface.interface as epde_alg

from epde.cache.cache import upload_complex_token
from epde.interface.prepared_tokens import Custom_tokens, Cache_stored_tokens
from epde.interface.equation_translator import translate_equation
from epde.evaluators import CustomEvaluator, simple_function_evaluator, inverse_function_evaluator

if __name__ == "__main__":
    t_max = 400
    
    file = np.loadtxt('/home/maslyaev/epde/EPDE/tests/cylinder/data/Data_32_points_.dat', 
                      delimiter=' ', usecols=range(33))

    x = np.linspace(0.5, 16, 32)
    t = file[:t_max, 0]    
    grids = np.meshgrid(t, x, indexing = 'ij')
    u = file[:t_max, 1:] 

    boundary = [10, 4]

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


    epde_search_obj.create_pool(data = u, max_deriv_order=(1, 2), boundary=(10, 8), coordinate_tensors = grids, 
                        additional_tokens = [custom_grid_tokens, custom_inv_fun_tokens], 
                        method='ANN', method_kwargs = {'epochs_max':2000}, 
                        memory_for_cache=5, prune_domain = True,
                        division_fractions = (int(u.shape[0] / 10.), int(u.shape[0] / 4.)))
    
    grid_cache, token_cache = epde_search_obj.cache

    upload_complex_token('1/x_[dim]', OrderedDict([('power', 1.0), ('dim', 1.0)]), custom_inv_fun_evaluator, 
                         token_cache, grid_cache)

    dudt = token_cache.memory_default[('du/dx1', (1.0,))]
    dudx = token_cache.memory_default[('du/dx2', (1.0,))]
    r_inv = token_cache.memory_default[('1/x_[dim]', (1.0, 1.0))]
    d2udx22 = token_cache.memory_default[('d^2u/dx2^2', (1.0,))]
    
    t1 = dudt.reshape(-1)
    t2 = (dudx*r_inv).reshape(-1)
    t3 = d2udx22.reshape(-1)
    features = np.vstack((t2, t3))
    
    lr = LinearRegression()
    lr.fit(features.T, t1)
    lr.coef_
    
    
    coeff = 
    equation_string = ('0.0 * du/dx2{power: 1} * du/dx2{power: 1}'
                       ' + 0.0 * d^3u/dx1^3{power: 1}'
                       ' + 0.0'
                       ' = d^2u/dx1^2{power: 1} * du/dx1{power: 1}')
    equation = 