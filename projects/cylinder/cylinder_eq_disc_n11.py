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

    file = np.loadtxt('/home/maslyaev/epde/EPDE_stable/tests/cylinder/data/Noise_1/Data_20_points_07.dat', 
                  delimiter=' ', skiprows = 3, usecols=range(21))

    x = np.linspace(0.5, 10, 20)
    t = file[:t_max, 0]
    grids = np.meshgrid(t, x, indexing = 'ij')
    u = file[:t_max, 1:]

    boundary = [10, 4]

    dimensionality = u.ndim
    
    # train_results = []
    
    # for train_idx in np.arange(10):
    epde_search_obj = epde_alg.epde_search(use_solver=False, eq_search_iter = 100, dimensionality=2,
                                           ) # verbose_params={'show_moeadd_epochs' : True}
    epde_search_obj.set_memory_properties(u, mem_for_cache_frac = 20)

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
    
    epde_search_obj.set_moeadd_params(population_size=3, training_epochs = 5)
    
    epde_search_obj.fit(data = u, max_deriv_order=(1, 2), boundary=boundary, equation_terms_max_number = 4,
                        equation_factors_max_number = 2, deriv_method='ANN', eq_sparsity_interval = (1e-6, 3.0), #'smooth' : True, 'sigma' : 5
                        deriv_method_kwargs = {'epochs_max':2500}, coordinate_tensors = grids, 
                        additional_tokens = [custom_grid_tokens, custom_inv_fun_tokens], 
                        memory_for_cache=25, prune_domain = True,
                        division_fractions = (int(u.shape[0] / 10.), int(u.shape[0] / 2.)))
    
    epde_search_obj.equation_search_results(only_print = True, level_num = 1)    