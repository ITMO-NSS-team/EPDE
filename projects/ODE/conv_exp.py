#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 16 14:00:20 2023

@author: maslyaev
"""

import numpy as np
import matplotlib.pyplot as plt

import epde.globals as global_var
import epde.interface.interface as epde_alg

from epde.interface.prepared_tokens import CustomTokens, TrigonometricTokens, CacheStoredTokens, GridTokens
from epde.evaluators import CustomEvaluator

if __name__ == "__main__":
    use_ann = False
    t = np.linspace(0, 4*np.pi, 1000)
    u = np.load('/home/maslyaev/epde/EPDE_main/projects/ODE/data/fill366.npy') # loading data with the solution of ODE
    # Trying to create population for mulit-objective optimization with only 
    # derivatives as allowed tokens. Here only one equation structure will be 
    # discovered, thus MOO algorithm will not be launched.
    
    dimensionality = t.ndim - 1
    
    multiobjective = False
    runs = 10
    histories = []
        
    for idx in range(runs):
        epde_search_obj = epde_alg.EpdeSearch(multiobjective_mode = multiobjective, use_solver = False, dimensionality = dimensionality,
                                              boundary = 30, coordinate_tensors = [t,])
        if use_ann:
            epde_search_obj.set_preprocessor(default_preprocessor_type='ANN', # use_smoothing = True poly
                                             preprocessor_kwargs={'epochs_max' : 50000})# 
        else:
            epde_search_obj.set_preprocessor(default_preprocessor_type='poly', # use_smoothing = True poly
                                             preprocessor_kwargs={'use_smoothing' : False, 'sigma' : 1, 'polynomial_window' : 3, 'poly_order' : 3}) # 'epochs_max' : 10000})# 
                                         # preprocessor_kwargs={'use_smoothing' : True, 'polynomial_window' : 3, 'poly_order' : 2, 'sigma' : 3})#'epochs_max' : 10000}) 'polynomial_window' : 3, 'poly_order' : 3
        popsize = 10
        if multiobjective:
            epde_search_obj.set_moeadd_params(population_size = popsize, training_epochs=55)
        else:
            epde_search_obj.set_singleobjective_params(population_size = popsize, training_epochs=150)
        trig_tokens = TrigonometricTokens(freq = (1 - 0.005, 1 + 0.005), 
                                          dimensionality = dimensionality)
        factors_max_number = {'factors_num' : [1, 2], 'probas' : [0.65, 0.35]}
    
        custom_grid_tokens = GridTokens(dimensionality = dimensionality)
        
        # custom_inverse_eval_fun = lambda *grids, **kwargs: np.power(grids[int(kwargs['dim'])], kwargs['power']) 
        # custom_inv_fun_evaluator = CustomEvaluator(custom_inverse_eval_fun, 
        #                                            eval_fun_params_labels = ['dim', 'power'], 
        #                                            use_factors_grids = True)    
    
        # grid_params_ranges = {'power' : (1, 2), 'dim' : (0, dimensionality)}
        
        # custom_grid_tokens = CustomTokens(token_type = 'grid', # Выбираем название для семейства токенов - обратных функций.
        #                                   token_labels = ['1/x_{dim}',], # Задаём названия токенов семейства в формате python-list'a.
        #                                                                  # Т.к. у нас всего один токен такого типа, задаём лист из 1 элемента
        #                                   evaluator = custom_inv_fun_evaluator, # Используем заранее заданный инициализированный объект для функции оценки токенов.
        #                                   params_ranges = grid_params_ranges, # Используем заявленные диапазоны параметров
        #                                   params_equality_ranges = None)    
        
        epde_search_obj.fit(data=[u,], variable_names=['u',], max_deriv_order=(1,),
                            equation_terms_max_number=5, data_fun_pow = 1, 
                            additional_tokens=[trig_tokens, custom_grid_tokens], #
                            equation_factors_max_number=factors_max_number,
                            eq_sparsity_interval=(5e-1, 5e-1))
    
        epde_search_obj.equations(only_print = True, num = 1)
        histories.append((global_var.history.history, epde_search_obj.equations(only_print = False, num = 1)[0]))

    ranges = np.arange(0, 120, 12)
    
    data = {}
    for r in ranges:
        data_slice = [h[0][r][1] for h in histories]
        data[r] = data_slice
        
    fig, ax = plt.subplots()
    ax.set_yscale('log')
    ax.grid(alpha = 0.5)
    ax.boxplot(data.values(), whis=[5, 95])
    ax.set_xticklabels(data.keys())
    plt.savefig('boxplot_ode_conv_log.png', dpi = 300, format = 'png', bbox_inches = 'tight')        