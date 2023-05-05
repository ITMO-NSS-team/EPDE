#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  3 14:04:20 2023

@author: maslyaev
"""

import numpy as np

import torch
import os
import sys

sys.path.append('../')

sys.path.pop()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..')))

import matplotlib.pyplot as plt
import matplotlib

SMALL_SIZE = 12
matplotlib.rc('font', size=SMALL_SIZE)
matplotlib.rc('axes', titlesize=SMALL_SIZE)

from epde.preprocessing.interpolation_oversampling import BesselInterpolator

import epde.interface.interface as epde_alg
from epde.interface.prepared_tokens import TrigonometricTokens, CacheStoredTokens, GridTokens, CustomEvaluator, CustomTokens
from epde.interface.solver_integration import BoundaryConditions, BOPElement

def epde_discovery(grids, data, derivs, use_ann = False):
    multiobjective_mode = True
    dimensionality = data.ndim - 1
    
    epde_search_obj = epde_alg.EpdeSearch(multiobjective_mode=multiobjective_mode, use_solver = False, 
                                          dimensionality = dimensionality, boundary = 10,
                                          coordinate_tensors = grids)    
    # epde_search_obj.set_preprocessor(default_preprocessor_type='spectral', # use_smoothing = True
    #                                  preprocessor_kwargs={})
    popsize = 7
    if multiobjective_mode:
        epde_search_obj.set_moeadd_params(population_size = popsize, 
                                          training_epochs=25)
    else:
        epde_search_obj.set_singleobjective_params(population_size = popsize, 
                                                   training_epochs=25)
    # epde_search_obj.set_moeadd_params(population_size = popsize, training_epochs=50)
    print(dimensionality+1)
    grid_tokens = GridTokens(['t', 'x'], dimensionality = dimensionality)
    
    custom_inverse_eval_fun = lambda *grids, **kwargs: np.power(grids[int(kwargs['dim'])], - kwargs['power']) 
    custom_inv_fun_evaluator = CustomEvaluator(custom_inverse_eval_fun, 
                                               eval_fun_params_labels = ['dim', 'power'], use_factors_grids = True)    

    inv_fun_params_ranges = {'power' : (1, 2), 'dim' : (0, dimensionality - 1)}

    custom_inv_fun_tokens = CustomTokens(token_type = 'inverse', # Выбираем название для семейства токенов - обратных функций.
                                          token_labels = ['1/x_[dim]',], # Задаём названия токенов семейства в формате python-list'a.
                                                                      # Т.к. у нас всего один токен такого типа, задаём лист из 1 элемента
                                          evaluator = custom_inv_fun_evaluator, # Используем заранее заданный инициализированный объект для функции оценки токенов.
                                          params_ranges = inv_fun_params_ranges, # Используем заявленные диапазоны параметров
                                          params_equality_ranges = None) # Используем None, т.к. значения по умолчанию 
                                                                      # (равенство при лишь полном совпадении дискретных параметров)
                                                                      # нас устраивает.
    
    factors_max_number = {'factors_num' : [1, 2], 'probas' : [0.6, 0.4]}
    
    opt_val = 1e-1
    bounds = (1e-10, 1e0) if multiobjective_mode else (opt_val, opt_val)    
    epde_search_obj.fit(data=data, variable_names=['u',], max_deriv_order=(1, 2),
                        equation_terms_max_number=5, data_fun_pow = 1, additional_tokens=[grid_tokens, custom_inv_fun_tokens], 
                        equation_factors_max_number=factors_max_number,
                        eq_sparsity_interval=bounds)
    # sys = epde_search_obj.get_equations_by_complexity(4)
    return epde_search_obj

if __name__ == "__main__":
    filename = '/home/maslyaev/epde/EPDE_main/projects/heat/0_6_up_ex_My.dat'
    file = np.loadtxt(filename)
    
    idxs_x = (6, 9, 12, 15, 18, 21) 
    slice_start, slice_finish, slice_step = 1000, file.shape[0]-1000, 25
    
    idxs_t = slice(slice_start, slice_finish, slice_step)
    
    u_smol = file[idxs_t, idxs_x]
    dimensionality = file.ndim
    
    time = file[idxs_t, 0]
    x = np.array([0.6, 1.1, 1.6, 2.1, 2.6, 3.1]) * 1e-3

    oversampling_size = 30
    oversampling_x = np.linspace(x[0], x[-1], oversampling_size)  
    
    def oversampling_approx(x_init, x_new, row, order = 4):
        BI = BesselInterpolator(x_init, row, max_order = order)
        return BI.approximate(x_new)    
    
    # print(x, oversampling_x, u_smol.shape)
    u = np.vstack([oversampling_approx(x, oversampling_x, u_smol[idx, :]) 
                   for idx in range(u_smol.shape[0])])
    grids = np.meshgrid(time, oversampling_x, indexing = 'ij')
    
    search_res = epde_discovery(grids, u, None)