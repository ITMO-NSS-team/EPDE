#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 17:11:03 2023

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

import epde.interface.interface as epde_alg
from epde.interface.prepared_tokens import TrigonometricTokens, CustomEvaluator, CustomTokens, GridTokens
from epde.interface.solver_integration import BoundaryConditions, BOPElement


def epde_discovery(t, x, use_ann = False, derivs = None):
    dimensionality = x.ndim - 1
    

    epde_search_obj = epde_alg.EpdeSearch(use_solver = False, dimensionality = dimensionality, boundary = 80,
                                           coordinate_tensors = [t,])
    if use_ann:
        epde_search_obj.set_preprocessor(default_preprocessor_type='ANN', # use_smoothing = True poly
                                         preprocessor_kwargs={'epochs_max' : 10000})# 
    else:
        epde_search_obj.set_preprocessor(default_preprocessor_type='poly', # use_smoothing = True poly
                                         preprocessor_kwargs={'use_smoothing' : True, 'sigma' : 2, 'polynomial_window' : 3, 'poly_order' : 3}) # 'epochs_max' : 10000})# 
                                     # preprocessor_kwargs={'use_smoothing' : True, 'polynomial_window' : 3, 'poly_order' : 2, 'sigma' : 3})#'epochs_max' : 10000}) 'polynomial_window' : 3, 'poly_order' : 3
    popsize = 8
    epde_search_obj.set_moeadd_params(population_size = popsize, training_epochs=55)
    trig_tokens = TrigonometricTokens(freq = (2 - 0.0000005, 2 + 0.0000005), 
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
    
    epde_search_obj.fit(data=[x,], variable_names=['u',], max_deriv_order=(2,),
                        equation_terms_max_number=4, data_fun_pow = 1, derivs=derivs,
                        additional_tokens=[trig_tokens, custom_grid_tokens], 
                        equation_factors_max_number=factors_max_number,
                        eq_sparsity_interval=(1e-12, 1e-4))

    epde_search_obj.equations(only_print = True, num = 1)
    
    # syss = epde_search_obj.equations(only_print = False, num = 1) 
    '''
    Having insight about the initial ODE structure, we are extracting the equation with complexity of 5
    
    In other cases, you should call sys.equations(only_print = True),
    where the algorithm presents Pareto frontier of optimal equations.
    '''
    # sys = epde_search_obj.get_equations_by_complexity(5)[0]
    return epde_search_obj, epde_search_obj.saved_derivaties

if __name__ == "__main__":
    # try:
    #     t_file = os.path.join(os.path.dirname( __file__ ), 'projects/ODE/data/ODE_t.npy')
    #     t = np.load(t_file)
    # except FileNotFoundError:
    #     t_file = '/home/maslyaev/epde/EPDE_main/projects/ODE/data/ODE_t.npy'
    #     t = np.load(t_file)
        
    
    try:
        data_file =  os.path.join(os.path.dirname( __file__ ), 'data/85_15.csv')
        data = np.genfromtxt(data_file)
    except FileNotFoundError:
        data_file = '/home/maslyaev/epde/EPDE_main/projects/misc/data/85_15.csv'
        data = np.genfromtxt(data_file)
        
    t=np.arange(len(data))
    t_max = data.size-1
    t_train = t[:t_max]; t_test = t[t_max:] 
    x_train = data[:t_max]; x_test = data[t_max:]

    magnitude = 0.5*1e-2
    x_n = x_train #+ np.random.normal(scale = magnitude*x, size = x.shape)
    plt.plot(t_train, x_n)
    plt.show()
    epde = True
    
    if epde:
        epde_search_obj = epde_discovery(t_train, x_train, False)
        
        # def get_ode_bop(key, var, term, grid_loc, value):
        #     bop = BOPElement(axis = 0, key = key, term = term, power = 1, var = var)
        #     bop_grd_np = np.array([[grid_loc,]])
        #     bop.set_grid(torch.from_numpy(bop_grd_np).type(torch.FloatTensor))
        #     bop.values = torch.from_numpy(np.array([[value,]])).float()
        #     return bop
            
        
        # bop_u = get_ode_bop('u', 0, [None], t_test[0], x_test[0])
        # bop_du = get_ode_bop('dudt', 0, [0,], t_test[0], (x_test[1] - x_train[-1])/(2*(t_test[1] - t_test[0])))        
        
        # pred_u = epde_search_obj.predict(system=sys, boundary_conditions=[bop_u(), bop_du()], 
        #                                   grid = [t_test,], strategy='autograd')
        
        # plt.plot(t_test, x_test, '+', label = 'test data')
        # plt.plot(t_test, pred_u, color = 'r', label='solution of the discovered ODE')
        # plt.grid()
        # plt.legend(loc='upper right')       
        # plt.show()