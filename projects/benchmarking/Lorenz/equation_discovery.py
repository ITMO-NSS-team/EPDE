#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 15 15:47:47 2022

@author: maslyaev
"""

import numpy as np
import torch
import matplotlib.pyplot as plt

import epde.interface.interface as epde_alg
from epde.interface.prepared_tokens import TrigonometricTokens
from epde.interface.solver_integration import BoundaryConditions, BOPElement

import os
import sys

SOLVER_STRATEGY = 'autograd'

sys.path.append('../')

sys.path.pop()
sys.path.append(os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..')))

def write_pareto(dict_of_exp):
    for key, item in dict_of_exp.items():
        test_key = str(key[0]).replace('.', '_') + '__' + str(key[1]).replace('.', '_')
        with open('/home/maslyaev/epde/EPDE_main/projects/hunter-prey/param_var/'+test_key+'.txt', 'w') as f:
            for iteration in range(len(item)):
                f.write(f'Iteration {iteration}\n\n')
                for ind in [pareto.text_form for pareto in item[iteration][0]]:
                    f.write(ind + '\n\n')

if __name__ == '__main__':
    '''
    Подгружаем данные, содержащие временные ряды динамики "вида-охотника" и "вида-жертвы"
    '''
    try:
        t_file = os.path.join(os.path.dirname( __file__ ), 'data/t_1.npy')
        t = np.load(t_file)
    except FileNotFoundError:
        t_file = '/home/maslyaev/epde/EPDE_main/projects/benchmarking/Lorenz/data/t_1.npy'
        t = np.load(t_file)
    
    try:
        data_file =  os.path.join(os.path.dirname( __file__ ), 'data/lorenz_1.npy')
        data = np.load(data_file)
    except FileNotFoundError:
        data_file = '/home/maslyaev/epde/EPDE_main/projects/benchmarking/Lorenz/data/lorenz_1.npy'
        data = np.load(data_file)
        
    end_train, end_test = 5000, 10000
    t_train, t_test = t[:end_train], t[end_train : end_test]
    
    u_train, u_test = data[:end_train, 0], data[:end_train, 0]
    v_train, v_test = data[:end_train, 1], data[:end_train, 1]
    w_train, w_test = data[:end_train, 2], data[:end_train, 2]
        
    dimensionality = u_train.ndim - 1
    
    '''
    Подбираем Парето-множество систем дифф. уравнений.
    '''
    epde_search_obj = epde_alg.EpdeSearch(use_solver = False, dimensionality = dimensionality, boundary = 10,
                                           coordinate_tensors = [t_train,])    
    epde_search_obj.set_preprocessor(default_preprocessor_type='poly', # use_smoothing = True
                                     preprocessor_kwargs={})
    popsize = 20
    epde_search_obj.set_moeadd_params(population_size = popsize, training_epochs=40)
    trig_tokens = TrigonometricTokens(dimensionality = dimensionality)
    factors_max_number = {'factors_num' : [1, 2], 'probas' : [0.8, 0.2]}
    
    epde_search_obj.fit(data=[u_train, v_train, w_train], variable_names=['u', 'v', 'w'], max_deriv_order=(1,),
                        equation_terms_max_number=5, data_fun_pow = 1, additional_tokens=[trig_tokens,], 
                        equation_factors_max_number=factors_max_number,
                        eq_sparsity_interval=(1e-10, 1e-4))
    '''
    Смотрим на найденное Парето-множество, 
    
    "идеальное уравнение" имеет вид:
     / -1.3048888807580532 * u{power: 1.0} * v{power: 1.0} + 0.3922851274813135 * u{power: 1.0} + -0.0003917278536547386 = du/dx1{power: 1.0}
     \ -0.9740492564964498 * v{power: 1.0} + 0.9717873909925121 * u{power: 1.0} * v{power: 1.0} + 0.0003500773115704403 = dv/dx1{power: 1.0}
    {'terms_number': {'optimizable': False, 'value': 3}, 'max_factors_in_term': {'optimizable': False, 'value': {'factors_num': [1, 2], 'probas': [0.8, 0.2]}}, 
     ('sparsity', 'u'): {'optimizable': True, 'value': 0.00027172388370453704}, ('sparsity', 'v'): {'optimizable': True, 'value': 0.00019292375116125682}} , with objective function values of [0.2800438  0.18041074 4.         4.        ]         
    '''
    epde_search_obj.equation_search_results(only_print = True, num = 1)
    
    res = epde_search_obj.equation_search_results(only_print = False, num = 1)
    eq = epde_search_obj.get_equations_by_complexity(complexity=[3., 5., 4.])[0]
    
    def get_ode_bop(key, var, grid_loc, value):
        bop = BOPElement(axis = 0, key = key, term = [None], power = 1, var = var)
        bop_grd_np = np.array([[grid_loc,]])
        bop.set_grid(torch.from_numpy(bop_grd_np).type(torch.FloatTensor))
        bop.values = torch.from_numpy(np.array([[value,]])).float()
        return bop
    
    bop_u = get_ode_bop('u', 0, t_test[0], u_test[0])
    bop_v = get_ode_bop('v', 0, t_test[0], v_test[0])
    bop_w = get_ode_bop('w', 0, t_test[0], w_test[0])
    
    pred_u_v_w = epde_search_obj.predict(system=eq, boundary_conditions=[bop_u(), bop_v(), bop_w()], 
                                        grid = [t_train,], strategy=SOLVER_STRATEGY)
    
    plt.plot(t_test, u_test, '+', color = 'b', label = 'u_data')
    plt.plot(t_test, v_test, '*', color = 'r', label = "v_data")
    plt.plot(t_test, w_test, '1', color = 'k', label = "w_data")
    plt.plot(t_test, pred_u_v_w[:, 0], color = 'b', label='u_NN')
    plt.plot(t_test, pred_u_v_w[:, 1], color = 'r', label='v_NN')
    plt.plot(t_test, pred_u_v_w[:, 2], color = 'k', label='v_NN')
    plt.xlabel('t')
    plt.ylabel('value')
    plt.grid()
    plt.show()
    # sys = res[0][1]
    # sys.text_form
    # pareto_frontiers[(err_factor, sigma)].append(res)
'''
Решаем уравнение (систему уравнений) при помощи метода .predict(...)
'''
# predictions = epde_search_obj.predict(system = sys, boundary_conditions=None)