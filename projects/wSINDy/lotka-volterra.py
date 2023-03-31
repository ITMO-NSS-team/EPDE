#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 12:53:27 2023

@author: maslyaev
"""

import numpy as np

import epde.interface.interface as epde_alg
from epde.interface.prepared_tokens import TrigonometricTokens
from epde.interface.solver_integration import BoundaryConditions, BOPElement

import torch
import os
import sys

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
        t_file = os.path.join(os.path.dirname( __file__ ), 'projects/hunter-prey/t_20.npy')
        t = np.load(t_file)
    except FileNotFoundError:
        t_file = '/home/maslyaev/epde/EPDE_main/projects/hunter-prey/t_20.npy'
        t = np.load(t_file)
    
    try:
        data_file =  os.path.join(os.path.dirname( __file__ ), 'projects/hunter-prey/data_20.npy')
        data = np.load(data_file)
    except FileNotFoundError:
        data_file = '/home/maslyaev/epde/EPDE_main/projects/hunter-prey/data_20.npy'
        data = np.load(data_file)
        
    t_max = 150
    t_train = t[:t_max]; t_test = t[t_max:] 
    x = data[:t_max, 0]; x_test = data[t_max:, 0]
    y = data[:t_max, 1]; y_test = data[t_max:, 1]
        
    dimensionality = x.ndim - 1
    
    '''
    Подбираем Парето-множество систем дифф. уравнений.
    '''
    epde_search_obj = epde_alg.EpdeSearch(use_solver = False, dimensionality = dimensionality, boundary = 30,
                                           coordinate_tensors = [t_train,])    
    epde_search_obj.set_preprocessor(default_preprocessor_type='poly', # use_smoothing = True
                                     preprocessor_kwargs={'use_smoothing' : False, 'polynomial_window' : 3, 'poly_order' : 2})#'epochs_max' : 10000}) 'polynomial_window' : 3, 'poly_order' : 3
    popsize = 12
    epde_search_obj.set_moeadd_params(population_size = popsize, training_epochs=25)
    trig_tokens = TrigonometricTokens(dimensionality = dimensionality)
    factors_max_number = {'factors_num' : [1, 2], 'probas' : [0.8, 0.2]}
    
    epde_search_obj.fit(data=[x, y], variable_names=['u', 'v'], max_deriv_order=(1,),
                        equation_terms_max_number=3, data_fun_pow = 1, additional_tokens=[trig_tokens,], 
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
    
    # res = epde_search_obj.equation_search_results(only_print = False, num = 1)
    
    sys = epde_search_obj.get_equations_by_complexity([4, 4])
    
    def get_ode_bop(key, var, grid_loc, value):
        bop = BOPElement(axis = 0, key = key, term = [None], power = 1, var = var)
        bop_grd_np = np.array([[grid_loc,]])
        bop.set_grid(torch.from_numpy(bop_grd_np).type(torch.FloatTensor))
        bop.values = torch.from_numpy(np.array([[value,]])).float()
        
    # bop_x = BOPElement(axis = 0, key = 'u', term = [None], power = 1, var = 0)
    # bop_y = BOPElement(axis = 0, key = 'v', term = [None], power = 1, var = 1)
    
    # bop_grd_np = np.array([[t_test[0],]])
    # bop_x.set_grid(torch.from_numpy(bop_grd_np).type(torch.FloatTensor))    
    # bop_y.set_grid(torch.from_numpy(bop_grd_np).type(torch.FloatTensor))
    
    # bop_x.values = torch.from_numpy(np.array([[x_test[0],]])).float()
    # bop_y.values = torch.from_numpy(np.array([[y_test[0],]])).float()
    
    bop_x = get_ode_bop('u', 0, t_test[0], x_test[0])
    bop_y = get_ode_bop('v', 1, t_test[0], y_test[0])
    
    
    pred_u_v = epde_search_obj.predict(system=sys[0], boundary_conditions=[bop_x(), bop_y()], 
                                        grid = [t_test,])
