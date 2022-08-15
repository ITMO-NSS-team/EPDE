#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 15:43:29 2021

@author: mike_ubuntu
"""

import numpy as np
from collections import OrderedDict
import os
import sys
import getopt

global opt, args
#opt, args = getopt.getopt(sys.argv[2:], '', ['path='])

#sys.path.append(opt[0][1])

import epde.globals as global_var

from epde.moeadd.moeadd import *
from epde.moeadd.moeadd_supplementary import *

from epde.prep.DomainPruning import Domain_Pruner

import epde.operators.sys_search_operators as operators
#from epde.src.evo_optimizer import Operator_director
from epde.evaluators import simple_function_evaluator, trigonometric_evaluator, inverse_function_evaluator
from epde.supplementary import Define_Derivatives
from epde.cache.cache import upload_simple_tokens, upload_grids, prepare_var_tensor, np_ndarray_section
from epde.prep.derivatives import Preprocess_derivatives
from epde.interface.token_family import TF_Pool, TokenFamily

from epde.eq_search_strategy import Strategy_director, Strategy_director_solver
from epde.operators.ea_stop_criteria import Iteration_limit

if __name__ == "__main__":
    x = np.linspace(0.5, 5, 10)
    file = np.loadtxt('/home/maslyaev/epde/EPDE/tests/cylinder/data/Data_32_points_.dat', 
                      delimiter=' ', usecols=range(33))
    t_max = 3000
    t = file[:t_max, 0]
    T_vals = file[:t_max, 1:11] 
    grids = np.meshgrid(t, x, indexing = 'ij')
    
    
    # ff_filename = '/media/mike_ubuntu/DATA/EPDE_publication/tests/cylinder/data/grids_saved.npy'
    # output_file_name = '/media/mike_ubuntu/DATA/EPDE_publication/tests/cylinder/data/derivs.npy'
    
    max_order = 3
    
    poly_kwargs = {'grid' : grids, 'smooth' : False, 'max_order' : max_order, 
                   'polynomial_window' : 6, 'poly_order' : 5}
    derivs = Preprocess_derivatives(T_vals, method = 'poly', method_kwargs = poly_kwargs)
    
    derivatives = derivs[1].reshape((T_vals.shape[0], T_vals.shape[1], -1))
    
    global_var.init_globals(set_grids=True)
    global_var.tensor_cache.memory_usage_properties(obj_test_case = grids[0], mem_for_cache_frac = 5)  
    global_var.grid_cache.memory_usage_properties(obj_test_case = grids[0], mem_for_cache_frac = 5)    
    global_var.set_time_axis(0)
    
    boundary = [10, 5]
    grids = list(map(lambda idx: np_ndarray_section(grids[idx], boundary), range(len(grids))))
    upload_grids(grids, global_var.grid_cache)   
    u_derivs_stacked = prepare_var_tensor(T_vals, derivs[1], time_axis = 0, boundary = boundary)

    u_names, u_deriv_orders = Define_Derivatives('u', grids[0].ndim, max_order) 
    u_names = u_names; u_deriv_orders = u_deriv_orders 

    u_tokens = TokenFamily('Function', family_of_derivs = True)
    u_tokens.set_status(unique_specific_token=False, unique_token_type=False, s_and_d_merged = False, 
                        meaningful = True, unique_for_right_part = True)
    u_token_params = OrderedDict([('power', (1, 1))])
    u_equal_params = {'power' : 0}
    u_tokens.set_params(u_names, u_token_params, u_equal_params, u_deriv_orders)
    u_tokens.set_evaluator(simple_function_evaluator, [])
    upload_simple_tokens(u_names, global_var.tensor_cache, u_derivs_stacked)

    grid_names = ['t', 'r']    
    grid_tokens = TokenFamily('Grids')
    grid_tokens.set_status(unique_specific_token=True, unique_token_type=True, s_and_d_merged = False, 
                        meaningful = False, unique_for_right_part = False)
    grid_token_params = OrderedDict([('power', (1, 1))])
    grid_equal_params = {'power' : 0}
    grid_tokens.set_params(grid_names, grid_token_params, grid_equal_params)
    grid_tokens.set_evaluator(simple_function_evaluator, [])
    upload_simple_tokens(grid_names, global_var.tensor_cache, grids)    
    
    inv_grid_tokens = TokenFamily('Inverse')
    inv_grid_names = ['1/x_dim']
    inv_grid_tokens.set_status(unique_specific_token=True, unique_token_type=True, s_and_d_merged = False, 
                           meaningful = False, unique_for_right_part = False)
    inv_grid_token_params = OrderedDict([('power', (1, 2)), ('dim', (0, 1))])
    inv_grid_equal_params = {'power' : 0, 'dim' : 0}
    inv_grid_tokens.set_params(inv_grid_names, inv_grid_token_params, inv_grid_equal_params)
    inv_grid_tokens.set_evaluator(inverse_function_evaluator, [])    
    
    
    global_var.tensor_cache.use_structural()
    pool = TF_Pool([u_tokens, grid_tokens, inv_grid_tokens]) # grid_tokens, 
    print('cardinality', pool.families_cardinality())  
    
    test_strat = Strategy_director(Iteration_limit, {'limit' : 500})
    test_strat.strategy_assembly()    
    
    pop_constructor = operators.Systems_population_constructor(pool = pool, terms_number=6, 
                                                               max_factors_in_term=2, eq_search_evo=test_strat.constructor.strategy,
                                                               sparsity_interval = (0.0, 0.5))
    
    optimizer = moeadd_optimizer(pop_constructor, 4, 4, delta = 1/50., neighbors_number = 3, solution_params = {})
    evo_operator = operators.sys_search_evolutionary_operator(operators.mixing_xover, 
                                                              operators.gaussian_mutation)

    optimizer.set_evolutionary(operator=evo_operator)
    best_obj = np.concatenate((np.ones([1,]), 
                              np.zeros(shape=len([1 for token_family in pool.families if token_family.status['meaningful']]))))  
    optimizer.pass_best_objectives(*best_obj)
    
    def simple_selector(sorted_neighbors, number_of_neighbors = 4):
        return sorted_neighbors[:number_of_neighbors]

    '''
    Запускаем оптимизацию
    '''
    
    optimizer.optimize(simple_selector, 0.95, (4,), 100, 0.75) # Простая форма искомого уравнения найдется и за 10 итераций           
    
    for idx in range(len(optimizer.pareto_levels.levels)):
        print('\n')
        print(f'{idx}-th non-dominated level')
        [print(solution.structure[0].text_form, solution.evaluate(normalize = False))  for solution in optimizer.pareto_levels.levels[idx]]
        
