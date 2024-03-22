#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 21 17:36:43 2021

@author: mike_ubuntu
"""

import numpy as np
from collections import OrderedDict
import os
import sys
import getopt

global opt, args
#opt, args = getopt.getopt(sys.argv[2:], '', ['path='])
#
#sys.path.append(opt[0][1])

import epde.globals as global_var

from epde.moeadd.moeadd import *
from epde.moeadd.moeadd_supplementary import *

import epde.operators.sys_search_operators as operators
#from epde.src.evo_optimizer import Operator_director
from epde.evaluators import simple_function_evaluator, trigonometric_evaluator
from epde.supplementary import Define_Derivatives
from epde.cache.cache import upload_simple_tokens, upload_grids, prepare_var_tensor
from epde.prep.derivatives import Preprocess_derivatives
from epde.interface.token_family import TF_Pool, Token_family

from epde.eq_search_strategy import Strategy_director, Strategy_director_solver
from epde.operators.ea_stop_criteria import Iteration_limit

#if __name__ == '__main__':
def test_ode_auto():
    '''
    
    В этой задаче мы ищем уравнение u sin(x) + u' cos(x) = 1 по его решению: u = sin(x) + C cos(x), 
    где у частного решения C = 1.3.
    
    Задаём x - координатную ось по времени; ts - временной ряд условных измерений
    ff_filename - имя файла, куда сохраняется временной ряд; output_file_name - имя файла для производных
    step - шаг по времени
    '''
    
#    delim = '/' if sys.platform == 'linux' else '\\'
    
    x = np.linspace(0, 4*np.pi, 1000)
    print('path:', sys.path)
    ts = np.load('/media/mike_ubuntu/DATA/EPDE_publication/tests/system/Test_data/fill366.npy') # tests/system/
    new_derivs = True
    
    ff_filename = '/media/mike_ubuntu/DATA/EPDE_publication/tests/system/Test_data/smoothed_ts.npy'
    output_file_name = '/media/mike_ubuntu/DATA/EPDE_publication/tests/system/Test_data/derivs.npy'
    step = x[1] - x[0]
    
    '''

    '''
    
    max_order = 1 # presence of the 2nd order derivatives leads to equality u = d^2u/dx^2 on this data (elaborate)
    
    if new_derivs:
        _, derivs = Preprocess_derivatives(ts, data_name = ff_filename, 
                                output_file_name = output_file_name,
                                steps = (step,), smooth = False, sigma = 1, max_order = max_order)
        ts_smoothed = np.load(ff_filename)        
    else:
        try:
            ts_smoothed = np.load(ff_filename)
            derivs = np.load(output_file_name)
        except FileNotFoundError:
            _, derivs = Preprocess_derivatives(ts, data_name = ff_filename, 
                                    output_file_name = output_file_name,
                                    steps = (step,), smooth = False, sigma = 1, max_order = max_order)            
            ts_smoothed = np.load(ff_filename) 
            
    global_var.init_caches(set_grids=True)
    global_var.tensor_cache.memory_usage_properties(obj_test_case=ts, mem_for_cache_frac = 5)  
    global_var.grid_cache.memory_usage_properties(obj_test_case=x, mem_for_cache_frac = 5)

    print(type(derivs))

    upload_grids(x, global_var.grid_cache)   
    u_derivs_stacked = prepare_var_tensor(ts_smoothed, derivs, time_axis = 0)
    
    u_names, u_deriv_orders = Define_Derivatives('u', 1, 1) 
    u_names = u_names; u_deriv_orders = u_deriv_orders 
    upload_simple_tokens(u_names, global_var.tensor_cache, u_derivs_stacked)
    
    u_tokens = Token_family('Function', family_of_derivs = True)
    u_tokens.set_status(unique_specific_token=False, unique_token_type=False, s_and_d_merged = False, 
                        meaningful = True, unique_for_right_part = False)
    u_token_params = OrderedDict([('power', (1, 1))])
    u_equal_params = {'power' : 0}
    u_tokens.set_params(u_names, u_token_params, u_equal_params, u_deriv_orders)
    u_tokens.set_evaluator(simple_function_evaluator, [])

    grid_names = ['t',]    
    grid_tokens = Token_family('Grids')
    grid_tokens.set_status(unique_specific_token=True, unique_token_type=True, s_and_d_merged = False, 
                        meaningful = False, unique_for_right_part = False)
    grid_token_params = OrderedDict([('power', (1, 1))])
    grid_equal_params = {'power' : 0}
    grid_tokens.set_params(grid_names, grid_token_params, grid_equal_params)
    grid_tokens.set_evaluator(simple_function_evaluator, [])
#    
    trig_tokens = Token_family('Trigonometric')
    trig_names = ['sin', 'cos']
    trig_tokens.set_status(unique_specific_token=True, unique_token_type=True, 
                           meaningful = False, unique_for_right_part = False)
    trig_token_params = OrderedDict([('power', (1, 1)), ('freq', (0.95, 1.05)), ('dim', (0, 0))])
    trig_equal_params = {'power' : 0, 'freq' : 0.05, 'dim' : 0}
    trig_tokens.set_params(trig_names, trig_token_params, trig_equal_params)
    trig_tokens.set_evaluator(trigonometric_evaluator, [])

    upload_simple_tokens(grid_names, global_var.tensor_cache, [x,])    
    global_var.tensor_cache.use_structural()

    pool = TF_Pool([u_tokens, trig_tokens]) # grid_tokens, 
    pool.families_cardinality()
    
    '''
    Используем базовый эволюционный оператор.
    '''
#    test_strat = Strategy_director(Iteration_limit, {'limit' : 300})
    test_strat = Strategy_director_solver(Iteration_limit, {'limit' : 50})
    test_strat.strategy_assembly()
    
#    test_system = SoEq(pool = pool, terms_number = 4, max_factors_in_term=2, sparcity = (0.1,))
#    test_system.set_eq_search_evolutionary(director.constructor.operator)
#    test_system.create(population_size=16, eq_search_iters=300)    
    
#    tokens=[h_tokens, trig_tokens]
    '''
    Настраиваем генератор новых уравнений, которые будут составлять популяцию для 
    алгоритма многокритериальной оптимизации.
    '''
    pop_constructor = operators.systems_population_constructor(pool = pool, terms_number=6, 
                                                               max_factors_in_term=2, eq_search_evo=test_strat.constructor.strategy,
                                                               sparcity_interval = (0.0, 0.5))
    
    '''
    Задаём объект многокритериального оптимизатора, эволюционный оператор и задаём лучшие возможные 
    значения целевых функций.
    '''
    optimizer = moeadd_optimizer(pop_constructor, 3, 3, delta = 1/50., neighbors_number = 3, solution_params = {})
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
        [print(solution.structure[0].text_form, solution.evaluate())  for solution in optimizer.pareto_levels.levels[idx]]
    
    raise NotImplementedError
    '''
    В результате мы должны получить фронт Парето, который должен включать в себя одно уравнение с 
    "0-ём слагаемых в левой части", т.к. равенство какого-то токена константе (скорее всего, 0), а также 
    1 уравнение с "1им слагаемым помимо константы и правой части из одного слагаемого", которое будет либо исходным 
    (т.е. искомым уравнением, либо уравнением u cos(x) - 1.3 = u' sin(x), которое имеет 
    частное решение, совпадающее с рассматриваемым частным решением исходного уравнения.)
    '''
    