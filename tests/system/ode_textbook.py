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
opt, args = getopt.getopt(sys.argv[2:], '', ['path='])

sys.path.append(opt[0][1])

import epde.src.globals as global_var

from epde.src.moeadd.moeadd import *
from epde.src.moeadd.moeadd_supplementary import *

import epde.src.sys_search_operators as operators
#from epde.src.evo_optimizer import Operator_director
from epde.src.evaluators import simple_function_evaluator, trigonometric_evaluator
from epde.src.supplementary import Define_Derivatives
from epde.src.cache.cache import upload_simple_tokens, upload_grids, prepare_var_tensor
from epde.prep.derivatives import Preprocess_derivatives
from epde.src.token_family import TF_Pool, Token_family

from epde.src.eq_search_strategy import Strategy_director
from epde.src.operators.ea_stop_criteria import Iteration_limit

#if __name__ == '__main__':
def test_ode_auto():
    '''
    
    В этой задаче мы ищем уравнение u sin(x) + u' cos(x) = 1 по его решению: u = sin(x) + C cos(x), 
    где у частного решения C = 1.3.
    
    Задаём x - координатную ось по времени; ts - временной ряд условных измерений
    ff_filename - имя файла, куда сохраняется временной ряд; output_file_name - имя файла для производных
    step - шаг по времени
    '''
    
    delim = '/' if sys.platform == 'linux' else '\\'
    
    x = np.linspace(0, 4*np.pi, 1000)
    ts = np.load('tests/system/Test_data' + delim + 'fill366.npy')
    new_derivs = False
    
    ff_filename = 'tests/system/Test_data' + delim + 'smoothed_ts.npy'
    output_file_name = 'tests/system/Test_data' + delim + 'derivs.npy'
    step = x[1] - x[0]
    
    '''

    Рекомендуемый максимальный порядок производных в этой задаче - 1ый, т.к. в данном случае u = - u'', 
    и в силу простоты структуры, алгоритм в больше случаев обнаруживает её, а не исходное уравнение.
    В следующем фрагменте - пример вычисления производных при помощи метода Preprocess_derivatives(...),
    который вызывается, если булева переменная new_derivs == True, т.е. указано пересчитать производные.
    
    '''
    
    max_order = 1 # presence of the 2nd order derivatives leads to equality u = d^2u/dx^2 on this data (elaborate)
    
    if new_derivs:
        _, derivs = Preprocess_derivatives(ts, ff_name = ff_filename, 
                                output_file_name = output_file_name,
                                steps = (step,), smooth = True, sigma = 1, max_order = max_order)
        ts_smoothed = np.load(ff_filename)        
    else:
        try:
            ts_smoothed = np.load(ff_filename)
            derivs = np.load(output_file_name)
        except FileNotFoundError:
            _, derivs = Preprocess_derivatives(ts, ff_name = ff_filename, 
                                    output_file_name = output_file_name,
                                    steps = (step,), smooth = True, sigma = 1, max_order = max_order)            
            ts_smoothed = np.load(ff_filename)
#    print(derivs.shape)
    
    '''
    Инициализируем кэш для хранения вычисленных векторов слагаемых, чтобы не пересчитывать их каждый 
    выпуск, и не хранить в отдельных слагаемых, создавая возможные повторные вычисления.
    
    global_var - модуль с глобальными переменными; у него метод init_caches() - создаёт кэши
    global_var.tensor_cache - кэш со значениями множителей и слагаемых;
    global_var.grid_cache - кэш, хранящий в себе тензоры значений координат в узлах.
    
    Метод .memory_usage_properties задаёт свойства использования кэшем памяти.
    
    '''
    
    global_var.init_caches(set_grids=True)
    global_var.tensor_cache.memory_usage_properties(obj_test_case=ts, mem_for_cache_frac = 25)  
    global_var.grid_cache.memory_usage_properties(obj_test_case=x, mem_for_cache_frac = 5)

    '''
    Задаём пулл токенов, из которых будут создавать уравнения. Граница в 10 элементов позволяет 
    избавиться от ошибок в значении производных, которые встречаются на границах исследумой области.
    Также, выполняем предварительную загрузку данных в кэш.
    '''

    boundary = 10
    upload_grids(x[boundary:-boundary], global_var.grid_cache)   
    u_derivs_stacked = prepare_var_tensor(ts_smoothed, derivs, time_axis = 0, boundary = boundary, axes = [x,])
    u_names = ['t',] + Define_Derivatives('u', 1, 1) 
    upload_simple_tokens(u_names, global_var.tensor_cache, u_derivs_stacked)
    global_var.tensor_cache.use_structural()
    '''

    Далее ряд операций для задания семейств токенов (коорд. ось, исх. функция и её производные в первом 
    семействе, а во втором - тригонометрические функции): 
    задание статуса использования токенов через метод .set_status(...)
    выбор параметров; названия индивидуальных токенов из семейства; параметры равенства двух множителей 
    одного типа, но с разными параметрами (т.е. когда f(x, p1) == f(x, p2), где p1 и p2 - параметры
    вроде частоты, степени и т.д.), задание метода оценки значений токена на сетке через .set_evaluator(...)
    и т.д.

    '''
    u_tokens = Token_family('U')
    u_tokens.use_glob_cache()
    u_tokens.set_status(unique_specific_token=False, unique_token_type=False, s_and_d_merged = False, 
                        meaningful = True, unique_for_right_part = False)
    u_token_params = OrderedDict([('power', (1, 1))])
    u_equal_params = {'power' : 0}
    u_tokens.set_params(u_names, u_token_params, u_equal_params)
    u_tokens.set_evaluator(simple_function_evaluator, [])
    
    trig_tokens = Token_family('trig')
    trig_names = ['sin', 'cos']
    trig_tokens.use_glob_cache()
    trig_tokens.set_status(unique_specific_token=True, unique_token_type=True, 
                           meaningful = False, unique_for_right_part = False)
    trig_token_params = OrderedDict([('power', (1, 1)), ('freq', (0.95, 1.05)), ('dim', (0, 0))])
    trig_equal_params = {'power' : 0, 'freq' : 0.05, 'dim' : 0}
    trig_tokens.set_params(trig_names, trig_token_params, trig_equal_params)
    trig_tokens.set_evaluator(trigonometric_evaluator, [])
    
    '''
    Объединяем заданные семейства токенов в пулл, из которого будут строиться уравнения.
    '''
    pool = TF_Pool([u_tokens, trig_tokens])
    pool.families_cardinality()
    
    '''
    Используем базовый эволюционный оператор.
    '''
    test_strat = Strategy_director(Iteration_limit, {'limit' : 100})
    test_strat.strategy_assembly()
    
#    test_system = SoEq(pool = pool, terms_number = 4, max_factors_in_term=2, sparcity = (0.1,))
#    test_system.set_eq_search_evolutionary(director.constructor.operator)
#    test_system.create_equations(population_size=16, eq_search_iters=300)    
    
#    tokens=[h_tokens, trig_tokens]
    '''
    Настраиваем генератор новых уравнений, которые будут составлять популяцию для 
    алгоритма многокритериальной оптимизации.
    '''
    pop_constructor = operators.systems_population_constructor(pool = pool, terms_number=6, 
                                                               max_factors_in_term=2, eq_search_evo=test_strat.constructor.strategy,
                                                               sparcity_interval = (0.001, 1.2))
    
    '''
    Задаём объект многокритериального оптимизатора, эволюционный оператор и задаём лучшие возможные 
    значения целевых функций.
    '''
    optimizer = moeadd_optimizer(pop_constructor, 4, 4, delta = 1/50., neighbors_number = 3, solution_params = {'eq_search_iters':50})
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
    
    optimizer.optimize(simple_selector, 0.95, (4,), 10, 0.75) # Простая форма искомого уравнения найдется и за 10 итераций           
    
    [print(solution.structure[0].text_form, solution.evaluate())  for solution in optimizer.pareto_levels.levels[0]]
    
    '''
    В результате мы должны получить фронт Парето, который должен включать в себя одно уравнение с 
    "0-ём слагаемых в левой части", т.к. равенство какого-то токена константе (скорее всего, 0), а также 
    1 уравнение с "1им слагаемым помимо константы и правой части из одного слагаемого", которое будет либо исходным 
    (т.е. искомым уравнением, либо уравнением u cos(x) - 1.3 = u' sin(x), которое имеет 
    частное решение, совпадающее с рассматриваемым частным решением исходного уравнения.)
    '''
    