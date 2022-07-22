#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 13:32:54 2021

@author: mike_ubuntu
"""

import numpy as np
from collections import OrderedDict
import sys
import getopt

global opt, args
opt, args = getopt.getopt(sys.argv[2:], '', ['path='])

sys.path.append(opt[0][1])

import epde.globals as global_var
from epde.cache.cache import upload_grids

from epde.interface.token_family import Token_family, TF_Pool
from epde.structure import Term, Equation, SoEq
from epde.evaluators import trigonometric_evaluator
#from epde.src.evo_optimizer import Operator_director

def mock_eval_function(factor,  structural = False, **kwargs):
    return np.full(shape = 100, fill_value = 2.)

#class mock_token_family(Token_family):
#    def __init__(self, names = [], evaluator = None):
#        super().__init__('mock')
#        super().set_status(meaningful = True)
#      
#        mock_equal_params = {'not_power' : 0, 'power' : 0}
#        mock_params = OrderedDict([('not_power', (1, 4)), ('power', (1, 1))])
#        super().set_evaluator(evaluator)
#        super().set_params(names, mock_params, mock_equal_params)   
        
def set_family(family, names, params, equal_params, evaluator, meaningful):
    family.set_status(meaningful = meaningful)
    family.set_evaluator(evaluator)
    family.set_params(names, params, equal_params)
        
def test_term():
    '''
    Check both (random and determined) ways to initialize the term, check correctness of term value evaluation & terms equality, 
    output format, latex format.

    '''

    x = np.linspace(0, 2*np.pi, 100)   
    global_var.init_caches(set_grids=True)
    global_var.tensor_cache.memory_usage_properties(obj_test_case=x, mem_for_cache_frac = 5)  
    global_var.grid_cache.memory_usage_properties(obj_test_case=x, mem_for_cache_frac = 5)  
    upload_grids(x, global_var.grid_cache)    
    mock_equal_params = {'not_power' : 0, 'power' : 0}
    mock_params = {'not_power' : (1, 4), 'power' : (1, 1)}

    f1 = Token_family('type_1')
    f2 = Token_family('type_2')
    f3 = Token_family('type_3')
    set_family(f1, names = ['t1_1', 't1_2', 't1_3'], params = mock_params, 
               equal_params=mock_equal_params, evaluator=mock_eval_function, meaningful=True);
    set_family(f2, names = ['t2_1', 't2_2', 't2_3', 't2_4'], params = mock_params, 
               equal_params=mock_equal_params, evaluator=mock_eval_function, meaningful=False); 
    set_family(f3, names = ['t3_1', 't3_2'], params = mock_params, 
               equal_params=mock_equal_params, evaluator=mock_eval_function, meaningful=True)
    
    pool = TF_Pool([f1, f2, f3])
    test_term_1 = Term(pool)

    print(test_term_1.name)
    print('avalable', test_term_1.available_tokens[0].tokens)
#    assert test_term_1.available_tokens[0].tokens == names
#    assert type(test_term_1) == Term

    test_term_2 = Term(pool, passed_term = 't1_1')
    print(test_term_2.name)
    print(test_term_2.solver_form)
    assert test_term_2.solver_form[1] is None and test_term_2.solver_form[2] == 1
#    raise Exception('Kek')


def test_equation():
    '''
    
    Use trigonometric identity sin^2 (x) + cos^2 (x) = 1 to generate data, with it: initialize the equation, 
    equation splitting & weights discovery? output format, latex format. Additionally, test evaluator for trigonometric functions
    
    '''
    global_var.init_caches(set_grids=True)
    global_var.tensor_cache.memory_usage_properties(obj_test_case=np.ones((10,10,10)), mem_for_cache_frac = 25)  
    
    from epde.eq_search_strategy import Strategy_director
    from epde.operators.ea_stop_criteria import Iteration_limit
    
    director = Strategy_director(Iteration_limit, {'limit' : 100})
    director.strategy_assembly()
    
    x = np.linspace(0, 2*np.pi, 100)
    global_var.grid_cache.memory_usage_properties(obj_test_case=x, mem_for_cache_frac = 25)  
    upload_grids(x, global_var.grid_cache)
    names = ['sin', 'cos'] # simple case: single parameter of a token - power
    
    trig_tokens = Token_family('trig')    
    trig_tokens.set_status(unique_specific_token=False, unique_token_type=False, meaningful = True, 
                           unique_for_right_part = False)
    
    equal_params = {'power' : 0, 'freq' : 0.2, 'dim' : 0}
    
    trig_params = OrderedDict([('power', (1., 1.)), ('freq', (0.5, 1.5)), ('dim', (0., 0.))])
    trig_tokens.set_params(names, trig_params, equal_params)
    trig_tokens.set_evaluator(trigonometric_evaluator)  
    
    set_family(trig_tokens, names, trig_params, equal_params, trigonometric_evaluator, True)
    pool = TF_Pool([trig_tokens,])
    
    eq1 = Equation(pool , basic_structure = [], 
                   terms_number = 3, max_factors_in_term = 2)   # Задать возможности выбора вероятностей кол-ва множителей
    director.constructor.strategy.modify_block_params(block_label = 'rps1', param_label = 'sparsity', 
                                                             value = 1., suboperator_sequence = ['eq_level_rps', 'fitness_calculation', 'sparsity'])

    director._constructor._strategy.apply_block('rps1', 
                                                {'population' : [eq1,], 'unexplained_vars' : []})
    
    eq1.described_variables
    eq1.evaluate(normalize = False, return_val = True)
    eq1.weights_internal
    eq1.weights_final
    print(eq1.text_form)
#    raise Exception('Exception to print test function')
    
    
def test_solver_forms():
    from epde.cache.cache import upload_simple_tokens, upload_grids, prepare_var_tensor
    from epde.prep.derivatives import Preprocess_derivatives
    from epde.supplementary import Define_Derivatives
    from epde.evaluators import simple_function_evaluator, trigonometric_evaluator
    from epde.operators.ea_stop_criteria import Iteration_limit
    from epde.eq_search_strategy import Strategy_director
    
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
    for i in np.arange(10):
        global_var.init_caches(set_grids=True)
        global_var.tensor_cache.memory_usage_properties(obj_test_case=ts, mem_for_cache_frac = 5)  
        global_var.grid_cache.memory_usage_properties(obj_test_case=x, mem_for_cache_frac = 5)
    
        
        print(type(derivs))
    
        boundary = 10
        upload_grids(x[boundary:-boundary], global_var.grid_cache)   
        u_derivs_stacked = prepare_var_tensor(ts_smoothed, derivs, time_axis = 0, boundary = boundary)
        
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
        upload_simple_tokens(grid_names, global_var.tensor_cache, [x[boundary:-boundary],])    
        global_var.tensor_cache.use_structural()
    
    
        grid_tokens = Token_family('Grids')
        grid_tokens.set_status(unique_specific_token=True, unique_token_type=True, s_and_d_merged = False, 
                            meaningful = True, unique_for_right_part = False)
        grid_token_params = OrderedDict([('power', (1, 1))])
        grid_equal_params = {'power' : 0}
        grid_tokens.set_params(grid_names, grid_token_params, grid_equal_params)
        grid_tokens.set_evaluator(simple_function_evaluator, [])
        
        trig_tokens = Token_family('Trigonometric')
        trig_names = ['sin', 'cos']
        trig_tokens.set_status(unique_specific_token=True, unique_token_type=True, 
                               meaningful = False, unique_for_right_part = False)
        trig_token_params = OrderedDict([('power', (1, 1)), ('freq', (0.95, 1.05)), ('dim', (0, 0))])
        trig_equal_params = {'power' : 0, 'freq' : 0.05, 'dim' : 0}
        trig_tokens.set_params(trig_names, trig_token_params, trig_equal_params)
        trig_tokens.set_evaluator(trigonometric_evaluator, [])
    
        pool = TF_Pool([grid_tokens, u_tokens, trig_tokens])
        pool.families_cardinality()
    
        test_strat = Strategy_director(Iteration_limit, {'limit' : 300})
        test_strat.strategy_assembly()        
        
        eq1 = Equation(pool , basic_structure = [], 
                       terms_number = 6, max_factors_in_term = 2)   # Задать возможности выбора вероятностей кол-ва множителей
        test_strat.constructor.strategy.modify_block_params(block_label = 'rps1', param_label = 'sparsity', 
                                                                 value = 0.1, suboperator_sequence = ['eq_level_rps', 'fitness_calculation', 'sparsity'])
    
        test_strat._constructor._strategy.apply_block('rps1', 
                                                    {'population' : [eq1,], 'unexplained_vars' : []})
        
        print('text form:', eq1.text_form)
    #    print('solver form:', eq1.solver_form())
        print(eq1.max_deriv_orders())
    raise Exception('Test exception')