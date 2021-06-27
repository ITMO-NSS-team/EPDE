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

import epde.src.globals as global_var
from epde.src.cache.cache import upload_grids

from epde.src.token_family import Token_family, TF_Pool
from epde.src.structure import Term, Equation, SoEq
from epde.src.evaluators import trigonometric_evaluator
#from epde.src.evo_optimizer import Operator_director

def mock_eval_function(factor,  structural = False, **kwargs):
    return np.ones((10, 10, 10))

#class mock_token_family(Token_family):
#    def __init__(self, names = [], evaluator = None):
#        super().__init__('mock')
#        super().use_glob_cache()
#        super().set_status(meaningful = True)
#      
#        mock_equal_params = {'not_power' : 0, 'power' : 0}
#        mock_params = OrderedDict([('not_power', (1, 4)), ('power', (1, 1))])
#        super().set_evaluator(evaluator)
#        super().set_params(names, mock_params, mock_equal_params)   
        
def set_family(family, names, params, equal_params, evaluator, meaningful):
    family.use_glob_cache()
    family.set_status(meaningful = meaningful)
    family.set_evaluator(evaluator)
    family.set_params(names, params, equal_params)
        
def test_term():
    '''
    Check both (random and determined) ways to initialize the term, check correctness of term value evaluation & terms equality, 
    output format, latex format.
    '''
    
    global_var.init_caches(set_grids=True)
    global_var.tensor_cache.memory_usage_properties(obj_test_case=np.ones((10,10,10)), mem_for_cache_frac = 5)  
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
    assert type(test_term_2) == Term    
#    
#    test_term_3 = Term([mock,], passed_term = ['mock3', 'mock1'])
#    print(test_term_3.name)
#    assert type(test_term_3) == Term   
#    
#    test_term_2.evaluate, test_term_3.evaluate
##    assert False
#
def test_equation():
    '''
    Use trigonometric identity sin^2 (x) + cos^2 (x) = 1 to generate data, with it: initialize the equation, 
    equation splitting & weights discovery? output format, latex format. Additionally, test evaluator for trigonometric functions
    '''
    global_var.init_caches(set_grids=True)
    global_var.tensor_cache.memory_usage_properties(obj_test_case=np.ones((10,10,10)), mem_for_cache_frac = 25)  
    
    from epde.src.eq_search_strategy import Strategy_director
    from epde.src.operators.ea_stop_criteria import Iteration_limit
    
    director = Strategy_director(Iteration_limit, {'limit' : 100})
    director.strategy_assembly()
    
#    .operator_assembly()    
    
    x = np.linspace(0, 2*np.pi, 100)
    global_var.grid_cache.memory_usage_properties(obj_test_case=x, mem_for_cache_frac = 25)  
    upload_grids(x, global_var.grid_cache)
#    print(global_var.grid_cache.memory)
    names = ['sin', 'cos'] # simple case: single parameter of a token - power
    
    trig_tokens = Token_family('trig')    
    trig_tokens.set_status(unique_specific_token=False, unique_token_type=False, meaningful = True, 
                           unique_for_right_part = False)
    
    equal_params = {'power' : 0, 'freq' : 0.2, 'dim' : 0}

    trig_tokens.use_glob_cache()
    
    trig_params = OrderedDict([('power', (1., 1.)), ('freq', (0.5, 1.5)), ('dim', (0., 0.))])
    trig_tokens.set_params(names, trig_params, equal_params)
    trig_tokens.set_evaluator(trigonometric_evaluator)  
    
    def set_family(family, names, params, equal_params, evaluator, meaningful):
        family.use_glob_cache()
        family.set_status(meaningful = meaningful)
        family.set_evaluator(evaluator)
        family.set_params(names, params, equal_params)
        
    set_family(trig_tokens, names, trig_params, equal_params, trigonometric_evaluator, True)
    pool = TF_Pool([trig_tokens,])
    
    eq1 = Equation(pool , basic_structure = [], 
                   terms_number = 3, max_factors_in_term = 2)   # Задать возможности выбора вероятностей кол-ва множителей
#    assert False  
#    director.constructor.operator.set_sparcity(sparcity_value = 1.)    
#    eq1.select_target_idx(target_idx_fixed = 0)
#    eq1.select_target_idx(operator = director.constructor.operator)
#    print([term.name for term in eq1.structure])
#    print(eq1.fitness_value)
    director._constructor._strategy.apply_block('rps', 
                                                {'population' : [eq1,], 'separate_vars' : []})
    
    eq1.described_variables
    eq1.evaluate(normalize = False, return_val = True)
    eq1.weights_internal
    eq1.weights_final