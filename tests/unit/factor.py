#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 13:29:59 2021

@author: mike_ubuntu
"""

import sys
import getopt

global opt, args
opt, args = getopt.getopt(sys.argv[2:], '', ['path='])

sys.path.append(opt[0][1])


import numpy as np
from collections import OrderedDict


import epde.src.globals as global_var
from epde.src.token_family import Token_family, Evaluator
from epde.src.factor import Factor
from epde.src.cache.cache import upload_grids, upload_simple_tokens
from epde.src.supplementary import Define_Derivatives

def mock_eval_function(factor,  structural = False, **kwargs):
    return np.ones((10, 10, 10))

#class mock_token_family(Token_family):
#    def __init__(self, names = [], evaluator = None):
#        super().__init__('mock')
#        super().use_glob_cache()
#        super().set_status()
#      
#        mock_equal_params = {'not_power' : 0, 'power' : 0}
#        mock_params = OrderedDict([('not_power', (1, 4)), ('power', (1, 1))])
#        super().set_evaluator(evaluator)
#        super().set_params(names, mock_params, mock_equal_params)       

#def test_factor():
#    global_var.init_caches(set_grids=False)
#    global_var.tensor_cache.memory_usage_properties(obj_test_case=np.ones((10,10,10)), mem_for_cache_frac = 25)    
##    names = ['mock1', 'mock2', 'mock3']
##    mock = mock_token_family(names, mock_evaluator)
#    test_factor_1 = Factor(names[0], mock, randomize=True)
#    test_factor_2 = Factor(names[0], mock, randomize=True)
#    print(test_factor_1.params, test_factor_1.params_description) 
#    print(test_factor_2.params, test_factor_2.params_description)     
##    print(test_factor_3.params, test_factor_3.params_description)
#    
#    assert type(test_factor_1.cache_label) == tuple and type(test_factor_1.cache_label[0]) == str and type(test_factor_1.cache_label[1]) == tuple
#    assert np.all(test_factor_1.evaluate() == test_factor_2.evaluate())
#    print(test_factor_1.params, test_factor_2.params)
##    assert test_factor_1 == test_factor_2, 'Equally defined tokens are not equal'
#    
#    test_factor_3 = Factor(names[1], mock, randomize=False)
#    test_factor_3.Set_parameters(random=False, not_power = 2, power = 1)
#    test_factor_4 = Factor(names[1], mock, randomize=False)
#    test_factor_4.Set_parameters(random=False, not_power = 2, power = 1)
#    assert test_factor_3 == test_factor_4, 'Equally defined tokens are not equal'
#    
#    print(test_factor_3.name)
#    assert False
    
def test_grids_cache():
    global_var.init_caches(set_grids=True)
    global_var.tensor_cache.memory_usage_properties(obj_test_case=np.ones((10,10,10)), mem_for_cache_frac = 5)  
    
    assert type(global_var.tensor_cache.base_tensors) == list, f'Creating wrong obj for base tensors: {type(global_var.tensor_cache.base_tensors)} instead of list'
    
    x = np.linspace(0, 2*np.pi, 100)
    global_var.grid_cache.memory_usage_properties(obj_test_case=x, mem_for_cache_frac = 5)  

    upload_grids(x, global_var.grid_cache)
    
    print(global_var.grid_cache.memory_default.keys(), global_var.grid_cache.memory_default.values())
    assert '0' in global_var.grid_cache 
    assert x in global_var.grid_cache 
    assert (x, False) in global_var.grid_cache 
    assert not (x, True) in global_var.grid_cache
    
    x_returned = global_var.grid_cache.get('0')
    assert np.all(x == x_returned)
    global_var.grid_cache.clear(full = True)
    
    y = np.linspace(0, 10, 200)
    grids = np.meshgrid(x, y)
    upload_grids(grids, global_var.grid_cache)
    print('memory for cache:', global_var.grid_cache.available_mem, 'B')
    print('consumed memory:', global_var.grid_cache.consumed_memory, 'B')
    print(global_var.grid_cache.memory_default.keys())
    global_var.grid_cache.delete_entry(entry_label = '0')
    assert not '0' in global_var.grid_cache.memory_default.keys()
    
def test_tensor_cache():
    global_var.init_caches(set_grids=False)
    global_var.tensor_cache.memory_usage_properties(obj_test_case=np.ones((10,10,10)), mem_for_cache_frac = 5)  
    
    tensors = np.zeros((5, 10, 10, 10))
    t_labels = Define_Derivatives('t', 2, 2)
    upload_simple_tokens(t_labels, global_var.tensor_cache, tensors)
    global_var.tensor_cache.use_structural(use_base_data = True)
    replacing_tensors = np.ones((5, 10, 10, 10))
    replacing_data = {}
    for idx, key in enumerate(global_var.tensor_cache.memory_default.keys()):
        replacing_data[key] = replacing_tensors[idx, ... ]
        
    print(replacing_data[list(global_var.tensor_cache.memory_default.keys())[0]].shape, list(global_var.tensor_cache.memory_default.values())[0].shape)
    global_var.tensor_cache.use_structural(use_base_data = False, replacing_data = replacing_data)
    print(global_var.tensor_cache.memory_default.keys())
    print(list(global_var.tensor_cache.memory_default.keys()))
    key = list(global_var.tensor_cache.memory_default.keys())[np.random.randint(low = 0,
                                                                                high = len(global_var.tensor_cache.memory_default.keys()))]
    global_var.tensor_cache.use_structural(use_base_data = False, 
                                           replacing_data = np.full(shape = (10, 10, 10), fill_value=2.),
                                           label = key)
    
def test_factor():
    global_var.init_caches(set_grids=False)
    global_var.tensor_cache.memory_usage_properties(obj_test_case=np.ones((10,10,10)), mem_for_cache_frac = 5)
    test_status = {'meaningful':True, 'unique_specific_token':False, 'unique_token_type':False, 
              'unique_for_right_part':False, 'requires_grid':False}
    try:
        name = 'made_to_fail'
        test_factor = Factor(name, test_status,  family_type = 'some family', randomize = True)
    except AssertionError:
        pass
    name = 'made_to_success'
    test_equal_params = {'not_power' : 0, 'power' : 0}
    test_params = {'not_power' : (1, 4), 'power' : (1, 1)}    
    test_factor = Factor(name, test_status, family_type = 'some family', randomize = True, params_description = test_params, 
                         equality_ranges = test_equal_params)
    _evaluator = Evaluator(mock_eval_function, [])
    test_factor.Set_evaluator(_evaluator)
    test_factor.evaluate()
    assert test_factor.name is not None
#    raise NotImplementedError