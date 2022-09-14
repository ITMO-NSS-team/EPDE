#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 23 18:05:09 2021

@author: mike_ubuntu
"""

import numpy as np
import sys
from collections import OrderedDict
sys.path.append('/media/mike_ubuntu/DATA/ESYS/')

import src.globals as global_var
from src.token_family import Token_family, TF_Pool

def mock_evaluator(factor):
    return np.ones((10, 10, 10))

def test_TF():
    global_var.init_caches(set_grids=True)
    global_var.tensor_cache.memory_usage_properties(obj_test_case=np.ones((10,10,10)), mem_for_cache_frac = 25)  
    family = Token_family('test_type')
    family.use_glob_cache()
    family.set_status()
  
    mock_equal_params = {'not_power' : 0, 'power' : 0}
    mock_params = {'not_power' : (1, 4), 'power' : (1, 1)}
    
    family.set_evaluator(mock_evaluator)
    names = ['n1', 'n2', 'n3']
    family.set_params(names, mock_params, mock_equal_params)           
    print(family.cardinality())

    occ, token_example = family.create('n1')
    print(occ, token_example.name)

    occ, token_example = family.create()
    print(occ, token_example.name)

    occ, token_example = family.create(occupied = ['n1', 'n2'])
    print(occ, token_example.name, token_example.status)
    
#    raise NotImplementedError
    
def set_family(family, names, params, equal_params, evaluator, meaningful):
    family.use_glob_cache()
    family.set_status(meaningful = meaningful)
    family.set_evaluator(evaluator)
    family.set_params(names, params, equal_params)
    
    
def test_pool():
    global_var.init_caches(set_grids=True)
    global_var.tensor_cache.memory_usage_properties(obj_test_case=np.ones((10,10,10)), mem_for_cache_frac = 25)  
    mock_equal_params = {'not_power' : 0, 'power' : 0}
    mock_params = {'not_power' : (1, 4), 'power' : (1, 1)}

    f1 = Token_family('type_1')
    f2 = Token_family('type_2')
    f3 = Token_family('type_3')
    set_family(f1, names = ['t1_1', 't1_2', 't1_3'], params = mock_params, 
               equal_params=mock_equal_params, evaluator=mock_evaluator, meaningful=True);
    set_family(f2, names = ['t2_1', 't2_2', 't2_3', 't2_4'], params = mock_params, 
               equal_params=mock_equal_params, evaluator=mock_evaluator, meaningful=False); 
    set_family(f3, names = ['t3_1', 't3_2'], params = mock_params, 
               equal_params=mock_equal_params, evaluator=mock_evaluator, meaningful=True)
    
    pool = TF_Pool([f1, f2, f3])
    print('meaningful:', [(family.ftype, family.tokens) for family in pool.families_meaningful])
    print('all:', [(family.ftype, family.tokens) for family in pool.families])
    pool.families_cardinality(meaningful_only = True)
    pool.families_cardinality(meaningful_only = False)
#    raise NotImplementedError
    
    
    
    
    