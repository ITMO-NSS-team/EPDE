#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 17 14:12:05 2021

@author: maslyaev
"""

import numpy as np
from collections import OrderedDict
from typing import Union
import epde.globals as global_var

from epde.interface.prepared_tokens import Prepared_tokens
from epde.evaluators import simple_function_evaluator
from epde.interface.token_family import TokenFamily, TF_Pool
from epde.interface.interface import Input_data_entry

def create_pool(data : Union[np.ndarray, list, tuple], time_axis : int = 0, boundary : int = 0, 
                variable_names = ['u',], derivs = None, max_deriv_order = 1, additional_tokens = [], 
                coordinate_tensors = None, memory_for_cache = 5, data_fun_pow : int = 1):
    assert (isinstance(derivs, list) and isinstance(derivs[0], np.ndarray)) or derivs is None
    if isinstance(data, np.ndarray):
        data = [data,]

    set_grids = coordinate_tensors is not None
    set_grids_among_tokens = coordinate_tensors is not None
    if derivs is None:
        if len(data) != len(variable_names):
            print(len(data), len(variable_names))
            raise ValueError('Mismatching lengths of data tensors and the names of the variables')
    else:
        if not (len(data) == len(variable_names) == len(derivs)): 
            raise ValueError('Mismatching lengths of data tensors, names of the variables and passed derivatives')            
    data_tokens = []
    for data_elem_idx, data_tensor in enumerate(data):
        assert isinstance(data_tensor, np.ndarray), 'Input data must be in format of numpy ndarrays or iterable (list or tuple) of numpy arrays'
        entry = Input_data_entry(var_name = variable_names[data_elem_idx],
                                 data_tensor = data_tensor, 
                                 coord_tensors = coordinate_tensors)
        derivs_tensor = derivs[data_elem_idx] if derivs is not None else None
        entry.set_derivatives(deriv_tensors = derivs_tensor, max_order = max_deriv_order)
        print(f'set grids parameter is {set_grids}')
        entry.use_global_cache(grids_as_tokens = set_grids_among_tokens,
                               set_grids=set_grids, memory_for_cache=memory_for_cache, boundary=boundary)
        set_grids = False; set_grids_among_tokens = False
        
        entry_token_family = TokenFamily(entry.var_name, family_of_derivs = True)
        entry_token_family.set_status(unique_specific_token=False, unique_token_type=False, 
                             s_and_d_merged = False, meaningful = True, 
                             unique_for_right_part = True)     
        entry_token_family.set_params(entry.names, OrderedDict([('power', (1, data_fun_pow))]),
                                      {'power' : 0}, entry.d_orders)
        entry_token_family.set_evaluator(simple_function_evaluator, [])
            
        print(entry_token_family.tokens)
        data_tokens.append(entry_token_family)
 
    if isinstance(additional_tokens, list):
        if not all([isinstance(tf, (TokenFamily, Prepared_tokens)) for tf in additional_tokens]):
            raise TypeError(f'Incorrect type of additional tokens: expected list or TokenFamily/Prepared_tokens - obj, instead got list of {type(additional_tokens[0])}')                
    elif isinstance(additional_tokens, (TokenFamily, Prepared_tokens)):
        additional_tokens = [additional_tokens,]
    else:
        print(isinstance(additional_tokens, Prepared_tokens))
        raise TypeError(f'Incorrect type of additional tokens: expected list or TokenFamily/Prepared_tokens - obj, instead got {type(additional_tokens)}')
    return TF_Pool(data_tokens + [tf if isinstance(tf, TokenFamily) else tf.token_family 
                                  for tf in additional_tokens])

if __name__ == '__main__':
    global_var.time_axis = 0
    global_var.init_caches(set_grids = False)
    dummy_data = np.ones((10, 10))
    global_var.tensor_cache.memory_usage_properties(dummy_data, 5)

    dummy_derivs = np.ones((100, 2))
    dummy_pool = create_pool(data = dummy_data, derivs = [dummy_derivs,], max_deriv_order=1)
    _, test_factor = dummy_pool.create()
    print(test_factor.label, test_factor.params)
