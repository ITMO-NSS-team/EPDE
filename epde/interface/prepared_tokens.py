#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 13:26:56 2021

@author: mike_ubuntu
"""
import numpy as np
from abc import ABC
from collections import OrderedDict
from typing import Union, Callable
import time

import epde.globals as global_var
from epde.interface.token_family import TokenFamily
from epde.evaluators import CustomEvaluator, EvaluatorTemplate, trigonometric_evaluator, simple_function_evaluator 
from epde.evaluators import velocity_heating_eval_fun, vhef_grad
from epde.cache.cache import upload_simple_tokens, np_ndarray_section

class Prepared_tokens(ABC):
    def __init__(self, *args, **kwargs):
        self._token_family = TokenFamily(token_type = 'Placeholder')
        
    @property
    def token_family(self):
        if not (self._token_family.evaluator_set and self._token_family.params_set):
            raise AttributeError(f'Some attributes of the token family have not been declared.')
        return self._token_family
    
    
class Trigonometric_tokens(Prepared_tokens):
    def __init__(self, freq : tuple = (np.pi/2., 2*np.pi), dimensionality = 1):
        assert freq[1] > freq[0] and len(freq) == 2, 'The tuple, defining frequncy interval, shall contain 2 elements with first - the left boundary of interval and the second - the right one. '

        self._token_family = TokenFamily(token_type='trigonometric')
        self._token_family.set_status(unique_specific_token=True, unique_token_type=True, 
                           meaningful = False)
        
        trig_token_params = OrderedDict([('power', (1, 1)), 
                                         ('freq', freq), 
                                         ('dim', (0, dimensionality))])
        freq_equality_fraction = 0.05 # fraction of allowed frequency interval, that is considered as the same
        trig_equal_params = {'power' : 0, 'freq' : (freq[1] - freq[0]) / freq_equality_fraction, 
                             'dim' : 0}
        self._token_family.set_params(['sin', 'cos'], trig_token_params, trig_equal_params)
        self._token_family.set_evaluator(trigonometric_evaluator, [])
        
        
class Logfun_tokens(Prepared_tokens):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError
        
class Custom_tokens(Prepared_tokens):
    def __init__(self, token_type : str, token_labels : list, 
                 evaluator : Union[CustomEvaluator, EvaluatorTemplate, Callable], 
                 params_ranges : dict, params_equality_ranges : Union[None, dict], dimensionality : int = 1,
                 unique_specific_token=True, unique_token_type=True, meaningful = False):
        self._token_family = TokenFamily(token_type = token_type)
        self._token_family.set_status(unique_specific_token = unique_specific_token, 
                                      unique_token_type = unique_token_type, meaningful = meaningful)
        default_param_eq_fraction = 0.5
        if params_equality_ranges is not None:
            for param_key, interval in params_ranges.items():
                if param_key not in params_equality_ranges.keys():
                    if isinstance(interval[0], float):
                        params_equality_ranges[param_key] = (interval[1] - interval[0]) * default_param_eq_fraction
                    elif isinstance(interval[0], int):
                        params_equality_ranges[param_key] = 0
        else:
            params_equality_ranges = dict()
            for param_key, interval in params_ranges.items():
                if isinstance(interval[0], float):
                    params_equality_ranges[param_key] = (interval[1] - interval[0]) * default_param_eq_fraction
                elif isinstance(interval[0], int):
                    params_equality_ranges[param_key] = 0

        self._token_family.set_params(token_labels, params_ranges, params_equality_ranges)
        self._token_family.set_evaluator(evaluator, [])      
        
class Cache_stored_tokens(Custom_tokens):
    def __init__(self, token_type : str, boundary : Union[list, tuple],
                 token_labels : list, token_tensors : dict, params_ranges : dict,
                 params_equality_ranges : Union[None, dict], dimensionality : int = 1,
                 unique_specific_token=True, unique_token_type=True, meaningful = False):
        if set(token_labels) != set(list(token_tensors.keys())):
            raise KeyError('The labels of tokens do not match the labels of passed tensors')
        for key, val in token_tensors.items():
            token_tensors[key] = np_ndarray_section(val, boundary = boundary)
        upload_simple_tokens(list(token_tensors.keys()), global_var.tensor_cache, list(token_tensors.values()))
        super().__init__(token_type = token_type, token_labels = token_labels, evaluator = simple_function_evaluator, 
                         params_ranges = params_ranges, params_equality_ranges = params_equality_ranges, 
                         dimensionality = dimensionality, unique_specific_token = unique_specific_token, 
                         unique_token_type = unique_token_type, meaningful = meaningful)

class Velocity_HEQ_tokens(Prepared_tokens):
    def __init__(self, param_ranges):
        assert len(param_ranges) == 15
        self._token_family = TokenFamily(token_type='velocity_assuption')
        self._token_family.set_status(unique_specific_token=True, unique_token_type=True, 
                           meaningful = False)
        
        opt_params = [('p' + str(idx), p_range) for idx, p_range in enumerate(param_ranges)]
        token_params = OrderedDict([('power', (1, 1)),] + opt_params)

        p_equality_fraction = 0.05 # fraction of allowed frequency interval, that is considered as the same
        opt_params_equality = {'p' + str(idx) : (p_range[1] - p_range[0]) / p_equality_fraction for idx, p_range in enumerate(param_ranges)}
        equal_params = {'power' : 0} + opt_params_equality
        self._token_family.set_params(['v'], token_params, equal_params)
        self._token_family.set_evaluator(velocity_heating_eval_fun, vhef_grad, [])