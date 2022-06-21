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

from epde.supplementary import Define_Derivatives
from epde.prep.derivatives import Preprocess_derivatives

import epde.globals as global_var
from epde.interface.token_family import TokenFamily
from epde.evaluators import CustomEvaluator, EvaluatorTemplate, trigonometric_evaluator, simple_function_evaluator
from epde.evaluators import const_evaluator, const_grad_evaluator
from epde.evaluators import velocity_evaluator, velocity_grad_evaluators
from epde.cache.cache import upload_simple_tokens, np_ndarray_section, prepare_var_tensor

class PreparedTokens(ABC):
    def __init__(self, *args, **kwargs):
        self._token_family = TokenFamily(token_type = 'Placeholder')
        
    @property
    def token_family(self):
        if not (self._token_family.evaluator_set and self._token_family.params_set):
            raise AttributeError(f'Some attributes of the token family have not been declared.')
        return self._token_family
    
    
class TrigonometricTokens(PreparedTokens):
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
        
        
class LogfunTokens(PreparedTokens):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError
        
class CustomTokens(PreparedTokens):
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
        
class CacheStoredTokens(CustomTokens):
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

class ExternalDerivativesTokens(CustomTokens):
    def __init__(self, token_type : str, boundary : Union[list, tuple], time_axis : int,
                 base_token_label : list, token_tensor : np.ndarray, max_orders : Union[int, tuple],
                 deriv_method : str, deriv_method_kwargs : dict, params_ranges : dict,
                 params_equality_ranges : Union[None, dict], dimensionality : int = 1,
                 unique_specific_token=True, unique_token_type=True, meaningful = False):
        deriv_method_kwargs['max_order'] = max_orders
        data_tensor, derivs_tensor = Preprocess_derivatives(token_tensor, method = deriv_method, 
                                                          method_kwargs = deriv_method_kwargs)
        deriv_names, deriv_orders = Define_Derivatives(base_token_label, dimensionality=token_tensor.ndim, 
                                                       max_order = max_orders)

        derivs_stacked = prepare_var_tensor(token_tensor, derivs_tensor, time_axis, boundary)
        upload_simple_tokens(deriv_names, global_var.tensor_cache, derivs_stacked)

        super().__init__(token_type = token_type, token_labels = deriv_names,
                         evaluator = simple_function_evaluator, params_ranges = params_ranges,
                         params_equality_ranges = params_equality_ranges,
                         dimensionality = dimensionality, unique_specific_token = unique_specific_token,
                         unique_token_type = unique_token_type, meaningful = meaningful)

class ConstantToken(PreparedTokens):
    def __init__(self, values_range = (-np.inf, np.inf)):
        assert len(values_range) == 2 and values_range[0] < values_range[1], 'Range of the values has not been stated correctly.'
        self._token_family = TokenFamily(token_type='constants')
        self._token_family.set_status(unique_specific_token=True, unique_token_type=True, 
                                      meaningful = False)
        const_token_params = OrderedDict([('power', (1, 1)),
                                          ('value', values_range)])
        if - values_range[0] == np.inf or values_range[0] == np.inf:
            value_equal_range = 0.1
        else:
            val_equality_fraction = 0.01
            value_equal_range = (values_range[1] - values_range[0]) / val_equality_fraction
        const_equal_params = {'power' : 0, 'value' : value_equal_range}
        print('Conducting init procedure for ConstantToken:')
        self._token_family.set_params(['const'], const_token_params, const_equal_params)
        print('Parameters set')
        self._token_family.set_evaluator(const_evaluator, []) 
        print('Evaluator set')
        self._token_family.set_deriv_evaluator({'value' : const_grad_evaluator}, [])

class Velocity_HEQ_tokens(PreparedTokens):
    def __init__(self, param_ranges):
        assert len(param_ranges) == 15
        self._token_family = TokenFamily(token_type='velocity_assuption')
        self._token_family.set_status(unique_specific_token=True, unique_token_type=True, 
                                      meaningful = False)
        
        opt_params = [('p' + str(idx+1), p_range) for idx, p_range in enumerate(param_ranges)]
        token_params = OrderedDict([('power', (1, 1)),] + opt_params)
        # print(token_params)

        p_equality_fraction = 0.05 # fraction of allowed frequency interval, that is considered as the same
        opt_params_equality = {'p' + str(idx+1) : (p_range[1] - p_range[0]) / p_equality_fraction for idx, p_range in enumerate(param_ranges)}
        equal_params = {'power' : 0}
        equal_params.update(opt_params_equality)
        self._token_family.set_params(['v'], token_params, equal_params)
        self._token_family.set_evaluator(velocity_evaluator, [])
        grad_eval_labeled = {'p'+str(idx+1) : fun for idx, fun in enumerate(velocity_grad_evaluators)}
        self._token_family.set_deriv_evaluator(grad_eval_labeled, [])
