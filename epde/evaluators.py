#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 13:41:07 2021

@author: mike_ubuntu
"""

import numpy as np
import epde.globals as global_var
from epde.supplementary import factor_params_to_str

def simple_function_evaluator(factor, structural = False, **kwargs):
    '''
    
    Example of the evaluator of token values, appropriate for case of trigonometric functions to be calculated on grid, with results in forms of tensors
    
    OLD DESCRIPTION
    
    Parameters
    ----------
    token: {'u', 'du/dx', ...}
        symbolic form of the function to be evaluated: 
    token_params: dictionary: key - symbolic form of the parameter, value - parameter value
        names and values of the parameters for trigonometric functions: amplitude, frequency & dimension
    eval_params : dict
        Dictionary, containing parameters of the evaluator: in this example, it contains coordinates np.ndarray with pre-calculated values of functions, 
        names of the token parameters (power). Additionally, the names of the token parameters must be included with specific key 'params_names', 
        and parameters range, for which each of the tokens if consedered as "equal" to another, like sin(1.0001 x) can be assumed as equal to (0.9999 x)
    
    Returns
    ----------
    value : numpy.ndarray
        Vector of the evaluation of the token values, that shall be used as target, or feature during the LASSO regression.
        
    '''
    
    for param_idx, param_descr in factor.params_description.items():
        if param_descr['name'] == 'power': power_param_idx = param_idx
        
    if factor.params[power_param_idx] == 1:
        value = global_var.tensor_cache.get(factor.cache_label, structural = structural)
        return value
    else:
        value = global_var.tensor_cache.get(factor_params_to_str(factor, set_default_power = True, power_idx = power_param_idx), 
                                            structural = structural)
        value = value**(factor.params[power_param_idx])
        return value


def trigonometric_evaluator(factor, structual = False, **kwargs):
    
    '''
    
    Example of the evaluator of token values, appropriate for case of trigonometric functions to be calculated on grid, with results in forms of tensors
    
    OLD DESCRIPTION
    
    Parameters
    ----------
    token: {'sin', 'cos'}
        symbolic form of the function to be evaluated: 
    token_params: dictionary: key - symbolic form of the parameter, value - parameter value
        names and values of the parameters for trigonometric functions: amplitude, frequency & dimension
    eval_params : dict
        Dictionary, containing parameters of the evaluator: in this example, it contains coordinates np.meshgrid with coordinates for points, 
        names of the token parameters (frequency, axis and power). Additionally, the names of the token parameters must be included with specific key 'params_names', 
        and parameters range, for which each of the tokens if consedered as "equal" to another, like sin(1.0001 x) can be assumed as equal to (0.9999 x)
    
    Returns
    ----------
    value : numpy.ndarray
        Vector of the evaluation of the token values, that shall be used as target, or feature during the LASSO regression.
        
    '''
    
    assert factor.grid_set, 'Evaluation grid is not defined for the trigonometric token'
    trig_functions = {'sin' : np.sin, 'cos' : np.cos}
    function = trig_functions[factor.label]
    for param_idx, param_descr in factor.params_description.items():
        if param_descr['name'] == 'freq': freq_param_idx = param_idx
        if param_descr['name'] == 'dim': dim_param_idx = param_idx
        if param_descr['name'] == 'power': power_param_idx = param_idx
    grid_function = np.vectorize(lambda *args: function(factor.params[freq_param_idx] * #args[int(factor.params[dim_param_idx])]
                                                        args[int(factor.params[dim_param_idx])])**factor.params[power_param_idx])
    value = grid_function(factor.grids)#[int(factor.params[dim_param_idx])]
    return value