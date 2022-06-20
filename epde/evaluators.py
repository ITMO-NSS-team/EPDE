#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 13:41:07 2021

@author: mike_ubuntu
"""

import numpy as np
from abc import ABC
from typing import Callable, Union

import epde.globals as global_var
from epde.supplementary import factor_params_to_str

class EvaluatorTemplate(ABC):
    def __init__(self):
        pass
    
    def __call__(self, factor, structural : bool = False, **kwargs):
        raise NotImplementedError('Trying to call the method of an abstract class')

class CustomEvaluator(EvaluatorTemplate):
    def __init__(self, evaluation_functions : Union[Callable, dict], 
                 eval_fun_params_labels : Union[list, tuple, set], use_factors_grids : bool = True):
        if isinstance(evaluation_functions, dict):
            self.single_function_token = False
        else:
            self.single_function_token = True
            
        self.evaluation_functions = evaluation_functions
        self.use_factors_grids = use_factors_grids
        self.eval_fun_params_labels = eval_fun_params_labels
        
    def __call__(self, factor, structural : bool = False, **kwargs):
        if not self.single_function_token and factor.label not in self.evaluation_functions.keys():
            raise KeyError('The label of the token function does not match keys of the evaluator functions')
        if self.single_function_token:
            evaluation_function = self.evaluation_functions 
        else:
            evaluation_function = self.evaluation_functions[factor.label]
            
        eval_fun_kwargs = dict()
        for key in self.eval_fun_params_labels:
            for param_idx, param_descr in factor.params_description.items():
                if param_descr['name'] == key: eval_fun_kwargs[key] = factor.params[param_idx]
        
        grid_function = np.vectorize(lambda args: evaluation_function(*args, **eval_fun_kwargs))
                                     #, excluded=eval_fun_kwargs.keys())
                                     
        try:
            self.indexes_vect
        except AttributeError:
            self.indexes_vect = np.empty_like(factor.grids[0], dtype = object)
            for tensor_idx, _ in np.ndenumerate(factor.grids[0]):
                self.indexes_vect[tensor_idx] = tuple([grid[tensor_idx]
                                                        for grid in factor.grids])

        value = grid_function(self.indexes_vect)
        return value
        

def simple_function_evaluator(factor, structural : bool = False, **kwargs):
    '''

    Example of the evaluator of token values, that can be used for uploading values of stored functions from cache. Cases, when 
    this approach can be used, include evaluating derivatives, coordinates, etc.

    
    Parameters
    ----------

    factor : epde.factor.Factor object,
        Object, that represents a factor from the equation terms, for that we want to calculate the values.
        
    structural : bool,
        Mark, if the evaluated value will be used for discovering equation structure (True), or calculating coefficients (False)
        
    Returns
    ----------
    value : numpy.ndarray
        Vector of the evaluation of the token values, that can be used as target, or feature during the LASSO regression.
        
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

trig_eval_fun = {'cos' : lambda *grids, **kwargs: np.cos(kwargs['freq'] * grids[int(kwargs['dim'])]) ** kwargs['power'], 
                   'sin' : lambda *grids, **kwargs: np.sin(kwargs['freq'] * grids[int(kwargs['dim'])]) ** kwargs['power']}
inverse_eval_fun = lambda *grids, **kwargs: np.power(grids[int(kwargs['dim'])], - kwargs['power']) 

def const_eval_fun(*grids, **kwargs):
    return np.full_like(a = grids[0], fill_value = kwargs['value'])

def const_grad_fun(*grids, **kwargs):
    return np.ones_like(a = grids[0])


def velocity_heating_eval_fun(*grids, **kwargs):
    '''
    Assumption of the velocity field for two-dimensional heat equation with convetion.
    '''
    a = [kwargs['p' + str(idx*3+1)] * grids[0]**2 + kwargs['p' + str(idx*3 + 2)] * grids[0] + kwargs['p' + str(idx*3 + 3)] for idx in range(5)]
    alpha = np.exp(a[0] * grids[1] + a[1]); beta = a[2] * grids[1]**2 + a[3] * grids[1] + a[4]
    return alpha * beta

# Proof of concept, if works properly, replace with permutations approach to gradient construction

def vhef_grad_1(*grids, **kwargs):
    a = [kwargs['p' + str(idx*3+1)] * grids[0]**2 + kwargs['p' + str(idx*3 + 2)] * grids[0] + kwargs['p' + str(idx*3 + 3)] for idx in range(5)]
    alpha = np.exp(a[0] * grids[1] + a[1]); beta = a[2] * grids[1]**2 + a[3] * grids[1] + a[4]
    return grids[0]**2 * grids[1] * alpha * beta

def vhef_grad_2(*grids, **kwargs):
    a = [kwargs['p' + str(idx*3+1)] * grids[0]**2 + kwargs['p' + str(idx*3 + 2)] * grids[0] + kwargs['p' + str(idx*3 + 3)] for idx in range(5)]
    alpha = np.exp(a[0] * grids[1] + a[1]); beta = a[2] * grids[1]**2 + a[3] * grids[1] + a[4]
    return grids[0]  * grids[1] * alpha * beta

def vhef_grad_3(*grids, **kwargs):
    a = [kwargs['p' + str(idx*3+1)] * grids[0]**2 + kwargs['p' + str(idx*3 + 2)] * grids[0] + kwargs['p' + str(idx*3 + 3)] for idx in range(5)]
    alpha = np.exp(a[0] * grids[1] + a[1]); beta = a[2] * grids[1]**2 + a[3] * grids[1] + a[4]
    return grids[1] * alpha * beta

def vhef_grad_4(*grids, **kwargs):
    a = [kwargs['p' + str(idx*3+1)] * grids[0]**2 + kwargs['p' + str(idx*3 + 2)] * grids[0] + kwargs['p' + str(idx*3 + 3)] for idx in range(5)]
    alpha = np.exp(a[0] * grids[1] + a[1]); beta = a[2] * grids[1]**2 + a[3] * grids[1] + a[4]
    return grids[0]**2 * alpha * beta

def vhef_grad_5(*grids, **kwargs):
    a = [kwargs['p' + str(idx*3+1)] * grids[0]**2 + kwargs['p' + str(idx*3 + 2)] * grids[0] + kwargs['p' + str(idx*3 + 3)] for idx in range(5)]
    alpha = np.exp(a[0] * grids[1] + a[1]); beta = a[2] * grids[1]**2 + a[3] * grids[1] + a[4]
    return grids[0] * alpha * beta

def vhef_grad_6(*grids, **kwargs):
    a = [kwargs['p' + str(idx*3+1)] * grids[0]**2 + kwargs['p' + str(idx*3 + 2)] * grids[0] + kwargs['p' + str(idx*3 + 3)] for idx in range(5)]
    alpha = np.exp(a[0] * grids[1] + a[1]); beta = a[2] * grids[1]**2 + a[3] * grids[1] + a[4]
    return alpha * beta
    
def vhef_grad_7(*grids, **kwargs):
    a = [kwargs['p' + str(idx*3+1)] * grids[0]**2 + kwargs['p' + str(idx*3 + 2)] * grids[0] + kwargs['p' + str(idx*3 + 3)] for idx in range(5)]
    alpha = np.exp(a[0] * grids[1] + a[1])#; beta = a[2] * grids[1]**2 + a[3] * grids[1] + a[4]
    return grids[0]**2 * grids[1]**2 * alpha

def vhef_grad_8(*grids, **kwargs):
    a = [kwargs['p' + str(idx*3+1)] * grids[0]**2 + kwargs['p' + str(idx*3 + 2)] * grids[0] + kwargs['p' + str(idx*3 + 3)] for idx in range(5)]
    alpha = np.exp(a[0] * grids[1] + a[1])#; beta = a[2] * grids[1]**2 + a[3] * grids[1] + a[4]
    return grids[0] * grids[1]**2 * alpha

def vhef_grad_9(*grids, **kwargs):
    a = [kwargs['p' + str(idx*3+1)] * grids[0]**2 + kwargs['p' + str(idx*3 + 2)] * grids[0] + kwargs['p' + str(idx*3 + 3)] for idx in range(5)]
    alpha = np.exp(a[0] * grids[1] + a[1])#; beta = a[2] * grids[1]**2 + a[3] * grids[1] + a[4]
    return grids[1]**2 * alpha

def vhef_grad_10(*grids, **kwargs):
    a = [kwargs['p' + str(idx*3+1)] * grids[0]**2 + kwargs['p' + str(idx*3 + 2)] * grids[0] + kwargs['p' + str(idx*3 + 3)] for idx in range(5)]
    alpha = np.exp(a[0] * grids[1] + a[1])#; beta = a[2] * grids[1]**2 + a[3] * grids[1] + a[4]
    return grids[0]**2 * grids[1] * alpha

def vhef_grad_11(*grids, **kwargs):
    a = [kwargs['p' + str(idx*3+1)] * grids[0]**2 + kwargs['p' + str(idx*3 + 2)] * grids[0] + kwargs['p' + str(idx*3 + 3)] for idx in range(5)]
    alpha = np.exp(a[0] * grids[1] + a[1])#; beta = a[2] * grids[1]**2 + a[3] * grids[1] + a[4]
    return grids[0] * grids[1] * alpha

def vhef_grad_12(*grids, **kwargs):
    a = [kwargs['p' + str(idx*3+1)] * grids[0]**2 + kwargs['p' + str(idx*3 + 2)] * grids[0] + kwargs['p' + str(idx*3 + 3)] for idx in range(5)]
    alpha = np.exp(a[0] * grids[1] + a[1])#; beta = a[2] * grids[1]**2 + a[3] * grids[1] + a[4]
    return grids[1] * alpha

def vhef_grad_13(*grids, **kwargs):
    a = [kwargs['p' + str(idx*3+1)] * grids[0]**2 + kwargs['p' + str(idx*3 + 2)] * grids[0] + kwargs['p' + str(idx*3 + 3)] for idx in range(5)]
    alpha = np.exp(a[0] * grids[1] + a[1])#; beta = a[2] * grids[1]**2 + a[3] * grids[1] + a[4]
    return grids[0]**2 * alpha

def vhef_grad_14(*grids, **kwargs):
    a = [kwargs['p' + str(idx*3+1)] * grids[0]**2 + kwargs['p' + str(idx*3 + 2)] * grids[0] + kwargs['p' + str(idx*3 + 3)] for idx in range(5)]
    alpha = np.exp(a[0] * grids[1] + a[1])#; beta = a[2] * grids[1]**2 + a[3] * grids[1] + a[4]
    return grids[0] * alpha

def vhef_grad_15(*grids, **kwargs):
    a = [kwargs['p' + str(idx*3+1)] * grids[0]**2 + kwargs['p' + str(idx*3 + 2)] * grids[0] + kwargs['p' + str(idx*3 + 3)] for idx in range(5)]
    alpha = np.exp(a[0] * grids[1] + a[1])#; beta = a[2] * grids[1]**2 + a[3] * grids[1] + a[4]
    return alpha

vhef_grad = [vhef_grad_1, vhef_grad_2, vhef_grad_3,
             vhef_grad_4, vhef_grad_5, vhef_grad_6,
             vhef_grad_7, vhef_grad_8, vhef_grad_9,
             vhef_grad_10, vhef_grad_11, vhef_grad_12,
             vhef_grad_13, vhef_grad_14, vhef_grad_15]

trigonometric_evaluator = CustomEvaluator(trig_eval_fun, eval_fun_params_labels = ['freq', 'dim', 'power'])

inverse_function_evaluator = CustomEvaluator(inverse_eval_fun, eval_fun_params_labels = ['dim', 'power'], use_factors_grids = True)    

const_evaluator = CustomEvaluator(const_eval_fun, ['power', 'value']) 
const_grad_evaluator = CustomEvaluator(const_grad_fun, ['power', 'value']) 

velocity_evaluator = CustomEvaluator(velocity_heating_eval_fun, ['p' + str(idx+1) for idx in range(15)]) 
velocity_grad_evaluators = [CustomEvaluator(component, ['p' + str(idx+1) for idx in range(15)]) 
                            for component in vhef_grad] 


#def trigonometric_evaluator(factor, structual = False, **kwargs):
#    
#    '''
#    
#    Example of the evaluator of token values, appropriate for case of trigonometric functions to be calculated on grid, with results in forms of tensors
#    
#    Parameters
#    ----------
#
#    factor : epde.factor.Factor object,
#        Object, that represents a factor from the equation terms, for that we want to calculate the values.
#        
#    structural : bool,
#        Mark, if the evaluated value will be used for discovering equation structure (True), or calculating coefficients (False)
#        
#    Returns
#    ----------
#    value : numpy.ndarray
#        Vector of the evaluation of the token values, that can be used as target, or feature during the LASSO regression.
#        
#    '''
#    
#    assert factor.grid_set, 'Evaluation grid is not defined for the trigonometric token'
#    trig_functions = {'sin' : np.sin, 'cos' : np.cos}
#    function = trig_functions[factor.label]
#    for param_idx, param_descr in factor.params_description.items():
#        if param_descr['name'] == 'freq': freq_param_idx = param_idx
#        if param_descr['name'] == 'dim': dim_param_idx = param_idx
#        if param_descr['name'] == 'power': power_param_idx = param_idx
#    grid_function = np.vectorize(lambda *args: function(factor.params[freq_param_idx] * #args[int(factor.params[dim_param_idx])]
#                                                        args[int(factor.params[dim_param_idx])])**factor.params[power_param_idx])
#    value = grid_function(factor.grids)#[int(factor.params[dim_param_idx])]
#    return value


#def inverse_function_evaluator(factor, structual = False, **kwargs):
#    
#    '''
#    
#    Example of the evaluator of token values, appropriate for case of inverse functions of grid values to be calculated on grid, 
#    with results in forms of tensors. 
#    
#    Parameters
#    ----------
#
#    factor : epde.factor.Factor object,
#        Object, that represents a factor from the equation terms, for that we want to calculate the values.
#        
#    structural : bool,
#        Mark, if the evaluated value will be used for discovering equation structure (True), or calculating coefficients (False)
#        
#    Returns
#    ----------
#    value : numpy.ndarray
#        Vector of the evaluation of the token values, that can be used as target, or feature during the LASSO regression.
#        
#    '''
#    
#    assert factor.grid_set, 'Evaluation grid is not defined for the trigonometric token'
#    for param_idx, param_descr in factor.params_description.items():
#        if param_descr['name'] == 'dim': dim_param_idx = param_idx
#        if param_descr['name'] == 'power': power_param_idx = param_idx
#    grid_function = np.vectorize(lambda *args: pow(args[int(factor.params[dim_param_idx])], - factor.params[power_param_idx]))
#    value = grid_function(factor.grids)#[int(factor.params[dim_param_idx])]
#    return value
