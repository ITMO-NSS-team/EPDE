#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 13:41:07 2021

@author: mike_ubuntu
"""

import numpy as np
import torch
device = torch.device('cpu')

from abc import ABC
from typing import Callable, Union, List, Tuple

import epde.globals as global_var
from epde.supplementary import factor_params_to_str

class EvaluatorTemplate(ABC):
    def __init__(self):
        pass

    def __call__(self, factor, structural: bool = False, grids: list = None, **kwargs):
        raise NotImplementedError(
            'Trying to call the method of an abstract class')


class CustomEvaluator(EvaluatorTemplate):
    def __init__(self, evaluation_functions_np: Union[Callable, dict] = None, evaluation_functions_torch: Union[Callable, dict] = None,
                 eval_fun_params_labels: Union[list, tuple, set] = ['power'], use_factors_grids: bool = True):
        self._evaluation_functions_np = evaluation_functions_np
        self._evaluation_functions_torch = evaluation_functions_torch

        if (evaluation_functions_np is None) and (evaluation_functions_torch is None):
            raise ValueError('No evaluation function set in the initialization of CustomEvaluator.')

        if isinstance(evaluation_functions_np, dict):
            self._single_function_token = False
        else:
            self._single_function_token = True

        self.use_factors_grids = use_factors_grids
        self.eval_fun_params_labels = eval_fun_params_labels

    def __call__(self, factor, grids: List[torch.Tensor, np.ndarray] = None, **kwargs): # structural: bool = False, 
        if not self.single_function_token and factor.label not in self.evaluation_functions.keys():
            raise KeyError(
                'The label of the token function does not match keys of the evaluator functions')
        if isinstance(grids[0], np.ndarray) or self._evaluation_functions_torch is None:
            funcs = self._evaluation_functions_np if self._single_function_token else self._evaluation_functions_np[factor.label]
        elif isinstance(grids[0], torch.Tensor) or self._evaluation_functions_np is None:
            funcs = self._evaluation_functions_torch if self._single_function_token else self._evaluation_functions_torch[factor.label]

        # if self.single_function_token:
        #     evaluation_function = self.evaluation_functions
        # else:
        #     evaluation_function = self.evaluation_functions[factor.label]

        eval_fun_kwargs = dict()
        for key in self.eval_fun_params_labels:
            for param_idx, param_descr in factor.params_description.items():
                if param_descr['name'] == key:
                    eval_fun_kwargs[key] = factor.params[param_idx]

        grid_function = np.vectorize(lambda args: funcs(*args, **eval_fun_kwargs))

        if grids is None:
            new_grid = False
            grids = factor.grids
        else:
            new_grid = True
        try:
            if new_grid:
                raise AttributeError
            self.indexes_vect
        except AttributeError:
            self.indexes_vect = np.empty_like(grids[0], dtype=object)
            for tensor_idx, _ in np.ndenumerate(grids[0]):
                self.indexes_vect[tensor_idx] = tuple([grid[tensor_idx]
                                                       for grid in grids])

        value = grid_function(self.indexes_vect)
        return value


def simple_function_evaluator(factor, structural: bool = False, grids=None, **kwargs):
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
        if param_descr['name'] == 'power':
            power_param_idx = param_idx
            
    if grids is not None:
        # base_val = global_var.tensor_cache.get(factor.cache_label, structural=structural)

        value = factor.predict_with_ann(grids)
        value = value**(factor.params[power_param_idx])
        return value

    else:
        if factor.params[power_param_idx] == 1:
            value = global_var.tensor_cache.get(factor.cache_label, structural=structural)
            return value
        else:
            value = global_var.tensor_cache.get(factor_params_to_str(factor, set_default_power=True, power_idx=power_param_idx),
                                                structural=structural)
            value = value**(factor.params[power_param_idx])
            return value

def sign_function_evaluator(factor, structural: bool = False, grids=None, **kwargs):
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
        if param_descr['name'] == 'power':
            power_param_idx = param_idx
            
    assert factor.params[power_param_idx] == 1, 'Power of sgn function token must be "1".'
    if grids is not None:
        # base_val = global_var.tensor_cache.get(factor.cache_label, structural=structural)

        value = factor.predict_with_ann(grids)
        return np.sign(value)

    else:
        value = global_var.tensor_cache.get(factor.cache_label, structural=structural)
        return np.sign(value)



trig_eval_fun = {'cos': lambda *grids, **kwargs: np.cos(kwargs['freq'] * grids[int(kwargs['dim'])]) ** kwargs['power'],
                 'sin': lambda *grids, **kwargs: np.sin(kwargs['freq'] * grids[int(kwargs['dim'])]) ** kwargs['power']}

inverse_eval_fun = lambda *grids, **kwargs: np.power(grids[int(kwargs['dim'])], - kwargs['power'])

grid_eval_fun = lambda *grids, **kwargs: np.power(grids[int(kwargs['dim'])], kwargs['power'])

def phased_sine(*grids, **kwargs):
    coordwise_elems = [kwargs['freq'][dim] * 2*np.pi*(grids[dim] + kwargs['phase'][dim]) 
                       for dim in range(len(grids))]
    return np.power(np.sin(np.sum(coordwise_elems, axis = 0)), kwargs['power'])

def phased_sine_1d(*grids, **kwargs):
    coordwise_elems = kwargs['freq'] * 2*np.pi*(grids[0] + kwargs['phase']/kwargs['freq']) 
    return np.power(np.sin(coordwise_elems), kwargs['power'])

def const_eval_fun(*grids, **kwargs):
    return np.full_like(a=grids[0], fill_value=kwargs['value'])

def const_grad_fun(*grids, **kwargs):
    return np.zeros_like(a=grids[0])

def get_velocity_common(*grids, **kwargs):
    a = [kwargs['p' + str(idx*3+1)] * grids[0]**2 + kwargs['p' + str(idx*3 + 2)] * grids[0] + kwargs['p' + str(idx*3 + 3)] for idx in range(5)]
    alpha = np.exp(a[0] * grids[1] + a[1]); beta = a[2] * grids[1]**2 + a[3] * grids[1] + a[4]
    return alpha, beta

def velocity_heating_eval_fun(*grids, **kwargs):
    '''
    Assumption of the velocity field for two-dimensional heat equation with convetion.
    '''
    alpha, beta = get_velocity_common(*grids, **kwargs)
    return alpha * beta

def vhef_grad_1(*grids, **kwargs):
    alpha, beta = get_velocity_common(*grids, **kwargs)
    return grids[0]**2 * grids[1] * alpha * beta

def vhef_grad_2(*grids, **kwargs):
    alpha, beta = get_velocity_common(*grids, **kwargs)
    return grids[0]  * grids[1] * alpha * beta

def vhef_grad_3(*grids, **kwargs):
    alpha, beta = get_velocity_common(*grids, **kwargs)
    return grids[1] * alpha * beta

def vhef_grad_4(*grids, **kwargs):
    alpha, beta = get_velocity_common(*grids, **kwargs)
    return grids[0]**2 * alpha * beta

def vhef_grad_5(*grids, **kwargs):
    alpha, beta = get_velocity_common(*grids, **kwargs)
    return grids[0] * alpha * beta

def vhef_grad_6(*grids, **kwargs):
    alpha, beta = get_velocity_common(*grids, **kwargs)
    return alpha * beta

def vhef_grad_7(*grids, **kwargs):
    alpha, beta = get_velocity_common(*grids, **kwargs)
    return grids[0]**2 * grids[1]**2 * alpha

def vhef_grad_8(*grids, **kwargs):
    alpha, beta = get_velocity_common(*grids, **kwargs)
    return grids[0] * grids[1]**2 * alpha

def vhef_grad_9(*grids, **kwargs):
    alpha, beta = get_velocity_common(*grids, **kwargs)
    return grids[1]**2 * alpha

def vhef_grad_10(*grids, **kwargs):
    alpha, beta = get_velocity_common(*grids, **kwargs)
    return grids[0]**2 * grids[1] * alpha

def vhef_grad_11(*grids, **kwargs):
    alpha, beta = get_velocity_common(*grids, **kwargs)
    return grids[0] * grids[1] * alpha

def vhef_grad_12(*grids, **kwargs):
    alpha, beta = get_velocity_common(*grids, **kwargs)
    return grids[1] * alpha

def vhef_grad_13(*grids, **kwargs):
    alpha, beta = get_velocity_common(*grids, **kwargs)
    return grids[0]**2 * alpha

def vhef_grad_14(*grids, **kwargs):
    alpha, beta = get_velocity_common(*grids, **kwargs)
    return grids[0] * alpha

def vhef_grad_15(*grids, **kwargs):
    alpha, beta = get_velocity_common(*grids, **kwargs)
    return alpha


vhef_grad = [vhef_grad_1, vhef_grad_2, vhef_grad_3,
             vhef_grad_4, vhef_grad_5, vhef_grad_6,
             vhef_grad_7, vhef_grad_8, vhef_grad_9,
             vhef_grad_10, vhef_grad_11, vhef_grad_12,
             vhef_grad_13, vhef_grad_14, vhef_grad_15]

phased_sine_evaluator = CustomEvaluator(phased_sine_1d, eval_fun_params_labels=['power', 'freq', 'phase'], use_factors_grids=True)
trigonometric_evaluator = CustomEvaluator(trig_eval_fun, eval_fun_params_labels=['freq', 'dim', 'power'], use_factors_grids=True)
grid_evaluator = CustomEvaluator(grid_eval_fun, eval_fun_params_labels=['dim', 'power'], use_factors_grids=True)

inverse_function_evaluator = CustomEvaluator(inverse_eval_fun, eval_fun_params_labels=['dim', 'power'], use_factors_grids=True)

const_evaluator = CustomEvaluator(const_eval_fun, ['power', 'value'])
const_grad_evaluator = CustomEvaluator(const_grad_fun, ['power', 'value'])

velocity_evaluator = CustomEvaluator(velocity_heating_eval_fun, ['p' + str(idx+1) for idx in range(15)])
velocity_grad_evaluators = [CustomEvaluator(component, ['p' + str(idx+1) for idx in range(15)])
                            for component in vhef_grad]