#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  5 13:41:07 2021

@author: mike_ubuntu
"""

import numpy as np
import torch
# device = torch.device('cpu')

from abc import ABC, abstractmethod
from typing import Callable, Union, List, Dict

import epde.globals as global_var
from epde.supplementary import factor_params_to_str

class EvaluatorTemplate(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def __call__(self, factor, grids: list = None, 
                 torch_mode: bool = False, **kwargs):
        raise NotImplementedError(
            'Trying to call the method of an abstract class')


class CustomEvaluator(EvaluatorTemplate):
    def __init__(self, evaluation_functions_np: Union[Callable, dict],
                 eval_fun_params_labels: Union[list, tuple, set] = ['power'],
                 native_vectorized: bool = False):
        """Wrap one or many evaluation functions for use as a factor evaluator.

        ``native_vectorized=True`` skips the per-element ``np.vectorize``
        dispatch on the hot path: the func is called ONCE with the full
        grid arrays. The built-in evaluators in this module all set this
        flag because their numpy ops (``np.cos``, ``np.sin``, ``np.power``,
        ``np.full_like``, etc.) vectorize natively. User code passing a
        non-vectorising callable should leave the default ``False``.
        """
        self._evaluation_functions_np = evaluation_functions_np
        self.indexes_vect = {}

        # if evaluation_functions_np is None:
        #     raise ValueError('No evaluation function set in the initialization of CustomEvaluator.')

        if isinstance(evaluation_functions_np, dict):
            self._single_function_token = False
        else:
            self._single_function_token = True

        self.eval_fun_params_labels = eval_fun_params_labels
        self.native_vectorized = native_vectorized

    def __call__(self, factor, func_args: Dict[int, List[np.ndarray]] = None, 
                 **kwargs) -> Dict[int, np.ndarray]:
        # if torch_mode: # TODO: rewrite
        #     torch_mode_explicit = True
        if not self._single_function_token and factor.label not in self._evaluation_functions_np.keys():
            raise KeyError(
                'The label of the token function does not match keys of the evaluator functions')
        # if func_args is not None:
        #     if isinstance(func_args[0], np.ndarray) or self._evaluation_functions_torch is None:
        #         funcs = self._evaluation_functions_np if self._single_function_token else self._evaluation_functions_np[factor.label]
        #     elif isinstance(func_args[0], torch.Tensor) or self._evaluation_functions_np is None or torch_mode_explicit:
        #         funcs = self._evaluation_functions_torch if self._single_function_token else self._evaluation_functions_torch[factor.label]
        # elif torch_mode:
        #     funcs = self._evaluation_functions_torch if self._single_function_token else self._evaluation_functions_torch[factor.label]
        # else:
        funcs = self._evaluation_functions_np if self._single_function_token else self._evaluation_functions_np[factor.label]

        eval_fun_kwargs = dict()
        for key in self.eval_fun_params_labels:
            for param_idx, param_descr in factor.params_description.items():
                if param_descr['name'] == key:
                    eval_fun_kwargs[key] = factor.params[param_idx]

        if func_args is None or all([all([grid_like is None for grid_like in sample]) for sample in func_args.values()]):
            new_grid = False
            func_args : Dict[int, List[np.ndarray]] = global_var.samples_manager.grids() # factor.grids
        else:
            assert isinstance(func_args, dict), \
                f'Arg. func_args for CustomEvaluator must be None or DICT of lists of np.ndarrays, got {type(func_args)}.'
            assert all([isinstance(func_arg, list) for func_arg in func_args.values()]), \
                f'Arg. func_args for CustomEvaluator must be None or dict of LISTS of np.ndarrays, got {func_args}.'
            assert all([all([(isinstance(arr, np.ndarray) or arr is None) for arr in func_arg]) for func_arg in func_args.values()]), \
                f'Arg. func_args for CustomEvaluator must be None or dict of lists of NP.NDARRAYS, got {func_args}.'

            new_grid = True

        gfunc_masks = global_var.samples_manager.gFunc('m')

        if self.native_vectorized:
            # Fast path: call funcs once with the full grid arrays. The
            # built-in numpy evaluators (trig, sign, grid, inverse,
            # const, velocity) all return an array of shape
            # ``func_args[0].shape``. This skips an N-element
            # ``np.vectorize`` loop that on Wave (65k samples)
            # dominated evaluator self-time at ~35 s per run.
            values = {sample_ID: funcs(*func_arg, **eval_fun_kwargs)[gfunc_masks[sample_ID]].reshape(-1)
                      for sample_ID, func_arg in func_args.items()}
            return values
        else:
            grid_function = np.vectorize(lambda args: funcs(*args, **eval_fun_kwargs))

            values = {}
            # Two single-trajectory assumptions survived the multisample port
            # here, on the path custom (non-vectorized) evaluators take:
            #   * ``sampleIDs`` never existed on TrajectoriesManager -- the
            #     property is ``trajecatoryIDs``. The identical slip is called
            #     out and fixed in integrate/interface.py; this occurrence was
            #     missed, so any CustomTokens evaluator raised AttributeError.
            #   * ``func_args[0]`` indexed the per-sample dict with a literal
            #     0 while iterating sample IDs, which only happened to work
            #     when a trajectory was keyed 0.
            for sample_ID in global_var.samples_manager.trajecatoryIDs:
                if sample_ID in self.indexes_vect.keys():
                    assert not new_grid, 'Trying to call pre-computed vectorized indexes, while a new grid is passed.'
                else:
                    self.indexes_vect[sample_ID] = np.empty_like(func_args[sample_ID][0], dtype=object)
                    for tensor_idx, _ in np.ndenumerate(func_args[sample_ID][0]):
                        self.indexes_vect[sample_ID][tensor_idx] = tuple([subarg[tensor_idx]
                                                                          for subarg in func_args[sample_ID]])
                
                values[sample_ID] = grid_function(self.indexes_vect[sample_ID])[gfunc_masks[sample_ID]].reshape(-1)
            return values



def simple_function_evaluator(factor,
                              # structural: bool = False,
                              grids: Union[Dict[int, List[np.ndarray]], List[np.ndarray]] = None,
                              **kwargs) -> Dict[int, np.ndarray]:
    '''

    Example of the evaluator of token values, that can be used for uploading values of stored functions from cache. Cases, when
    this approach can be used, include evaluating derivatives, coordinates, etc.


    Parameters
    ----------

    factor : epde.factor.Factor object,
        Object, that represents a factor from the equation terms, for that we want to calculate the values.

    structural : bool,
        Mark, if the evaluated value will be used for discovering equation structure (True), 
        or calculating coefficients (False)

    Returns
    ----------
    values : Dict[int, numpy.ndarray]
        Dict of vector of the evaluation of the token values, that can be used as target,
        or feature during the LASSO regression.

    '''

    for param_idx, param_descr in factor.params_description.items():
        if param_descr['name'] == 'power':
            power_param_idx = param_idx

    if isinstance(grids, dict):
        if all([isinstance(grid, list) for grid in grids.values()]):  
            if any([any([subgr is None for subgr in grid]) for grid in grids.values()]):
                none_cond = True
            else:
                assert all([isinstance(key, int) and isinstance(grid, (np.ndarray, list)) for key, grid in grids.items()]), \
                    f'Some domain keys and grids do not match desired types: \
                      got {[(type(key), type(grid)) for key, grid in grids.items]}.'
                none_cond = False
        elif all([isinstance(grid, np.ndarray) for grid in grids.values()]):
            if any([grid is None for grid in grids.values()]):
                none_cond = True
            else:
                assert all([isinstance(key, int) and isinstance(grid, np.ndarray) for key, grid in grids.items()]), \
                    f'Some domain keys and grids do not match desired types:  \
                      got {[(type(key), type(grid)) for key, grid in grids.items]}.'
                none_cond = False
        else:
            raise TypeError(f'Incorrect type of values in grids dict, \
                            expected np.ndarrays or list, got {[type(grid) for grid in grids.values()]}')            
    elif isinstance(grids, list):
        if any([grid is None for grid in grids]):
            none_cond = True
        else:
            assert all([isinstance(grid, np.ndarray) for grid in grids]), \
                f'Some domain reeval grids do not match desired np.ndarray type: got {[type(grid) for grid in grids]}.'
            none_cond = False
    else:
        assert grids is None, f'Non-default behavior, expected other grids, got {type(grids)}'
        none_cond = True

    if not none_cond: # grids is not None or any([for grid in ]):
        if isinstance(grids[0], np.ndarray):
            values = factor.predict_with_ann(grids)
            values = {-1: values**(factor.params[power_param_idx])}
        else:
            values = {}
            for key, domain_grids in grids.items():
                assert all([isinstance(domain_grid, np.ndarray) for domain_grid in domain_grids]), \
                    'In multiple grids evalutation, the grids have to be passed as a dict with values - lists of np.ndarrays!' 
                values[key] = factor.predict_with_ann(domain_grids)**(factor.params[power_param_idx])
        return values

    else:
        if factor.params[power_param_idx] == 1:
            # Same bucketed key Factor.evaluate uses so trig factors with
            # within-tolerance freq share a single cached evaluation.
            values = global_var.samples_manager.get(factor.structural_label) # , structural = structural
            return values
        else:
            values = global_var.samples_manager.get(factor_params_to_str(factor, set_default_power = True,
                                                                         power_idx = power_param_idx))
            values = {ID : value**(factor.params[power_param_idx]) for ID, value in values.items()}
            return values


sign_eval_fun_np = lambda *args, **kwargs: np.sign(args[0]) # If dim argument is needed here: int(kwargs['dim'])
# sign_eval_fun_torch = lambda *args, **kwargs: torch.sign(args[0])

trig_eval_fun_np = {'cos': lambda *grids, **kwargs: np.cos(kwargs['freq'] * grids[int(kwargs['dim'])]) ** kwargs['power'],
                    'sin': lambda *grids, **kwargs: np.sin(kwargs['freq'] * grids[int(kwargs['dim'])]) ** kwargs['power']}

# trig_eval_fun_torch = {'cos': lambda *grids, **kwargs: torch.cos(kwargs['freq'] * grids[int(kwargs['dim'])]) ** kwargs['power'],
#                        'sin': lambda *grids, **kwargs: torch.sin(kwargs['freq'] * grids[int(kwargs['dim'])]) ** kwargs['power']}

inverse_eval_fun_np = lambda *grids, **kwargs: np.power(grids[int(kwargs['dim'])], - kwargs['power'])
# inverse_eval_fun_torch = lambda *grids, **kwargs: torch.pow(grids[int(kwargs['dim'])], - kwargs['power'])

grid_eval_fun_np = lambda *grids, **kwargs: np.power(grids[int(kwargs['dim'])], kwargs['power'])
# grid_eval_fun_torch = lambda *grids, **kwargs: torch.pow(grids[int(kwargs['dim'])], kwargs['power'])

def phased_sine_np(*grids, **kwargs):
    coordwise_elems = [kwargs['freq'][dim] * 2*np.pi*(grids[dim] + kwargs['phase'][dim]) 
                       for dim in range(len(grids))]
    return np.power(np.sin(np.sum(coordwise_elems, axis = 0)), kwargs['power'])

# def phased_sine_torch(*grids, **kwargs):
#     coordwise_elems = [kwargs['freq'][dim] * 2*torch.pi*(grids[dim] + kwargs['phase'][dim]) 
#                        for dim in range(len(grids))]
#     return torch.pow(torch.sin(torch.sum(coordwise_elems, axis = 0)), kwargs['power'])    

def phased_sine_1d_np(*grids, **kwargs):
    coordwise_elems = kwargs['freq'] * 2*np.pi*(grids[0] + kwargs['phase']/kwargs['freq']) 
    return np.power(np.sin(coordwise_elems), kwargs['power'])

# def phased_sine_1d_torch(*grids, **kwargs):
#     coordwise_elems = kwargs['freq'] * 2*torch.pi*(grids[0] + kwargs['phase']/kwargs['freq']) 
#     return torch.pow(torch.sin(coordwise_elems), kwargs['power'])

def const_eval_fun_np(*grids, **kwargs):
    return np.full_like(a=grids[0], fill_value=kwargs['value'])

# def const_eval_fun_torch(*grids, **kwargs):
#     return torch.full_like(a=grids[0], fill_value=kwargs['value'])    

def const_grad_fun_np(*grids, **kwargs):
    return np.zeros_like(a=grids[0])

# def const_grad_fun_torch(*grids, **kwargs):
#     return torch.zeros_like(a=grids[0])

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

sign_evaluator = CustomEvaluator(evaluation_functions_np=sign_eval_fun_np,
                                eval_fun_params_labels = ['power', 'dim'],
                                native_vectorized=True)

phased_sine_evaluator = CustomEvaluator(evaluation_functions_np = phased_sine_1d_np,
                                        eval_fun_params_labels = ['power', 'freq', 'phase'],
                                        native_vectorized=True)

trigonometric_evaluator = CustomEvaluator(evaluation_functions_np = trig_eval_fun_np,
                                          eval_fun_params_labels=['freq', 'dim', 'power'],
                                          native_vectorized=True)

grid_evaluator = CustomEvaluator(evaluation_functions_np = grid_eval_fun_np,
                                 eval_fun_params_labels=['dim', 'power'],
                                 native_vectorized=True)

inverse_function_evaluator = CustomEvaluator(evaluation_functions_np = inverse_eval_fun_np,
                                             eval_fun_params_labels=['dim', 'power'],
                                             native_vectorized=True)

const_evaluator = CustomEvaluator(evaluation_functions_np = const_eval_fun_np,
                                  eval_fun_params_labels = ['power', 'value'],
                                  native_vectorized=True)

const_grad_evaluator = CustomEvaluator(evaluation_functions_np = const_grad_fun_np,
                                       eval_fun_params_labels = ['power', 'value'],
                                       native_vectorized=True)

velocity_evaluator = CustomEvaluator(velocity_heating_eval_fun, ['p' + str(idx+1) for idx in range(15)])

velocity_grad_evaluators = [CustomEvaluator(component, ['p' + str(idx+1) for idx in range(15)])
                            for component in vhef_grad]
