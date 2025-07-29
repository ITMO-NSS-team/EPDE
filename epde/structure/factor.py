#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 13:16:43 2020

@author: mike_ubuntu
"""

import numpy as np
import copy
import torch
from typing import Callable
try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable

import epde.globals as global_var
from epde.structure.Tokens import TerminalToken
from epde.supplementary import factor_params_to_str, train_ann, use_ann_to_predict, exp_form
from epde.evaluators import simple_function_evaluator

class EvaluatorContained(object):
    """
    Class for evaluator of token (factor of the term in the sought equation) values with arbitrary function

    Attributes:
        _evaluator (`callable`): a function, which returns the vector of token values, evaluated on the studied area;
        params (`dict`): dictionary, containing parameters of the evaluator (like grid, on which the function is evaluated or matrices of pre-calculated function)

    Methods:
        set_params(**params)
            set the parameters of the evaluator, using keyword arguments
        apply(token, token_params)
            apply the defined evaluator to evaluate the token with specific parameters
    """

    def __init__(self, eval_function): # , eval_kwargs_keys={}
        self._evaluator = eval_function
        # self.eval_kwargs_keys = eval_kwargs_keys

    def apply(self, token, structural=False, func_args=None, torch_mode=False): # , **kwargs
        """
        Apply the defined evaluator to evaluate the token with specific parameters.

        Args:
            token (`epde.main_structures.factor.Factor`): symbolic label of the specific token, e.g. 'cos';
        token_params (`dict`): dictionary with keys, naming the token parameters (such as frequency, axis and power for trigonometric function) 
            and values - specific values of corresponding parameters.

        Raises:
            `TypeError`
                If the evaluator could not be applied to the token.
        """
        # assert list(kwargs.keys()) == self.eval_kwargs_keys, f'Kwargs {kwargs.keys()} != {self.eval_kwargs_keys}'
        return self._evaluator(token, structural, func_args, torch_mode = torch_mode)


class Factor(TerminalToken):
    __slots__ = ['_params', '_params_description', '_hash_val', '_latex_constructor', 'label',
                 'ftype', '_variable', '_all_vars', 'grid_set', 'grid_idx', 'is_deriv', 'deriv_code',
                 'cache_linked', '_status', 'equality_ranges', '_evaluator', 'saved']

    def __init__(self, token_name: str, status: dict, family_type: str, latex_constructor: Callable,
                 variable: str = None, all_vars: list = None, randomize: bool = False, 
                 params_description=None, deriv_code=None, equality_ranges = None):
        self.label = token_name
        self.ftype = family_type
        self._variable = variable
        self._all_vars = all_vars
        
        self.status = status
        self.grid_set = False
        self._hash_val = np.random.randint(0, 1e9)
        self._latex_constructor = latex_constructor

        self.is_deriv = not (deriv_code is None)
        self.deriv_code = deriv_code

        self.reset_saved_state()
        if global_var.tensor_cache is not None:
            self.use_cache()
        else:
            self.cache_linked = False

        if randomize:
            assert params_description is not None and equality_ranges is not None
            self.set_parameters(params_description,
                                equality_ranges, random=True)

            if self.status['requires_grid']:
                self.use_grids_cache()
    
    @property
    def variable(self):
        if self._variable is None:
            return self.ftype
        else:
            return self._variable
        
    def manual_reconst(self, attribute:str, value, except_attrs:dict):
        from epde.loader import obj_to_pickle, attrs_from_dict        
        supported_attrs = []
        if attribute not in supported_attrs:
            raise ValueError(f'Attribute {attribute} is not supported by manual_reconst method.')

    @property
    def ann_representation(self) -> torch.nn.modules.container.Sequential:
        try:
            return self._ann_repr
        except AttributeError:
            _, grids = global_var.grid_cache.get_all()
            self._ann_repr = train_ann(grids = grids, data=self.evaluate())
            return self._ann_repr

    def predict_with_ann(self, grids: list):
        return use_ann_to_predict(self.ann_representation, grids)

    def reset_saved_state(self):
        self.saved = {'base': False, 'structural': False}

    @property
    def status(self):
        return self._status

    @status.setter
    def status(self, status_dict):
        '''
        Parameters
        ----------
        status_dict : dict
            Description of token behaviour during the equation construction and processsing.
            Keys:
                'mandatory' - if True, a token from the family must be present in every term; 

                'unique_token_type' - if True, only one token of the family can be present in the term; 

                'unique_specific_token' - if True, a specific token can be present only once per term;            

                'requires_grid' - if True, the token requires grid for evaluation, if False, the tokens will be
                loaded from cache.
        '''
        self._status = status_dict

    def set_parameters(self, params_description: dict, equality_ranges: dict,
                       random=True, **kwargs):
        '''

        Avoid periodic parameters (e.g. phase shift) 

        '''
        _params_description = {}
        if not random:
            _params = np.empty(len(kwargs))
            if len(kwargs) != len(params_description):
                print('Not all parameters have been declared. Partial randomization TBD')
                print(f'kwargs {kwargs}, while params_descr {params_description}')
                raise ValueError('...')
            for param_idx, param_info in enumerate(kwargs.items()):
                _params[param_idx] = param_info[1]
                _params_description[param_idx] = {'name': param_info[0],
                                                  'bounds': params_description[param_info[0]]}
        else:
            _params = np.empty(len(params_description))
            for param_idx, param_info in enumerate(params_description.items()):
                if param_info[0] != 'power' or self.status['non_default_power']:
                    _params[param_idx] = (np.random.randint(param_info[1][0], param_info[1][1] + 1) if isinstance(param_info[1][0], int)
                                          else np.random.uniform(param_info[1][0], param_info[1][1])) if param_info[1][1] > param_info[1][0] else param_info[1][0]
                else:
                    _params[param_idx] = 1
                _params_description[param_idx] = {'name': param_info[0],
                                                  'bounds': param_info[1]}
        self.equality_ranges = equality_ranges
        super().__init__(number_params=_params.size, params_description=_params_description,
                         params=_params)
        if not self.grid_set:
            self.use_grids_cache()

    def __eq__(self, other):
        if type(self) != type(other):
            return False
        elif self.label != other.label:
            return False
        elif any([abs(self.params[idx] - other.params[idx]) > self.equality_ranges[self.params_description[idx]['name']]
                  for idx in np.arange(self.params.size)]):
            return False
        else:
            return True
        
    def partial_equlaity(self, other):
        for param_idx, param_info in self.params_description.items():
            if param_info['name'] == 'power':
                power_idx = param_idx
                break
            
        if type(self) != type(other):
            return False
        elif self.label != other.label:
            return False
        elif any([abs(self.params[idx] - other.params[idx]) > self.equality_ranges[self.params_description[idx]['name']]
                  for idx in np.arange(self.params.size) if idx != power_idx]):
            return False
        else:
            return True

    @property
    def evaluator(self):
        return self._evaluator

    @evaluator.setter
    def evaluator(self, evaluator):
        if isinstance(evaluator, EvaluatorContained):
            self._evaluator = evaluator
        else:
            factor_family = [family for family in evaluator.families if family.ftype == self.ftype][0]
            self._evaluator = factor_family._evaluator # TODO: fix calling private attribute
            
    def evaluate(self, structural=False, grids=None, torch_mode: bool = False):
        assert self.cache_linked, 'Missing linked cache.'
        if self.is_deriv and grids is not None:
            raise Exception(
                'Derivatives have to evaluated on the initial grid')

        key = 'structural' if structural else 'base'
        if (self.cache_label, structural) in global_var.tensor_cache and grids is None:
            # print(f'Asking for {self.cache_label} in tmode {torch_mode}')
            # print(f'From numpy cache of {global_var.tensor_cache.memory_structural["numpy"].keys()}')
            # print(f'And torch cache of {global_var.tensor_cache.memory_structural["torch"].keys()}')

            return global_var.tensor_cache.get(self.cache_label,
                                               structural=structural, torch_mode = torch_mode)
 
        else:
            if self.is_deriv and self.evaluator._evaluator != simple_function_evaluator:
                if grids is not None:
                    raise Exception('Data-reliant tokens shall not get grids as arguments for evaluation.')
                if isinstance(self.variable, str):
                    var = self._all_vars.index(self.variable)
                    func_arg = [global_var.tensor_cache.get(label=None, torch_mode=torch_mode,
                                                            deriv_code=(var, self.deriv_code)),]
                elif isinstance(self.variable, (list, tuple)):
                    func_arg = []
                    for var_idx, code in enumerate(self.deriv_code):
                        assert len(self.variable) == len(self.deriv_code)
                        func_arg.append(global_var.tensor_cache.get(label=None, torch_mode=torch_mode,
                                                                    deriv_code=(self.variable[var_idx], code)))

                value = self.evaluator.apply(self, structural=structural, func_args=func_arg, torch_mode=torch_mode)
            else:
                value = self.evaluator.apply(self, structural=structural, func_args=grids, torch_mode=torch_mode)
            if grids is None:
                if self.is_deriv and self.evaluator._evaluator == simple_function_evaluator:
                    full_deriv_code = (self._all_vars.index(self.variable), self.deriv_code)
                else:
                    full_deriv_code = None      

                if key == 'structural' and self.status['structural_and_defalut_merged']:
                    self.saved[key] = global_var.tensor_cache.add(self.cache_label, value, structural=False, 
                                                                  deriv_code=full_deriv_code)                    
                    global_var.tensor_cache.use_structural(use_base_data=True,
                                                           label=self.cache_label)
                elif key == 'structural' and not self.status['structural_and_defalut_merged']:
                    global_var.tensor_cache.use_structural(use_base_data=False,
                                                           label=self.cache_label,
                                                           replacing_data=value)
                else:
                    self.saved[key] = global_var.tensor_cache.add(self.cache_label, value, structural=False, 
                                                                  deriv_code=full_deriv_code)
            return value

    @property
    def cache_label(self):
        cache_label = factor_params_to_str(self)
        return cache_label

    @property
    def name(self):
        form = self.label + '{'
        for param_idx, param_info in self.params_description.items():
            form += param_info['name'] + ': ' + str(self.params[param_idx])
            if param_idx < len(self.params_description.items()) - 1:
                form += ', '
        form += '}'
        return form

    @property
    def latex_name(self):
        if self._latex_constructor is not None:
            params_dict = {}
            for param_idx, param_info in self.params_description.items():
                mnt, exp = exp_form(self.params[param_idx], 3)
                exp_str = r'\cdot 10^{{{0}}} '.format(str(exp)) if exp != 0 else ''

                params_dict[param_info['name']] = (self.params[param_idx], str(mnt) + exp_str)
            return self._latex_constructor(self.label, **params_dict)
        else:
            return self.name # other implementations are possible
    
    @property
    def hash_descr(self) -> int:
        return self._hash_val

    @property
    def grids(self):
        _, grids = global_var.grid_cache.get_all()
        return grids

    def use_grids_cache(self):
        dim_param_idx = np.inf
        dim_set = False
        for param_idx, param_descr in self.params_description.items():
            if param_descr['name'] == 'dim':
                dim_param_idx = param_idx
                dim_set = True
        self.grid_idx = int(self.params[dim_param_idx]) if dim_set else 0
        self.grid_set = True

    def __deepcopy__(self, memo=None):
        clss = self.__class__
        new_struct = clss.__new__(clss)
        memo[id(self)] = new_struct

        new_struct.__dict__.update(self.__dict__)

        attrs_to_avoid_copy = []
        for k in self.__slots__:
            try:
                if k not in attrs_to_avoid_copy:
                    if not isinstance(k, list):
                        setattr(new_struct, k, copy.deepcopy(
                            getattr(self, k), memo))
                    else:
                        temp = []
                        for elem in getattr(self, k):
                            temp.append(copy.deepcopy(elem, memo))
                        setattr(new_struct, k, temp)
                else:
                    setattr(new_struct, k, None)
            except AttributeError:
                pass

        return new_struct

    def use_cache(self):
        self.cache_linked = True
