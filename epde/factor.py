#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 13:16:43 2020

@author: mike_ubuntu
"""

import numpy as np
import copy

from epde.Tokens import TerminalToken
import epde.globals as global_var
from epde.supplementary import factor_params_to_str

class Factor(TerminalToken):
    __slots__ = ['_params', '_params_description', 
                 'label', 'type', 'grid_set', 'grid_idx', 'is_deriv', 'deriv_code', 
                 'cache_linked', '_status', 'equality_ranges', '_evaluator', 'saved']
    
    def __init__(self, token_name : str, status : dict, family_type : str, 
                 randomize : bool = False, params_description = None, 
                 deriv_code = None, equality_ranges = None):#, token_family, randomize = False):
        self.label = token_name
        self.type = family_type
        self.status = status
        self.grid_set = False
        
        
        self.is_deriv = not (deriv_code is None)
        self.deriv_code = deriv_code
        
        self.reset_saved_state()
        if global_var.tensor_cache is not None:
            self.use_cache()
        else:
            self.cache_linked = False

        if randomize:
            assert params_description is not None and equality_ranges is not None
            self.set_parameters(params_description, equality_ranges, random = True)
    
            if self.status['requires_grid']:
                self.use_grids_cache()

    def reset_saved_state(self):
        self.saved = {'base':False, 'structural':False}

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
                
                'unique_for_right_part' - if True, the tokens, present in the "right part" of the equation, can not 
                be present in the terms of the "left part". Recommended to select "True", if any token of the 
                familiy can have 0 values on the majority of studied area;
            
                'unique_specific_token' - if True, a specific token can be present only once per term;            

                'requires_grid' - if True, the token requires grid for evaluation, if False, the tokens will be
                loaded from cache.
        '''
        self._status = status_dict
        
    def set_parameters(self, params_description : dict, equality_ranges : dict, 
                       random = True, **kwargs):
        '''
        
        Avoid periodic parameters (e.g. phase shift) 
        
        '''
        _params_description = {}
        if not random:
#            assert set(self.token._evaluator.params['params_names']) != set(kwargs.keys()), 'Incorrect/partial set of parameters used to define factor'
#                raise Exception()
            _params = np.empty(len(kwargs))
            assert len(kwargs) == len(params_description), 'Not all parameters have been declared. Partial randomization TBD'
            for param_idx, param_info in enumerate(kwargs.items()): #param_name, param_val 
                _params[param_idx] = param_info[1]
                _params_description[param_idx] = {'name' : param_info[0], 
                                                          'bounds' : params_description[param_info[0]]} 
        else:
            _params = np.empty(len(params_description))#OrderedDict()
            for param_idx, param_info in enumerate(params_description.items()):
                if param_info[0] != 'power':
                    _params[param_idx] = (np.random.randint(param_info[1][0], param_info[1][1] + 1) if isinstance(param_info[1][0], int) 
                    else np.random.uniform(param_info[1][0], param_info[1][1])) if param_info[1][1] > param_info[1][0] else param_info[1][0]
                else:
                    _params[param_idx] = 1
                _params_description[param_idx] = {'name' : param_info[0], 
                                                      'bounds' : param_info[1]} 
        self.equality_ranges = equality_ranges    
        super().__init__(number_params = _params.size, params_description = _params_description, 
                         params = _params)
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
    
    def __call__(self):
        '''
        Return vector of evaluated values
        '''
        raise NotImplementedError('Delete me')
        return self.evaluate(self)    
    
    def set_evaluator(self, evaluator):
        self._evaluator = evaluator
    
    def evaluate(self, structural = False): # Переработать/удалить __call__, т.к. его функции уже тут
        assert self.cache_linked
        key = 'structural' if structural else 'base'
        if self.saved[key]:
            return global_var.tensor_cache.get(self.cache_label,
                                               structural = structural)
        else:
            value = self._evaluator.apply(self)
            if key == 'structural' and self.status['structural_and_defalut_merged']:
                global_var.tensor_cache.use_structural(use_base_data = True)
            elif key == 'structural' and not self.status['structural_and_defalut_merged']:
                global_var.tensor_cache.use_structural(use_base_data = False, 
                                                       label = self.cache_label,
                                                       replacing_data = value)            
            else:
                self.saved[key] = global_var.tensor_cache.add(self.cache_label, value, structural = False)
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
    
    def __deepcopy__(self, memo = None):
        clss = self.__class__
        new_struct = clss.__new__(clss)
        memo[id(self)] = new_struct
        
        new_struct.__dict__.update(self.__dict__)
        
        attrs_to_avoid_copy = []
        for k in self.__slots__:
            try:
                if k not in attrs_to_avoid_copy:
                    if not isinstance(k, list):
                        setattr(new_struct, k, copy.deepcopy(getattr(self, k), memo))
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
