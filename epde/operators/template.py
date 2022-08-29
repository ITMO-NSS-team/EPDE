#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import inspect

from functools import wraps

from epde.structure.main_structures import Term, Equation, SoEq
from epde.structure.encoding import Gene, Chromosome
from epde.moeadd.moeadd import ParetoLevels

OPERATOR_LEVELS = ('custom level', 'term level', 'gene level', 'chromosome level',
                   'population level')

OPERATOR_LEVELS_SUPPORTED_TYPES = {'custom level' : [], 'term level' : [Term], 'gene level' : [Gene], 
                                   'chromosome level' : [Chromosome], 'population level' : [ParetoLevels]}


def add_param_to_operator(operator, target_dict, labeled_base_val):
    for key, base_val in labeled_base_val.items():
        if base_val is None and key not in target_dict.keys():
            raise ValueError('Mandatory parameter for evolutionary operator')
        operator.param[key] = target_dict[key] if key in target_dict.keys() else base_val


class CompoundOperator():
    '''
    Universal class for operator of an arbitrary purpose
    '''
    def __init__(self, param_keys : list = []):
        self.param_keys = param_keys
        self.use_default_tags()

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, param_dict : dict):
        if set(self.param_keys) != set(param_dict.keys()):
            print('self.param_keys:', set(self.param_keys), ' param_dict.keys():', set(param_dict.keys()))
            raise KeyError('Wrong keys of param dict')
        self._params = param_dict

    @property
    def suboperators(self):
        return self._suboperators

    def set_suboperators(self, operators : dict, probas : dict = {}):
        if not all([isinstance(key, str) and isinstance(value, (CompoundOperator, list, tuple, dict)) for key, value
                    in operators.items()]):
            raise TypeError('The suboperators of an evolutionary operator must be declared in format key : value, where key is str and value - CompoundOperator, list, tuple or dict')
        self._suboperators = SuboperatorContainer(suboperators = operators, probas = probas) 

    def get_suboperator_args(self):
        '''
        
        
        Returns
        -------
        args : list
            Arguments of the operator and its suboperators.

        '''
        args = [operator.get_suboperator_args() for operator in self.suboperators]
        args.extend(inspect.getfullargspec(self.apply).args)
        
        if 'objective' in args:
            args.remove('objective')
        return args

    def _check_objective_type(method):
        @wraps
        def wrapper(self, *args, **kwargs):
            objective = args[0]
            try:
                level_descr = [tag for tag in self.operator_tags if 'level' in tag][0]
            except IndexError:
                level_descr = 'custom level'
            if level_descr == 'custom level':
                result = method(self, *args, **kwargs)
            else:
                processing_type = OPERATOR_LEVELS_SUPPORTED_TYPES[level_descr]
                # TODO: переписать выбор типа и тэги под кастомные объекты
                if isinstance(objective, self.operator_tags[processing_type]):
                    result = method(self, *args, **kwargs)
                else:
                    raise TypeError(f'Incorrect input type of the EA operator objective: {type(objective)} does not match {processing_type}')
                return result
        return wrapper
    
    _check_objective_type = staticmethod(_check_objective_type)
    
    @_check_objective_type
    def apply(self, objective):
        pass

    @property
    def level_index(self):
        try:
            return [(idx, level) for idx, level in enumerate(OPERATOR_LEVELS) 
                    if level in self.operator_tags][0]
        except IndexError:
            return (0, 'custom level')

    def use_default_tags(self):
        self._tags = set()

    @property
    def operator_tags(self):
        return self._tags


class SuboperatorContainer():
    def __init__(self, suboperators : dict = {}, probas : dict = {}):
        '''
        Object, implemented to contain the suboperators of a CompoundOperator object. 
        Main purpose: support of uneven probabilities of similar suboperator application.
        '''
        self.suboperators = suboperators
        self.probas = {}

        for label, oper in suboperators:
            if isinstance(oper, CompoundOperator):
                operator_probability = 1
            elif isinstance(oper, (list, tuple, np.ndarray)) and label not in probas.keys():
                operator_probability = np.full(fill_value=1./len(oper), shape=len(oper))
            elif isinstance(oper, (list, tuple, np.ndarray)) and label in probas.keys():
                if len(oper) != len(probas[label]):
                    raise ValueError(f'Number of passed suboperators for {label} does not match defined probabilities.')
                operator_probability = probas[label]
            self.probas[label] = operator_probability
    
    def __call__(self, label):
        if isinstance(self.suboperators[label], CompoundOperator):
            return self.suboperators[label]
        elif isinstance(self.suboperators[label], (list, tuple, np.ndarray)):
            return np.random.choice(a = self.suboperators[label], p = self.probas[label])