#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from functools import wraps

from epde.structure.main_structures import Term, Equation, SoEq
from epde.structure.encoding import Gene, Chromosome

OPERATOR_LEVELS = ('custom level', 'term level', 'gene level', 'chromosome level',
                   'population level')

OPERATOR_LEVELS_SUPPORTED_TYPES = {'custom level' : [], 'term level' : [Term], 'gene level' : [Gene, SoEq], 
                                   'chromosome level' : [Chromosome]}


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

    @suboperators.setter
    def suboperators(self, operators : dict):
        if not all([isinstance(key, str) and isinstance(value, (CompoundOperator, list, tuple, dict)) for key, value
                    in operators.items()]):
            raise TypeError('The suboperators of an evolutionary operator must be declared in format key : value, where key is str and value - CompoundOperator, list, tuple or dict')
        self._suboperators = operators

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