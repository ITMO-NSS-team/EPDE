#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import inspect

from functools import wraps
from epde.operators.utils.default_parameter_loader import EvolutionaryParams

from epde.structure.main_structures import Term, Equation, SoEq
from epde.structure.encoding import Gene, Chromosome
from epde.optimizers.moeadd.moeadd import ParetoLevels

OPERATOR_LEVELS = ('custom level', 'term level', 'gene level', 'chromosome level',
                   'population level')

OPERATOR_LEVELS_SUPPORTED_TYPES = {'custom level': None, 'term level': Term, 'gene level': Gene,
                                   'chromosome level': Chromosome, 'population level': ParetoLevels}


def add_base_param_to_operator(operator):
    params_container = EvolutionaryParams()
    for param_key, param_value in params_container[operator.key].items():
        operator.params[param_key] = param_value

def add_base_param_to_operator(operator, target_dict):
    params_container = EvolutionaryParams()
    for param_key, param_value in params_container.get_default_params_for_operator(operator.key).items():
        operator.params[param_key] = target_dict[param_key] if param_key in target_dict.keys(
        ) else param_value
# def add_param_to_operator(operator, target_dict, labeled_base_val):
#     for key, base_val in labeled_base_val.items():
#         if base_val is None and key not in target_dict.keys():
#             raise ValueError('Mandatory parameter for evolutionary operator')
#         operator.params[key] = target_dict[key] if key in target_dict.keys() else base_val

class CompoundOperator():
    '''
    Universal class for operator of an arbitrary purpose

    Attributes:
        suboperators (`dict`): dictionary with name of suboperators and its argumetns
        params (`dict`): dictionary with names and values of parameters for the operator
        param_keys (`list`): names of parameters of the operator
        level_index (`tuple`): abstraction level, to indicate which classes of objects the operator is applied to
        operator_tags (`set`): log about operator
    '''

    def __init__(self, param_keys: list = []):
        self.param_keys = param_keys
        self._params = {}
        self._suboperators = {}

        self.use_default_tags()

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, param_dict: dict):
        if set(self.param_keys) != set(param_dict.keys()):
            print('self.param_keys:', set(self.param_keys),
                  ' param_dict.keys():', set(param_dict.keys()))
            raise KeyError('Wrong keys of param dict')
        self._params = param_dict

    @property
    def suboperators(self):
        return self._suboperators

    def set_suboperators(self, operators: dict, probas: dict = {}):
        """
        Setting suboperators

        Args:
            operators (`dict`): dictionary with names and methods for evoluation of operators
            probas (`dict`): dictionary with names of operators and them probability of execution

        Returns:
            None
        """
        if not all([isinstance(key, str) and (isinstance(value, (list, tuple, dict)) or
                                              issubclass(type(value), CompoundOperator))
                    for key, value in operators.items()]):
            print([(key, isinstance(key, str),  
                    value, (isinstance(value, (list, tuple, dict)) or 
                            issubclass(type(value), CompoundOperator)))
                    for key, value in operators.items()])
            raise TypeError('The suboperators of an evolutionary operator must be declared in format key : value, where key is str and value - CompoundOperator, list, tuple or dict')
        self._suboperators = SuboperatorContainer(suboperators = operators, probas = probas) 

    def get_suboperator_args(self, personal=False):
        """
        Get arguments of the operator and its suboperators

        Args:
            personal (`boolean`): if True - gives unique arguments, default - True

        Returns:
            args (`list`): arguments of the operator and its suboperators.
        """
        args = self.arguments
        if not personal:
            for operator in self.suboperators:
                args = args.union(operator.get_suboperator_args())

        technical_args = ['arguments', 'objective', 'obj', 'self']
        for arg in technical_args:
            if arg in args:
                args.remove(arg)

        return args

    def _check_objective_type(method):
        @wraps
        def wrapper(self, *args, **kwargs):
            objective = args[0]
            try:
                level_descr = [
                    tag for tag in self.operator_tags if 'level' in tag][0]
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
                    raise TypeError(
                        f'Incorrect input type of the EA operator objective: {type(objective)} does not match {processing_type}')
                return result
        return wrapper

    def parse_suboperator_args(self, arguments: dict):
        """
        Getting args of suboperators in the from of a dictionary

        Args:
            arguments (`dict`): dictionary with names of operator's arguments and its values 

        Returns:
            `dict` with all parameters in one pile
            `dict` with parameters separated by each operator
        """
        def parse_args(keys, args):
            return {key: args[key] for key in keys}

        operators_args = {}
        for key in self.suboperators.keys():
            operators_args[key] = parse_args(
                self.suboperators[key].get_suboperator_args(), arguments)

        return parse_args(self.get_suboperator_args(True), arguments), operators_args

    @_check_objective_type
    def apply(self, objective, arguments: dict):
        self_args, subop_args = self.parse_suboperator_args(
            arguments=arguments)
        raise NotImplementedError('Trying to apply abstract superclass of the operator.')

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

    @property
    def arguments(self):
        return set()


class SuboperatorContainer():
    def __init__(self, suboperators: dict = {}, probas: dict = {}):
        '''
        Object, implemented to contain the suboperators of a CompoundOperator object. 
        Main purpose: support of uneven probabilities of similar suboperator application.
        '''
        self.suboperators = suboperators
        self.probas = {}

        for label, oper in suboperators.items():
            if isinstance(oper, CompoundOperator):
                operator_probability = 1
            elif isinstance(oper, (list, tuple, np.ndarray)) and label not in probas.keys():
                operator_probability = np.full(
                    fill_value=1./len(oper), shape=len(oper))
            elif isinstance(oper, (list, tuple, np.ndarray)) and label in probas.keys():
                if len(oper) != len(probas[label]):
                    raise ValueError(
                        f'Number of passed suboperators for {label} does not match defined probabilities.')
                operator_probability = probas[label]
            self.probas[label] = operator_probability

    def __getitem__(self, item):
        if isinstance(self.suboperators[item], CompoundOperator):
            return self.suboperators[item]
        elif isinstance(self.suboperators[item], (list, tuple, np.ndarray)):
            return np.random.choice(a=self.suboperators[item], p=self.probas[item])

    def keys(self):
        return self.suboperators.keys()

    def __iter__(self):
        return SuboperatorContainerIterator(self)


class SuboperatorContainerIterator(object):
    def __init__(self, container):
        self._suboperators = []
        for val in container.suboperators.values():
            self._suboperators.append(val) if isinstance(
                val, CompoundOperator) else self._suboperators.extend(val)
        self._idx = 0

    def __next__(self):
        if self._idx < len(self._suboperators):
            res = self._suboperators[self._idx]
            self._idx += 1
            return res
        else:
            raise StopIteration
