#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 11 13:57:40 2022

@author: maslyaev
"""
import numpy as np

import copy
from functools import reduce
from typing import Union, Callable

from epde.operators.template import OPERATOR_LEVELS, CompoundOperator


class OperatorMapper(CompoundOperator):
    def __init__(self, operator_to_map : CompoundOperator, objective_tag : str, source_tag : str, 
                 objective_condition : Callable = None):
        super().__init__()

        self.set_suboperators({'to_map' : operator_to_map}); self.objective_condition = objective_condition
        if not source_tag in operator_to_map.operator_tags:
            raise ValueError(f'Only {source_tag}-level operators can be mapped to the elements of a {objective_tag}.')
        self._tags = copy.copy(operator_to_map.operator_tags)
        self._tags.remove(source_tag)
        self._tags.add(objective_tag)
        # print(f'Initializing operator mapper from {source_tag} to {objective_tag}')

    def apply(self, objective : CompoundOperator, arguments : dict):
        if self.objective_condition is None or self.objective_condition(objective):
            if 'inplace' in self.operator_tags:
                for elem in objective:
                    self.suboperators['to_map'].apply(elem, arguments)
            elif 'standard' in self.operator_tags:
                for idx, elem in enumerate(objective):
                    objective[idx] = self.suboperators['to_map'].apply(elem, arguments)
            else:
                raise TypeError('Incorrect type of mapping operator: not inplace nor returns similar object, as input.')


def map_operator_between_levels(operator : CompoundOperator, original_level : Union[str, int], 
                           target_level : Union[str, int], objective_condition : Callable = None):
    if isinstance(original_level, str): original_level = OPERATOR_LEVELS.index(original_level)
    if isinstance(target_level, str): target_level = OPERATOR_LEVELS.index(target_level)
    
    # print(f'mapping between {original_level} and {target_level}, that is {np.arange(original_level, target_level + 1)}')
    resulting_operator = reduce(lambda x, y: OperatorMapper(operator_to_map     = x, 
                                                            objective_tag       = OPERATOR_LEVELS[y], 
                                                            source_tag          = OPERATOR_LEVELS[y-1],
                                                            objective_condition = objective_condition),
                                np.arange(original_level + 1, target_level + 1),
                                operator)
    return resulting_operator