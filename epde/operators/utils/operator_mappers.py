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

from epde.operators.utils.template import OPERATOR_LEVELS, CompoundOperator

class OperatorCondition(CompoundOperator):
    """
    Wraps a compound operator and applies it only when a condition is met.
    
        Class Methods:
        - __init__:
    """

    def __init__(self, operator: CompoundOperator, condition: Callable = None):
        """
        Initializes a ConditionalOperator.
        
                Wraps a compound operator and applies it only when a condition is met. This allows to construct more complex equation structures by enabling or disabling parts of the equation based on specific criteria, thus improving the equation discovery process.
        
                Args:
                    operator (CompoundOperator): The compound operator to be conditionally applied.
                    condition (Callable): A callable that determines whether the operator should be applied.
        
                Returns:
                    None
        """
        super().__init__()

        self._conditioned_operator = operator
        self.condition = condition
        self._tags = operator.operator_tags

    def apply(self, objective, arguments: dict):
        """
        Applies a specific transformation to an objective based on a predefined condition and the nature of the transformation itself.
        
                This method orchestrates the application of a transformation to a given objective, but only if a specified condition is met.
                The type of transformation (e.g., in-place modification or creation of a new object) is determined by the operator tags.
                This ensures that the appropriate transformation is applied to the objective, facilitating the evolutionary search for the best equation structure.
                If the '_tags' contain 'population level', the objective is returned without transformation.
        
                Args:
                    objective: The objective to be transformed. This could represent a candidate equation or a population of equations.
                    arguments: A dictionary containing arguments required by the transformation operator.
        
                Returns:
                    The transformed objective if the operator creates a new object or if the '_tags' contain 'population level'. Returns None if the operator modifies the objective in-place.
        """
        if self.condition(objective):
            if 'inplace' in self.operator_tags:
                self._conditioned_operator.apply(objective, arguments)
            elif 'standard' in self.operator_tags:
                objective = self._conditioned_operator.apply(objective, arguments)
            else:
                raise TypeError('Incorrect type of mapping operator: not inplace nor returns similar object, as input.')
                
        if 'population level' in self._tags: # Edited 21.05.2024
            return objective


class OperatorMapper(CompoundOperator):
    """
    Maps operators based on specified conditions.
    
        This class facilitates the mapping of one operator onto elements of another,
        allowing for conditional application based on objective and element criteria.
    
        Class Methods:
        - __init__
        - apply
    
        Attributes:
            _suboperators (dict): A dictionary storing sub-operators, with 'to_map' key holding the operator_to_map.
            objective_condition (Callable): A callable that determines whether the operator should be applied to a specific objective.
            element_condition (Callable): A callable that determines whether the operator should be applied to a specific element.
            _tags (set): A set of tags associated with the operator, derived from the operator_to_map's tags, with the source_tag removed and the objective_tag added.
    """

    def __init__(self, operator_to_map: CompoundOperator, objective_tag: str, source_tag: str, 
                 objective_condition: Callable = None, element_condition: Callable = None):
        """
        Initializes the MapToElementsOperator.
        
                This operator prepares a given operator for application within a broader equation discovery process. It configures how the operator will be applied to specific elements of a larger expression, based on user-defined conditions. This setup is crucial for exploring the solution space of possible differential equations.
        
                Args:
                    operator_to_map: The operator to be mapped to the elements.
                    objective_tag: The tag to be assigned to the resulting operator.
                    source_tag: The tag identifying the elements to which the operator is mapped.
                    objective_condition: An optional callable that determines whether the operator
                        should be applied to a specific objective. Defaults to None.
                    element_condition: An optional callable that determines whether the operator
                        should be applied to a specific element. Defaults to None.
        
                Raises:
                    ValueError: If the source_tag is not present in the operator_to_map's tags.
        
                Returns:
                    None
        
                Class Fields:
                    _suboperators (dict): A dictionary storing sub-operators, with 'to_map' key holding the operator_to_map.
                    objective_condition (Callable): A callable that determines whether the operator should be applied to a specific objective.
                    element_condition (Callable): A callable that determines whether the operator should be applied to a specific element.
                    _tags (set): A set of tags associated with the operator, derived from the operator_to_map's tags, with the source_tag removed and the objective_tag added.
        
                Why:
                    This method sets up the operator for selective application to different parts of a potential equation, enabling the evolutionary search to explore diverse equation structures effectively.
        """
        super().__init__()

        self.set_suboperators({'to_map' : operator_to_map})
        self.objective_condition = objective_condition; self.element_condition = element_condition
        if not source_tag in operator_to_map.operator_tags:
            raise ValueError(f'Only {source_tag}-level operators can be mapped to the elements of a {objective_tag}. \
                               Recieved operator with tags: {operator_to_map.operator_tags}')
        self._tags = copy.copy(operator_to_map.operator_tags)
        self._tags.remove(source_tag)
        self._tags.add(objective_tag)

    def apply(self, objective, arguments: dict):
        """
        Applies a sub-operator to elements of the objective based on specified conditions and operator tags.
        
                This method facilitates the application of a sub-operator to each element of the objective
                that satisfies a given element condition. The behavior adapts based on whether the operator
                is marked as 'inplace' or 'standard'. For 'inplace' operators, the sub-operator directly
                modifies the elements. For 'standard' operators, the sub-operator returns a new value that
                replaces the original element. This mapping is essential for evolving and refining the
                objective (e.g., a population of equations) by applying transformations defined by the
                sub-operators.
        
                Args:
                    objective: The objective to be mapped over (e.g., a population of equations).
                    arguments: A dictionary of arguments to be passed to the sub-operator.
        
                Returns:
                    The modified objective if the operator tag contains 'population level'. Otherwise, returns None.
                    If the operator tag contains 'inplace', the original objective is modified directly.
                    If the operator tag contains 'standard', a new objective with modified elements is returned.
        
                Raises:
                    TypeError: If the operator tags do not include 'inplace' or 'standard'.
        """
        if self.objective_condition is None or self.objective_condition(objective):
            if 'inplace' in self.operator_tags:
                for elem in objective:
                    if self.element_condition is None or self.element_condition(elem):
                        self.suboperators['to_map'].apply(elem, arguments)
            elif 'standard' in self.operator_tags:
                for idx, elem in enumerate(objective):
                    if self.element_condition is None or self.element_condition(elem):
                        objective[idx] = self.suboperators['to_map'].apply(elem, arguments)
            else:
                raise TypeError('Incorrect type of mapping operator: not inplace nor returns similar object, as input.')

        if 'population level' in self._tags: # Edited 21.05.2024
            return objective


def map_operator_between_levels(operator, original_level: Union[str, int], target_level: Union[str, int],
                                objective_condition: Callable = None, element_condition: Callable = None) -> CompoundOperator:
    """
    Maps an operator between two levels by sequentially refining it through a series of transformations.
    
        This method takes an initial operator and progressively adapts it to a target level of complexity.
        It achieves this by applying a chain of `OperatorMapper` instances, each responsible for a single level transition.
        This iterative refinement allows the operator to evolve, incorporating more detailed features and dependencies
        as it moves closer to the target level. This is useful for gradually building up complex operators from simpler ones.
    
        Args:
            operator: The initial operator to be mapped between levels.
            original_level: The starting level for the operator mapping (int or str).
            target_level: The desired level for the operator after mapping (int or str).
            objective_condition: An optional callable specifying a condition for the objective.
            element_condition: An optional callable specifying a condition for individual elements.
    
        Returns:
            CompoundOperator: The resulting operator after mapping through the specified levels.
    """
    if isinstance(original_level, str): original_level = OPERATOR_LEVELS.index(original_level)
    if isinstance(target_level, str): target_level = OPERATOR_LEVELS.index(target_level)
    
    resulting_operator = reduce(lambda x, y: OperatorMapper(operator_to_map     = x, 
                                                            objective_tag       = OPERATOR_LEVELS[y], 
                                                            source_tag          = OPERATOR_LEVELS[y-1],
                                                            objective_condition = objective_condition,
                                                            element_condition   = element_condition),
                                np.arange(original_level + 1, target_level + 1),
                                operator)
    return resulting_operator