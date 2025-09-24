#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 15:56:10 2021

@author: mike_ubuntu
"""

import numpy as np

from functools import wraps
from typing import Union

import epde.globals as global_var

changelog_entry_templates = {}


class ResetEquationStatus:
    """
    Resets the status of equation parts (input, output, right part).
    
        Attributes:
            reset_input (bool): Whether to reset the input.
            reset_output (bool): Whether to reset the output.
            reset_right_part (bool): Whether to reset the right part.
    """

    def __init__(self, reset_input: bool = False, reset_output: bool = False,
                 reset_right_part: bool = True):
        """
        Initializes the ResetEquationStatus object.
        
        This object determines which parts of an equation should be reset during the equation simplification process.
        Resetting parts of the equation allows the algorithm to explore different equation structures and avoid getting stuck in local optima.
        
        Args:
            reset_input (bool): Whether to reset the input part of the equation. Defaults to False.
            reset_output (bool): Whether to reset the output part of the equation. Defaults to False.
            reset_right_part (bool): Whether to reset the right-hand side of the equation. Defaults to True.
        
        Returns:
            None.
        """
        self.reset_input = reset_input
        self.reset_output = reset_output
        self.reset_right_part = reset_right_part

    def __call__(self, method):
        """
        Wraps a method to reset the state of input and output equation objects after execution.
        
                This decorator is applied to methods involved in equation discovery. After the method executes, it resets the state of the input arguments (typically equation components or data) and the resulting equation(s). This ensures a clean slate for subsequent evolutionary steps, preventing state accumulation from influencing the search process and maintaining the integrity of each generation. The reset operation is applied recursively to elements within lists, tuples, or sets.
        
                Args:
                    method: The method to be wrapped.
        
                Returns:
                    The wrapped method.
        """
        @wraps(method)
        def wrapper(obj, *args, **kwargs):
            result = method(obj, *args, **kwargs)

            if self.reset_input:
                for element in [obj,] + list(args):
                    if isinstance(element, (list, tuple, set)):
                        for subelement in element:
                            try:
                                subelement.reset_state(self.reset_right_part)
                            except AttributeError:
                                pass
                    else:
                        try:
                            element.reset_state(self.reset_right_part)
                        except AttributeError:
                            pass
            if self.reset_output:
                if isinstance(result, (list, tuple, set)):
                    for equation in result:
                        try:
                            equation.reset_state(self.reset_right_part)
                        except AttributeError:
                            pass
                else:
                    try:
                        result.reset_state(self.reset_right_part)
                    except AttributeError:
                        pass
            return result
        return wrapper


class HistoryExtender():
    """
    The `HistoryExtender` class manages and extends the history of a symbolic regression process. It stores past populations, expressions, and their associated fitness scores, enabling the algorithm to revisit promising areas of the search space and avoid premature convergence. The class provides methods for adding new populations to the history, retrieving the best individuals from the history, and managing the size of the historical archive. This allows for a more comprehensive exploration of the solution space and can lead to the discovery of more accurate and robust models.
    
    
        Extend histroy log of the complex structure
    
        '''
    """


    def __init__(self, action_log_entry: str = '',
                 state_writing_points='n'):
        """
        Initializes a HistoryExtender instance, configuring how historical states are managed during the equation discovery process.
        
                This configuration determines when the state of the evolutionary process
                is recorded, allowing for detailed analysis and potential rollback
                during the search for the optimal equation.
        
                Args:
                    action_log_entry (str): A descriptive string to log actions performed. Defaults to ''.
                    state_writing_points (str): Specifies points in the evolutionary algorithm's cycle
                        where state should be saved.  Valid values are:
                        - 'n': Never write state.
                        - 'ba': Write state before and after each generation.
                        - 'b': Write state before each generation.
                        - 'a': Write state after each generation.
                        Defaults to 'n'.
        
                Returns:
                    None.
        
                Class Fields:
                    action_log_entry (str): Stores the action log entry for logging purposes.
                    state_writing_points (str): Stores the state writing points configuration,
                        controlling the frequency of state saving.
        """
        assert (state_writing_points == 'n' or state_writing_points == 'ba' or
                state_writing_points == 'b' or state_writing_points == 'a')
        self.action_log_entry = action_log_entry
        self.state_writing_points = state_writing_points

    def __call__(self, method):
        """
        Wraps a method to track changes in objects relevant to differential equation discovery.
        
                This wrapper adds history entries to objects involved in the execution of the wrapped method,
                specifically those possessing `_history` and `add_history` attributes. These entries,
                derived from the class's `action_log_entry` and object states, capture the evolution
                of objects during equation discovery, aiding in understanding how different operations
                affect the search for the best-fitting differential equation. This is important to track the changes of the objects and debug the process of equation discovery.
        
                Args:
                    method: The method to wrap.
        
                Returns:
                    The wrapped method.
        """
        @wraps(method)
        def wrapper(obj, *args, **kwargs):
            def historized(h_obj):
                res = hasattr(h_obj, '_history') and hasattr(h_obj, 'add_history')
                return res

            for element in [obj,] + list(args):
                if historized(element):
                    element.add_history(self.action_log_entry)

            if 'b' in self.state_writing_points:
                ender = ' ' if 'a' in self.state_writing_points else ' || \n'
                for element in [obj,] + list(args):
                    if historized(element):
                        element.add_history(
                            ' || before operation: ' + element.state + ender)

            result = method(obj, *args, **kwargs)
            if 'a' in self.state_writing_points:
                beginner = ' | ' if 'b' in self.state_writing_points else ' || '
                for element in [obj,] + list(args):
                    if historized(element):
                        element.add_history(
                            beginner + 'after operation: ' + element.state + ' || \n')
            return result
        return wrapper


class BoundaryExclusion():
    """
    Class designed to modify a test function, effectively treating values at specified boundaries as zero. This allows the function to disregard boundary values during calculations or evaluations.
    
    
        Attributes:
            boundary_width (`int|list|tuple`): the number of unaccounted elements at the edges
    """

    def __init__(self, boundary_width=0):
        """
        Initializes a BoundaryExclusion object.
        
        This object defines an exclusion zone around the boundaries of the domain.
        The boundary width determines the size of this zone, which is used to avoid
        evaluating equation terms too close to the edges of the data domain. This
        helps to improve the accuracy and stability of the equation discovery process
        by mitigating boundary effects.
        
        Args:
            boundary_width (int): The width of the boundary exclusion zone. Defaults to 0.
        
        Returns:
            None
        
        Attributes:
            boundary_width (int): The width of the boundary exclusion zone.
        """
        self.boundary_width = boundary_width

    def __call__(self, func):
        """
        Applies a mask to the output of a function, effectively excluding boundary regions.
        
                This method acts as a decorator. It wraps a given function, modifying its
                output by applying a mask. This mask sets values within a specified
                `boundary_width` of the grid edges to zero. This is useful to exclude boundary effects from consideration when discovering underlying equations.
        
                Args:
                    func (callable): The function to be decorated. This function should
                        accept a tuple of grids as input and return a numerical array
                        representing the function's output on those grids.
        
                Returns:
                    callable: A wrapped function that, when called, applies the boundary
                        exclusion mask to the original function's output.
        """
        @wraps(func)
        def wrapper(grids, boundary_width: Union[int, list] = 0):
            assert len(grids) == grids[0].ndim
            if isinstance(self.boundary_width, int):
               
                self.boundary_width = len(grids)*[self.boundary_width,]
            indexes_shape = grids[0].shape
            indexes = np.indices(indexes_shape)

            mask_partial = np.array([np.where((indexes[idx, ...] >= self.boundary_width[idx]) &
                                              (indexes[idx, ...] < indexes_shape[idx] -
                                               self.boundary_width[idx]),
                                              1, 0)
                                     for idx in np.arange(indexes.shape[0])])

            mask = np.multiply.reduce(mask_partial, axis=0)
            g_function_res = func(grids)
            assert np.shape(g_function_res) == np.shape(mask)
            return func(grids) * mask

        return wrapper