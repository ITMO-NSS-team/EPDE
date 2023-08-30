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
    def __init__(self, reset_input: bool = False, reset_output: bool = False,
                 reset_right_part: bool = True):
        self.reset_input = reset_input
        self.reset_output = reset_output
        self.reset_right_part = reset_right_part

    def __call__(self, method):
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
    '''

    Extend histroy log of the complex structure

    '''

    def __init__(self, action_log_entry: str = '',
                 state_writing_points='n'):
        assert (state_writing_points == 'n' or state_writing_points == 'ba' or
                state_writing_points == 'b' or state_writing_points == 'a')
        self.action_log_entry = action_log_entry
        self.state_writing_points = state_writing_points

    def __call__(self, method):
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
    Class for modifing test function to ignore (i.e. considered zeros) values at the bounderies

    Attributes:
        boundary_width (`int|list|tuple`): the number of unaccounted elements at the edges
    """
    def __init__(self, boundary_width=0):
        self.boundary_width = boundary_width

    def __call__(self, func):
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