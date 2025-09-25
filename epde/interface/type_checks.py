#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 16:36:01 2021

@author: mike_ubuntu
"""

import numpy as np
import typing


def check_nparray(obj):
    """
    Checks if an object is a NumPy array.
    
    This check is crucial to ensure that the data is in the correct format 
    before proceeding with equation discovery, as the evolutionary algorithm 
    relies on efficient numerical operations provided by NumPy.
    
    Args:
        obj: The object to check.
    
    Returns:
        None.
    
    Raises:
        ValueError: If the object is not a NumPy array.
    """
    if type(obj) != np.ndarray:
        raise ValueError(
            f'mismatching types of object: passed {type(obj)} instead of numpy.ndarray')


def check_nparray_iterable(objs):
    """
    Verifies that all elements within the provided iterable are NumPy arrays.
    
    This ensures data consistency before proceeding with equation discovery.
    
    Args:
        objs: An iterable containing objects to be checked.
    
    Returns:
        None. The function raises an exception if the check fails.
    
    Raises:
        ValueError: If any object in the iterable is not a NumPy array,
            indicating a type mismatch that could disrupt the equation discovery process.
    """
    if any([type(obj) != np.ndarray for obj in objs]):
        raise ValueError(
            f'mismatching types of object: passed {[type(obj) for obj in objs]} instead of numpy.ndarray')
