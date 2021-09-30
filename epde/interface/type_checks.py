#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 16:36:01 2021

@author: mike_ubuntu
"""

import numpy as np
import typing

def check_nparray(obj):
    if type(obj) != np.ndarray:
        raise ValueError(f'mismatching types of object: passed {type(obj)} instead of numpy.ndarray')
        
def check_nparray_iterable(objs):
    if any([type(obj) != np.ndarray for obj in objs]):
        raise ValueError(f'mismatching types of object: passed {[type(obj) for obj in objs]} instead of numpy.ndarray')
    