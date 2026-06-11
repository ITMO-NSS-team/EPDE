#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 13:38:20 2022

@author: maslyaev
"""

import copy
import numpy as np
from functools import reduce
try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable


def _deepcopy_slots(src, memo, attrs_to_avoid_copy=(), attrs_to_share_by_ref=()):
    """Slot-aware deep copy used by Term/Equation/SoEq/Factor.

    Replicates the loop that previously lived in each class's
    ``__deepcopy__``: iterate ``__slots__``, skip attrs in
    ``attrs_to_avoid_copy`` (sets them to None instead), tolerate slots
    that are not yet set (AttributeError -> skip), deepcopy lists
    element-by-element so subclassed list types survive.

    ``attrs_to_share_by_ref`` aliases the named slots from ``src``
    directly instead of deep-copying them -- used for immutable /
    single-instance objects (e.g. ``pool``, ``_evaluator``) that the
    same population shares.

    Hosted here (not in ``main_structures``) so ``Factor`` can call it
    without creating a circular import (``main_structures`` already
    imports ``Factor``).
    """
    clss = src.__class__
    new_struct = clss.__new__(clss)
    memo[id(src)] = new_struct
    for k in src.__slots__:
        try:
            if k in attrs_to_avoid_copy:
                setattr(new_struct, k, None)
            elif k in attrs_to_share_by_ref:
                setattr(new_struct, k, getattr(src, k))
            else:
                value = getattr(src, k)
                if isinstance(value, list):
                    setattr(new_struct, k, [copy.deepcopy(elem, memo) for elem in value])
                else:
                    setattr(new_struct, k, copy.deepcopy(value, memo))
        except AttributeError:
            pass
    return new_struct


def check_uniqueness(obj, background):
    return not any([elem == obj for elem in background])


class ComplexStructure(object):
    def __init__(self, interelement_operator=np.add, *params):
        self._history = ''
        self.structure = None
        self.interelement_operator = interelement_operator
    
    def manual_reconst(self, attribute:str, value, except_attrs:dict):
        from epde.loader import obj_to_pickle, attrs_from_dict        
        supported_attrs = []
        if attribute not in supported_attrs:
            raise ValueError(f'Attribute {attribute} is not supported by manual_reconst method.')
    
    def __eq__(self, other):
        if type(other) != type(self):
            raise ValueError('Type of self and other are different')
        return (all([any([other_elem == self_elem for other_elem in other.structure]) for self_elem in self.structure]) and
                all([any([other_elem == self_elem for self_elem in self.structure]) for other_elem in other.structure]) and
                len(other.structure) == len(self.structure))

    def __iter__(self):
        return CSIterator(self)

    def hash_descr(self):
        return tuple([term.hash_descr for term in self.structure])

    def set_evaluator(self, evaluator):
        raise NotImplementedError(
            'Functionality of this method has been moved to the evolutionary operator declaration')

    def evaluate(self, structural=False):
        assert len(self.structure) > 0, 'Attempt to evaluate an empty complex structure'
        if len(self.structure) == 1:
            return self.structure[0].evaluate(structural)
        try:
            evaluated = [elem.evaluate(structural) for elem in self.structure]
            return reduce(self.interelement_operator, evaluated)
        except ValueError:
            print([element.name for element in self.structure])
            raise ValueError('operands could not be broadcast together with shapes')

    def reset_saved_state(self):
        self.saved = {True: False, False: False}
        self.saved_as = {True: None, False: None}
        for elem in self.structure:
            elem.reset_saved_state()

    @property
    def name(self):
        pass


class CSIterator(object):
    def __init__(self, complex_structure: ComplexStructure):
        self._idx = 0
        self._complex_structure = complex_structure

    def __next__(self):
        if self._idx < len(self._complex_structure.structure):
            res = self._complex_structure.structure[self._idx]
            self._idx += 1
            return res
        else:
            raise StopIteration
