#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 13:38:20 2022

@author: maslyaev
"""

import numpy as np
from functools import reduce


def check_uniqueness(obj, background):
    return not any([elem == obj for elem in background])


class ComplexStructure(object):
    def __init__(self, interelement_operator=np.add, *params):
        self._history = ''
        self.structure = None
        self.interelement_operator = interelement_operator

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
        else:
            return reduce(lambda x, y: self.interelement_operator(x, y.evaluate(structural)),
                          self.structure[1:], self.structure[0].evaluate(structural))

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
