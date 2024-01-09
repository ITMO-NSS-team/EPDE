#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 13:38:20 2022

@author: maslyaev
"""

import numpy as np
from functools import reduce
from collections import Iterable

def check_uniqueness(obj, background):
    return not any([elem == obj for elem in background])


class ComplexStructure(object):
    def __init__(self, interelement_operator=np.add, *params):
        self._history = ''
        self.structure = None
        self.interelement_operator = interelement_operator

    def attrs_from_dict(self, attributes, except_attrs: dict = {}):
        except_attrs['obj_type'] = None
        try:
            slots = self.__slots__
            for slot in slots:
                setattr(self, slot, attributes[slot]) if slot in attributes.keys() else setattr(self, slot, except_attrs[slot])
        except AttributeError:
            pass
        self.__dict__ = {key : item for key, item in attributes.items()
                         if key not in except_attrs.keys() and key not in slots}
        for key, elem in except_attrs.items():
            if elem is not None and key not in self.__slots__:
                self.__dict__[key] = elem

    def to_pickle(self, not_to_pickle:list, manual_pickle: list = []):
        '''

        Template method for adapting pickling of an object. Shall be copied to objects, that are 
        to be pickable with local rules.

        Parameters
        ----------
        except_attrs : list of strings
            Attributes to keep from saving to the resulting dict.

        Returns
        -------
        dict_to_pickle : dict
            Dictionary representation of the object attributes.

        '''
        dict_to_pickle = {}
        
        for key, elem in self.__dict__.items():
            if key in not_to_pickle:
                continue
            elif key in manual_pickle:
                if isinstance(elem, dict):
                    dict_to_pickle[key] = {'type' : dict, 'keys' : [ekey for ekey in elem.keys()],
                                           'elements' : [val.to_pickle() for val in elem.values()]}
                elif isinstance(elem, Iterable):
                    dict_to_pickle[key] = {'type' : type(elem), 'elements' : [list_elem.to_pickle() for list_elem in elem]}
                else:
                    dict_to_pickle[key] = {'type' : type(elem), 'elements' : elem.to_pickle()}
            else:
                dict_to_pickle[key] = elem
        
        for slot in self.__slots__():
            elem = getattr(self, slot)
            if slot in not_to_pickle:
                continue
            elif key in manual_pickle:
                if isinstance(elem, dict):
                    dict_to_pickle[slot] = {'type' : dict, 'keys' : [key for key in elem.keys()],
                                           'elements' : [val.to_pickle() for val in elem.values()]}
                elif isinstance(elem, Iterable):
                    dict_to_pickle[slot] = {'type' : type(elem), 'elements' : [list_elem.to_pickle() for list_elem in elem]}
                else:
                    dict_to_pickle[slot] = {'type' : type(elem), 'elements' : elem.to_pickle()}
            else:
                dict_to_pickle[key] = elem
        
        return dict_to_pickle

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
