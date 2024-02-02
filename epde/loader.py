#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 13:27:53 2023

@author: maslyaev
"""

import sys
import os
import dill as pickle
import time

from collections import Iterable

from epde.structure.factor import Factor
from epde.structure.main_structures import SoEq, Equation, Term 
from epde.interface.token_family import TFPool
from epde.optimizers.moeadd.moeadd import ParetoLevels
from epde.optimizers.single_criterion.optimizer import Population
from epde.cache.cache import Cache

TYPES = {'SoEq' : SoEq, 'TFPool' : TFPool, 'cache' : Cache,
         'ParetoLevels' : ParetoLevels, 'Population' : Population}    

# In TYPESPEC_ATTRS the first element of the value tuple is list of attributes not to be pickled
# while the second represents attributes for manual reconstruction.
TYPESPEC_ATTRS = {'SoEq' : (['tokens_for_eq', 'tokens_supp', 'latex_form'], ['vals']), 
                  'Factor' : (['latex_form', '_ann_repr', '_latex_constructor', '_evaluator'], []), 
                  'Equation' : (['pool', 'latex_form', '_history', '_features', '_target'], ['structure']), 
                  'Term' : (['pool', 'latex_form'], ['structure']), 'TFPool' : ([], []), 'cache' : ([], []), 
                  'ParetoLevels' : (['levels'], ['population']), 'Population' : ([], [])}
                  

LOADING_PRESETS = {'SoEq' : {'SoEq' : []}}

def get_size(obj, seen=None):
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    try:
        slots = obj.__slots__
    except AttributeError:
        slots = []
    for slot in slots:
        size += get_size(getattr(obj, slot), seen)
    return size

def parse_obj_type(obj):
    return str(type(obj)).split('.')[-1][:-2]
    
def get_typespec_attrs(obj):
    key = parse_obj_type(obj)   # Get key of the attribute, strapping extra symbols from type()
    return TYPESPEC_ATTRS[key]

def obj_to_pickle(obj, not_to_pickle: list = [], manual_pickle: list = []):
    '''

    Template method for adapting pickling of an object. Shall be copied to objects, that are 
    to be pickable with local rules.

    Parameters
    ----------
    
    obj : 
        Object, which is translated into a dictionary for pickling.
    
    not_to_pickle : list of strings
        ``obj`` object attributes to keep from saving to the resulting dict.

    manual_pickle : list of strings
        ``obj`` object attributes that will be pickled with their own ``obj_to_pickle()`` calls.

    Returns
    -------
    dict_to_pickle : dict
        Dictionary representation of the object attributes.

    '''
    ntp_add, mp_add = get_typespec_attrs(obj)
    not_to_pickle = ntp_add
    manual_pickle = mp_add
    
    dict_to_pickle = {'obj_type' : parse_obj_type(obj)}
    
    for key, elem in obj.__dict__.items():
        if key in not_to_pickle:
            continue
        elif key in manual_pickle:
            if isinstance(elem, dict):
                
                dict_to_pickle[key] = {'obj_type' : dict, 'keys' : [ekey for ekey in elem.keys()],
                                       'elements' : [obj_to_pickle(val, get_typespec_attrs(val)[0], get_typespec_attrs(val)[1])
                                                     for val in elem.values()]}
            elif isinstance(elem, Iterable):
                
                dict_to_pickle[key] = {'obj_type' : type(elem), 'elements' : [obj_to_pickle(list_elem,
                                                                                            get_typespec_attrs(list_elem)[0],
                                                                                            get_typespec_attrs(list_elem)[1])
                                                                              for list_elem in elem]}
            else:
                dict_to_pickle[key] = {'obj_type' : type(elem), 'elements' : obj_to_pickle(elem, get_typespec_attrs(elem)[0], 
                                                                                           get_typespec_attrs(elem)[1])}
        else:
            dict_to_pickle[key] = elem

    try:
        slots = obj.__slots__
    except AttributeError:
        slots = []

    for slot in slots:
        elem = getattr(obj, slot)
        if slot in not_to_pickle:
            continue
        elif slot in manual_pickle:
            if isinstance(elem, dict):
                dict_to_pickle[slot] = {'obj_type' : dict, 'keys' : [ekey for ekey in elem.keys()],
                                        'elements' : [obj_to_pickle(val, get_typespec_attrs(val)[0], get_typespec_attrs(val)[1])
                                                      for val in elem.values()]}
            elif isinstance(elem, Iterable):
                dict_to_pickle[slot] = {'obj_type' : type(elem), 'elements' : [obj_to_pickle(list_elem,
                                                                                             get_typespec_attrs(list_elem)[0],
                                                                                             get_typespec_attrs(list_elem)[1])
                                                                               for list_elem in elem]}
            else:
                dict_to_pickle[slot] = {'obj_type' : type(elem), 'elements' : obj_to_pickle(elem, get_typespec_attrs(elem)[0], 
                                                                                            get_typespec_attrs(elem)[1])}
        else:
            dict_to_pickle[slot] = elem
    
    return dict_to_pickle

def attrs_from_dict(obj, attributes, except_attrs: dict = {}):
    except_dict = except_attrs[parse_obj_type(obj)]
    if 'obj_type' not in except_dict.keys():
        except_dict['obj_type'] = None
    manual_reconstr = get_typespec_attrs(obj)[1]
    
    try:
        slots = obj.__slots__
        for slot in slots:
            if slot not in manual_reconstr:
                setattr(obj, slot, attributes[slot]) if slot in attributes.keys() else setattr(obj, slot, except_dict[slot])
    except AttributeError:
        slots = []
    
    obj.__dict__ = {key : item for key, item in attributes.items()
                    if key not in except_attrs.keys() and key not in slots and key not in manual_reconstr}
    for key, elem in except_attrs.items():
        if elem is not None and key not in slots and key not in manual_reconstr:
            obj.__dict__[key] = elem
    
    for man_attr in manual_reconstr:
        if man_attr not in attributes.keys():
            raise AttributeError(f'Object {man_attr} for reconstruction is missing from attributes dictionary.')

        obj.manual_reconst(man_attr, attributes[man_attr]['elements'], except_attrs)

class LoaderAssistant(object):
    def __init__(self):
        pass
    
    @staticmethod
    def system_preset(pool: TFPool): # Validate correctness of attribute definitions
        return {'SoEq' :     {'tokens_for_eq' : TFPool(pool.families_demand_equation),
                              'tokens_supp' : TFPool(pool.families_equationless), 
                              'latex_form' : None}, # TBD, make better loading procedure
                'Equation' : {'pool' : pool, 
                              'latex_form' : None,
                              '_history' : None,
                              '_features' : None,
                              '_target' : None},
                'Term'     : {'pool' : pool, 
                              'latex_form' : None},
                'Factor'   : {'latex_form' : None}}
    
    @staticmethod
    def pool_preset():
        return {'TFPool' : {}}
    
    @staticmethod
    def cache_preset():
        return {}
    
    @staticmethod
    def population_preset():
        return {}
    
    @staticmethod
    def pareto_levels_preset(pool: TFPool):
        return {
                'SoEq'     : {'tokens_for_eq' : TFPool(pool.families_demand_equation), 
                              'tokens_supp' : TFPool(pool.families_equationless), 
                              'latex_form' : None}, # TBD, make better loading procedure
                'Equation' : {'pool' : pool, 
                              'latex_form' : None},
                'Term'     : {'pool' : pool, 
                              'latex_form' : None},
                'Factor'   : {'latex_form' : None}}
        

class EPDELoader(object):
    '''
    Universal loader for EPDE objects, applicable to system of equations as 
    ``SoEq``, token families pool as ``TFPool``, populations as 
    '''
    
    def __init__(self, directory = None):
        if directory is not None:
            if not isinstance(directory, str):
                raise TypeError(f'Incorrect format of repo to save objects, expected str, got {type(directory)}.')

            if not os.path.isdir(directory):
                try:
                    os.mkdir(path=directory)
                except FileNotFoundError:
                    raise TypeError(f'Wrong path passed, can not create a directory with path {directory}')

            self._directory = directory
        else:
            self._directory = os.path.normpath((os.path.join(os.path.dirname(os.getcwd()), 
                                                             '..','epde_cache')))
    
    def save(self, obj, filename:str = None, not_to_pickle:list = [], manual_pickle:list = []):
        with open(filename, mode = 'wb') as file:
            pickle.dump(obj_to_pickle(obj, not_to_pickle, manual_pickle), file)
    
    def saves(self, obj, not_to_pickle:list = [], manual_pickle:list = []):
        pickling_form = obj_to_pickle(obj, not_to_pickle, manual_pickle)
        return pickle.dumps(pickling_form)        

    def use_pickles(self, obj_pickled, **kwargs):
        obj = TYPES[obj_pickled['obj_type']].__new__(TYPES[obj_pickled['obj_type']])
        attrs_from_dict(obj, obj_pickled, except_attrs = kwargs)
        return obj
    
    def load(self, filename:str, **kwargs):
        with open(filename, mode = 'rb') as file:
            obj_pickled = pickle.load(file)
        return self.use_pickles(obj_pickled, **kwargs)
    
    def loads(self, byteobj, **kwargs):
        obj_pickled = pickle.loads(byteobj)
        return self.use_pickles(obj_pickled, **kwargs)