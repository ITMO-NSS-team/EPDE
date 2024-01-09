#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 22 13:27:53 2023

@author: maslyaev
"""

import os
import pickle

from epde.structure.main_structures import SoEq
from epde.interface.token_family import TFPool
from epde.optimizers.moeadd.moeadd import ParetoLevels
from epde.optimizers.single_criterion.optimizer import Population
from epde.cache.cache import Cache

class EPDELoader(object):
    '''
    Universal loader for EPDE objects, applicable to system of equations as 
    ``SoEq``, token families pool as ``TFPool``, populations as 
    '''
    _types = {'SoEq' : SoEq, 'TFPool' : TFPool, 'cache' : Cache,
             'multiobj_pop' : ParetoLevels, 'singleobj_pop' : Population}    
    
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
            pickle.dump(obj.to_pickle(not_to_pickle, manual_pickle), file)
    
    def saves(self, obj, not_to_pickle:list = [], manual_pickle:list = []):
        pickling_form = obj.to_pickle(not_to_pickle, manual_pickle)
        pickle.dumps(pickling_form)        

    def use_pickles(self, obj_pickled, **kwargs):
        obj = self._types[obj_pickled['obj_type']].__new__(self._types[obj_pickled['obj_type']])
        obj.attrs_from_dict(obj_pickled, except_attrs = kwargs)
        return obj

    def load(self, filename:str, **kwargs):
        with open(filename, mode = 'rb') as file:
            obj_pickled = pickle.load(file)
        return self.use_pickles(obj_pickled, **kwargs)

    def loads(self, byteobj, **kwargs):
        obj_pickled = pickle.loads(byteobj)
        return self.use_pickles(obj_pickled, **kwargs)