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
from epde.cache import cache

class EPDELoader(object):
    '''
    Universal loader for EPDE objects, associated with 
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
        
    def save(self, obj, filename = None):
        #Make save methods for equations, pool and populations
        pickling_form = obj.to_pickle()
        with open(filename, mode = 'wb') as file:
            pickle.dump(pickling_form, file)

    def load(self, filename):
        types = {'equation' : SoEq, 'pool' : TFPool,
                 'multiobj_pop' : ParetoLevels, 'singleobj_pop' : Population}
        
        with open(filename, mode = 'rb') as file:
            obj_pickled = pickle.load(file)
        
        obj = types[obj_pickled['obj_type']].__new__(types[obj_pickled['obj_type']])
        obj.attrs_from_dict(obj_pickled)
        return obj