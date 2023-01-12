#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 16:05:11 2022

@author: maslyaev
"""

import numpy as np
from typing import Union

from functools import reduce
from abc import ABC, abstractmethod, abstractproperty

class GeneralizedPrepBuilder(ABC):
    def __init__(self):
        pass
    
    @abstractproperty
    def prep_pipeline(self):
        pass
    
    @abstractmethod
    def set_smoother(self):
        pass
    
    @abstractmethod    
    def set_deriv_calculator(self):
        pass
    
    @abstractmethod
    def check_preprocessing_correctness(self):
        pass


class ConcretePrepBuilder(GeneralizedPrepBuilder):
    def __init__(self):
        self.reset()
        
    def reset(self):
        self._prep_pipeline = PreprocessingPipe()
        self.deriv_calc_set = False
        self.smoother_set = False
        self.output_tests = []
        
    def set_tests(self, tests):
        self.output_tests = tests
        
    def set_smoother(self, smoother, *args, **kwargs):
        self._prep_pipeline.smoother = smoother()
        self._prep_pipeline.smoother_args = args
        self._prep_pipeline.smoother_kwargs = kwargs
        if self.deriv_calc_set and self.smoother_set:
            self.check_preprocessing_correctness()
        
    def set_deriv_calculator(self, deriv_calculator, *args, **kwargs):
        self._prep_pipeline.deriv_calculator = deriv_calculator()
        self._prep_pipeline.deriv_calculator_args = args
        self._prep_pipeline.deriv_calculator_kwargs = kwargs
        if self.deriv_calc_set and self.smoother_set:
            self.check_preprocessing_correctness()

    def check_preprocessing_correctness(self):
        print("Checking correctness of the preprocessing tool:")
        try:
            test_call = self._prep_pipeline.run()
            if len(self.output_tests) > 0:
                test_call = reduce(lambda x, y: y(x), self.output_tests, test_call)
            _ = test_call(np.ones((100, 100)))
        except:
            print("Incorrect selection of preprocessing tools.")
            
    @property
    def prep_pipeline(self):
        pipeline = self._prep_pipeline
        self.reset()
        return pipeline


class PreprocessingPipe(object):
    def __init__(self):
        self.smoother = None
        self.deriv_calculator = None
    
        self.smoother_args = None
        self.smoother_kwargs = None
        
        self.deriv_calculator_kwargs = None
        self.deriv_calculator_args = None
    
    def use_grid(self, grid):
        if 'grid' in self.smoother_kwargs.keys():
            self.smoother_kwargs['grid'] = grid
        if 'grid' in self.deriv_calculator_kwargs.keys():
            self.deriv_calculator_kwargs['grid'] = grid
    
    def run(self, data, grid = None, max_order : Union[list, int] = 1):
        self.deriv_calculator_kwargs['max_order'] = max_order
        if grid is not None:
            self.use_grid(grid)
            
        # TODO: add an arbitrary preprocssing operators
        if self.smoother is not None:
            data = self.smoother(data, *self.smoother_args, **self.smoother_kwargs)
        return data, self.deriv_calculator(data, *self.deriv_calculator_args, **self.deriv_calculator_kwargs)