#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 20:13:46 2022

@author: maslyaev
"""

from epde.preprocessing.smoothers import gaussian, ann_smoother
from epde.preprocessing.deriv_calculators import adaptive_finite_difference, polynomial_diff

from epde.preprocessing.preprocessor import ConcretePrepBuilder

class PreprocessorSetup:
    def __init__(self):
        self._builder = None

    @property
    def builder(self):
        return self._builder

    @builder.setter
    def builder(self, builder : ConcretePrepBuilder):
        self._builder = builder

    def build_ANN_preprocessing(self, test_output = False, epochs_max = 1e3,
                                loss_mean = 1000, batch_frac = 0.5):
        smoother_args = ()
        smoother_kwargs = {'grid' : None, 'test_output' : test_output, 'epochs_max' : epochs_max,
                           'loss_mean' : loss_mean, 'batch_frac' : batch_frac}
        
        deriv_calculator_args = ()
        deriv_calculator_kwargs = {'grid' : None}
        
        self.builder.set_smoother(ann_smoother, args = smoother_args, 
                                  kwargs = smoother_kwargs)
        self.builder.set_deriv_calculator(adaptive_finite_difference, args = deriv_calculator_args, 
                                          kwargs = deriv_calculator_kwargs)
        
    def build_poly_diff_preprocessing(self, sigma = 9, mp_poolsize = 4, polynomial_window = 9,
                                      poly_order = None):
        smoother_args = ()
        smoother_kwargs = {'sigma' : sigma}
        
        deriv_calculator_args = ()
        deriv_calculator_kwargs = {'grid' : None, 'mp_poolsize' : mp_poolsize, 'polynomial_window' : polynomial_window,
                                   'poly_order' : poly_order}
        
        self.builder.set_smoother(gaussian, args = smoother_args,
                                  kwargs = smoother_kwargs)
        self.builder.set_deriv_calculator(polynomial_diff, args = deriv_calculator_args,
                                          kwargs = deriv_calculator_kwargs)
    