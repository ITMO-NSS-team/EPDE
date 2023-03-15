#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 13:35:18 2021

@author: mike_ubuntu
"""

from typing import Union, Callable
import numpy as np
from sklearn.linear_model import Lasso

import epde.globals as global_var
from epde.operators.utils.template import CompoundOperator
from epde.structure.main_structures import Equation


class LASSOSparsity(CompoundOperator):
    """
    The operator, which applies LASSO regression to the equation object to detect the 
    valuable term coefficients.
    
    Notable attributes:
    -------------------
        
    params : dict
        Inhereted from the ``CompoundOperator`` class. 
        Parameters of the operator; main parameters: 
            
            sparsity - value of the sparsity constant in the LASSO operator;
            
    g_fun : np.ndarray or None:
        values of the function, used during the weak derivatives estimations. 
            
    Methods:
    -----------
    apply(equation)
        calculate the coefficients of the equation, that will be stored in the equation.weights np.ndarray.    
        
    """
    key = 'LASSOBasedSparsity'
    
    def apply(self, objective : Equation, arguments : dict):
        """
        Apply the operator, to fit the LASSO regression to the equation object to detect the 
        valueable terms. In the Equation class, a term is selected to represent the right part of
        the equation, and its values are used here as the target, and the values of the other 
        terms are utilizd as the features. The method does not return the vector of coefficients, 
        but rather assigns the result to the equation attribute ``equation.weights_internal``
        
        Parameters:
        ------------
        equation : Equation object
            the equation object, to that the coefficients are obtained.
            
        Returns:
        ------------
        None
        """
        # print(f'Metaparameter: {objective.metaparameters}, objective.metaparameters[("sparsity", objective.main_var_to_explain)]')
        self_args, subop_args = self.parse_suboperator_args(arguments = arguments)

        estimator = Lasso(alpha = objective.metaparameters[('sparsity', objective.main_var_to_explain)]['value'], 
                          copy_X=True, fit_intercept=True, max_iter=1000, 
                          positive=False, precompute=False, random_state=None,
                          selection='cyclic', tol=0.0001, warm_start=False)
        _, target, features = objective.evaluate(normalize = True, return_val = False)
        self.g_fun_vals = global_var.grid_cache.g_func.reshape(-1)

        estimator.fit(features, target, sample_weight = self.g_fun_vals)
        objective.weights_internal = estimator.coef_

    def use_default_tags(self):
        self._tags = {'sparsity', 'gene level', 'no suboperators', 'inplace'}
