#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 13:35:18 2021

@author: mike_ubuntu
"""

import numpy as np
from sklearn.linear_model import Lasso

from epde.operators.template import Compound_Operator

#class Poplevel_sparsity(Compound_Operator):
#    def apply(self, population):
#        for 

class LASSO_sparsity(Compound_Operator):
    """
    The operator, which applies LASSO regression to the equation object to detect the 
    valuable term coefficients.
    
    Notable attributes:
    -------------------
        
    params : dict
        Inhereted from the ``Compound_Operator`` class. 
        Parameters of the operator; main parameters: 
            
            sparsity - value of the sparsity constant in the LASSO operator;
            
    Methods:
    -----------
    apply(equation)
        calculate the coefficients of the equation, that will be stored in the equation.weights np.ndarray.    
        
    """
    def apply(self, equation):
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

        estimator = Lasso(alpha = self.params['sparsity'], copy_X=True, fit_intercept=True, max_iter=1000,
                               normalize=False, positive=False, precompute=False, random_state=None,
                               selection='cyclic', tol=0.0001, warm_start=False)
        _, target, features = equation.evaluate(normalize = True, return_val = False)

        estimator.fit(features, target)
#        print('Set intermediate weights')        
        equation.weights_internal = estimator.coef_
        
    @property
    def operator_tags(self):
        return {'sparsity', 'equation level', 'no suboperators'}        