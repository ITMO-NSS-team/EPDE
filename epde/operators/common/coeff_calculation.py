#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 13:58:18 2021

@author: mike_ubuntu
"""

import numpy as np
from sklearn.linear_model import LinearRegression

import epde.globals as global_var
from epde.operators.utils.template import CompoundOperator
from epde.structure.main_structures import Equation

class LinRegBasedCoeffsEquation(CompoundOperator):
    '''
    
    The operatror, dedicated to the calculation of the weights of the equation (for the free coefficient and 
    each of its terms except the target one).

    Attributes:
        _tags (`set`): 
        g_fun_vals (`numpy.ndarray`): 
    
    Methods:
        apply(equation)
            Calculate the coefficients of the equation, using the linear regression. The result is stored in the 
            equation.weights_final attribute
    '''
    key = 'LinRegCoeffCalc'
    
    def apply(self, objective : Equation, arguments : dict = None):
        """
        Calculate the coefficients of the equation, using the linear regression.The result is stored in the 
        objective.weights_final attribute

        Args:
            objective (`Equation`): the equation object, to that the fitness function is obtained.
            arguments (`dict`): 
            
        Returns:
            None
        """        
        # self_args, subop_args = self.parse_suboperator_args(arguments = arguments)
        
        assert objective.weights_internal_evald, 'Trying to calculate final weights before evaluating intermeidate ones (no sparsity).'
        target = objective.structure[objective.target_idx]
    
        target_vals = target.evaluate(False)
        features_vals = []
        nonzero_features_indexes = []
        for i in range(len(objective.structure)):
            if i == objective.target_idx:
                continue
            idx = i if i < objective.target_idx else i-1
            if objective.weights_internal[idx] != 0:
                features_vals.append(objective.structure[i].evaluate(False))
                nonzero_features_indexes.append(idx)

        if len(features_vals) == 0:
            objective.weights_final = np.zeros(len(objective.structure))
        else:
            features = features_vals[0]
            if len(features_vals) > 1:
                for i in range(1, len(features_vals)):
                    features = np.vstack([features, features_vals[i]])
            features = np.vstack([features, np.ones(features_vals[0].shape)]) # Добавляем константную фичу
            features = np.transpose(features)
            estimator = LinearRegression(fit_intercept=False)
            if features.ndim == 1:
                features = features.reshape(-1, 1)
            try:
                self.g_fun_vals = global_var.grid_cache.g_func.reshape(-1)
            except AttributeError:
                self.g_fun_vals = None
            estimator.fit(features, target_vals, sample_weight = self.g_fun_vals)

            valueable_weights = estimator.coef_
            weights = np.zeros(len(objective.structure))
            for weight_idx in range(len(weights)-1):
                if weight_idx in nonzero_features_indexes:
                    weights[weight_idx] = valueable_weights[nonzero_features_indexes.index(weight_idx)]
            weights[-1] = valueable_weights[-1]    
            objective.weights_final_evald = True
            objective.weights_final = weights
            
    def use_default_tags(self):
        self._tags = {'coefficient calculation', 'chromosome level', 'no suboperators', 'inplace'}
