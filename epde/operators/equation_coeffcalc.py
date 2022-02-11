#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 17 13:58:18 2021

@author: mike_ubuntu
"""

import numpy as np
from sklearn.linear_model import LinearRegression

from epde.operators.template import Compound_Operator

#class PopLevel_true_coeffs(Compound_Operator):
#    def apply(self, population):
#        for equation in population:
#            if not equation.weights_final_evald:
#                self.suboperators['True_coeff_calc'].apply(equation)
##                equation.weights_final_evald = True
#        return population
    
class LinReg_based_coeffs(Compound_Operator):
    '''
    
    The operatror, dedicated to the calculation of the weights of the equation (for the free coefficient and 
    each of its terms except the target one). 
    
    Methods:
    -----------
    apply(equation)
        Calculate the coefficients of the equation, using the linear regression. The result is stored in the 
        equation.weights_final attribute
            
    
    '''
    def apply(self, equation):
        """
        Calculate the coefficients of the equation, using the linear regression.The result is stored in the 
        equation.weights_final attribute

        Parameters:
        ------------
        equation : Equation object
            the equation object, to that the fitness function is obtained.
            
        Returns:
        ------------
        
        None
        """        
        assert equation.weights_internal_evald, 'Trying to calculate final weights before evaluating intermeidate ones (no sparsity).'
        target = equation.structure[equation.target_idx]
    
        target_vals = target.evaluate(False)
        features_vals = []
        nonzero_features_indexes = []
        for i in range(len(equation.structure)):
            if i == equation.target_idx:
                continue
            idx = i if i < equation.target_idx else i-1
            if equation.weights_internal[idx] != 0:
                features_vals.append(equation.structure[i].evaluate(False))
                nonzero_features_indexes.append(idx)
                
    #    print('Indexes of nonzero elements:', nonzero_features_indexes)
        if len(features_vals) == 0:
            equation.weights_final = np.zeros(len(equation.structure)) #Bind_Params([(token.label, token.params) for token in target.structure]), [('0', 1)]
        else:
            features = features_vals[0]
            if len(features_vals) > 1:
                for i in range(1, len(features_vals)):
                    features = np.vstack([features, features_vals[i]])
            features = np.vstack([features, np.ones(features_vals[0].shape)]) # Добавляем константную фичу
            features = np.transpose(features)  
    #        print('Done 2')        
            estimator = LinearRegression(fit_intercept=False)
            if features.ndim == 1:
                features = features.reshape(-1, 1)
                estimator.fit(features, target_vals)
            else:                
                # print('features', features.shape, 'target', target_vals.shape)
                # print((features == None).any())
                # print(features[:100, 1])
                estimator.fit(features, target_vals)
                
            valueable_weights = estimator.coef_
            weights = np.zeros(len(equation.structure))
            for weight_idx in range(len(weights)-1):
                if weight_idx in nonzero_features_indexes:
                    weights[weight_idx] = valueable_weights[nonzero_features_indexes.index(weight_idx)]
            weights[-1] = valueable_weights[-1]    
        #    print('weights check:', weights, equation.weights_internal)
            equation.weights_final_evald = True
    #        print('Set final weights')
    #        print('Done 3')
            equation.weights_final = weights
            
    @property
    def operator_tags(self):
        return {'coefficient calculation', 'equation level', 'no suboperators'}
            
