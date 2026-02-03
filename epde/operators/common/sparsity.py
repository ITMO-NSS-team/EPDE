#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 13:35:18 2021

@author: mike_ubuntu
"""

import numpy as np
import epde.globals as global_var
from epde.operators.utils.template import CompoundOperator
from epde.structure.main_structures import Equation
import time
from sklearn.base import BaseEstimator, RegressorMixin
# import seaborn as sns
import matplotlib.pyplot as plt
from epde.supplementary import calculate_weights


class PhysicsInformedLasso(BaseEstimator, RegressorMixin):
    def __init__(self, max_iter=20, tol=1e-4, grid_shape=None):
        self.max_iter = max_iter
        self.tol = tol
        self.grid_shape = grid_shape

    def _soft_threshold(self, x, lambda_):
        return np.sign(x) * np.maximum(np.abs(x) - lambda_, 0)

    def get_cv(self, weights):
        # Calculate Coefficient of Variation (CV)
        weights_arr = np.array(weights)
        std = weights_arr.std(axis=0, ddof=1)
        mu = weights_arr.mean(axis=0)

        # Safe division
        with np.errstate(divide='ignore', invalid='ignore'):
            cv = (std ** 2) / (mu ** 2)
            cv[mu == 0] = 0.0  # Handle zero mean

        return np.nan_to_num(cv)

    def fit(self, X, y, sample_weights):
        self.n_samples, self.n_features = X.shape

        # 1. Initial Weights
        weights = calculate_weights(X, y, sample_weights=sample_weights, grid_shape=self.grid_shape)
        cv = self.get_cv(weights[:, :-1])

        self.coef_ = weights.mean(axis=0)[:-1]
        self.intercept_ = weights.mean(axis=0)[-1]

        norm_sq_features = np.sum(X ** 2, axis=0)
        residual = y - (X @ self.coef_ + self.intercept_)

        # 2. Coordinate Descent Loop
        for iteration in range(self.max_iter):
            max_change = self.tol

            if all(self.coef_ == 0):
                break

            # Sort features by instability (highest CV first)
            for j in np.argsort(cv)[::-1]:
                old_coef = self.coef_[j]

                if old_coef == 0:
                    continue

                norm_sq = norm_sq_features[j]
                y_sq_sum = np.sum((y - self.intercept_) ** 2)

                # Partial residual correlation
                rho = np.dot(X[:, j], residual) + old_coef * norm_sq

                # Use CV-based Thresholding
                threshold = cv[j] * y_sq_sum
                # threshold = cv[j] * self.n_samples
                # threshold = cv[j] * norm_sq * abs(old_coef)
                new_coef = self._soft_threshold(rho, threshold) / norm_sq

                self.coef_[j] = new_coef

                if new_coef == 0:
                    weights = calculate_weights(X[:, self.coef_ != 0], y, sample_weights=sample_weights, grid_shape=self.grid_shape)
                    new_cv = iter(self.get_cv(weights[:, :-1]))
                    cv = np.array([next(new_cv) if _ else 0 for _ in self.coef_ != 0])

                    new_coef = iter(weights.mean(axis=0)[:-1])
                    self.coef_ = np.array([next(new_coef) if _ else 0 for _ in self.coef_ != 0])
                    self.intercept_ = weights.mean(axis=0)[-1]
                    residual = y - (X @ self.coef_ + self.intercept_)
                    break

                residual -= (new_coef - old_coef) * X[:, j]
                change = abs(new_coef - old_coef) / abs(old_coef)
                # change = abs(self.intercept_ - old_intercept) / abs(old_intercept)
                max_change = max(max_change, change)

            if max_change < self.tol:
                break
        # print(iteration)
        return self


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

        estimator = PhysicsInformedLasso(grid_shape=global_var.grid_cache.inner_shape)

        _, target, features = objective.evaluate(normalize = True, return_val = False)

        self.g_fun_vals = global_var.grid_cache.g_func[global_var.grid_cache.g_func != 0]

        estimator.fit(features, target, self.g_fun_vals)
        objective.weights_internal = estimator.coef_
        objective.weights_internal_evald = True
        objective.weights_final = np.append(objective.weights_internal, estimator.intercept_)
        objective.weights_final_evald = True
        objective.weights_final = [weight for weight in objective.weights_final if weight != 0]


    def use_default_tags(self):
        self._tags = {'sparsity', 'gene level', 'no suboperators', 'inplace'}

        
