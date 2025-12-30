#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 13:35:18 2021

@author: mike_ubuntu
"""

import numpy as np
from sklearn.linear_model import Lasso, LassoLars, OrthogonalMatchingPursuit, Ridge, ElasticNet, SGDRegressor
# from cuml.linear_model import Ridge
# from pysindy import STLSQ, SR3
from scipy.linalg import lstsq
import epde.globals as global_var
from epde.operators.utils.template import CompoundOperator
from epde.structure.main_structures import Equation
import time
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class CustomPhysicsLasso(BaseEstimator, RegressorMixin):
    def __init__(self, max_iter=100, tol=1e-4):
        self.max_iter = max_iter
        self.tol = tol

    def _soft_threshold(self, x, lambda_):
        return np.sign(x) * np.maximum(np.abs(x) - lambda_, 0)

    def get_cv(self, weights):
        std = np.array(weights).std(axis=0, ddof=1)
        mu = np.array(weights).mean(axis=0)
        # cv = std ** 2 / (std ** 2 + mu ** 2)
        # cv = np.sqrt(std ** 2 / (std ** 2 + mu ** 2))
        cv = std ** 2 / (mu ** 2)
        return cv

    def calculate_weights(self, X, y):
        X_aug = np.column_stack([X, np.ones(self.n_samples)])
        weights = []
        for _ in range(30):
            idx = np.random.choice(self.n_samples, self.batch_size, replace=False)
            X_batch = X_aug[idx]
            y_batch = y[idx]
            w_full, _, _, _ = np.linalg.lstsq(X_batch, y_batch, rcond=None)
            weights.append(w_full)

        return weights

    def fit(self, X, y):
        X, y = check_X_y(X, y, dtype=np.float64)
        self.n_samples, self.n_features = X.shape
        self.batch_size = int(self.n_samples * 0.5)  # 50% of data

        # --- 1. Initialization ---
        # Add column of 1s to solve for intercept correctly via OLS
        weights = self.calculate_weights(X, y)
        cv = self.get_cv(weights)

        self.coef_ = np.array(weights).mean(axis=0)[:-1]
        self.intercept_ = np.array(weights).mean(axis=0)[-1]

        # Pre-compute norms of features (optimization)
        # These are constant throughout the loop
        norm_sq_features = np.sum(X ** 2, axis=0)

        # Pre-compute initial residual: r = y - (Xw + b)
        y_pred = X @ self.coef_ + self.intercept_
        residual = y - y_pred

        max_change_old = 0.0

        # --- 2. Coordinate Descent Loop ---
        for iteration in range(self.max_iter):
            max_change = 0.0

            # A. Update Intercept (Unpenalized)
            # The optimal intercept shift is simply the mean of the residuals
            # because we want mean(y - Xw - b_new) = 0
            intercept_shift = np.mean(residual)
            self.intercept_ += intercept_shift
            residual -= intercept_shift

            # B. Update Coefficients
            for j in range(self.n_features):
                if self.coef_[j] == 0:
                    continue

                old_coef = self.coef_[j]
                norm_sq = norm_sq_features[j]

                # Skip constant columns to avoid division by zero
                # if norm_sq == 0:
                #     continue

                # 1. Calculate partial residual correlation
                # This represents the correlation between feature j and the target
                # if feature j were removed from the model.
                # rho = dot(X_j, residual + old_coef * X_j)
                rho = np.dot(X[:, j], residual) + old_coef * norm_sq

                # 2. Soft Thresholding
                # Threshold is N * alpha
                threshold = cv[j] * sum(y ** 2)
                new_coef = self._soft_threshold(rho, threshold) / norm_sq

                # 3. Update State
                self.coef_[j] = new_coef
                # Update residual vector efficiently
                # r_new = r_old - (w_new - w_old) * X_j
                residual -= (new_coef - old_coef) * X[:, j]
                max_change = max(max_change, abs((new_coef - old_coef) / old_coef))

            if abs(max_change - max_change_old) < self.tol:
                break

            max_change_old = max_change

        self.n_iter_ = iteration + 1
        # print("-------")
        # print(self.n_iter_)
        # print(np.mean(cv[:-1]))
        # print(sum(abs(y - X @ self.coef_ - self.intercept_)) / sum(abs(y)))
        # print(self.coef_, self.intercept_)
        return self

    def predict(self, X):
        check_is_fitted(self)
        X = check_array(X)
        return X @ self.coef_ + self.intercept_


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

        # estimator = Lasso(alpha = objective.metaparameters[('sparsity', objective.main_var_to_explain)]['value'],
        #                   copy_X=True, fit_intercept=True, max_iter=1000,
        #                   positive=False, precompute=False, random_state=None,
        #                   selection='random', tol=0.0001, warm_start=True)
        # estimator = SGDRegressor(alpha=objective.metaparameters[('sparsity', objective.main_var_to_explain)]['value'],
        #                   penalty='l1', fit_intercept=True, max_iter=1000,
        #                   random_state=None, tol=0.0001, warm_start=False)
        # estimator = Ridge(alpha=objective.metaparameters[('sparsity', objective.main_var_to_explain)]['value'],
        #                   copy_X=True, fit_intercept=True,
        #                   positive=False, random_state=None,
        #                   tol=0.0001, solver='cholesky')
        estimator = CustomPhysicsLasso()
        # estimator = ElasticNet(alpha=objective.metaparameters[('sparsity', objective.main_var_to_explain)]['value'],
        #                        l1_ratio=objective.metaparameters[('threshold', objective.main_var_to_explain)]['value'],
        #                        copy_X=True, fit_intercept=True, max_iter=1000,
        #                        positive=False, precompute=False, random_state=None,
        #                        selection='random', tol=0.0001, warm_start=False
        #                        )
        # estimator = OrthogonalMatchingPursuit(n_nonzero_coefs=objective.metaparameters[('nonzero_terms', objective.main_var_to_explain)]['value'], fit_intercept=True)
        # estimator = STLSQ(threshold=objective.metaparameters[('sparsity', objective.main_var_to_explain)]['value'],
        #                   copy_X=True, unbias=True, max_iter=20, alpha=objective.metaparameters[('threshold', objective.main_var_to_explain)]['value'], ridge_kw={"tol": 1e-10})
        # estimator = SR3(reg_weight_lam=objective.metaparameters[('sparsity', objective.main_var_to_explain)]['value'],
        #                 regularizer='L2', relax_coeff_nu=objective.metaparameters[('nu', objective.main_var_to_explain)]['value'],
        #                 copy_X=True, unbias=True)

        start_time = time.time()  # record start time
        _, target, features = objective.evaluate(normalize = True, return_val = False)
        end_time = time.time()  # record end time
        elapsed_time = end_time - start_time
        # print(f"Elapsed time for evaluating: {elapsed_time / len(features.reshape(-1)):.12f} seconds")

        self.g_fun_vals = global_var.grid_cache.g_func[global_var.grid_cache.g_func != 0]

        # fraction = 0.1
        # num_subsample = int(len(target) * fraction)
        #
        # probabilities = self.g_fun_vals / np.sum(self.g_fun_vals)
        # indices = np.random.choice(
        #     a=np.arange(len(target)),
        #     size=num_subsample,
        #     replace=False,
        #     p=probabilities
        # )
        #
        # features_subsampled = features[indices]
        # target_subsampled = target[indices]
        # weights_subsampled = self.g_fun_vals[indices]

        start_time = time.time()  # record start time
        # estimator.fit(features, target, sample_weight = self.g_fun_vals)
        end_time = time.time()  # record end time
        elapsed_time = end_time - start_time
        # print(f"Elapsed time for fitting: {elapsed_time / len(features.reshape(-1)):.12f} seconds")

        # estimator.fit(features_subsampled, target_subsampled, sample_weight=weights_subsampled)
        estimator.fit(features, target)
        objective.weights_internal = estimator.coef_
        # print(estimator.coef_)
        # objective.weights_internal = np.where(
        #     np.abs(estimator.w) < objective.metaparameters[('threshold', objective.main_var_to_explain)]['value'],
        #     0, estimator.w)
        # objective.weights_internal = estimator.coef_
        # objective.weights_internal = np.where(np.abs(estimator.coef_) < objective.metaparameters[('threshold', objective.main_var_to_explain)]['value'], 0, estimator.coef_)
        # objective.weights_internal = estimator.coef_[0]
        # objective.weights_internal = np.where(np.abs(estimator.coef_[0]) < objective.metaparameters[('threshold', objective.main_var_to_explain)]['value'], 0, estimator.coef_[0])
        objective.weights_internal_evald = True

    def use_default_tags(self):
        self._tags = {'sparsity', 'gene level', 'no suboperators', 'inplace'}

        
