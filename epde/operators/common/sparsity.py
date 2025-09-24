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
        Applies LASSO regression to identify significant terms within an equation.
        
                This method leverages LASSO regression to automatically determine the importance of each term in the equation.
                By treating one term as the target variable and the remaining terms as features, LASSO regression identifies the coefficients that best reconstruct the target term.
                The L1 regularization of LASSO encourages sparsity, effectively selecting the most relevant terms and simplifying the equation.
                The resulting coefficients are then stored within the equation object, providing a compact representation of the equation's structure.
        
                Args:
                    objective (Equation): The equation object to which the coefficients are applied.
                    arguments (dict): A dictionary containing arguments for the operator.
        
                Returns:
                    None: The method modifies the equation object in place, assigning the LASSO coefficients to the `weights_internal` attribute.
        """
        # print(f'Metaparameter: {objective.metaparameters}, objective.metaparameters[("sparsity", objective.main_var_to_explain)]')
        self_args, subop_args = self.parse_suboperator_args(arguments = arguments)

        estimator = Lasso(alpha = objective.metaparameters[('sparsity', objective.main_var_to_explain)]['value'],
                          copy_X=True, fit_intercept=True, max_iter=1000,
                          positive=False, precompute=False, random_state=None,
                          selection='random', tol=0.0001, warm_start=False)
        _, target, features = objective.evaluate(normalize = True, return_val = False)
        self.g_fun_vals = global_var.grid_cache.g_func.reshape(-1)

        estimator.fit(features, target, sample_weight = self.g_fun_vals)
        objective.weights_internal = estimator.coef_

    def use_default_tags(self):
        """
        Sets the operator's tags to a predefined default set.
        
        This ensures that the operator is correctly categorized with general characteristics 
        related to sparsity, gene-level operations, lack of sub-operators, and in-place computation.
        This is important for the framework to properly identify and utilize this operator within 
        the equation discovery process.
        
        Args:
            self: The operator instance.
        
        Returns:
            None.
        
        Class Fields:
            _tags (set): A set containing default tags: 'sparsity', 'gene level', 'no suboperators', and 'inplace'. These tags describe general characteristics of the operator.
        """
        self._tags = {'sparsity', 'gene level', 'no suboperators', 'inplace'}

        
