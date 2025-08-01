#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 13:35:18 2021

@author: mike_ubuntu
"""

from typing import Union, Callable
import numpy as np
from sklearn.linear_model import Lasso, ElasticNet
from sklearn.preprocessing import normalize

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
                          selection='random', tol=0.0001, warm_start=False)

        # estimator = ElasticNet(alpha=objective.metaparameters[('sparsity', objective.main_var_to_explain)]['value'],
        #                   copy_X=True, fit_intercept=True, max_iter=1000,
        #                   positive=False, precompute=False, random_state=None,
        #                   selection='random', tol=0.0001, warm_start=False)

        _, target, features = objective.evaluate(normalize = True, return_val = False)
        # features = normalize(features)
        self.g_fun_vals = global_var.grid_cache.g_func.reshape(-1)

        estimator.fit(features, target, sample_weight = self.g_fun_vals)
        objective.weights_internal = estimator.coef_

        # Remove common terms
        nonzero_terms_mask = np.array([False if weight == 0 else True for weight in objective.weights_internal], dtype=np.integer)
        nonzero_terms_mask = np.append(nonzero_terms_mask, True) # Include right side
        nonzero_terms = [item for item, keep in zip(objective.structure, nonzero_terms_mask) if keep]
        nonzero_terms_labels = [list(term.cache_label) if not isinstance(term.cache_label[0], tuple) else term.cache_label for term in nonzero_terms]
        for i in range(len(nonzero_terms_labels)):
            if isinstance(nonzero_terms_labels[i][0], tuple):
                nonzero_terms_labels[i] = [list(_) for _ in nonzero_terms_labels[i]]
            else:
                nonzero_terms_labels[i] = [nonzero_terms_labels[i]]
        if nonzero_terms_labels[1:]:
            common_item = nonzero_terms_labels[0]
            for term in common_item:
                if all([term in label for label in nonzero_terms_labels[1:]]):
                    for item in nonzero_terms:
                        idxs_to_delete = []
                        for factor_idx in range(len(item.structure)):
                            if item.structure[factor_idx].cache_label == tuple(term):
                                idxs_to_delete.append(factor_idx)
                                last_removed_mandatory = item.structure[factor_idx].mandatory
                                last_removed_deriv = item.structure[factor_idx].is_deriv
                        new_terms = [val for i, val in enumerate(item.structure) if i not in idxs_to_delete]
                        if len(new_terms) == 0:
                            item.randomize(mandatory_family=last_removed_mandatory, create_derivs=last_removed_deriv)
                            while objective.structure.count(item) > 1:
                                item.randomize(mandatory_family=last_removed_mandatory,
                                               create_derivs=last_removed_deriv)
                            item.reset_saved_state()
                            continue
                        item.structure = new_terms
                        while objective.structure.count(item) > 1:
                            item.randomize(mandatory_family=last_removed_mandatory,
                                           create_derivs=last_removed_deriv)
                        item.reset_saved_state()
                    objective.reset_state(reset_right_part=True)
                    self.apply(objective, arguments)
                    return
        # print(objective.text_form)


    def use_default_tags(self):
        self._tags = {'sparsity', 'gene level', 'no suboperators', 'inplace'}
