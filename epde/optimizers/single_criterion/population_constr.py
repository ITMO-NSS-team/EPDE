#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 11:57:44 2022

@author: maslyaev
"""

from typing import Callable
from warnings import warn

from epde.structure.main_structures import SoEq

class SystemsPopulationConstructor(object):
    """
    Constructor for population with systems of equations

    Attributes:
        pool (`TFPool`): familys of tokens, that can be in equation 
        terms_number (`int`): maximum number of terms in equation. Insignificant terms will be filtered out by LASSO operator, but still be keeped with 0 weights. 
        max_factors_in_term (`int`): maximum number of tokens in term
        vars_demand_equation (`list`): types of token's families, that require an individual equation to be represented in the system
        sparsity_interval (`tuple`): 
    """
    def __init__(self, pool, terms_number : int = 8, max_factors_in_term : int = 2, 
                 obj_functions : Callable = None, sparsity_interval : tuple = (1., 1.)):
        if sparsity_interval[0] != sparsity_interval[1]:
            warn(message = 'Single criterion optimization requires use of fixed sparsity constant. \
                            The right boundary of iterval will be used the value')
            # sparsity_interval[0] = sparsity_interval[1]
        
        self.pool = pool
        self.terms_number = terms_number
        self.max_factors_in_term = max_factors_in_term 
        self.vars_demand_equation = [family.ftype for family in self.pool.families_demand_equation]
        if len(self.vars_demand_equation) > 1:
            raise ValueError("Single criterion optimization should bot be used to discover a system of equations. Use multiobjective mode instead.")
        
        self.sparsity_interval = sparsity_interval

    def create(self, **kwargs):
        """
        Creating solution for equation
        """
        sparsity = kwargs.get('sparsity', self.sparsity_interval[1])
        terms_number = kwargs.get('terms_number', self.terms_number)
        max_factors_in_term = kwargs.get('max_factors_in_term', self.max_factors_in_term)
        
        print(f'Creating new equation, sparsity value {sparsity}')
        metaparameters = {'terms_number'        : {'optimizable' : False, 'value' : terms_number},
                          'max_factors_in_term' : {'optimizable' : False, 'value' : max_factors_in_term}}
        for idx, variable in enumerate(self.vars_demand_equation):
            metaparameters[('sparsity', variable)] = {'optimizable' : False, 'value' : sparsity}

        created_solution = SoEq(pool = self.pool, metaparameters = metaparameters)

        try:
            created_solution.set_objective_functions(kwargs['obj_funs'])
        except KeyError:
            created_solution.use_default_singleobjective_function()

        created_solution.create_equations()
        return created_solution