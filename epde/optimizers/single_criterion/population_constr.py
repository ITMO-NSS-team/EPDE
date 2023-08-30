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
    def __init__(self, pool, terms_number : int = 8, max_factors_in_term : int = 2, 
                 obj_functions : Callable = None, sparsity_interval : tuple = (1., 1.)):
        if sparsity_interval[0] != sparsity_interval[1]:
            warn(message = 'Single criterion optimization requires use of fixed sparsity constant. \
                            The right boundary of iterval will be used the value')
            # sparsity_interval[0] = sparsity_interval[1]
        
        self.pool = pool; self.terms_number = terms_number
        self.max_factors_in_term = max_factors_in_term 
        self.vars_demand_equation = [family.ftype for family in self.pool.families_demand_equation]
        if len(self.vars_demand_equation) > 1:
            raise ValueError('Trying to use single criterion optimization to discover a system of equations.')
        
        self.sparsity_interval = sparsity_interval

    def create(self, **kwargs):
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