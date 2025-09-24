#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  8 11:57:44 2022

@author: maslyaev
"""

import numpy as np
from typing import Callable

from epde.structure.main_structures import SoEq

class SystemsPopulationConstructor(object):
    """
    Class for constructing an initial population of symbolic expressions to represent potential solutions. The construction process is tailored to the chosen optimization method and the defined search space.
    
    
        Attributes:
            pool (``):
            terms_number (`int`):
            max_factors_in_term (`int`):
            vars_demand_equation (``):
            sparsity_internal (`tuple`):
    """

    def __init__(self, pool, use_pic: bool = True, terms_number : int = 8, max_factors_in_term : int = 2, 
                 obj_functions : Callable = None, sparsity_interval : tuple = (0, 1)):
        """
        Initializes the SystemsPopulationConstructor.
        
                This constructor sets up the necessary parameters for generating a population of symbolic expressions representing potential equation structures. It configures the search space by defining the pool of families (building blocks for equations), the complexity of terms, and sparsity constraints. Extracting variables from the demand equation ensures that the generated expressions are relevant to the specific problem being addressed.
        
                Args:
                    pool: A pool of families, defining the basic elements for constructing equations.
                    use_pic: A boolean indicating whether to use PIC (Physics-Informed Collocation) loss function. Defaults to True.
                    terms_number: The number of terms in the symbolic expression. Defaults to 8.
                    max_factors_in_term: The maximum number of factors allowed within a single term. Defaults to 2.
                    obj_functions: Objective functions used for evaluating the fitness of candidate equations. Defaults to None.
                    sparsity_interval: A tuple representing the desired sparsity range for the generated equations. Defaults to (0, 1).
        
                Returns:
                    None.
        """
        self.pool = pool
        self.use_pic = use_pic 
        self.terms_number = terms_number
        self.max_factors_in_term = max_factors_in_term 
        self.vars_demand_equation = set([family.variable for family in self.pool.families_demand_equation])
        self.sparsity_interval = sparsity_interval
        print('self.vars_demand_equation', self.vars_demand_equation)        

    def applyToPassed(self, passed_solution: SoEq, **kwargs):
        """
        Applies objective functions to a passed solution.
        
                This method is crucial for evaluating the fitness of a candidate solution (equation) within the evolutionary process. It determines how well the equation describes the observed data by applying specified objective functions.
        
                Args:
                    passed_solution: The solution (equation) to which the objective functions will be applied. This solution's performance will be assessed based on these functions.
                    **kwargs: Keyword arguments. If 'obj_funs' is provided, these objective functions will be set for the solution.
        
                Returns:
                    None. The method modifies the passed_solution in place by setting or using objective functions.
        """
        try:
            passed_solution.set_objective_functions(kwargs['obj_funs'])
        except KeyError:
            passed_solution.use_default_multiobjective_function(self.use_pic)

    def create(self, **kwargs):
        """
        Creates a new equation (SoEq) tailored for the equation discovery process.
        
                This method constructs a `SoEq` object, configuring its objective functions
                and defining its structural components based on provided parameters. The
                equation's structure is crucial for the evolutionary search of governing
                equations.
        
                Args:
                    **kwargs: Keyword arguments containing parameters for equation creation.
                        sparsity: Sparsity value for the equation. If not provided, it's
                            randomly generated within the `sparsity_interval`.
                        terms_number: The number of terms in the equation. Defaults to
                            `self.terms_number`.
                        max_factors_in_term: The maximum number of factors in a term.
                            Defaults to `self.max_factors_in_term`.
                        obj_funs: Objective functions for the equation. If not provided,
                            default multiobjective function is used.
        
                Returns:
                    SoEq: The created `SoEq` object representing the new equation.
        """
        sparsity = kwargs.get('sparsity', 10 ** (np.random.uniform(low = np.log10(self.sparsity_interval[0]),
                                                                      high = np.log10(self.sparsity_interval[1]),
                                                                      size = len(self.vars_demand_equation))))
        terms_number = kwargs.get('terms_number', self.terms_number)
        max_factors_in_term = kwargs.get('max_factors_in_term', self.max_factors_in_term)
        
        print(f'Creating new equation, sparsity value {sparsity}')
        metaparameters = {'terms_number'        : {'optimizable' : False, 'value' : terms_number},
                          'max_factors_in_term' : {'optimizable' : False, 'value' : max_factors_in_term}}
        for idx, variable in enumerate(self.vars_demand_equation):
            metaparameters[('sparsity', variable)] = {'optimizable' : True, 'value' : sparsity[idx]}

        created_solution = SoEq(pool = self.pool, metaparameters = metaparameters)

        try:
            created_solution.set_objective_functions(kwargs['obj_funs'])
        except KeyError:
            created_solution.use_default_multiobjective_function(use_pic=self.use_pic)



        created_solution.create()

        return created_solution
