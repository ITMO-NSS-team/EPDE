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
    Constructs populations of systems of equations.
    
        This class is responsible for creating and managing populations of systems of equations,
        allowing for the configuration of equation sparsity, number of terms, and other parameters.
    
        Methods:
        - __init__
        - create
    
        Attributes:
            pool: The pool of families.
            terms_number: The number of terms.
            max_factors_in_term: The maximum number of factors in a term.
            vars_demand_equation: A set of variables from the families' demand equations.
            sparsity_interval: The sparsity interval.
    
        Class Methods:
        - __init__:
    """

    def __init__(self, pool, terms_number : int = 8, max_factors_in_term : int = 2, 
                 obj_functions : Callable = None, sparsity_interval : tuple = (1., 1.)):
        """
        Initializes the SystemsPopulationConstructor.
        
        This constructor sets up the necessary components for generating a population of equation system candidates.
        It prepares the search space by extracting variables from the provided pool of equation families and
        configuring the complexity constraints for the equation terms. The goal is to create a diverse initial
        population that can be evolved to discover the underlying system of equations.
        
        Args:
            pool: A pool of families, each representing a potential building block for the equation system.
            terms_number: The number of terms in each equation candidate. Defaults to 8.
            max_factors_in_term: The maximum number of factors allowed in a single term. Defaults to 2.
            obj_functions: Objective functions (not used). Defaults to None.
            sparsity_interval: The sparsity interval. Defaults to (1., 1.).
        
        Raises:
            ValueError: If the pool contains demand equations with more than one variable, indicating an attempt
                to discover a system of equations using a single-criterion optimization setup, which is not supported.
        
        Returns:
            None.
        
        Fields:
            pool: The pool of families.
            terms_number: The number of terms.
            max_factors_in_term: The maximum number of factors in a term.
            vars_demand_equation: A set of variables from the families' demand equations.
            sparsity_interval: The sparsity interval.
        """
        if sparsity_interval[0] != sparsity_interval[1]:
            warn(message = 'Single criterion optimization requires use of fixed sparsity constant. \
                            The right boundary of iterval will be used the value')
            # sparsity_interval[0] = sparsity_interval[1]
        
        self.pool = pool; self.terms_number = terms_number
        self.max_factors_in_term = max_factors_in_term 
        self.vars_demand_equation = set([family.variable for family in self.pool.families_demand_equation])
        print('self.vars_demand_equation', self.vars_demand_equation)
        if len(self.vars_demand_equation) > 1:
            raise ValueError('Trying to use single criterion optimization to discover a system of equations.')
        
        self.sparsity_interval = sparsity_interval

    def create(self, **kwargs):
        """
        Creates a new equation (SoEq) tailored for the evolutionary search process.
        
                This method instantiates a `SoEq` object, configuring its search space
                based on provided parameters like sparsity and complexity (number of terms,
                factors per term). These parameters define the landscape within which the
                evolutionary algorithm will explore potential equation solutions. The method
                also sets the objective functions that guide the search towards optimal
                equation discovery.
        
                Args:
                    **kwargs: Keyword arguments to customize equation parameters.
                        Possible keys include:
                            - sparsity (float): Sparsity value influencing equation complexity.
                            - terms_number (int): Number of terms in the equation.
                            - max_factors_in_term (int): Maximum number of factors per term.
                            - obj_funs (list): Objective functions to evaluate equation fitness.
        
                Returns:
                    SoEq: The newly created `SoEq` object, ready for evolutionary search.
        
                Class Fields (initialized in `SoEq.__init__`):
                    - obj_funs (list): Objective functions for the equation. Initialized to None.
                    - metaparameters (dict): Metaparameters dictionary for the search. Key - label of the parameter (e.g. 'sparsity'), value - tuple, containing flag for metaoptimization and initial value.
                    - tokens_for_eq (TFPool): Pool, containing token families for the equation search algorithm.
                    - tokens_supp (TFPool): Pool, containing token families for the equationless part.
                    - moeadd_set (bool): Flag indicating if the MOEADD solution is set. Initialized to False.
                    - vars_to_describe (list): List of variables to describe in the equation.
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

        created_solution.create()
        return created_solution