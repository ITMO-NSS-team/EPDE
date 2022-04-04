#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 16:59:24 2021

@author: mike_ubuntu
"""

import numpy as np
from copy import deepcopy
#import 
from epde.structure import SoEq

def set_argument(var, fun_kwargs, base_value):
    try:
        res = fun_kwargs[var]
    except KeyError:
        res = base_value
    return res

class Systems_population_constructor(object):
    def __init__(self, pool, terms_number, max_factors_in_term, eq_search_evo, 
                 obj_functions = None, sparsity_interval = (0, 1)):
        self.pool = pool; self.terms_number = terms_number
        self.eq_search_evo = eq_search_evo
        self.max_factors_in_term = max_factors_in_term; #self.eq_search_iters = eq_search_iters
        self.equation_number = len(self.pool.families_meaningful)
        self.sparsity_interval = sparsity_interval
#        print(self.equation_number)
#        raise NotImplementedError
        #len([1 for token_family in self.tokens if token_family.status['meaningful']])
    
    def create(self, **kwargs): # Дописать
        pop_size = set_argument('population_size', kwargs, 16)
        sparsity = set_argument('sparsity', kwargs, np.power(np.e, np.random.uniform(low = np.log(self.sparsity_interval[0]),
                                                                      high = np.log(self.sparsity_interval[1]),
                                                                      size = self.equation_number)))
#        eq_search_iters = set_argument('eq_search_iters', kwargs, 50)
        
        print(f'Creating new equation, sparsity value {sparsity}')
        created_solution = SoEq(pool = self.pool, terms_number = self.terms_number,
                                max_factors_in_term = self.max_factors_in_term, 
                                sparsity = sparsity)
        try:
            created_solution.set_objective_functions(kwargs['obj_funs'])
        except KeyError:
            created_solution.use_default_objective_function()
        created_solution.set_eq_search_evolutionary(self.eq_search_evo)
#        print('searching equations with ', eq_search_iters, 'iterations, and popsize of ', pop_size)
        created_solution.create_equations(pop_size, sparsity)
        print('Equation created', type(created_solution))
        return created_solution        
    

class sys_search_evolutionary_operator(object): # Возможно, организовать наследование от эвол. оператора из eq_search_operators
    def __init__(self, xover, mutation):
        '''
        Define the evolutionary operator to be used in the search of system of differential equations. 
        
        Parameters:
            xover : function
                The crossover/recombination operator for the evolutionary algorithm, specified in form of function, that
                must take two parent individuals as arguments and returns two offsprings.
                
            mutation : function
                The mutation operator for the evolutionary, which must take an individual as parameter and will return the changed 
                copy of the input individual.
                
                
        '''
        self._xover = xover
        self._mutation = mutation
        
    def mutation(self, solution):
        output = self._mutation(solution)
        output.create_equations()
        return output

    def crossover(self, parents_pool):
        offspring_pool = []
        for idx in np.arange(np.int(np.floor(len(parents_pool)/2.))):
#            print(parents_pool[2*idx].vals, parents_pool[2*idx+1].vals)
            offsprings_generated = self._xover((parents_pool[2*idx], parents_pool[2*idx+1]))
            offspring_pool.extend(offsprings_generated)
        for offspring in offspring_pool:
            offspring.create_equations()
        return offspring_pool
    
    
def gaussian_mutation(solution):
    assert isinstance(solution, SoEq), 'Object of other type, than the system of equation (SoEq), has been passed to the mutation operator'
    solution_new = deepcopy(solution)
    solution_new.set_eq_search_evolutionary(solution.eq_search_evolutionary_strategy)
    
    solution_new.vals += np.random.normal(size = solution_new.vals.size)
    return solution_new


def mixing_xover(parents):
    assert all([isinstance(parent, SoEq) for parent in parents]), 'Object of other type, than the system of equation (SoEq), has been passed to the crossover operator'
    proportion = np.random.uniform(low = 1e-6, high = 0.5-1e-6)
    offsprings = [deepcopy(parent) for parent in parents]

    offsprings[0].precomputed_value = False; offsprings[1].precomputed_value = False
    offsprings[0].precomputed_domain = False; offsprings[1].precomputed_domain = False
    
    # strategy = parents[0].eq_search_evolutionary_strategy
    # offsprings[0].set_eq_search_evolutionary(strategy)
    # offsprings[1].set_eq_search_evolutionary(strategy)
    # offsprings[0].def_eq_search_iters = parents[0].def_eq_search_iters
    # offsprings[1].def_eq_search_iters = parents[0].def_eq_search_iters

    offsprings[0].vals = parents[0].vals + proportion * (parents[1].vals - parents[0].vals)
    offsprings[1].vals = parents[0].vals + (1 - proportion) * (parents[1].vals - parents[0].vals)

    return offsprings

