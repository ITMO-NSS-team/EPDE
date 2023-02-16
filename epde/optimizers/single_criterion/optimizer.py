#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 20:17:30 2023

@author: maslyaev
"""

from typing import Iterable, Callable
import warnings

import numpy as np

import epde.globals as global_var
from epde.optimizers.strategy import Strategy
from epde.optimizers.single_criterion.ea_stop_conds import IterationLimit
from epde.optimizers.single_criterion.supplementary import simple_sorting

# class PopulationProcesserBuilder(StrategyBuilder):
#     """
#     Class of sector process builder for moeadd. 
    
#     Attributes:
#     ------------
    
#     operator : Evolutionary_operator object
#         the evolutionary operator, which is being constructed for the evolutionary algorithm, which is applied to a population;
        
#     Methods:
#     ------------
    
#     reset()
#         Reset the evolutionary operator, deleting all of the declared suboperators.
        
#     set_evolution(crossover_op, mutation_op)
#         Set crossover and mutation operators with corresponding evolutionary operators, each of the Specific_Operator type object, to improve the 
#         quality of the population.
    
#     set_param_optimization(param_optimizer)
#         Set parameter optimizer with pre-defined Specific_Operator type object to optimize the parameters of the factors, present in the equation.
        
#     set_coeff_calculator(coef_calculator)
#         Set coefficient calculator with Specific_Operator type object, which determines the weights of the terms in the equations.
        
#     set_fitness(fitness_estim)
#         Set fitness function value estimator with the Specific_Operator type object. 
    
#     """
#     def reset(self): # stop_criterion, stop_criterion_kwargs
#         self._processer = EvolutionaryStrategy() # stop_criterion, stop_criterion_kwargs
#         super().__init__()
    
#     @property
#     def processer(self):
#         return self._processer

class EvolutionaryStrategy(Strategy):
    '''
    Evolutionary strategy for the single criterion optimization. Includes method ``iteration`` for a single 
    iteration of the algotirhm & ``run`` for executing a full optimization.
    '''
    def __init__(self, stop_criterion = IterationLimit, sc_init_kwargs: dict = {'limit' : 50}):
        super().__init__()
        self._stop_criterion = stop_criterion(**sc_init_kwargs)
        self.run_performed = False
            
    def iteration(self, population_subset, EA_kwargs = None):
        self.check_integrity()
        self.linked_blocks.blocks_labeled['initial'].set_output(population_subset)
        self.linked_blocks.traversal(EA_kwargs)
        return self.linked_blocks.output
    
    def run(self, initial_population: Iterable, EA_kwargs: dict, stop_criterion_params: dict = {}):
        self._stop_criterion.reset(**stop_criterion_params)
        population = initial_population
        while not self._stop_criterion.check():
            self.linked_blocks.traversal(population, EA_kwargs)
            population = self.linked_blocks.output
        self.run_performed = True

class Population(object):
    def __init__(self, elements: list, sorting_method: Callable):
        self.population = elements
        self.length = len(elements)
        self._sorting_method = sorting_method
        
    def sort(self):
        '''
        Method, that returns sorted population of the candidates. 
        Does not change anything inside the population.
        '''
        return self._sorting_method(self.population) # TODO: finish that piece of code.
    
    def sorted(self):
        '''
        Method to sort the population. Operates in "in-place" mode.
        '''
        self.population = self.sort() 

    def update(self, point):
        self.population.append(point)
        
    def delete_point(self, point):
        self.population = [solution for solution in self.population if solution != point]

    def __setitem__(self, key, value):
        self.population[key] = value

    def __getitem__(self, key):
        return self.population[key]
        
    def __iter__(self):
        return PopulationIterator(self)
    
    
class PopulationIterator(object):
    def __init__(self, population):
        self._population = population
        self._idx = 0

    def __next__(self):
        if self._idx < len(self._population.population):
            res = self._population.population[self._idx]
            self._idx += 1
            return res
        else:
            raise StopIteration


class SimpleOptimizer(object):
    def __init__(self, pop_constructor, pop_size, solution_params, sorting_method = simple_sorting): 
        soluton_creation_attempts_softmax = 10
        soluton_creation_attempts_hardmax = 100
        
        assert type(solution_params) == type(None) or type(solution_params) == dict, 'The solution parameters, passed into population constructor must be in dictionary'
        initial_population = []
        for solution_idx in range(pop_size):
            solution_gen_idx = 0
            while True:
                if type(solution_params) == type(None): solution_params = {}
                temp_solution = pop_constructor.create(**solution_params)
                if not np.any([temp_solution == solution for solution in initial_population]):
                    initial_population.append(temp_solution)
                    print(f'New solution accepted, confirmed {len(initial_population)}/{pop_size} solutions.')
                    break
                if solution_gen_idx == soluton_creation_attempts_softmax and global_var.verbose.show_warnings:
                    print('solutions tried:', solution_gen_idx)
                    warnings.warn('Too many failed attempts to create unique solutions for multiobjective optimization. Change solution parameters to allow more diversity.')
                if solution_gen_idx == soluton_creation_attempts_hardmax:
                    raise RuntimeError('Can not place an individual into the population even with many attempts.')
                solution_gen_idx += 1
                
        self.population = Population(elements = initial_population, sorting_method = sorting_method)

    def set_strategy(self, strategy: EvolutionaryStrategy):
        self.strategy = strategy
        
    def optimize(self, EA_kwargs: dict = {},  epochs: int = None):
        scp = {} if epochs is None else {'limit' : 50}

        self.strategy.run(initial_population = self.population, EA_kwargs = EA_kwargs, 
                          stop_criterion_params = scp)