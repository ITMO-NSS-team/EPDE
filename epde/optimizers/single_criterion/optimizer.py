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
from epde.optimizers.single_criterion.population_constr import SystemsPopulationConstructor
from epde.optimizers.builder import OptimizationPatternDirector

class EvolutionaryStrategy(Strategy):
    """
    Evolutionary strategy for the single criterion optimization. Includes method ``iteration`` for a single 
    iteration of the algotirhm & ``run`` for executing a full optimization.

    Attributes:
        _stop_criterion (`IteratorLimit`): object for stored the condition for stopping the evolutionary algorithm
        run_performed (`bool`): flag about ranning evolutionary strategy
        linked_blocks (`LinkedBlocks`): the sequence (not necessarily chain: divergencies can be present) of blocks with evolutionary operators
    """
    def __init__(self, stop_criterion = IterationLimit, sc_init_kwargs: dict = {'limit' : 50}):
        super().__init__()
        self._stop_criterion = stop_criterion(**sc_init_kwargs)
        self.run_performed = False
    
    def run(self, initial_population: Iterable, EA_kwargs: dict, stop_criterion_params: dict = {}):
        """
        Running evolutionary strategy for input population

        Args:
            intial_population (`Iterable`): population only after initialized
            EA_kwargs (`dict`): arguments for evolutionary algoritm
            stop_criterion_params (`dict`): parameters for stoping evolutionary

        Returns:
            None
        """
        self._stop_criterion.reset(**stop_criterion_params)
        population = initial_population
        while not self._stop_criterion.check():
            self.linked_blocks.traversal(population, EA_kwargs)
            population = self.linked_blocks.output
        self.run_performed = True

class Population(object):
    """
    Class for keeping population

    Attributes:
        population (`list`): list of individs
        length (`int`): number of individs in population
    """
    def __init__(self, elements: list, sorting_method: Callable):
        """
        Args:
            elements (`list`): list of individs
            sorting_method (`Callable`): method for sortiing of individs in population
        """
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
        """
        Adding new candidate to population
        """
        self.population.append(point)
        
    def delete_point(self, point):
        """
        Deleting specified candidate solution
        """
        self.population = [solution for solution in self.population if solution != point]

    def get_stats(self):
        return np.array([element.obj_fun for element in self.population])

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
    """
    
    """
    def __init__(self, population_instruct, pop_size, solution_params, sorting_method = simple_sorting): 
        soluton_creation_attempts_softmax = 10
        soluton_creation_attempts_hardmax = 100

        pop_constructor = SystemsPopulationConstructor(**population_instruct)
        
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

    def set_strategy(self, strategy_director):
        builder = strategy_director.builder
        builder.assemble(True)
        self.strategy = builder.processer
        # self.strategy = strategy
        
    def optimize(self, EA_kwargs: dict = {},  epochs: int = None):
        scp = {'limit' : epochs} if epochs is not None else {'limit' : 50}
        global_var.reset_hist()
        
        self.strategy.run(initial_population = self.population, EA_kwargs = EA_kwargs, 
                          stop_criterion_params = scp)
