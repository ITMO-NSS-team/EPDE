#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 20:17:30 2023

@author: maslyaev
"""

from typing import Iterable, Callable, Union, List
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
    Evolutionary strategy for single-objective optimization. Includes the `iteration` method for performing a single step of the algorithm and the `run` method for executing a complete optimization process.
    
    
        Attributes:
            _stop_criterion (`IteratorLimit`): object for stored the condition for stopping the evolutionary algorithm
            run_performed (`bool`): flag about ranning evolutionary strategy
            linked_blocks (`LinkedBlocks`): the sequence (not necessarily chain: divergencies can be present) of blocks with evolutionary operators
    """

    def __init__(self, stop_criterion = IterationLimit, sc_init_kwargs: dict = {'limit' : 50}):
        """
        Initializes the EvolutionaryStrategy with a specified stopping criterion.
        
                This constructor configures the condition that will halt the evolutionary search
                for differential equations. The stopping criterion is instantiated based on the
                provided class and keyword arguments. A flag is also set to track whether the
                evolutionary process has been executed.
        
                Args:
                    stop_criterion: The class defining the stopping criterion. Defaults to IterationLimit,
                        which stops the search after a fixed number of iterations.
                    sc_init_kwargs: Keyword arguments used to initialize the stop_criterion class.
                        Defaults to {'limit' : 50}, setting the iteration limit to 50.
        
                Returns:
                    None
        
                Why:
                    The stopping criterion is essential for controlling the duration of the evolutionary
                    search. It prevents the algorithm from running indefinitely and allows for balancing
                    exploration of the search space with computational cost. The default `IterationLimit`
                    ensures that the search terminates after a predefined number of generations, providing
                    a basic mechanism for managing the search process.
        """
        super().__init__()
        self._stop_criterion = stop_criterion(**sc_init_kwargs)
        self.run_performed = False
    
    def run(self, initial_population: Iterable, EA_kwargs: dict, stop_criterion_params: dict = {}):
        """
        Evolves the initial population to discover a differential equation.
        
                This method iteratively refines a population of candidate equations
                using an evolutionary algorithm until a stopping criterion is met.
                The evolutionary process involves traversing a sequence of linked
                blocks that perform operations such as selection, mutation, and
                crossover on the population. The goal is to find an equation that
                accurately describes the underlying dynamics of the data.
        
                Args:
                    initial_population (Iterable): The initial set of candidate equations.
                    EA_kwargs (dict): Keyword arguments to be passed to the evolutionary algorithm.
                    stop_criterion_params (dict): Parameters for the stopping criterion.
        
                Returns:
                    None: The method modifies the internal state of the `EvolutionaryStrategy` object.
        """
        self._stop_criterion.reset(**stop_criterion_params)
        population = initial_population
        while not self._stop_criterion.check():
            self.linked_blocks.traversal(population, EA_kwargs)
            population = self.linked_blocks.output
        self.run_performed = True

class Population(object):
    """
    Represents a collection of candidate solutions. Facilitates operations such as selection, reproduction, and evaluation, driving the evolutionary search process. Manages the population's state and provides methods for accessing and manipulating its members.
    
    
        Attributes:
            population (`list`): list of individs
            length (`int`): number of individs in population
    """

    def __init__(self, elements: list, sorting_method: Callable = simple_sorting):
        """
        Initializes the population with a set of individuals and a sorting method.
        
        The population is initialized with a list of individuals, and its length is recorded.
        A sorting method is assigned to maintain order based on individual fitness,
        facilitating the evolutionary process of selecting better-performing individuals.
        
        Args:
            elements (`list`): A list of individuals to form the initial population.
            sorting_method (`Callable`): A method used to sort the population based on fitness. Defaults to `simple_sorting`.
        
        Returns:
            None
        """
        self.population = elements
        self.length = len(elements)
        self._sorting_method = sorting_method
    
    def manual_reconst(self, attribute:str, value, except_attrs:dict):
        """
        Reconstructs a specific attribute of the population's search space manually.
        
                This method facilitates targeted modifications to the population, allowing for fine-grained control over the evolutionary process. By directly setting the value of an attribute, such as 'vals', the search can be guided towards promising regions of the solution space or to correct deviations introduced by evolutionary operators. This is particularly useful for injecting domain knowledge or recovering from unfavorable population states. Currently, only the 'vals' attribute is supported.
                
                Args:
                    attribute: The attribute to reconstruct. Currently, only 'vals' is supported.
                    value: The new value for the specified attribute.
                    except_attrs: A dictionary of attributes to exclude during reconstruction.
        
                Returns:
                    None.
        
                Raises:
                    ValueError: If the specified attribute is not supported.
        """
        from epde.loader import attrs_from_dict, get_typespec_attrs      
        supported_attrs = ['vals']
        if attribute not in supported_attrs:
            raise ValueError(f'Attribute {attribute} is not supported by manual_reconst method.')
    
    def sort(self):
        """
        Sorts the population based on the defined sorting method to prioritize candidates that better represent the underlying dynamics of the system. This is crucial for the evolutionary process, as it allows the algorithm to focus on more promising equation structures.
        
                Args:
                    self: The Population object.
        
                Returns:
                    A new list containing the sorted population of candidates. The original population remains unchanged.
        """
        return self._sorting_method(self.population)
    
    def sorted(self):
        """
        Sorts the population based on the fitness of individuals, arranging them in ascending order of error. Operates in "in-place" mode, updating the population directly. This ensures that the evolutionary process prioritizes the selection of better-performing equation candidates for subsequent generations.
        
                Args:
                    self: The Population instance.
        
                Returns:
                    None. The population is sorted in place.
        """
        self.population = self.sort() 

    def update(self, point):
        """
        Adds a new candidate solution to the population.
        
        This expands the search space by incorporating a new potential solution
        into the existing set of candidates, allowing the evolutionary algorithm
        to explore a wider range of possibilities when searching for the
        optimal equation.
        
        Args:
            point: The candidate solution (equation representation) to add to the population.
        
        Returns:
            None
        """
        self.population.append(point)
        
    def delete_point(self, point):
        """
        Deletes a specified candidate solution from the population. This is necessary to maintain diversity and improve the overall quality of the population during the evolutionary process of discovering differential equations. By removing less promising solutions, the algorithm can focus on exploring more promising areas of the search space.
        
                Args:
                    point: The candidate solution to be removed from the population.
        
                Returns:
                    None
        """
        self.population = [solution for solution in self.population if solution != point]

    def get_stats(self):
        """
        Calculates the objective function values for each individual in the population.
        
                This is a crucial step in evaluating the fitness of candidate equation structures
                within the evolutionary process. By calculating these values, the algorithm
                can assess how well each equation represents the underlying dynamics of the
                system being modeled.
        
                Args:
                    self: The Population instance.
        
                Returns:
                    np.ndarray: A NumPy array containing the objective function values for each
                        individual in the population.
        """
        return np.array([element.obj_fun for element in self.population])

    def __setitem__(self, key, value):
        """
        Sets an individual within the population.
        
                This method allows you to directly modify or add an individual (equation) to the population.
                This is useful for injecting specific equations or modifying existing ones during the evolutionary process.
        
                Args:
                    key: The index or identifier of the individual in the population.
                    value: The individual (equation) to be stored in the population.
        
                Returns:
                    None
        """
        self.population[key] = value

    def __getitem__(self, key):
        """
        Retrieves an individual from the population.
        
                This allows accessing and utilizing specific candidate solutions
                within the evolutionary process.
        
                Args:
                    key: The index of the individual to retrieve.
        
                Returns:
                    The individual at the given index in the population.
        """
        return self.population[key]
        
    def __iter__(self):
        """
        Returns an iterator for the population.
        
                This allows to traverse individuals in the population, enabling the evolutionary process to evaluate and evolve the population towards better solutions.
        
                Returns:
                    PopulationIterator: An iterator object for traversing the individuals in the population.
        """
        return PopulationIterator(self)
    
    
class PopulationIterator(object):
    """
    An iterator for traversing a population.
    
        Class Methods:
        - __init__:
    """

    def __init__(self, population):
        """
        Initializes the iterator with a population.
        
        This iterator is designed to traverse a population, providing access to individual members during the evolutionary process. The index is initialized to 0 to start at the beginning of the population.
        
        Args:
            population: The initial population to iterate over.
        
        Returns:
            None
        """
        self._population = population
        self._idx = 0

    def __next__(self):
        """
        Retrieves the next individual from the population to be evaluated.
        
        This allows iterating through the population, providing individuals
        for fitness evaluation and subsequent evolutionary operations.
        
        Args:
            self: The PopulationIterator instance.
        
        Returns:
            The next individual from the population.
        
        Raises:
            StopIteration: If all individuals in the population have been processed.
        """
        if self._idx < len(self._population.population):
            res = self._population.population[self._idx]
            self._idx += 1
            return res
        else:
            raise StopIteration


class SimpleOptimizer(object):
    """
    
    """

    def __init__(self, population_instruct, pop_size, solution_params, sorting_method = simple_sorting, 
                 passed_population: Union[Population, List] = None): 
        """
        Initializes a new Population object for equation discovery.
        
                This constructor creates a population of candidate equation solutions. It either
                generates a new population from scratch using a `SystemsPopulationConstructor`
                or initializes it with a pre-existing `Population` object or a list of solutions.
                When creating a new population, the constructor ensures that each generated
                solution is unique within the population to promote diversity in the search space.
                This diversity is crucial for the evolutionary process to effectively explore
                potential equation structures.
        
                Args:
                    population_instruct: Instructions for the `SystemsPopulationConstructor`.
                        This argument is passed directly to the `SystemsPopulationConstructor`
                        to configure how individual equation structures are generated. For
                        example, it specifies the pool of potential equation terms and other
                        parameters influencing the structure of the generated equations.
                    pop_size: The desired size of the population, determining the number of
                        candidate equation solutions to be maintained.
                    solution_params: A dictionary of parameters passed to the
                        `SystemsPopulationConstructor.create()` method when generating new
                        solutions. These parameters can influence the specific form of the
                        generated equations. If None, an empty dictionary is used.
                    sorting_method: The method used to sort the population based on the
                        performance of the candidate equations. Defaults to `simple_sorting`.
                    passed_population: An optional `Population` object or a list of solutions
                        to initialize the population with. If None, a new population is
                        created. If a list, it's used as the initial population. If a
                        `Population` object, it's used directly as the population.
        
                Raises:
                    TypeError: If `passed_population` is not None, a list, or a `Population`
                        object.
                    RuntimeError: If the constructor fails to create a unique solution
                        after many attempts, indicating potential issues with the solution
                        generation process or parameter settings.
                    AssertionError: If `solution_params` is not None or a dictionary.
        
                Returns:
                    None
        
                Fields:
                    population (Population): The `Population` object representing the
                        collection of candidate equation solutions. It is initialized either
                        with newly generated solutions or with the `passed_population` if
                        provided.
        """
        soluton_creation_attempts_softmax = 10
        soluton_creation_attempts_hardmax = 100

        pop_constructor = SystemsPopulationConstructor(**population_instruct)
        
        assert type(solution_params) == type(None) or type(solution_params) == dict, 'The solution parameters, passed into population constructor must be in dictionary'
        if (passed_population is None) or isinstance(passed_population, list):
            initial_population = [] if passed_population is None else passed_population

            for _ in range(pop_size):
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

        else:
            if not isinstance(passed_population, Population):
                raise TypeError(f'Incorrect type of the population passed. Expected Population object, instead got \
                                 {type(passed_population)}')
            self.population = passed_population

    def set_strategy(self, strategy_director):
        """
        Sets the optimization strategy.
        
        This method configures the optimization strategy based on the provided
        strategy director. It utilizes the director's builder to assemble
        the strategy and then assigns the resulting processor to the object's
        strategy attribute. This allows the optimizer to adapt its behavior
        based on the specific problem and data characteristics, enabling
        the discovery of governing equations.
        
        Args:
            strategy_director: The director responsible for building the strategy.
        
        Returns:
            None. This method does not return any value.
        """
        builder = strategy_director.builder
        builder.assemble(True)
        self.strategy = builder.processer
        
    def optimize(self, EA_kwargs: dict = {},  epochs: int = None):
        """
        Optimizes the population by iteratively refining equation candidates using an evolutionary strategy.
        
                This method drives the search for the best differential equation model by
                evolving a population of candidate equations. It leverages the configured
                evolutionary strategy to explore the search space and improve the
                population's fitness based on how well the equations fit the data. The global history is reset to ensure a fresh optimization run.
        
                Args:
                    EA_kwargs: Keyword arguments to be passed to the evolutionary algorithm's run method,
                               allowing for customization of the evolutionary process.
                    epochs: The maximum number of epochs to run the optimization for. If None, a default limit of 50 is used.
        
                Returns:
                    None
        """
        scp = {'limit' : epochs} if epochs is not None else {'limit' : 50}
        global_var.reset_hist()
        
        self.strategy.run(initial_population = self.population, EA_kwargs = EA_kwargs, 
                          stop_criterion_params = scp)