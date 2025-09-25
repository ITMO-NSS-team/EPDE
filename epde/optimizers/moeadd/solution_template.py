#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 11:56:57 2022

@author: maslyaev
"""

import numpy as np
from abc import ABC, abstractmethod

from epde.optimizers.moeadd.supplementary import acute_angle


def get_domain_idx(solution, weights) -> int:
    """
    Finds the domain index for a given solution based on its objective function values and a set of weight vectors.
    
    The domain is determined by identifying the weight vector that forms the smallest acute angle with the solution's objective function vector. This assignment is crucial for effectively distributing solutions across different subproblems during the evolutionary optimization process, ensuring diversity and convergence towards the Pareto front.
    
    Args:
        solution (`np.ndarray|src.moeadd.moeadd_solution_template.MOEADDSolution`): The candidate solution, either as a NumPy array of objective function values or a `MOEADDSolution` object containing the objective function values.
        weights (`np.ndarray`): A NumPy array containing the weight vectors that define the different domains.
    
    Returns:
        `int`: The index of the weight vector (domain) to which the solution belongs, corresponding to the smallest acute angle.
    """

    if type(solution) == np.ndarray:
        return np.fromiter(map(lambda x: acute_angle(x, solution), weights), dtype=float).argmin()
    elif type(solution.obj_fun) == np.ndarray:
        return np.fromiter(map(lambda x: acute_angle(x, solution.obj_fun), weights), dtype=float).argmin()
    else:
        raise ValueError(
            'Can not detect the vector of objective function for solution')


class CrossoverSelectionCounter(object):
    """
    Counter of individs participating in the crossover
    """

    def __init__(self):
        """
        Initializes a new instance of the CrossoverSelectionCounter class.
        
        This class is designed to track and manage crossover selections during the evolutionary process of equation discovery.
        It initializes the counter by calling the reset method to ensure a clean state for each run or evaluation.
        
        Args:
            None
        
        Returns:
            None
        """
        self.reset()

    def reset(self):
        """
        Resets the selection counter.
        
        This method resets the internal counter `_counter` to 0. It is used to ensure that each generation starts with a fresh count of crossover selections, preventing bias towards earlier individuals in the population.
        
        Args:
            self: The instance of the class.
        
        Returns:
            None.
        """
        self._counter = 0

    def incr(self):
        """
        Increments the internal counter, tracking the number of crossover selection operations performed. This is crucial for monitoring the evolutionary algorithm's progress and ensuring sufficient exploration of the search space.
        
                Args:
                    self: The instance of the CrossoverSelectionCounter class.
        
                Returns:
                    None. This method modifies the internal state of the object by incrementing the counter.
        """
        self._counter += 1

    def __call__(self):
        """
        Return the number of crossover selections performed.
        
        This counter tracks the number of times crossover is applied during the evolutionary process, providing insights into the exploration of the search space. By monitoring this value, we can analyze the effectiveness of crossover in generating new candidate solutions.
        
        Args:
            self: The object instance.
        
        Returns:
            int: The current value of the crossover selection counter.
        """
        return self._counter



class MOEADDSolution(ABC):
    """
    Abstract base class for solutions within the multi-objective evolutionary algorithm. Subclasses must implement the *__hash__* method. The overloaded *__eq__* method relies on strict equality of the `self.vals` attribute, which may not be suitable for real-valued strings.
    
    
        Attributes:
            vals : arbitrary object
                An arbitrary object, representing the solution gene.
            obj_funs (`list of functions`): Objective functions, that would be optimized by the 
                evolutionary algorithm.
            precomputed_value (`bool`): Indicator, if the value of the objective functions is already calculated.
                Implemented to avoid redundant computations.
            precomputed_domain (`bool`): Indicator, if the solution has been already placed in a domain in objective function 
                space. Implemented to avoid redundant computations during the point placement.
            obj_fun (`np.array`): Property, that calculates/contains calculated value of objective functions.
            _domain (`int`): Index of the domain, to that the solution belongs.
    """

    def __init__(self, x, obj_funs):
        """
        Initializes a solution with its gene and objective functions.
        
        This method prepares a solution by storing its gene representation and the objective functions
        that will be used to evaluate its fitness. It also initializes flags to track precomputed values
        and crossover selection counts.
        
        Args:
            x : arbitrary object, 
                An arbitrary object, representing the solution gene. For example, 
                it can be a string of floating-point values, implemented as np.ndarray. This gene encodes a candidate equation.
            obj_funs (`list of functions`): Objective functions, that would be optimized by the 
                evolutionary algorithm. These functions quantify how well the candidate equation fits the data and satisfies other criteria.
        
        Returns:
            None
        """
        self.vals = x
        self.obj_funs = obj_funs
        self.precomputed_value = False
        self.precomputed_domain = False
        self.crossover_selected_times = CrossoverSelectionCounter()

    @property
    def obj_fun(self):
        """
        Calculates and returns the objective function value for this solution.
        
                This property provides access to the solution's objective function values.
                It efficiently computes these values only once and stores them for subsequent access.
                This ensures that the objective functions, which represent the solution's performance
                according to different criteria, are evaluated consistently and without redundant computation.
        
                Args:
                    self: The object instance.
        
                Returns:
                    np.ndarray: The calculated objective function value.
        
                Class Fields Initialized:
                    _obj_fun (np.ndarray): Stores the calculated objective function value. Initialized only if `precomputed_value` is False.
                    precomputed_value (bool): A flag indicating whether the objective function value has been precomputed. Set to True after the objective function value is computed for the first time.
        """
        if self.precomputed_value:
            return self._obj_fun
        else:
            self._obj_fun = np.fromiter(map(lambda obj_fun: obj_fun(self.vals), self.obj_funs), dtype=float)
            self.precomputed_value = True
            return self._obj_fun

    def get_domain(self, weights):
        """
        Determines the domain to which the solution belongs based on provided weights.
        
        This method optimizes performance by caching the domain index after its initial computation. Subsequent calls will return the cached value, avoiding redundant calculations. This is crucial for efficient multi-objective optimization, where domain assignments are frequently needed.
        
        Args:
            weights (`np.ndarray`): A NumPy array containing weights from the MOEA/D optimizer, used to determine the solution's domain.
        
        Returns:
            `int`: The index of the domain to which the solution belongs.
        """
        if self.precomputed_domain:
            return self._domain
        else:
            self._domain = get_domain_idx(self, weights)
            self.precomputed_domain = True
            return self._domain

    def set_domain(self, idx):
        """
        Sets the domain for the solution.
        
                This method marks the domain as precomputed and stores its index. This is done to efficiently manage and access different problem domains during the evolutionary search process, where each solution needs to be evaluated within a specific domain. By precomputing and indexing domains, the search algorithm can quickly switch between them without redundant calculations.
        
                Args:
                    idx (int): The index representing the precomputed domain.
        
                Returns:
                    None
        """
        self.precomputed_domain = True
        self._domain = idx

    def __eq__(self, other):
        """
        Compares this solution to another solution for equality based on their variable values.
        
                This is important for comparing solutions within the evolutionary algorithm,
                allowing the algorithm to identify and eliminate duplicate solutions,
                maintaining diversity in the population.
        
                Args:
                    other: The object to compare with.
        
                Returns:
                    bool: True if the objects are equal, NotImplemented if the other object is not of the same type.
        """
        if isinstance(other, type(self)):
            return self.vals == other.vals
        else:
            return NotImplemented

    def __call__(self):
        """
        Evaluates the solution by returning its objective function value.
        
        This allows the solution to be easily used in optimization algorithms
        where the objective function value is required for comparison and selection.
        
        Args:
            self: The MOEADDSolution instance.
        
        Returns:
            The objective function value associated with this solution.
        """
        return self.obj_fun

    @abstractmethod
    def __hash__(self):
        """
        Raises a NotImplementedError, enforcing implementation in subclasses.
        
        This abstract method ensures that concrete solution classes define their
        own hashing mechanism. This is crucial for utilizing solution objects
        in hash-based data structures, which is essential for efficient
        multi-objective optimization algorithms that rely on comparing and
        grouping solutions based on their properties.
        
        Args:
            self: The instance of the class.
        
        Returns:
            None.
        
        Raises:
            NotImplementedError: Always raised, prompting implementation in subclasses.
        """
        raise NotImplementedError(
            
            'The hash needs to be defined in the subclass')

    def incr_counter(self, incr_val: int = 1):
        """
        Increase the counter of crossover participations.
        
        This counter tracks how often an individual has been selected for crossover,
        influencing its likelihood of being chosen again and promoting diversity
        in the evolutionary process.
        
        Args:
            incr_val (int): The amount to increment the counter by. Defaults to 1.
        
        Returns:
            None
        """
        for i in range(incr_val):
            self.crossover_selected_times.incr()

    def crossover_times(self):
        """
        Returns the times when crossover operations were applied during the evolutionary process. These times indicate when genetic material was exchanged between solutions to explore the search space.
        
                Args:
                    None
        
                Returns:
                    list: A list of crossover times, representing the iterations at which crossover was performed. These times help track the evolution of solutions and the impact of crossover on the search process.
        """
        return self.crossover_selected_times()

    def reset_counter(self):
        """
        Resets the crossover selection counter.
        
        This method resets the internal counter that tracks how many times this
        crossover operator has been selected during the evolutionary process.
        This is important for maintaining a fair comparison of different crossover
        operators and ensuring diversity in the search for optimal equation structures.
        
        Args:
            self: The instance of the class.
        
        Returns:
            None.
        """
        self.crossover_selected_times.reset()
