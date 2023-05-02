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
    Function, devoted to finding the domain, defined by **weights**, to which the 
    **solutions** belongs. The belonging is determined by the acute angle between solution and 
    the weight vector, defining the domain.

    Args:
        solution (`np.ndarray|src.moeadd.moeadd_solution_template.MOEADDSolution`): The candidate solution, for which we are determining the domain, or its objective 
            function values, stored in np.ndarray.
        weights (`np.ndarray`): Numpy ndarray, containing weights from the moeadd optimizer. 

    Returns:
        idx (`int`): Index of the domain (i.e. index of corresponing weight vector), to which the solution belongs.
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
        self.reset()

    def reset(self):
        self._counter = 0

    def incr(self):
        self._counter += 1

    def __call__(self):
        return self._counter



class MOEADDSolution(ABC):
    """
    Abstract superclass of the moeadd solution. *__hash__* method must be declared in the subclasses. 
    Overloaded *__eq__* method of MOEADDSolution uses strict equatlity between self.vals attributes,
    therefore, can not be used with the real-valued strings.

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
        Args:
            x : arbitrary object, 
                An arbitrary object, representing the solution gene. For example, 
                it can be a string of floating-point values, implemented as np.ndarray
            obj_funs (`list of functions`): Objective functions, that would be optimized by the 
                evolutionary algorithm.
        """
        self.vals = x
        self.obj_funs = obj_funs
        self.precomputed_value = False
        self.precomputed_domain = False
        self.crossover_selected_times = CrossoverSelectionCounter()

    @property
    def obj_fun(self):
        if self.precomputed_value:
            return self._obj_fun
        else:
            self._obj_fun = np.fromiter(map(lambda obj_fun: obj_fun(self.vals), self.obj_funs), dtype=float)
            self.precomputed_value = True
            return self._obj_fun

    def get_domain(self, weights):
        """
        Method that regulates the execution of the function finding the definition area once

        Args:
            weights (`np.ndarray`): Numpy ndarray, containing weights from the moeadd optimizer

        Returns:
            domains (`int`): Index of the domain, to that the solution belongs.
        """
        if self.precomputed_domain:
            # print(self, 'DOMAIN IS:', self._domain)
            return self._domain
        else:
            self._domain = get_domain_idx(self, weights)
            self.precomputed_domain = True
            return self._domain

    def set_domain(self, idx):
        self.precomputed_domain = True
        self._domain = idx

    def __eq__(self, other):
        if isinstance(other, type(self)):
            return self.vals == other.vals
        else:
            return NotImplemented

    def __call__(self):
        return self.obj_fun

    @abstractmethod
    def __hash__(self):
        raise NotImplementedError(
            
            'The hash needs to be defined in the subclass')

    def incr_counter(self, incr_val: int = 1):
        """
        Increase counter of allowed participations in the crossover process for and individual.

        Args:
            incr_val (`int`): Value, for which the counter is increased.
                Optional, the default is 1.

        """
        for i in range(incr_val):
            self.crossover_selected_times.incr()

    def crossover_times(self):
        return self.crossover_selected_times()

    def reset_counter(self):
        self.crossover_selected_times.reset()
