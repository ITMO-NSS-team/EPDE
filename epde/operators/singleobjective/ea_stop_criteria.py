#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 20:51:55 2021

@author: mike_ubuntu
"""

from abc import ABC, abstractproperty


class StopCondition(ABC):
    """
    The `StopCondition` class provides a flexible mechanism for defining criteria to halt the equation discovery process. It allows users to specify conditions based on various metrics, such as the number of generations, the fitness score of the best equation, or a custom function. This enables fine-grained control over the search process and ensures that the algorithm terminates when satisfactory results are achieved or when further exploration is unlikely to yield significant improvements.
    
    
        Abstract class for generalized stop condition for the evolutionary algorithm.
    
        Methods:
        ----------
    
        reset():
            Reset the object. Mainly useful, if some sort of counter is implemented inside the instance. Returns None.
    
        check()
    
        '''
    """


    def __init__(self, estrategy):
        """
        Initializes the StopCondition object with an evaluation strategy.
        
        The evaluation strategy determines when the search for differential equations should terminate.
        This allows for flexible control over the trade-off between exploration and convergence
        during the equation discovery process.
        
        Args:
            estrategy: An object that implements a stopping condition strategy.
        
        Returns:
            None
        """
        pass

    def reset(self):
        """
        Resets the internal state of the stop condition.
        
        This ensures that the search for differential equations starts fresh, 
        allowing the evolutionary algorithm to explore the solution space 
        without being biased by previous iterations.
        
        Args:
            self: The StopCondition instance.
        
        Returns:
            None.
        """
        pass

    def check(self):
        """
        Checks if the stopping criteria for the evolutionary algorithm have been met.
        
        This method determines whether the search for differential equations should be terminated based on predefined conditions.
        
        Args:
            self: The object instance.
        
        Returns:
            bool: True if the stopping criteria are met, False otherwise.
        
        Why:
            This check is crucial for controlling the duration and efficiency of the equation discovery process. It prevents the algorithm from running indefinitely or prematurely terminating before finding a satisfactory solution.
        """
        pass


class IterationLimit(StopCondition):
    """
    A class to limit the number of iterations in a process.
    
        Class Methods:
        - __init__
        - reset
        - check
    """

    def __init__(self, limit=100):
        """
        Initializes the iteration limit for the evolutionary process.
        
        This method sets the maximum number of iterations allowed during the equation discovery process and initializes a counter to track the progress. Limiting iterations prevents the algorithm from running indefinitely and ensures timely convergence.
        
        Args:
            limit (int, optional): The maximum number of iterations. Defaults to 100.
        
        Returns:
            None
        """
        self.limit = limit + 1
        self.checks = 0

    def reset(self):
        """
        Resets the check counter.
        
        This method resets the internal counter (`checks`) that tracks the number of checks
        performed during the equation discovery process. This is useful for restarting or
        managing the search within a defined computational budget.
        
        Args:
            self: The object instance.
        
        Returns:
            None.
        """
        self.checks = 0

    def check(self):
        """
        Checks if the iteration limit has been reached.
        
        This method is crucial for controlling the evolutionary search process, ensuring it terminates after a specified number of iterations to prevent excessive computation. It increments the internal counter and determines if the search has exceeded its allocated budget.
        
        Args:
            self: The object instance.
        
        Returns:
            bool: True if the iteration limit has been reached, False otherwise.
        """
        self.checks += 1
        return self.checks > self.limit


class FitnessStabilized(StopCondition):
    """
    A fitness-stabilized evolutionary strategy.
    
        This class adapts the fitness function during evolution to stabilize
        selection pressure and improve search efficiency.
    
        Attributes:
            estrategy: The underlying evolutionary strategy.
    """

    def __init__(self, estrategy):
        """
        Initializes the FitnessStabilized object.
        
        This method must be overridden by subclasses to provide a concrete implementation for fitness evaluation within a stabilized evolutionary process.  The initialization should set up the fitness evaluation strategy.
        
        Args:
            estrategy: The fitness evaluation strategy to be used.
        
        Raises:
            NotImplementedError: Always raised, indicating that the method must be implemented in a subclass.
        
        Returns:
            None.
        
        Why: This abstract initialization ensures that each specific fitness stabilization technique defines its own setup, tailoring the fitness evaluation process to its particular needs within the broader equation discovery framework.
        """
        raise NotImplementedError
