#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun 15 20:51:55 2021

@author: mike_ubuntu
"""

from abc import ABC, abstractproperty

class StopCondition(ABC):
    '''
    Abstract class for generalized stop condition for the evolutionary algorithm.
    
        Methods:    
            reset():
                Reset the object. Mainly useful, if some sort of counter is implemented inside the instance. Returns None.
            check()
    '''

    def __init__(self, estrategy):
        """
        Initializes the stop condition with a specified evaluation strategy.
        
        The evaluation strategy determines how the evolutionary process is terminated, 
        ensuring the search for differential equations concludes based on predefined criteria.
        
        Args:
            estrategy: An object defining the strategy for evaluating when to stop the evolutionary process.
        
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
        - __init__:
    """

    def __init__(self, limit = 100):
        """
        Initializes the iteration limit with a specified upper bound and a check counter.
        
        This setup is crucial for controlling the evolutionary search process, preventing it from running indefinitely and ensuring efficient exploration of the solution space. By limiting the number of iterations or checks, the algorithm can focus on promising equation structures and avoid unnecessary computations.
        
        Args:
            limit (int, optional): The maximum number of iterations allowed. Defaults to 100.
        
        Returns:
            None
        """
        self.limit = limit + 1
        self.checks = 0 # shortcut^ checking a number of calls instead of the estrategy attr of some sort
    
    def reset(self, **kwargs):
        """
        Resets the iteration counter and optionally adjusts the maximum iteration limit.
        
                This method prepares the iteration limit for a new search by resetting the internal counter
                that tracks the number of iterations performed. It also allows for updating the maximum
                number of iterations permitted, which is useful for controlling the computational cost
                of the search process. This ensures that the equation discovery process can be restarted
                or reconfigured with potentially different constraints on the search effort.
        
                Args:
                    **kwargs: Keyword arguments.  If 'limit' is provided, the maximum iteration limit is updated.
        
                Returns:
                    None
        """
        self.checks = 0
        self.limit = kwargs.get('limit', self.limit)
    
    def check(self):
        """
        Checks if the iteration limit has been reached.
        
        This method is called to ensure that the search process does not run indefinitely.
        It increments the internal counter and returns `True` if the maximum number of iterations
        has been exceeded, signaling that the search should be terminated or adjusted.
        
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
    
        This class adapts the fitness function during evolution to maintain a stable selection pressure.
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
    
