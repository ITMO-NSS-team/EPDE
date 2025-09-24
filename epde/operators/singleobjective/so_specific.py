import numpy as np

import epde.globals as global_var
from epde.operators.utils.template import CompoundOperator
from epde.optimizers.single_criterion.optimizer import Population

class SizeRestriction(CompoundOperator):
    """
    Restricts the size of an objective to a specified maximum.
    """

    key = 'SizeRestriction'

    def apply(self, objective: Population, arguments: dict):
        """
        Applies a selection operator to the population.
        
                Sorts the population based on fitness and truncates it to the desired length, ensuring the population size remains consistent.
                Also, it records the fitness of the best individual in the history to track the evolutionary progress. This selection process refines the population, favoring individuals that better represent the underlying dynamics of the system being modeled.
        
                Args:
                  objective: The population to apply the selection to.
                  arguments: A dictionary of arguments for the operator.
        
                Returns:
                  The modified objective (population).
        """
        self_args, subop_args = self.parse_suboperator_args(arguments = arguments)          
        objective.population = objective.sort()[:objective.length]
        global_var.history.add([eq.fitness_value  for eq in objective.population[0]][0])        
        return objective

    def use_default_tags(self):
        """
        Applies a predefined set of tags to the object, ensuring consistency in categorization.
        
        This method overwrites any existing tags with a default set, providing a standardized
        classification for objects within the system. This is important for maintaining a
        uniform structure for objects that are subject to size restrictions, population levels,
        lack suboperators, and are considered standard.
        
        Args:
            self: The object instance.
        
        Returns:
            None.
        
        Class Fields:
            _tags (set): A set containing the default tags: 'size restriction', 'population level', 'no suboperators', and 'standard'.
        """
        self._tags = {'size restriction', 'population level', 'no suboperators', 'standard'}    

class FractionElitism(CompoundOperator):
    """
    Implements a fraction-based elitism strategy for evolutionary algorithms.
    
        This class ensures that a specified fraction of the best individuals
        in a population are preserved in the next generation.
    
        Attributes:
            fraction (float): The fraction of the population to be considered elite.
    """

    key = 'FractionElitism'

    def apply(self, objective: Population, arguments: dict):
        """
        Applies the elite strategy to the population, preserving the best solution found so far.
        
                This method sorts the population based on fitness and designates the
                best individual as the elite, making it immutable to ensure that the best-performing equation
                is retained throughout the evolutionary process. All other individuals
                are marked as non-elite. This ensures that the evolutionary process does not lose the best equation found.
        
                Args:
                  objective: The Population object to which the elite strategy is applied.
                  arguments: A dictionary containing arguments for the sub-operators.
        
                Returns:
                  Population: The modified Population object with the elite strategy applied.
        """
        self_args, subop_args = self.parse_suboperator_args(arguments = arguments)    

        objective.population = objective.sort()
        for idx, elem in enumerate(objective.population):
            if idx == 0:
                setattr(elem, 'elite', 'immutable')
            else:
                setattr(elem, 'elite', 'non-elite')
                
        return objective
    
    @property
    def operator_tags(self):
        """
        Returns a set of operator tags.
        
                This method returns a set of predefined tags associated with the operator.
                These tags provide metadata about the operator's functionality and characteristics,
                allowing the evolutionary algorithm to effectively explore the search space of possible equations
                by categorizing and selecting operators based on their properties.
        
                Args:
                    self: The instance of the class.
        
                Returns:
                    set: A set containing string literals representing operator tags.
        """
        return {'elitism', 'population level', 'auxilary', 'no suboperators', 'standard'}        