#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 19:14:25 2022

@author: maslyaev
"""

# from epde.optimizers.blocks import EvolutionaryBlock, InputBlock
from epde.optimizers.strategy import Strategy # rename to strategy_elems
from epde.optimizers.builder import OperatorBuilder

# class SectorProcesserBuilder(OperatorBuilder):
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
#         self._processer = MOEADDSectorProcesser() # stop_criterion, stop_criterion_kwargs
#         super().__init__()
    
#     @property
#     def processer(self):
#         return self._processer

class MOEADDSectorProcesser(Strategy):
    '''
    Specific implemtation of evolutionary strategy of MOEADD algorithm. Defines 
    processing of a population in respect to a weight vector.
    '''
    def run(self, population_subset, EA_kwargs : dict):
        self.check_integrity()
        self.linked_blocks.traversal(population_subset, EA_kwargs)
        return self.linked_blocks.output