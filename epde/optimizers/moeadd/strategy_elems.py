#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 19:14:25 2022

@author: maslyaev
"""

# from epde.optimizers.blocks import EvolutionaryBlock, InputBlock
from epde.optimizers.strategy import Strategy # rename to strategy_elems
from epde.optimizers.builder import StrategyBuilder

class MOEADDSectorProcesser(Strategy):
    '''
    Specific implemtation of evolutionary strategy of MOEADD algorithm. Defines 
        processing of a population in respect to a weight vector.
    '''

    def run(self, population_subset, EA_kwargs : dict):
        """
        Runs the linked blocks traversal on a population subset.
                
                This method orchestrates the search for governing equations by traversing linked blocks of equation components within a subset of the population. It leverages evolutionary algorithm parameters to guide the search process.
        
                Args:
                    population_subset: The subset of the population to process, representing candidate equation structures.
                    EA_kwargs (dict): Keyword arguments for the evolutionary algorithm, controlling parameters like mutation rates and selection pressures.
        
                Returns:
                    The output from the linked blocks traversal, representing the discovered equation structures and their associated fitness values. This output is crucial for identifying the best candidate equations that fit the observed data.
        """
        self.linked_blocks.traversal(population_subset, EA_kwargs)
        return self.linked_blocks.output
