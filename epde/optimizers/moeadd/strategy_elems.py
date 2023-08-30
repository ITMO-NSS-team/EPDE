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
        self.linked_blocks.traversal(population_subset, EA_kwargs)
        return self.linked_blocks.output
