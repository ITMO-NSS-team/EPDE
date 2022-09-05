#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 12:54:02 2022

@author: maslyaev
"""

from functools import partial, singledispatch

from epde.operators.operator_mappers import map_operator_between_levels
from epde.operators.selections import MOEADDSelection, MOEADDSelectionConstrained
from epde.operators.variation import get_basic_variation
from epde.operators.fitness import L2Fitness
from epde.operators.right_part_selection import PoplevelRightPartSelector
from epde.operators.moeadd_specific import get_pareto_levels_updater

from epde.moeadd.moeadd_strategy_elems import SectorProcesserBuilder, MOEADDSectorProcesser


class OptimizationPatternDirector(object):
    def __init__(self):
        self._builder = None

    @property
    def builder(self):
        return self._builder

    @builder.setter
    def builder(self, sector_processer_builder : SectorProcesserBuilder):
        self._builder = sector_processer_builder

    def use_unconstrained_eq_search(self, variation_params : dict, pareto_updater_params : dict, 
                                    **kwargs):
        add_kwarg_to_operator = partial(func = detect_in_dict, target_dict = kwargs)

        neighborhood_selector = SimpleNeighborSelector(['number_of_neighbors'])
        add_kwarg_to_operator(neighborhood_selector, {'number_of_neighbors' : 4})

        selection = MOEADDSelection(['delta', 'parents_fraction'])
        add_kwarg_to_operator(selection, {'delta' : 0.9, 'parents_fraction' : 4})
        selection.suboperators = {'neighborhood_selector' : neighborhood_selector}

        variation = get_basic_variation(variation_params)
        population_updater = get_pareto_levels_updater(pareto_updater_params)

        right_part_selector = PoplevelRightPartSelector()
        fitness = 

        self._builder.
    
    def use_constrained_eq_search(self):
        raise NotImplementedError('No constraints have been implemented yest')
    