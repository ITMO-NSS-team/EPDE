#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 12:54:02 2022

@author: maslyaev
"""

from functools import partial, singledispatch

from epde.operators.operator_mappers import map_operator_between_levels

from epde.operators.template import add_param_to_operator
from epde.operators.selections import MOEADDSelection
from epde.operators.variation import get_basic_variation
from epde.operators.fitness import L2Fitness
from epde.operators.right_part_selection import PoplevelRightPartSelector
from epde.operators.moeadd_specific import get_pareto_levels_updater, SimpleNeighborSelector
from epde.operators.sparsity import LASSOSparsity
from epde.operators.coeff_calculation import LinRegBasedCoeffsEquation

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

    def use_unconstrained_eq_search(self, variation_params : dict = {},  mutation_params : dict = {},
                                    pareto_combiner_params : dict = {}, pareto_updater_params : dict = {},
                                    **kwargs):
        add_kwarg_to_operator = partial(func = add_param_to_operator, target_dict = kwargs)

        neighborhood_selector = SimpleNeighborSelector(['number_of_neighbors'])
        add_kwarg_to_operator(neighborhood_selector, {'number_of_neighbors' : 4})

        selection = MOEADDSelection(['delta', 'parents_fraction'])
        add_kwarg_to_operator(selection, {'delta' : 0.9, 'parents_fraction' : 4})
        selection.suboperators = {'neighborhood_selector' : neighborhood_selector}

        variation = get_basic_variation(variation_params)

        right_part_selector = PoplevelRightPartSelector()
        
        eq_fitness = L2Fitness(['penalty_coeff'])
        add_kwarg_to_operator(eq_fitness, {'penalty_coeff' : 0.2})
        
        sparsity = LASSOSparsity()
        coeff_calc = LinRegBasedCoeffsEquation()

        eq_fitness.suboperators = {'sparsity' : sparsity, 'coeff_calc' : coeff_calc}
        fitness_cond = lambda x: getattr(x, 'fitness_calculated')
        sys_fitness = map_operator_between_levels(eq_fitness, 'gene level', 'chromosome level', fitness_cond)
        
        # Separate mutation from population updater for better customization.
        population_updater = get_pareto_levels_updater(right_part_selector, sys_fitness,
                                                       constrained = False, mutation_params = mutation_params, 
                                                       pareto_updater_params = pareto_updater_params, 
                                                       combiner_params = pareto_combiner_params)

        self._builder.add_init_operator('initial')

        self._builder.add_operator('selection', selection)
        self._builder.add_operator('variation', variation)
        self._builder.add_operator('pareto_updater', population_updater, terminal_operator = True)

        self._builder.set_input_combinator()

        self._builder.link('initial', 'selection')
        self._builder.link('rps1', 'selection')
        self._builder.link('selection', 'variation')
        self._builder.link('variation', 'pareto_updater')

        self._builder.assemble()

    
    def use_constrained_eq_search(self):
        raise NotImplementedError('No constraints have been implemented yest')
    