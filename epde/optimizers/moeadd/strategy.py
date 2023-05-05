#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 12:54:02 2022

@author: maslyaev
"""

import numpy as np
from functools import partial

from epde.operators.utils.operator_mappers import map_operator_between_levels
from epde.operators.utils.template import add_base_param_to_operator

from epde.operators.multiobjective.selections import MOEADDSelection
from epde.operators.multiobjective.variation import get_basic_variation
from epde.operators.common.fitness import L2Fitness
from epde.operators.common.right_part_selection import RandomRHPSelector
from epde.operators.multiobjective.moeadd_specific import get_pareto_levels_updater, SimpleNeighborSelector, get_initial_sorter
from epde.operators.common.sparsity import LASSOSparsity
from epde.operators.common.coeff_calculation import LinRegBasedCoeffsEquation

from epde.optimizers.builder import add_sequential_operators, OptimizationPatternDirector, StrategyBuilder
from epde.optimizers.moeadd.strategy_elems import MOEADDSectorProcesser

class MOEADDDirector(OptimizationPatternDirector):
    """
    Class for creating strategy builder of multicriterian optimization
    """
# class MOEADDDirector(OptimizationPatternDirector):
    def use_baseline(self, variation_params : dict = {}, mutation_params : dict = {}, sorter_params : dict = {},
                    pareto_combiner_params : dict = {}, pareto_updater_params : dict = {}, **kwargs):
        add_kwarg_to_operator = partial(add_base_param_to_operator, target_dict = kwargs)

        neighborhood_selector = SimpleNeighborSelector(['number_of_neighbors'])
        add_kwarg_to_operator(operator = neighborhood_selector)

        selection = MOEADDSelection(['delta', 'parents_fraction'])
        add_kwarg_to_operator(operator = selection)
        selection.set_suboperators({'neighborhood_selector' : neighborhood_selector})

        variation = get_basic_variation(variation_params)

        right_part_selector = RandomRHPSelector()
        
        eq_fitness = L2Fitness(['penalty_coeff'])
        add_kwarg_to_operator(operator = eq_fitness)
        
        sparsity = LASSOSparsity()
        coeff_calc = LinRegBasedCoeffsEquation()

        eq_fitness.set_suboperators({'sparsity' : sparsity, 'coeff_calc' : coeff_calc})
        fitness_cond = lambda x: not getattr(x, 'fitness_calculated')
        sys_fitness = map_operator_between_levels(eq_fitness, 'gene level', 'chromosome level', fitness_cond)

        rps_cond = lambda x: any([not elem_eq.right_part_selected for elem_eq in x.vals])
        sys_rps = map_operator_between_levels(right_part_selector, 'gene level', 'chromosome level', rps_cond)

        # Separate mutation from population updater for better customization.
        initial_sorter = get_initial_sorter(right_part_selector = sys_rps, chromosome_fitness = sys_fitness, 
                                            sorter_params = sorter_params)
        population_updater = get_pareto_levels_updater(right_part_selector = sys_rps, chromosome_fitness = sys_fitness,
                                                       constrained = False, mutation_params = mutation_params, 
                                                       pl_updater_params = pareto_updater_params, 
                                                       combiner_params = pareto_combiner_params)

        self.builder = add_sequential_operators(self.builder, [('initial_sorter', initial_sorter),
                                                               # ('pareto_updater_initial', population_updater),
                                                               ('selection', selection),
                                                               ('variation', variation),
                                                               ('pareto_updater_compl', population_updater)])
    
    def use_constrained_eq_search(self):
        raise NotImplementedError('No constraints have been implemented yest')
