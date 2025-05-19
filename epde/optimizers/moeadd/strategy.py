#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 12:54:02 2022

@author: maslyaev
"""

import numpy as np
from functools import partial

from epde.operators.utils.operator_mappers import map_operator_between_levels, OperatorCondition
from epde.operators.utils.template import add_base_param_to_operator

from epde.operators.multiobjective.selections import MOEADDSelection
from epde.operators.multiobjective.variation import get_basic_variation
from epde.operators.common.fitness import L2Fitness, L2LRFitness, SolverBasedFitness, PIC
from epde.operators.common.right_part_selection import RandomRHPSelector, EqRightPartSelector
from epde.operators.multiobjective.moeadd_specific import get_pareto_levels_updater, SimpleNeighborSelector, get_initial_sorter
from epde.operators.common.sparsity import LASSOSparsity
from epde.operators.common.coeff_calculation import LinRegBasedCoeffsEquation

from epde.optimizers.builder import add_sequential_operators, OptimizationPatternDirector, StrategyBuilder
from epde.optimizers.moeadd.strategy_elems import MOEADDSectorProcesser

class MOEADDDirector(OptimizationPatternDirector):
    """
    Class for creating strategy builder of multicriterian optimization
    """
    def use_baseline(self, use_solver: bool = False, use_pic: bool = True, variation_params : dict = {}, mutation_params : dict = {},  
                     sorter_params : dict = {}, pareto_combiner_params : dict = {},
                     pareto_updater_params : dict = {}, **kwargs):
        add_kwarg_to_operator = partial(add_base_param_to_operator, target_dict = kwargs)

        neighborhood_selector = SimpleNeighborSelector(['number_of_neighbors'])
        add_kwarg_to_operator(operator = neighborhood_selector)

        selection = MOEADDSelection(['delta', 'parents_fraction'])
        add_kwarg_to_operator(operator = selection)
        selection.set_suboperators({'neighborhood_selector' : neighborhood_selector})

        variation = get_basic_variation(variation_params)

        # right_part_selector = RandomRHPSelector()
        right_part_selector = EqRightPartSelector()
        
        sparsity = LASSOSparsity()
        coeff_calc = LinRegBasedCoeffsEquation()

        if use_solver:
            fitness = PIC(['penalty_coeff']) if use_pic else SolverBasedFitness(['penalty_coeff'])
            # self.best_objectives = [0., 1., 0.] if use_pic else [0., 1.]

            sparsity_c = map_operator_between_levels(sparsity, 'gene level', 'chromosome level')
            coeff_calc_c = map_operator_between_levels(coeff_calc, 'gene level', 'chromosome level')
        else:
            sparsity_c = sparsity; coeff_calc_c = coeff_calc 
            fitness = L2LRFitness(['penalty_coeff'])
        add_kwarg_to_operator(operator = fitness)

        fitness.set_suboperators({'sparsity' : sparsity_c, 'coeff_calc' : coeff_calc_c})
        fitness_cond = lambda x: not getattr(x, 'fitness_calculated')
        if use_solver:
            fitness_lightweight = L2LRFitness(['penalty_coeff'])
            fitness_lightweight.set_suboperators({'sparsity' : sparsity, 'coeff_calc' : coeff_calc})
            right_part_selector.set_suboperators({'fitness_calculation' : fitness_lightweight})

            fitness = OperatorCondition(fitness, fitness_cond)
        else:
            right_part_selector.set_suboperators({'fitness_calculation' : fitness})
            fitness = map_operator_between_levels(fitness, 'gene level', 'chromosome level',
                                                  objective_condition=fitness_cond)




        rps_cond = lambda x: any([not elem_eq.right_part_selected for elem_eq in x.vals])
        sys_rps = map_operator_between_levels(right_part_selector, 'gene level', 'chromosome level', 
                                              objective_condition=rps_cond)

        # Separate mutation from population updater for better customization.
        initial_sorter = get_initial_sorter(right_part_selector = sys_rps, chromosome_fitness = fitness, 
                                            sorter_params = sorter_params)
        population_updater = get_pareto_levels_updater(right_part_selector = sys_rps, chromosome_fitness = fitness,
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
