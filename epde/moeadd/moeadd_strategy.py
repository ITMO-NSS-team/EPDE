#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 12:54:02 2022

@author: maslyaev
"""

import numpy as np
from functools import partial

from epde.operators.operator_mappers import map_operator_between_levels

from epde.operators.template import add_param_to_operator
from epde.operators.selections import MOEADDSelection
from epde.operators.variation import get_basic_variation
from epde.operators.fitness import L2Fitness
from epde.operators.right_part_selection import RandomRHPSelector
from epde.operators.moeadd_specific import get_pareto_levels_updater, SimpleNeighborSelector, get_initial_sorter
from epde.operators.sparsity import LASSOSparsity
from epde.operators.coeff_calculation import LinRegBasedCoeffsEquation

from epde.moeadd.moeadd_strategy_elems import SectorProcesserBuilder, MOEADDSectorProcesser


def add_sequential_operators(builder : SectorProcesserBuilder, operators : list):
    '''
    

    Parameters
    ----------
    builder : SectorProcesserBuilder,
        MOEADD evolutionary strategy builder (sector processer), which will contain added operators.
    operators : list,
        Operators, which will be added into the processer. The elements of the list shall be tuple in 
        format of (label, operator), where the label is str (e.g. 'selection'), while the operator is 
        an object of subclass of CompoundOperator.

    Returns
    -------
    builder : SectorProcesserBuilder
        Modified builder.

    '''
    builder.add_init_operator('initial')
    for idx, operator in enumerate(operators):
        builder.add_operator(operator[0], operator[1], terminal_operator = (idx == len(operators) - 1))

    builder.set_input_combinator()
    builder.link('initial', operators[0][0])
    for op_idx, _ in enumerate(operators[:-1]):
        builder.link(operators[op_idx][0], operators[op_idx + 1][0])

    builder.assemble()
    return builder


class OptimizationPatternDirector(object):
    def __init__(self):
        self._builder = None

    @property
    def builder(self):
        return self._builder

    @builder.setter
    def builder(self, sector_processer_builder : SectorProcesserBuilder):
        print(f'setting builder with {sector_processer_builder}')
        self._builder = sector_processer_builder

    def use_unconstrained_eq_search(self, variation_params : dict = {}, mutation_params : dict = {}, sorter_params : dict = {},
                                    pareto_combiner_params : dict = {}, pareto_updater_params : dict = {},
                                    **kwargs):
        add_kwarg_to_operator = partial(add_param_to_operator, target_dict = kwargs)

        neighborhood_selector = SimpleNeighborSelector(['number_of_neighbors'])
        add_kwarg_to_operator(operator = neighborhood_selector, labeled_base_val = {'number_of_neighbors' : 4})

        selection = MOEADDSelection(['delta', 'parents_fraction'])
        add_kwarg_to_operator(operator = selection, labeled_base_val = {'delta' : 0.9, 'parents_fraction' : 0.4})
        selection.set_suboperators({'neighborhood_selector' : neighborhood_selector})

        variation = get_basic_variation(variation_params)

        right_part_selector = RandomRHPSelector()
        
        eq_fitness = L2Fitness(['penalty_coeff'])
        add_kwarg_to_operator(operator = eq_fitness, labeled_base_val = {'penalty_coeff' : 0.2})
        
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