#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 12:54:02 2022

@author: maslyaev
"""

from functools import partial, singledispatch

from epde.operators.operator_mappers import map_operator_between_levels
from epde.operators.selections import MOEADDSelection, MOEADDSelectionConstrained
from epde.operators.elitism import NDLElitism
from epde.operators.mutations import ,  # TODO: write pregen operator chains
from epde.operators.crossovers import (
                                       ParetoLevelCrossover, 
                                       SystemCrossover, 
                                       ParamsCrossover, 
                                       EquationCrossover,
                                       EquationExchangeCrossoverб
                                       )  # TODO: write pregen operator chains
from epde.operators.fitness import L2Fitness
from epde.operators.right_part_selection import PoplevelRightPartSelector

from epde.moeadd.moeadd_strategy_elems import SectorProcesserBuilder, MOEADDSectorProcesser

# @singledispatch
# def add_param_to_operator(operator, target_dict, label, base_val):
#     operator.param[label] = target_dict[label] if label in target_dict.keys() else base_val

# @add_param_to_operator.register
# def add_param_to_operator(operator, target_dict, label : str, base_val):
#     operator.param[label] = target_dict[label] if label in target_dict.keys() else base_val
    
# @add_param_to_operator.register
# def add_param_to_operator(operator, target_dict, label : list, base_val : list):
#     for idx, lbl in enumerate(label):
#         operator.param[lbl] = target_dict[lbl] if lbl in target_dict.keys() else base_val[idx]



def form_basic_crossover(**kwargs):
    add_kwarg_to_operator = partial(func = detect_in_dict, target_dict = kwargs)

    param_crossover = Param_crossover(['proportion'])
    add_kwarg_to_operator(param_crossover, {'proportion' : 0.4})
    term_crossover = Term_crossover(['crossover_probability'])
    add_kwarg_to_operator(term_crossover, {'crossover_probability' : 0.3})        
    eq_crossover = Equation_crossover()
    
    crossover = PopLevel_crossover()
    

def form_basic_mutation(**kwargs):
    pass


class OptimizationPatternDirector(object):
    def __init__(self):
        self._builder = None

    @property
    def builder(self):
        return self._builder

    @builder.setter
    def builder(self, sector_processer_builder : SectorProcesserBuilder):
        self._builder = sector_processer_builder

    def use_unconstrained_optimization(self, **kwargs):
        add_kwarg_to_operator = partial(func = detect_in_dict, target_dict = kwargs)
        
        neighborhood_selector = SimpleNeighborSelector(['number_of_neighbors'])
        add_kwarg_to_operator(neighborhood_selector, {'number_of_neighbors' : 4})

        selection = MOEADDSelection(['delta', 'parents_fraction'])
        add_kwarg_to_operator(selection, {'delta' : 0.9, 'parents_fraction' : 4})
        selection.suboperators = {'neighborhood_selector' : neighborhood_selector}
        
        elilism = NDLElitism(['pareto_levels_excluded'])
        add_kwarg_to_operator(elilism, {'pareto_levels_excluded' : 1})
        
        param_crossover = Param_crossover(['proportion'])
        add_kwarg_to_operator(param_crossover, {'proportion' : 0.4})
        term_crossover = Term_crossover(['crossover_probability'])
        add_kwarg_to_operator(term_crossover, {'crossover_probability' : 0.3})        
        eq_crossover = Equation_crossover()
        
        crossover = PopLevel_crossover()

        pareto_level_mutation = OperatorMapper(operator_to_map = EquationMutation, objective_tag = 'pareto level level')

        self._builder.
    
    def use_constrained_optimization(self):
        pass
    