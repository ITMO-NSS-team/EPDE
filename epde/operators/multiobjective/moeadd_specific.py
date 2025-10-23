#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 19:08:51 2022

@author: maslyaev
"""
import copy
import numpy as np
import time
from typing import Union, Tuple
from functools import reduce, partial

from epde.optimizers.moeadd.moeadd import ParetoLevels, ObjFunNormalizer
from epde.operators.utils.template import CompoundOperator, add_base_param_to_operator
from epde.operators.multiobjective.mutations import get_basic_mutation

from epde.structure.main_structures import SoEq
from copy import deepcopy


def penalty_based_intersection(sol_obj, weight, ideal_obj, 
                               penalty_factor = 1., obj_normalizer: ObjFunNormalizer = None) -> float:
    '''
    Calculation of the penalty pased intersection, that is minimized for the solutions inside the 
    domain, specified by **weight** vector. The calculations are held, according to the following formulas:
        
    .. math:: g^{pbi}(\mathbf{x}|\mathbf{w}, \mathbf{z^{*}}) = d_1 + \Theta d_2 \longrightarrow min
        
    subject to :math:`\mathbf{x} \in \Omega`

    where: 
        
    .. math::        
        d_1 = ||(\mathbf{f}(\mathbf{x}) - \mathbf{z^{*}})^{t}\mathbf{w}|| (||\mathbf{w}||)^{-1}

        d_2 = || \mathbf{f}(\mathbf{x}) - (\mathbf(z^{*}) + d_1 \mathbf{w} (||\mathbf{w}||)^{-1})||

    Arguments:
    ----------
    
    sol_obj : object of subclass of ``src.moeadd.moeadd_solution_template.MOEADDSolution``
        The solution, for which the penalty based intersection is calculated. In the equations above,
        it denotes :math:`\mathbf{x}`, with the :math:`\mathbf{F}(\mathbf{x})` representing the
        objective function values.
    
    weight : np.array
        Values of the weight vector, specific to the domain, in which the solution is located.
        Represents the :math:`\mathbf{w}` in the equations above.
    
    ideal_obj : `np.array`
        The value of best achievable objective functions values; denoted as 
        :math:`\mathbf{z^{*}} = (z^{*}_1, z^{*}_2, \; ... \;, z^{*}_m)`.
    
    penalty_factor : float, optional, default 1.
        The penalty parameter, represents :math:`\Theta` in the equations.

    obj_normalizer : ObjFunNormalizer obj., optional, defaut None.
        Normalizer for solution objective functions.
    
    '''
    # print(f'Objective before normalization: {sol_obj.obj_fun} for normalizer {obj_normalizer}')
    solution_objective = sol_obj.obj_fun if obj_normalizer is None else obj_normalizer(sol_obj.obj_fun)
    # print(f'Objective after expected normalization: {solution_objective}')
    
    d_1 = np.dot((solution_objective - ideal_obj), weight) / np.linalg.norm(weight)
    d_2 = np.linalg.norm(solution_objective - (ideal_obj + d_1 * weight/np.linalg.norm(weight)))
    return d_1 + penalty_factor * d_2


def population_to_sectors(population, weights):
    '''
    
    The distribution of the solutions into the domains, defined by weights vectors.
    
    Parameters:
    -----------
    
    population : list
        List, containing the candidate solutions for the evolutionary algorithm. Elements shall
        belong to the case-specific subclass of ``src.moeadd.moeadd_solution_template.MOEADDSolution``.
        
    weights : np.ndarray
        Numpy ndarray of weight vectors; first dimension - weight index, second dimension - 
        weight value in the objective function space.
        
    Returns:
    ---------
    
    population_divided : list
        List of candidate solutions, belonging to the weight domain. The outer index of the list - 
        the weight vector index, inner - the index of a particular candidate solution inside the domain.

        
    '''
    solution_selection = lambda weight_idx: [solution for solution in population 
                                             if solution.get_domain(weights) == weight_idx]
    return list(map(solution_selection, np.arange(len(weights))))    


def locate_pareto_worst(levels: ParetoLevels, weights: np.ndarray, best_obj: np.ndarray, penalty_factor: float = 1.):
    '''
    
    Function, dedicated to the selection of the worst solution on the Pareto levels.
    
    Arguments:
    ----------
    
    levels : pareto_levels obj
        The levels, on which the worst candidate solution is detected.
    
    weights : np.ndarray
        The weight vectors of the moeadd optimizer.
        
    best_obj : np.array
        Best achievable values of the objective functions.
    
    penalty_factor : float, optional, default 1.
        The penalty parameter, used during penalty based intersection value calculation.        
    
    '''
    domain_solutions = population_to_sectors(levels.population, weights)
    most_crowded_count = max([len(domain) for domain in domain_solutions]); crowded_domains = [domain_idx for domain_idx in np.arange(len(weights)) if 
                                                                           len(domain_solutions[domain_idx]) == most_crowded_count]
    if len(crowded_domains) == 1:
        most_crowded_domain = crowded_domains[0]
    else:
        PBI = lambda domain_idx: sum([penalty_based_intersection(sol_obj, weights[domain_idx], best_obj, penalty_factor, levels.normalizer)
                                      for sol_obj in domain_solutions[domain_idx]])
        PBIS = np.fromiter(map(PBI, crowded_domains), dtype = float)
        most_crowded_domain = crowded_domains[np.argmax(PBIS)]
        
    worst_NDL_section = []
    domain_solution_NDL_idxs = np.empty(most_crowded_count)
    for solution_idx, solution in enumerate(domain_solutions[most_crowded_domain]):
        domain_solution_NDL_idxs[solution_idx] = [level_idx for level_idx in np.arange(len(levels.levels)) 
                                                    if any([np.allclose(solution.obj_fun, level_solution.obj_fun) for level_solution in levels.levels[level_idx]])][0]
        
    max_level = np.max(domain_solution_NDL_idxs)
    worst_NDL_section = [domain_solutions[most_crowded_domain][sol_idx] for sol_idx in np.arange(len(domain_solutions[most_crowded_domain])) 
                        if domain_solution_NDL_idxs[sol_idx] == max_level]
    PBIS = np.fromiter(map(lambda solution: penalty_based_intersection(solution, weights[most_crowded_domain], best_obj, penalty_factor, levels.normalizer),
                           worst_NDL_section), dtype = float)
    return worst_NDL_section[np.argmax(PBIS)]


class PopulationUpdater(CompoundOperator):
    key = 'PopulationUpdater'
    
    def apply(self, objective : Tuple[Union[SoEq, ParetoLevels]], arguments : dict):
        '''
        Update population to get the pareto-nondomiated levels with the worst element removed. 
        Here, "worst" means the solution with highest PBI value (penalty-based boundary intersection)
        '''
        assert isinstance(objective, tuple), f'Expected input of PopulationUpdater to be a Tuple of SoEq and ParetoLevels.\n'\
                                             f'Did not get even a Tuple, instead got {type(objective)}!'
        assert isinstance(objective[0], SoEq), f'Expected input of PopulationUpdater to be a Tuple of SoEq and ParetoLevels.\n'\
                                               f'Did not get a SoEq obj in the first position, instead got {type(objective[0])}!'        
        assert isinstance(objective[1], ParetoLevels), f'Expected input of PopulationUpdater to be a Tuple of SoEq and ParetoLevels.\n'\
                                                       f'Did not get even a ParetoLevels in the second position, '\
                                                       f'instead got {type(objective[1])}!.'

        self_args, subop_args = self.parse_suboperator_args(arguments = arguments)
        # print(f'PopulationUpdater.params is {self.params}')
        
        # TODO: Init normalizer here!
        # print('objective is ', objective)
        # objective[1].set_normalizer()

        objective[1].update(objective[0])  #levels_updated = ndl_update(offspring, levels)
        if len(objective[1].levels) == 1:
            worst_solution = locate_pareto_worst(objective[1], self_args['weights'], 
                                                 self_args['best_obj'], self.params['PBI_penalty'])
        else:
            last_level_by_domains = population_to_sectors(objective[1].levels[-1],
                                                          self_args['weights'])
            most_crowded_count = np.max([len(domain) for domain in last_level_by_domains])
            crowded_domains = [domain_idx for domain_idx in np.arange(len(self_args['weights']))
                               if len(last_level_by_domains[domain_idx]) == most_crowded_count]

            if len(crowded_domains) == 1:
                most_crowded_domain = crowded_domains[0]
            else:
                PBI = lambda domain_idx: np.sum([penalty_based_intersection(sol_obj, self_args['weights'][domain_idx],
                                                                            self_args['best_obj'],
                                                                            self.params['PBI_penalty'],
                                                                            objective[1].normalizer)
                                                 for sol_obj in last_level_by_domains[domain_idx]])
                PBIS = np.fromiter(map(PBI, crowded_domains), dtype = float)
                most_crowded_domain = crowded_domains[np.argmax(PBIS)]

            if len(last_level_by_domains[most_crowded_domain]) == 1:
                worst_solution = last_level_by_domains[most_crowded_domain][0]
            else:
                PBIS = np.fromiter(map(lambda solution: penalty_based_intersection(solution,
                                                                                   self_args['weights'][most_crowded_domain],
                                                                                   self_args['best_obj'], self.params['PBI_penalty'],
                                                                                   objective[1].normalizer),
                                           last_level_by_domains[most_crowded_domain]), dtype = float)
                worst_solution = last_level_by_domains[most_crowded_domain][np.argmax(PBIS)]
        
        objective[1].delete_point(worst_solution)
        
    @property
    def arguments(self):
        return set(['weights', 'best_obj'])        

    def use_default_tags(self):
        self._tags = {'pareto level update', 'custom level', 'no suboperators', 'inplace'}
        

class PopulationUpdaterConstrained(object):
    key = 'PopulationUpdaterConstrined'
    
    def __init__(self, param_keys : list = [], constraints : Union[list, tuple, set] = []):
        super().__init__(param_keys = param_keys)
        raise NotImplementedError('Constrained optimization has not been implemented yet.')
        self.constraints = constraints
        # TODO: add constraint setting for the constructor        
        
    def apply(self, objective : ParetoLevels, arguments : dict):
        '''
        Update population to get the pareto-nondomiated levels with the worst element removed. 
        Here, "worst" means the solution with highest PBI value (penalty-based boundary intersection). 
        Additionally, the constraint violations are considered in the selection of the 
        "worst" individual.
        '''
        self_args, subop_args = self.parse_suboperator_args(arguments = arguments)

        objective[1].update(objective[0])
        cv_values = np.empty(len(objective[1])) #self.suboperators['constraint_violation'].apply(objective[0])
        
        for idx, elem in enumerate(objective[1]):
            cv_values[idx] = np.sum([constraint(elem) for constraint in self.constraints])
        
        if sum(cv_values) == 0:
            if len(objective[1].levels) == 1:
                worst_solution = locate_pareto_worst(objective[1], self_args['weights'], self_args['best_obj'], 
                                                     self.params['PBI_penalty'])
            else:
                if objective[1].levels[len(objective[1].levels) - 1] == 1:
                    domain_solutions = population_to_sectors(objective[1].population, self_args['weights'])
                    reference_solution = objective[1].levels[len(objective[1].levels) - 1][0]
                    reference_solution_domain = [idx for idx in np.arange(domain_solutions) if reference_solution in domain_solutions[idx]]
                    if len(domain_solutions[reference_solution_domain] == 1):
                        worst_solution = locate_pareto_worst(objective[1].levels, self_args['weights'], 
                                                             self_args['best_obj'], self.params['PBI_penalty'])
                    else:
                        worst_solution = reference_solution
                else:
                    last_level_by_domains = population_to_sectors(objective[1].levels[len(objective[1].levels)-1], 
                                                                  self_args['weights'])
                    most_crowded_count = np.max([len(domain) for domain in last_level_by_domains]); 
                    crowded_domains = [domain_idx for domain_idx in np.arange(len(self_args['weights'])) 
                                       if len(last_level_by_domains[domain_idx]) == most_crowded_count]
    
                    if len(crowded_domains) == 1:
                        most_crowded_domain = crowded_domains[0]
                    else:
                        PBI = lambda domain_idx: np.sum([penalty_based_intersection(sol_obj, self_args['weights'][domain_idx], 
                                                                                    self_args['best_obj'], self.params['PBI_penalty'],
                                                                                    objective.normalizer) 
                                                            for sol_obj in last_level_by_domains[domain_idx]])
                        PBIS = np.fromiter(map(PBI, crowded_domains), dtype = float)
                        most_crowded_domain = crowded_domains[np.argmax(PBIS)]
                        
                    if len(last_level_by_domains[most_crowded_domain]) == 1:
                        worst_solution = locate_pareto_worst(objective[1], self_args['weights'], 
                                                             self_args['best_obj'], self.params['PBI_penalty'])
                    else:
                        PBIS = np.fromiter(map(lambda solution: population_to_sectors(solution, self_args['weights'][most_crowded_domain], 
                                                                                      self_args['best_obj'],
                                                                                      self.params['PBI_penalty']), 
                                               last_level_by_domains[most_crowded_domain]), dtype = float)
                        worst_solution = last_level_by_domains[most_crowded_domain][np.argmax(PBIS)]                    
        else:
            infeasible = [solution for solution, _ in sorted(list(zip(objective[1].population, cv_values)), key = lambda pair: pair[1])]
            infeasible.reverse()
            infeasible = infeasible[:np.nonzero(cv_values)[0].size]
            deleted = False
            domain_solutions = population_to_sectors(objective[1].population, self_args['weights'])
            
            for infeasable_element in infeasible:
                domain_idx = [domain_idx for domain_idx, domain in enumerate(domain_solutions) if infeasable_element in domain][0]
                if len(domain_solutions[domain_idx]) > 1:
                    deleted = True
                    worst_solution = infeasable_element
                    break
            if not deleted:
                worst_solution = infeasible[0]

        objective[1].delete_point(worst_solution)

    @property
    def arguments(self):
        return set(['weights', 'best_obj'])   

    def use_default_tags(self):
        self._tags = {'pareto level update', 'custom level', 'no suboperators', 'inplace'}


def use_item_if_no_default(key, arg : dict, replacement_arg : dict):
    if key in replacement_arg.keys():
        arg[key] = replacement_arg[key]
    return arg


def get_basic_populator_updater(params : dict = {}):
    add_kwarg_to_operator = partial(add_base_param_to_operator, target_dict = params)    
    
    pop_updater = PopulationUpdater()
    add_kwarg_to_operator(operator = pop_updater)    
    # pop_updater.params = params
    return pop_updater


def get_constrained_populator_updater(params : dict = {}, constraints : list = []):
    add_kwarg_to_operator = partial(add_base_param_to_operator, target_dict = params)
    
    pop_updater = PopulationUpdaterConstrained(constraints = constraints)
    add_kwarg_to_operator(operator = pop_updater)        
    # pop_updater.params = params
    return pop_updater


class SimpleNeighborSelector(CompoundOperator):
    key = 'SortingBasedNeighborSelector'

    def apply(self, objective : list, arguments : dict):
        '''
            Simple selector of neighboring weight vectors: takes n-closest (*n = number_of_neighbors*)ones to the 
            processed one. Defined to be used inside the moeadd algorithm.
        
            Arguments:
            ----------
            
            sorted_neighbors : list
                proximity list of neighboring vectors, ranged in the ascending order of the angles between vectors.
                
            number_of_neighbors : int
                numbers of vectors to be considered as the adjacent ones
                
            Returns:
            ---------
            
            sorted_neighbors[:number_of_neighbors] : list
                self evident slice of proximity list
        '''
        self_args, subop_args = self.parse_suboperator_args(arguments = arguments)        
        return objective[:self.params['number_of_neighbors']]
    
    def use_default_tags(self):
        self._tags = {'neighbor selector', 'custom level', 'no suboperators', 'inplace'}    


def best_obj_values(levels : ParetoLevels):
    vals = np.array([solution.obj_fun for solution in levels])
    return np.sort(vals, axis = 0)[(0, -1), ...]


class OffspringUpdater(CompoundOperator):
    key = 'ParetoLevelUpdater'

    def apply(self, objective: ParetoLevels, arguments: dict):
        self_args, subop_args = self.parse_suboperator_args(arguments=arguments)

        while objective.unplaced_candidates:
            offspring = objective.unplaced_candidates.pop()
            attempt = 1
            mutation_attempt_limit = self.params['mutation_attempt_limit']
            offspring_attempt_limit = self.params['offspring_attempt_limit']
            temp_offspring = deepcopy(offspring)
            replaced = 0
            while True:
                temp_offspring = self.suboperators['chromosome_mutation'].apply(objective=temp_offspring,
                                                                                arguments=subop_args['chromosome_mutation'])
                self.suboperators['right_part_selector'].apply(objective=temp_offspring,
                                                               arguments=subop_args['right_part_selector'])
                temp_offspring.reset_state()
                system = temp_offspring.described_variables
                if system not in objective.history:
                    self.suboperators['chromosome_fitness'].apply(objective=temp_offspring,
                                                                  arguments=subop_args['chromosome_fitness'])
                    self.suboperators['pareto_level_updater'].apply(objective=(temp_offspring, objective),
                                                                    arguments=subop_args['pareto_level_updater'])
                    objective.history.add(system)
                    print(temp_offspring.obj_fun)
                    break
                elif replaced == offspring_attempt_limit:
                    print("Could not generate unique offspring")
                    break
                elif attempt == mutation_attempt_limit:
                    temp_offspring = deepcopy(offspring)
                    replaced += 1
                    attempt = 0
                attempt += 1
        return objective
    
def get_pareto_levels_updater(right_part_selector : CompoundOperator, chromosome_fitness : CompoundOperator,
                              mutation : CompoundOperator = None, constrained : bool = False, 
                              mutation_params : dict = {}, pl_updater_params : dict = {}, 
                              combiner_params : dict = {}):
    add_kwarg_to_updater = partial(add_base_param_to_operator, target_dict = combiner_params)
    updater = OffspringUpdater()
    add_kwarg_to_updater(operator = updater)
    
    if mutation is None:
        mutation = get_basic_mutation(mutation_params)
    pl_updater = get_basic_populator_updater(pl_updater_params)
    updater.set_suboperators(operators = {'chromosome_mutation' : mutation,
                                          'pareto_level_updater' : pl_updater,
                                          'right_part_selector' : right_part_selector,
                                          'chromosome_fitness' : chromosome_fitness})
    return updater

class InitialParetoLevelSorting(CompoundOperator):
    key = 'InitialParetoLevelSorting'  
    
    def apply(self, objective : ParetoLevels, arguments : dict):
        '''
        Initial sorting of the candidates in pareto levels. 

        Parameters
        ----------
        objective : ParetoLevels
            DESCRIPTION.
        arguments : dict
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        self_args, subop_args = self.parse_suboperator_args(arguments = arguments)

        if len(objective.population) == 0:
            for idx, candidate in enumerate(objective.unplaced_candidates):
                self.suboperators['right_part_selector'].apply(objective = candidate,
                                                                arguments = subop_args['right_part_selector'])
                system = candidate.described_variables
                while system in objective.history:
                    candidate.create()
                    self.suboperators['right_part_selector'].apply(objective=candidate,
                                                                   arguments=subop_args['right_part_selector'])
                    system = candidate.described_variables
                self.suboperators['chromosome_fitness'].apply(objective=candidate,
                                                              arguments=subop_args['chromosome_fitness'])
                objective.history.add(system)
                print(candidate.obj_fun)
            objective.initial_placing()
        
            # TODO: consider carefully, where normalizer init shall be held. If here, only the initial values are employed
        objective.set_normalizer()

        return objective
    
def get_initial_sorter(right_part_selector : CompoundOperator, 
                       chromosome_fitness : CompoundOperator, 
                       sorter_params : dict = {}):
    add_kwarg_to_updater = partial(add_base_param_to_operator, target_dict = sorter_params)
    sorter = InitialParetoLevelSorting()
    add_kwarg_to_updater(operator = sorter)
    sorter.set_suboperators(operators = {'right_part_selector' : right_part_selector,
                                          'chromosome_fitness' : chromosome_fitness})
    return sorter
