#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 19:08:51 2022

@author: maslyaev
"""
import copy
import numpy as np
import time
from typing import Union
from functools import reduce, partial

from epde.optimizers.moeadd.moeadd import ParetoLevels
from epde.operators.utils.template import CompoundOperator, add_base_param_to_operator
from epde.operators.multiobjective.mutations import get_basic_mutation


def penalty_based_intersection(sol_obj, weight, ideal_obj, penalty_factor = 1.) -> float:
    """
    Calculates the Penalty-based Intersection (PBI) value for a given solution, considering its objective function values, a weight vector, and an ideal point.
    
        This function evaluates how well a solution aligns with a specific subregion of the objective space,
        guided by the weight vector. The PBI value combines the distance along the weight vector (d1) and
        the distance to the weight vector (d2), penalizing solutions that are far from the ideal point
        and the direction indicated by the weight vector. This metric is used to promote solutions that
        are both close to the ideal point and well-distributed across the objective space.
    
        The calculations are held, according to the following formulas:
            
        .. math:: g^{pbi}(\mathbf{x}|\mathbf{w}, \mathbf{z^{*}}) = d_1 + \Theta d_2 \longrightarrow min
            
        subject to :math:`\mathbf{x} \in \Omega`
    
        where: 
            
        .. math::        
            d_1 = ||(\mathbf{f}(\mathbf{x}) - \mathbf{z^{*}})^{t}\mathbf{w}|| (||\mathbf{w}||)^{-1}
    
            d_2 = || \mathbf{f}(\mathbf{x}) - (\mathbf(z^{*}) + d_1 \mathbf{w} (||\mathbf{w}||)^{-1})||
    
        Args:
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
        
        Returns:
        -------
        float
            The calculated penalty-based intersection value.
    """
    d_1 = np.dot((sol_obj.obj_fun - ideal_obj), weight) / np.linalg.norm(weight)
    d_2 = np.linalg.norm(sol_obj.obj_fun - (ideal_obj + d_1 * weight/np.linalg.norm(weight)))
    return d_1 + penalty_factor * d_2


def population_to_sectors(population, weights):
    """
    Distributes the population of solutions across different subdomains, each associated with a specific weight vector.
    
        This division is crucial for localized evolutionary search within each subdomain, 
        allowing the algorithm to focus on improving solutions that are most relevant to a particular 
        region of the objective space, guided by the weight vectors.
    
        Args:
            population (list): A list of candidate solutions. Each solution should be an instance of a subclass
                of `src.moeadd.moeadd_solution_template.MOEADDSolution` and must implement a `get_domain` method
                that determines its associated weight vector index.
            weights (np.ndarray): A NumPy array containing the weight vectors. The first dimension represents the
                weight vector index, and the second dimension represents the weight values in the objective function space.
    
        Returns:
            list: A list of lists, where each inner list contains the solutions belonging to a specific weight vector's
                domain. The outer index corresponds to the weight vector index, and the inner index corresponds to the
                index of a particular solution within that domain.
    """
    solution_selection = lambda weight_idx: [solution for solution in population 
                                             if solution.get_domain(weights) == weight_idx]
    return list(map(solution_selection, np.arange(len(weights))))    


def locate_pareto_worst(levels, weights, best_obj, penalty_factor = 1.):
    """
    Locates the worst-performing solution within the Pareto levels to maintain diversity and convergence in the evolutionary process.
    
        This function identifies the solution that contributes least to the overall Pareto front, 
        focusing on densely populated regions and higher non-dominated levels. By removing such solutions, 
        the algorithm encourages exploration of less crowded areas and promotes better convergence towards the true Pareto front.
    
        Args:
            levels (pareto_levels obj): The Pareto levels containing the population's non-dominated sorting.
            weights (np.ndarray): The weight vectors used in the MOEA/D algorithm to decompose the multi-objective problem.
            best_obj (np.array): The best-known objective function values for normalization purposes.
            penalty_factor (float, optional): The penalty parameter used in the penalty-based intersection calculation. Defaults to 1.0.
    
        Returns:
            Individual: The individual identified as the worst-performing solution on the Pareto levels.
    """
    domain_solutions = population_to_sectors(levels.population, weights)
    most_crowded_count = max([len(domain) for domain in domain_solutions]); crowded_domains = [domain_idx for domain_idx in np.arange(len(weights)) if 
                                                                           len(domain_solutions[domain_idx]) == most_crowded_count]
    if len(crowded_domains) == 1:
        most_crowded_domain = crowded_domains[0]
    else:
        PBI = lambda domain_idx: sum([penalty_based_intersection(sol_obj, weights[domain_idx], best_obj, penalty_factor) for sol_obj in domain_solutions[domain_idx]])
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
    PBIS = np.fromiter(map(lambda solution: penalty_based_intersection(solution, weights[most_crowded_domain], best_obj, penalty_factor), worst_NDL_section), dtype = float)
    return worst_NDL_section[np.argmax(PBIS)]


class PopulationUpdater(CompoundOperator):
    """
    Updates a population by removing the worst element based on a penalty-based boundary intersection (PBI) value and updating Pareto levels.
    
        Attributes:
            problem (Problem): The optimization problem being solved.
            ref_points (np.ndarray): The reference points used for PBI calculation.
            ideal_point (np.ndarray): The ideal point for normalization.
            utopian_point (np.ndarray): The utopian point for normalization.
    """

    key = 'PopulationUpdater'
    
    def apply(self, objective : ParetoLevels, arguments : dict):
        """
        Updates the Pareto levels by removing the solution with the highest PBI value from the worst level. This focuses the search on promising areas of the Pareto front, maintaining diversity while prioritizing solutions that effectively balance multiple objectives.
        
                Args:
                    objective (ParetoLevels): The current Pareto levels, containing both the population and its non-dominated sorting.
                    arguments (dict): A dictionary containing necessary parameters, including weights for PBI calculation and the best objective values.
        
                Returns:
                    None: The method modifies the `objective` in place, updating the Pareto levels.
        """
        self_args, subop_args = self.parse_suboperator_args(arguments = arguments)
        # print(f'PopulationUpdater.params is {self.params}')
        
        objective[1].update(objective[0])  #levels_updated = ndl_update(offspring, levels)
        if len(objective[1].levels) == 1:
            worst_solution = locate_pareto_worst(objective[1], self_args['weights'], 
                                                 self_args['best_obj'], self.params['PBI_penalty'])
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
                                                                                self_args['best_obj'], 
                                                                                self.params['PBI_penalty'])
                                                     for sol_obj in last_level_by_domains[domain_idx]])
                    PBIS = np.fromiter(map(PBI, crowded_domains), dtype = float)
                    most_crowded_domain = crowded_domains[np.argmax(PBIS)]
                    
                if len(last_level_by_domains[most_crowded_domain]) == 1:
                    worst_solution = locate_pareto_worst(objective[1], self_args['weights'], 
                                                         self_args['best_obj'], self.params['PBI_penalty'])
                else:
                    PBIS = np.fromiter(map(lambda solution: penalty_based_intersection(solution, 
                                                                                       self_args['weights'][most_crowded_domain],
                                                                                       self_args['best_obj'], self.params['PBI_penalty']),
                                               last_level_by_domains[most_crowded_domain]), dtype = float)
                    worst_solution = last_level_by_domains[most_crowded_domain][np.argmax(PBIS)]                    
        
        objective[1].delete_point(worst_solution)
        
    @property
    def arguments(self):
        """
        Returns the set of arguments used by the object. These arguments are necessary for updating the population based on the optimization process. Specifically, 'weights' are used to control the contribution of each objective, and 'best_obj' represents the best objective value achieved so far, guiding the search towards better solutions.
                
                Returns:
                    set: A set containing the strings 'weights' and 'best_obj',
                        representing the arguments used.
        """
        return set(['weights', 'best_obj'])        

    def use_default_tags(self):
        """
        Sets the tags to the default set.
        
        This method overwrites any existing tags with a predefined set of default tags. This ensures that the PopulationUpdater operates with a consistent and predefined set of characteristics, which is useful for maintaining a standardized approach to equation discovery. By using default tags, the system can reliably apply specific update strategies and constraints during the evolutionary process.
        
        Args:
            self: The object instance.
        
        Returns:
            None.
        """
        self._tags = {'pareto level update', 'custom level', 'no suboperators', 'inplace'}
        

class PopulationUpdaterConstrained(object):
    """
    Updates a population by removing the worst element based on PBI and constraint violations to maintain Pareto-nondominated levels.
    
          Class Methods:
            - apply: Update population to get the pareto-nondomiated levels with the worst element removed.
            - arguments: Returns the set of arguments used by the object.
            - use_default_tags: Uses default tags for the object.
    """

    key = 'PopulationUpdaterConstrined'
    
    def __init__(self, param_keys : list = [], constraints : Union[list, tuple, set] = []):
        """
        Initializes a ConstrainedOptimizer object.
        
        This class is intended to manage parameter updates within specified constraints, ensuring that the evolved solutions adhere to predefined boundaries.
        Currently, constrained optimization is not implemented.
        
        Args:
            param_keys (list): List of parameter keys to optimize.
            constraints (Union[list, tuple, set]): Constraints to apply during optimization.
        
        Raises:
            NotImplementedError: Always raised, as constrained optimization is not yet implemented.
        
        Returns:
            None.
        
        Fields:
            constraints (Union[list, tuple, set]): Constraints to apply during optimization.  Currently, constrained optimization is not implemented, so this field is not used.
        
        Why:
            This initializer is designed to eventually handle constraints during the evolutionary process, ensuring that the discovered equations and their parameters remain within physically or empirically relevant ranges.
        """
        super().__init__(param_keys = param_keys)
        raise NotImplementedError('Constrained optimization has not been implemented yet.')
        self.constraints = constraints
        # TODO: add constraint setting for the constructor        
        
    def apply(self, objective : ParetoLevels, arguments : dict):
        """
        Update the Pareto-nondominated levels by removing the "worst" solution from the population.
        
                The "worst" solution is determined based on its PBI (Penalty-based Boundary Intersection) value,
                prioritizing solutions with constraint violations. This process refines the Pareto front approximation
                by iteratively removing less desirable solutions, guiding the search towards better equation discovery.
        
                Args:
                    objective (ParetoLevels): The current Pareto-nondominated levels. The first element is assumed to be the initial population, the second - population with levels
                    arguments (dict): A dictionary containing necessary parameters, including weights, best objective values,
                        and PBI penalty.
        
                Returns:
                    None: The method modifies the `objective` in place by removing the worst solution.
        """
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
                                                                                    self_args['best_obj'], self.params['PBI_penalty']) 
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
        """
        Returns the set of arguments used by the object.
        
                This method specifies the arguments required for updating the population, 
                ensuring that the evolutionary process has the necessary information 
                (weights and best objective value) to guide the search for optimal solutions.
        
                Returns:
                    set: A set containing the strings 'weights' and 'best_obj',
                         representing the arguments used.
        """
        return set(['weights', 'best_obj'])   

    def use_default_tags(self):
        """
        Uses a predefined set of tags to characterize the update operation.
        
                This method assigns a default set of tags to the updater, providing a standardized way to identify its key characteristics and ensure consistent behavior within the evolutionary process. These tags help in categorizing and managing different update strategies.
        
                Args:
                    self: The PopulationUpdaterConstrained instance.
        
                Returns:
                    None.
        
                This method initializes the following object properties:
                  - _tags: A set containing the default tags: 'pareto level update', 'custom level', 'no suboperators', and 'inplace'.
        """
        self._tags = {'pareto level update', 'custom level', 'no suboperators', 'inplace'}


def use_item_if_no_default(key, arg : dict, replacement_arg : dict):
    """
    Adds a key-value pair from a replacement dictionary to the original dictionary if the key is missing.
    
    This ensures that essential parameters, potentially discovered during the equation search process, 
    are incorporated into the configuration, even if they weren't initially specified. This helps to refine 
    the equation discovery process by ensuring that important terms are not overlooked.
    
    Args:
        key: The key to check for in both dictionaries.
        arg (dict): The original dictionary to be updated.
        replacement_arg (dict): The dictionary providing replacement values.
    
    Returns:
        dict: The updated dictionary.
    """
    if key in replacement_arg.keys():
        arg[key] = replacement_arg[key]
    return arg


def get_basic_populator_updater(params : dict = {}):
    """
    Creates a `PopulationUpdater` instance and configures it with essential parameters.
    
        This function initializes a `PopulationUpdater` and enriches it with
        fundamental parameters, preparing it for evolutionary processes. This
        ensures that the population update mechanism is properly set up with
        necessary configurations before the evolutionary search begins.
    
        Args:
            params (dict, optional): A dictionary holding parameters to be incorporated
                into the `PopulationUpdater`. Defaults to an empty dictionary.
    
        Returns:
            PopulationUpdater: A configured `PopulationUpdater` object, ready for use
                in the evolutionary process.
    """
    add_kwarg_to_operator = partial(add_base_param_to_operator, target_dict = params)    
    
    pop_updater = PopulationUpdater()
    add_kwarg_to_operator(operator = pop_updater)    
    # pop_updater.params = params
    return pop_updater


def get_constrained_populator_updater(params : dict = {}, constraints : list = []):
    """
    Creates and returns a population updater with specified constraints.
    
        This method initializes a `PopulationUpdaterConstrained` object, incorporating
        user-defined constraints to guide the evolutionary process. It also integrates
        essential parameters, ensuring the population update adheres to the problem's
        specific requirements. This ensures that the search for differential equations
        respects predefined boundaries and conditions.
    
        Args:
            params (dict, optional): A dictionary of parameters to be added to the operator. Defaults to {}.
            constraints (list, optional): A list of constraints to be applied to the population updater. Defaults to [].
    
        Returns:
            PopulationUpdaterConstrained: An instance of the population updater, configured with the provided constraints and parameters.
    """
    add_kwarg_to_operator = partial(add_base_param_to_operator, target_dict = params)
    
    pop_updater = PopulationUpdaterConstrained(constraints = constraints)
    add_kwarg_to_operator(operator = pop_updater)        
    # pop_updater.params = params
    return pop_updater


class SimpleNeighborSelector(CompoundOperator):
    """
    A simple selector of neighboring weight vectors.
    
        This class selects the n-closest neighboring weight vectors to a processed one,
        where n is the number_of_neighbors. It is designed for use within the MOEA/D algorithm.
    
        Methods:
            - apply
    """

    key = 'SortingBasedNeighborSelector'

    def apply(self, objective : list, arguments : dict):
        """
        Selects a subset of the most relevant objective vectors based on proximity.
        
        This method is used to focus the evolutionary search on the most promising regions
        of the objective space, thereby improving the efficiency of the equation discovery
        process. By selecting only a few neighboring objectives, the algorithm can
        concentrate its computational resources on refining the equation models that
        best represent those objectives.
        
        Args:
            objective (list): A list of objective vectors, typically sorted by a proximity metric.
            arguments (dict): A dictionary containing additional arguments (not directly used in this method).
        
        Returns:
            list: A slice of the input `objective` list, containing the top `number_of_neighbors` vectors.
        """
        self_args, subop_args = self.parse_suboperator_args(arguments = arguments)        
        return objective[:self.params['number_of_neighbors']]
    
    def use_default_tags(self):
        """
        Resets the selector's tags to the default set.
        
                This ensures the selector operates with a standard configuration, 
                suitable for typical equation discovery tasks. The default tags 
                define common characteristics and constraints for neighbor selection.
        
                Args:
                    self: The SimpleNeighborSelector instance.
        
                Returns:
                    None.
        """
        self._tags = {'neighbor selector', 'custom level', 'no suboperators', 'inplace'}    


def best_obj_values(levels : ParetoLevels):
    """
    Returns the extreme objective function values from a set of Pareto-optimal solutions.
    
        This function identifies the minimum and maximum values for each objective
        across all solutions within the provided Pareto levels. This helps to
        understand the range of possible objective values achievable within the
        Pareto-optimal set, providing insights into trade-offs between objectives.
    
        Args:
            levels: A ParetoLevels object containing a set of Pareto-optimal solutions.
    
        Returns:
            np.ndarray: A NumPy array containing the minimum and maximum objective
                function values across all solutions in the Pareto levels. The array
                is sorted along axis 0, and the first and last elements are returned,
                representing the minimum and maximum values respectively.
    """
    vals = np.array([solution.obj_fun for solution in levels])
    return np.sort(vals, axis = 0)[(0, -1), ...]


class OffspringUpdater(CompoundOperator):
    """
    Updates the offspring population by applying a series of suboperators.
    
        This class orchestrates the process of improving a Pareto front by
        iteratively applying mutation, selection, and fitness evaluation
        suboperators to unplaced candidate solutions. It manages attempts to
        avoid duplicates and enforces attempt limits.
    """

    key = 'ParetoLevelUpdater'

    def apply(self, objective: ParetoLevels, arguments: dict):
        """
        Applies a series of suboperators to integrate unplaced candidates into the Pareto front.
        
                This method iteratively processes unplaced candidates, applying mutation, selection,
                and fitness evaluation suboperators to find suitable solutions. It manages attempts
                to avoid duplicates and enforces attempt limits to ensure diversity and prevent
                premature convergence during the equation discovery process. The goal is to refine
                the Pareto front by incorporating new candidate solutions that effectively model
                the underlying dynamics.
        
                Args:
                    objective: The ParetoLevels object to be improved, representing the current
                        set of candidate equations and their performance metrics.
                    arguments: A dictionary containing arguments for the suboperators, configuring
                        their behavior during the optimization process.
        
                Returns:
                    ParetoLevels: The modified ParetoLevels object after applying the suboperators,
                        containing the updated Pareto front with potentially new and improved
                        candidate equations.
        """
        self_args, subop_args = self.parse_suboperator_args(arguments=arguments)

        while objective.unplaced_candidates:
            offspring = objective.unplaced_candidates.pop()
            attempt = 1
            attempt_limit = self.params['attempt_limit']
            temp_offspring = self.suboperators['chromosome_mutation'].apply(objective=offspring,
                                                                            arguments=subop_args['chromosome_mutation'])
            replaced = False
            while True:
                self.suboperators['right_part_selector'].apply(objective=temp_offspring,
                                                               arguments=subop_args['right_part_selector'])
                self.suboperators['chromosome_fitness'].apply(objective=temp_offspring,
                                                              arguments=subop_args['chromosome_fitness'])

                if (all([not np.allclose(temp_offspring.obj_fun, solution.obj_fun) for solution in objective.population])
                        and tuple(temp_offspring.obj_fun) not in objective.history):
                        # and all([not np.allclose(temp_offspring.obj_fun, obj_fun) for obj_fun in objective.history]):

                    self.suboperators['pareto_level_updater'].apply(objective=(temp_offspring, objective),
                                                                    arguments=subop_args['pareto_level_updater'])
                    objective.history.add(tuple(temp_offspring.obj_fun))
                    break
                elif replaced >= attempt_limit:
                    print("Allowed replication")
                    self.suboperators['pareto_level_updater'].apply(objective=(temp_offspring, objective),
                                                                    arguments=subop_args['pareto_level_updater'])
                    break
                elif attempt >= attempt_limit:
                    temp_offspring.create()
                    replaced += 1
                    attempt = 1
                self.suboperators['chromosome_mutation'].apply(objective=temp_offspring,
                                                               arguments=subop_args['chromosome_mutation'])
                attempt += 1
        return objective
    
def get_pareto_levels_updater(right_part_selector : CompoundOperator, chromosome_fitness : CompoundOperator,
                              mutation : CompoundOperator = None, constrained : bool = False, 
                              mutation_params : dict = {}, pl_updater_params : dict = {}, 
                              combiner_params : dict = {}):
    """
    Creates and configures an offspring updater for Pareto levels.
    
        This method constructs an `OffspringUpdater` and configures it with
        sub-operators responsible for mutation, Pareto level updating, right part selection,
        and chromosome fitness evaluation. This ensures that the evolutionary process
        effectively explores the search space and identifies optimal equation structures
        by iteratively refining the population of candidate solutions. It also handles the case where a mutation
        operator is not provided, creating a default one.
    
        Args:
            right_part_selector: Operator for selecting the right part of the offspring.
            chromosome_fitness: Operator for evaluating chromosome fitness.
            mutation: Operator for chromosome mutation (optional). If None, a basic mutation is used.
            constrained: A boolean indicating whether the problem is constrained (not used in the provided code).
            mutation_params: Parameters for the mutation operator.
            pl_updater_params: Parameters for the Pareto level updater.
            combiner_params: Parameters for combining the offspring.
    
        Returns:
            OffspringUpdater: The configured offspring updater.
    """
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
    """
    Initial sorting of the candidates in pareto levels.
    
        Class Attributes:
            key
    
        Class Methods:
            - apply: '''
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
    """

    key = 'InitialParetoLevelSorting'  
    
    def apply(self, objective : ParetoLevels, arguments : dict):
        """
        Sorts the initial population into Pareto levels based on their objective values.
        
                This method is crucial for initiating the evolutionary process by 
                organizing the candidate solutions based on dominance relationships. 
                It ensures that the selection process favors non-dominated solutions, 
                driving the evolution towards better-performing equations.
        
                Args:
                    objective (ParetoLevels): The ParetoLevels object containing the population and related information.
                    arguments (dict): A dictionary containing arguments for the sub-operators.
        
                Returns:
                    ParetoLevels: The updated ParetoLevels object with the population sorted into initial Pareto levels.
        """
        self_args, subop_args = self.parse_suboperator_args(arguments = arguments)
        
        if len(objective.population) == 0:
            for idx, candidate in enumerate(objective.unplaced_candidates):
                # while True:
                    # temp_candidate = copy.deepcopy(candidate)
                self.suboperators['right_part_selector'].apply(objective = candidate,
                                                                arguments = subop_args['right_part_selector'])                
                    # print('Hah, got ya!')
                    # if all([temp_candidate != solution for solution in objective.unplaced_candidates[:idx] + 
                    #         objective.unplaced_candidates[idx+1:]]):
                    #     objective.unplaced_candidates[idx] = temp_candidate
                    #     break
                        
                self.suboperators['chromosome_fitness'].apply(objective = objective.unplaced_candidates[idx],
                                                              arguments = subop_args['chromosome_fitness'])
                objective.history.add(tuple(candidate.obj_fun))
            objective.initial_placing()
        return objective
    
def get_initial_sorter(right_part_selector : CompoundOperator, 
                       chromosome_fitness : CompoundOperator, 
                       sorter_params : dict = {}):
    """
    Creates and configures the initial Pareto level sorting operator for equation discovery.
    
        This method instantiates an `InitialParetoLevelSorting` operator, configures it with provided parameters,
        and sets its sub-operators for right part selection and chromosome fitness evaluation.
        This sorting is crucial in the evolutionary process to identify promising equation structures
        by evaluating their fitness and complexity.
    
        Args:
            right_part_selector: Operator for selecting the right part of the chromosome (equation).
            chromosome_fitness: Operator for evaluating chromosome fitness (equation fitness).
            sorter_params: Optional dictionary of parameters to be added to the sorter.
    
        Returns:
            InitialParetoLevelSorting: The configured initial Pareto level sorting operator.
    """
    add_kwarg_to_updater = partial(add_base_param_to_operator, target_dict = sorter_params)
    sorter = InitialParetoLevelSorting()
    add_kwarg_to_updater(operator = sorter)
    sorter.set_suboperators(operators = {'right_part_selector' : right_part_selector,
                                          'chromosome_fitness' : chromosome_fitness})
    return sorter
