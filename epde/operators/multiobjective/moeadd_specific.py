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
import epde.globals as global_var
from epde.operators.multiobjective.mutations import get_basic_mutation

from epde.structure.main_structures import SoEq
from copy import deepcopy

from epde import _loop_stats


def penalty_based_intersection(sol_obj, weight, ideal_obj,
                               penalty_factor=1., obj_normalizer=None) -> float:
    '''
    Calculation of the penalty based intersection in an expanded 2N-D space.
    This ensures that individual equations within the system maintain the
    trade-off defined by the weight vector.
    '''
    solution_objective = sol_obj.obj_fun if obj_normalizer is None else obj_normalizer(sol_obj.obj_fun)

    weight_arr = np.asarray(weight)
    ideal_obj_arr = np.asarray(ideal_obj)
    n_eqs = len(sol_obj.vals)
    n_obj = solution_objective.shape[0]
    if weight_arr.size * n_eqs == n_obj:
        # MOEA/D weight is per objective TYPE -- expand to per-equation space.
        weight_full = np.repeat(weight_arr, n_eqs)
        ideal_obj_full = np.repeat(ideal_obj_arr, n_eqs)
    else:
        # Weight already lives in the full objective space (legacy
        # objective list of per-equation partials).
        weight_full = weight_arr
        ideal_obj_full = ideal_obj_arr

    weight_norm = np.linalg.norm(weight_full)

    d_1 = np.dot((solution_objective - ideal_obj_full), weight_full) / weight_norm
    d_2 = np.linalg.norm(solution_objective - (ideal_obj_full + d_1 * (weight_full / weight_norm)))

    return d_1 + penalty_factor * d_2


def population_to_sectors(population, weights):
    '''
    The distribution of the solutions into the domains, defined by weights vectors.
    '''
    solution_selection = lambda weight_idx: [solution for solution in population
                                             if solution.get_domain(weights) == weight_idx]
    return list(map(solution_selection, np.arange(len(weights))))


def decomposition_based_worst(solutions: list, weights: np.ndarray, best_obj: np.ndarray,
                              penalty_factor: float = 1., obj_normalizer=None):
    '''
    Decomposition-based worst-solution finder. Returns argmax PBI over the
    most-crowded subregion within ``solutions``; ties on niche count are
    broken by sum-PBI per paper Eq. (7).

    Used as a helper for two branches of Algorithm 4 in the MOEA/DD paper
    (Li, Deb, Zhang, Kwong, 2015):
      * ``PopulationUpdater`` case ``l = 1`` -- called with the full
        population; equivalent to ``LOCATE_WORST`` (Algorithm 5) since
        every solution lives on the single front.
      * ``PopulationUpdater`` case ``l > 1, |F_l| > 1, |Phi^h| > 1`` --
        called with ONLY the last front ``F_l``. This is a deliberate
        EPDE deviation from Algorithm 4 line 18, which says
        ``argmax_{x in Phi^h} g^pbi(x|w^h, z*)`` over the FULL subregion
        (potentially containing elite F_1..F_{l-1} solutions). Restricting
        to F_l preserves convergence elites at the cost of some selection
        pressure within Phi^h; see audit notes for the rationale.
    '''
    domain_solutions = population_to_sectors(solutions, weights)
    most_crowded_count = max(len(domain) for domain in domain_solutions)
    crowded_domains = [idx for idx, domain in enumerate(domain_solutions)
                       if len(domain) == most_crowded_count]

    if len(crowded_domains) == 1:
        most_crowded_domain = crowded_domains[0]
    else:
        # Tie-breaking via largest sum of PBI in the crowded subregions
        PBIS = [sum(penalty_based_intersection(sol, weights[domain_idx], best_obj, penalty_factor, obj_normalizer)
                    for sol in domain_solutions[domain_idx])
                for domain_idx in crowded_domains]
        most_crowded_domain = crowded_domains[np.argmax(PBIS)]

    candidates = domain_solutions[most_crowded_domain]

    # Find the solution with the largest individual PBI in the selected subregion
    PBIS_candidates = [
        penalty_based_intersection(s, weights[most_crowded_domain], best_obj, penalty_factor, obj_normalizer)
        for s in candidates]

    return candidates[np.argmax(PBIS_candidates)]


def locate_pareto_worst(levels, weights: np.ndarray, best_obj: np.ndarray, penalty_factor: float = 1.):
    '''
    Function dedicated to the selection of the worst solution on the Pareto levels.
    '''
    domain_solutions = population_to_sectors(levels.population, weights)
    most_crowded_count = max(len(domain) for domain in domain_solutions)

    crowded_domains = [domain_idx for domain_idx, domain in enumerate(domain_solutions)
                       if len(domain) == most_crowded_count]

    if len(crowded_domains) == 1:
        most_crowded_domain = crowded_domains[0]
    else:
        PBIS = [
            sum(penalty_based_intersection(sol_obj, weights[domain_idx], best_obj, penalty_factor, levels.normalizer)
                for sol_obj in domain_solutions[domain_idx])
            for domain_idx in crowded_domains]
        most_crowded_domain = crowded_domains[np.argmax(PBIS)]

    candidates = domain_solutions[most_crowded_domain]
    domain_solution_NDL_idxs = np.empty(len(candidates))

    # Optimized loop for locating the NDL index
    for solution_idx, solution in enumerate(candidates):
        # NOTE: If your solution objects have a `.rank` or `.ndl` attribute,
        # replace this inner loop entirely with: `domain_solution_NDL_idxs[solution_idx] = solution.rank`
        for level_idx, level in enumerate(levels.levels):
            if any(solution.equations_labels == level_solution.equations_labels for level_solution in level):
                domain_solution_NDL_idxs[solution_idx] = level_idx
                break

    max_level = np.max(domain_solution_NDL_idxs)
    worst_NDL_section = [candidates[sol_idx] for sol_idx in range(len(candidates))
                         if domain_solution_NDL_idxs[sol_idx] == max_level]

    PBIS_worst = [
        penalty_based_intersection(solution, weights[most_crowded_domain], best_obj, penalty_factor, levels.normalizer)
        for solution in worst_NDL_section]

    return worst_NDL_section[np.argmax(PBIS_worst)]


class PopulationUpdater(CompoundOperator):
    key = 'PopulationUpdater'

    def apply(self, objective: Tuple, arguments: dict):
        '''
        Update population to get the pareto-nondominated levels with the worst element removed.
        Here, "worst" means the solution with highest PBI value (penalty-based boundary intersection).
        '''
        self_args, subop_args = self.parse_suboperator_args(arguments=arguments)

        # objective[1] represents the ParetoLevels object
        levels_obj = objective[1]

        # Add offspring to population and update non-dominated levels
        levels_obj.update(objective[0])

        if len(levels_obj.levels) == 1:
            # Algorithm 4, Case 1: single front — decomposition on entire population
            worst_solution = decomposition_based_worst(levels_obj.population, self_args['weights'],
                                                       self_args['best_obj'], self.params['PBI_penalty'],
                                                       levels_obj.normalizer)
        else:
            if len(levels_obj.levels[-1]) == 1:
                # Algorithm 4, Case 2: single solution on last front
                solution = levels_obj.levels[-1][0]
                population_by_domains = population_to_sectors(levels_obj.population, self_args['weights'])
                solution_subregion = next(domain for domain in population_by_domains if solution in domain)

                if len(solution_subregion) > 1:
                    worst_solution = solution
                else:
                    # Subregion has only this solution — use NDL-aware decomposition
                    worst_solution = locate_pareto_worst(levels_obj, self_args['weights'],
                                                         self_args['best_obj'], self.params['PBI_penalty'])
            else:
                # Algorithm 4, Case 3: multiple solutions on last front F_l.
                # DEVIATION from the paper: the crowded-subregion search and
                # the worst-PBI argmax are both restricted to F_l, NOT to
                # the full Phi^h as Algorithm 4 line 18 specifies. The
                # paper would let us eliminate an elite F_1 solution if it
                # happened to have the largest PBI inside the crowded
                # subregion; we prefer to keep the elite and only churn
                # the last-front candidates. See ``decomposition_based_worst``
                # docstring for the full rationale.
                last_front = levels_obj.levels[-1]
                last_front_by_domains = population_to_sectors(last_front, self_args['weights'])
                most_crowded_count = max(len(d) for d in last_front_by_domains)

                if most_crowded_count > 1:
                    # Most crowded F_l subregion has >1 solutions -- drop
                    # the worst-PBI one among F_l members of that subregion.
                    worst_solution = decomposition_based_worst(last_front, self_args['weights'],
                                                               self_args['best_obj'], self.params['PBI_penalty'],
                                                               levels_obj.normalizer)
                else:
                    # Every F_l solution sits alone in its subregion: fall
                    # back to NDL-aware LOCATE_WORST over the full P'
                    # (Algorithm 5).
                    worst_solution = locate_pareto_worst(levels_obj, self_args['weights'],
                                                         self_args['best_obj'], self.params['PBI_penalty'])

        levels_obj.delete_point(worst_solution)
        
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
            attempt = 0
            replaced = 0
            mutation_attempt_limit = self.params['mutation_attempt_limit']
            offspring_attempt_limit = self.params['offspring_attempt_limit']
            # self.suboperators['sparsity'].apply(objective=offspring,
            #                                     arguments=subop_args['sparsity'])
            offspring.reset_state(True)
            # First iteration aliases ``offspring``; ``chromosome_mutation``
            # (SystemMutation) deepcopies its input before mutating, so the
            # parent ``offspring`` is never touched. The initial
            # ``deepcopy(offspring)`` here was discarded immediately by the
            # very next line below (SystemMutation reassigns
            # ``temp_offspring``) -- a per-pass SoEq deepcopy of pure waste.
            temp_offspring = offspring
            total_attempts = 0
            hit_offspring_cap = False
            while True:
                total_attempts += 1
                temp_offspring = self.suboperators['chromosome_mutation'].apply(objective=temp_offspring,
                                                                                arguments=subop_args['chromosome_mutation'])
                temp_offspring.reset_state(True)
                # SoEqRightPartSelector enforces cross-equation RPS
                # uniqueness inline (sequential pre-scrub), so no post-hoc
                # ``enforce_rps_uniqueness`` retry loop is needed here.
                self.suboperators['right_part_selector'].apply(objective=temp_offspring,
                                                               arguments=subop_args['right_part_selector'])

                system = temp_offspring.equations_labels
                if system not in objective.history:
                    self.suboperators['chromosome_fitness'].apply(objective=temp_offspring,
                                                                  arguments=subop_args['chromosome_fitness'])
                    self.suboperators['pareto_level_updater'].apply(objective=(temp_offspring, objective),
                                                                    arguments=subop_args['pareto_level_updater'])
                    objective.history.add(system)
                    if global_var.verbose.candidate_objectives:
                        print(temp_offspring.obj_fun)
                    break
                if replaced == offspring_attempt_limit:
                    hit_offspring_cap = True
                    if global_var.verbose.candidate_objectives:
                        print("Could not generate unique offspring")
                    break
                if attempt == mutation_attempt_limit:
                    temp_offspring.create()
                    replaced += 1
                    attempt = 0
                    # print("Could not generate unique offspring")
                    # break
                attempt += 1
            # Track total iters and cap-hits separately for the success vs failure paths.
            theoretical_cap = (offspring_attempt_limit + 1) * (mutation_attempt_limit + 1)
            _loop_stats.record(
                'OffspringUpdater.unique_offspring' + ('.FAIL' if hit_offspring_cap else ''),
                total_attempts, theoretical_cap,
            )
        return objective
    
def get_pareto_levels_updater(right_part_selector : CompoundOperator, chromosome_fitness : CompoundOperator,
                              sparsity : CompoundOperator,
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
                                          'sparsity' : sparsity,
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
            uniqueness_attempt_limit = self.params['uniqueness_attempt_limit']
            if global_var.verbose.show_iter_idx:
                print('\n========== Initial population ==========')
            for idx, candidate in enumerate(objective.unplaced_candidates):
                candidate.reset_state(True)
                # SoEqRightPartSelector handles cross-equation RPS
                # uniqueness inline; no post-hoc retry needed.
                self.suboperators['right_part_selector'].apply(objective = candidate,
                                                                arguments = subop_args['right_part_selector'])

                system = candidate.equations_labels
                attempts = 0
                hit_cap = False
                while system in objective.history:
                    if attempts >= uniqueness_attempt_limit:
                        hit_cap = True
                        break
                    attempts += 1
                    candidate.create()
                    candidate.reset_state(True)
                    self.suboperators['right_part_selector'].apply(objective=candidate,
                                                                   arguments=subop_args['right_part_selector'])
                    system = candidate.equations_labels
                _loop_stats.record(
                    'InitialParetoLevelSorting.unique_candidate' + ('.FAIL' if hit_cap else ''),
                    attempts, uniqueness_attempt_limit,
                )
                if hit_cap:
                    # Initial population must be duplicate-free for MOEA/D's
                    # per-sector uniqueness invariant. If the candidate pool
                    # is too small to satisfy pop_size, fail loud rather than
                    # silently corrupt the initial Pareto layer.
                    raise RuntimeError(
                        f"InitialParetoLevelSorting: could not generate a unique "
                        f"initial candidate after {uniqueness_attempt_limit} attempts "
                        f"(candidate index {idx}, {len(objective.history)} unique "
                        f"systems already placed). The search space appears smaller "
                        f"than the requested population. Reduce pop_size, widen the "
                        f"token pool, or raise InitialParetoLevelSorting's "
                        f"'uniqueness_attempt_limit' parameter."
                    )
                self.suboperators['chromosome_fitness'].apply(objective=candidate,
                                                              arguments=subop_args['chromosome_fitness'])
                objective.history.add(system)
                if global_var.verbose.candidate_objectives:
                    print(candidate.obj_fun)
            if global_var.verbose.show_iter_idx:
                print('\n========== Marriage (weight assignment) ==========')
            objective.associate_weights()
            objective.initial_placing()
            if global_var.verbose.show_iter_idx:
                print('\n========== Multiobjective optimization ==========')

            # TODO: consider carefully, where normalizer init shall be held. If here, only the initial values are employed
        # objective.set_normalizer()

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

from itertools import combinations

def has_subset_pair(collection_of_sets):
    """
    Checks if any two sets within a collection are subsets of one another.
    """
    # Iterate through all unique pairs of sets in the collection
    for set1, set2 in combinations(collection_of_sets, 2):
        # Check if set1 is a subset of set2, or vice versa
        if set1.issubset(set2):
            # Found a pair that has a subset relationship
            return True, set1, set2
        elif set2.issubset(set1):
            return True, set2, set1
    # No subset relationship found among any pairs
    return False, None, None

def _debug_assert_rps_unique(objective) -> list:
    """Debug helper: scan an SoEq's equations and return a per-equation
    list of bools indicating which equations contain at least one
    non-target term whose factor set is a superset of another equation's
    target term factor set.

    The post-hoc enforcement loop that used to call this and rewrite
    conflicting terms is gone -- ``SoEqRightPartSelector`` now propagates
    the uniqueness constraint forward across equations during the RPS
    sweep itself, so a correctly-implemented pipeline must produce an
    all-False result here. Use this in tests or temporary asserts to
    catch regressions; do NOT wire it back into the operator graph as a
    repair step.
    """
    equations = list(objective.vals)
    rsterms = [eq.structure[eq.target_idx].factors_labels for eq in equations]
    flagged = [False] * len(equations)

    for eq_idx, equation in enumerate(equations):
        other_rs = rsterms[:eq_idx] + rsterms[eq_idx + 1:]
        for term_idx, term in enumerate(equation.structure):
            if term_idx == equation.target_idx:
                continue
            if any(rs.issubset(term.factors_labels) for rs in other_rs):
                flagged[eq_idx] = True
                break
    return flagged


def is_rps_in_other_equation(objective):
    """Deprecated alias. The post-hoc uniqueness repair has been replaced
    by ``SoEqRightPartSelector`` (sequential pre-scrub), so this is now a
    pure assertion helper that returns a per-equation flag list without
    mutating anything. External callers should migrate to using the new
    operator and remove their ``while any(is_rps_in_other_equation(...))``
    retry loops; the new operator guarantees the result is all-False on a
    well-formed SoEq.
    """
    warnings.warn(
        'is_rps_in_other_equation is now a pure debug check; '
        'SoEqRightPartSelector enforces uniqueness during RPS dispatch. '
        'Drop your retry loop.',
        DeprecationWarning, stacklevel=2,
    )
    return _debug_assert_rps_unique(objective)


def enforce_rps_uniqueness(objective, *, max_iter: int = 100) -> list:
    """Deprecated. Retained as an assertion-only shim for code that
    imports the old name; mutates nothing. Wraps
    :func:`_debug_assert_rps_unique`.
    """
    warnings.warn(
        'enforce_rps_uniqueness is now a pure debug check; '
        'SoEqRightPartSelector enforces uniqueness during RPS dispatch.',
        DeprecationWarning, stacklevel=2,
    )
    return _debug_assert_rps_unique(objective)