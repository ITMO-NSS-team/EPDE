#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 19:08:51 2022

@author: maslyaev
"""
import copy
import numpy as np
import time
import warnings
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

    if obj_normalizer is not None:
        # The solution objective above is normalized; the ideal point must
        # live on the same scale (a no-op for the usual all-zero ideal).
        ideal_obj_full = obj_normalizer(np.asarray(ideal_obj_full, dtype=float))

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


def _most_crowded_domain(domain_solutions: list, weights: np.ndarray, best_obj: np.ndarray,
                         penalty_factor: float, obj_normalizer, candidate_idxs: list = None) -> int:
    '''
    Select the index of the most crowded subregion; ties on niche count are
    broken by the largest sum of PBI in the tied subregions (paper Eq. (7)).
    Shared by ``decomposition_based_worst``, ``locate_pareto_worst`` and the
    last-front branch of ``PopulationUpdater``.

    ``candidate_idxs`` optionally restricts the search to a subset of
    subregion indices (paper Algorithm 4, line 16: "the most crowded
    subregion associated with those solutions in F_l"); the niche counts
    themselves always run over the full ``domain_solutions`` content.
    '''
    if candidate_idxs is None:
        candidate_idxs = range(len(domain_solutions))
    most_crowded_count = max(len(domain_solutions[idx]) for idx in candidate_idxs)
    crowded_domains = [idx for idx in candidate_idxs
                       if len(domain_solutions[idx]) == most_crowded_count]

    if len(crowded_domains) == 1:
        return crowded_domains[0]

    PBIS = [sum(penalty_based_intersection(sol, weights[domain_idx], best_obj, penalty_factor, obj_normalizer)
                for sol in domain_solutions[domain_idx])
            for domain_idx in crowded_domains]
    return crowded_domains[np.argmax(PBIS)]


def decomposition_based_worst(solutions: list, weights: np.ndarray, best_obj: np.ndarray,
                              penalty_factor: float = 1., obj_normalizer=None, sectors: list = None):
    '''
    Decomposition-based worst-solution finder. Returns argmax PBI over the
    most-crowded subregion within ``solutions``; ties on niche count are
    broken by sum-PBI per paper Eq. (7).

    Used by ``PopulationUpdater`` for case ``l = 1`` of Algorithm 4 in the
    MOEA/DD paper (Li, Deb, Zhang, Kwong, 2015) -- called with the full
    population; equivalent to ``LOCATE_WORST`` (Algorithm 5) since every
    solution lives on the single front.
    '''
    domain_solutions = population_to_sectors(solutions, weights) if sectors is None else sectors
    most_crowded_domain = _most_crowded_domain(domain_solutions, weights, best_obj,
                                               penalty_factor, obj_normalizer)

    candidates = domain_solutions[most_crowded_domain]

    # Find the solution with the largest individual PBI in the selected subregion
    PBIS_candidates = [
        penalty_based_intersection(s, weights[most_crowded_domain], best_obj, penalty_factor, obj_normalizer)
        for s in candidates]

    return candidates[np.argmax(PBIS_candidates)]


def locate_pareto_worst(levels, weights: np.ndarray, best_obj: np.ndarray, penalty_factor: float = 1.,
                        sectors: list = None):
    '''
    Function dedicated to the selection of the worst solution on the Pareto levels.
    '''
    domain_solutions = population_to_sectors(levels.population, weights) if sectors is None else sectors
    most_crowded_domain = _most_crowded_domain(domain_solutions, weights, best_obj,
                                               penalty_factor, levels.normalizer)

    candidates = domain_solutions[most_crowded_domain]
    domain_solution_NDL_idxs = np.empty(len(candidates))

    for solution_idx, solution in enumerate(candidates):
        try:
            domain_solution_NDL_idxs[solution_idx] = next(
                level_idx for level_idx, level in enumerate(levels.levels)
                if any(solution is level_solution for level_solution in level))
        except StopIteration:
            raise RuntimeError(
                'locate_pareto_worst: a candidate solution from the population is '
                'missing from the Pareto levels; population and levels are out of sync.')

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
        weights = self_args['weights']

        # Add offspring to population and update non-dominated levels
        levels_obj.update(objective[0])

        # Sector association is computed once per update and passed into
        # the worst-finders (previously recomputed up to 3x per insertion).
        population_sectors = population_to_sectors(levels_obj.population, weights)

        if len(levels_obj.levels) == 1:
            # Algorithm 4, Case 1: single front — decomposition on entire population
            worst_solution = decomposition_based_worst(levels_obj.population, weights,
                                                       self_args['best_obj'], self.params['PBI_penalty'],
                                                       levels_obj.normalizer, sectors=population_sectors)
        else:
            if len(levels_obj.levels[-1]) == 1:
                # Algorithm 4, Case 2: single solution on last front
                solution = levels_obj.levels[-1][0]
                solution_subregion = next(domain for domain in population_sectors if solution in domain)

                if len(solution_subregion) > 1:
                    worst_solution = solution
                else:
                    # Subregion has only this solution — use NDL-aware decomposition
                    worst_solution = locate_pareto_worst(levels_obj, weights,
                                                         self_args['best_obj'], self.params['PBI_penalty'],
                                                         sectors=population_sectors)
            else:
                # Algorithm 4, Case 3 (lines 16-22): multiple solutions on
                # the last front F_l. Identify the most crowded subregion
                # Phi^h among the subregions associated with F_l members;
                # niche counts and the worst-PBI argmax both run over the
                # FULL subregion content, so an elite solution from an
                # earlier front is eliminated if it owns the largest PBI
                # inside Phi^h -- exactly as the paper specifies.
                last_front = levels_obj.levels[-1]
                last_front_ids = {id(sol) for sol in last_front}
                fl_domain_idxs = [idx for idx, domain in enumerate(population_sectors)
                                  if any(id(sol) in last_front_ids for sol in domain)]
                most_crowded_idx = _most_crowded_domain(population_sectors, weights,
                                                        self_args['best_obj'], self.params['PBI_penalty'],
                                                        levels_obj.normalizer, candidate_idxs=fl_domain_idxs)
                subregion = population_sectors[most_crowded_idx]

                if len(subregion) > 1:
                    # |Phi^h| > 1: eliminate argmax PBI over the whole
                    # subregion (Algorithm 4, lines 17-19).
                    PBIS = [penalty_based_intersection(sol, weights[most_crowded_idx],
                                                       self_args['best_obj'], self.params['PBI_penalty'],
                                                       levels_obj.normalizer)
                            for sol in subregion]
                    worst_solution = subregion[np.argmax(PBIS)]
                else:
                    # |Phi^h| = 1: every F_l member is associated with an
                    # isolated subregion -- preserve them and fall back to
                    # LOCATE_WORST over the full P' (Algorithm 5, lines
                    # 20-22 of Algorithm 4).
                    worst_solution = locate_pareto_worst(levels_obj, weights,
                                                         self_args['best_obj'], self.params['PBI_penalty'],
                                                         sectors=population_sectors)

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
            Selector of neighboring weight vectors: randomly chooses
            *number_of_neighbors* indices from the proximity list E(i) of the
            processed weight vector, as prescribed by Algorithm 3, line 2 of
            the MOEA/DD paper ("Randomly choose k indices from E(i)").
            Defined to be used inside the moeadd algorithm.

            Arguments:
            ----------

            sorted_neighbors : list
                proximity list of neighboring vectors, ranged in the ascending order of the angles between vectors.

            number_of_neighbors : int
                numbers of vectors to be considered as the adjacent ones

            Returns:
            ---------

            selected_neighbors : list
                random subset of the proximity list of size *number_of_neighbors*
        '''
        self_args, subop_args = self.parse_suboperator_args(arguments = arguments)
        n_select = min(self.params['number_of_neighbors'], len(objective))
        return list(np.random.choice(objective, size = n_select, replace = False))
    
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
            # RPS-state reset is no longer done here in bulk; it is coupled to
            # the structure-changing operators (EquationCrossover / Equation
            # Mutation), so only changed equations are re-selected.
            # Crossover offspring are deepcopies of their parents and so
            # inherit the parent's cached sector domain / objective vector
            # (precomputed_domain / precomputed_value). Reset, so niching
            # operates on the offspring's OWN objective-space position --
            # consistent with create()-rebuilt solutions, whose caches are
            # reset by MOEADDSolution.__init__.
            offspring.reset_moeadd_state()
            # ``chromosome_mutation`` (SystemMutation) mutates its input
            # IN PLACE and returns the same object -- the pop() above is
            # the only live reference, so no defensive deepcopy is needed.
            # The retry loop below depends on that identity contract:
            # each pass keeps editing the same SoEq until it is unique
            # or replaced wholesale by ``temp_offspring.create()``.
            temp_offspring = offspring
            total_attempts = 0
            hit_offspring_cap = False
            while True:
                total_attempts += 1
                temp_offspring = self.suboperators['chromosome_mutation'].apply(objective=temp_offspring,
                                                                                arguments=subop_args['chromosome_mutation'])
                # EquationMutation already reset the RPS state of the changed
                # equations; only the moeadd niching cache must be re-derived
                # from the post-mutation objectives.
                temp_offspring.reset_moeadd_state()
                # SoEqRightPartSelector resolves system degeneracy inline
                # (no two equations may share an identical active
                # structure), so no post-hoc ``enforce_rps_uniqueness``
                # retry loop is needed here.
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
                # SoEqRightPartSelector resolves system degeneracy
                # inline; no post-hoc retry needed.
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
                    # Search-space collapse: the RPS+simplify pipeline canonicalises
                    # ``uniqueness_attempt_limit`` consecutive create() outputs into
                    # one of the already-placed structures. Diagnostic builds want
                    # to observe the partial population's objective values rather
                    # than abort. Policy:
                    # warn, register the placed candidates in ``population`` /
                    # ``levels[0]`` so downstream consumers can read their
                    # ``text_form`` + ``obj_fun``, set ``_init_collapsed=True``,
                    # and stop -- MOEA/D's ``optimize`` checks this flag after
                    # init and skips the epoch loop so mutation/crossover never
                    # runs on the degenerate population.
                    warnings.warn(
                        f"InitialParetoLevelSorting: search-space collapse at "
                        f"candidate {idx} after {uniqueness_attempt_limit} attempts "
                        f"({len(objective.history)} unique systems placed of "
                        f"{len(objective.unplaced_candidates)} requested). Stopping "
                        f"the search; placed candidates are accessible via "
                        f"pareto_levels.levels[0]."
                    )
                    placed = list(objective.unplaced_candidates[:idx])
                    objective.population = placed
                    objective.unplaced_candidates = []
                    # Single-level dump: Pareto-correctness is moot here because
                    # we're terminating. Consumers reading ``levels[0]`` get
                    # every placed candidate.
                    objective.levels = [placed] if placed else [[]]
                    objective._init_collapsed = True
                    init_collapsed = True
                    break
                self.suboperators['chromosome_fitness'].apply(objective=candidate,
                                                              arguments=subop_args['chromosome_fitness'])
                objective.history.add(system)
                if global_var.verbose.candidate_objectives:
                    print(candidate.obj_fun)
            else:
                init_collapsed = False
            if init_collapsed:
                if global_var.verbose.show_iter_idx:
                    print(f'\n*** Search-space collapse: stopping early with '
                          f'{len(objective.levels[0])} placed candidates. ***')
                return
            if global_var.verbose.show_iter_idx:
                print('\n========== Marriage (weight assignment) ==========')
            objective.associate_weights()
            objective.initial_placing()
            # Initialize the PBI objective normalizer from the placed
            # population; MOEADDOptimizer.optimize refreshes it per epoch.
            objective.set_normalizer()
            if global_var.verbose.show_iter_idx:
                print('\n========== Multiobjective optimization ==========')

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
    """Debug helper: scan an SoEq's equations pairwise and return a
    per-equation list of bools flagging equations whose ACTIVE structure
    (target + nonzero-weight terms, ``Equation.active_terms_labels``)
    coincides with an EARLIER equation's -- i.e. the system carries the
    same law twice (rearranged), the degenerate state that
    ``SoEqRightPartSelector`` resolves during RPS dispatch.

    Cross-equation term sharing is NOT flagged: an equation for ``v``
    keeping ``du/dx0`` as a coupling term is legitimate. A correctly-
    implemented pipeline must produce an all-False result here. Use this
    in tests or temporary asserts to catch regressions; do NOT wire it
    back into the operator graph as a repair step.
    """
    equations = list(objective.vals)
    sigs = [eq.active_terms_labels for eq in equations]
    flagged = [False] * len(equations)

    for eq_idx in range(1, len(equations)):
        for other_idx in range(eq_idx):
            if sigs[eq_idx] and sigs[eq_idx] == sigs[other_idx]:
                flagged[eq_idx] = True
                break
    return flagged


def is_rps_in_other_equation(objective):
    """Deprecated alias. The post-hoc uniqueness repair has been replaced
    by ``SoEqRightPartSelector`` (system-degeneracy resolution: no two
    equations may share an identical active structure), so this is now a
    pure assertion helper that returns a per-equation flag list without
    mutating anything. External callers should migrate to using the new
    operator and remove their ``while any(is_rps_in_other_equation(...))``
    retry loops; the new operator guarantees the result is all-False on a
    well-formed SoEq.
    """
    warnings.warn(
        'is_rps_in_other_equation is now a pure debug check; '
        'SoEqRightPartSelector resolves system degeneracy during RPS '
        'dispatch. Drop your retry loop.',
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
        'SoEqRightPartSelector resolves system degeneracy during RPS dispatch.',
        DeprecationWarning, stacklevel=2,
    )
    return _debug_assert_rps_unique(objective)