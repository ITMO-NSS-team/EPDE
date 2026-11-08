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
from itertools import combinations
from typing import Union, Tuple
from functools import reduce, partial

from epde.optimizers.moeadd.moeadd import ParetoLevels, ObjFunNormalizer
from epde.operators.utils.template import CompoundOperator, add_base_param_to_operator
import epde.globals as global_var
from epde.operators.multiobjective.mutations import get_basic_mutation

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


def population_to_sectors(population, weights, obj_normalizer=None):
    '''
    The distribution of the solutions into the domains, defined by weights vectors.

    ``obj_normalizer`` reaches MOEA/DD Eq. (6) through
    ``MOEADDSolution.get_domain``, so a solution's subregion is decided on the
    same scale ``penalty_based_intersection`` and ``marriageSolutionAssignment``
    use. It only affects solutions whose association has not been derived yet.
    '''
    solution_selection = lambda weight_idx: [solution for solution in population
                                             if solution.get_domain(weights, obj_normalizer) == weight_idx]
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


def _worst_by_pbi(candidates: list, weight, best_obj, penalty_factor, obj_normalizer):
    '''
    Argmax of the individual PBI over ``candidates`` for one weight vector
    (first max wins on ties, per ``np.argmax``). The elimination core shared
    by ``decomposition_based_worst``, ``locate_pareto_worst`` and the
    last-front branch of ``PopulationUpdater``.
    '''
    PBIS = [penalty_based_intersection(candidate, weight, best_obj, penalty_factor, obj_normalizer)
            for candidate in candidates]
    return candidates[np.argmax(PBIS)]


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
    domain_solutions = (population_to_sectors(solutions, weights, obj_normalizer)
                        if sectors is None else sectors)
    most_crowded_domain = _most_crowded_domain(domain_solutions, weights, best_obj,
                                               penalty_factor, obj_normalizer)

    candidates = domain_solutions[most_crowded_domain]

    # The solution with the largest individual PBI in the selected subregion.
    return _worst_by_pbi(candidates, weights[most_crowded_domain], best_obj,
                         penalty_factor, obj_normalizer)


def locate_pareto_worst(levels, weights: np.ndarray, best_obj: np.ndarray, penalty_factor: float = 1.,
                        sectors: list = None):
    '''
    Function dedicated to the selection of the worst solution on the Pareto levels.
    '''
    domain_solutions = (population_to_sectors(levels.population, weights, levels.normalizer)
                        if sectors is None else sectors)
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

    return _worst_by_pbi(worst_NDL_section, weights[most_crowded_domain], best_obj,
                         penalty_factor, levels.normalizer)


# ---------------------------------------------------------------------------
# evolution.niching='per_equation': every equation is ranked in its own
# (discrepancy, instability) plane; nothing is summed across equations.
# ---------------------------------------------------------------------------
def niching_per_equation(population) -> bool:
    """True when ``evolution.niching='per_equation'`` applies: the setting is on
    and the systems hold more than one equation (a single-equation system keeps
    the published MOEA/DD)."""
    from epde.interface.search_config import active_config
    return (active_config().evolution.niching == 'per_equation'
            and len(population) > 0 and len(population[0].vals) > 1)


def equation_planes(population):
    """``(planes, labels)``. ``planes[e]`` is an array with one row per system:
    that system's values of equation ``e``'s objectives only (``SoEq.obj_fun``
    is grouped by objective type, then by equation). ``labels[e]`` holds each
    system's structure of equation ``e`` -- ``check_dominance``'s own
    (``supplementary._structure_label``): the terms, and in plain mode whether
    the constant is in, so S and S+{1} are two structures there."""
    from epde.optimizers.moeadd.supplementary import _structure_label
    n_eq = len(population[0].vals)
    keys = population[0].vals.equation_keys
    objs = [np.asarray(system.obj_fun, dtype=float) for system in population]
    n_types = objs[0].size // n_eq
    planes = [np.array([obj[[t * n_eq + e for t in range(n_types)]] for obj in objs])
              for e in range(n_eq)]
    labels = [[_structure_label(system.vals[keys[e]]) for system in population]
              for e in range(n_eq)]
    return planes, labels


def plane_levels(points, labels) -> np.ndarray:
    """Non-domination level of every system in ONE equation's plane. Two
    systems holding the same structure of that equation never dominate each
    other: the same structure is the same equation (``check_dominance``)."""
    points = np.asarray(points, dtype=float)
    levels = np.full(len(points), -1, dtype=int)
    remaining = list(range(len(points)))
    level = 0
    while remaining:
        current = [i for i in remaining
                   if not any(labels[j] != labels[i]
                              and np.all(points[j] <= points[i])
                              and np.any(points[j] < points[i])
                              for j in remaining if j != i)]
        for i in current:
            levels[i] = level
        current_set = set(current)
        remaining = [i for i in remaining if i not in current_set]
        level += 1
    return levels


def equation_front_holders(population, system_level=None) -> list:
    """For each equation, the indices of the systems that HOLD a structure on
    that equation's own front. One holder per structure -- the same structure
    is the same equation -- the copy on the shallowest system level
    (``system_level(i)``, when given), then the first in population order."""
    planes, labels = equation_planes(population)
    holders = []
    for points, labs in zip(planes, labels):
        levels = plane_levels(points, labs)
        chosen = {}
        for i, lab in enumerate(labs):
            if levels[i] != 0:
                continue
            rank = (system_level(i) if system_level is not None else 0, i)
            if lab not in chosen or rank < chosen[lab][0]:
                chosen[lab] = (rank, i)
        holders.append({i for _, i in chosen.values()})
    return holders


def _crowding_distances(points) -> np.ndarray:
    """NSGA-II crowding distance of points on one front (larger = sparser)."""
    points = np.asarray(points, dtype=float)
    n = len(points)
    distance = np.zeros(n)
    if n <= 2:
        return np.full(n, np.inf)
    for col in range(points.shape[1]):
        order = np.argsort(points[:, col], kind='stable')
        span = points[order[-1], col] - points[order[0], col]
        distance[order[0]] = distance[order[-1]] = np.inf
        if span > 0:
            gaps = (points[order[2:], col] - points[order[:-2], col]) / span
            distance[order[1:-1]] += gaps
    return distance


def per_equation_worst(levels_obj):
    """The system to delete under ``niching='per_equation'``.

    Systems holding a structure on some equation's own front are protected.
    Among the rest, the victim is the one whose BEST per-equation plane rank is
    the worst (its sorted ranks compared lexicographically, then the deeper
    system level, then the later position): a system good in one equation
    outlives one that is mediocre everywhere, and no errors are added across
    equations. When every system is protected, the victim holds the fewest
    equation fronts and, among those, sits most crowded in its own plane
    (smallest crowding distance; ties to the later position).
    """
    population = levels_obj.population
    level_of = {id(system): idx for idx, level in enumerate(levels_obj.levels)
                for system in level}
    system_level = lambda i: level_of.get(id(population[i]), 0)          # noqa: E731
    planes, labels = equation_planes(population)
    ranks = np.array([plane_levels(points, labs)
                      for points, labs in zip(planes, labels)]).T      # (systems, equations)
    # The same structure is the same equation: in each plane only one copy of
    # a structure -- the shallowest system, then the first -- keeps its rank.
    # Every other copy brings nothing of its own there and ranks last, so a
    # clone never outlives a system with an equation of its own.
    for e, labs in enumerate(labels):
        representative = {}
        for i, lab in enumerate(labs):
            key = (system_level(i), i)
            if lab not in representative or key < representative[lab][0]:
                representative[lab] = (key, i)
        keep = {i for _, i in representative.values()}
        for i in range(len(population)):
            if i not in keep:
                ranks[i, e] = len(population)
    holders = equation_front_holders(population, system_level)
    counts = [sum(i in held for held in holders) for i in range(len(population))]
    unprotected = [i for i, count in enumerate(counts) if count == 0]
    if unprotected:
        _loop_stats.record('PerEqSurvival.unprotected_victim', 1, 1)
        victim = max(unprotected,
                     key=lambda i: (tuple(sorted(ranks[i])), system_level(i), i))
        return population[victim]
    _loop_stats.record('PerEqSurvival.all_protected', 1, 1)
    fewest = min(counts)
    tied = [i for i, count in enumerate(counts) if count == fewest]
    crowding = {}
    for e, held in enumerate(holders):
        members = sorted(held)
        distances = _crowding_distances(planes[e][members])
        for i, dist in zip(members, distances):
            if i in tied:
                crowding[i] = min(crowding.get(i, np.inf), dist)
    victim = min(tied, key=lambda i: (crowding.get(i, np.inf), -i))
    return population[victim]


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

        if niching_per_equation(levels_obj.population):
            levels_obj.delete_point(per_equation_worst(levels_obj))
            return

        # Sector association is computed once per update and passed into
        # the worst-finders (previously recomputed up to 3x per insertion).
        # The normalizer goes with it so Eq. (6) is evaluated on the scale PBI
        # selects on -- the same scale the initial marriage assignment used.
        population_sectors = population_to_sectors(levels_obj.population, weights,
                                                   levels_obj.normalizer)

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
                    worst_solution = _worst_by_pbi(subregion, weights[most_crowded_idx],
                                                   self_args['best_obj'], self.params['PBI_penalty'],
                                                   levels_obj.normalizer)
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


def _equation_is_degenerate(eq) -> bool:
    """An equation the in-place fitness host flagged fit-degenerate: its
    primary objective was set to ``LOSS_NAN_VAL`` (sparsity collapsed it, its
    discrepancy exceeded the degeneracy threshold, or RPS left it unfitted)."""
    from epde.operators.common.objectives import LOSS_NAN_VAL
    fv = getattr(eq, 'fitness_value', None)
    return fv is not None and fv >= LOSS_NAN_VAL


def has_degenerate_equation(offspring) -> bool:
    """True if ANY equation of ``offspring`` is still fit-degenerate."""
    return any(_equation_is_degenerate(eq) for eq in offspring.vals)


class _FirstLevelCount(object):
    """The first Pareto level of the systems placed so far, kept as each
    system's count of placed systems dominating it (``check_dominance``):
    level 0 is the systems with a count of 0, as in Deb's sort."""

    def __init__(self):
        self.members, self.dominators = [], []

    def place(self, system) -> None:
        from epde.optimizers.moeadd.supplementary import check_dominance
        count = 0
        for k, other in enumerate(self.members):
            if check_dominance(other, system):
                count += 1
            elif check_dominance(system, other):
                self.dominators[k] += 1
        self.members.append(system)
        self.dominators.append(count)

    @property
    def size(self) -> int:
        return sum(1 for count in self.dominators if count == 0)

    def holds(self) -> bool:
        """At most half the placed systems on the first level."""
        return 2 * self.size <= len(self.members)


#: Redraws :func:`regenerate_degenerate_equations` gives one equation that
#: right-part selection DECLINED (``EqRightPartSelector._decline``) before its
#: system is given up.
#:
#: The selector used to redraw such an equation itself, in place
#: (``objective.randomize()``), up to its outer cap of 50 passes per call. It
#: now declines, so its output is a function of its input, and this budget
#: moves that redraw to the caller at the same size. Measured on Lorenz under
#: the chi2 keep rule (seed 0, 5 epochs, the selector that still redrew): 12422
#: in-place redraws over 2629 selector calls, about 4.7 per call. That puts
#: the acceptance of one redraw at 0.18 at most. The declined equations that
#: three plain redraw passes left unplaced (4279 of 6296) put it near 0.12. At those
#: rates all 50 redraws of one equation fail with probability 1e-4 to 2e-3,
#: and the mean cost is 6 to 8 redraws. Three passes, the budget before, lost
#: 68% of the declined equations. Two of three seeds lost their initial
#: population, and the third placed 171 offspring where the selector that
#: redrew placed 480.
RESAMPLE_BUDGET = 50


def _declined_by_selection(eq) -> bool:
    """A fit-degenerate equation that right-part selection DECLINED, as opposed
    to a fitted one the fitness host stamped (a discrepancy over the
    degeneracy threshold, an amplified fit, a failed solve). The decline
    zeroes every weight (``right_part_selection._mark_unassignable``); a fit
    the sweep accepted keeps a non-zero term weight, since the sweep declines
    an all-zero support."""
    weights = getattr(eq, 'weights_internal', None)
    return (_equation_is_degenerate(eq) and weights is not None
            and not np.any(np.asarray(weights)))


def regenerate_degenerate_equations(offspring, right_part_selector, chromosome_fitness,
                                    rps_args, fitness_args, max_passes: int = 3,
                                    budget: int = None) -> bool:
    """Redraw ONLY the fit-degenerate equations of ``offspring`` in place;
    True when none is left.

    A degenerate equation carries ``LOSS_NAN_VAL`` on every objective. Rather
    than discard the whole system -- wasting its healthy equations -- each one
    is redrawn (``randomize``), the system is selected and scored again, and
    this repeats until no equation is degenerate or a budget runs out:

    * an equation right-part selection DECLINED (:func:`_declined_by_selection`;
      in ``'repair'`` mode, where the selector used to redraw it in place) gets
      up to ``budget`` redraws, :data:`RESAMPLE_BUDGET` by default;
    * any other degenerate equation -- a fit the fitness host stamped, or in
      ``'assignment'`` / ``'plain'`` mode an equation the system-level choice
      left unassigned -- shares ``max_passes`` redraw passes, as before.

    Either budget running out loses the system, whatever the other equations
    do, and nothing more is redrawn. With ``max_passes=0`` (the
    single-objective search) a stamped fit therefore gives the system up
    before any redraw, a declined equation beside it included: the fit is
    kept, so its stamp stands whatever the redraws give.

    The redraw is selected inside ``resample_scope``: the equations still
    fitted keep their fits, so a pass fits only the redrawn equations. The
    chromosome fitness then scores the system as it always does. With every
    other equation scored, a redraw declined again is not scored at all. A
    solver host does not solve a system that holds a declined equation.

    Equations still degenerate when a budget runs out keep their stamp, and
    the caller is expected to DROP the offspring (rather than place a
    ``LOSS_NAN_VAL`` form) or, in the initial population, draw a new system. A
    system with no degenerate equation is returned untouched -- no draw, no
    selection, no scoring. A single-equation system degrades to a plain
    redraw of that equation.

    Shared by ``OffspringUpdater`` and ``InitialParetoLevelSorting`` so the
    initial population is held to the same no-degenerate-forms contract as the
    per-epoch offspring, and by the single-objective search
    (``so_specific.ResamplingRightPartSelector``).
    """
    from epde.interface.search_config import active_config
    from epde.operators.common.right_part_selection import resample_scope
    if budget is None:
        budget = RESAMPLE_BUDGET
    repair = active_config().search_space.system_rps_mode == 'repair'
    redraws = {}        # equation position -> redraws its declines used
    passes = 0          # passes spent on the other degenerate equations
    while True:
        equations = list(offspring.vals)
        degenerate = [i for i, eq in enumerate(equations) if _equation_is_degenerate(eq)]
        if not degenerate:
            break
        declined = {i for i in degenerate if repair and _declined_by_selection(equations[i])}
        # One equation out of redraws loses the system, whatever the others do.
        if any(redraws.get(i, 0) >= budget for i in declined):
            break
        if len(declined) < len(degenerate):
            if passes >= max_passes:
                break
            passes += 1
        for i in degenerate:
            # ``randomize`` rebuilds through ``__init__``, which resets.
            equations[i].randomize()
            if i in declined:
                redraws[i] = redraws.get(i, 0) + 1
        offspring.reset_moeadd_state()
        with resample_scope():
            right_part_selector.apply(objective=offspring, arguments=rps_args)
        chromosome_fitness.apply(objective=offspring, arguments=fitness_args)

    left = {i for i, eq in enumerate(offspring.vals) if _equation_is_degenerate(eq)}
    for i, used in redraws.items():
        _loop_stats.record('regenerate_degenerate_equations.redraws', used, budget)
    if passes:
        _loop_stats.record('regenerate_degenerate_equations.passes', passes, max_passes)
    # Every system that came in degenerate records its outcome, one given up
    # before any redraw too: with nothing spent, ``left`` is empty unless the
    # system came in degenerate.
    if redraws or passes or left:
        _loop_stats.record('regenerate_degenerate_equations.'
                           + ('exhausted' if left else 'resampled'), 1, 1)
    return not left


def _rps_fitness_regenerate(candidate, right_part_selector, chromosome_fitness,
                            rps_args, fitness_args, skip_fitness_if=None):
    """Right-part selection, then fitness, then re-roll of any equations the
    fitness host flagged fit-degenerate -- the evaluation sequence every fresh
    or re-created candidate goes through before its verdicts (degenerate?
    duplicate?) can be read. Used by ``InitialParetoLevelSorting``.

    ``skip_fitness_if`` is a predicate on the post-RPS candidate: when it
    holds, fitness (and the degenerate re-roll that depends on it) is NOT run
    and ``False`` is returned. That is what keeps the initial population from
    scoring candidates it is about to throw away -- on the solver path a
    discarded score is a full PDE solve, and the retry budget is 100.

    Skipping is only sound because RPS is the last step that can change the
    structure: it fits every candidate target, then prunes the winner
    (``remove_zero_terms``). The fitness hosts are pure scorers, so the label
    RPS leaves behind IS the label the scored candidate would carry, and a
    duplicate verdict read here cannot be overturned by the fit.
    """
    right_part_selector.apply(objective=candidate, arguments=rps_args)
    if skip_fitness_if is not None and skip_fitness_if(candidate):
        return False
    chromosome_fitness.apply(objective=candidate, arguments=fitness_args)
    regenerate_degenerate_equations(candidate, right_part_selector,
                                    chromosome_fitness, rps_args, fitness_args)
    return True


class OffspringUpdater(CompoundOperator):
    key = 'ParetoLevelUpdater'

    def apply(self, objective: ParetoLevels, arguments: dict):
        # The system right-part selector in ``system_rps_mode='assignment'``
        # skips the variants this population has already seen; 'repair' mode
        # does not read the history.
        from epde.operators.common.right_part_selection import system_history
        with system_history(objective.history):
            return self._update(objective, arguments)

    def _update(self, objective: ParetoLevels, arguments: dict):
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
                # (no equation carries another's target as a standalone term,
                # and no two equations share an identical active structure),
                # so no post-hoc ``enforce_rps_uniqueness`` retry loop is
                # needed here.
                self.suboperators['right_part_selector'].apply(objective=temp_offspring,
                                                               arguments=subop_args['right_part_selector'])

                system = temp_offspring.equations_labels
                if system in objective.history and self._swap_arrangement(temp_offspring, objective,
                                                                          subop_args):
                    break
                if system not in objective.history:
                    self.suboperators['chromosome_fitness'].apply(objective=temp_offspring,
                                                                  arguments=subop_args['chromosome_fitness'])
                    # Redraw ONLY the fit-degenerate equations (declined by
                    # right-part selection or stamped by the fitness host),
                    # each within its budget, leaving the system's good
                    # equations and their fits intact -- a single bad gene no
                    # longer dominates the whole offspring out.
                    regenerate_degenerate_equations(
                        temp_offspring,
                        self.suboperators['right_part_selector'],
                        self.suboperators['chromosome_fitness'],
                        subop_args['right_part_selector'],
                        subop_args['chromosome_fitness'])
                    # The pre-fit check above already keyed the PRUNED active
                    # structure -- ``right_part_selector`` fits and prunes, and
                    # the fitness host changes neither. What can still move the
                    # label is ``regenerate_degenerate_equations``: it redraws
                    # the degenerate genes and re-runs RPS on them. Re-check, so
                    # a duplicate active form is never placed twice (the
                    # repeated-objective leak). An offspring still degenerate
                    # after regeneration is DROPPED, not inserted, so
                    # LOSS_NAN_VAL never reaches the front/history.
                    final_label = temp_offspring.equations_labels
                    if (final_label not in objective.history
                            and not has_degenerate_equation(temp_offspring)):
                        self.suboperators['pareto_level_updater'].apply(objective=(temp_offspring, objective),
                                                                        arguments=subop_args['pareto_level_updater'])
                        objective.history.add(final_label)
                        if global_var.verbose.candidate_objectives:
                            print(temp_offspring.obj_fun)
                        self._place_siblings(temp_offspring, objective, subop_args)
                        break
                    # Duplicate active form or unrecoverable degenerate: record
                    # the active label so an identical pruned candidate
                    # short-circuits, then keep searching for a novel one.
                    objective.history.add(final_label)
                self._place_siblings(temp_offspring, objective, subop_args)
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

    def _place_siblings(self, offspring, objective, subop_args) -> None:
        """system_rps_mode='plain': the S+{1} versions of ``offspring`` that its
        fit could not rank against S enter the population too -- each one not
        yet in the history and not degenerate, through the same level updater.
        Nothing is handed over in any other mode."""
        from epde.operators.common.plain_fit import take_siblings
        for sibling in take_siblings(offspring):
            label = sibling.equations_labels
            if label in objective.history or has_degenerate_equation(sibling):
                continue
            sibling.reset_moeadd_state()
            self.suboperators['pareto_level_updater'].apply(objective=(sibling, objective),
                                                            arguments=subop_args['pareto_level_updater'])
            objective.history.add(label)

    def _swap_arrangement(self, offspring, objective, subop_args) -> bool:
        """system_rps_mode='plain': one arrangement per structure (USER, Oct 23
        2026). An offspring whose structures a population member already holds,
        solved for other targets, replaces that member when its own objectives
        dominate the member's -- compared value by value, since
        ``check_dominance`` never compares the same structure. True when it did;
        otherwise the offspring is a duplicate, as any candidate in the history.

        MOEA/DD re-derives the levels (``delete_point``, ``update``); NSGA-II
        swaps the member in its population, which it sorts once per generation."""
        from epde.interface.search_config import active_config
        config = active_config()
        if config.search_space.system_rps_mode != 'plain':
            return False
        label = offspring.equations_labels
        incumbent = next((member for member in objective.population
                          if member.equations_labels == label), None)
        if incumbent is None or incumbent.targets_labels == offspring.targets_labels:
            return False
        self.suboperators['chromosome_fitness'].apply(objective=offspring,
                                                      arguments=subop_args['chromosome_fitness'])
        if has_degenerate_equation(offspring) or offspring.equations_labels != label:
            return False
        new, old = np.asarray(offspring.obj_fun, dtype=float), np.asarray(incumbent.obj_fun, dtype=float)
        if not (new.shape == old.shape and np.all(new <= old) and np.any(new < old)):
            _loop_stats.record('OffspringUpdater.arrangement_kept', 1, 1)
            return False
        _loop_stats.record('OffspringUpdater.arrangement_swapped', 1, 1)
        offspring.reset_moeadd_state()
        if config.evolution.engine == 'nsga2':
            objective.population = [offspring if member is incumbent else member
                                    for member in objective.population]
            objective.levels = [[offspring if member is incumbent else member for member in level]
                                for level in objective.levels]
        else:
            objective.delete_point(incumbent)
            objective.update(offspring)
        return True

def get_pareto_levels_updater(right_part_selector : CompoundOperator, chromosome_fitness : CompoundOperator,
                              sparsity : CompoundOperator,
                              mutation : CompoundOperator = None, constrained : bool = False,
                              mutation_params : dict = {}, pl_updater_params : dict = {},
                              combiner_params : dict = {}, pl_updater : CompoundOperator = None):
    """``pl_updater`` replaces the level updater MOEA/DD's survival uses
    (``PopulationUpdater``), e.g. with ``evolution.engine='nsga2'``'s
    ``AppendUpdater``; None builds the default."""
    add_kwarg_to_updater = partial(add_base_param_to_operator, target_dict = combiner_params)
    updater = OffspringUpdater()
    add_kwarg_to_updater(operator = updater)

    if mutation is None:
        mutation = get_basic_mutation(mutation_params)
    if pl_updater is None:
        pl_updater = get_basic_populator_updater(pl_updater_params)
    updater.set_suboperators(operators = {'chromosome_mutation' : mutation,
                                          'pareto_level_updater' : pl_updater,
                                          # Dormant hook retained for the legacy
                                          # LASSO mode; on the live path sparsity
                                          # runs inside the fitness host (see the
                                          # MOEADD strategy wiring).
                                          'sparsity' : sparsity,
                                          'right_part_selector' : right_part_selector,
                                          'chromosome_fitness' : chromosome_fitness})
    return updater

class InitialParetoLevelSorting(CompoundOperator):
    key = 'InitialParetoLevelSorting'

    def apply(self, objective : ParetoLevels, arguments : dict):
        # See OffspringUpdater.apply: the history is read only by the
        # 'assignment' system right-part selector. A new initial population
        # never inherits the previous fit's mating sector (variant_pick='sector').
        from epde.operators.common.right_part_selection import system_history, set_mating_sector
        if len(objective.population) == 0:
            set_mating_sector(None)
        with system_history(objective.history):
            return self._sort(objective, arguments)

    @staticmethod
    def _grow_if_crowded(objective, idx, candidate, first_level, extension, pool_exhausted) -> None:
        """Past the last slot, while more than half the placed systems are on
        the first Pareto level and new systems can still be drawn, one more
        slot: a fresh random system (``SoEq.create``), drawn like any other."""
        if idx != len(objective.unplaced_candidates) - 1 or pool_exhausted or first_level.holds():
            return
        extra = copy.deepcopy(candidate)
        extra.create()
        objective.unplaced_candidates.append(extra)
        extension.add(id(extra))

    def _sort(self, objective : ParetoLevels, arguments : dict):
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
            # MOEA/D needs every popsize slot filled (one weight vector per
            # population member), but the token pool may yield fewer DISTINCT
            # non-degenerate forms than popsize. Policy: degeneracy is a HARD
            # reject (a LOSS_NAN_VAL form is never seeded); a duplicate is a SOFT
            # reject (prefer unique, but a non-degenerate duplicate is an
            # acceptable fill). Once a slot proves the unique pool is exhausted,
            # ``pool_exhausted`` drops the uniqueness retries for the remaining
            # slots so they fill immediately instead of each burning the full
            # budget. A true collapse (cannot find ANY non-degenerate form for a
            # slot) still stops the search.
            pool_exhausted = False
            init_collapsed = False
            # population_size='auto' (``ParetoLevels.population_floor`` set): once
            # at least that many systems are placed, the first slot that finds no
            # new one within the attempt limit ends the population instead -- it is
            # as large as the pool's distinct systems allow, up to the ceiling.
            floor = getattr(objective, 'population_floor', None)
            cut_at = None
            rps_op = self.suboperators['right_part_selector']
            fitness_op = self.suboperators['chromosome_fitness']
            rps_args = subop_args['right_part_selector']
            fitness_args = subop_args['chromosome_fitness']
            # At most half the initial population on the first Pareto level,
            # whatever the size asked for: past the last slot, fresh systems are
            # drawn while more than half are non-dominated (``extension``), and
            # an extension slot that finds no new system is dropped.
            requested = len(objective.unplaced_candidates)
            # evolution.niching='per_variable': at most half on level 0 of
            # every equation's own plane.
            from epde.optimizers.nsga.per_variable import PerPlaneFirstLevelCount, niching_per_variable
            first_level = (PerPlaneFirstLevelCount() if niching_per_variable(objective.unplaced_candidates)
                           else _FirstLevelCount())
            extension = set()
            # system_rps_mode='plain': a placed system's S+{1} versions that its
            # fit could not rank against S (``plain_fit.take_siblings``) take the
            # next slots, already fitted.
            from epde.operators.common.plain_fit import take_siblings
            fitted_siblings = set()

            for idx, candidate in enumerate(objective.unplaced_candidates):
                if pool_exhausted and floor is not None and idx >= floor:
                    # population_size='auto' whose pool ran dry below the floor:
                    # duplicates fill the slots up to the floor, not to the ceiling.
                    cut_at = idx
                    break
                if id(candidate) in fitted_siblings:
                    objective.history.add(candidate.equations_labels)
                    first_level.place(candidate)
                    self._grow_if_crowded(objective, idx, candidate, first_level, extension,
                                          pool_exhausted)
                    continue
                attempts = 0
                hit_cap = False
                while True:
                    candidate.reset_state(True)
                    # SoEqRightPartSelector resolves system degeneracy inline
                    # and leaves the pruned ACTIVE structure behind; the retry
                    # below additionally rejects fit-degenerate candidates and
                    # (while uniques remain) duplicate ACTIVE forms.
                    #
                    # THE DUPLICATE VERDICT IS READ BEFORE FITNESS. Only the
                    # degeneracy verdict needs the fit; a candidate whose
                    # post-RPS label is already in history gets re-rolled
                    # whatever the fit says, so scoring it is pure waste -- one
                    # discarded PDE solve per attempt on the solver path, with
                    # ``uniqueness_attempt_limit`` at 100. This mirrors the
                    # pre-fitness gate ``OffspringUpdater.apply`` already has.
                    scored = _rps_fitness_regenerate(
                        candidate, rps_op, fitness_op, rps_args, fitness_args,
                        skip_fitness_if=(
                            None if pool_exhausted else
                            lambda c: c.equations_labels in objective.history))
                    duplicate = candidate.equations_labels in objective.history
                    # An unscored candidate cannot READ as degenerate: the flag
                    # is the fitness host's LOSS_NAN_VAL stamp, and scoring was
                    # skipped precisely because ``duplicate`` already decided
                    # the retry.
                    degenerate = scored and has_degenerate_equation(candidate)
                    if not (degenerate or (duplicate and not pool_exhausted)):
                        break
                    if attempts >= uniqueness_attempt_limit and (
                            id(candidate) in extension or (floor is not None and idx >= floor)):
                        cut_at = idx
                        break
                    if attempts >= uniqueness_attempt_limit:
                        if not scored:
                            # This slot is about to be FILLED with the
                            # duplicate, so it has to carry real objective
                            # values -- and only the fit can say whether it is
                            # degenerate as well.
                            _rps_fitness_regenerate(candidate, rps_op, fitness_op,
                                                    rps_args, fitness_args)
                            degenerate = has_degenerate_equation(candidate)
                        if degenerate:
                            # No non-degenerate form found for this slot at all
                            # -> genuine search-space collapse.
                            hit_cap = True
                        else:
                            # Non-degenerate but only duplicates remain: the
                            # unique pool is exhausted. Accept this duplicate as
                            # a fill, and stop retrying uniqueness on later slots.
                            pool_exhausted = True
                        break
                    attempts += 1
                    candidate.create()
                if cut_at is not None:
                    _loop_stats.record('InitialParetoLevelSorting.unique_candidate.CUT',
                                       attempts, uniqueness_attempt_limit)
                    break
                system = candidate.equations_labels
                _loop_stats.record(
                    'InitialParetoLevelSorting.unique_candidate' + ('.FAIL' if hit_cap else ''),
                    attempts, uniqueness_attempt_limit,
                )
                if hit_cap:
                    # Search-space collapse: every one of
                    # ``uniqueness_attempt_limit`` consecutive create() outputs
                    # stayed fit-degenerate even after regeneration (each
                    # declined equation redrawn up to RESAMPLE_BUDGET times),
                    # so this slot has no valid (non-LOSS_NAN_VAL) form to seed. (Duplicates
                    # do NOT trigger this -- they are filled, see pool_exhausted.)
                    # Diagnostic builds want to observe the partial population's
                    # objective values rather than abort. Policy:
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
                # Fitness already computed inside the uniqueness/degeneracy loop;
                # ``system`` is the placed candidate's active structure.
                objective.history.add(system)
                first_level.place(candidate)
                if global_var.verbose.candidate_objectives:
                    print(candidate.obj_fun)
                inserted = 0
                for sibling in take_siblings(candidate):
                    if (sibling.equations_labels in objective.history
                            or has_degenerate_equation(sibling)):
                        continue
                    inserted += 1
                    objective.unplaced_candidates.insert(idx + inserted, sibling)
                    fitted_siblings.add(id(sibling))
                    # The sibling takes a slot a fresh draw would have had: the
                    # last one still waiting, if there is one.
                    waiting = [j for j in range(idx + inserted + 1, len(objective.unplaced_candidates))
                               if id(objective.unplaced_candidates[j]) not in fitted_siblings]
                    if waiting:
                        del objective.unplaced_candidates[waiting[-1]]
                self._grow_if_crowded(objective, idx, candidate, first_level, extension,
                                      pool_exhausted)
            if init_collapsed:
                if global_var.verbose.show_iter_idx:
                    print(f'\n*** Search-space collapse: stopping early with '
                          f'{len(objective.levels[0])} placed candidates. ***')
                return
            if cut_at is not None:
                allowed = len(objective.unplaced_candidates)
                objective.unplaced_candidates = list(objective.unplaced_candidates[:cut_at])
                # Below the floor a dry pool fills slots with duplicates, so the
                # slots are not all distinct systems: say how many are.
                distinct = len({c.equations_labels for c in objective.unplaced_candidates})
                print(f'Population size: {cut_at} systems, {distinct} distinct (the pool yields no '
                      f'further one in {uniqueness_attempt_limit} attempts; up to {allowed} allowed).')
            final = len(objective.unplaced_candidates)
            if final != requested:
                # One weight vector per member: the optimizer adopts the resized
                # weights after this call (MOEADDOptimizer._sync_weights).
                objective.resize_weights(final)
            if extension and first_level.holds():
                warnings.warn(
                    f'Initial population extended from {requested} to {final} systems so that at '
                    f'most half are on the first Pareto level ({first_level.size} of {final}).',
                    global_var.EPDEUsageWarning)
            elif not first_level.holds():
                warnings.warn(
                    f'Initial population: {first_level.size} of {final} systems are on the first '
                    'Pareto level, more than half, and the pool yields no further system.',
                    global_var.EPDEUsageWarning)
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
    per-equation list of bools flagging the two states that
    ``SoEqRightPartSelector`` resolves during RPS dispatch: an equation
    carrying another equation's target as a whole, standalone term
    (``_carries_target_of``), and an equation whose ACTIVE structure
    (target + nonzero-weight terms, ``Equation.active_terms_labels``)
    coincides with an EARLIER equation's -- the same law twice, rearranged.

    Other cross-equation sharing is NOT flagged: a common coupling term, or
    another equation's target as a FACTOR of a product (``v*dv/dx0`` in the
    u-equation), is legitimate. A correctly-implemented pipeline must produce
    an all-False result here. Use this in tests or temporary asserts to catch
    regressions; do NOT wire it back into the operator graph as a repair step.
    """
    # Local import: right_part_selection sits above this module's imports.
    from epde.operators.common.right_part_selection import _carries_target_of
    equations = list(objective.vals)
    sigs = [eq.active_terms_labels for eq in equations]
    flagged = [any(_carries_target_of(owner, carrier)
                   for owner in equations if owner is not carrier)
               for carrier in equations]

    for eq_idx in range(1, len(equations)):
        for other_idx in range(eq_idx):
            if sigs[eq_idx] and sigs[eq_idx] == sigs[other_idx]:
                flagged[eq_idx] = True
                break
    return flagged


def is_rps_in_other_equation(objective):
    """Deprecated alias. The post-hoc uniqueness repair has been replaced
    by ``SoEqRightPartSelector`` (system-degeneracy resolution: no equation
    carries another's target as a standalone term, and no two equations share
    an identical active structure), so this is now a
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
