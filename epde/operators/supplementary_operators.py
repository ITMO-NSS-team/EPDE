#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 29 19:08:51 2022

@author: maslyaev
"""
import copy
import numpy as np
from functools import reduce

from epde.moeadd.moeadd_stc import Constraint
from epde.operators.template import CompoundOperator


class SimpleNeighborSelector(CompoundOperator):
    def apply(self, neighbors):
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
        return neighbors[:self.params['number_of_neighbors']]


class PopulationUpdater(CompoundOperator):
    def apply(self, offspring, pareto_levels, PBI_penalty):
        '''
        Update population to get the pareto-nondomiated levels with the worst element removed. 
        Here, "worst" means the solution with highest PBI value (penalty-based boundary intersection)
        '''         
        pareto_levels.update(offspring)  #levels_updated = ndl_update(offspring, levels)
        if len(pareto_levels.levels) == 1:
            worst_solution = self.suboperators['locate_pareto_worst'].apply(pareto_levels, self.weights, 
                                                                            self.best_obj, self.params['PBI_penalty'])
        else:
            if pareto_levels.levels[len(pareto_levels.levels) - 1] == 1:
                domain_solutions = self.suboperators['population_to_sectors'].apply(pareto_levels.population, 
                                                                                    self.weights)
                reference_solution = pareto_levels.levels[len(pareto_levels.levels) - 1][0]
                reference_solution_domain = [idx for idx in np.arange(domain_solutions) if reference_solution in domain_solutions[idx]]
                if len(domain_solutions[reference_solution_domain] == 1):
                    worst_solution = self.suboperators['locate_pareto_worst'].apply(pareto_levels.levels, self.weights,
                                                                                    self.best_obj, self.params['PBI_penalty'])                            
                else:
                    worst_solution = reference_solution
            else:
                last_level_by_domains = self.suboperators['population_to_sectors'].apply(pareto_levels.levels[len(pareto_levels.levels)-1], 
                                                                                         self.weights)
                most_crowded_count = np.max([len(domain) for domain in last_level_by_domains]); 
                crowded_domains = [domain_idx for domain_idx in np.arange(len(self.weights)) if len(last_level_by_domains[domain_idx]) == most_crowded_count]
        
                if len(crowded_domains) == 1:
                    most_crowded_domain = crowded_domains[0]
                else:
                    PBI = lambda domain_idx: np.sum([self.suboperators.apply['penalty_based_intersection'].apply(sol_obj, self.weights[domain_idx], 
                                                                                                                 self.best_obj, PBI_penalty) 
                                                     for sol_obj in last_level_by_domains[domain_idx]])
                    PBIS = np.fromiter(map(PBI, crowded_domains), dtype = float)
                    most_crowded_domain = crowded_domains[np.argmax(PBIS)]
                    
                if len(last_level_by_domains[most_crowded_domain]) == 1:
                    worst_solution = self.suboperators['locate_pareto_worst'].apply(self.pareto_levels, self.weights, 
                                                                                    self.best_obj, self.params['PBI_penalty'])
                else:
                    PBIS = np.fromiter(map(lambda solution: self.suboperators.apply['penalty_based_intersection'].apply(solution, self.weights[most_crowded_domain], 
                                                                                                                        self.best_obj, PBI_penalty),
                                               last_level_by_domains[most_crowded_domain]), dtype = float)
                    worst_solution = last_level_by_domains[most_crowded_domain][np.argmax(PBIS)]                    
        
        pareto_levels.delete_point(worst_solution)


class PopulationUpdaterConstrained(object):
    def apply(self, offspring, pareto_levels, PBI_penalty):
        '''
        
        Update population to get the pareto-nondomiated levels with the worst element removed. 
        Here, "worst" means the solution with highest PBI value (penalty-based boundary intersection). 
        Additionally, the constraint violations are considered in the selection of the 
        "worst" individual.
        
        '''        
        pareto_levels.update(offspring)
        cv_values = self.suboperators['constaint_violation'].apply(pareto_levels)
        
        if sum(cv_values) == 0:
            if len(pareto_levels.levels) == 1:
                worst_solution = self.suboperators['locate_pareto_worst'].apply(pareto_levels, self.weights, 
                                                                                self.best_obj, self.params['PBI_penalty'])
            else:
                if pareto_levels.levels[len(pareto_levels.levels) - 1] == 1:
                    domain_solutions = self.suboperators['population_to_sectors'].apply(pareto_levels.population, 
                                                                                        self.weights)
                    reference_solution = pareto_levels.levels[len(pareto_levels.levels) - 1][0]
                    reference_solution_domain = [idx for idx in np.arange(domain_solutions) if reference_solution in domain_solutions[idx]]
                    if len(domain_solutions[reference_solution_domain] == 1):
                        worst_solution = self.suboperators['locate_pareto_worst'].apply(pareto_levels.levels, 
                                                                                        self.weights, self.best_obj, 
                                                                                        self.params['PBI_penalty'])
                    else:
                        worst_solution = reference_solution
                else:
                    last_level_by_domains = self.suboperators['population_to_sectors'].apply(pareto_levels.levels[len(pareto_levels.levels)-1], 
                                                                                             self.weights)
                    most_crowded_count = np.max([len(domain) for domain in last_level_by_domains]); 
                    crowded_domains = [domain_idx for domain_idx in np.arange(len(self.weights)) 
                                       if len(last_level_by_domains[domain_idx]) == most_crowded_count]
    
                    if len(crowded_domains) == 1:
                        most_crowded_domain = crowded_domains[0]
                    else:
                        PBI = lambda domain_idx: np.sum([self.suboperators['penalty_based_intersection'].apply(sol_obj, self.weights[domain_idx], 
                                                                                                               self.best_obj, self.params['PBI_penalty']) 
                                                            for sol_obj in last_level_by_domains[domain_idx]])
                        PBIS = np.fromiter(map(PBI, crowded_domains), dtype = float)
                        most_crowded_domain = crowded_domains[np.argmax(PBIS)]
                        
                    if len(last_level_by_domains[most_crowded_domain]) == 1:
                        worst_solution = self.suboperators['locate_pareto_worst'].apply(pareto_levels, self.weights, 
                                                                                        self.best_obj, self.params['PBI_penalty'])
                    else:
                        PBIS = np.fromiter(map(lambda solution: self.suboperators['population_to_sectors'].apply(solution, self.weights[most_crowded_domain], 
                                                                                                                 self.best_obj, self.params['PBI_penalty']), 
                                               last_level_by_domains[most_crowded_domain]), dtype = float)
                        worst_solution = last_level_by_domains[most_crowded_domain][np.argmax(PBIS)]                    
        else:
            infeasible = [solution for solution, _ in sorted(list(zip(self.pareto_levels.population, cv_values)), key = lambda pair: pair[1])]
            infeasible.reverse()
            infeasible = infeasible[:np.nonzero(cv_values)[0].size]
            deleted = False
            domain_solutions = self.suboperators['population_to_sectors'].apply(pareto_levels.population, 
                                                                                self.weights)
            
            for infeasable_element in infeasible:
                domain_idx = [domain_idx for domain_idx, domain in enumerate(domain_solutions) if infeasable_element in domain][0]
                if len(domain_solutions[domain_idx]) > 1:
                    deleted = True
                    worst_solution = infeasable_element
                    break
            if not deleted:
                worst_solution = infeasible[0]

        pareto_levels.delete_point(worst_solution)
        return pareto_levels


class ConstraintViolationCalculator(object):
    def __init__(self, constraints = None, param_keys = []):
        assert all(isinstance(constraints, Constraint) for constr in constraints)
        self._constraints = constraints
        super().init(param_keys)
    
    def apply(self, pareto_levels, indexes = None):
        def violation(individual):
            return reduce(lambda y, z: y + z(individual.vals()), self._constraints, initial = 0)

        if indexes is None:
            indexes = np.arange(len(pareto_levels.population))
        constraint_violations = np.array(list(map(lambda x: violation(pareto_levels.population[x]), indexes)))
        
        return constraint_violations
    
    def use_default_tags(self):
        self._tags = {'constraint violation calculation', 'mixed input level', 'auxilary'}        