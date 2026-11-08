#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  4 13:49:36 2021

@author: mike_ubuntu
"""

import numpy as np
from functools import reduce

from epde.optimizers.moeadd.moeadd import ParetoLevels
from epde.optimizers.moeadd.supplementary import Constraint

from epde.operators.utils.template import CompoundOperator
from epde import _loop_stats


def _plane_domains(population, eq_idx, weights, normalizer):
    """Each system's subregion in equation ``eq_idx``'s own plane: MOEA/DD
    Eq. (6) on that equation's objective values alone, on the same scale the
    rest of the optimizer uses."""
    from epde.optimizers.moeadd.solution_template import get_domain_idx
    n_eq = len(population[0].vals)
    domains = []
    for system in population:
        obj = np.asarray(system.obj_fun, dtype=float)
        if normalizer is not None:
            obj = normalizer(obj)
        n_types = obj.size // n_eq
        plane = obj[[t * n_eq + eq_idx for t in range(n_types)]]
        domains.append(get_domain_idx(plane, weights))
    return domains


def _complementary_partner(population, parent_idxs):
    """``evolution.complementary_mating_prob``: with that probability the second
    parent is redrawn from the systems holding some equation's own front where
    the first parent does not, so a crossover can bring, e.g., an exact u and
    an exact v together. ``0`` (the default) draws nothing.

    Returns ``(parent_idxs, pair)``; ``pair`` holds the two forced mates'
    indices, which the crossover pairs with each other before it shuffles the
    rest of the pool (``ParetoLevels.forced_pair``), or None."""
    from epde.interface.search_config import active_config
    prob = active_config().evolution.complementary_mating_prob
    if not prob or len(population) < 2 or len(population[0].vals) < 2 or parent_idxs.size < 2:
        return parent_idxs, None
    if np.random.uniform() >= prob:
        return parent_idxs, None
    from epde.operators.multiobjective.moeadd_specific import equation_front_holders
    holders = equation_front_holders(population)
    first = int(parent_idxs.reshape(-1)[0])
    partners = [i for i in range(len(population)) if i != first
                and any(i in held and first not in held for held in holders)]
    if not partners:
        _loop_stats.record('Mating.no_complement', 1, 1)
        return parent_idxs, None
    _loop_stats.record('Mating.complement', 1, 1)
    parent_idxs = np.array(parent_idxs, copy=True)
    partner = int(np.random.choice(partners))
    parent_idxs.reshape(-1)[1] = partner
    return parent_idxs, (first, partner)


class MOEADDSelection(CompoundOperator):
    key = 'MOEADDSelection'
    
    def apply(self, objective : ParetoLevels, arguments : dict): # pareto_levels
        r'''
        
        The mating operator, designed to select parents for the crossover with respect 
        to the location of the point in the objective functions values space and the 
        connected weight vector.
        
        Parameters:
        ------------
        
        weight_idx : int,
            Index of the processed weight vector.
            
        weights : np.ndarray,        
            Numpy array, containing weight vectors.
            
        neighborhood_vectors : list,
            List of lists, containing indexes: i-th element is the list of 
            k - closest to the i-the weight vector weight vectors.
            
        population : list,
            List of candidate solutions.
            
        neighborhood_selector : function,
            Method of finding "close neighbors" of the vector with proximity list.
            The baseline example of the selector, presented in 
            ``moeadd.moeadd_stc.simple_selector``, selects n-adjacent ones.
            
        delta : float
            The probability of mating selection to be limited only to the selected
            subregions (adjacent to the weight vector domain). :math:`\delta \in [0., 1.)`
            
        Returns:
        ---------
        
        parent_idxs : list
            List of the selected parents in the population pool.
        
        '''
        self_args, subop_args = self.parse_suboperator_args(arguments = arguments)
        
        parents_number_counted = int(len(objective.population) * self.params['parents_fraction']) # Странное упрощение 
        parents_number_counted = parents_number_counted if not parents_number_counted % 2 else parents_number_counted + 1
        
        parents_number = min(max(2, parents_number_counted), len(objective.population))
                             
        from epde.operators.multiobjective.moeadd_specific import niching_per_equation
        from epde.operators.common.right_part_selection import set_mating_sector
        per_equation = niching_per_equation(objective.population)
        # The sector the next offspring belong to, for variant_pick='sector'
        # (read by the system right-part selector; no random draw here).
        set_mating_sector(self_args['weights'][self_args['weight_idx']], objective.normalizer,
                          self_args['weight_idx'] % len(objective.population[0].vals)
                          if per_equation else None)
        if np.random.uniform() < self.params['delta']:
            selected_regions_idxs = self.suboperators['neighborhood_selector'].apply(self_args['neighborhood_vectors'][self_args['weight_idx']],
                                                                                     arguments = subop_args['neighborhood_selector'])

            if per_equation:
                # niching='per_equation': weight vector k serves equation
                # k mod n, in that equation's own (discrepancy, instability)
                # plane, so every equation gets its own spread of directions.
                candidate_solution_domains = _plane_domains(
                    objective.population, self_args['weight_idx'] % len(objective.population[0].vals),
                    self_args['weights'], objective.normalizer)
            else:
                candidate_solution_domains = [candidate.get_domain(self_args['weights'])
                                              for candidate in objective.population]
            solution_mask = [domain in selected_regions_idxs for domain in candidate_solution_domains]
            available_in_proximity = sum(solution_mask)
            if available_in_proximity == 0:
                # Algorithm 3, lines 3-4 of the MOEA/DD paper: no solution is
                # associated with the selected subregions -- mate from the
                # whole population instead.
                parent_idxs = np.random.choice(np.arange(len(objective.population)),
                                               size = parents_number, replace = False)
            else:
                # Algorithm 3, line 6: parents come exclusively from the
                # selected subregions. When they hold fewer solutions than
                # the requested number of parents, sample with replacement
                # to preserve the parent count.
                parent_idxs = np.random.choice([idx for idx in np.arange(len(objective.population)) if solution_mask[idx]],
                                               size = parents_number,
                                               replace = available_in_proximity < parents_number)
        else:
            parent_idxs = np.random.choice(np.arange(len(objective.population)), size = parents_number, replace = False)
        parent_idxs, pair = _complementary_partner(objective.population, parent_idxs)
        # The crossover pairs these two before shuffling the rest of its pool.
        objective.forced_pair = (None if pair is None else
                                 (objective.population[pair[0]], objective.population[pair[1]]))
        for idx in parent_idxs.reshape(-1):
            objective.population[int(idx)].incr_counter()
        return objective
    
    @property
    def arguments(self):
        return set(['neighborhood_vectors', 'weight_idx', 'weights'])    
    
    def use_default_tags(self):
        self._tags = {'selection', 'population level', 'auxilary', 'suboperators', 'standard'}


class MOEADDSelectionConstrained(CompoundOperator):
    key = 'MOEADDSelectionConstrained'
    
    def apply(self, objective : ParetoLevels, arguments : dict):
        r'''
        
        The mating operator, designed to select parents for the crossover with respect 
        to the location of the point in the objective functions values space and the 
        connected weight vector.
        
        Parameters:
        ------------
        
        weight_idx : int,
            Index of the processed weight vector.
            
        weights : np.ndarray,        
            Numpy array, containing weight vectors.
            
        neighborhood_vectors : list,
            List of lists, containing indexes: i-th element is the list of 
            k - closest to the i-the weight vector weight vectors.
            
        population : list,
            List of candidate solutions.
            
        neighborhood_selector : function,
            Method of finding "close neighbors" of the vector with proximity list.
            The baseline example of the selector, presented in 
            ``moeadd.moeadd_stc.simple_selector``, selects n-adjacent ones.
            
        delta : float
            The probability of mating selection to be limited only to the selected
            subregions (adjacent to the weight vector domain). :math:`\delta \in [0., 1.)`
            
        Returns:
        ---------
        
        parent_idxs : list
            List of the selected parents in the population pool.
        
        '''
        self_args, subop_args = self.parse_suboperator_args(arguments = arguments)
    
        if 'constraint_processer' in list(self.suboperators.keys()):
            multiplier = self.suboperators['constraint_processer'].params['group_size']
            
        parents_number = int(len(objective.population) * self.params['parents_fraction'] * multiplier)
        if np.random.uniform() < self.params['delta']:
            selected_regions_idxs = self.suboperators['neighborhood_selector'].apply(self_args['neighborhood_vectors'][self_args['weight_idx']])
            candidate_solution_domains = list(map(lambda x: x.get_domain(self_args['weights']), [candidate for candidate in 
                                                                                                 objective.population]))
    
            solution_mask = [(objective.population[solution_idx].get_domain(self_args['weights']) in selected_regions_idxs) 
                             for solution_idx in candidate_solution_domains]
            available_in_proximity = sum(solution_mask)
            parent_idxs = np.random.choice([idx for idx in np.arange(len(objective.population)) if solution_mask[idx]], 
                                            size = min(available_in_proximity, parents_number), 
                                            replace = False)
            if available_in_proximity < parents_number:
                parent_idxs_additional = np.random.choice([idx for idx in np.arange(len(objective.population))
                                                           if not solution_mask[idx]],
                                                          size = parents_number - available_in_proximity,
                                                          replace = False)
                parent_idxs_temp = np.empty(shape = parent_idxs.size + parent_idxs_additional.size)
                parent_idxs_temp[:parent_idxs.size] = parent_idxs; parent_idxs_temp[parent_idxs.size:] = parent_idxs_additional
                parent_idxs = parent_idxs_temp
        else:
            parent_idxs = np.random.choice(np.arange(len(objective.population)), size = parents_number, replace = False)

        if 'constraint_processer' in list(self.suboperators.keys()):
            assert 'constraints' in self.suboperators['constraint_processer'].tags
            parent_idxs = self.suboperators['constraint_processer'].apply(parent_idxs, objective)

        for idx in parent_idxs:
            objective.population[int(idx)].incr_counter()
        return objective
    
    @property
    def arguments(self):
        return set(['neighborhood_vectors', 'weight_idx', 'weights'])    

    def use_default_tags(self):
        self._tags = {'selection', 'population level', 'auxilary', 'suboperators', 'standard'}


class SelectionConstraintProcesser(object):
    key = 'SelectionConstraintProcesser'
    
    def __init__(self, constraints = None, param_keys = []):
        assert all(isinstance(constraints, Constraint) for constr in constraints)
        self._constraints = constraints
        super().init(param_keys)
    
    def apply(self, indexes, pareto_levels):
        def violation(individual):
            return reduce(lambda y, z: y + z(individual.vals()), self._constraints, initial = 0)

        constraint_violations = {idx : violation(pareto_levels.population[idx]) for idx in indexes}
        selected_idxs = []
        for idx in np.arange(int(indexes/self.params['group_size'])):
            first_elem_idx = idx * self.params['group_size']
            if all([constraint_violations[indexes[first_elem_idx]] == constraint_violations[indexes[first_elem_idx + incr]] 
                    for incr in range(self.params['group_size'])]):
                selected_idxs.append(np.random.choice(a = indexes[first_elem_idx : first_elem_idx + self.params['group_size']]))
        if len(selected_idxs) % 2: selected_idxs = selected_idxs[:-1]
        np.random.shuffle(selected_idxs)
        return selected_idxs
    
    def use_default_tags(self):
        self._tags = {'constraints', 'selection', 'custom level', 'auxilary'}
