#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 15:46:31 2021

@author: mike_ubuntu
"""

import numpy as np
from copy import deepcopy
from functools import partial
from typing import Union

from epde.optimizers.moeadd.moeadd import ParetoLevels

from epde.structure.main_structures import Equation, SoEq, Term
from epde.structure.structure_template import check_uniqueness
from epde.supplementary import filter_powers
from epde.operators.utils.template import CompoundOperator, add_base_param_to_operator


from epde.decorators import HistoryExtender, ResetEquationStatus


class SystemMutation(CompoundOperator):
    key = 'SystemMutation'
    def apply(self, objective : SoEq, arguments : dict): # TODO: add setter for best_individuals & worst individuals 
        self_args, subop_args = self.parse_suboperator_args(arguments = arguments)    
    
        altered_objective = deepcopy(objective)
        
        eqs_keys = altered_objective.vals.equation_keys; params_keys = altered_objective.vals.params_keys
        # eq_key = np.random.choice(eqs_keys)
        # altered_eq = self.suboperators['equation_mutation'].apply(altered_objective.vals[eq_key],
        #                                                           subop_args['equation_mutation'])
        for eq_key in eqs_keys:
            affected_by_mutation = np.random.random() < (self.params['indiv_mutation_prob'] / len(eq_key))
            if affected_by_mutation:
                altered_eq = self.suboperators['equation_mutation'].apply(altered_objective.vals[eq_key],
                                                                          subop_args['equation_mutation'])

            altered_objective.vals.replace_gene(gene_key = eq_key, value = altered_eq)

        for param_key in params_keys:
            altered_param = self.suboperators['param_mutation'].apply(altered_objective.vals[param_key],
                                                                      subop_args['param_mutation'])
            altered_objective.vals.replace_gene(gene_key = param_key, value = altered_param)
            altered_objective.vals.pass_parametric_gene(key = param_key, value = altered_param)

        return altered_objective

    def use_default_tags(self):
        self._tags = {'mutation', 'chromosome level', 'contains suboperators'}
    

class EquationMutation(CompoundOperator):
    key = 'EquationMutation'
    @HistoryExtender(f'\n -> mutating equation', 'ba')
    def apply(self, objective : Equation, arguments : dict):
        self_args, subop_args = self.parse_suboperator_args(arguments = arguments)

        # for term_idx in range(objective.n_immutable, len(objective.structure)):
        #     if np.random.uniform(0, 1) <= self.params['r_mutation']:
        #         objective.structure[term_idx] = self.suboperators['mutation'].apply(objective = (term_idx, objective),
        #                                                                             arguments = subop_args['mutation'])
        # nonzero_terms_mask = np.array([False if weight == 0 else True for weight in objective.weights_internal],
        #                               dtype=np.integer)
        # nonrs_terms_idx = [i for i, term in enumerate(objective.structure) if i != objective.target_idx]
        # nonzero_terms_idx = [item for item, keep in zip(nonrs_terms_idx, nonzero_terms_mask) if keep]
        # nonzero_terms_idx.append(objective.target_idx)
        # if len(nonzero_terms_idx) > 0:
        #     term_idx = np.random.choice(nonzero_terms_idx)
        # else:
        #     term_idx = objective.target_idx
        term_idx = np.random.choice(range(len(objective.structure)))
        objective.structure[term_idx] = self.suboperators['mutation'].apply(objective=(term_idx, objective),
                                                                            arguments=subop_args['mutation'])
        objective.structure[term_idx].reset_saved_state()
        return objective

    def use_default_tags(self):
        self._tags = {'mutation', 'gene level', 'contains suboperators'}


class MetaparameterMutation(CompoundOperator):
    key = 'MetaparameterMutation'

    def apply(self, objective : Union[int, float], arguments : dict):
        self_args, subop_args = self.parse_suboperator_args(arguments = arguments)

        altered_objective = np.random.normal(objective, objective)
        if altered_objective < 0:
            altered_objective = - altered_objective
        # if altered_objective > 1:
        #     altered_objective = 1

        # altered_objective = objective + np.random.randint(-1, 2)
        # if altered_objective < 1:
        #     altered_objective = 1
        # if altered_objective > 4:
        #     altered_objective = 4
        #
        # return altered_objective
        return np.float64(altered_objective)

    def use_default_tags(self):
        self._tags = {'mutation', 'gene level', 'no suboperators'}

    
class TermMutation(CompoundOperator):
    """
    Specific operator of the term mutation, where the term is replaced with a randomly created new one.
    """
    key = 'TermMutation'
    
    def apply(self, objective : tuple, arguments : dict): #term_idx, equation):
        """
        Return a new term, randomly created to be unique from other terms of this particular equation.
        
        Parameters:
        -----------
        term_idx : integer
            The index of the mutating term in the equation.
            
        equation : Equation object
            The equation object, in which the term is present.
        
        Returns:
        ----------
        new_term : Term object
            A new, randomly created, term.
            
        """       
        self_args, subop_args = self.parse_suboperator_args(arguments = arguments)

        temp = deepcopy(objective[1].structure[objective[0]])
        objective[1].structure[objective[0]].randomize()
        objective[1].structure[objective[0]].reset_saved_state()
        while (len(objective[1].described_variables_full) != len(objective[1].structure)
               or objective[1].structure[objective[0]].described_variables_full == temp.described_variables_full):
            objective[1].structure[objective[0]].randomize()
            objective[1].structure[objective[0]].reset_saved_state()
        # print(f'CREATED DURING MUTATION: {new_term.name}, while contatining {objective[1].structure[objective[0]].descr_variable_marker}')
        return objective[1].structure[objective[0]]

    def use_default_tags(self):
        self._tags = {'mutation', 'term level', 'exploration', 'no suboperators'}


class TermParameterMutation(CompoundOperator):
    """
    Specific operator of the term mutation, where the term parameters are changed with a random increment.
    """
    key = 'TermParameterMutation'    
    
    def apply(self, objective : tuple, arguments : dict): # term_idx, objective
        """ 
        Specific operator of the term mutation, where the term parameters are changed with a random increment.
        
        Parameters:
        -----------
        term_idx : integer
            The index of the mutating term in the equation.
            
        equation : Equation object
            The equation object, in which the term is present.
        
        Returns:
        ----------
        new_term : Term object
            The new, created from the previous one with random parameters increment, term.
            
        """
        self_args, subop_args = self.parse_suboperator_args(arguments = arguments)
        
        unmutable_params = {'dim', 'power'}
        # objective[1] = deepcopy(objective[1])
        while True:
            # Костыль!
            print('ENTERING LOOP')
            try:
                objective[1].target_idx
            except AttributeError:
                objective[1].target_idx = 0
            #
            term = objective[1].structure[objective[0]] 
            for factor in term.structure:
                if objective[0] == objective[1].target_idx:
                    continue
                # if objective[0] < altered_objective.target_idx:
                #     corresponding_weight = altered_objective.weights_internal[objective[0]] 
                # else:
                #     corresponding_weight = altered_objective.weights_internal[objective[0] - 1]
                # if corresponding_weight == 0:                
                parameter_selection = deepcopy(factor.params)
                for param_idx, param_properties in factor.params_description.items():
                    if np.random.random() < self.params['r_param_mutation'] and param_properties['name'] not in unmutable_params:
                        interval = param_properties['bounds']
                        if interval[0] == interval[1]:
                            shift = 0
                            continue
                        if isinstance(interval[0], int):
                            shift = np.rint(np.random.normal(loc= 0, scale = self.params['multiplier']*(interval[1] - interval[0]))).astype(int) #
                        elif isinstance(interval[0], float):
                            shift = np.random.normal(loc= 0, scale = self.params['multiplier']*(interval[1] - interval[0]))
                        else:
                            raise ValueError('In current version of framework only integer and real values for parameters are supported') 
                        if self.params['strict_restrictions']:
                            parameter_selection[param_idx] = np.min((np.max((parameter_selection[param_idx] + shift, interval[0])), interval[1]))
                        else:
                            parameter_selection[param_idx] = parameter_selection[param_idx] + shift
                    factor.params = parameter_selection
            term.structure = filter_powers(term.structure)
            print(f'checking presence of {term.name} as {objective[0]}-th element in {objective[1].text_form}')
            # if check_uniqueness(term, objective[1].structure[:objective[0]] +
            #                     objective[1].structure[objective[0]+1:]):
            if len(objective[1].described_variables_full) == len(objective[1].structure):
                break
        term.reset_saved_state()
        return term
    
    def use_default_tags(self):
        self._tags = {'mutation', 'term level', 'exploitation', 'no suboperators'}


def get_basic_mutation(mutation_params):
    add_kwarg_to_operator = partial(add_base_param_to_operator, target_dict = mutation_params)

    term_mutation = TermMutation([])

    equation_mutation = EquationMutation(['r_mutation', 'type_probabilities'])
    add_kwarg_to_operator(operator = equation_mutation)
    
    metaparameter_mutation = MetaparameterMutation(['std', 'mean'])
    add_kwarg_to_operator(operator = metaparameter_mutation)

    chromosome_mutation = SystemMutation(['indiv_mutation_prob'])
    add_kwarg_to_operator(operator = chromosome_mutation)

    equation_mutation.set_suboperators(operators = {'mutation' : term_mutation})#, [term_param_mutation, ]
                                       # probas = {'equation_crossover' : [0.0, 1.0]})

    chromosome_mutation.set_suboperators(operators = {'equation_mutation' : equation_mutation, 
                                                      'param_mutation' : metaparameter_mutation})
    return chromosome_mutation
