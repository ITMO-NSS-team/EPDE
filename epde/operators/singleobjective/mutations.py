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
from epde.supplementary import filter_powers, try_iterable
from epde.operators.utils.template import CompoundOperator, add_base_param_to_operator


from epde.decorators import HistoryExtender, ResetEquationStatus


class SystemMutation(CompoundOperator):
    key = 'SystemMutation'

    def apply(self, objective : SoEq, arguments : dict): # TODO: add setter for best_individuals & worst individuals 
        self_args, subop_args = self.parse_suboperator_args(arguments = arguments)    

        altered_objective = deepcopy(objective)
        if objective.elite == 'immutable':
            return altered_objective
        
        eqs_keys = altered_objective.vals.equation_keys; params_keys = altered_objective.vals.params_keys
        affected_by_mutation = True

        if affected_by_mutation:
            for eq_key in eqs_keys:
                altered_eq = self.suboperators['equation_mutation'].apply(altered_objective.vals[eq_key],
                                                                          subop_args['equation_mutation'])

                altered_objective.vals.replace_gene(gene_key = eq_key, value = altered_eq)

            for param_key in params_keys:
                altered_param = self.suboperators['param_mutation'].apply(altered_objective.vals[param_key],
                                                                          subop_args['param_mutation'])
                altered_objective.vals.replace_gene(gene_key = param_key, value = altered_param)
                altered_objective.vals.pass_parametric_gene(key = param_key, value = altered_param)
        
        altered_objective.reset_state() # Использовать ли reset_right_part
        return altered_objective

    def use_default_tags(self):
        self._tags = {'mutation', 'chromosome level', 'contains suboperators', 'standard'}
    

class EquationMutation(CompoundOperator):
    key = 'EquationMutation'

    @HistoryExtender(f'\n -> mutating equation', 'ba')
    def apply(self, objective : Equation, arguments : dict):
        self_args, subop_args = self.parse_suboperator_args(arguments = arguments)  
        for term_idx in range(objective.n_immutable, len(objective.structure)):
            if np.random.uniform(0, 1) <= self.params['r_mutation']:
                objective.structure[term_idx] = self.suboperators['mutation'].apply(objective = (term_idx, objective),
                                                                                    arguments = subop_args['mutation'])
        return objective

    def use_default_tags(self):
        self._tags = {'mutation', 'gene level', 'contains suboperators', 'standard'}


class MetaparameterMutation(CompoundOperator):
    key = 'MetaparameterMutation'

    def apply(self, objective : Union[int, float], arguments : dict):
        self_args, subop_args = self.parse_suboperator_args(arguments = arguments)

        # print('objective', objective)
        
        altered_objective = objective + np.random.normal(loc = self.params['mean'], scale = self.params['std'])        
        if altered_objective < 0:
            altered_objective = - altered_objective
        
        return altered_objective

    def use_default_tags(self):
        self._tags = {'mutation', 'gene level', 'no suboperators', 'standard'}

    
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
        
        create_derivs = bool(objective[1].structure[objective[0]].descr_variable_marker)
        
        new_term = Term(objective[1].pool, mandatory_family = objective[1].structure[objective[0]].descr_variable_marker, 
                        create_derivs=create_derivs,
                        max_factors_in_term = objective[1].metaparameters['max_factors_in_term']['value'])
        while not check_uniqueness(new_term, objective[1].structure[:objective[0]] + objective[1].structure[objective[0]+1:]):
            new_term = Term(objective[1].pool, mandatory_family = objective[1].structure[objective[0]].descr_variable_marker, 
                            create_derivs=create_derivs,
                            max_factors_in_term = objective[1].metaparameters['max_factors_in_term']['value'])
        new_term.use_cache()
        # print(f'CREATED DURING MUTATION: {new_term.name}, while contatining {objective[1].structure[objective[0]].descr_variable_marker}')
        return new_term

    def use_default_tags(self):
        self._tags = {'mutation', 'term level', 'exploration', 'no suboperators', 'standard'}


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
            if check_uniqueness(term, objective[1].structure[:objective[0]] + 
                                objective[1].structure[objective[0]+1:]):
                break
        term.reset_saved_state()
        return term
    
    def use_default_tags(self):
        self._tags = {'mutation', 'term level', 'exploitation', 'no suboperators', 'standard'}

# TODO: reorganize mutation and similar operators into the blocks of "common" operators.

def get_multiobjective_mutation(mutation_params): # TODO: rename function calls where necessary
    # TODO: generalize initiation with test runs and simultaneous parameter and object initiation.
    add_kwarg_to_operator = partial(add_base_param_to_operator, target_dict = mutation_params)    

    term_mutation = TermMutation([])

    equation_mutation = EquationMutation(['r_mutation', 'type_probabilities'])
    add_kwarg_to_operator(operator = equation_mutation)
    
    metaparameter_mutation = MetaparameterMutation(['std', 'mean'])
    add_kwarg_to_operator(operator = metaparameter_mutation)

    chromosome_mutation = SystemMutation(['indiv_mutation_prob'])

    equation_mutation.set_suboperators(operators = {'mutation' : term_mutation})

    chromosome_mutation.set_suboperators(operators = {'equation_mutation' : equation_mutation, 
                                                      'param_mutation' : metaparameter_mutation})
    return chromosome_mutation


def get_singleobjective_mutation(mutation_params):
    # TODO: generalize initiation with test runs and simultaneous parameter and object initiation.
    add_kwarg_to_operator = partial(add_base_param_to_operator, target_dict = mutation_params)    

    term_mutation = TermMutation([])

    equation_mutation = EquationMutation(['r_mutation', 'type_probabilities'])
    add_kwarg_to_operator(operator = equation_mutation)
    
    chromosome_mutation = SystemMutation(['indiv_mutation_prob'])
    
    equation_mutation.set_suboperators(operators = {'mutation' : term_mutation})

    chromosome_mutation.set_suboperators(operators = {'equation_mutation' : equation_mutation})
    return chromosome_mutation    