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
    """
    Applies mutation suboperators to an objective function.
    
        This class orchestrates the application of mutation suboperators to a
        given objective function, facilitating the evolution of the system.
    """

    key = 'SystemMutation'
    def apply(self, objective : SoEq, arguments : dict): # TODO: add setter for best_individuals & worst individuals 
        self_args, subop_args = self.parse_suboperator_args(arguments = arguments)    
    
        altered_objective = deepcopy(objective)
        
        eqs_keys = altered_objective.vals.equation_keys; params_keys = altered_objective.vals.params_keys
        affected_by_mutation = np.random.random() < self.params['indiv_mutation_prob']

        if affected_by_mutation:
        """
        Applies equation and parameter mutation suboperators to an objective function.
        
                This method orchestrates the mutation of an objective function by applying
                equation and parameter mutation suboperators. The mutation is applied probabilistically,
                allowing for exploration of the search space by introducing variations into the
                objective function's equations and parameters. This is a crucial step in the evolutionary
                process, enabling the algorithm to discover new and potentially better-fitting equation structures.
        
                Args:
                    objective (SoEq): The objective function to be mutated.
                    arguments (dict): A dictionary of arguments for the suboperators.
        
                Returns:
                    SoEq: The mutated objective function.
        """
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
        """
        Uses a predefined set of tags to categorize the mutation operator.
        
                This method overwrites any existing tags, ensuring the operator is correctly identified for downstream processing and analysis within the EPDE framework. This ensures consistent categorization and facilitates proper handling during the equation discovery process.
        
                Args:
                    self: The object instance.
        
                Returns:
                    None.
        
                Class Fields:
                    _tags (set): A set containing the tags 'mutation', 'chromosome level', and 'contains suboperators'.
        """
        self._tags = {'mutation', 'chromosome level', 'contains suboperators'}
    

class EquationMutation(CompoundOperator):
    """
    Applies mutation to the equation's structure based on a mutation rate.
    
        Class Attributes:
        - mutation_rate: The probability of applying a mutation sub-operator to a term.
        - sub_operators: A list of mutation sub-operators to apply.
    """

    key = 'EquationMutation'
    @HistoryExtender(f'\n -> mutating equation', 'ba')
    def apply(self, objective : Equation, arguments : dict):
        """
        Applies mutation to the equation's structure based on a mutation rate.
        
                It iterates through the mutable terms of the equation's structure,
                and for each term, it applies a mutation sub-operator with a certain probability
                defined by the mutation rate. This process refines the equation's structure
                to better fit the observed data by exploring alternative mathematical
                relationships.
        
                Args:
                    objective (Equation): The equation to be mutated.
                    arguments (dict): A dictionary of arguments for the sub-operators.
        
                Returns:
                    Equation: The mutated equation.
        """
        self_args, subop_args = self.parse_suboperator_args(arguments = arguments)

        for term_idx in range(objective.n_immutable, len(objective.structure)):
            if np.random.uniform(0, 1) <= self.params['r_mutation']:
                objective.structure[term_idx] = self.suboperators['mutation'].apply(objective = (term_idx, objective),
                                                                                    arguments = subop_args['mutation'])
        return objective

    def use_default_tags(self):
        """
        Applies a predefined set of tags to categorize and characterize the equation mutation operation.
        
        This method assigns default tags to the mutation, providing a standardized way to identify and group mutations based on their characteristics. This facilitates analysis and comparison of different mutation strategies within the evolutionary process.
        
        Args:
            self: The EquationMutation object instance.
        
        Returns:
            None.
        
        This method initializes the following object properties:
          - _tags (set): A set containing the default tags: 'mutation', 'gene level', and 'contains suboperators'.
        """
        self._tags = {'mutation', 'gene level', 'contains suboperators'}


class MetaparameterMutation(CompoundOperator):
    """
    Represents a mutation operation that modifies metaparameters.
    
        This class provides a base for implementing mutation strategies that
        alter the values of metaparameters within a search space.
    
        Attributes:
            params: A dictionary containing parameters for the mutation.
    """

    key = 'MetaparameterMutation'

    def apply(self, objective : Union[int, float], arguments : dict):
        """
        Applies a random perturbation to the objective value.
        
                This method introduces slight variations to the objective value, simulating the inherent noise and uncertainty present in real-world data. By adding a normally distributed random value (defined by the `mean` and `std` parameters), the method explores the solution space more broadly, potentially escaping local optima and promoting the discovery of more robust and generalizable equation structures. The non-negativity constraint ensures that the objective value remains physically meaningful within the context of equation discovery.
        
                Args:
                    objective (Union[int, float]): The original objective value.
                    arguments (dict): A dictionary of arguments that can be used to configure the perturbation.
        
                Returns:
                    float: The perturbed objective value, guaranteed to be non-negative.
        """
        self_args, subop_args = self.parse_suboperator_args(arguments = arguments)

        altered_objective = objective + np.random.normal(loc = self.params['mean'], scale = self.params['std'])        
        if altered_objective < 0:
            altered_objective = - altered_objective
        
        return altered_objective

    def use_default_tags(self):
        """
        Sets the object's tags to a predefined set, ensuring consistency in identifying the type and characteristics of the mutation being applied. This standardization is crucial for the evolutionary process, allowing the system to effectively track and manage different mutation strategies.
        
                Args:
                    self: The object instance.
        
                Returns:
                    None.
        """
        self._tags = {'mutation', 'gene level', 'no suboperators'}

    
class TermMutation(CompoundOperator):
    """
    Specific operator of the term mutation, where the term is replaced with a randomly created new one.
    """

    key = 'TermMutation'
    
    def apply(self, objective : tuple, arguments : dict): #term_idx, equation):
        """
        Return a new, randomly generated term to replace an existing one within an equation.
        
                This ensures diversity in the equation population during the evolutionary search process. The new term is created based on the equation's pool of available operators and variables, and it is checked for uniqueness to maintain the integrity of the equation structure.
        
                Args:
                    objective (tuple): A tuple containing the index of the term to be mutated and the equation object.
                    arguments (dict): A dictionary containing additional arguments (not directly used in this method, but potentially used in sub-methods).
        
                Returns:
                    Term: A new, randomly created term that is unique within the equation.
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
        """
        Sets the default tags for this term mutation operator. These tags categorize the operator's behavior and capabilities within the equation discovery process.
        
                Args:
                    self: The object instance.
        
                Returns:
                    None.
        
                Class Fields Initialized:
                    _tags (set): A set containing the default tags: 'mutation', 'term level', 'exploration', and 'no suboperators'.
        
                Why:
                    These tags help the system understand the nature of this mutation operator, 
                    allowing for informed decisions during the equation search process, such as 
                    whether it introduces new terms ('exploration'), modifies existing ones ('term level'), 
                    or avoids creating sub-operators ('no suboperators').
        """
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
        """
        Specific operator for refining equation terms by randomly adjusting their parameters.
        
                This method fine-tunes the parameters of a selected term within an equation to improve the overall fit and accuracy of the model. By introducing small, random increments to the parameters, the algorithm explores the solution space and seeks to minimize the error between the equation's predictions and the observed data. This process helps to discover the optimal parameter values that best describe the underlying dynamics of the system.
        
                Args:
                    objective (tuple): A tuple containing the index of the term to be mutated and the equation object.
                    arguments (dict): A dictionary containing additional arguments required for the mutation process.
        
                Returns:
                    Term: The modified term with updated parameter values.
        """
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
        """
        Sets the operator's tags to a predefined default. This ensures that the operator is correctly categorized with respect to its function within the equation discovery process.
        
                Args:
                    self: The object instance.
        
                Returns:
                    None. The operator's tags are updated in place.
        """
        self._tags = {'mutation', 'term level', 'exploitation', 'no suboperators'}


def get_basic_mutation(mutation_params):
    """
    Generates a basic mutation operator for evolving equation systems.
    
    This method creates and configures a system mutation operator composed
    of equation and metaparameter mutation sub-operators. Term mutation is
    nested under equation mutation to modify individual terms within equations.
    This hierarchical structure allows for fine-grained control over the
    evolutionary process, enabling the discovery of increasingly accurate
    and parsimonious equation models.
    
    Args:
        mutation_params: A dictionary containing mutation parameters that
            control the behavior of the mutation operators.
    
    Returns:
        SystemMutation: A configured system mutation operator ready to be
            used in the evolutionary process.
    """
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
