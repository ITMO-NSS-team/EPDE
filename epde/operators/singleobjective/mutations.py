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
    Represents a mutation operator that applies sub-operators to an objective's equations and parameters.
    """

    key = 'SystemMutation'

    def apply(self, objective : SoEq, arguments : dict): # TODO: add setter for best_individuals & worst individuals 
        self_args, subop_args = self.parse_suboperator_args(arguments = arguments)    

        altered_objective = deepcopy(objective)
        if objective.elite == 'immutable':
        """
        Applies mutation to the objective's equations and parameters.
        
                This method takes an objective function representing a system of equations (SoEq) and a set of arguments for mutation sub-operators.
                It selectively mutates the equations and parameters within the objective, creating diversity in the population of candidate solutions.
                The mutation process involves iterating through the equation and parameter keys, applying the corresponding mutation sub-operator to each.
                The mutated equations and parameters are then updated within a copy of the objective.
                This ensures exploration of the search space by introducing variations in the equation structures and parameter values.
                Finally, the objective's state is reset to ensure consistency.
        
                Args:
                    objective (SoEq): The objective function (system of equations) to be mutated.
                    arguments (dict): A dictionary containing arguments for the sub-operators, specifying mutation parameters.
        
                Returns:
                    SoEq: A mutated copy of the objective function.
        """
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
        """
        Applies a predefined set of tags to the object.
        
        This method resets the object's tags to a default configuration, ensuring consistency in identifying the type of operation performed. This is useful for categorizing and managing different operations within the system.
        
        Args:
            self: The object instance.
        
        Returns:
            None.
        """
        self._tags = {'mutation', 'chromosome level', 'contains suboperators', 'standard'}
    

class EquationMutation(CompoundOperator):
    """
    Applies mutation to the objective equation's structure.
    
        Attributes:
            mutation_rate: The probability of applying a mutation to each term.
            sub_operators: A list of mutation sub-operators to apply.
    """

    key = 'EquationMutation'

    @HistoryExtender(f'\n -> mutating equation', 'ba')
    def apply(self, objective : Equation, arguments : dict):
        """
        Applies mutation to the structure of the equation.
        
                Iterates through the mutable terms of the equation's structure, applying a mutation sub-operator to each term with a probability determined by the mutation rate. This process aims to explore the space of possible equation structures to find one that better fits the observed data.
        
                Args:
                    objective (Equation): The equation to be mutated.
                    arguments (dict): A dictionary of arguments for the operators.
        
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
        Applies a predefined set of tags to the equation mutation object.
        
        This method resets the object's tags to a default set, ensuring consistency
        in identifying and categorizing equation mutations. This is useful for
        standardizing the representation of mutations within the evolutionary
        process, facilitating filtering and analysis based on predefined criteria.
        
        Args:
            self: The EquationMutation object instance.
        
        Returns:
            None. The method modifies the object's internal state by updating its tags.
        """
        self._tags = {'mutation', 'gene level', 'contains suboperators', 'standard'}


class MetaparameterMutation(CompoundOperator):
    """
    Represents a mutation operation that modifies a metaparameter.
    
        This class encapsulates the logic for altering a metaparameter's value
        during an optimization or evolutionary process.
    
        Attributes:
            parameter_name: The name of the metaparameter to be mutated.
            mutation_range: The range within which the metaparameter can be mutated.
    """

    key = 'MetaparameterMutation'

    def apply(self, objective : Union[int, float], arguments : dict):
        """
        Applies a random normal perturbation to the objective value.
        
                This method introduces controlled noise to the objective value, simulating the inherent uncertainty and variability often encountered in real-world data and model evaluations. By adding a random value drawn from a normal distribution, parameterized by a mean and standard deviation, the method explores the solution space more robustly. This helps the evolutionary algorithm escape local optima and discover more generalizable equation structures. If the altered objective becomes negative, its absolute value is returned to maintain a valid objective range.
        
                Args:
                    objective (Union[int, float]): The objective value to be altered.
                    arguments (dict): A dictionary of arguments, including those for sub-operators.
        
                Returns:
                    float: The altered objective value after applying the random perturbation and ensuring it is non-negative.
        """
        self_args, subop_args = self.parse_suboperator_args(arguments = arguments)

        # print('objective', objective)
        
        altered_objective = objective + np.random.normal(loc = self.params['mean'], scale = self.params['std'])        
        if altered_objective < 0:
            altered_objective = - altered_objective
        
        return altered_objective

    def use_default_tags(self):
        """
        Applies a pre-defined set of tags to the mutation, ensuring consistency and adherence to established categories.
        
        This method overwrites any existing tags with a default set, providing a standardized categorization. This is useful for maintaining a consistent vocabulary across different mutation types, facilitating filtering and analysis within the evolutionary process.
        
        Args:
            self: The object instance.
        
        Returns:
            None.
        """
        self._tags = {'mutation', 'gene level', 'no suboperators', 'standard'}

    
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
        Resets the operator's tags to the default set.
        
                This ensures the operator is configured with a standard set of characteristics,
                allowing it to be used in a general-purpose equation discovery process.
                This is useful for ensuring a consistent starting point for different search strategies.
        
                Args:
                    self: The object instance.
        
                Returns:
                    None. This method modifies the object's tags in place.
        """
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
        Sets the tags to the default set.
        
        This method resets the current set of tags, ensuring the mutation operation is characterized by a standard set of properties. This is useful for reverting to a known state or ensuring consistency in the mutation process.
        
        Args:
            self: The object instance.
        
        Returns:
            None.
        """
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
    add_kwarg_to_operator(operator = chromosome_mutation)

    equation_mutation.set_suboperators(operators = {'mutation' : term_mutation})

    chromosome_mutation.set_suboperators(operators = {'equation_mutation' : equation_mutation, 
                                                      'param_mutation' : metaparameter_mutation})
    return chromosome_mutation


def get_singleobjective_mutation(mutation_params):
    """
    Creates and configures a mutation operator tailored for single-objective equation discovery.
    
        This method constructs a hierarchical mutation strategy, comprising chromosome,
        equation, and term mutation components. This layered approach allows for
        fine-grained control over the evolutionary process, enabling efficient
        exploration of the equation space. The relationships between these components
        are established to ensure coordinated mutation.
    
        Args:
            mutation_params: A dictionary containing parameters for the mutation operators,
                such as probabilities and mutation types. These parameters guide the
                mutation process at each level of the hierarchy.
    
        Returns:
            SystemMutation: A configured chromosome mutation operator, ready to be
                integrated into the evolutionary algorithm. This operator orchestrates
                the mutation of entire equation systems.
    
        Why:
            This method is crucial for evolving populations of equation candidates. By
            providing a structured mutation strategy, it facilitates the discovery of
            equations that accurately describe the underlying dynamics of the data.
            The hierarchical design allows for targeted exploration of different aspects
            of the equation structure, improving the efficiency of the search process.
    """
    # TODO: generalize initiation with test runs and simultaneous parameter and object initiation.
    add_kwarg_to_operator = partial(add_base_param_to_operator, target_dict = mutation_params)    

    term_mutation = TermMutation([])

    equation_mutation = EquationMutation(['r_mutation', 'type_probabilities'])
    add_kwarg_to_operator(operator = equation_mutation)
    
    chromosome_mutation = SystemMutation(['indiv_mutation_prob'])
    add_kwarg_to_operator(operator = chromosome_mutation)

    equation_mutation.set_suboperators(operators = {'mutation' : term_mutation})

    chromosome_mutation.set_suboperators(operators = {'equation_mutation' : equation_mutation})
    return chromosome_mutation    