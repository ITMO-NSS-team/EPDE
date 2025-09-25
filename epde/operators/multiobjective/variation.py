#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 15:43:19 2021

@author: mike_ubuntu
"""

from ast import operator
from operator import eq
import numpy as np
from copy import deepcopy

from functools import partial

from epde.structure.structure_template import check_uniqueness
from epde.optimizers.moeadd.moeadd import ParetoLevels

from epde.supplementary import detect_similar_terms, flatten
from epde.decorators import HistoryExtender, ResetEquationStatus

from epde.operators.utils.template import CompoundOperator, add_base_param_to_operator
from epde.operators.multiobjective.moeadd_specific import get_basic_populator_updater
from epde.operators.multiobjective.mutations import get_basic_mutation


class ParetoLevelsCrossover(CompoundOperator):
    """
    This crossover operator combines parameter crossover for terms sharing the same factors but differing in parameters, and exchanges entire terms between dissimilar equations. It leverages sub-operators for parent selection, parameter crossover, term crossover, coefficient calculation, and fitness evaluation.
    
    
        Methods:
        -----------
        apply(population)
            return the new population, created with the noted operators and containing both parent individuals and their offsprings.    
        copy_properties_to
    """

    key = 'ParetoLevelsCrossover'
    
    def apply(self, objective : ParetoLevels, arguments : dict):
        """
        Method to generate new candidate solutions (offsprings) by combining existing solutions from the population using crossover. This process aims to explore the solution space and discover novel equation structures that better fit the data.
        
                Args:
                    objective (ParetoLevels): An object containing the population of equations, organized by Pareto levels, and unplaced candidates.
                    arguments (dict): A dictionary containing arguments for the crossover operator and its sub-operators.
        
                Returns:
                    ParetoLevels: The updated ParetoLevels object, now containing the generated offsprings in the `unplaced_candidates` attribute. The offsprings are equations created by crossing over parent equations selected based on their Pareto level.
        """
        self_args, subop_args = self.parse_suboperator_args(arguments = arguments)        
        
        crossover_pool = []
        for solution in objective.population:
            crossover_pool.extend([solution,] * solution.crossover_times())
            solution.reset_counter()

        if len(crossover_pool) == 0:
            raise ValueError('crossover pool not created, probably solution.crossover_selected_times error')
        np.random.shuffle(crossover_pool)
        if len(crossover_pool) % 2:
            crossover_pool = crossover_pool[:-1]
        crossover_pool = np.array(crossover_pool, dtype = object).reshape((-1,2))

        offsprings = []
        for pair_idx in np.arange(crossover_pool.shape[0]):
            if len(crossover_pool[pair_idx, 0].vals) != len(crossover_pool[pair_idx, 1].vals):
                raise IndexError('Equations have diffferent number of terms')
            new_system_1 = deepcopy(crossover_pool[pair_idx, 0])
            new_system_2 = deepcopy(crossover_pool[pair_idx, 1])
            new_system_1.reset_state(); new_system_2.reset_state()
            
            new_system_1, new_system_2 = self.suboperators['chromosome_crossover'].apply(objective = (new_system_1, new_system_2), 
                                                                                         arguments = subop_args['chromosome_crossover'])
            offsprings.extend([new_system_1, new_system_2])

        objective.unplaced_candidates = offsprings
        return objective

    def use_default_tags(self):
        """
        Applies a predefined set of tags to categorize the crossover operator. These tags provide a concise description of the operator's characteristics, such as its type, level of operation, and structural properties.
        
                Args:
                    self: The object instance.
        
                Returns:
                    None. The method modifies the object's tags in place to reflect its default categorization.
        
                Why:
                    Tagging operators facilitates their organization and selection within the evolutionary process, enabling the framework to effectively manage and explore diverse search strategies for identifying differential equations.
        """
        self._tags = {'crossover', 'population level', 'contains suboperators', 'standard'}


class ChromosomeCrossover(CompoundOperator):
    """
    Applies crossover at the chromosome level using sub-operators.
    
        This class facilitates crossover between two chromosomes (offspring) by
        applying specialized sub-operators to their constituent genes. It manages
        the application of these sub-operators for both equation and parameter
        genes, effectively recombining genetic material to create new offspring.
    
        Class Methods:
        - apply: Applies crossover to the equation and parameter genes of two offspring.
        - use_default_tags: Uses default tags for the object.
    """

    key = 'ChromosomeCrossover'
    
    def apply(self, objective : tuple, arguments : dict):
        """
        Applies crossover to the equation and parameter genes of two offspring.
        
                This method exchanges genetic material between two offspring to create new candidate solutions.
                It iterates through the equation and parameter keys, applying specialized sub-operators for each.
                The resulting genes are then used to replace the original genes in the offspring.
        
                Args:
                    objective (tuple): A tuple containing two offspring objects to be crossed over.
                    arguments (dict): A dictionary containing arguments for the sub-operators.
        
                Returns:
                    tuple: A tuple containing the two modified offspring after crossover. This ensures diversity
                           in the population, which is crucial for effective equation discovery.
        """
        self_args, subop_args = self.parse_suboperator_args(arguments = arguments)
   
        assert objective[0].vals.same_encoding(objective[1].vals)
        offspring_1 = objective[0]; offspring_2 = objective[1]
                
        eqs_keys = objective[0].vals.equation_keys; params_keys = objective[1].vals.params_keys
        for eq_key in eqs_keys:
            temp_eq_1, temp_eq_2 = self.suboperators['equation_crossover'].apply(objective = (objective[0].vals[eq_key],
                                                                                              objective[1].vals[eq_key]),
                                                                                 arguments = subop_args['equation_crossover'])
            objective[0].vals.replace_gene(gene_key = eq_key, value = temp_eq_1)
            offspring_2.vals.replace_gene(gene_key = eq_key, value = temp_eq_2)
            
        for param_key in params_keys:
            temp_param_1, temp_param_2 = self.suboperators['param_crossover'].apply(objective = (objective[0].vals[param_key],
                                                                                                 objective[1].vals[param_key]),
                                                                                    arguments = subop_args['param_crossover'])
            objective[0].vals.replace_gene(gene_key = param_key, value = temp_param_1)
            objective[1].vals.replace_gene(gene_key = param_key, value = temp_param_2)

            objective[0].vals.pass_parametric_gene(key = param_key, value = temp_param_1)
            objective[1].vals.pass_parametric_gene(key = param_key, value = temp_param_2)

        return objective[0], objective[1]

    def use_default_tags(self):
        """
        Applies a predefined set of tags to categorize the crossover operation.
        
        This method assigns default tags to the crossover object, providing a standardized way to identify and classify its characteristics. This tagging facilitates filtering and selection of appropriate crossover operators during the evolutionary search process.
        
        Args:
            self: The ChromosomeCrossover instance.
        
        Returns:
            None. The method modifies the object's internal `_tags` attribute.
        """
        self._tags = {'crossover', 'chromosome level', 'contains suboperators', 'standard'}


class MetaparamerCrossover(CompoundOperator):
    """
    Applies a metaparameter-controlled crossover operation on two parent solutions.
    
        This class implements a crossover operator that uses a metaparameter
        to blend the genetic material of two parent solutions, creating two offspring.
    
        Class Attributes:
            - metaparameter: The metaparameter that controls the blend of the parents' genetic material.
    """

    key = 'MetaparamerCrossover'
    
    def apply(self, objective : tuple, arguments : dict):
        """
        Applies the blend crossover operator to generate two offspring.
        
                This method refines the search for differential equation structures by
                creating new candidate solutions (offspring) through a linear combination
                of existing solutions (parents). The `metaparam_proportion` controls the
                balance between the parent solutions, influencing the exploration of the
                solution space. This exploration is crucial for discovering equations that
                accurately represent the underlying dynamics of the data.
        
                Args:
                    objective: A tuple containing the objective values of two parent solutions.
                    arguments: A dictionary containing additional arguments for the operator.
        
                Returns:
                    tuple: A tuple containing the objective values of the two offspring.
        """
        self_args, subop_args = self.parse_suboperator_args(arguments = arguments)
        
        offspring_1 = objective[0] + self.params['metaparam_proportion'] * (objective[1] - objective[0])
        offspring_2 = objective[0] + (1 - self.params['metaparam_proportion']) * (objective[1] - objective[0])
        return offspring_1, offspring_2

    def use_default_tags(self):
        """
        Sets a predefined set of tags for the crossover operator.
        
        This method assigns a default set of tags to the crossover operator, 
        characterizing its behavior and properties within the evolutionary process.
        These tags help in categorizing and selecting appropriate operators 
        during the equation discovery process.
        
        Args:
            self: The MetaparamerCrossover instance.
        
        Returns:
            None.
        
        Initializes:
            _tags (set): A set containing the default tags: 'crossover', 'gene level', and 'no suboperators'.
        """
        self._tags = {'crossover', 'gene level', 'no suboperators'}


class EquationCrossover(CompoundOperator):
    """
    Applies equation crossover to a pair of equations within an objective.
    
        This class performs equation crossover between two equations within a
        given objective. It identifies similar terms in both equations and applies
        different sub-operators ('term_param_crossover' and 'term_crossover') to
        these terms to generate new equation structures. The method also ensures
        the uniqueness of the generated terms within their respective equations.
    
        Class Methods:
        - apply: Applies equation crossover to the given objective using sub-operators.
        - use_default_tags: Uses the default set of tags for this object.
    """

    key = 'EquationCrossover'
    
    @HistoryExtender(f'\n -> performing equation crossover', 'ba')
    def apply(self, objective : tuple, arguments : dict):
        """
        Applies equation crossover to a pair of equations, exploring the space of possible equation structures.
        
                This method aims to evolve more accurate and generalizable equation models
                by recombining building blocks from existing equations. It identifies
                and exchanges both similar and dissimilar terms between two equations,
                leveraging sub-operators to modify and refine the equation structures.
                The method prioritizes the creation of unique and valid equation terms
                to maintain diversity and prevent redundancy in the search process.
                This crossover operation facilitates the discovery of novel equation forms
                that better capture the underlying dynamics of the system.
        
                Args:
                    objective (tuple): A tuple containing two equation objects to be crossed over.
                    arguments (dict): A dictionary containing arguments for the sub-operators,
                                      organized by sub-operator name (e.g., 'term_param_crossover', 'term_crossover').
        
                Returns:
                    tuple: A tuple containing the two modified equation objects after crossover.
                           These equations now have potentially different structures resulting
                           from the exchange and modification of terms.
        """
        self_args, subop_args = self.parse_suboperator_args(arguments = arguments)
        
        equation1_terms, equation2_terms = detect_similar_terms(objective[0], objective[1])
        assert len(equation1_terms[0]) == len(equation2_terms[0]) and len(equation1_terms[1]) == len(equation2_terms[1])
        same_num = len(equation1_terms[0]); similar_num = len(equation1_terms[1])
        objective[0].structure = flatten(equation1_terms); objective[1].structure = flatten(equation2_terms)
    
        for i in range(same_num, same_num + similar_num):
            temp_term_1, temp_term_2 = self.suboperators['term_param_crossover'].apply(objective = (objective[0].structure[i], 
                                                                                                    objective[1].structure[i]),
                                                                                       arguments = subop_args['term_param_crossover']) 
            if (check_uniqueness(temp_term_1, objective[0].structure[:i] + objective[0].structure[i+1:]) and 
                check_uniqueness(temp_term_2, objective[1].structure[:i] + objective[1].structure[i+1:])):                     
                objective[0].structure[i] = temp_term_1; objective[1].structure[i] = temp_term_2

        for i in range(same_num + similar_num, len(objective[0].structure)):
            if check_uniqueness(objective[0].structure[i], objective[1].structure) and check_uniqueness(objective[1].structure[i], objective[0].structure):
                objective[0].structure[i], objective[1].structure[i] = self.suboperators['term_crossover'].apply(objective = (objective[0].structure[i], 
                                                                                                                              objective[1].structure[i]),
                                                                                                                 arguments = subop_args['term_crossover'])
                
        return objective[0], objective[1]

    def use_default_tags(self):
        """
        Applies a pre-defined set of tags to the operator.
        
        This method resets the operator's tags to a default configuration, ensuring consistency in identifying its characteristics. This is useful for standardizing the representation of operators within the evolutionary process.
        
        Args:
            self: The object instance.
        
        Returns:
            None.
        """
        self._tags = {'crossover', 'gene level', 'contains suboperators', 'standard'}

class EquationExchangeCrossover(CompoundOperator):
    """
    Applies equation exchange crossover to a pair of objectives.
    
        Class Methods:
        - apply: Applies equation exchange crossover to a pair of objectives.
        - use_default_tags: Uses the default set of tags for this object.
    """

    key = 'EquationExchangeCrossover'
    
    @HistoryExtender(f'\n -> performing equation exchange crossover', 'ba')
    def apply(self, objective : tuple, arguments : dict):
        """
        Applies equation exchange crossover to a pair of objectives.
        
                This method facilitates the exploration of the search space by recombining equation structures. It swaps the 'structure' attributes of the two objective functions, effectively exchanging building blocks between them. This allows the evolutionary algorithm to explore new combinations of equation terms and potentially discover better-fitting models.
        
                Args:
                    objective (tuple): A tuple containing two objective objects, whose equation structures will be exchanged.
                    arguments (dict): A dictionary containing arguments for the operator and its sub-operators.
        
                Returns:
                    tuple: A tuple containing the two objective objects with their structures swapped, enabling the exploration of new equation combinations.
        """
        self_args, subop_args = self.parse_suboperator_args(arguments = arguments)
        
        objective[0].structure, objective[1].structure = objective[1].structure, objective[0].structure
        return objective[0], objective[1]

    def use_default_tags(self):
        """
        Resets the operator's tags to the default set.
        
        This ensures the operator is correctly categorized with its fundamental properties. This is important for proper identification and selection of operators within the evolutionary process.
        
        Args:
            self: The object instance.
        
        Returns:
            None.
        """
        self._tags = {'crossover', 'gene level', 'contains suboperators', 'standard'}


class TermParamCrossover(CompoundOperator):
    """
    Performs crossover by exchanging parameters between parent terms that share the same factor functions but differ in their parameter values.
    
    
        Noteable attributes:
        -----------
        params : dict
            Inhereted from the Specific_Operator class. 
            Main key - 'proportion', value - proportion, in which the offsprings' parameter values are chosen.
    
        Methods:
        -----------
        apply(population)
            return the offspring terms, constructed as the parents' factors with parameter values, selected between the parents' ones.
    """

    key = 'TermParamCrossover'
        
    def apply(self, objective : tuple, arguments : dict):
        """
        Applies crossover to the parameters of two parent terms, creating two offspring terms with a blend of their parents' characteristics.
        
                This method iterates through the tokens of the parent terms, adjusting the numerical parameter values of each token based on a defined proportion.
                The goal is to explore the space of possible parameter combinations, potentially leading to improved equation discovery.
        
                Args:
                    objective (tuple): A tuple containing two `Term` objects, representing the parent terms to be crossed over.
                    arguments (dict): A dictionary containing arguments required for the sub-operators.
        
                Returns:
                    tuple: A tuple containing two `Term` objects, representing the offspring terms resulting from the crossover operation.
        """
        self_args, subop_args = self.parse_suboperator_args(arguments = arguments)
        
        objective[0].reset_saved_state(); objective[1].reset_saved_state()
        
        if len(objective[0].structure) != len(objective[1].structure):
            print([(token.label, token.params) for token in objective[0].structure], [(token.label, token.params) for token in objective[1].structure])
            raise Exception('Wrong terms passed:')
        for term1_token_idx in np.arange(len(objective[0].structure)):
            term2_token_idx = [i for i in np.arange(len(objective[1].structure)) 
                               if objective[1].structure[i].label == objective[0].structure[term1_token_idx].label][0]
            for param_idx, param_descr in objective[0].structure[term1_token_idx].params_description.items():
                if param_descr['name'] == 'power': power_param_idx = param_idx
                if param_descr['name'] == 'dim': dim_param_idx = param_idx
            
            try:                # TODO: refactor logic
                dim_param_idx
            except:
                dim_param_idx = power_param_idx

            for param_idx in np.arange(objective[0].structure[term1_token_idx].params.size):
                if param_idx != power_param_idx and param_idx != dim_param_idx:
                    try:
                        objective[0].structure[term1_token_idx].params[param_idx] = (objective[0].structure[term1_token_idx].params[param_idx] + 
                                                                                     self.params['term_param_proportion'] 
                                                                                     * (objective[1].structure[term2_token_idx].params[param_idx] 
                                                                                        - objective[0].structure[term1_token_idx].params[param_idx]))
                    except KeyError:
                        print([(token.label, token.params) for token in objective[0].structure], [(token.label, token.params) for token in objective[1].structure])
                        raise Exception('Wrong set of parameters:', objective[0].structure[term1_token_idx].params_description, objective[1].structure[term1_token_idx].params_description)
                    objective[1].structure[term2_token_idx].params[param_idx] = (objective[0].structure[term1_token_idx].params[param_idx] + 
                                                                                (1 - self.params['term_param_proportion']) 
                                                                                * (objective[1].structure[term2_token_idx].params[param_idx] 
                                                                                - objective[0].structure[term1_token_idx].params[param_idx]))
        objective[0].reset_occupied_tokens(); objective[1].reset_occupied_tokens()
        return objective[0], objective[1]

    def use_default_tags(self):
        """
        Uses a predefined set of tags to categorize the crossover operator.
        
                This method overwrites any existing tags with a predefined set of default tags. This ensures consistency and allows the system to quickly identify the operator's characteristics during the equation discovery process. By using a standard set of tags, the framework can efficiently filter and select appropriate operators based on the desired search criteria.
        
                Args:
                    self: The object instance.
        
                Returns:
                    None.
        """
        self._tags = {'crossover', 'term level', 'exploitation', 'no suboperators', 'standard'}

class TermCrossover(CompoundOperator):
    """
    Performs crossover by exchanging complete terms between parent equations.
    
    
        Noteable attributes:
        -----------
        params : dict
            Inhereted from the Specific_Operator class. 
            Main key - 'crossover_probability', value - probabilty of the term exchange.
    
        Methods:
        -----------
        apply(population)
            return the offspring terms, which are the same parents' ones, but in different order, if the crossover occured.
            .
    """
    
    key = 'TermCrossover'

    def apply(self, objective : tuple, arguments : dict):
        """
        Performs crossover between two terms based on a defined probability.
        
                This method determines whether to swap the order of two parent terms
                to produce offspring. The decision is based on a crossover probability
                and whether the parent terms share a common variable marker.
        
                Args:
                    objective (tuple): A tuple containing two Term objects (the parent terms).
                    arguments (dict): A dictionary containing arguments needed for the sub-operators.
        
                Returns:
                    tuple: A tuple containing two Term objects (the offspring terms). The order
                           of the terms may be swapped depending on the crossover probability
                           and variable marker condition.
        
                Why:
                This crossover operation helps explore different combinations of terms
                within the equation search space, potentially leading to the discovery
                of more accurate or parsimonious models.
        """
        self_args, subop_args = self.parse_suboperator_args(arguments = arguments)
        
        if (np.random.uniform(0, 1) <= self.params['crossover_probability'] and 
            objective[1].descr_variable_marker == objective[0].descr_variable_marker):
                return objective[1], objective[0]
        else:
                return objective[0], objective[1]
        
    def use_default_tags(self):
        """
        Resets the operator's tags to the default set. This ensures that the operator is correctly categorized with its fundamental properties, such as being a crossover operator that operates at the term level, encouraging exploration, not using sub-operators and being a standard operator. This is important for maintaining consistency in the search process and ensuring that the evolutionary algorithm explores the space of possible operators effectively.
        
                Args:
                    self: The object instance.
        
                Returns:
                    None.
        """
        self._tags = {'crossover', 'term level', 'exploration', 'no suboperators', 'standard'}


def get_basic_variation(variation_params : dict = {}):
    """
    Creates a basic variation operator setup for equation discovery.
    
        This method initializes a Pareto Levels Crossover operator, configuring it with
        sub-operators like term parameter crossover, term crossover, equation crossover,
        metaparameter crossover, equation exchange crossover, and chromosome crossover.
        These sub-operators are structured to explore diverse equation structures during
        the evolutionary search process. The relationships and probabilities between these
        sub-operators are carefully set to guide the variation process towards potentially
        better-fitting equations. This setup facilitates the exploration of the equation
        search space by combining different genetic operators at various levels of granularity.
    
        Args:
            variation_params: A dictionary to store base parameters for operators.
    
        Returns:
            ParetoLevelsCrossover: A ParetoLevelsCrossover operator configured with the basic variation setup.
    """
    # TODO: generalize initiation with test runs and simultaneous parameter and object initiation.
    add_kwarg_to_operator = partial(add_base_param_to_operator, target_dict = variation_params)    

    term_param_crossover = TermParamCrossover(['term_param_proportion'])
    add_kwarg_to_operator(operator = term_param_crossover)
    term_crossover = TermCrossover(['crossover_probability'])
    add_kwarg_to_operator(operator = term_crossover)

    equation_crossover = EquationCrossover()
    metaparameter_crossover = MetaparamerCrossover(['metaparam_proportion'])
    add_kwarg_to_operator(operator = metaparameter_crossover)
    equation_exchange_crossover = EquationExchangeCrossover()

    chromosome_crossover = ChromosomeCrossover()

    pl_cross = ParetoLevelsCrossover([])
    
    equation_crossover.set_suboperators(operators = {'term_param_crossover' : term_param_crossover, 
                                                     'term_crossover' : term_crossover})
    chromosome_crossover.set_suboperators(operators = {'equation_crossover' : [equation_crossover, equation_exchange_crossover],
                                                       'param_crossover' : metaparameter_crossover},
                                          probas = {'equation_crossover' : [0.9, 0.1]})
    pl_cross.set_suboperators(operators = {'chromosome_crossover' : chromosome_crossover})
    return pl_cross
