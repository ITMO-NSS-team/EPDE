#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 16 20:50:55 2021

@author: mike_ubuntu
"""
import time

import numpy as np
from copy import deepcopy
import warnings

import epde.globals as global_var
from epde.operators.utils.template import CompoundOperator
from epde.decorators import HistoryExtender
from epde.structure.main_structures import Term, Equation
    
class EqRightPartSelector(CompoundOperator):
    """
    Operator for selecting the right-hand side of an equation to approximate a function.
    
    This operator iterates through each term in an equation, treating it as the right-hand side. For each such division, a fitness function is evaluated. The term that yields the highest fitness function value when isolated on the right-hand side is then selected as the appropriate right-hand side.
    
    Key Attributes:
    -----------
    suboperators : dict
            Inherited from the CompoundOperator class.
            key - str, value - instance of a class, inherited from the CompoundOperator.
            Suboperators responsible for equation processing tasks. In this case, it typically includes a 'fitness_calculation' suboperator, which calculates the fitness function value.
    
    
        Methods:
        -----------
        apply(equation)
            return None
            Inplace detection of index of the best separation into right part, saved into ``equation.target_idx``
    
    
        '''
    """
    
    key = 'FitnessCheckingRightPartSelector'

    @HistoryExtender('\n -> The equation structure was detected: ', 'a')        
    def apply(self, objective : Equation, arguments : dict):
        """
        Applies a series of sub-operators to refine the equation's right-hand side.
        
                This method iteratively selects and simplifies terms on the right-hand side
                of the equation. It focuses on terms containing both the main variable and its
                derivative, aiming to isolate the most relevant components for explanation.
                The process continues until a suitable term is identified and the equation is simplified.
                This iterative refinement helps in discovering the underlying structure of the equation
                by strategically dissecting its components.
        
                Args:
                    objective (Equation): The equation to be solved, containing the structure to be refined.
                    arguments (dict): A dictionary of arguments for the sub-operators, guiding the simplification process.
        
                Returns:
                    None: The method modifies the `objective` in place, refining its structure.
        """
        self_args, subop_args = self.parse_suboperator_args(arguments = arguments)

        objective.reset_state(True)

        while not (objective.right_part_selected and objective.simplified):
            min_fitness = np.inf
            weights_internal = np.zeros_like(objective.structure)
            min_idx = 0
            if not any(term.contains_variable(objective.main_var_to_explain) and term.contains_deriv(objective.main_var_to_explain) for term in objective.structure):
                objective.restore_property(mandatory_family=objective.main_var_to_explain, deriv=True)
                
            for target_idx, target_term in enumerate(objective.structure):
                if not (objective.structure[target_idx].contains_variable(objective.main_var_to_explain) and objective.structure[target_idx].contains_deriv(objective.main_var_to_explain)):
                    continue
                objective.target_idx = target_idx
                fitness = self.suboperators['fitness_calculation'].apply(objective,
                                                                            arguments = subop_args['fitness_calculation'],
                                                                            force_out_of_place = True)
                if fitness < min_fitness:
                    min_fitness = fitness
                    min_idx = target_idx
                    weights_internal = objective.weights_internal
                else:
                    pass

            objective.weights_internal = weights_internal
            objective.target_idx = min_idx
            # self.suboperators['fitness_calculation'].apply(objective, arguments = subop_args['fitness_calculation'])
            # if not np.isclose(objective.fitness_value, max_fitness) and global_var.verbose.show_warnings:
            #     warnings.warn('Reevaluation of fitness function for equation has obtained different result. Not an error, if ANN DE solver is used.')
            self.simplify_equation(objective)
        else:
            objective.reset_explaining_term(objective.target_idx)

    def simplify_equation(self, objective: Equation):
        """
        Simplifies the given equation to enhance the search for the optimal equation structure.
        
                It identifies non-zero terms in the equation, finds common factors among them,
                and attempts to reduce the order of these factors if they share the same dimension.
                If a term's order becomes zero after simplification, it is replaced with a new random term to maintain diversity in the equation's structure.
                This simplification process helps to refine the equation and potentially reduce its complexity,
                making it easier to find a solution that accurately represents the underlying dynamics of the system.
        
                Args:
                    objective: The equation to simplify.
        
                Returns:
                    None. The method modifies the equation object in place.
        """
        # Get nonzero terms
        nonzero_terms_mask = np.array([False if weight == 0 else True for weight in objective.weights_internal], dtype=np.integer)
        nonrs_terms = [term for i, term in enumerate(objective.structure) if i != objective.target_idx]
        nonzero_terms = [item for item, keep in zip(nonrs_terms, nonzero_terms_mask) if keep]
        nonzero_terms.append(objective.structure[objective.target_idx])
        nonzero_terms_labels = [[term.cache_label[0]] if not isinstance(term.cache_label[0], tuple) else list(next(zip(*term.cache_label))) for term in nonzero_terms]

        # If amount nonzero terms is more than one -- get their intersection
        if len(nonzero_terms) > 1:
            common_factor = np.array(list(set.intersection(*map(set, nonzero_terms_labels)))).flatten()
            common_dim = []
            if len(common_factor) > 0:
                # Find if this intersection in the same dimension (i.e. trigonometry functions) + it's minimal order
                min_order = np.inf
                for term in nonzero_terms:
                    for factor in term.structure:
                        if factor.cache_label[0] == common_factor[0]:
                            if len(factor.params) > 1:
                                common_dim.append(factor.params[-1])
                            if factor.cache_label[1][0] < min_order:
                                min_order = factor.cache_label[1][0]
                if len(set(common_dim)) < 2:
                    # If dimension is the same -- reduce order of terms' factor
                    for term in nonzero_terms:
                        temp = deepcopy(term)
                        factors_simplified = []
                        for factor in term.structure:
                            if factor.cache_label[0] == common_factor[0]:
                                for i, value in enumerate(factor.params_description):
                                    if factor.params_description[i]["name"] == "power":
                                        factor.params[i] -= min_order
                                        if factor.params[i] == 0:
                                            factors_simplified.append(factor)
                        term.structure = [factor for factor in term.structure if factor not in factors_simplified]
                        term.reset_saved_state()
                        # If term's order became zero -- replace term
                        if len(term.structure) == 0:
                            term.randomize()
                            term.reset_saved_state()
                        while objective.structure.count(term) > 1 or term == temp:
                            term.randomize()
                            term.reset_saved_state()
                        objective.simplified = False
                        objective.right_part_selected = False
                        return
        objective.simplified = True
        objective.right_part_selected = True

    def use_default_tags(self):
        """
        Uses a predefined set of tags to categorize and manage the selection process.
        
                This method overwrites any existing tags with a predefined set, ensuring consistency in how equation right-part selections are handled within the evolutionary process. This standardization aids in the effective discovery of differential equations by maintaining a clear and consistent categorization of operations.
        
                Args:
                    self: The object instance.
        
                Returns:
                    None.
        
                Class Fields:
                    _tags (set): A set containing the default tags: 'equation right part selection', 'gene level', 'contains suboperators', and 'inplace'.
        """
        self._tags = {'equation right part selection', 'gene level', 'contains suboperators', 'inplace'}

        
class RandomRHPSelector(CompoundOperator):
    """
    Operator for selecting the optimal right-hand side of an equation by iteratively evaluating each term as a potential candidate. For each term, a fitness function is computed, and the term yielding the highest fitness value is chosen as the right-hand side.
    
    Key attributes:
    
    *   `suboperators`: A dictionary containing sub-operators for equation processing. In this case, it includes a `fitness_calculation` sub-operator responsible for computing the fitness function value.
    
        Methods:
        -----------
        apply(equation)
            return None
            Inplace detection of index of the best separation into right part, saved into ``equation.target_idx``
    
    
        '''
    """

    key = 'RandomRightPartSelector'

    @HistoryExtender('\n -> The equation structure was detected: ', 'a')
    def apply(self, objective : Equation, arguments : dict):
        """
        Applies a selection strategy to identify a suitable term on the right-hand side of the equation for further refinement.
        
                This method focuses on selecting a term within the equation's structure that can be modified or expanded to better represent the underlying dynamics. It prioritizes terms that already contain the derivative of the main variable, as these are more likely to contribute meaningfully to the equation's explanatory power. If no such term exists, the method introduces a new term containing the derivative to guide the search process. This ensures that the equation evolves towards a form that accurately captures the relationships between variables and their rates of change.
        
                Args:
                  objective: The equation to which the selection strategy is applied.
                  arguments: A dictionary of arguments (currently unused in the selection logic).
        
                Returns:
                  None. This method modifies the `objective` in place.
        """
        # print(f'CALLING RIGHT PART SELECTOR FOR {objective.text_form}')
        self_args, subop_args = self.parse_suboperator_args(arguments = arguments)

        if not objective.right_part_selected:
            term_selection = [term_idx for term_idx, term in enumerate(objective.structure)
                              if term.contains_deriv(variable = objective.main_var_to_explain)]
            
            if len(term_selection) == 0:
                idx = np.random.choice([term_idx for term_idx, _ in enumerate(objective.structure)])
                prev_term = objective.structure[idx]
                while True:
                    candidate_term = Term(pool = prev_term.pool, mandatory_family = objective.main_var_to_explain,
                                          max_factors_in_term = len(prev_term.structure), 
                                          create_derivs = True)
                    if candidate_term.contains_deriv(variable = objective.main_var_to_explain):
                        break
                
                objective.structure[idx] = candidate_term
            else:
                idx = np.random.choice(term_selection)

            objective.target_idx = idx
            # print('Selected right part term', objective.structure[idx].name)
            objective.reset_explaining_term(idx)
            objective.right_part_selected = True


    def use_default_tags(self):
        """
        Sets the default operational tags.
        
                This method initializes the internal tag set with a predefined collection of tags.
                These tags indicate specific characteristics and constraints relevant to the equation discovery process,
                such as operations on the equation's right-hand side, gene-level considerations, the inclusion of sub-operators,
                and whether operations are performed in place. This configuration ensures that the equation search and simplification
                process starts with a consistent and relevant set of constraints.
        
                Args:
                    self: The object instance.
        
                Returns:
                    None.
        """
        self._tags = {'equation right part selection', 'gene level', 'contains suboperators', 'inplace'}
