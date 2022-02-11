#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 11 16:22:17 2021

@author: mike_ubuntu
"""

import numpy as np
from functools import reduce
import copy
import gc
import time
import datetime
import pickle
import warnings

import epde.globals as global_var

import torch
from epde.decorators import History_Extender, Reset_equation_status
from epde.interface.token_family import TF_Pool
from epde.factor import Factor
from epde.supplementary import Filter_powers, Population_Sort, flatten
import epde.moeadd.moeadd_stc as moeadd


def Check_Unqueness(obj, background):
    return not any([elem == obj for elem in background]) 


def normalize_ts(Input):
#    print('normalize_ts Input:', Input)
    matrix = np.copy(Input)
#    print(Matrix.shape)
    if np.ndim(matrix) == 0:
        raise ValueError('Incorrect input to the normalizaton: the data has 0 dimensions')
    elif np.ndim(matrix) == 1:
        return matrix
    else:
        for i in np.arange(matrix.shape[0]):
            # print(matrix[i].shape)
            std = np.std(matrix[i])
            if std != 0:
                matrix[i] = (matrix[i] - np.mean(matrix[i])) / std
            else:
                matrix[i] = 1
        return matrix


class Complex_Structure(object):
    def __init__(self, interelement_operator = np.add, *params):
        self._history = ''
        self.structure = None        
        self.interelement_operator = interelement_operator
    
    def __eq__(self, other):
        if type(other) != type(self):
            raise ValueError('Type of self and other are different')
        return (all([any([other_elem == self_elem for other_elem in other.structure]) for self_elem in self.structure]) and 
                all([any([other_elem == self_elem for self_elem in self.structure]) for other_elem in other.structure]) and 
                len(other.structure) == len(self.structure))

    def set_evaluator(self, evaluator):
        raise NotImplementedError('Functionality of this method has been moved to the evolutionary operator declaration')
    
    def evaluate(self, structural = False):
        assert len(self.structure) > 0, 'Attempt to evaluate an empty complex structure'
        if len(self.structure) == 1:
            return self.structure[0].evaluate(structural)
        else:
            return reduce(lambda x, y: self.interelement_operator(x, y.evaluate(structural)),
                          self.structure[1:], self.structure[0].evaluate(structural))        

    def reset_saved_state(self):
        self.saved = {True:False, False:False}
        self.saved_as = {True:None, False:None}
        for elem in self.structure:
            elem.reset_saved_state()

    @property
    def name(self):
        pass


class Term(Complex_Structure):
    __slots__ = ['_history', 'structure', 'interelement_operator', 'saved', 'saved_as', 
                 'pool', 'max_factors_in_term', 'cache_linked', 'occupied_tokens_labels']
    
    def __init__(self, pool, passed_term = None, max_factors_in_term = 1, forbidden_tokens = None, 
                 interelement_operator = np.multiply):
        super().__init__(interelement_operator)
        self.pool = pool
        self.max_factors_in_term = max_factors_in_term
        
        if passed_term is None:
            self.randomize(forbidden_tokens)     
        else:
            self.defined(passed_term)

        if global_var.tensor_cache is not None:
            self.use_cache()
        self.reset_saved_state() # key - state of normalization, value - if the variable is saved in cache

        
    @property
    def cache_label(self):
        if len(self.structure) > 1:
            structure_sorted = sorted(self.structure, key = lambda x: x.cache_label)
            cache_label = tuple([elem.cache_label for elem in structure_sorted])#reduce(form_label, structure_sorted, '')
        else:
            cache_label = self.structure[0].cache_label
        return cache_label

    def use_cache(self):    
        self.cache_linked = True
#        print('structure:', self.structure)
        for idx, _ in enumerate(self.structure):
            if not self.structure[idx].cache_linked:
                self.structure[idx].use_cache()


    def defined(self, passed_term):
        self.structure = []
        print('passed_term:', passed_term)

        if isinstance(passed_term, (list, tuple)): #type(passed_term) == list or type(passed_term) == tuple:
            for i, factor in enumerate(passed_term):
                if isinstance(factor, str):
                    _, temp_f = self.pool.create(label = factor)
                    self.structure.append(temp_f)#; raise NotImplementedError
                elif isinstance(factor, Factor):
                    self.structure.append(factor)
                else:
                    raise ValueError('The structure of a term should be declared with str or factor.Factor obj, instead got', type(factor))
        else:   # Случай, если подается лишь 1 токен
            if isinstance(passed_term, str):
                _, temp_f = self.pool.create(label = passed_term)
                self.structure.append(temp_f)#; raise NotImplementedError
            elif isinstance(passed_term, Factor):
                self.structure.append(passed_term)
            else:
                raise ValueError('The structure of a term should be declared with str or factor.Factor obj, instead got', type(passed_term))
                
                
    def randomize(self, forbidden_factors = None, **kwargs):
        if np.sum(self.pool.families_cardinality(meaningful_only = True)) == 0:
            raise ValueError('No token families are declared as meaningful for the process of the system search')
        factors_num = np.random.randint(1, self.max_factors_in_term +1)
        if forbidden_factors is None:
            forbidden_factors = []
            
        while True:
            self.occupied_tokens_labels = copy.copy(forbidden_factors)
            occupied_by_factor, factor = self.pool.create(label = None, create_meaningful = True, 
                                                           occupied = self.occupied_tokens_labels, **kwargs)
            self.structure = [factor,]
            self.occupied_tokens_labels.extend(occupied_by_factor)
            factors_powers = {factor.label : 1}
            
            for i in np.arange(1, factors_num):
                occupied_by_factor, factor = self.pool.create(label = None, create_meaningful = False, 
                                                               occupied = self.occupied_tokens_labels, 
                                                               def_term_tokens = [token.label for token in self.structure], 
                                                               **kwargs) 
                if factor.label in factors_powers:
                    factors_powers[factor.label] += 1
                else:
                    factors_powers[factor.label] = 1
                    
                for param_idx, param_descr in factor.params_description.items():
                    if param_descr['name'] == 'power': power_param_idx = param_idx
                        
                if factors_powers[factor.label] == factor.params_description[power_param_idx]['bounds'][1]:
                    self.occupied_tokens_labels.append(factor.label)
                self.structure.append(factor)
                self.occupied_tokens_labels.extend(occupied_by_factor)                
            self.structure = Filter_powers(self.structure)
            if forbidden_factors is None:
                break
            elif all([(Check_Unqueness(factor, forbidden_factors) or not factor.status['unique_for_right_part']) for factor in self.structure]):
                break            


    def evaluate(self, structural):
        assert global_var.tensor_cache is not None, 'Currently working only with connected cache'
        normalize = structural
        if self.saved[structural] or (self.cache_label, normalize) in global_var.tensor_cache: #  
            value = global_var.tensor_cache.get(self.cache_label, normalized = normalize,
                                                saved_as = self.saved_as[normalize]) 
            value = value.reshape(value.size)
            return value
        else:
            self.prev_normalized = normalize
            value = super().evaluate(structural)
            if normalize and np.ndim(value) != 1:
                value = normalize_ts(value)    
            elif normalize and np.ndim(value) == 1 and np.std(value) != 0:
                value = (value - np.mean(value))/np.std(value)
            elif normalize and np.ndim(value) == 1 and np.std(value) == 0:
                value = (value - np.mean(value))   
            if np.all([len(factor.params) == 1 for factor in self.structure]):
                self.saved[normalize] = global_var.tensor_cache.add(self.cache_label, value, normalized = normalize) # Место возможных проблем: сохранение/загрузка нормализованных данных
                if self.saved[normalize]: self.saved_as[normalize] = self.cache_label
            value = value.reshape(value.size)
            return value            

    def Filter_tokens_by_right_part(self, reference_target, equation, equation_position):
        taken_tokens = [factor.label for factor in reference_target.structure if factor.status['unique_for_right_part']]
        meaningful_taken = any([factor.status['meaningful'] for factor in reference_target.structure
                            if factor.status['unique_for_right_part']])
        
        accept_term_try = 0
        while True:
            accept_term_try += 1
            new_term = copy.deepcopy(self)            
            for factor_idx, factor in enumerate(new_term.structure):
                if factor.label in taken_tokens:
                    new_term.Reset_occupied_tokens()
                    _, new_term.structure[factor_idx] = self.pool.create(create_meaningful=meaningful_taken, 
                                                             occupied = new_term.occupied_tokens_labels + taken_tokens)
                    # print('try:', accept_term_try, 'suggested:', new_term.name)
                    #Возможная ошибка из-за (не) использования "create meaningful"
            if Check_Unqueness(new_term, equation.structure[:equation_position] + 
                                         equation.structure[equation_position + 1 :]):
                self.structure = new_term.structure
                self.structure = Filter_powers(self.structure)
                self.reset_saved_state()
                break
            if accept_term_try == 10 and global_var.verbose.show_warnings:
                warnings.warn('Can not create unique term, while filtering equation tokens in regards to the right part.')
            if accept_term_try >= 10:
                self.randomize(forbidden_factors = new_term.occupied_tokens_labels + taken_tokens)
            if accept_term_try == 100:
                print('Something wrong with the random generation of term while running "Filter_tokens_by_right_part"')
                print('proposed', new_term.name, 'for ', equation.text_form, 'with respect to', reference_target.name)

    def Reset_occupied_tokens(self):
        occupied_tokens_new = []
        for factor in self.structure:
            for token_family in self.pool.families:
                if factor in token_family.tokens and factor.status['unique_token_type']:
                    occupied_tokens_new.extend([token for token in token_family.tokens])
                elif factor.status['unique_specific_token']:
                    occupied_tokens_new.append(factor.label)
        self.occupied_tokens_labels = occupied_tokens_new

    @property
    def available_tokens(self): #Переделать, т.к. меняется пул токенов: старая имплементация через лист
        available_tokens = []
        for token in self.pool.families:
            if not all([label in self.occupied_tokens_labels for label in token.tokens]):
                token_new = copy.deepcopy(token)
                token_new.tokens = [label for label in token.tokens if label not in self.occupied_tokens_labels]
                available_tokens.append(token_new)
        return available_tokens

    @property
    def total_params(self):
        return max(sum([len(element.params) - 1 for element in self.structure]), 1)
    
    @property
    def name(self):
        form = ''
        for token_idx in range(len(self.structure)):
            form += self.structure[token_idx].name
            if token_idx < len(self.structure) - 1:
                form += ' * '
        return form
    
    @property
    def contains_deriv(self):
        return any([factor.is_deriv and factor.deriv_code != [None,] for factor in self.structure])
    
    @property
    def solver_form(self):
        
        deriv_orders = []
        deriv_powers = []
        try:
            coeff_tensor = np.ones_like(global_var.grid_cache.get('0'))
        except KeyError:
            raise NotImplementedError('No cache implemented')
        for factor in self.structure:
            if factor.is_deriv:
                for param_idx, param_descr in factor.params_description.items():
                    if param_descr['name'] == 'power': power_param_idx = param_idx
                deriv_orders.append(factor.deriv_code); deriv_powers.append(factor.params[power_param_idx])
            else:
                coeff_tensor = coeff_tensor * factor.evaluate()
        if len(deriv_powers) == 1:
            deriv_powers = deriv_powers[0]
            deriv_orders = deriv_orders[0]
            
        coeff_tensor = torch.from_numpy(coeff_tensor)
        return [coeff_tensor, deriv_orders, deriv_powers]
                
    def __eq__(self, other):
        return (all([any([other_elem == self_elem for other_elem in other.structure]) for self_elem in self.structure]) and 
                all([any([other_elem == self_elem for self_elem in self.structure]) for other_elem in other.structure]) and 
                len(other.structure) == len(self.structure))

    @History_Extender('\n -> was copied by deepcopy(self)', 'n')
    def __deepcopy__(self, memo = None):
        clss = self.__class__
        new_struct = clss.__new__(clss)
        memo[id(self)] = new_struct
        
        attrs_to_avoid_copy = [] # Add attributes to avoid copying during deepcopy, if needed
        for k in self.__slots__:
            try:
                if k not in attrs_to_avoid_copy:
                    if not isinstance(k, list):
                        setattr(new_struct, k, copy.deepcopy(getattr(self, k), memo))
                    else:
                        temp = []
                        for elem in getattr(self, k):
                            temp.append(copy.deepcopy(elem, memo))
                        setattr(new_struct, k, temp)
                else:
                    setattr(new_struct, k, None)
            except AttributeError:
                pass

        return new_struct


class Equation(Complex_Structure): 
    __slots__ = ['_history', 'structure', 'interelement_operator', 'saved', 'saved_as', 
                 'n_immutable', 'pool', 'terms_number', 'max_factors_in_term', 'operator', 
                  '_target', 'target_idx', '_features', 'right_part_selected', 
                  '_weights_final', 'weights_final_evald', '_weights_internal', 'weights_internal_evald', 
                  'fitness_calculated', 'solver_form_defined', '_solver_form', '_fitness_value', 
                  'crossover_selected_times', 'elite']

    def __init__(self, pool, basic_structure, terms_number = 6, max_factors_in_term = 2, 
                 interelement_operator = np.add): #eq_weights_eval

        """

        Class for the single equation for the dynamic system.
            
        attributes:
            structure : list of Term objects \r\n
            List, containing all terms of the equation; first 2 terms are reserved for constant value and the input function;
        
            target_idx : int \r\n
            Index of the target term, selected in the Split phase;
        
            target : 1-d array of float \r\n
            values of the Term object, reshaped into 1-d array, designated as target for application in sparse regression;
            
            features : matrix of float \r\n
            matrix, composed of terms, not included in target, value columns, designated as features for application in sparse regression;
        
            fitness_value : float \r\n
            Inverse value of squared error for the selected target 2function and features and discovered weights; 
        
            estimator : sklearn estimator of selected type \r\n
        
        parameters:

            Matrix of derivatives: first axis through various orders/coordinates in order: ['1', 'f', all derivatives by one coordinate axis
            in increasing order, ...]; second axis: time, further - spatial coordinates;

            tokens : list of strings \r\n
            Symbolic forms of functions, including derivatives;

            terms_number : int, base value of 6 \r\n
            Maximum number of terms in the discovered equation; 

            max_factors_in_term : int, base value of 2\r\n
            Maximum number of factors, that can form a term (e.g. with 2: df/dx_1 * df/dx_2)

        """
        super().__init__(interelement_operator)
        self.reset_state()

        self.n_immutable = len(basic_structure)
        self.pool = pool
        self.structure = []
        self.terms_number = terms_number; self.max_factors_in_term = max_factors_in_term
        self.operator = None
        if (terms_number < self.n_immutable): 
            raise Exception('Number of terms ({}) is too low to even contain all of the pre-determined ones'.format(terms_number))        

        for passed_term in basic_structure:
            if isinstance(passed_term, Term):
                self.structure.append(passed_term)
            elif isinstance(passed_term, str):
                self.structure.append(Term(self.pool, passed_term = passed_term, 
                                           max_factors_in_term = self.max_factors_in_term))
            
        for i in range(len(basic_structure), terms_number):
            check_test = 0
            while True:
                check_test += 1 
                new_term = Term(self.pool, max_factors_in_term = self.max_factors_in_term, passed_term = None)
                if Check_Unqueness(new_term, self.structure):
                    break
            self.structure.append(new_term)

        for idx, _ in enumerate(self.structure):
            self.structure[idx].use_cache()            
        
    @property
    def contains_deriv(self):
        return any([term.contains_deriv for term in self.structure])
    
    @property 
    def forbidden_token_labels(self):
        target_symbolic = [factor.label for factor in self.structure[self.target_idx].structure]
        forbidden_tokens = set()

        for token_family in self.pool.families:
            for token in token_family.tokens:
                if token in target_symbolic and token_family.status['unique_for_right_part']:
                    forbidden_tokens.add(token)        
        return forbidden_tokens

    def reconstruct_to_contain_deriv(self):
        while True:
            replacement_idx = np.random.randint(low = 0, high = len(self.structure))
            temp = Term(self.pool, max_factors_in_term=self.max_factors_in_term) # ,  forbidden_tokens=self.forbidden_token_labels
            if temp.contains_deriv:
                self.structure[replacement_idx] = temp
                break

    def reconstruct_by_right_part(self, right_part_idx):
        new_eq = copy.deepcopy(self)
        self.copy_properties_to(new_eq)
        new_eq.target_idx = right_part_idx
        if any([factor.status['unique_for_right_part'] for factor in new_eq.structure[right_part_idx].structure]):
            for term_idx, term in enumerate(new_eq.structure):
                if term_idx != right_part_idx:
                    term.Filter_tokens_by_right_part(new_eq.structure[right_part_idx], self, term_idx)

        new_eq.reset_saved_state()
        return new_eq

    def evaluate(self, normalize = True, return_val = False, save = True):
        self._target = self.structure[self.target_idx].evaluate(normalize)
        feature_indexes = list(range(len(self.structure)))
        feature_indexes.remove(self.target_idx)
        for feat_idx in range(len(feature_indexes)):
            if feat_idx == 0:
                self._features = self.structure[feature_indexes[feat_idx]].evaluate(normalize)
            elif feat_idx != 0:
                temp = self.structure[feature_indexes[feat_idx]].evaluate(normalize)
                self._features = np.vstack([self._features, temp])
            else:
                continue
        if self._features.ndim == 1:
            # print('before dim extension', self._features.shape)
            self._features = np.expand_dims(self._features, 1).T
            # print('after dim extension', self._features.shape)
        temp_feats = np.vstack([self._features, np.ones(self._features.shape[1])])
        self._features = np.transpose(self._features); temp_feats = np.transpose(temp_feats)
        if return_val:
            self.prev_normalized = normalize
            if normalize:
                elem1 = np.expand_dims(self._target, axis = 1)
                value = np.add(elem1, - reduce(lambda x,y: np.add(x, y), [np.multiply(weight, temp_feats[:,feature_idx])
                                                    for feature_idx, weight in np.ndenumerate(self.weights_internal)])) 
            else:
                elem1 = np.expand_dims(self._target, axis = 1)

                value = np.add(elem1, - reduce(lambda x,y: np.add(x, y), [np.multiply(weight, temp_feats[:,feature_idx])
                                                    for feature_idx, weight in np.ndenumerate(self.weights_final)])) 
            return value, self._target, self._features
        else:
            return None, self._target, self._features
        
    def reset_state(self, reset_right_part : bool = True):
        if reset_right_part: self.right_part_selected = False
        self.weights_internal_evald = False
        self.weights_final_evald = False
        self.fitness_calculated = False
        self.solver_form_defined = False
    
    @Reset_equation_status(reset_input = False, reset_output = True)
    @History_Extender('\n -> was copied by deepcopy(self)', 'n')
    def __deepcopy__(self, memo = None):
        clss = self.__class__
        new_struct = clss.__new__(clss)
        memo[id(self)] = new_struct
        
        attrs_to_avoid_copy = ['_features', '_target']
        for k in self.__slots__:
            try:
                if k not in attrs_to_avoid_copy:
                    if not isinstance(k, list):
                        setattr(new_struct, k, copy.deepcopy(getattr(self, k), memo))
                    else:
                        temp = []
                        for elem in getattr(self, k):
                            temp.append(copy.deepcopy(elem, memo))
                        setattr(new_struct, k, temp)
                else:
                    setattr(new_struct, k, None)
            except AttributeError:
                pass

        return new_struct
            
    def copy_properties_to(self, new_equation):
        new_equation.weights_internal_evald = self.weights_internal_evald
        new_equation.weights_final_evald = self.weights_final_evald
        new_equation.right_part_selected = self.right_part_selected
        new_equation.fitness_calculated = self.fitness_calculated
        new_equation.solver_form_defined = False

        try:
            new_equation._fitness_value = self._fitness_value
        except AttributeError:
            pass
            
    def add_history(self, add):
        self._history += add
            
    @property
    def history(self):
        return self._history
    
    @property
    def fitness_value(self):
        return self._fitness_value
    
    @fitness_value.setter
    def fitness_value(self, val):
        self._fitness_value = val
    
    def penalize_fitness(self, coeff = 1.):
        self._fitness_value = self._fitness_value*coeff

    @property
    def weights_internal(self):    
        if self.weights_internal_evald:
            return self._weights_internal
        else:
            raise AttributeError('Internal weights called before initialization')
        
    @weights_internal.setter
    def weights_internal(self, weights):
        self._weights_internal = weights
        self.weights_internal_evald = True
        self.weights_final_evald = False
        
    @property
    def weights_final(self): 
        if self.weights_final_evald:
            return self._weights_final
        else:
            print(self.text_form)
            raise AttributeError('Final weights called before initialization')
            
    @weights_final.setter
    def weights_final(self, weights):
        self._weights_final = weights
        self.weights_final_evald = True
            
    @property
    def latex_form(self):
        form = r""
        for term_idx in range(len(self.structure)):
            if term_idx != self.target_idx:
                form += str(self.weights_final[term_idx]) if term_idx < self.target_idx else str(self.weights_final[term_idx-1])
                form += ' * ' + self.structure[term_idx].latex_form + ' + '
        form += str(self.weights_final[-1]) + ' = ' + self.structure[self.target_idx].text_form
        return form 
        
    @property
    def text_form(self):
        form = ''
        if self.weights_final_evald:
            for term_idx in range(len(self.structure)):
                if term_idx != self.target_idx:
                    form += str(self.weights_final[term_idx]) if term_idx < self.target_idx else str(self.weights_final[term_idx-1])
                    form += ' * ' + self.structure[term_idx].name + ' + '
            form += str(self.weights_final[-1]) + ' = ' + self.structure[self.target_idx].name            
        else:
            for term_idx in range(len(self.structure)):
                    form += 'k_'+ str(term_idx) + ' ' + self.structure[term_idx].name + ' + '
            form +=  'k_' + str(len(self.structure)) + ' = 0'
        return form

    def solver_form(self):
        if self.solver_form_defined:
            print(self.text_form)
            return self._solver_form
        else:
            self._solver_form = []
            for term_idx in range(len(self.structure)):
                if term_idx != self.target_idx:
                    term_form = self.structure[term_idx].solver_form
                    weight = self.weights_final[term_idx] if term_idx < self.target_idx else self.weights_final[term_idx-1]
                    term_form[0] = term_form[0] * weight
                    term_form[0] = torch.flatten(term_form[0]).unsqueeze(1).type(torch.FloatTensor)
                    self._solver_form.append(term_form)
                    
            free_coeff_weight = torch.from_numpy(np.full_like(a = global_var.grid_cache.get('0'), 
                                                              fill_value = self.weights_final[-1]))
            free_coeff_weight = torch.flatten(free_coeff_weight).unsqueeze(1).type(torch.FloatTensor)
            target_weight = torch.from_numpy(np.full_like(a = global_var.grid_cache.get('0'), 
                                                              fill_value = -1.))            
            target_form = self.structure[self.target_idx].solver_form
            target_form[0] = target_form[0] * target_weight
            target_form[0] = torch.flatten(target_form[0]).unsqueeze(1).type(torch.FloatTensor)
            
            self._solver_form.append([free_coeff_weight, [None], 0])
            self._solver_form.append(target_form)
            self.solver_form_defined = True
            return self._solver_form
    
    @property
    def state(self):
        return self.text_form    
    
    @property
    def described_variables(self):
        eps=1e-7
        described = set()
        for term_idx, term in enumerate(self.structure):
            if term_idx == self.target_idx:
                described.update({factor.type for factor in term.structure})
            else:
                weight_idx = term_idx if term_idx < term_idx else term_idx - 1
                if np.abs(self.weights_final[weight_idx]) > eps:
                    described.update({factor.type for factor in term.structure})
        described = frozenset(described)
        return described
    
    def max_deriv_orders(self):
        solver_form = self.solver_form()
        max_orders = np.zeros(global_var.grid_cache.get('0').ndim)
        
        def count_order(obj, deriv_ax):
            if obj is None:
                return 0
            else:
                return obj.count(deriv_ax)
                
        for term in solver_form:
            if isinstance(term[2], list):
                for deriv_factor in term[1]:
                    orders = np.array([count_order(deriv_factor, ax) for ax
                                       in np.arange(max_orders.size)])
                    max_orders = np.maximum(max_orders, orders)
            else:
                orders = np.array([count_order(term[1], ax) for ax
                                   in np.arange(max_orders.size)])
                max_orders = np.maximum(max_orders, orders)
        if np.max(max_orders) > 4:
            raise NotImplementedError('The current implementation allows does not allow higher orders of equation, than 2.')
        return max_orders
    
    def boundary_conditions(self, main_var_key = ('u', (1.0,)), full_domain : bool = False):
        required_bc_ord = self.max_deriv_orders()   # We assume, that the maximum order of the equation here is 2
        if global_var.grid_cache is None:
            raise NameError('Grid cache has not been initialized yet.')
        
        bconds = []
        hardcoded_bc_relative_locations = {0 : (), 1 : (0,), 2 : (0, 1), 
                                           3 : (0., 0.5, 1.), 4 : (0., 1/3., 2/3., 1.)}
    
        if full_domain:
            grid_cache = global_var.initial_data_cache
            tensor_cache = global_var.initial_data_cache
        else:
            grid_cache = global_var.grid_cache
            tensor_cache = global_var.tensor_cache
            
        tensor_shape = grid_cache.get('0').shape

        def get_boundary_ind(tensor_shape, axis, rel_loc):
            return tuple(np.meshgrid(*[np.arange(shape) if dim_idx != axis else min(int(rel_loc * shape), shape-1)
                       for dim_idx, shape in enumerate(tensor_shape)], indexing = 'ij'))
        print(required_bc_ord)
        print(self.text_form)
        print('hardcoded_bc_relative_locations', hardcoded_bc_relative_locations)
        for ax_idx, ax_ord in enumerate(required_bc_ord):
            print(ax_idx, ax_ord)
            for loc_fraction in hardcoded_bc_relative_locations[ax_ord]:
                indexes = get_boundary_ind(tensor_shape, axis = ax_idx, rel_loc = loc_fraction)
                # if len(tensor_shape) > 0:                
                coords_raw = np.array([grid_cache.get(str(idx))[indexes] for idx 
                                  in np.arange(len(tensor_shape))]) 
                coords = coords_raw.T # np.squeeze(coords_raw).T
                if coords.ndim > 2:
                    coords = coords.squeeze()
                vals = np.expand_dims(tensor_cache.get(main_var_key)[indexes], axis = 0).T # np.squeeze(tensor_cache.get(main_var_key)[indexes]).T
                
                coords = torch.from_numpy(coords).type(torch.FloatTensor)
                
                vals = torch.from_numpy(vals).type(torch.FloatTensor)
                bconds.append([coords, vals])    

        print('shape of the grid', grid_cache.get('0').shape)
        print('Obtained boundary conditions', len(bconds[0]))
        return bconds   
    
    def clear_after_solver(self):
        del self.model
        del self._solver_form; self.solver_form_defined = False
        gc.collect()
                

def solver_formed_grid():
    keys, training_grid = global_var.grid_cache.get_all()
    assert len(keys) == training_grid[0].ndim, 'Mismatching dimensionalities'
    
    training_grid = np.array(training_grid).reshape((len(training_grid), -1))
    return torch.from_numpy(training_grid).T.type(torch.FloatTensor)
    


class SoEq(Complex_Structure, moeadd.moeadd_solution):
    # __slots__ = ['tokens_indep', 'tokens_dep', 'equation_number']
    def __init__(self, pool, terms_number, max_factors_in_term, sparsity = None, eq_search_iters = 100):
        self.tokens_indep = TF_Pool(pool.families_meaningful) #[family for family in token_families if family.status['meaningful']]
        self.tokens_dep = TF_Pool(pool.families_supplementary) #[family for family in token_families if not family.status['meaningful']]
        self.equation_number = np.size(self.tokens_indep.families_cardinality())
        
        if sparsity is not None: self.vals = sparsity 
        self.max_terms_number = terms_number; self.max_factors_in_term = max_factors_in_term
        self.moeadd_set = False; self.eq_search_operator_set = False# ; self.evaluated = False
        self.def_eq_search_iters = eq_search_iters
        
    def use_default_objective_function(self):
        from epde.eq_mo_objectives import system_discrepancy, system_complexity_by_terms, system_complexity_by_factors
        self.set_objective_functions([system_discrepancy, system_complexity_by_factors])
        
    def set_objective_functions(self, obj_funs):
        '''
        Method to set the objective functions to evaluate the "quality" of the system of equations.
        
        Parameters:
        -----------
            obj_funs - callable or list of callables;
            function/functions to evaluate quality metrics of system of equations. Can return a single 
            metric (for example, quality of the process modelling with specific system), or 
            a list of metrics (for example, number of terms for each equation in the system).
            The function results will be flattened after their application. 
            
        '''
        assert callable(obj_funs) or all([callable(fun) for fun in obj_funs])
        self.obj_funs = obj_funs
#        import time
#        print(len(self.obj_funs))
#        time.sleep(10)
    
    def set_eq_search_evolutionary(self, evolutionary):
#        raise NotImplementedError('In current version, the evolutionary operatorshall be taken from global variables')
#        assert type(evolutionary.coeff_calculator) != type(None), 'Defined evolutionary operator lacks coefficient calculator'
        self.eq_search_evolutionary_strategy = evolutionary
        self.eq_search_operator_set = True
        
    def create_equations(self, population_size = 16, sparsity = None, eq_search_iters = None, EA_kwargs = dict()):
#        if type(eq_search_iters) == type(None) and type(self.def_eq_search_iters) == type(None):
#            raise ValueError('Number of iterations is not defied both in method parameter or in object attribute')
        assert self.eq_search_operator_set
        
        if eq_search_iters is None: eq_search_iters = self.def_eq_search_iters
        if sparsity is None: 
            sparsity = self.vals
        else:
            self.vals = sparsity

        self.population_size = population_size
        self.eq_search_evolutionary_strategy.modify_block_params(block_label = 'truncation', 
                                                                 param_label = 'population_size', 
                                                                 value = population_size)
        
        self.structure = []; self.eq_search_iters = eq_search_iters
        token_selection = self.tokens_indep
        
        self.vars_to_describe = {token_family.type for token_family in self.tokens_dep.families}
        self.vars_to_describe = self.vars_to_describe.union({token_family.type for token_family in self.tokens_indep.families})
        self.separated_vars = set()
        
        for eq_idx in range(self.equation_number):
            current_tokens = token_selection + self.tokens_dep
#            print('Equation index', eq_idx, self.vals)
            self.eq_search_evolutionary_strategy.modify_block_params(block_label = 'rps1', param_label = 'sparsity', 
                                                                     value = self.vals[eq_idx], suboperator_sequence = ['eq_level_rps', 'fitness_calculation', 'sparsity'])#(sparsity_value = self.vals[eq_idx])
            self.eq_search_evolutionary_strategy.modify_block_params(block_label = 'rps2', param_label = 'sparsity', 
                                                                     value = self.vals[eq_idx], suboperator_sequence = ['eq_level_rps', 'fitness_calculation', 'sparsity'])#(sparsity_value = self.vals[eq_idx])

            cur_equation, cur_eq_operator_error_abs, cur_eq_operator_error_structural = self.optimize_equation(pool = current_tokens, 
                                                  strategy = self.eq_search_evolutionary_strategy, population_size = self.population_size,
                                                  separate_vars = self.separated_vars, EA_kwargs = EA_kwargs)
            self.vars_to_describe.difference_update(cur_equation.described_variables)
            self.separated_vars.add(frozenset(cur_equation.described_variables))

            self.structure.append(cur_equation)
#            self.single_vars_in_equation.update()
#            cache.clear(full = False)
            if not eq_idx == self.equation_number - 1:
                global_var.tensor_cache.change_variables(cur_eq_operator_error_abs, 
                                                         cur_eq_operator_error_structural)
#            for idx, _ in enumerate(token_selection):
#                 token_selection[idx].change_variables(cur_eq_operator_error)               
        
#        obj_funs = np.array(flatten([func(self) for func in self.obj_funs]))
#        np.array([self.evaluate(normalize = False),] + [eq.L0_norm for eq in self.structure])
        moeadd.moeadd_solution.__init__(self, self.vals, self.obj_funs) # , return_val = True, self) super(
        self.moeadd_set = True
            
    def optimize_equation(self, pool, strategy, population_size, basic_terms : list = [], 
                          separate_vars : set = None, EA_kwargs = dict()):
        population = [Equation(pool, basic_terms, self.max_terms_number, self.max_factors_in_term) 
                        for i in range(population_size)]
        EA_kwargs['separate_vars'] = separate_vars
        strategy.run(initial_population = population, EA_kwargs = EA_kwargs)
        result = strategy.result
        
        return result[0], result[0].evaluate(normalize = False, return_val=True)[0], result[0].evaluate(normalize = True, return_val=True)[0]
        
    @staticmethod
    def equation_opt_iteration(population, evol_operator, population_size, iter_index, separate_vars, strict_restrictions = True):
        for equation in population:
            if equation.described_variables in separate_vars:
                equation.penalize_fitness(coeff = 0.)           
        population = Population_Sort(population)
        population = population[:population_size]
        gc.collect()
        population = evol_operator.apply(population, separate_vars)
        return population
        
    def evaluate(self, normalize = True):
        if len(self.structure) == 1:
            value = self.structure[0].evaluate(normalize = normalize, return_val = True)[0]
        else:
            value = np.sum([equation.evaluate(normalize, return_val = True)[0] for equation in self.structure])
        value = np.sum(np.abs(value))
        return value

    @property
    def obj_fun(self):
#        print('objective functions:', self.obj_funs)
#        print('objective function values:', [func(self) for func in self.obj_funs], flatten([func(self) for func in self.obj_funs]))
        return np.array(flatten([func(self) for func in self.obj_funs]))
        
    def __call__(self):
        assert self.moeadd_set, 'The structure of the equation is not defined, therefore no moeadd operations can be called'
        return self.obj_fun

    @property
    def text_form(self):
        form = ''
        if len(self.structure) > 1:
            for eq_idx, equation in enumerate(self.structure):
                if eq_idx == 0:
                    form += ' / ' + equation.text_form + '\n'                                        
                elif eq_idx == len(self.structure) - 1:
                    form += ' \ ' + equation.text_form + '\n'
                else:
                    form += ' | ' + equation.text_form + '\n'
        else:
            form += self.structure[0].text_form + '\n'
        return form

    def __eq__(self, other):
        assert self.moeadd_set, 'The structure of the equation is not defined, therefore no moeadd operations can be called'        
#        eps = 1e-9
        return (all([any([other_elem == self_elem for other_elem in other.structure]) for self_elem in self.structure]) and 
                all([any([other_elem == self_elem for self_elem in self.structure]) for other_elem in other.structure]) and 
                len(other.structure) == len(self.structure)) or all(np.isclose(self.obj_fun, other.obj_fun))        

    @property
    def latex_form(self):
        form = r"\begin{eqnarray*}"
        for equation in self.structure:
            form += equation.latex_form + r", \\ "
        form += r"\end{eqnarray*}"
        
            
    def __hash__(self):
        return hash(tuple(self.vals))
        
    def __deepcopy__(self, memo = None):
        clss = self.__class__
        new_struct = clss.__new__(clss)
        memo[id(self)] = new_struct
        
        new_struct.__dict__.update(self.__dict__)
        
        for k in self.__slots__:
                try:
                    if not isinstance(k, list):
                        setattr(new_struct, k, copy.deepcopy(getattr(self, k), memo))
                    else:
                        temp = []
                        for elem in getattr(self, k):
                            temp.append(copy.deepcopy(elem, memo))
                        setattr(new_struct, k, temp)                            
                except AttributeError:
                    pass     

        for idx, eq in enumerate(self.structure):
            eq.copy_properties_to(new_struct)
        return new_struct
    
    def solver_params(self, full_domain):
        '''
        Returns solver form, grid and boundary conditions
        '''
        if len(self.structure) > 1:
            raise Exception('Solver form is defined only for a "system", that contains a single equation.')
        else:
            form = self.structure[0].solver_form()
            grid = solver_formed_grid()
            bconds = self.structure[0].boundary_conditions(full_domain = full_domain)
            return form, grid, bconds
