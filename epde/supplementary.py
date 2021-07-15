#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 16:33:34 2020

@author: mike_ubuntu
"""

import numpy as np
import copy
from functools import reduce

def flatten(obj):
    '''
    Method to flatten list, passed as ``obj`` - the function parameter.
    '''
    assert type(obj) == list 
    
    for idx, elem in enumerate(obj):
        if not isinstance(elem, (list, tuple)):
            obj[idx] = [elem,]
    return reduce(lambda x,y: x+y, obj)

def try_iterable(arg):
    
    try:
        _ = [elem for elem in arg]
    except TypeError:
        return False
    return True

def memory_assesment():
    try:
        h=hpy()
    except NameError:
        from guppy import hpy
        h=hpy()
    print(h.heap())
    del h

def factor_params_to_str(factor, set_default_power = False, power_idx = 0):
#    power_idx = [key for (key, value) in factor.params_description if value['name'] == 'power']#[0]
#    print('params:', factor.params, factor.params_description, power_idx)
    param_label = np.copy(factor.params)            
    if set_default_power:
        param_label[power_idx] = 1.
    return (factor.label, tuple(param_label))

def form_label(x, y):
    print(type(x), type(y.cache_label))
    return x + ' * ' + y.cache_label if len(x) > 0 else x + y.cache_label

#def Detect_Similar_Terms_bugged(base_equation_1, base_equation_2): # Переделать!
#    equation_1 = copy.deepcopy(base_equation_1); equation_2 = copy.deepcopy(base_equation_2)
#    same_terms_from_eq1 = []
#    same_terms_from_eq2 = []    
#    
#    similar_terms_from_eq1 = []
#    similar_terms_from_eq2 = []
#    
#    different_terms_from_eq1 = []
#    different_terms_from_eq2 = []    
#    
#    for eq1_term in base_equation_1.structure:
##        print('processing', eq1_term.name)
#        for eq2_term in base_equation_2.structure:
#            print('processing', eq1_term.name, 'with', eq2_term.name)
#            if eq1_term == eq2_term:
#                same_terms_from_eq1.append(eq1_term); same_terms_from_eq2.append(eq2_term);
#                print('deleting', eq1_term.name, 'with', eq2_term.name)   
#                try:
#                    equation_1.structure.remove(eq1_term); equation_2.structure.remove(eq2_term); break
#                except ValueError:
#                    print('term 1:', eq1_term.name, 'term 2:', eq2_term.name)
#                    print(equation_1.text_form, '\n', equation_2.text_form)
#                    raise ValueError                 
#            elif set([token.label for token in eq1_term.structure]) == set([token.label for token in eq2_term.structure]) and len(eq1_term.structure) == len(eq2_term.structure):
#                similar_terms_from_eq1.append(eq1_term); similar_terms_from_eq2.append(eq2_term); 
#                try:
#                    equation_1.structure.remove(eq1_term)
#                except ValueError:
#                    print(eq1_term.name, [factor.name for factor in eq1_term.structure])
##                    print([(term == eq1_term, term.name, [factor.name for factor in eq1_term.structure], (term.structure[0] == eq1_term.structure[0]), (all([any([other_elem == self_elem for other_elem in eq1_term.structure]) for self_elem in term.structure]) and 
##                all([any([other_elem == self_elem for self_elem in term.structure]) for other_elem in eq1_term.structure]) and 
##                len(term.structure) == len(eq1_term.structure))) for term in equation_1.structure])
##                    print([(term == eq1_term, term.name, [factor.name for factor in eq1_term.structure]) for term in base_equation_1.structure])
#                    print([(term.name, eq1_term == term, eq1_term is term) for term in equation_1.structure])
#                    print([term.name for term in equation_1.structure])
#                    raise ValueError
#                try:
#                    equation_2.structure.remove(eq2_term)
#                except ValueError:
#                    print(eq2_term.name, [factor.name for factor in eq2_term.structure])
##                    print([(term == eq1_term, term.name, [factor.name for factor in eq1_term.structure], (term.structure[0] == eq1_term.structure[0]), (all([any([other_elem == self_elem for other_elem in eq1_term.structure]) for self_elem in term.structure]) and 
##                all([any([other_elem == self_elem for self_elem in term.structure]) for other_elem in eq1_term.structure]) and 
##                len(term.structure) == len(eq1_term.structure))) for term in equation_1.structure])
##                    print([(term == eq1_term, term.name, [factor.name for factor in eq1_term.structure]) for term in base_equation_1.structure])
#                    print(base_equation_1.text_form, '\n', base_equation_2.text_form)
#                    print([(term.name, eq2_term == term, eq2_term is term) for term in equation_2.structure])
#                    print(eq2_term in equation_2.structure)
#                    raise ValueError
#
#                break
#
#    for term_idx in np.arange(len(equation_1.structure)):
#        different_terms_from_eq1.append(equation_1.structure[term_idx]); different_terms_from_eq2.append(equation_2.structure[term_idx])
#    return [same_terms_from_eq1, similar_terms_from_eq1, different_terms_from_eq1], [same_terms_from_eq2, similar_terms_from_eq2, different_terms_from_eq2]


def Detect_Similar_Terms(base_equation_1, base_equation_2): # Переделать!
#    equation_1 = copy.deepcopy(base_equation_1); equation_2 = copy.deepcopy(base_equation_2)
    same_terms_from_eq1 = []
    same_terms_from_eq2 = []    
    eq2_processed = np.full(shape = len(base_equation_2.structure), fill_value = False)
    
    similar_terms_from_eq1 = []
    similar_terms_from_eq2 = []
    
    different_terms_from_eq1 = []
    different_terms_from_eq2 = []    
#    print(base_equation_1.text_form)
#    print(base_equation_2.text_form)
    for eq1_term in base_equation_1.structure:
        found_similar = False
        for idx, eq2_term in enumerate(base_equation_2.structure):
            if eq1_term == eq2_term and not eq2_processed[idx]:
                found_similar = True
                same_terms_from_eq1.append(eq1_term)
                same_terms_from_eq2.append(eq2_term)
                eq2_processed[idx] = True
#                print('Written:', eq1_term.name, '&', eq2_term.name, ': they are the same', 'idx = ', idx)
                break
            elif ({token.label for token in eq1_term.structure} == {token.label for token in eq2_term.structure} and 
                  len(eq1_term.structure) == len(eq2_term.structure) and not eq2_processed[idx]):                
                found_similar = True
                similar_terms_from_eq1.append(eq1_term); 
                similar_terms_from_eq2.append(eq2_term)
                eq2_processed[idx] = True
#                print('Written:', eq1_term.name, '&', eq2_term.name, ': they are similar', 'idx = ', idx)
                break
        if not found_similar:
#            print('Written:', eq1_term.name, 'from eq2 : it is unique')
            different_terms_from_eq1.append(eq1_term)
            
    for idx, elem in enumerate(eq2_processed):
        if not elem: 
#            print(idx)
#            print('Written:', base_equation_2.structure[idx].name, 'from eq2 : it is unique')            
            different_terms_from_eq2.append(base_equation_2.structure[idx])
        
#    print(len(same_terms_from_eq1), len(similar_terms_from_eq1), len(different_terms_from_eq1), len(base_equation_1.structure))
#    print(len(same_terms_from_eq2), len(similar_terms_from_eq2), len(different_terms_from_eq2), len(base_equation_2.structure))

    assert len(same_terms_from_eq1) + len(similar_terms_from_eq1) + len(different_terms_from_eq1) == len(base_equation_1.structure)
    assert len(same_terms_from_eq2) + len(similar_terms_from_eq2) + len(different_terms_from_eq2) == len(base_equation_2.structure)    
    return [same_terms_from_eq1, similar_terms_from_eq1, different_terms_from_eq1], [same_terms_from_eq2, similar_terms_from_eq2, different_terms_from_eq2]


def Filter_powers(gene):    # Разобраться и переделать
    gene_filtered = []
    for token_idx in range(len(gene)):
        total_power = gene.count(gene[token_idx])
        powered_token = copy.deepcopy(gene[token_idx])
        
        power_idx = np.inf
        for param_idx, param_info in powered_token.params_description.items():
            if param_info['name'] == 'power': 
                power_idx = param_idx
                break
        powered_token.params[power_idx] = total_power
        if powered_token not in gene_filtered:
            gene_filtered.append(powered_token)
    return gene_filtered

def Bind_Params(zipped_params):
    param_dict = {}
    for token_props in zipped_params:
        param_dict[token_props[0]] = token_props[1]
    return param_dict

def Slice_Data_3D(matrix, part = 4, part_tuple = None):     # Input matrix slicing for separate domain calculation
    if part_tuple:
        for i in range(part_tuple[0]):
            for j in range(part_tuple[1]):
                yield matrix[:, i*int(matrix.shape[1]/float(part_tuple[0])):(i+1)*int(matrix.shape[1]/float(part_tuple[0])), 
                             j*int(matrix.shape[2]/float(part_tuple[1])):(j+1)*int(matrix.shape[2]/float(part_tuple[1]))], i, j   
    part_dim = int(np.sqrt(part))
    for i in range(part_dim):
        for j in range(part_dim):
            yield matrix[:, i*int(matrix.shape[1]/float(part_dim)):(i+1)*int(matrix.shape[1]/float(part_dim)), 
                         j*int(matrix.shape[2]/float(part_dim)):(j+1)*int(matrix.shape[2]/float(part_dim))], i, j

def Define_Derivatives(var_name = 'u', dimensionality = 1, max_order = 2):
    deriv_names = [var_name,]
    var_deriv_orders = [[] ,]
    for var_idx in range(dimensionality):
        for order in range(max_order):
            var_deriv_orders.append([var_idx,] * (order+1))
            if order == 0:
                deriv_names.append('d'+ var_name + '/dx'+str(var_idx+1))
            else:
                deriv_names.append('d^'+str(order+1) + var_name + '/dx'+str(var_idx+1)+'^'+str(order+1))
    return deriv_names, var_deriv_orders

#def Create_Var_Matrices(U_input, max_order = 3):
#    var_names = ['u']
#
#    for var_idx in range(U_input.ndim):
#        for order in range(max_order):
#            if order == 0:
#                var_names.append('du/dx'+str(var_idx+1))
#            else:
#                var_names.append('d^'+str(order+1)+'u/dx'+str(var_idx+1)+'^'+str(order+1))
#
#    variables = np.ones((len(var_names),) + U_input.shape)      
#    return variables, tuple(var_names)
#

#def Prepare_Data_matrixes(raw_matrix, dim_info):
#    resulting_matrix = np.reshape(raw_matrix, dim_info)
#    return resulting_matrix 


#def Decode_Gene(gene, token_names, parameter_labels, n_params = 1):
#    term_dict = {}
#    for token_idx in range(0, gene.shape[0], n_params):
#        term_params = {}#coll.OrderedDict()
#        for param_idx in range(0, n_params):
#            term_params[parameter_labels[param_idx]] = gene[token_idx*n_params + param_idx]    
#        term_dict[token_names[int(token_idx/n_params)]] = term_params
#    return term_dict
#
#
#def Encode_Gene(label_dict, token_names, parameter_labels, n_params = 1):
#    gene = np.zeros(shape = len(token_names) * n_params)
#
#    for i in range(len(token_names)):
#        if token_names[i] in label_dict:
#            #print(token_names, label_dict[token_names[i]])
#            for key, value in label_dict[token_names[i]].items():
#                gene[i*n_params + parameter_labels.index(key)] = value
#    return gene

def Population_Sort(input_population):
    individ_fitvals = [individual.fitness_value if individual.fitness_calculated else 0 for individual in input_population ]
    pop_sorted = [x for x, _ in sorted(zip(input_population, individ_fitvals), key=lambda pair: pair[1])]
    return list(reversed(pop_sorted))