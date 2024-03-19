#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 13:25:47 2024

@author: maslyaev
"""
import os
import numpy as np
import pandas as pd
import re
regex = re.compile('freq:\s\d\S\d+') # Using regular expression for frequency delete (sin/cos)

import itertools
import tempfile
import pickle

from epde.interface.interface import EpdeSearch
from epde.loader import EPDELoader, temp_pickle_save

def dict_update(d_main, term, coeff, k):

    str_t = '_r' if '_r' in term else ''
    arr_term = re.sub('_r', '', term).split(' * ')

    # if structure recorded b * a provided, that a * b already exists (for all case - generalization)
    perm_set = list(itertools.permutations([i for i in range(len(arr_term))]))
    structure_added = False

    for p_i in perm_set:
        temp = " * ".join([arr_term[i] for i in p_i]) + str_t
        if temp in d_main:
            d_main[temp] += [0 for i in range(k - len(d_main[temp]))] + [coeff]
            structure_added = True

    if not structure_added:
        d_main[term] = [0 for i in range(k)] + [coeff]

    return d_main

def equation_table(k, equation, dict_main, dict_right):
    """
        Collecting the obtained values (coefficients and structures) into a common table (the right and left parts of the equation are considered separately)

        Parameters
        ----------
        equation:
        k : Number of equations (final)
        dict_main : dict/table coefficients left parts of the equation
        dict_right : -//- the right parts of the equation

        Returns
        -------
        dict_main, dict_right, k
    """

    equation_s = equation.structure  # list of class objects 'epde.structure.Term'
    equation_c = equation.weights_final  # coefficients of the right part
    text_form_eq = regex.sub('', equation.text_form)  # full equation line

    flag = False  # flag of the right part
    for t_eq in equation_s:
        term = regex.sub('', t_eq.name)  # full name term
        for t in range(len(equation_c)):
            c = equation_c[t]
            if f'{c} * {term} +' in text_form_eq:
                dict_main = dict_update(dict_main, term, c, k)
                equation_c = np.delete(equation_c, t)
                break
            elif f'+ {c} =' in text_form_eq:
                dict_main = dict_update(dict_main, "C", c, k)
                equation_c = np.delete(equation_c, t)
                break
        if f'= {term}' == text_form_eq[text_form_eq.find('='):] and flag is False:
            flag = True
            term += '_r'
            dict_right = dict_update(dict_right, term, -1., k)

    return [dict_main, dict_right]

def object_table(res, variable_name, table_main, k):
    """
        Collecting the obtained objects (system/equation) into a common table

        Parameters
        ----------
        variable_name: List of objective function names
        res : Pareto front of detected equations/systems
        table_main: List of dictionaries
        k : Number of equations/system (final)

        Returns
        -------
        table_main: [{'variable_name1': [{'structure1':[coef1, coef2,...],'structure2':[],...},{'structure1_r':[],'structure2_r':[],...}]},
                    {'variable_name2': [{'structure1':[coef1, coef2,...],'structure2':[],...},{'structure1_r':[],'structure2_r':[],...}]}]
    """
    for list_SoEq in res:  # List SoEq - an object of the class 'epde.structure.main_structures.SoEq'
        for SoEq in list_SoEq:
            # if 3 < max(SoEq.obj_fun[2:]) < 5: # to filter the complexity of equations/system
            # variable_name = SoEq.vals.equation_keys # param for object_epde_search
            for n, value in enumerate(variable_name):
                gene = SoEq.vals.chromosome.get(value)
                table_main[n][value] = equation_table(k, gene.value, *table_main[n][value])

            k += 1
            print(k)

    return table_main, k

def equation_fit(data, grid, derives, config_epde):
    dimensionality = config_epde.params["global_config"]["dimensionality"]

    deriv_method_kwargs = {}
    if config_epde.params["fit"]["deriv_method"] == "poly":
        deriv_method_kwargs = {'smooth': config_epde.params["fit"]["deriv_method_kwargs"]["smooth"], 'grid': grid}
    elif config_epde.params["fit"]["deriv_method"] == "ANN":
        deriv_method_kwargs = {'epochs_max': config_epde.params["fit"]["deriv_method_kwargs"]["epochs_max"]}

    epde_search_obj = EpdeSearch(use_solver=config_epde.params["epde_search"]["use_solver"],
                                 dimensionality=dimensionality,
                                 boundary=config_epde.params["epde_search"]["boundary"],
                                 coordinate_tensors=grid,
                                 verbose_params=config_epde.params["epde_search"]["verbose_params"])

    # epde_search_obj.set_memory_properties(data, mem_for_cache_frac=config_epde.params["set_memory_properties"][
    #     "mem_for_cache_frac"])
    epde_search_obj.set_moeadd_params(population_size=config_epde.params["set_moeadd_params"]["population_size"],
                                      training_epochs=config_epde.params["set_moeadd_params"]["training_epochs"])

    # custom_grid_tokens = CacheStoredTokens(token_type=config_epde.params["Cache_stored_tokens"]["token_type"],
    #                                        boundary=config_epde.params["fit"]["boundary"],
    #                                        token_labels=config_epde.params["Cache_stored_tokens"]["token_labels"],
    #                                        token_tensors=dict(
    #                                            zip(config_epde.params["Cache_stored_tokens"]["token_labels"], grid)),
    #                                        params_ranges=config_epde.params["Cache_stored_tokens"]["params_ranges"],
    #                                        params_equality_ranges=config_epde.params["Cache_stored_tokens"][
    #                                            "params_equality_ranges"])
    '''
    Method epde_search.fit() is used to initiate the equation search.
    '''
    epde_search_obj.fit(data=data, variable_names=config_epde.params["fit"]["variable_names"],
                        data_fun_pow=config_epde.params["fit"]["data_fun_pow"],
                        max_deriv_order=config_epde.params["fit"]["max_deriv_order"],
                        equation_terms_max_number=config_epde.params["fit"]["equation_terms_max_number"],
                        equation_factors_max_number=config_epde.params["fit"]["equation_factors_max_number"],
                        coordinate_tensors=grid, eq_sparsity_interval=config_epde.params["fit"]["eq_sparsity_interval"],
                        derivs=[derives] if derives is not None else None,
                        deriv_method=config_epde.params["fit"]["deriv_method"],
                        deriv_method_kwargs=deriv_method_kwargs,
                        # additional_tokens=[custom_grid_tokens, ],
                        memory_for_cache=config_epde.params["fit"]["memory_for_cache"],
                        prune_domain=config_epde.params["fit"]["prune_domain"])

    '''
    The results of the equation search have the following format: if we call method 
    .equation_search_results() with "only_print = True", the Pareto frontiers 
    of equations of varying complexities will be shown, as in the following example:

    If the method is called with the "only_print = False", the algorithm will return list 
    of Pareto frontiers with the desired equations.
    '''

    epde_search_obj.equation_search_results(only_print=True, level_num=config_epde.params["results"]["level_num"])

    return epde_search_obj

def preprocessing_bamt(variable_name, table_main, k):
    data_frame_total = pd.DataFrame()

    for dict_var in table_main:
        for var_name, list_structure in dict_var.items(): # object - {'variable_name1': [{'structure1':[coef1, coef2,...],'structure2':[],...},{'structure1_r':[],'structure2_r':[],...}]}
            general_dict = {}
            for structure in list_structure:
                general_dict.update(structure)
            dict_var[var_name] = general_dict

    # filling with zeros
    for dict_var in table_main:
        for var_name, general_dict in dict_var.items():
            for key, value in general_dict.items():  # value - it's general dictionary (list dict -> dict)
                if len(value) < k:
                    general_dict[key] = general_dict[key] + [0. for i in range(k - len(general_dict[key]))]

    data_frame_main = [{i: pd.DataFrame()} for i in variable_name]
    # creating dataframe from a table and updating the data
    for n, dict_var in enumerate(table_main):
        for var_name, general_dict in dict_var.items():
            data_frame_main[n][var_name] = pd.DataFrame(general_dict)

    for n, dict_var in enumerate(variable_name):
        data_frame_temp = data_frame_main[n].get(dict_var).copy()
        # renaming columns for every dataframe (column_{variable_name})
        list_columns = [f'{i}_{dict_var}' for i in data_frame_temp.columns]
        data_frame_temp.columns = list_columns
        # combine dataframes
        data_frame_total = pd.concat([data_frame_total, data_frame_temp], axis=1)

    return data_frame_total

def token_check(columns, objects_res, config_bamt):
    list_correct_structures_unique = config_bamt.params["correct_structures"]["list_unique"]
    variable_names = config_bamt.params["fit"]["variable_names"]

    list_correct_structures = set()
    for term in list_correct_structures_unique:
        str_r = '_r' if '_r' in term else ''
        str_elem = ''
        if any(f'_{elem}' in term for elem in variable_names):
            for elem in variable_names:
                if f'_{elem}' in term:
                    term = term.replace(f'_{elem}', "")
                    str_elem = f'_{elem}'
        # for case if several terms exist
        arr_term = re.sub('_r', '', term).split(' * ')
        perm_set = list(itertools.permutations([i for i in range(len(arr_term))]))
        for p_i in perm_set:
            temp = " * ".join([arr_term[i] for i in p_i]) + str_r + str_elem
            list_correct_structures.add(temp)

    def out_red(text):
        print("\033[31m {}".format(text), end='')

    def out_green(text):
        print("\033[32m {}".format(text), end='')

    met, k_sys = 0, len(objects_res)
    k_min = k_sys if k_sys < 5 else 5

    for object_row in objects_res[:k_min]:
        k_c, k_l = 0, 0
        for col in columns:
            if col in object_row:
                if col in list_correct_structures:
                    k_c += 1
                    out_green(f'{col}')
                    print(f'\033[0m:{object_row[col]}')
                else:
                    k_l += 1
                    out_red(f'{col}')
                    print(f'\033[0m:{object_row[col]}')
        print(f'correct structures = {k_c}/{len(list_correct_structures_unique)}')
        print(f'incorrect structures = {k_l}')
        print('--------------------------')

    for object_row in objects_res:
        for temp in object_row.keys():
            if temp in list_correct_structures:
                met += 1

    print(f'average value (equation - {k_sys}) = {met / k_sys}')


def get_objects(synth_data, config_bamt):
    """
        Parameters
        ----------
        synth_data : pd.dataframe
            The fields in the table are structures of received systems/equations,
            where each record/row contains coefficients at each structure
        config_bamt:  class Config from TEDEouS/config.py contains the initial configuration of the task

        Returns
        -------
        objects_result - list objects (combination of equations or systems)
    """
    objects = []  # equations or systems
    for i in range(len(synth_data)):
        object_row = {}
        for col in synth_data.columns:
            object_row[synth_data[col].name] = synth_data[col].values[i]
        objects.append(object_row)

    objects_result = []
    for i in range(len(synth_data)):
        object_res = {}
        for key, value in objects[i].items():
            if abs(float(value)) > config_bamt.params["glob_bamt"]["lambda"]:
                object_res[key] = value
        objects_result.append(object_res)
    return objects_result