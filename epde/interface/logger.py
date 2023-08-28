#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 16:21:21 2023

@author: maslyaev
"""

import numpy as np
import json
import pandas as pd

from itertools import product
from typing import Union

from epde.structure.main_structures import SoEq, Equation
from epde.interface.equation_translator import translate_equation, parse_equation_str

def equations_match(eq_checked: Equation, eq_ref: Equation):
    def parse_equation(equation: Equation, eps = 1e-9):
        term_weights = []
        for idx, term in enumerate(equation.structure):
            if idx < equation.target_idx:
                term_weights.append(equation.weights_final[idx])
            elif idx == equation.target_idx:
                term_weights.append(1)
            else:
                term_weights.append(equation.weights_final[idx-1])
        print(term_weights)
        return {frozenset({(factor.label, factor.param(name = 'power')) 
                           for factor in term.structure}) 
                for idx, term in enumerate(equation.structure) if np.abs(term_weights[idx]) > eps}
    
    checked_parsed = parse_equation(eq_checked)
    ref_parsed = parse_equation(eq_ref)
    
    matching = sum([term in checked_parsed for term in ref_parsed])
    missing = len(ref_parsed) - matching
    extra = sum([term not in ref_parsed for term in checked_parsed])
    return (matching, missing, extra)

def systems_match(checked_system: SoEq, reference_system: SoEq):
    return [equations_match(checked_system.vals[var], reference_system.vals[var]) 
            for var in checked_system.vars_to_describe]
        
class Logger():
    def __init__(self, name, referential_equation = None, pool = None):
        self.reset(name = name)
        if isinstance(referential_equation, str) or isinstance(referential_equation, dict):
            if pool is None:
                raise ValueError('Can not translate equations: the pool of tokens is missing')
            referential_equation = translate_equation(referential_equation, pool)
        self._referential_equation = referential_equation
        
    def reset(self, name = None):
        try:
            self.log_out()
        except AttributeError:
            pass
        
        self._log = {}
        self._meta = {'aggregation_key' : []}
        if name is not None:
            self.name = name
        
    def dump(self):
        self._log['meta'] = self._meta
        with open(self.name, 'w') as outfile:
            json.dump(self._log, outfile)
        self.reset()
        
    def add_log(self, key, entry, aggregation_key = None, **kwargs):
        match = systems_match(entry, self._referential_equation) if self._referential_equation is not None else (0, 0, 0)
        try:
            mae = [np.mean(np.abs(eq.evaluate(False, True)[0])) for eq in entry]
        except KeyError:
            mae = 0
        
        log_entry = {'equation_form': entry.text_form,
                     'term_match': match,
                     'mae_train': mae,
                     'aggregation_key': aggregation_key
                     }

        if aggregation_key not in self._meta['aggregation_key']:
            self._meta['aggregation_key'].append(aggregation_key)
        log_entry = {**log_entry, **kwargs}
        self._log[key] = log_entry
        
class LogLoader(object):
    '''
    Object for the basic analytics of the equation discovery process
    '''
    def __init__(self, filename: Union[str, list]):
        self.reset()
        if isinstance(filename, str):
            file = open(filename, 'r')    
            self._log.append(json.load(file))
            file.close()
        else:
            for specific_filename in filename:
                file = open(specific_filename, 'r')    
                self._log.append(json.load(file))
                file.close()

    def reset(self):
        self._log = []
        self._variables = None
    
    @staticmethod
    def eq_analytics(equation_string: str):
        eq_terms = parse_equation_str(equation_string)
        term_stats = {frozenset(eq_terms[-1]) : 1.}
        for term in eq_terms[:-2]:
            term_stats[frozenset(term[1:])] = str(term[0])
        return term_stats

    def get_aggregation_keys(self):
        keys = []
        for log in self._log:
            keys.append(log['meta']['aggregation_key'])
        return keys

    def obtain_parsed_log(self, variables: list = ['u',], aggregation_key: tuple = None):
        if self._variables is None:
            self._variables = variables
        else:
            assert self._variables == variables
            
        def parse_system_str(system_string: str):
            def strap_cases(eq_string: str):
                return eq_string.replace(' / ', '').replace(' | ', '').replace(' \\ ', '')
            
            if '/' in system_string:
                return [strap_cases(eq_string) for eq_string in system_string.split(sep = '\n')[:-1]]
            else:
                return system_string.split(sep = '\n')[:-1]
        
        term_presence_log = {key : {} for key in range(len(variables))} # replaced self._term_presence_log
        for log_entry in self._log:
            for exp_key, exp_log in log_entry.items():
                if exp_key == 'meta':
                    continue
                
                if not 'aggregation_key' in exp_log.keys() or aggregation_key != exp_log['aggregation_key']:
                    continue
                
                
                system_list = parse_system_str(exp_log['equation_form'])
                assert len(variables) == len(system_list)
                for eq_idx in range(len(system_list)):
                    term_stats = self.eq_analytics(system_list[eq_idx])
                    for key, value in term_stats.items():
                        if key in term_presence_log[eq_idx].keys():
                            term_presence_log[eq_idx][key].append(float(value))
                        else:
                            term_presence_log[eq_idx][key] = [float(value),]
        return term_presence_log
        
    @staticmethod
    def get_stats(terms: Union[list, tuple], parsed_log: dict, metrics: list = [np.mean, np.var, np.size], 
                  metric_names: list = ['mean', 'var', 'disc_num'], variable_eq: str = 'u', variables = ['u',]):
        assert all([isinstance(term, frozenset) for term in terms])
        stats = []
        for term in terms:
            if term in parsed_log[variables.index(variable_eq)].keys():
                term_coeff_vals = np.array(parsed_log[variables.index(variable_eq)][term])
                stats.append([metric(np.abs(term_coeff_vals[term_coeff_vals != 0])) 
                              for metric in metrics])
            else:
                stats.append([np.nan for metric in metrics])

        stats = np.array(stats).reshape(-1)

        label_sep = '_'; term_sep = '*'
        term_names = [term_sep.join(term) for term in terms]
        labels = [label_sep.join(map(str, x)) for x in product(*[term_names, metric_names])]
        print(metric_names)
        print(labels)
        
        return labels, stats
    
    def to_pandas(self, terms: Union[list, tuple], metrics: list = [np.mean, np.var, np.size], metric_names: list = ['mean', 'var', 'disc_num'], 
                  variable: str = 'u', variables: list = ['u',]):
        metric_frames = []
        for log_entry in self._log:
            aggregation_keys = log_entry['meta']['aggregation_key']
            data = []
            row_labels = ['_'.join(map(str, key)) for key in aggregation_keys]
            for aggr_key in aggregation_keys:
                parsed_log = self.obtain_parsed_log(variables=variables, aggregation_key=aggr_key)
                keys, stats = self.get_stats(terms = terms, parsed_log = parsed_log, metrics = metrics, metric_names = metric_names, 
                                             variable_eq = variable, variables = variables)
                data.append({keys[idx] : stats[idx] for idx in range(len(keys))})
                
            
            metric_frames.append(pd.DataFrame(data, index = row_labels))
        return metric_frames
            
            