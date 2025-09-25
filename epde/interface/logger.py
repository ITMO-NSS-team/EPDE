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
    """
    Checks the structural similarity between two equations by comparing their terms.
    
        This method quantifies how well the term structure of one equation matches
        another. It identifies common terms, missing terms, and extra terms, providing
        a measure of structural agreement. This comparison is crucial for assessing
        the quality of discovered equations against a reference equation.
    
        Args:
            eq_checked: The equation to be checked.
            eq_ref: The reference equation.
    
        Returns:
            tuple: A tuple containing three integers:
                - matching: The number of terms in the reference equation that are also
                  present in the checked equation.
                - missing: The number of terms in the reference equation that are not
                  present in the checked equation.
                - extra: The number of terms in the checked equation that are not
                  present in the reference equation.
    
        WHY: This method is used to evaluate the discovered equations by comparing them to a reference equation.
    """
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
    """
    Compares the equations associated with each variable in two systems.
    
    This function is essential for evaluating the similarity between candidate equation systems
    generated during the equation discovery process. By comparing equations variable-by-variable,
    the evolutionary algorithm can effectively assess the fitness and diversity of the population.
    
    Args:
        checked_system: The system to be checked (SoEq object).
        reference_system: The reference system to compare against (SoEq object).
    
    Returns:
        A list of booleans, where each boolean indicates whether the equations
        for a corresponding variable match between the two systems. The order of booleans
        corresponds to the order of variables in `checked_system.vars_to_describe`.
    """
    return [equations_match(checked_system.vals[var], reference_system.vals[var]) 
            for var in checked_system.vars_to_describe]
        
class Logger():
    """
    A simple logger class for recording and dumping log data.
    
        Class Methods:
        - __init__
        - reset
        - dump
        - add_log
    
        Class Fields:
            _log (dict): An empty dictionary to store log data.
            _meta (dict): A dictionary to store metadata, initialized with an empty list for 'aggregation_key'.
            name (str): The name of the logger, updated only if the 'name' argument is provided.
    """

    def __init__(self, name, referential_equation = None, pool = None):
        """
        Initializes a Logger instance with a name and an optional referential equation.
        
        This method sets up the logger with a given name and associates it with
        a referential equation, which can be used to guide the equation discovery
        process. If the equation is provided as a string or dictionary, it is
        translated into a suitable format using a provided pool of tokens. This
        translation step ensures that the equation is compatible with the
        equation discovery algorithms.
        
        Args:
            name (str): The name of the logger.
            referential_equation (str, dict, or pre-translated equation, optional):
                An optional equation that serves as a reference point for the
                equation discovery process. Defaults to None.
            pool (list of Token, optional): A pool of tokens used for translating the
                equation if it's a string or dictionary. Required if
                `referential_equation` is a string or dictionary. Defaults to None.
        
        Raises:
            ValueError: If the referential equation is a string or dictionary,
                but the pool of tokens is missing, as translation is impossible.
        
        Returns:
            None
        
        Class Fields:
            _name (str): The name of the object, initialized by `reset()`.
            _referential_equation: The referential equation associated with the object.
        """
        self.reset(name = name)
        if isinstance(referential_equation, str) or isinstance(referential_equation, dict):
            if pool is None:
                raise ValueError('Can not translate equations: the pool of tokens is missing')
            referential_equation = translate_equation(referential_equation, pool)
        self._referential_equation = referential_equation
        
    def reset(self, name = None):
        """
        Resets the logger to a clean state, preparing it for a new equation discovery task.
        
                This method clears all stored log data and metadata, effectively starting a new search. It also allows for renaming the logger, which can be useful for tracking different equation discovery runs or experiments.
        
                Args:
                    name (str, optional): A new name for the logger. If None, the current name is retained. Defaults to None.
        
                Returns:
                    None
        
                This method ensures that previous equation search results do not interfere with new ones by clearing the internal state.
                It initializes the following class fields:
                  _log (dict): An empty dictionary to store log data.
                  _meta (dict): A dictionary to store metadata, initialized with an empty list for 'aggregation_key'.
                  name (str): The name of the logger, updated only if the 'name' argument is provided.
        """
        try:
            self.log_out()
        except AttributeError:
            pass
        
        self._log = {}
        self._meta = {'aggregation_key' : []}
        if name is not None:
            self.name = name
        
    def dump(self):
        """
        Dumps the current log data to a JSON file and resets the log.
        
                This method serializes the internal log data, including metadata,
                into a JSON file specified by the object's name attribute. After
                dumping the data, it resets the internal log to prepare for new entries.
                This is done to persist the log data for later analysis or use in equation discovery.
        
                Args:
                    self: The Logger instance.
        
                Returns:
                    None.
        """
        self._log['meta'] = self._meta
        with open(self.name, 'w') as outfile:
            json.dump(self._log, outfile)
        self.reset()
        
    def add_log(self, key, entry, aggregation_key = None, **kwargs):
        """
        Adds a new equation to the internal log, tracking its characteristics and performance.
        
                This method stores the equation's textual form, term matching statistics,
                mean absolute error (MAE) on the training data, and an optional aggregation key
                for grouping similar equations. This information is crucial for analyzing
                the population of discovered equations and selecting the most promising candidates
                during the evolutionary search process.
        
                Args:
                    key: A unique identifier for the equation being logged.
                    entry: The equation object to log.
                    aggregation_key: An optional key for grouping similar equations.
                    **kwargs: Additional keyword arguments to store in the log entry.
        
                Returns:
                    None.
        """
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
        """
        Initializes the LogLoader, reading log data from specified file(s) to prepare for equation discovery.
        
                This method loads log data, which serves as the foundation for identifying underlying differential equations.
                It handles both single and multiple log files, appending the data from each into a unified log structure.
        
                Args:
                    filename (Union[str, list]): The path to a single log file or a list of paths to multiple log files.
        
                Returns:
                    None: The method initializes the LogLoader instance and loads data, but does not return any value.
        """
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
        """
        Resets the internal state, preparing for a new data loading and equation discovery cycle.
        
                This method clears the current log and any stored variables, ensuring a clean slate
                for subsequent data processing and equation search operations. This is crucial for
                isolating the impact of each data loading attempt and avoiding contamination from
                previous states.
        
                Args:
                    self: The object instance.
        
                Returns:
                    None. This method modifies the object's internal state.
        
                Initializes:
                    _log (list): An empty list to store log messages.
                    _variables (None): A variable to store variables, initialized to None.
        """
        self._log = []
        self._variables = None
    
    @staticmethod
    def eq_analytics(equation_string: str):
        """
        Analyzes an equation string to extract term statistics for equation discovery.
        
                This method parses an equation string, extracts the terms, and
                calculates statistics based on the terms. It uses the
                `parse_equation_str` function to split the equation into its
                constituent terms. The statistics are stored in a dictionary where
                the keys are frozensets of the terms (excluding the coefficient for
                terms on the left-hand side) and the values are either the
                coefficient (as a string) or 1.0 for the right-hand side term.
                This information is crucial for identifying potential terms and their
                contributions within the equation.
        
                Args:
                    equation_string: The equation string to analyze.  The expected
                        format is a string where terms are separated by ' + ' on the
                        left-hand side and ' * ' within each term. The left and right
                        sides are separated by ' = '.
        
                Returns:
                    dict: A dictionary containing term statistics. The keys are
                        frozensets of strings representing the terms (excluding the
                        coefficient for terms on the left-hand side), and the values
                        are either the coefficient (as a string) or 1.0 for the
                        right-hand side term.
        """
        eq_terms = parse_equation_str(equation_string)
        term_stats = {frozenset(eq_terms[-1]) : 1.}
        for term in eq_terms[:-2]:
            term_stats[frozenset(term[1:])] = str(term[0])
        return term_stats

    def get_aggregation_keys(self):
        """
        Retrieves the aggregation keys from the log entries.
        
        This method iterates through the internal log data and extracts the
        'aggregation_key' from the 'meta' dictionary of each log entry.
        The aggregation keys are essential for grouping and processing log entries
        that share common characteristics, enabling the framework to identify
        patterns and relationships within the data relevant to discovering
        governing differential equations.
        
        Args:
            self: The object instance.
        
        Returns:
            list: A list of aggregation keys extracted from the log entries.
        """
        keys = []
        for log in self._log:
            keys.append(log['meta']['aggregation_key'])
        return keys

    def obtain_parsed_log(self, variables: list = ['u',], aggregation_key: tuple = None):
        """
        Obtains and parses the system log to extract term statistics for equation discovery.
        
                This method processes log entries, extracts equation forms, and analyzes terms
                based on specified variables and an aggregation key. The extracted term statistics
                are essential for identifying candidate equation structures within the EPDE framework.
                It filters log entries based on the aggregation key and parses the equation forms
                associated with each entry. The term statistics are then aggregated to provide
                a comprehensive view of term presence and their values across different equations.
                This information is crucial for the evolutionary algorithm to effectively search
                for the best-fitting differential equation.
        
                Args:
                    variables: A list of variable names to consider during parsing. Defaults to ['u'].
                    aggregation_key: A tuple representing the aggregation key to filter log entries.
        
                Returns:
                    dict: A dictionary containing term presence logs, where keys are variable indices
                        and values are dictionaries of term statistics (term -> list of values).
                        The term statistics are represented as lists of floating-point values.
        """
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
        """
        Calculates statistical features of equation terms based on their coefficients in the parsed log data.
        
                This method extracts coefficient values for each specified term from the parsed log, computes statistical
                metrics on the absolute values of these coefficients (excluding zeros), and returns the calculated
                statistics along with corresponding labels. This helps to quantify the importance and characteristics
                of each term within the discovered equation.
        
                Args:
                    terms: A list or tuple of frozensets, where each frozenset represents a term in the equation.
                    parsed_log: A dictionary containing parsed log data, with keys corresponding to variable indices and terms.
                    metrics: A list of functions to apply as metrics (default: [np.mean, np.var, np.size]).
                    metric_names: A list of names corresponding to the metrics (default: ['mean', 'var', 'disc_num']).
                    variable_eq: The variable to extract from the parsed log (default: 'u').
                    variables: List of variables to consider
        
                Returns:
                    tuple: A tuple containing two lists:
                        - labels: A list of strings representing the labels for each statistic, constructed from term names and metric names.
                        - stats: A list of calculated statistics, providing a numerical representation of term characteristics.
        """
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
        """
        Converts the internal log data into a list of Pandas DataFrames, facilitating subsequent equation discovery.
        
                This method transforms the raw log data into a structured format suitable for analysis by EPDE's equation discovery algorithms. It aggregates data based on specified terms and metrics, constructing a Pandas DataFrame for each log entry. This aggregation is crucial for summarizing the data and extracting relevant features that can be used to identify underlying differential equations.
        
                Args:
                    terms: The terms to use for aggregation, guiding the feature extraction process.
                    metrics: A list of metric functions to apply (e.g., mean, variance). Defaults to [np.mean, np.var, np.size]. These metrics quantify different aspects of the data relevant to equation discovery.
                    metric_names: A list of names corresponding to the metrics. Defaults to ['mean', 'var', 'disc_num'].
                    variable: The primary variable to analyze. Defaults to 'u'.
                    variables: A list of variables to include in the analysis. Defaults to ['u'].
        
                Returns:
                    list: A list of Pandas DataFrames, where each DataFrame represents a log entry and contains aggregated statistics for the specified terms and metrics. These DataFrames serve as input for the equation discovery process.
        """
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