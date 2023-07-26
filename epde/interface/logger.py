#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 16:21:21 2023

@author: maslyaev
"""

import numpy as np
import json

from epde.structure.main_structures import SoEq, Equation
from epde.interface.equation_translator import translate_equation

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
        if name is not None:
            self.name = name
        
    def dump(self):
        with open(self.name, 'w') as outfile:
            json.dump(self._log, outfile)
        self.reset()
        
    def add_log(self, key, entry, **kwargs):
        match = systems_match(entry, self._referential_equation) if self._referential_equation is not None else (0, 0, 0)
        try:
            mae = [np.mean(np.abs(eq.evaluate(False, True)[0])) for eq in entry]
        except KeyError:
            mae = 0
        
        log_entry = {'equation_form': entry.text_form,
                     'term_match': match,
                     'mae_train': mae
                     }

        log_entry = {**log_entry, **kwargs}
        self._log[key] = log_entry