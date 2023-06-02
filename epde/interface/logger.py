#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 16:21:21 2023

@author: maslyaev
"""

import numpy as np

import json
from epde.structure.main_structures import SoEq, Equation

def equations_match(eq_checked: Equation, eq_ref: Equation):
    def parse_equation(equation: Equation):
        return {frozenset({(factor.label, factor.param(name = 'power')) 
                           for factor in term.structure}) 
                for term in equation.structure}
    
    checked_parsed = parse_equation(eq_checked)
    ref_parsed = parse_equation(eq_ref)
    
    matching = sum([term in checked_parsed for term in ref_parsed])
    missing = len(ref_parsed) - matching
    extra = sum([term not in ref_parsed for term in checked_parsed])
    return (matching, missing, extra)

def systems_match(checked_system: SoEq, reference_system: SoEq):
    return [equations_match(checked_system.vars[var], reference_system.vars[var]) 
            for var in checked_system.vars_to_describe]
        
class Logger():
    def __init__(self, name, referential_equation = None):
        self.reset(name = name)
        self._referential_equation = referential_equation
        
    def reset(self, name = None):
        try:
            self.log_out()
        except AttributeError:
            pass
        
        self._log = {}
        self.name = name
        
    def dump(self):
        # json_str = json.dumps(self.log)
        with open(self.name, 'w') as outfile:
            outfile.dump(self._log, outfile)
        
    def add_log(self, key, entry, **kwargs):
        log_entry = {'equation_form' : entry.text_form,
                     'term_match' : systems_match(entry, self._referential_equation),
                     'mae_train' : np.mean(np.abs(entry.evaluate(False, True)[0]))}

        log_entry = {**log_entry, **kwargs}
        self._log[key] = log_entry