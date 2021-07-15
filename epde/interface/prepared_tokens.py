#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 14 13:26:56 2021

@author: mike_ubuntu
"""
import numpy as np
from abc import ABC
from collections import OrderedDict

from epde.interface.token_family import Token_family
from epde.evaluators import trigonometric_evaluator

class Prepared_tokens(ABC):
    def __init__(self, *args, **kwargs):
        self._token_family = Token_family(token_type = 'Placeholder')
        
    @property
    def token_family(self):
        if not (self._token_family.evaluator_set and self._token_family.params_set):
            raise AttributeError(f'Some attributes of the token family have not been declared.')
        return self._token_family
    
    
class Trigonometric_tokens(Prepared_tokens):
    def __init__(self, freq : tuple = (np.pi/2., 2*np.pi), dimensionality = 1):
        assert freq[1] > freq[0] and len(freq) == 2, 'The tuple, defining frequncy interval, shall contain 2 elements with first - the left boundary of interval and the second - the right one. '

        self._token_family = Token_family(token_type='trigonometric')
        self._token_family.set_status(unique_specific_token=True, unique_token_type=True, 
                           meaningful = False, unique_for_right_part = False)
        
        trig_token_params = OrderedDict([('power', (1, 1)), 
                                         ('freq', freq), 
                                         ('dim', (0, dimensionality))])
        freq_equality_fraction = 0.02 # fraction of allowed frequency interval, that is considered as the same
        trig_equal_params = {'power' : 0, 'freq' : (freq[1] - freq[0]) / freq_equality_fraction, 
                             'dim' : 0}
        self._token_family.set_params(['sin', 'cos'], trig_token_params, trig_equal_params)
        self._token_family.set_evaluator(trigonometric_evaluator, [])
        
        
class Logfun_tokens(Prepared_tokens):
    def __init__(self, *args, **kwargs):
        raise NotImplementedError
        