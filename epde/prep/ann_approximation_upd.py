#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  2 15:27:56 2021

@author: maslyaev
"""

import time
from collections import Counter

import numpy as np
import torch
import copy
from scipy.interpolate import RegularGridInterpolator

     
class Differentiable_Function(object):
    def __init__(self, function : torch.nn.modules.container.Sequential, history : list = []):
        self.function = function
        self.deriv_history = history
        
    def differentiate(self, data, axes_names, orders = 1):
        function_evald = self.function(data)
        
        # gradient = tf.gradients(ys=self.function, xs=data)
        if orders == 1:
            deriv_history = [self.deriv_history + [axis_name] for axis_name in axes_names] 
            return gradient, deriv_history
        else:
            derivatives = []; deriv_history = []
            for idx in np.arange(len(data)):
                hist = self.deriv_history + [axes_names[idx]]
                temp_fun = Differentiable_Function(gradient[idx], history = hist)
                der_fun, history = temp_fun.differentiate(data, axes_names, orders = orders - 1)
                derivatives.extend(der_fun); deriv_history.extend(history)
            return derivatives, deriv_history
        
        