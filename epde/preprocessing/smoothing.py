#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 19:33:43 2020

@author: mike_ubuntu
"""

import numpy as np
from scipy.ndimage import gaussian_filter


def smoothing(data, kernel_fun, **kwargs):
    smoothed_data = np.empty_like(data)        
    if kernel_fun == 'gaussian':
        if kernel_fun == 'gaussian':
            if not kwargs['include_time'] and np.ndim(data) > 1:
                for time_idx in np.arange(data.shape[0]):
                    smoothed_data[time_idx, ...] = gaussian_filter(data[time_idx, ...],
                                                                   sigma=kwargs['sigma'])
            else:
                smoothed_data = gaussian_filter(data, sigma=kwargs['sigma'])
    else:
        raise Exception('Wrong kernel passed into function')

    return smoothed_data
