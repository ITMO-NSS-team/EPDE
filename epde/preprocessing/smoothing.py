#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 19:33:43 2020

@author: mike_ubuntu
"""

import numpy as np
from scipy.ndimage import gaussian_filter


def Smoothing(data, kernel_fun, **params):
    smoothed = np.empty(data.shape)
    if kernel_fun == 'gaussian':
        if np.ndim(data) > 1 or params['include_time']:
            for time_idx in np.arange(data.shape[0]):
                if np.ndim(data) == 3:
                    smoothed[time_idx, :, :] = gaussian_filter(data[time_idx, :, :], sigma=params['sigma'])
                elif np.ndim(data) == 2:
                    smoothed[time_idx, :] = gaussian_filter(data[time_idx, :], 
							      sigma=params['sigma'])
        else:  # if np.ndim(data) == 1:
            smoothed = gaussian_filter(data, sigma=params['sigma'])
    else:
        raise Exception('Wrong kernel passed into function')

    return smoothed
