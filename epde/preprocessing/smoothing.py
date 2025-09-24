#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 19:33:43 2020

@author: mike_ubuntu
"""

import numpy as np
from scipy.ndimage import gaussian_filter


def smoothing(data, kernel_fun, **kwargs):
    """
    Applies a smoothing filter to the input data, enhancing the signal and reducing noise before equation discovery.
    
        Args:
            data (np.ndarray): The data to be smoothed.
            kernel_fun (str): The type of smoothing kernel to use (e.g., 'gaussian'). Currently only 'gaussian' is supported.
            **kwargs: Keyword arguments to pass to the smoothing function.
                Must include 'sigma' for gaussian kernel. If 'include_time' is False and the data has more than one dimension, smoothing is applied along each time slice.
    
        Returns:
            np.ndarray: The smoothed data.
    """
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
