#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 19:18:48 2020

@author: mike_ubuntu
"""

import numpy as np

def FD_derivatives(matrix, axis, idx, grid, max_order, poly_bound):
    assert idx[axis] < poly_bound or idx[axis] > matrix.shape[axis] - poly_bound
    if idx[axis] < poly_bound:
        I = idx[axis] + np.arange(6) 
    else:
        I = idx[axis] - np.arange(6)
    
    x = grid[axis].take(I, axis = axis)
    F = matrix.take(I , axis = axis)    
    for i in range(idx.size):
        if i < axis:
            F = F.take(idx[i], axis = 0)
            x = x.take(idx[i], axis = 0)            
        elif i > axis:
#            print(i, idx, F.shape, x_raw.shape)
            F = F.take(idx[i], axis = 1)
            x = x.take(idx[i], axis = 1)     

    derivatives = np.empty(3)            
    derivatives[0] = (F[1] - F[0]) / (x[1] - x[0])
    derivatives[1] = (2*F[0] - 5*F[1] + 4*F[2] - F[3]) / (x[1] - x[0]) ** 2
    derivatives[2] = (-2.5*F[0] + 9*F[1] - 12*F[2] + 7*F[3] - 1.5*F[4]) / (x[1] - x[0]) ** 3
    if max_order > 3:
        raise ValueError('Attempting to calculate derivatives up to order higher, than 3. Option not implemented yet.')
    return derivatives[:max_order]        
        