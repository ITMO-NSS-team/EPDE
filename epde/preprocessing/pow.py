#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 19:29:43 2020

@author: mike_ubuntu
"""

import numpy as np
from edpe.prep.fin_der import FD_derivatives
from sklearn.linear_model import LinearRegression

def Get_LSQ_for_point(matrix, axis, idx, grid, max_der_order = 3, points = 9):
    max_power = max_der_order + 1
    I = np.array([np.int(-(points-1)/2 + i) for i in np.arange(points)]) + idx[axis]
    F = matrix.take(I , axis = axis)
    x_raw = grid[axis].take(I, axis = axis)
    for i in range(idx.size):
        if i < axis:
            F = F.take(idx[i], axis = 0)
            x_raw = x_raw.take(idx[i], axis = 0)            
        elif i > axis:
            F = F.take(idx[i], axis = 1)
            x_raw = x_raw.take(idx[i], axis = 1)     
            
    X = np.array([np.power(x_raw[0], max_power - i) for i in np.arange(1, max_power)] + [1])
    for j in np.arange(1, points):
        X = np.vstack((X, np.array([np.power(x_raw[j], max_power - i) for i in np.arange(1, max_power)] + [1])))
    estimator = LinearRegression()
    estimator.fit(X, F)
    return x_raw[int(x_raw.size/2.)], np.flip(np.array(estimator.coef_)[:-1])


def Process_Point_LSQ(args):
    global PolyBoundary
    idx = np.array(args[0]); matrix = args[1]; grid = args[2]; points = args[3]; n_der = args[4]
    print(args[0])
    poly_mask = [idx[dim] >= PolyBoundary and idx[dim] <= matrix.shape[dim] - PolyBoundary for dim in np.arange(matrix.ndim)]
    coeffs = np.empty((matrix.ndim, n_der))
    x = np.empty(idx.shape)
    for i in range(coeffs.shape[0]):
        if poly_mask[i]:
            x_temp, coeffs_temp = Get_LSQ_for_point(matrix, i, idx, grid, max_der_order=n_der, points = points)
            x[i] = x_temp
            coeffs[i, :] = coeffs_temp 

    derivatives = np.empty(coeffs.shape[0] * (n_der))
    for var_idx in np.arange(coeffs.shape[0]):
        if poly_mask[var_idx]:
            for der_idx in np.arange(1, n_der+1):
                derivatives[var_idx*(n_der) + (der_idx-1)] = np.sum([coeffs[var_idx, j-1] * np.math.factorial(j)/
                                     np.math.factorial(j - der_idx) * x[var_idx] ** (j - der_idx) for j in range(der_idx, n_der+1)])          
        else:
            derivatives[var_idx*(n_der) : (var_idx+1)*(n_der)] = FD_derivatives(matrix, axis = var_idx, idx = idx, grid = grid, max_order = n_der)
    return(derivatives)
