#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 19:18:48 2020

@author: mike_ubuntu
"""

import numpy as np


def FDderivatives(matrix, axis, idx, grid, max_order, poly_bound):
    """
    Calculates finite difference derivatives of a matrix along a specified axis at a given index.
    
        This method approximates derivatives using finite differences. It selects a stencil of points
        around the given index along the specified axis and uses these points to compute the derivatives.
        The order of the derivatives calculated is limited to a maximum of 3. This function is used to estimate the rate of change of the data, which is a crucial step in identifying the underlying differential equations.
    
        Args:
            matrix (np.ndarray): The input matrix for which to calculate derivatives.
            axis (int): The axis along which to calculate the derivatives.
            idx (tuple): The index at which to calculate the derivatives.
            grid (np.ndarray): The grid coordinates corresponding to the matrix.
            max_order (int): The maximum order of derivatives to calculate (limited to 3).
            poly_bound (int): A boundary parameter that defines the region near the edges where a different stencil is used.
    
        Returns:
            np.ndarray: An array containing the calculated derivatives up to the specified `max_order`. These derivatives serve as features for discovering the underlying differential equation.
    """
    assert idx[axis] < poly_bound or idx[axis] > matrix.shape[axis] - poly_bound
    if idx[axis] < poly_bound:
        I = idx[axis] + np.arange(6)
    else:
        I = idx[axis] - np.arange(6)

    x = grid[axis].take(I, axis=axis)
    F = matrix.take(I, axis=axis)
    for i in range(idx.size):
        if i < axis:
            F = F.take(idx[i], axis=0)
            x = x.take(idx[i], axis=0)
        elif i > axis:
            #            print(i, idx, F.shape, x_raw.shape)
            F = F.take(idx[i], axis=1)
            x = x.take(idx[i], axis=1)

    derivatives = np.empty(3)
    derivatives[0] = (F[1] - F[0]) / (x[1] - x[0])
    derivatives[1] = (2*F[0] - 5*F[1] + 4*F[2] - F[3]) / (x[1] - x[0]) ** 2
    derivatives[2] = (-2.5*F[0] + 9*F[1] - 12*F[2] + 7*F[3] - 1.5*F[4]) / (x[1] - x[0]) ** 3
    if max_order > 3:
        raise ValueError(
            'Attempting to calculate derivatives up to order higher, than 3. Option not implemented yet.')
    return derivatives[:max_order]
