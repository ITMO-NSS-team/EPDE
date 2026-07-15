#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 19:18:48 2020

@author: mike_ubuntu
"""

import numpy as np


def FDderivatives(matrix, axis, idx, grid, max_order, poly_bound):
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

    # One-sided stencils on the 6 signed-spacing points above. Direction
    # reversal at the right boundary is handled by the signed step
    # x[1]-x[0]: even orders divide by an even power (stencil unchanged),
    # odd orders by an odd power (stencil negation absorbed by the sign).
    derivatives = np.empty(5)
    derivatives[0] = (F[1] - F[0]) / (x[1] - x[0])
    derivatives[1] = (2*F[0] - 5*F[1] + 4*F[2] - F[3]) / (x[1] - x[0]) ** 2
    derivatives[2] = (-2.5*F[0] + 9*F[1] - 12*F[2] + 7*F[3] - 1.5*F[4]) / (x[1] - x[0]) ** 3
    derivatives[3] = (3*F[0] - 14*F[1] + 26*F[2] - 24*F[3] + 11*F[4] - 2*F[5]) / (x[1] - x[0]) ** 4
    derivatives[4] = (-F[0] + 5*F[1] - 10*F[2] + 10*F[3] - 5*F[4] + F[5]) / (x[1] - x[0]) ** 5
    if max_order > 5:
        raise ValueError(
            'Attempting to calculate derivatives up to order higher, than 5. Option not implemented yet.')
    return derivatives[:max_order]
