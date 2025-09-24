#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 19:16:50 2020

@author: mike_ubuntu
"""

import numpy as np
from warnings import warn
from epde.preprocessing.fin_der import FDderivatives


def get_cheb_for_point(matrix, axis, idx, grid, max_der_order=3, points=9, poly_order=None):
    """
    Computes a Chebyshev polynomial fit at a specific location within a multi-dimensional array.
    
        This method isolates a subset of data points along a given axis centered on a specified index,
        and subsequently fits a Chebyshev polynomial to these points. This provides a smooth,
        continuous representation of the data in the neighborhood of the point, which can be used
        for tasks such as estimating derivatives or interpolating values.
    
        Args:
            matrix (numpy.ndarray): The multi-dimensional array containing the data.
            axis (int): The axis along which to extract the data points.
            idx (tuple or array-like): A tuple or array of indices specifying the location of the point.
            grid (list of numpy.ndarray): A list of arrays, where each array represents the coordinate grid for each axis.
            max_der_order (int, optional): The maximum derivative order to consider when automatically determining the polynomial order. Defaults to 3.
            points (int, optional): The number of points to use for the Chebyshev fit. Defaults to 9.
            poly_order (int, optional): The order of the Chebyshev polynomial. If None, it is determined based on max_der_order. Defaults to None.
    
        Returns:
            tuple: A tuple containing:
                - The x-coordinate of the central point (float).
                - The Chebyshev polynomial object (numpy.polynomial.chebyshev.Chebyshev) fitted to the data.
    """
    if poly_order is None:
        max_power = max_der_order + 1
    else:
        max_power = poly_order
        
    if points == max_power:
        warn('Overfitting of the Chebyshev polynomial, that tries to represent the data, reducing order by one.')
        max_power -= 1
        
    I = np.array([int(-(points-1)/2 + i)
                 for i in np.arange(points)]) + idx[axis]
    F = matrix.take(I, axis=axis)
    x_raw = grid[axis].take(I, axis=axis)
    for i in range(idx.size):
        if i < axis:
            F = F.take(idx[i], axis=0)
            x_raw = x_raw.take(idx[i], axis=0)
        elif i > axis:
            F = F.take(idx[i], axis=1)
            x_raw = x_raw.take(idx[i], axis=1)

    poly = np.polynomial.chebyshev.Chebyshev.fit(x_raw, F, max_power)
    return x_raw[int(x_raw.size/2.)], poly


def process_point_cheb(args):
    """
    Processes a single point to calculate derivatives using Chebyshev polynomials or finite differences.
    
        This method strategically chooses between Chebyshev polynomials and finite difference approximations
        to compute derivatives at a specific point within a multi-dimensional data tensor. It prioritizes
        Chebyshev polynomials for points within a defined boundary to leverage their accuracy and efficiency,
        reverting to finite differences for points outside this boundary. This hybrid approach ensures
        accurate derivative calculations across the entire data domain.
    
        Args:
            args: A list containing the following elements:
                idx (numpy.ndarray): The index of the point in the matrix.
                matrix (numpy.ndarray): The multi-dimensional data tensor.
                grid (numpy.ndarray): The grid spacing for each dimension.
                points (numpy.ndarray): The points at which to evaluate the Chebyshev polynomials.
                n_der (int or tuple/list of int): The order of derivatives to calculate for each dimension.
                poly_bound (int): The boundary within which to use Chebyshev polynomials.
                poly_order (int): The order of the Chebyshev polynomials to use.
    
        Returns:
            numpy.ndarray: An array containing the calculated derivatives. The length of the array is the sum of the
                derivative orders for each dimension.
    """
    global PolyBoundary
    idx = np.array(args[0])
    matrix = args[1]
    grid = args[2]
    points = args[3]
    n_der = args[4]
    poly_bound = args[5]
    poly_order = args[6]
#    print(args[0])

    if isinstance(n_der, int):
        n_der = tuple([n_der for i in range(matrix.ndim)])
    elif isinstance(n_der, (list, tuple)):
        assert len(n_der) == matrix.ndim, 'Given derivative orders do not match the data tensor dimensionality'
    else:
        raise TypeError(
            'Derivatives were given in the incorrect format. A single integer or list/tuple of integers required')

    poly_mask = [idx[dim] >= poly_bound and idx[dim] <= matrix.shape[dim] - poly_bound for dim in np.arange(matrix.ndim)]
    polynomials = np.empty(matrix.ndim, dtype=np.polynomial.chebyshev.Chebyshev)
    x = np.empty(idx.shape)
    for i in range(matrix.ndim):
        if poly_mask[i]:
            x_temp, poly_temp = get_cheb_for_point(matrix, i, idx, grid, max_der_order=n_der[i], 
                                                   points=points, poly_order=poly_order)
            x[i] = x_temp
            polynomials[i] = poly_temp

    derivatives = np.empty(sum(n_der))

    deriv_idx = 0
    for var_idx in np.arange(matrix.ndim):
        if poly_mask[var_idx]:
            for der_idx in np.arange(1, n_der[var_idx]+1):
                derivatives[deriv_idx] = polynomials[var_idx].deriv(m=der_idx)(x[var_idx])
                deriv_idx += 1
        else:
            derivatives[deriv_idx: deriv_idx + n_der[var_idx]] = FDderivatives(matrix, axis=var_idx, idx=idx, grid=grid, 
                                                                               max_order=n_der[var_idx], poly_bound=poly_bound)
            deriv_idx += n_der[var_idx]
#    print(derivatives.shape)
#    print('derivatives length', len(derivatives), 'type', type(derivatives))
    return (derivatives)
