#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 18:20:23 2022

@author: maslyaev
"""

import numpy as np
from scipy.special import jv, yv
from sklearn.linear_model import LinearRegression


class BesselInterpolator(object):
    """
    Represents an interpolator using Bessel functions.
    
        This class approximates a function using a linear combination of Bessel
        functions of the first kind. It fits the coefficients of the Bessel
        functions to the given data.
    
        Class Methods:
        - __init__
        - bf_vals
        - approximate
    
        Class Fields:
            x (np.ndarray): The independent variable data.
            data (np.ndarray): The dependent variable data.
            max_order (int): The maximum order of the Bessel functions to use.
            coef_ (np.ndarray): The coefficients of the fitted Bessel function expansion.
    """

    def __init__(self, x, data, max_order=5):
        """
        Initializes the interpolator by fitting a linear regression model to the data using Bessel basis functions.
        
                This method prepares the interpolator to approximate a function by determining the coefficients
                that best combine the Bessel basis functions to fit the provided data. The linear regression model
                is fitted using the provided 'x' values as the independent variable and 'data' as the dependent variable.
                Fitting happens during initialization to precompute regression coefficients for faster interpolation later.
        
                Args:
                    x (np.ndarray): The independent variable data (1D array).
                    data (np.ndarray): The dependent variable data (array with the same shape as x).
                    max_order (int): The maximum order of the Bessel functions to use as basis functions (default is 5).
        
                Returns:
                    None
        
                Class Fields:
                    max_order (int): The maximum order of the Bessel basis functions.
                    coef_ (np.ndarray): The coefficients of the fitted linear regression model, representing the weights of each basis function.
        """
        assert x.ndim == 1, 'Prototype for 1D - data'
        assert data.shape == x.shape

        self.max_order = max_order
        model = LinearRegression(fit_intercept=False)
        A = np.vstack(self.bf_vals(x)).T
        # print(A.shape)
        model.fit(A, data)
        self.coef_ = model.coef_

    def bf_vals(self, arg):
        """
        Calculates Bessel function values required for basis function expansion.
        
                This method computes a list of Bessel function values of the first kind (jv)
                for orders 0 up to `self.max_order`, evaluated at the input `arg`. These values
                serve as the foundation for constructing a basis set used in representing
                solutions to differential equations. A list of ones is appended to provide
                a constant term in the basis.
        
                Args:
                    arg: The argument at which to evaluate the Bessel functions.
                         Can be a scalar (int or float) or a NumPy array.
        
                Returns:
                    A list of NumPy arrays. The first `self.max_order` elements are the
                    Bessel function values of the first kind for orders 0 to
                    `self.max_order - 1`. The last element is a NumPy array of ones
                    with the same size as `arg` if `arg` is an array, or a list containing
                    a single float 1.0 if `arg` is a scalar.
        """
        ones = [1.,] if isinstance(arg, (int, float)) else [np.ones(arg.size),]
        res = ([jv(order, arg) for order in range(self.max_order)] +  # [yv(order, arg) for order in range(self.max_order)] +
               ones)  # Конкат. листов - неоптимальная операция
        # print('res', len(res))
        return res

    def approximate(self, point):
        """
        Approximates the function value at a specified point using a linear combination of basis functions.
        
        This method leverages pre-computed coefficients and basis function values to efficiently estimate the function's value at the given point. This is a core step in reconstructing the underlying function based on its representation in the basis function space.
        
        Args:
            point (float or array-like): The point at which to evaluate the approximation.
        
        Returns:
            float: The approximated function value at the given point.
        """
        return np.dot(self.coef_, np.array(self.bf_vals(point)))
