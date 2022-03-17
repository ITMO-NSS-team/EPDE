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
    def __init__(self, x, data, max_order = 5):
        assert x.ndim == 1, 'Prototype for 1D - data'
        assert data.shape == x.shape

        self.max_order = max_order
        model = LinearRegression(fit_intercept=False)
        A = np.vstack(self.bf_vals(x)).T
        # print(A.shape)
        model.fit(A, data)
        self.coef_ = model.coef_

    def bf_vals(self, arg):
        ones = [1.,] if isinstance(arg, (int, float)) else [np.ones(arg.size),]
        res = ([jv(order, arg) for order in range(self.max_order)] +  #[yv(order, arg) for order in range(self.max_order)] + 
               ones) # Конкат. листов - неоптимальная операция
        # print('res', len(res))
        return res

    def approximate(self, point):
        return np.dot(self.coef_, np.array(self.bf_vals(point)))