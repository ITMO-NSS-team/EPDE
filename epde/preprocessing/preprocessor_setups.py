#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 20:13:46 2022

@author: maslyaev
"""

from epde.preprocessing.smoothers import GaussianSmoother, ANNSmoother, PlaceholderSmoother
from epde.preprocessing.deriv_calculators import AdaptiveFiniteDeriv, PolynomialDeriv, SpectralDeriv

from epde.preprocessing.preprocessor import ConcretePrepBuilder



class PreprocessorSetup:
    def __init__(self):
        self._builder = None

    @property
    def builder(self):
        return self._builder

    @builder.setter
    def builder(self, builder: ConcretePrepBuilder):
        self._builder = builder

    def build_ANN_preprocessing(self, test_output=False, epochs_max=1e5,
                                loss_mean=1000, batch_frac=0.8):
        smoother_args = ()
        smoother_kwargs = {'grid': None, 'epochs_max': epochs_max,
                           'loss_mean': loss_mean, 'batch_frac': batch_frac}

        deriv_calculator_args = ()
        deriv_calculator_kwargs = {'grid': None}

        self.builder.set_smoother(ANNSmoother, *smoother_args, **smoother_kwargs)
        self.builder.set_deriv_calculator(AdaptiveFiniteDeriv, *deriv_calculator_args,
                                          **deriv_calculator_kwargs)

    def build_spectral_preprocessing(self, n=None, steepness=1):
        smoother_args = ()
        smoother_kwargs = {}

        deriv_calculator_args = ()
        deriv_calculator_kwargs = {'grid': None, 'n': n, 'steepness': steepness}

        self.builder.set_smoother(PlaceholderSmoother, *smoother_args, **smoother_kwargs)
        self.builder.set_deriv_calculator(SpectralDeriv, *deriv_calculator_args,
                                          **deriv_calculator_kwargs)

    def build_poly_diff_preprocessing(self, use_smoothing=False, sigma=1, mp_poolsize=4,
                                      polynomial_window=9, poly_order=None, include_time=False):
        smoother_args = ()
        smoother_kwargs = {'sigma': sigma, 'include_time' : include_time}

        deriv_calculator_args = ()
        deriv_calculator_kwargs = {'grid': None, 'mp_poolsize': mp_poolsize, 'polynomial_window': polynomial_window,
                                   'poly_order': poly_order}

        smoother = GaussianSmoother if use_smoothing else PlaceholderSmoother
        self.builder.set_smoother(smoother, *smoother_args, **smoother_kwargs)
        self.builder.set_deriv_calculator(PolynomialDeriv, *deriv_calculator_args,
                                          **deriv_calculator_kwargs)

    # def build_spectral_deriv_preprocessing(self, *args, **kwargs):
    #     pass

    # def build_with_custom_deriv()
