#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 20:13:46 2022

@author: maslyaev
"""

from epde.preprocessing.smoothers import GaussianSmoother, ANNSmoother, PlaceholderSmoother
from epde.preprocessing.deriv_calculators import (AdaptiveFiniteDeriv, BackwardIncrementDeriv,
                                                  ForwardIncrementDeriv, PolynomialDeriv, SpectralDeriv,
                                                  _check_accuracy)

#: ``FD``'s rule along the time axis: central (as every other axis), or a one-sided increment.
FD_SCHEMES = {'central': AdaptiveFiniteDeriv, 'forward': ForwardIncrementDeriv,
              'backward': BackwardIncrementDeriv}

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

    def build_FD_preprocessing(self, scheme='central', accuracy=4, lag=1, time_axis=0):
        """Unsmoothed data, finite differences.

        ``scheme`` is the rule along the time axis ``time_axis``: ``'central'`` (as every other
        axis; ``AdaptiveFiniteDeriv``), ``'forward'`` -- the increment ``(u[k+lag] - u[k]) / (lag
        dt)``, which estimates the drift of stochastic data where a time-symmetric rule cannot
        (``ForwardIncrementDeriv``) -- or ``'backward'``, ``(u[k] - u[k-lag]) / (lag dt)``
        (``BackwardIncrementDeriv``). ``lag`` and ``time_axis`` serve the one-sided schemes.
        ``accuracy`` is the order of the central differences: 4 (the five-point stencil, the
        default since Oct 2026 -- second-order truncation-error terms such as ``h^2 u_x u_xxx``
        otherwise fit the truth's residual and push it off the front) or 2 (``np.gradient``). An
        axis the stencil cannot serve (under five samples, uneven spacing) takes the second-order
        rule."""
        if scheme not in FD_SCHEMES:
            raise ValueError(f'scheme must be one of {sorted(FD_SCHEMES)}, got {scheme!r}.')
        _check_accuracy(accuracy)
        if scheme != 'central' and (int(lag) != lag or lag < 1):
            raise ValueError(f'lag must be a positive integer number of samples, got {lag!r}.')
        smoother_args = ()
        smoother_kwargs = {}

        deriv_calculator_args = ()
        # Only what the calculator uses: the default central rule is called exactly as before.
        deriv_calculator_kwargs = {'grid': None}
        if scheme != 'central':
            deriv_calculator_kwargs.update(lag=lag, time_axis=time_axis)
        if accuracy != 2:
            deriv_calculator_kwargs['accuracy'] = accuracy

        self.builder.set_smoother(PlaceholderSmoother, *smoother_args, **smoother_kwargs)
        self.builder.set_deriv_calculator(FD_SCHEMES[scheme], *deriv_calculator_args,
                                          **deriv_calculator_kwargs)

    def build_forward_preprocessing(self, lag=1, time_axis=0):
        """``FD`` with ``scheme='forward'`` (the former 'forward' preprocessor)."""
        self.build_FD_preprocessing(scheme='forward', lag=lag, time_axis=time_axis)

    def build_spectral_preprocessing(self, n=None, steepness=1, periodic='auto'):
        """Unsmoothed data, Fourier derivatives (see SpectralDeriv): ``n`` / ``steepness`` set the
        optional Butterworth low-pass (None: no filter); ``periodic`` (one value or one per axis:
        'auto' = decided from the data, True / 'included' = periodic, False = boundary-corrected
        even extension)."""
        smoother_args = ()
        smoother_kwargs = {}

        deriv_calculator_args = ()
        deriv_calculator_kwargs = {'grid': None, 'n': n, 'steepness': steepness, 'periodic': periodic}

        self.builder.set_smoother(PlaceholderSmoother, *smoother_args, **smoother_kwargs)
        self.builder.set_deriv_calculator(SpectralDeriv, *deriv_calculator_args,
                                          **deriv_calculator_kwargs)

    def build_poly_diff_preprocessing(self, use_smoothing=False, sigma=1, mp_poolsize=4,
                                      polynomial_window='auto', poly_order=None, include_time=False):
        """Local Chebyshev fits (see PolynomialDeriv), optionally after Gaussian smoothing.
        ``polynomial_window`` (odd, or 'auto': 9, wider for a higher degree, narrower on a short
        axis) and ``poly_order`` (None: derivative order + 1) are reconciled with the derivative
        orders when the pool is built (``PolynomialDeriv.resolve``), which refuses settings that
        conflict with them."""
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
