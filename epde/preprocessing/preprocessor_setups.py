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
    """
    Sets up preprocessing configurations for different derivative calculation methods.
    
        This class provides methods to configure preprocessing steps for various
        derivative calculation techniques, including ANN-based, finite difference,
        spectral, and polynomial differentiation methods. It uses a builder pattern
        to construct the preprocessing pipeline.
    
        Class Methods:
        - __init__
        - builder
        - build_ANN_preprocessing
        - build_FD_preprocessing
        - build_spectral_preprocessing
        - build_poly_diff_preprocessing
    """

    def __init__(self):
        """
        Initializes the PreprocessorSetup object.
        
                This method prepares the object for configuring and managing data preprocessing steps. It sets up the internal state required to build a preprocessing pipeline.
        
                Args:
                    self: The PreprocessorSetup instance.
        
                Returns:
                    None.
        
                Class Fields:
                    _builder: An internal builder object, initially set to None. This builder will be used to construct the preprocessing pipeline.
        """
        self._builder = None

    @property
    def builder(self):
        """
        Returns the equation builder associated with this setup.
        
                This builder is responsible for constructing and managing the symbolic
                representation of the differential equations. It encapsulates the
                grammar and available operations used in the equation discovery process.
                Accessing the builder allows for customization and extension of the
                equation search space.
        
                Returns:
                    EquationBuilder: The builder object.
        
                Why:
                    The equation builder is a core component for defining the search space
                    of possible differential equations. Accessing it allows users to
                    inspect and modify the building blocks used in the equation discovery
                    process, such as available operators, functions, and variables.
        """
        return self._builder

    @builder.setter
    def builder(self, builder: ConcretePrepBuilder):
        """
        Sets the concrete builder responsible for constructing the preprocessing pipeline.
        
        This method configures the setup with a specific builder, which defines the steps
        involved in preparing the data for equation discovery. By setting the builder,
        the setup ensures that the data is properly transformed and formatted before
        being used to identify underlying differential equations.
        
        Args:
            builder: The concrete builder to use for constructing the preprocessing pipeline.
        
        Returns:
            None.
        
        Class Fields:
            _builder: The builder object used to construct the product.
        """
        self._builder = builder

    def build_ANN_preprocessing(self, test_output=False, epochs_max=1e5,
                                loss_mean=1000, batch_frac=0.8):
        """
        Builds the preprocessing steps using an Artificial Neural Network (ANN) model.
        
                This method configures the smoother and derivative calculator components
                of the data preprocessing pipeline using ANN-based methods. It sets up
                an `ANNSmoother` for smoothing and an `AdaptiveFiniteDeriv` for
                derivative calculation. This is a crucial step in preparing the data
                for equation discovery, as it reduces noise and provides accurate
                derivative estimates, which are essential for identifying the underlying
                differential equations.
        
                Args:
                  test_output: A flag for testing the output (currently unused).
                  epochs_max: The maximum number of epochs for training the ANN smoother.
                  loss_mean: The target mean loss value for training the ANN smoother.
                  batch_frac: The fraction of data to use in each batch during ANN smoother training.
        
                Returns:
                  None
        """
        smoother_args = ()
        smoother_kwargs = {'grid': None, 'epochs_max': epochs_max,
                           'loss_mean': loss_mean, 'batch_frac': batch_frac}

        deriv_calculator_args = ()
        deriv_calculator_kwargs = {'grid': None}

        self.builder.set_smoother(ANNSmoother, *smoother_args, **smoother_kwargs)
        self.builder.set_deriv_calculator(AdaptiveFiniteDeriv, *deriv_calculator_args,
                                          **deriv_calculator_kwargs)

    def build_FD_preprocessing(self):
        """
        Builds the preprocessing steps for functional data within the EPDE framework.
        
                This method configures the smoother and derivative calculator
                used in the preprocessing pipeline. It sets the smoother to a
                PlaceholderSmoother and the derivative calculator to an
                AdaptiveFiniteDeriv. This ensures that the data is properly prepared
                for subsequent equation discovery by providing a standardized and
                flexible preprocessing approach.
        
                Args:
                    self: The instance of the class.
        
                Returns:
                    None.
        """
        smoother_args = ()
        smoother_kwargs = {}

        deriv_calculator_args = ()
        deriv_calculator_kwargs = {'grid': None}

        self.builder.set_smoother(PlaceholderSmoother, *smoother_args, **smoother_kwargs)        
        self.builder.set_deriv_calculator(AdaptiveFiniteDeriv, *deriv_calculator_args,
                                          **deriv_calculator_kwargs)

    def build_spectral_preprocessing(self, n=None, steepness=1):
        """
        Builds the spectral preprocessing pipeline for derivative estimation.
        
                This method configures the smoother and derivative calculator within the
                builder to use spectral methods. It sets a placeholder smoother and a spectral derivative calculator.
                This is a crucial step in preparing the data for the equation discovery process,
                as accurate derivative estimation is essential for identifying the underlying differential equations.
        
                Args:
                    n: The number of points to use in the spectral derivative calculation.
                    steepness: A parameter controlling the steepness of the spectral filter.
        
                Returns:
                    None
        """
        smoother_args = ()
        smoother_kwargs = {}

        deriv_calculator_args = ()
        deriv_calculator_kwargs = {'grid': None, 'n': n, 'steepness': steepness}

        self.builder.set_smoother(PlaceholderSmoother, *smoother_args, **smoother_kwargs)
        self.builder.set_deriv_calculator(SpectralDeriv, *deriv_calculator_args,
                                          **deriv_calculator_kwargs)

    def build_poly_diff_preprocessing(self, use_smoothing=False, sigma=1, mp_poolsize=4,
                                      polynomial_window=9, poly_order=None, include_time=False):
        """
        Builds a preprocessing pipeline optimized for polynomial differentiation, a crucial step in revealing underlying patterns from noisy data.
        
                This method configures the preprocessing steps, setting up a smoother (Gaussian or a placeholder for no smoothing) and a polynomial derivative calculator. These components work together to prepare the data for accurate derivative estimation, which is essential for discovering the underlying differential equations.
        
                Args:
                  use_smoothing (bool): Whether to use Gaussian smoothing before differentiation to reduce noise.
                  sigma (int): Standard deviation for Gaussian smoothing (if used).
                  mp_poolsize (int): Number of processes to use for multiprocessing in derivative calculation, speeding up the computation.
                  polynomial_window (int): Size of the window for polynomial fitting, influencing the smoothness of the derivative.
                  poly_order (int, None): Order of the polynomial to fit. If None, it will be automatically determined.
                  include_time (bool): Whether to include time as a feature in the Gaussian smoothing, useful for time-series data.
        
                Returns:
                  None: This method configures the internal state of the `builder` object.
        """
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
