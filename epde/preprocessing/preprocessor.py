#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 16:05:11 2022

@author: maslyaev
"""

import numpy as np
from typing import Union

from functools import reduce
from abc import ABC, abstractmethod, abstractproperty


class GeneralizedPrepBuilder(ABC):
    """
    A base class for building data preparation pipelines.
    
        This class provides a structure for creating data preparation pipelines,
        including methods for setting smoothers, derivative calculators, and
        checking the correctness of preprocessing steps. It is intended to be
        subclassed by more specific pipeline builders.
    
        Class Methods:
        - prep_pipeline
        - set_smoother
        - set_deriv_calculator
        - check_preprocessing_correctness
    """

    def __init__(self):
        """
        Initializes a new instance of the GeneralizedPrepBuilder class.
        
        This class facilitates the preparation of data for the equation discovery process.
        It sets up the necessary data structures and configurations required for subsequent steps
        like data preprocessing and feature engineering, ensuring a smooth transition into the
        equation search phase.
        
        Args:
            self: The object instance.
        
        Returns:
            None.
        """
        pass

    @abstractproperty
    def prep_pipeline(self):
        """
        Prepare the data processing pipeline.
        
                This abstract property must be implemented by subclasses to define the specific steps for preparing the data pipeline.
                Subclasses should configure data loading, preprocessing, and any other transformations required before the equation discovery process.
        
                Args:
                    self: The object instance.
        
                Returns:
                    Pipeline: Configured data processing pipeline.
        
                Why: This step is essential to ensure that the data is in a suitable format for the equation discovery algorithms, enabling effective and accurate identification of underlying relationships.
        """
        pass

    @abstractmethod
    def set_smoother(self):
        """
        Sets the smoother to be used during data preparation.
        
        This abstract method enforces that subclasses define how smoothing is applied
        to the data before equation discovery. Smoothing helps to reduce noise and
        improve the accuracy of the identified equations.
        
        Args:
            self: The object instance.
        
        Returns:
            None.
        """
        pass

    @abstractmethod
    def set_deriv_calculator(self):
        """
        Sets up the derivative calculation method.
        
        This abstract method must be implemented by subclasses to define how derivatives are computed.
        The derivative calculation is a crucial step in constructing the equation terms.
        
        Args:
            self: The object instance.
        
        Returns:
            None.
        """
        pass

    @abstractmethod
    def check_preprocessing_correctness(self):
        """
        Checks the correctness of the preprocessing steps.
        
        This method verifies that the data preprocessing steps are correctly applied,
        ensuring data integrity before equation discovery. Implementations should
        raise an exception if inconsistencies or errors are found, preventing
        the evolutionary algorithm from operating on flawed data.
        
        Args:
            self: The object instance.
        
        Returns:
            None.
        
        Why: To guarantee the reliability of the equation discovery process by
        validating the consistency and correctness of the preprocessed data.
        """
        pass


class ConcretePrepBuilder(GeneralizedPrepBuilder):
    """
    A builder class for constructing a preprocessing pipeline for concrete data.
    
        This class provides methods to set the derivative calculator, smoother,
        and output tests for the preprocessing pipeline. It also includes a method
        to check the correctness of the configured preprocessing steps.
    
        Class Methods:
        - __init__
        - reset
        - set_tests
        - set_smoother
        - set_deriv_calculator
        - check_preprocessing_correctness
        - prep_pipeline
    """

    def __init__(self):
        """
        Initializes a ConcretePrepBuilder object.
        
        This constructor prepares the object for configuring and creating data preprocessing pipelines.
        It ensures a clean state by calling the reset method, allowing for the construction of a new pipeline.
        
        Args:
            None
        
        Returns:
            None
        """
        self.reset()

    def reset(self):
        """
        Resets the preprocessing pipeline to its initial state.
        
                This method prepares the builder for constructing a new preprocessing pipeline by clearing any previously configured steps. It ensures a clean slate for defining a new sequence of preprocessing operations. This is essential when exploring different preprocessing strategies during the equation discovery process.
        
                Args:
                    self: The object instance.
        
                Returns:
                    None. The method modifies the internal state of the `ConcretePrepBuilder` object.
        """
        self._prep_pipeline = PreprocessingPipe()
        self.deriv_calc_set = False
        self.smoother_set = False
        self.output_tests = []

    def set_tests(self, tests):
        """
        Sets the tests to be used for evaluating the generated equation.
        
        This ensures that the discovered equation is rigorously validated against a predefined set of test cases.
        
        Args:
            tests (list): A list of test functions or data points to evaluate the equation's performance.
        
        Returns:
            None
        
        Class Fields:
            output_tests (list): The list of tests to be outputted.
        """
        self.output_tests = tests

    def set_smoother(self, smoother, *args, **kwargs):
        """
        Sets the smoothing method for the data preprocessing pipeline.
        
                This method configures the data preprocessing stage by specifying the smoothing technique to be applied.
                It initializes the smoother with the given arguments and stores it within the pipeline.
                After setting the smoother, the method verifies the consistency of the preprocessing steps if both the derivative calculation and smoothing methods have been defined.
                This ensures that the data is properly prepared for subsequent equation discovery.
        
                Args:
                    smoother: The smoother class to be used.
                    *args: Positional arguments to be passed to the smoother's constructor.
                    **kwargs: Keyword arguments to be passed to the smoother's constructor.
        
                Returns:
                    None.
        
                Class Fields Initialized:
                    _prep_pipeline.smoother: An instance of the smoother class.
                    _prep_pipeline.smoother_args: The positional arguments passed to the smoother.
                    _prep_pipeline.smoother_kwargs: The keyword arguments passed to the smoother.
        """
        self._prep_pipeline.smoother = smoother()
        self._prep_pipeline.smoother_args = args
        self._prep_pipeline.smoother_kwargs = kwargs
        if self.deriv_calc_set and self.smoother_set:
            self.check_preprocessing_correctness()

    def set_deriv_calculator(self, deriv_calculator, *args, **kwargs):
        """
        Sets the derivative calculator for the preprocessing pipeline.
        
        This method configures the derivative calculation step within the preprocessing pipeline.
        It initializes the specified derivative calculator with provided arguments,
        preparing it to estimate derivatives from the input data. This is a crucial step
        in transforming raw data into a format suitable for equation discovery, as
        accurate derivative estimation is essential for identifying the relationships
        between variables and their rates of change. After setting the derivative calculator,
        the method checks if both the derivative calculator and smoother have been set,
        and if so, triggers a check for preprocessing correctness to ensure that the
        data is being prepared appropriately for the subsequent equation discovery process.
        
        Args:
            deriv_calculator: The derivative calculator class to be instantiated.
            *args: Positional arguments to be passed to the derivative calculator.
            **kwargs: Keyword arguments to be passed to the derivative calculator.
        
        Returns:
            None.
        
        Class Fields:
            _prep_pipeline.deriv_calculator: An instance of the derivative calculator class.
            _prep_pipeline.deriv_calculator_args: Positional arguments to be passed to the derivative calculator.
            _prep_pipeline.deriv_calculator_kwargs: Keyword arguments to be passed to the derivative calculator.
        """
        self._prep_pipeline.deriv_calculator = deriv_calculator()
        self._prep_pipeline.deriv_calculator_args = args
        self._prep_pipeline.deriv_calculator_kwargs = kwargs
        if self.deriv_calc_set and self.smoother_set:
            self.check_preprocessing_correctness()

    def check_preprocessing_correctness(self):
        """
        Checks the correctness of the preprocessing tool.
        
        It validates that the chosen preprocessing steps are compatible with the data and the subsequent equation discovery process.
        It attempts to run the preprocessing pipeline with a test input and applies a series of output tests to verify the result.
        If any error occurs during the process, it indicates an incorrect selection of preprocessing tools,
        preventing issues during the equation search phase.
        
        Args:
            self: The object instance.
        
        Returns:
            None.
        """
        print("Checking correctness of the preprocessing tool:")
        try:
            test_call = self._prep_pipeline.run()
            if len(self.output_tests) > 0:
                test_call = reduce(lambda x, y: y(x), self.output_tests, test_call)
            _ = test_call(np.ones((100, 100)))
        except:
            print("Incorrect selection of preprocessing tools.")

    @property
    def prep_pipeline(self):
        """
        Prepare the pipeline for equation discovery.
        
                This method retrieves the prepared pipeline and resets the internal state to ensure a clean slate for the next equation search. This is done to avoid any potential contamination from previous runs and to guarantee the reproducibility of results.
        
                Args:
                    self: The object instance.
        
                Returns:
                    The prepared pipeline.
        """
        pipeline = self._prep_pipeline
        self.reset()
        return pipeline


class PreprocessingPipe(object):
    """
    Class for preparing data for equation discovery, including scaling, smoothing, and derivative calculation.
    
    
        Attributes:
            smoother (`callable`): method for smoothing input data before calculate derivatives  
            smoother_args (`list`): args for `self.smoother`
            smoother_kwargs (`dict`): kwargs fot `self.smoother`
    
            deriv_calculator (`callable`): method for calculating derivatives from data
            deriv_calculator_args (`list`): args for `self.deriv_calculator`
            deriv_calculator_kwargs (`dict`): kwargs for `self.deriv_calculator`
    """

    def __init__(self):
        """
        Initializes the PreprocessingPipe with tools for data preparation.
        
                The PreprocessingPipe utilizes a smoother and derivative calculator to prepare data
                for differential equation discovery. This initialization sets up these tools
                with their respective arguments and keyword arguments, readying the pipeline
                for data processing.
        
                Args:
                    None
        
                Returns:
                    None
        
                Attributes:
                    smoother: The smoother object, initially set to None. Used for smoothing data.
                    deriv_calculator: The derivative calculator object, initially set to None. Used for calculating derivatives.
                    smoother_args: Positional arguments for the smoother, initially set to None.
                    smoother_kwargs: Keyword arguments for the smoother, initialized as an empty dictionary.
                    deriv_calculator_kwargs: Keyword arguments for the derivative calculator, initialized as an empty dictionary.
                    deriv_calculator_args: Positional arguments for the derivative calculator, initially set to None.
        
                Why:
                    Initializing these attributes allows the PreprocessingPipe to be configured
                    with specific smoothing and derivative calculation methods. This is crucial
                    for preparing the data in a way that facilitates accurate discovery of
                    underlying differential equations. The flexibility in configuring these
                    tools enables the pipeline to adapt to different data characteristics
                    and equation discovery tasks.
        """
        self.smoother = None
        self.deriv_calculator = None

        self.smoother_args = None
        self.smoother_kwargs = dict()

        self.deriv_calculator_kwargs = dict()
        self.deriv_calculator_args = None

    def use_grid(self, grid):
        """
        Set the grid used for smoothing and derivative calculations.
        
        This method ensures that both the smoother and derivative calculator utilize the same grid,
        which is crucial for maintaining consistency and accuracy in subsequent equation discovery steps.
        By aligning the grid, we ensure that the derivatives and smoothed data are computed on the same
        domain, preventing potential artifacts or inaccuracies in the identified equations.
        
        Args:
            grid (`np.ndarray`): The grid on which smoothing and derivative calculations will be performed.
        
        Returns:
            None
        """
        if 'grid' in self.smoother_kwargs.keys():
            self.smoother_kwargs['grid'] = grid
        if 'grid' in self.deriv_calculator_kwargs.keys():
            self.deriv_calculator_kwargs['grid'] = grid

    def run(self, data, grid=None, max_order: Union[list, int] = 1):
        """
        Calculates derivatives of the input data, optionally smoothing it beforehand. This is a crucial step in identifying the underlying differential equations, as derivatives are fundamental components of such equations.
        
                Args:
                    data (`np.ndarray`): The input data for derivative calculation.
                    grid (`np.ndarray`, optional): The grid on which the data is defined. If provided, it's used for derivative calculations. Defaults to None.
                    max_order (`list`|`int`, optional): The maximum order of derivatives to compute. Defaults to 1.
        
                Returns:
                    `np.ndarray`: The smoothed data if a smoother is applied; otherwise, the original data.
                    `np.ndarray`: The calculated derivatives of the input data.
        """
        self.deriv_calculator_kwargs['max_order'] = max_order
        if grid is not None:
            self.use_grid(grid)

        # TODO: add an arbitrary preprocssing operators
        if self.smoother is not None:
            data = self.smoother(data, *self.smoother_args,
                                 **self.smoother_kwargs)
        return data, self.deriv_calculator(data, *self.deriv_calculator_args, **self.deriv_calculator_kwargs)
