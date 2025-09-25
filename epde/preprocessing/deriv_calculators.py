#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 12:11:05 2022

@author: maslyaev
"""

import numpy as np
import time

import matplotlib.pyplot as plt

import multiprocessing as mp
from typing import Union

from abc import ABC
from epde.preprocessing.cheb import process_point_cheb

def Heatmap(Matrix, interval = None, area = ((0, 1), (0, 1)), xlabel = '', ylabel = '', figsize=(8,6), filename = None, title = ''):
    """
    Generates and displays a heatmap from a given matrix.
    
        This function visualizes the relationships and patterns within a matrix,
        allowing for the identification of key areas and trends. The heatmap
        representation aids in understanding the structure and distribution of
        data, which is crucial for tasks such as model validation and feature
        selection.
    
        Args:
            Matrix (numpy.ndarray): The input matrix to be visualized as a heatmap.
            interval (tuple, optional): The colorbar interval (vmin, vmax). If None, it's automatically determined based on the matrix values. Defaults to None.
            area (tuple, optional): The area covered by the heatmap ((ymin, ymax), (xmin, xmax)). Defaults to ((0, 1), (0, 1)).
            xlabel (str, optional): Label for the x-axis. Defaults to ''.
            ylabel (str, optional): Label for the y-axis. Defaults to ''.
            figsize (tuple, optional): Figure size (width, height) in inches. Defaults to (8,6).
            filename (str, optional): If provided, the plot is saved to a file with this name ('.eps' extension added). Defaults to None.
            title (str, optional): The title of the heatmap plot. Defaults to ''.
    
        Returns:
            None. Displays the heatmap plot and optionally saves it to a file. The heatmap visually represents the input matrix.
    """
    y, x = np.meshgrid(np.linspace(area[0][0], area[0][1], Matrix.shape[0]), np.linspace(area[1][0], area[1][1], Matrix.shape[1]))
    fig, ax = plt.subplots(figsize = figsize)
    plt.xlabel(xlabel)
    ax.set(ylabel=ylabel)
    ax.xaxis.labelpad = -10
    if interval:
        c = ax.pcolormesh(x, y, np.transpose(Matrix), cmap='RdBu', vmin=interval[0], vmax=interval[1])    
    else:
        c = ax.pcolormesh(x, y, np.transpose(Matrix), cmap='RdBu', vmin=min(-abs(np.max(Matrix)), -abs(np.min(Matrix))),
                          vmax=max(abs(np.max(Matrix)), abs(np.min(Matrix)))) 
    # set the limits of the plot to the limits of the data
    ax.axis([x.min(), x.max(), y.min(), y.max()])
    fig.colorbar(c, ax=ax)
    plt.title(title)
    plt.show()
    if type(filename) != type(None): plt.savefig(filename + '.eps', format='eps')

class AbstractDeriv(ABC):
    """
    Abstract base class for differentiation objects.
    
        This class serves as a template for implementing different differentiation methods.
        It defines the basic interface that all differentiation objects should implement.
    
        Class Methods:
        - __call__:
    """

    def __init__(self, *args, **kwargs):
        """
        Initializes the AbstractDeriv object.
        
        This base class provides a foundation for representing differential equation terms.
        Subclasses will extend this to define specific term structures and their derivatives.
        This initialization currently performs no operations, but serves as a placeholder
        for future common initialization logic across different derivative terms.
        
        Args:
            *args: Variable length argument list.  These arguments are passed to the superclass constructor.
            **kwargs: Arbitrary keyword arguments. These keyword arguments are passed to the superclass constructor.
        
        Returns:
            None.
        """
        pass

    def __call__(self, data: np.ndarray, grid: list, max_order: Union[int, list], *args, **kwargs):
        """
        Applies the abstract differentiation operator.
        
        This method serves as the entry point for applying a specific differentiation scheme
        to the input data. It ensures that all concrete differentiation methods implement
        a consistent interface, facilitating their use within the equation discovery process.
        Since this is an abstract method, it raises a NotImplementedError to enforce
        implementation in derived classes.
        
        Args:
            data (np.ndarray): The input data to be differentiated.
            grid (list): The grid points at which the data is defined.
            max_order (Union[int, list]): The maximum order of derivatives to compute.
            *args: Additional positional arguments passed to the differentiation method.
            **kwargs: Additional keyword arguments passed to the differentiation method.
        
        Returns:
            None
        
        Raises:
            NotImplementedError: Always raised as this is an abstract method.
        """
        raise NotImplementedError('Calling abstract differentiation object')


class AdaptiveFiniteDeriv(AbstractDeriv):
    """
    A class for calculating finite derivatives of data with adaptive grid refinement.
    
         Attributes:
            stencil_width: The width of the stencil used for derivative calculation.
            accuracy_order: The accuracy order of the finite difference scheme.
            step_ratio_limit: The maximum allowed ratio between consecutive grid spacings.
    """

    def __init__(self):
        """
        Initializes the AdaptiveFiniteDeriv class.
        
        This class is designed to dynamically adjust finite difference approximations
        for derivative estimation. Initialization prepares the object for subsequent
        derivative calculations without requiring any initial parameters.
        
        Args:
            self: The object instance.
        
        Returns:
            None.
        
        Why:
            The class needs to be initialized before it can be used to estimate derivatives.
            This initialization sets up the internal state of the object, preparing it for
            subsequent calculations.
        """
        pass

    def differentiate(self, data: np.ndarray, max_order: Union[int, list],
                      mixed: bool = False, axis=None, *grids) -> list:
        """
        Calculates derivatives of the input data up to a specified order.
        
                This method computes derivatives of a given data array with respect to specified grids.
                It can compute derivatives along a specific axis or mixed derivatives (under certain conditions).
                The computed derivatives are essential for constructing candidate differential equations.
        
                Args:
                    data (np.ndarray): The input data as a NumPy array.
                    max_order (Union[int, list]): The maximum order of derivative to compute. It can be an integer or a list of integers,
                        where each element corresponds to the maximum order along each axis.
                    mixed (bool, optional): A boolean flag indicating whether to compute mixed derivatives. Defaults to False.
                    axis (optional): The axis along which to calculate the derivative. Defaults to None.
                    *grids: Variable number of grid spacings for each dimension.
        
                Returns:
                    list: A list of NumPy arrays containing the computed derivatives.
                        The order of derivatives in the list corresponds to the order in which they are calculated.
        """
        if isinstance(max_order, int):
            max_order = [max_order,] * data.ndim
        if any([ord_ax != max_order[0] for ord_ax in max_order]) and mixed:
            raise Exception(
                'Mixed derivatives can be taken only if the orders are same along all axes.')
        if data.ndim != len(grids) and (len(grids) != 1 or data.ndim != 2):
            print(data.ndim, len(grids))
            raise ValueError('Data dimensionality does not fit passed grids.')

        if len(grids) == 1 and data.ndim == 2:
            assert np.any(data.shape) == 1
            derivs = []
            dim_idx = 0 if data.shape[1] == 1 else 1
            grad = np.gradient(data, grids[0], axis=dim_idx)
            derivs.append(grad)
            ord_marker = max_order[dim_idx] if axis is None else max_order[axis]
            if ord_marker > 1:
                higher_ord_der_axis = None if mixed else dim_idx
                ord_reduced = [ord_ax - 1 for ord_ax in max_order]
                derivs.extend(self.differentiate(grad, ord_reduced, mixed, higher_ord_der_axis, *grids))
        else:
            derivs = []
            if axis == None:
                for dim_idx in np.arange(data.ndim):
                    if max_order[dim_idx] == 0:
                        continue
                    grad = np.gradient(data, grids[dim_idx], axis=dim_idx)
                    derivs.append(grad)
                    ord_marker = max_order[dim_idx] if axis is None else max_order[axis]
                    if ord_marker > 1:
                        higher_ord_der_axis = None if mixed else dim_idx
                        ord_reduced = [ord_ax - 1 for ord_ax in max_order]
                        derivs.extend(self.differentiate(
                            grad, ord_reduced, mixed, higher_ord_der_axis, *grids))
            else:
                grad = np.gradient(data, grids[axis], axis=axis)
                derivs.append(grad)
                if max_order[axis] > 1:
                    ord_reduced = [ord_ax - 1 for ord_ax in max_order]
                    derivs.extend(self.differentiate(
                        grad, ord_reduced, False, axis, *grids))
        return derivs

    def __call__(self, data: np.ndarray, grid: list, max_order: Union[int, list, tuple],
                 mixed: bool = False) -> np.ndarray:
        """
        Computes derivatives of the input data using adaptive finite difference schemes.
        
                This method is essential for estimating the rates of change within the provided data,
                which is a crucial step in identifying the underlying differential equations.
                The derivatives are calculated based on the provided grid points and user-specified derivative orders.
        
                Args:
                    data (np.ndarray): Input data to differentiate.
                    grid (list): Grid points along each axis.
                    max_order (Union[int, list, tuple]): Maximum order of derivative to compute.
                    mixed (bool): Whether to compute mixed derivatives.
        
                Returns:
                    np.ndarray: Array of derivatives.
        """
        grid_unique = [np.unique(ax_grid) for ax_grid in grid]

        derivs = self.differentiate(data, max_order, mixed, None, *grid_unique)
        derivs = np.vstack([der.reshape(-1) for der in derivs]).T
        return derivs


# class ANNBasedFiniteDifferences(AbstractDeriv):
#     def __init__(self):
#         self._internal_ann = None

#     def differentiate(self, data: np.ndarray, max_order: Union[int, list],
#                       mixed: bool = False, axis=None, *grids) -> list:

class PolynomialDeriv(AbstractDeriv):
    """
    Calculates derivatives of data using Chebyshev polynomials.
    """

    def __init__(self):
        """
        Initializes a new instance of the PolynomialDeriv class.
        
        This class is designed to represent and manipulate polynomial derivatives within the EPDE framework.
        The initialization prepares the object for subsequent operations related to symbolic differentiation
        and equation discovery.
        
        Args:
            self: The object instance.
        
        Returns:
            None.
        """
        pass

    def __call__(self, data: np.ndarray, grid: list, max_order: Union[int, list, tuple],
                 mp_poolsize: int, polynomial_window: int, poly_order: int) -> np.ndarray:
        """
        Computes derivatives of the input data using local polynomial approximations.
        
                This method calculates derivatives at each point by fitting Chebyshev polynomials
                within a specified window. This approach allows for accurate estimation of derivatives,
                even in the presence of noise or irregularities in the data. Multiprocessing is used
                to accelerate the computation when a pool size greater than 1 is specified.
                This is a crucial step in identifying the underlying differential equations
                that govern the data.
        
                Args:
                    data (np.ndarray): Input data as a NumPy array.
                    grid (list): Grid coordinates corresponding to the data points.
                    max_order (Union[int, list, tuple]): Maximum order of the derivative to compute. Can be an integer, list, or tuple.
                    mp_poolsize (int): Size of the multiprocessing pool. If greater than 1, multiprocessing is used.
                    polynomial_window (int): Size of the window used for polynomial fitting.
                    poly_order (int): Order of the polynomial to fit.
        
                Returns:
                    np.ndarray: NumPy array containing the computed derivatives for each point in the input data.
        """
        polynomial_boundary = polynomial_window//2 + 1
        index_array = []

        for idx, _ in np.ndenumerate(data):
            index_array.append((idx, data, grid, polynomial_window, max_order, 
                                polynomial_boundary, poly_order))
        print(len(index_array))

        if mp_poolsize > 1:
            pool = mp.Pool(mp_poolsize)
            derivatives = pool.map_async(process_point_cheb, index_array)
            pool.close()
            pool.join()
            derivatives = derivatives.get()
        else:
            derivatives = list(map(process_point_cheb, index_array))

        return np.array(derivatives)


class SpectralDeriv(AbstractDeriv):
    '''
    Класс спектральной производной by https://github.com/KimKitsurag1
        adapted to match AbstractDeriv
    '''


    def __init__(self):
        """
        Initializes a `SpectralDeriv` object.
        
                This class is designed to work with spectral representations of data, 
                preparing it for subsequent differential equation discovery. 
                The initialization sets up the object to handle spectral data, 
                allowing for efficient computation of derivatives in the spectral domain, 
                which is a crucial step in identifying underlying differential equations.
        
                Args:
                    self: The object instance.
        
                Returns:
                    None.
        """
        pass

    @staticmethod
    def butterworth_filter(freqs, number_of_freqs, steepness):
        """
        Applies a Butterworth filter to dampen high-frequency components in a frequency spectrum. This helps to smooth the spectrum and reduce noise, which is crucial for identifying the dominant frequencies that contribute most to the underlying dynamics of the system.
        
                Args:
                    freqs (np.ndarray): Array of frequencies.
                    number_of_freqs (int):  Cutoff frequency index; frequencies above this index are attenuated.
                    steepness (int): Order of the Butterworth filter, controlling the sharpness of the cutoff.
        
                Returns:
                    np.ndarray: Filtered frequency array.
        """
        freqs_copy = np.copy(freqs)
        freqs_copy = np.abs(freqs_copy)
        freqs_copy.sort()
        # print('freqs', freqs)
        # print('freqs_copy', freqs_copy[number_of_freqs - 1])
        
        butterworth_filter_multiplier = 1 / \
            (1 + (freqs / freqs_copy[number_of_freqs - 1]) ** (2 * steepness))
        # print(butterworth_filter_multiplier)
        return freqs * butterworth_filter_multiplier

    def spectral_derivative_1d(self, func: np.ndarray, grid: np.ndarray, n=None, steepness=1):
        """
        Applies a spectral derivative operator to a 1D function.
        
                This method transforms the input function to the frequency domain, applies a derivative operator, and then transforms the result back to the spatial domain.
                Optionally, a Butterworth filter can be applied in the frequency domain to reduce noise.
                This is useful for calculating derivatives of noisy data, where direct differentiation may amplify noise.
        
                Args:
                    func (np.ndarray): The input 1D function values.
                    grid (np.ndarray): The grid coordinates corresponding to the function values.
                    n (int, optional): The number of frequencies to use for the Butterworth filter. If None, no filtering is applied. Defaults to None.
                    steepness (float, optional): The steepness of the Butterworth filter. Defaults to 1.
        
                Returns:
                    np.ndarray: The spectral derivative of the input function.
        """

        # print(func.shape, grid.shape)
        # if isinstance(func, type(None)) and isinstance(grid, type(None)):
        #     func = self.func
        #     grid = self.grid
        func_projection = np.fft.rfft(func)
        if n is None:
            n = func_projection.size
        func_projection_copy = np.copy(func_projection)
        spacing_vector = np.reshape(grid, (1, grid.size))
        func_projection_filtered = func_projection_copy
        frequencies = np.fft.rfftfreq(spacing_vector.size, d=(spacing_vector[0][1] - spacing_vector[0][0]))
        frequencies_filtered = self.butterworth_filter(frequencies, n, steepness)
        return np.real(np.fft.irfft(1j * 2 * np.pi * frequencies_filtered * func_projection_filtered))
    
    def spectral_derivative_nd(self, func : np.ndarray, grid : list, n = None, steepness = 1, 
                               deriv_hist : list = []):
        """
        Computes the spectral derivative of a function along specified grid dimensions. This method leverages Fourier transforms to estimate derivatives, providing a way to analyze the rate of change within the function's spectral representation.
        
                Args:
                    func (np.ndarray): The input function represented as a multi-dimensional array.
                    grid (list): A list of arrays, where each array represents the grid coordinates along a specific dimension.
                    n (int or None): The number of frequencies to consider for the Butterworth filter. If None, it defaults to the minimum dimension size of the input function.
                    steepness (int): The steepness parameter for the Butterworth filter, controlling the filter's sharpness.
                    deriv_hist (list): A list to keep track of the derivative dimensions. Defaults to an empty list.
        
                Returns:
                    list: A list of tuples, where each tuple contains:
                        - A tuple describing the derivative dimensions.
                        - The derivative of the function along the corresponding dimension.
        
                Why:
                    This method calculates spectral derivatives to analyze the function's rate of change in the frequency domain, which is useful for identifying dominant modes and patterns, and ultimately, for constructing differential equation models.
        """

        if isinstance(n, int):
            n = np.full(shape=len(grid), fill_value=n)
        if isinstance(n, type(None)):
            n = np.min(func.shape)
        all_dim_derivative = []
        func_projection = np.fft.fftn(func)
        print(func_projection.shape) #marker
        inverter = lambda x: 1 if x == 0 else (x if x != 1 else 0)
        
        for counter, i in enumerate(grid):
            spacing_vector = np.reshape(grid[counter], (1, grid[counter].size))
            frequencies = np.fft.fftfreq(spacing_vector.size, d=(spacing_vector[0][1] - spacing_vector[0][0]))
            frequencies_filtered = self.butterworth_filter(frequencies, n, steepness)
            deriv_descr = tuple(sorted(deriv_hist + [counter,])) # inverter(counter)
            derivative = np.apply_along_axis(np.multiply, counter, func_projection, frequencies_filtered)
            derivative = np.real(np.fft.ifftn(derivative*1j * 2 * np.pi))
            all_dim_derivative.append((deriv_descr, derivative))
        return all_dim_derivative
    
    def differentiate(self, field : np.ndarray, grid : list, max_order : Union[int, list], 
                      mixed : bool = False, n = None, steepness = 1, deriv_hist : list = []) -> list:
        """
        Calculates derivatives of a field using spectral differentiation.
        
                This method computes derivatives of a given field with respect to
                the spatial grid up to a specified order. It supports both single-axis and mixed derivatives.
                The derivatives are calculated to facilitate the discovery of underlying differential equations
                that govern the behavior of the input field.
        
                Args:
                    field: The input field to differentiate (numpy ndarray).
                    grid: The spatial grid coordinates (list of numpy arrays).
                    max_order: The maximum order of derivative to compute. Can be an
                        integer or a list specifying the maximum order for each axis.
                    mixed: A boolean indicating whether to compute mixed derivatives.
                        Defaults to False.
                    n: Number of modes to use in spectral derivative. Defaults to None.
                    steepness: Steepness parameter for spectral derivative. Defaults to 1.
                    deriv_hist: A list to store the history of derivatives taken.
                        Defaults to [].
        
                Returns:
                    A dictionary where keys are tuples representing the derivative
                    orders along each axis, and values are the corresponding derivative
                    fields (numpy ndarrays).
        """
        if isinstance(max_order, int):
            max_order = [max_order,] * field.ndim
        else:
            if mixed:
                max_order = np.full_like(max_order, np.max(max_order))

        if mixed:
            def num_of_derivs_with_ord(ords):
                temp = ords + field.ndim - 1
                numerator = np.math.factorial(temp)
                denominator = np.math.factorial(ords) * np.math.factorial(field.ndim - 1)
                return int(numerator / denominator)
            expeced_num_of_derivs = sum([num_of_derivs_with_ord(cur_ord + 1) for cur_ord in range(max_order[0])])
        else:
            expeced_num_of_derivs = sum(max_order)
        derivatives = {}     
        
        if mixed:
            for axis in range(field.ndim):
                higher_ord_derivs = self.spectral_derivative_nd(field, grid, n = n, steepness = steepness, 
                                                                deriv_hist = deriv_hist)
                part_derivs = []
                for history, field in higher_ord_derivs:
                    part_derivs.extend(self.differentiate(field, grid, max_order - 1, deriv_hist=history))
                # self.differentiate(field, grid, max_order)
                for key, deriv in part_derivs:
                    if key in derivatives.keys():
                        assert np.all(np.isclose(deriv, derivatives[key])), 'Shuffle in differentiation orders shall not affect the values.'
                    derivatives[key] = deriv
        else:
            for axis in range(field.ndim):
                axis_derivs = self.spectral_derivative_high_ord(field, grid, axis = axis, max_order = max_order[axis],
                                                                n = n, steepness = steepness)
                for key, deriv in axis_derivs:
                    derivatives[key] = deriv
                
        print(f'derivatives orders are {[deriv[0] for deriv in derivatives]}')
        if len(derivatives) != expeced_num_of_derivs:
            raise Exception(f'Expected number of derivatives {expeced_num_of_derivs} does not match obtained {len(derivatives)}')
        return derivatives
    
    def __call__(self, data : np.ndarray, grid : list, max_order : Union[int, list], 
                 mixed : bool = False, n = None, steepness = 1) -> np.ndarray:
        """
        Computes derivatives of the input data using spectral methods to facilitate equation discovery.
        
                This method calculates derivatives of the input data with respect to the given grid, up to a specified order.
                These derivatives are essential components for constructing candidate differential equations within the EPDE framework.
                The method supports mixed derivatives and allows for adjusting the steepness parameter to fine-tune the differentiation process.
                It is used to generate features that represent different derivative terms, which are then used in the equation discovery process.
        
                Args:
                    data (np.ndarray): Input data as a numpy array.
                    grid (list): Grid points for differentiation.
                    max_order (Union[int, list]): Maximum order of derivative to compute.
                    mixed (bool, optional): Flag to include mixed derivatives. Defaults to False.
                    n (optional): Order of the derivative for each dimension. If None, defaults to 0.
                    steepness (int, optional): Steepness parameter for the derivative calculation. Defaults to 1.
        
                Returns:
                    np.ndarray: Derivatives of the data as a numpy array.
        """
        def make_unsparse_sparse(*grids): # TODO^ find more e;egant solution
            unique_vals = [np.unique(grid) for grid in grids]
            return np.meshgrid(*unique_vals, sparse = True, indexing = 'ij')
        
        if len(grid) > 1 and grid[0].shape == grid[1].shape:
            grid = make_unsparse_sparse(*grid)
        if isinstance(n, int) or isinstance(n, type(None)):
            n = np.full(shape=len(grid), fill_value=n)        
        
        derivatives = self.differentiate(data, grid, max_order, mixed, n, steepness).values()
        derivatives = np.vstack([der.reshape(-1) for der in derivatives]).T
        
        return derivatives
    
    def spectral_derivative_high_ord(self, func : np.ndarray, grid : list, axis : int = 0, 
                                     max_order : int = 1, n = None, steepness = 1) -> list:
        """
        Computes spectral derivatives of a function along a specified axis.
        
                This method calculates derivatives of a given function using the spectral
                method, which involves transforming the function to Fourier space,
                multiplying by the corresponding frequencies, and then transforming back
                to real space. It applies a Butterworth filter to the frequencies to
                reduce noise, ensuring stable and accurate derivative estimation, which is crucial for equation discovery.
                This is done to accurately represent the function's behavior in the frequency domain, enabling precise derivative calculations required for identifying underlying differential equations.
                
                Args:
                    func: The input function as a NumPy array.
                    grid: A list of arrays representing the grid coordinates.
                    axis: The axis along which to compute the derivative (default: 0).
                    max_order: The maximum order of the derivative to compute (default: 1).
                    n: Cutoff frequency for the butterworth filter. If None, defaults to the minimum dimension of func.
                    steepness: Steepness of the butterworth filter.
                
                Returns:
                    A list of tuples, where each tuple contains:
                      - A tuple describing the derivative (axis, repeated deriv_idx times).
                      - The derivative as a NumPy array.
        """
        derivs = []
        cur_deriv = func
        func_projection = np.fft.fftn(func)
        spacing_vector = np.reshape(grid[axis], (1, grid[axis].size))
        frequencies = np.fft.fftfreq(spacing_vector.size, d=(spacing_vector[0][1] - spacing_vector[0][0]))
        print(func_projection.shape) #marker
        # inverter = lambda x: 1 if x == 0 else (x if x != 1 else 0)
        if isinstance(n, int):
            n = np.full(shape=len(grid), fill_value=n)
        if isinstance(n, type(None)):
            n = np.min(func.shape)
        frequencies_filtered = self.butterworth_filter(frequencies, n[axis], steepness)
        for deriv_idx in range(1,max_order+1):
            deriv_descr = tuple([axis,] * (deriv_idx)) # inverter(axis) V
            derivative = np.apply_along_axis(np.multiply, axis, func_projection,
                                             np.power(frequencies_filtered*1j * 2 * np.pi, deriv_idx,dtype = complex))
            cur_deriv= np.real(np.fft.ifftn(derivative))
            derivs.append((deriv_descr, cur_deriv))
            
        return derivs

class TotalVariation(AbstractDeriv):
    """
    Total variation (TV) regularization.
    
        This class implements total variation regularization, a technique used to
        smooth images or signals while preserving edges. It leverages the
        Alternating Direction Method of Multipliers (ADMM) to solve the optimization
        problem.
    
        Methods:
        - initial_guess
        - admm_step
        - optimize_with_admm
    """

    @staticmethod
    def initial_guess(data: np.ndarray, dimensionality: tuple):
        """
        Generates initial estimates for optimization variables.
        
                This method computes initial values for the gradient (`grad`), an approximation
                of the Hessian (`w`), and the Laplacian multiplier (`lap_mul`). These initial
                values are crucial for the subsequent optimization process, guiding the search
                towards a solution that minimizes the total variation while adhering to the
                constraints imposed by the data. The gradient is computed using `np.gradient`
                along each dimension of the input data. The 'w' is calculated as the gradient
                of the gradient along each dimension. The Laplacian multiplier is initialized
                as an array of zeros with the same shape as 'w'. These initial values provide
                a starting point for refining the solution and discovering the underlying
                differential equation.
        
                Args:
                    data: The input data as a multi-dimensional numpy array.
                    dimensionality: A tuple representing the dimensions of the input data.
        
                Returns:
                    A tuple containing:
                      - grad: The gradient of the input data.
                      - w: An approximation of the Hessian matrix.
                      - lap_mul: The initialized Laplacian multiplier (zeros).
        """
        grad = np.array([np.gradient(data, axis=dim_idx) for dim_idx, dim in enumerate(dimensionality)])
        # print(grad.shape)
        # w = np.array([np.zeros(dimensionality) for idx in np.arange(len(dimensionality)**2)], 
        #              dtype = np.ndarray).reshape([len(dimensionality), len(dimensionality),] + list(dimensionality))
        w = np.array([np.gradient(grad[int(idx/2.)], axis = idx % 2) 
                      for idx in np.arange(len(dimensionality)**2)]).reshape([len(dimensionality), 
                                                                              len(dimensionality),] + list(dimensionality))
        
        lap_mul = np.array([np.zeros(dimensionality) 
                            for idx in np.arange(len(dimensionality)**2)]).reshape([len(dimensionality), 
                                                                                    len(dimensionality),] + list(dimensionality))
        return grad, w, lap_mul

    @staticmethod
    def admm_step(data: np.ndarray, steps: list, initial_u: np.ndarray, initial_w: np.ndarray, 
                  initial_lap: np.ndarray, lbd: float, reg_strng: float, c_const: float) -> tuple:
        """
        Performs a single ADMM (Alternating Direction Method of Multipliers) step to refine the solution.
                This step optimizes the variables `u`, `w`, and `lap` in an iterative manner to minimize the total variation of the input `data`,
                subject to constraints that enforce consistency between the variables.
        
                Args:
                    data (np.ndarray): The input data, which should already be transformed into the Fourier domain.
                    steps (list): List of stepsizes for each dimension
                    initial_u (np.ndarray): Initial estimate of the variable `u` in Fourier space.
                    initial_w (np.ndarray): Initial estimate of the variable `w` in Fourier space.
                    initial_lap (np.ndarray): Initial estimate of the Lagrange multiplier `lap` in Fourier space.
                    lbd (float): Regularization parameter controlling the strength of the total variation penalty.
                    reg_strng (float): Regularization strength parameter.
                    c_const (float): Parameter for updating the Lagrange multiplier.
        
                Returns:
                    tuple: A tuple containing the updated values of `u`, `w`, and `lap` after the ADMM step.
                           These updated values represent a refined solution that better fits the data while minimizing its total variation.
                Why:
                    This method refines the estimates of the variables u, w, and lap, driving the solution towards a state that minimizes the total variation of the input data while adhering to the constraints imposed by the problem formulation.
        """
        def soft_thresholding(arg: np.ndarray, lbd: float) -> np.ndarray:
            norm = np.linalg.norm(arg)
            return max(norm - lbd, 0) * arg / norm
        
        def diff_factor(N: int, d: float = 1.) -> np.ndarray:
            #freqs = np.fft.fftfreq(N, d = d)
            freqs = np.arange(N)
            return np.exp(-2*np.pi*freqs/N) - 1
        
        initial_u = np.fft.fftn(initial_u, s = initial_u.shape[1:])
        # print(initial_u[0].shape)
        # print(np.max(np.abs(initial_u[0])), np.min(np.abs(initial_u[0])))
        Heatmap(np.abs(initial_u[0]), title = 'FFT')
        initial_w_fft = np.fft.fftn(initial_w, s = initial_w.shape[2:])
        initial_lap_fft = np.fft.fftn(initial_lap, s = initial_lap.shape[2:])
        
        diff_factors = np.array([np.moveaxis(np.broadcast_to(diff_factor(dim_size, d = steps[comp_idx]),
                                                             shape = initial_u[0].shape),
                                             source = -1, destination = comp_idx)
                                 for comp_idx, dim_size in enumerate(initial_u[0].shape)])
        
        lbd_inv = lbd**(-1)
        
        u_freq = np.copy(initial_u)
        
        for grad_idx in range(initial_u.shape[0]):
            u_nonzero_freq = np.zeros_like(u_freq[grad_idx])
            
            section_len = u_freq[grad_idx].shape[grad_idx]
            putting_shape = np.ones(u_freq[grad_idx].ndim, dtype = np.int8)
            putting_shape[grad_idx] = section_len-1
            
            putting_args = {'indices' : np.arange(1, u_freq[grad_idx].shape[grad_idx]).reshape(putting_shape),
                            'axis' : grad_idx}
            # print(f'putting_args {putting_args}')
            taking_args = {'indices' : range(1, u_freq[grad_idx].shape[grad_idx]),
                           'axis' : grad_idx}
            # print(f'taking_args {taking_args}')
            
            def take(arr: np.ndarray, taking_args: dict = taking_args):
                return np.take(arr, **taking_args)

            denum_part = lbd_inv * np.sum([np.abs(take(factor))**2 for factor in diff_factors])            
            partial_sum = [take(diff_factors[arg_idx]) * (take(initial_w_fft[grad_idx, arg_idx]) - 
                                                          take(initial_lap_fft[grad_idx, arg_idx])) 
                           for arg_idx in range(u_freq.shape[0])]
            print([(np.min(elem), np.max(elem)) for elem in partial_sum]) # * take(data) * take(data)
            print(np.min(take(data)), np.max(take(data)))
            print('3rd term:', np.min(reg_strng * take(diff_factors[grad_idx]) ), np.max(reg_strng * take(diff_factors[grad_idx]) ))
            
            grad_nonzero_upd = (lbd_inv * np.sum(partial_sum, axis = 0) + reg_strng * take(diff_factors[grad_idx]) * take(data) /
                                (denum_part + reg_strng/np.abs(take(diff_factors[grad_idx]))))
            print('shit shape', (lbd_inv * np.sum(partial_sum, axis = 0) + reg_strng * take(diff_factors[grad_idx]) * take(data)).shape)
            print(np.max(lbd_inv * np.sum(partial_sum, axis = 0) + reg_strng * take(diff_factors[grad_idx]) * take(data)),
                  np.min(lbd_inv * np.sum(partial_sum, axis = 0) + reg_strng * take(diff_factors[grad_idx]) * take(data)))
            Heatmap(np.real(lbd_inv * np.sum(partial_sum, axis = 0) + reg_strng * take(diff_factors[grad_idx]) * take(data)), title = 'partial')
            
            np.put_along_axis(u_nonzero_freq, values = grad_nonzero_upd, **putting_args)
            u_freq[grad_idx] = u_nonzero_freq
        Heatmap(np.real(u_freq[0]), title = 'inside the optimization')            
    
        initial_u = np.real(np.fft.ifftn(u_freq, u_freq.shape[1:]))
        for i in range(initial_w.shape[0]):
            for j in range(initial_w.shape[1]):
                initial_w[i, j] = soft_thresholding(np.gradient(initial_u[j], axis = i) + initial_lap[i, j], lbd)
                
        for i in range(initial_w.shape[0]):
            for j in range(initial_w.shape[1]):
                initial_lap[i, j] = c_const*(initial_lap[i, j] + np.gradient(initial_u[j], axis = i)
                                             - initial_w[i, j])
    
        time.sleep(15)
        return initial_u, initial_w, initial_lap

    def optimize_with_admm(self, data, lbd: float, reg_strng: float, c_const: float, nsteps: int = 1e5):
        """
        Optimizes the given data using the Alternating Direction Method of Multipliers (ADMM).
        
                This method refines an estimate of the underlying signal by iteratively updating primal (u), dual (w) variables, and the Lagrangian multiplier (lap_mul) based on the ADMM algorithm. This optimization process helps in identifying the governing differential equations from the input data.
        
                Args:
                  data: The input data to be optimized.
                  lbd: Lambda, a regularization parameter.
                  reg_strng: Regularization strength parameter.
                  c_const: Constant parameter used in the ADMM update.
                  nsteps: The number of ADMM iterations to perform. Defaults to 1e5.
        
                Returns:
                  The optimized primal variable (u) after the ADMM iterations. This represents a refined solution that aids in discovering the underlying differential equations.
        """
        u, w, lap_mul = self.initial_guess(data=data, dimensionality=data.shape)
        data_fft = np.fft.fftn(data)
        print(f'For some reason has to be abysmal: {np.min(np.real(data_fft)), np.max(np.real(data_fft))}')
        for epoch in range(int(nsteps)):
            print(epoch)
            if epoch % 100 == 0:
                Heatmap(u[1], title=str(epoch))
                plt.plot(u[1, :, int(u.shape[2]/2.)])
                plt.show()
            u, w, lap_mul = self.admm_step(data = data_fft, steps = np.ones(data.ndim), initial_u = u, initial_w = w, 
                                           initial_lap=lap_mul, lbd = lbd, reg_strng = reg_strng, 
                                           c_const = c_const)
        return u
        
    # def differentiate(self, data: np.ndarray, max_order: Union[int, list],
    #                   mixed: bool = False, axis=None, *grids) -> list:
    #     if isinstance(max_order, int):
    #         max_order = [max_order,] * data.ndim
    #     else:
    #         max_order = np.full_like(max_order, np.max(max_order))
        
    #     if len(grids) == 1 and data.ndim == 2:
