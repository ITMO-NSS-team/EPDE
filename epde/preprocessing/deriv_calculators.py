#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 21 12:11:05 2022

@author: maslyaev
"""

import numpy as np

import multiprocessing as mp
from typing import Union

from abc import ABC
from epde.preprocessing.cheb import process_point_cheb


class AbstractDeriv(ABC):
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, data: np.ndarray, grid: list, max_order: Union[int, list], *args, **kwargs):
        raise NotImplementedError('Calling abstract differentiation object')


class AdaptiveFiniteDeriv(AbstractDeriv):
    def __init__(self):
        pass

    def differentiate(self, data: np.ndarray, max_order: Union[int, list],
                      mixed: bool = False, axis=None, *grids) -> list:
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
    def __init__(self):
        pass

    def __call__(self, data: np.ndarray, grid: list, max_order: Union[int, list, tuple],
                 mp_poolsize: int, polynomial_window: int, poly_order: int) -> np.ndarray:
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
        pass

    @staticmethod
    def butterworth_filter(freqs, number_of_freqs, steepness):
        '''
        Фильтр Баттерворта с порядком крутизны steepness, гасящий высокочастотные 
        составляющие с частотой выше number_of_freqs-ой преобразования Фурье
        '''
        freqs_copy = np.copy(freqs)
        freqs_copy = np.abs(freqs_copy)
        freqs_copy.sort()
        butterworth_filter_multiplier = 1 / \
            (1 + (freqs / freqs_copy[number_of_freqs - 1]) ** (2 * steepness))
        return freqs * butterworth_filter_multiplier

    def spectral_derivative_1d(self, func: np.ndarray, grid: np.ndarray, n=None, steepness=1):
        '''Одномерная спектральная производная,принимает на вход количество частот и крутизну для фильтра Баттерворта, если они не указаны - фильтрация не производится'''

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

    def spectral_derivative_nd(self, func: np.ndarray, grid: list, n=None, steepness=1,
                               deriv_hist: list = []):
        '''Многомерная спектральная производная,принимает на вход количество частот по каждой размерности и крутизну для фильтра Баттерворта, если они не указаны-фильтрация не производится'''
    
        if isinstance(n, int):
            n = np.full(shape=len(grid), fill_value=n)
        if n is None:
            n = np.min(func.shape)
        all_dim_derivative = []
        func_projection = np.fft.fftn(func, axes=[0,1])
            
        for counter, i in enumerate(grid):
            spacing_vector = np.reshape(grid[counter], (1, grid[counter].size))
            frequencies = np.fft.fftfreq(spacing_vector.size, d=(spacing_vector[0][1] - spacing_vector[0][0]))
            frequencies_filtered = self.butterworth_filter(frequencies, n, steepness)
            deriv_descr = tuple(sorted(deriv_hist + [counter,])) # inverter(counter)
            derivative = np.apply_along_axis(np.multiply, counter, func_projection, frequencies_filtered)
            derivative = np.real(np.fft.ifftn(derivative*1j * 2 * np.pi))
            all_dim_derivative.append((deriv_descr, derivative))
        return all_dim_derivative

    def differentiate(self, field: np.ndarray, grid: list, max_order: Union[int, list],
                      mixed: bool = False, n=None, steepness=1, deriv_hist: list = []) -> list:
        if isinstance(max_order, int):
            max_order = [max_order,] * field.ndim
        else:
            if mixed:
                max_order = np.full_like(max_order, np.max(max_order))

        if mixed:
            def num_of_derivs_with_ord(ords):
                temp = ords + field.ndim - 1
                numerator = np.math.factorial(temp)
                denominator = np.math.factorial(
                    ords) * np.math.factorial(field.ndim - 1)
                return int(numerator / denominator)
            expeced_num_of_derivs = sum([num_of_derivs_with_ord(
                cur_ord + 1) for cur_ord in range(max_order[0])])
        else:
            expeced_num_of_derivs = sum(max_order)
        derivatives = {}

        if mixed:
            for axis in range(field.ndim):
                higher_ord_derivs = self.spectral_derivative_nd(field, grid, n=n, steepness=steepness,
                                                                deriv_hist=deriv_hist)
                part_derivs = []
                for history, field in higher_ord_derivs:
                    part_derivs.extend(self.differentiate(
                        field, grid, max_order - 1, deriv_hist=history))
                # self.differentiate(field, grid, max_order)
                for key, deriv in part_derivs:
                    if key in derivatives.keys():
                        assert np.all(np.isclose(
                            deriv, derivatives[key])), 'Shuffle in differentiation orders shall not affect the values.'
                    derivatives[key] = deriv
        else:
            for axis in range(field.ndim):
                axis_derivs = self.spectral_derivative_high_ord(field, grid, axis=axis, max_order=max_order[axis],
                                                                n=n, steepness=steepness)
                for key, deriv in axis_derivs:
                    derivatives[key] = deriv

        # print(f'derivatives orders are {[deriv[0] for deriv in derivatives]}')
        if len(derivatives) != expeced_num_of_derivs:
            raise Exception(
                f'Expected number of derivatives {expeced_num_of_derivs} does not match obtained {len(derivatives)}')
        return derivatives

    def __call__(self, data: np.ndarray, grid: list, max_order: Union[int, list],
                 mixed: bool = False, n=None, steepness=1) -> np.ndarray:
        def make_unsparse_sparse(*grids):  # TODO^ find more elegant solution
            unique_vals = [np.unique(grid) for grid in grids]
            return np.meshgrid(*unique_vals, sparse=True, indexing='ij')

        if len(grid) > 1 and grid[0].shape == grid[1].shape:
            grid = make_unsparse_sparse(*grid)
        if isinstance(n, int) or n is None:
            n = np.full(shape=len(grid), fill_value=n)

        derivatives = self.differentiate(data, grid, max_order, mixed, n, steepness).values()
        derivatives = np.vstack([der.reshape(-1) for der in derivatives]).T

        return derivatives

    def spectral_derivative_high_ord(self, func: np.ndarray, grid: list, axis: int = 0,
                                     max_order: int = 1, n: np.ndarray = None, steepness=1) -> list:
        derivs = []
        cur_deriv = func
        # inverter = lambda x: 1 if x == 0 else (x if x != 1 else 0)

        for deriv_idx in range(max_order):
            deriv_descr = tuple([axis,] * (deriv_idx + 1))
            cur_deriv = np.apply_along_axis(self.spectral_derivative_1d, axis, cur_deriv, grid[axis],
                                            n[axis], steepness)
            derivs.append((deriv_descr, cur_deriv))

        return derivs
