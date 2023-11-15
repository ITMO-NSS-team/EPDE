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
        # print('freqs', freqs)
        # print('freqs_copy', freqs_copy[number_of_freqs - 1])
        
        butterworth_filter_multiplier = 1 / \
            (1 + (freqs / freqs_copy[number_of_freqs - 1]) ** (2 * steepness))
        # print(butterworth_filter_multiplier)
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

class TotalVariation(AbstractDeriv):
    @staticmethod
    def initial_guess(data: np.ndarray, dimensionality: tuple):
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
        '''
        *data* has to be already Fourier-transformed
        All inputs initial_u, initial_w & initial_lap have to be transformed by DFT.
        '''
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
