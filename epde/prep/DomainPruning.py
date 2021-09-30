#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  6 15:14:41 2021

@author: mike_ubuntu
"""

import numpy as np
from typing import Union, Callable
import copy

import epde.globals as global_var

def scale_values(tensor):
    return tensor/np.max(tensor)

def split_tensor(tensor, fraction_along_axis : Union[int, list, tuple]):
    assert isinstance(tensor, np.ndarray)
    fragments = [tensor,]
    indexes = [[0 for idx in range(tensor.ndim)],]
    split_indexes_along_axis = []
    
    for axis_idx in np.arange(tensor.ndim):
        temp_fragments = []
        temp_indexes = []
        if isinstance(fraction_along_axis, int):
            fraction = fraction_along_axis
        elif isinstance(fraction_along_axis, (tuple, list)):
            fraction = fraction_along_axis[axis_idx]
        else:
            raise TypeError('Tensor fraction shall be declared in form of integer or list/tuple with elements for each axis in data')
        split_indexes = [int(tensor.shape[axis_idx]/fraction * element) for element in range(fraction) if element != 0]
        split_indexes_along_axis.append([0,] + split_indexes + [tensor.shape[axis_idx]])
        for ts_idx, tensor_section in enumerate(fragments):
            splitted_op = np.split(tensor_section, split_indexes, axis = axis_idx)
            temp_fragments.extend(splitted_op)
            for index_idx, idx in enumerate([indexes[ts_idx] for i in range(fraction)]):
                temp = copy.copy(idx); temp[axis_idx] = index_idx
                temp_indexes.append(temp)
        fragments = temp_fragments
        indexes = temp_indexes
    if isinstance(fraction_along_axis, (list, tuple)): 
        block_matrix = np.empty(fraction_along_axis, dtype = object) 
    else:
        block_matrix = np.empty([fraction_along_axis for i in range(tensor.ndim)], dtype = object)
    for idx, tensor_fragment in enumerate(fragments):
        block_matrix[tuple(indexes[idx])] = tensor_fragment
    return split_indexes_along_axis, block_matrix

def majority_rule(tensors : Union[list, np.ndarray], threshold : float = 1e-2):
    non_constancy_cond = lambda x: np.linalg.norm(x) > (threshold * x.size)
    return sum([non_constancy_cond(x_elem) for x_elem in tensors])/len(tensors) >= 0.5

def get_subdomains_mask(tensor, division_fractions : Union[int, list, tuple], domain_selector : Callable, 
                        domain_selector_kwargs : dict, time_axis : int):
    if isinstance(division_fractions, int):
        sd_shape = tuple([division_fractions for idx in range(tensor.ndim) if idx != time_axis]) 
    else:
        sd_shape = tuple([div_frac for idx, div_frac in enumerate(division_fractions) if idx != time_axis])
    accepted_spatial_domains = np.full(shape = (sd_shape), fill_value = True, dtype = bool) # [division_fraction for i in self.]
    tensor = scale_values(tensor)    
    split_idxs, time_deriv_fragments = split_tensor(tensor, division_fractions)
    time_deriv_fragments = np.moveaxis(a = time_deriv_fragments, source = time_axis, destination = -1)#    for 
    
    temp = []; counter = 0
    for idx, partial_tensor in np.ndenumerate(time_deriv_fragments):
        if counter < time_deriv_fragments.shape[-1] - 1:
            temp.append(partial_tensor)
            counter += 1
        else:
            temp.append(partial_tensor)
            accepted_spatial_domains[idx[:-1]] = domain_selector(temp, **domain_selector_kwargs)
            temp = []; counter = 0
    return split_idxs, accepted_spatial_domains

def pruned_domain_boundaries(mask : np.ndarray, split_idxs : list, time_axis : int, 
                             rectangular : bool = True, threshold : float = 0.5): # Разбораться с методом
    if rectangular:
        boundaries = []
        for dim in np.arange(mask.ndim):
            mask = np.moveaxis(mask, source = dim, destination = 0)
            accepted_slices = np.empty(shape = mask.shape[0], dtype = bool)
            for idx_outer in np.arange(mask.shape[0]):
                mask_elems = []
                for idx_inner, mask_val in np.ndenumerate(mask[idx_outer, ...]):
                    mask_elems.append(mask_val) # split_idxs[dim]
                accepted_slices[idx_outer] = sum(mask_elems)/len(mask_elems) >= threshold
            dim_corrected = dim if not dim >= time_axis else dim + 1
            boundaries.append((split_idxs[dim_corrected][np.where(accepted_slices)[0][0]], 
                               split_idxs[dim_corrected][np.where(accepted_slices)[0][-1] + 1]))
            mask = np.moveaxis(mask, source = 0, destination = dim)
    else:
        raise NotImplementedError
    return boundaries


class Domain_Pruner(object):
    def __init__(self, domain_selector : Callable = majority_rule, domain_selector_kwargs : dict = dict()):
        self.domain_selector = domain_selector
        self.domain_selector_kwargs = domain_selector_kwargs
        self.bds_init = False
    
    def get_boundaries(self, pivotal_tensor : np.ndarray, division_fractions : Union[int, list, tuple] = 3, 
                       time_axis = None, rectangular : bool = True):
        if time_axis is None:
            try:
                time_axis = global_var.time_axis
            except AttributeError:
                time_axis = 0
        self.time_axis = time_axis
        self.split_idxs, self.accepted_spatial_domains = get_subdomains_mask(pivotal_tensor, division_fractions, 
                                                                   self.domain_selector, 
                                                                   self.domain_selector_kwargs, self.time_axis)
        self.bds = pruned_domain_boundaries(self.accepted_spatial_domains, self.split_idxs, self.time_axis,
                                            rectangular)
        self.bds_init = True

    def prune(self, tensor):
        if not self.bds_init:
            raise AttributeError('Tring to use domain pruning boundaries before defining them. Use self.get_boundaries(...) beforehand.')
        tensor_new = np.copy(tensor)
        for axis_idx in np.arange(tensor_new.ndim):
            if axis_idx != self.time_axis:
                tensor_new = np.moveaxis(tensor_new, source = axis_idx, destination=0)
                bd_idx = axis_idx if axis_idx < self.time_axis else axis_idx - 1
                tensor_new = tensor_new[self.bds[bd_idx][0] : self.bds[bd_idx][1], ...]
                tensor_new = np.moveaxis(tensor_new, source = 0, destination=axis_idx)                
        return tensor_new