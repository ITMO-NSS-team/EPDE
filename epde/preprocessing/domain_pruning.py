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
    """
    Scales the input tensor by dividing it by its maximum value.
    
    This ensures that all values in the tensor are normalized to the range [0, 1].
    
    Args:
        tensor (np.ndarray): The input tensor to be scaled.
    
    Returns:
        np.ndarray: The scaled tensor with values in the range [0, 1].
    
    Why:
    This scaling is crucial for ensuring stable and efficient equation discovery.
    By normalizing the data, we prevent any single variable from dominating the equation search process due to its magnitude.
    """
    return tensor/np.max(tensor)


def split_tensor(tensor, fraction_along_axis: Union[int, list, tuple]):
    """
    Method for splitting a tensor into smaller blocks along specified axes.
    
        This function divides the input tensor into a grid of sub-tensors (blocks) based on the provided fractions for each axis.
        This is a preliminary step for parallel or distributed processing of the tensor, enabling the framework to analyze different
        sections of the data independently.
    
        Args:
            tensor (`np.ndarray`): The input tensor to be split.
            fraction_along_axis (`int|list|tuple`):  The number of fractions to divide each axis into.
                If an integer is provided, all axes will be split into the same number of fractions.
                If a list or tuple is provided, each element specifies the number of fractions for the corresponding axis.
    
        Returns:
            split_indexes_along_axis (`list`): A list of lists, where each inner list contains the indices along each axis where the tensor was split.
                These indices define the boundaries of the resulting blocks.
            block_matrix (`np.ndarray`): A NumPy array (object dtype) containing the split tensor blocks. The shape of this array
                corresponds to the fractions specified for each axis. Each element of this array holds a sub-tensor (block) of the original tensor.
    """
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
            raise TypeError(
                'Tensor fraction shall be declared in form of integer or list/tuple with elements for each axis in data')
        split_indexes = [int(tensor.shape[axis_idx]/fraction * element)
                         for element in range(fraction) if element != 0]
        split_indexes_along_axis.append(
            [0,] + split_indexes + [tensor.shape[axis_idx]])
        for ts_idx, tensor_section in enumerate(fragments):
            splitted_op = np.split(
                tensor_section, split_indexes, axis=axis_idx)
            temp_fragments.extend(splitted_op)
            for index_idx, idx in enumerate([indexes[ts_idx] for i in range(fraction)]):
                temp = copy.copy(idx)
                temp[axis_idx] = index_idx
                temp_indexes.append(temp)
        fragments = temp_fragments
        indexes = temp_indexes
    if isinstance(fraction_along_axis, (list, tuple)):
        block_matrix = np.empty(fraction_along_axis, dtype=object)
    else:
        block_matrix = np.empty(
            [fraction_along_axis for i in range(tensor.ndim)], dtype=object)
    for idx, tensor_fragment in enumerate(fragments):
        block_matrix[tuple(indexes[idx])] = tensor_fragment
    return split_indexes_along_axis, block_matrix


def majority_rule(tensors: Union[list, np.ndarray], threshold: float = 1e-2):
    """
    Applies majority rule to a list of tensors based on a non-constancy condition.
    
    This method checks if a majority of the input tensors exhibit sufficient variation,
    indicating they are not constant. This is crucial for identifying active terms
    in a differential equation. The non-constancy condition checks if the norm
    of a tensor is greater than a threshold times its size.
    
    Args:
      tensors: A list or numpy array of tensors to evaluate.
      threshold: A threshold value used to determine non-constancy.
    
    Returns:
      bool: True if the majority of tensors satisfy the non-constancy condition,
        False otherwise.
    """
    def non_constancy_cond(x): return np.linalg.norm(x) > (threshold * x.size)
    return sum([non_constancy_cond(x_elem) for x_elem in tensors])/len(tensors) >= 0.5


def get_subdomains_mask(tensor, division_fractions: Union[int, list, tuple], domain_selector: Callable,
                        domain_selector_kwargs: dict, time_axis: int):
    """
    Method for creating a mask that identifies valuable subdomains within the input tensor, which are then used for equation discovery.
    
        Args:
            tensor (`np.ndarray`): The input data tensor.
            division_fractions (`int|list|tuple`): The number of divisions for each axis (or a single number for all non-time axes).
            domain_selector (`callable`): A function to select valuable subdomains.
            domain_selector_kwargs (`dict`): Keyword arguments for the `domain_selector` function.
            time_axis (`int`): The index of the time axis in the tensor.
    
        Returns:
            split_idxs (`list`): The indices where the tensor was split.
            accepted_spatial_domains (`np.ndarray with boolen values`): A boolean mask indicating the valuable subdomains.
    
        WHY: This method divides the input tensor into smaller subdomains and uses a domain selector function to identify those subdomains that are most informative for discovering the underlying differential equations. The mask is then used to focus the equation discovery process on the most relevant parts of the data.
    """
    if isinstance(division_fractions, int):
        sd_shape = tuple([division_fractions for idx in range(
            tensor.ndim) if idx != time_axis])
    else:
        sd_shape = tuple([div_frac for idx, div_frac in enumerate(
            division_fractions) if idx != time_axis])
    # [division_fraction for i in self.]
    accepted_spatial_domains = np.full(shape=(sd_shape), fill_value=True, dtype=bool)
    tensor = scale_values(tensor)
    split_idxs, time_deriv_fragments = split_tensor(tensor, division_fractions)
    time_deriv_fragments = np.moveaxis(a=time_deriv_fragments, source=time_axis, 
					 destination=-1)  # for

    temp = []
    counter = 0
    for idx, partial_tensor in np.ndenumerate(time_deriv_fragments):
        if counter < time_deriv_fragments.shape[-1] - 1:
            temp.append(partial_tensor)
            counter += 1
        else:
            temp.append(partial_tensor)
            accepted_spatial_domains[idx[:-1]] = domain_selector(temp, **domain_selector_kwargs)
            temp = []
            counter = 0
    return split_idxs, accepted_spatial_domains


def pruned_domain_boundaries(mask: np.ndarray, split_idxs: list, time_axis: int,
                             rectangular: bool = True, threshold: float = 0.5):  # Разбораться с методом
    """
    Method for finding boundaries by mask

    Args:
    """
    Method for determining the spatial and temporal boundaries of a subdomain based on a significance mask.
    
        This method identifies the extent of a region where the dynamics are considered significant,
        effectively pruning the domain to focus on areas of interest. This is achieved by filtering
        slices along each axis based on the density of significant points within them.
    
        Args:
            mask (`np.ndarray`): Boolean mask indicating the significance of dynamics within the subdomain.
            split_idxs (`list`): Indices defining the divisions along each axis.
            time_axis (`int`): Index of the time axis.
            rectangular (`bool`): Flag indicating whether the area is rectangular (default: True).
            threshold (`float`): Percentage of significant data required in a slice to be considered part of the boundary (default: 0.5).
    
        Returns:
            boundaries (`list`): A list of tuples, where each tuple represents the start and end indices of the significant region along each axis.
    """
        mask (`np.ndarray`): boolean mask containg flags about significance all the dynamics in subdomain
        split_idxs (`list`): indexs of place of division into fraction
        time_axis (`int`): number of time axis
        rectangular (`bool`): flag indecating that area is rectangle
        threshold (`float`): optional, default - 0.5
            Percentage of significant data in a series to save it. Using for filtration of values on each axis.
    
    Returns:
        boundaries (`list`): boundaries for each axis with significant data
    """
    if rectangular:
        boundaries = []
        for dim in np.arange(mask.ndim):
            mask = np.moveaxis(mask, source=dim, destination=0)
            accepted_slices = np.empty(shape=mask.shape[0], dtype=bool)
            for idx_outer in np.arange(mask.shape[0]):
                mask_elems = 0
                count_elems = 0
                for idx_inner, mask_val in np.ndenumerate(mask[idx_outer, ...]):
                    # mask_elems.append(mask_val)  # split_idxs[dim]
                    mask_elems += mask_val
                    count_elems += 1
                accepted_slices[idx_outer] = mask_elems/count_elems >= threshold
            dim_corrected = dim if not dim >= time_axis else dim + 1
            boundaries.append((split_idxs[dim_corrected][np.where(accepted_slices)[0][0]],
                               split_idxs[dim_corrected][np.where(accepted_slices)[0][-1] + 1]))
            mask = np.moveaxis(mask, source=0, destination=dim)
    else:
        raise NotImplementedError
    return boundaries


class DomainPruner(object):
    """
    Class for pruning the domain based on specified criteria, focusing on data selection and preparation for equation discovery.
    
    
        Attribites:
            domain_selector (`callable`): default - majority_rule. 
                Rule for selecting region.
            domain_selector_kwargs (`dict`): parameters for method in `domain_selector`.
            bds_init (`int`): flag for marking execution of the selector
    """

    def __init__(self, domain_selector: Callable = majority_rule, domain_selector_kwargs: dict = dict()):
        """
        Initializes the SimpleDomainSelector with a specific domain selection strategy.
        
                This class sets up the domain selection mechanism used to refine the search space
                during the equation discovery process. By storing the provided domain selector
                function and its associated keyword arguments, it prepares for subsequent
                domain pruning operations. This initialization is crucial for focusing the search
                on the most promising areas of the solution space, thereby improving the
                efficiency and accuracy of the equation discovery process.
        
                Args:
                    domain_selector: The function to use for selecting the domain. Defaults to majority_rule.
                    domain_selector_kwargs: Keyword arguments to pass to the domain selector function. Defaults to an empty dictionary.
        
                Returns:
                    None
        """
        self.domain_selector = domain_selector
        self.domain_selector_kwargs = domain_selector_kwargs
        self.bds_init = False

    def get_boundaries(self, pivotal_tensor: np.ndarray, division_fractions: Union[int, list, tuple] = 3,
                       time_axis=None, rectangular: bool = True):
        """
        Method for determining the boundaries of the refined domain based on significant regions in the data.
        
                This method identifies and defines the boundaries of the most relevant areas within the data
                by analyzing a given tensor and dividing the domain into subregions.
                It focuses on isolating the portions of the data that exhibit the most prominent features,
                effectively narrowing down the search space for equation discovery.
                This is done to improve the efficiency and accuracy of the equation discovery process
                by focusing on the most informative parts of the data.
                
                Args:
                    pivotal_tensor (`np.ndarray`): A tensor highlighting important areas within the data,
                        guiding the domain refinement process by emphasizing regions with significant values.
                    division_fractions (`int|list|tuple`): Specifies the number of subregions to divide the domain into.
                        If an integer is provided, the domain is divided equally along each dimension.
                        A list or tuple allows for specifying different division fractions for each dimension. Defaults to 3.
                    time_axis (`int`): The index of the axis representing time. If None, it attempts to retrieve
                        the time axis from a global variable. Defaults to None.
                    rectangular (`bool`): A flag indicating whether the refined domain should be constrained to a rectangular shape.
                        Defaults to True.
        
                Returns:
                    None. The method updates the internal state of the `DomainPruner` object,
                    specifically `self.bds` with the calculated boundaries.
        """
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
        """
        Applies the detected domain boundaries to the input tensor, effectively focusing on the dynamically relevant regions.
        
                This method refines the input tensor by removing data points outside the identified domain boundaries.
                This ensures that subsequent equation discovery focuses on the areas where the system exhibits meaningful dynamics,
                avoiding regions with minimal or no change.
        
                Args:
                    tensor (`np.ndarray`): Input data for pruning.
        
                Returns:
                    `np.ndarray`: The tensor with areas outside the learned domain boundaries removed.
        """
        if not self.bds_init:
            raise AttributeError(
                'Tring to use domain pruning boundaries before defining them. Use self.get_boundaries(...) beforehand.')
        tensor_new = np.copy(tensor)
        for axis_idx in np.arange(tensor_new.ndim):
            if axis_idx != self.time_axis:
                tensor_new = np.moveaxis(
                    tensor_new, source=axis_idx, destination=0)
                bd_idx = axis_idx if axis_idx < self.time_axis else axis_idx - 1
                tensor_new = tensor_new[self.bds[bd_idx][0]:self.bds[bd_idx][1], ...]
                tensor_new = np.moveaxis(tensor_new, source=0, destination=axis_idx)
        return tensor_new
