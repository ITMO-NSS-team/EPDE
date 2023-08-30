'''
The cache object is introduced to reduce memory usage by storing the values of terms/factors of the discovered equations.

Functions:
    upload_simple_tokens: uploads the basic factor into the cache with its value in ndimensional numpy.array
    download_variable: download a variable from the disc by its and its derivatives file names, select axis for time (for normalization purposes) & cut values near area boundary

Objects:
    Cache: see object description (tbd)

The recommended way to declare the cache object isto declare it as a global variable:
    >>> import src.globals as global_var
    >>> global_var.cache.memory_usage_properties(obj_test_case=XX, mem_for_cache_frac = 25) #  XX - np.ndarray from np.meshgrid, mem_for_cache_frac - max part of memory to be used for cache, %
    >>> print(global_var.cache.consumed_memory)

'''

"""
This is an example of Google style.

Args:
    param1: This is the first param.
    param2: This is a second param.

Returns:
    This is a description of what is returned.

Raises:
    KeyError: Raises an exception.
"""

# !/usr/bin/env python3
# -*- coding: utf-8 -*-

from warnings import warn
import numpy as np
import psutil
from functools import partial

from typing import Union, Callable

from copy import deepcopy
from collections import OrderedDict


def upload_simple_tokens(labels, cache, tensors, grid_setting=False):
    """
    Uploads the basic factor into the cache with its value in ndimensional numpy.array

    Args:
        labels: list or 1-d array with string name of coefficients
        cache (`Cache`): keeping values of terms/factors of equations.
        tensors (`numpy.ndarray`): values for coefficients, shape of array: (n, m, ...), where n is number of coefficients
        grid_settings:  optional, boolean argument, default - False

    Returns:
        None
    """
    for idx, label in enumerate(labels):
        if grid_setting:
            label_completed = label
        else:
            label_completed = (label, (1.0,))
        cache.add(label_completed, tensors[idx])
        cache.add_base_matrix(label_completed)


def upload_grids(grids, cache):
    """
    Grids are saved into the base matrices of the cache

    Args:
        grids (`list|tuple|numpy.ndarray`): value of grids
        cache (`Cache`): object where grids wiil be stored
    
    Returns:
        None
    """
    if type(grids) == list or type(grids) == tuple:
        labels = [str(idx) for idx, _ in enumerate(grids)]
        tensors = grids
    elif type(grids) == np.ndarray:
        labels = ['0',]
        tensors = [grids,]
    upload_simple_tokens(labels=labels, cache=cache, tensors=tensors, grid_setting=True)
    cache.use_structural()


def prepare_var_tensor(var_tensor, derivs_tensor, time_axis):
    """
    Method for transformation of the input data, the time axis is placed first

    Args:
        var_tensor: numpy.array, 
        derivs_tensor: numpy.ndarray, 
        time_axis:
    Returns:
        result (`numpy.ndarray`): formed data for the algorithm

    """
    initial_shape = var_tensor.shape
    print('initial_shape', initial_shape, 'derivs_tensor.shape', derivs_tensor.shape)
    var_tensor = np.moveaxis(var_tensor, time_axis, 0)
    result = np.ones((1 + derivs_tensor.shape[-1], ) + tuple([shape for shape in var_tensor.shape]))  # - 2*boundary

    increment = 1
    result[increment - 1, :] = var_tensor#, boundary, cut_except)
    if derivs_tensor.ndim == 2:
        for i_outer in range(0, derivs_tensor.shape[1]):
            result[i_outer+increment, ...] = np.moveaxis(derivs_tensor[:, i_outer].reshape(initial_shape), # np_ndarray_section( , boundary, cut_except)
                         source=time_axis, destination=0)
    else:
        for i_outer in range(0, derivs_tensor.shape[-1]):
            assert derivs_tensor.ndim == var_tensor.ndim + increment, 'The shape of tensor of derivatives does not correspond '
            result[i_outer+increment, ...] = np.moveaxis(derivs_tensor[..., i_outer], # np_ndarray_section( -1, , boundary, [])
                             source=time_axis, destination=0)    
    return result


def download_variable(var_filename, deriv_filename, time_axis):
    """

    Args:
        var_filename: str, 
        deriv_filename: str, 
    Returns:

    """
    var = np.load(var_filename)
    derivs = np.load(deriv_filename)
    tokens_tensor = prepare_var_tensor(var, derivs, time_axis)
    return tokens_tensor


class Cache(object):
    """Class for keeping values of terms/factors of equations.

    Args:
        max_allowed_tensors (`int`): limitation on the number of allowed tensors to load into rhe cache.
        memory_default (`dict`): key - name of tensor (tuple - (name_of_term, params)), value - derivative. Objects without changes after evolutional step
        memory_normalized (`dict`): key - name of tensor (tuple - (name_of_term, params)), value - derivative. Objects with normalize
        memory_structural (`dict`): key - name of tensor (tuple - (name_of_term, params)), value - derivative. NOT USED ДОПИСАТЬ ПРОСМОТРЯ КОД
    """
    def __init__(self):
        self.max_allowed_tensors = None
        self.memory_default = dict()
        self.memory_normalized = dict()
        self.memory_structural = dict()
        self.mem_prop_set = False 
        self.base_tensors = []  # storage of non-normalized tensors, that will not be affected by change of variables
        self.structural_and_base_merged = dict()

    def use_structural(self, use_base_data=True, label=None, replacing_data=None):
        # print(f'Setting structural data for {label}, for it: {use_base_data} - use_base_data')
        assert use_base_data or replacing_data is not None, 'Structural data must be declared with base data or by additional tensors.'
        if label is None:
            # self.structural_used = True
            if use_base_data:
                self.memory_structural = {key: val for key, val in self.memory_default.items()}
                try:
                    for key in self.memory_structural.keys():
                        self.structural_and_base_merged[key] = True
                except AttributeError as e:
                    print(f"Error in class Cache {e}")
            else:
                if type(replacing_data) != dict:
                    raise TypeError('Replacing data shall be set with dict of format: tuple - memory key: np.ndarray ')
                if np.any([type(entry) != np.ndarray for entry in replacing_data.values()]):
                    raise TypeError('Replacing data shall be set with dict of format: tuple - memory key: np.ndarray ')
                if replacing_data.keys() != self.memory_default.keys():
                    print(replacing_data.keys(), self.memory_default.keys())
                    raise ValueError('Labels for the new structural data do not with the baseline data ones.')
                if np.any([entry.shape != self.memory_default[label].shape for label, entry in replacing_data.items()]):
                    print([(entry.shape, self.memory_default[label].shape) for label, entry in replacing_data.items()])
                    raise ValueError('Shapes of tensors in new structural data do not match their counterparts in the base data')
                for key in self.memory_default.keys():
                    self.structural_and_base_merged[label] = False
                self.memory_structural = replacing_data
        elif type(label) == tuple:
            if use_base_data:
                self.structural_and_base_merged[label] = True
                if label not in self.memory_default.keys():
                    self.add(label=label, tensor=replacing_data)
            else:
                # print(self.structural_and_base_merged)
                self.structural_and_base_merged[label] = False
                if type(replacing_data) != np.ndarray:
                    raise TypeError('Replacing data with provided label shall be set with np.ndarray ')
                if label in self.memory_default.keys():
                    if replacing_data.shape != self.memory_default[label].shape:
                        raise ValueError('Shapes of tensors in new structural data do not match their counterparts in the base data')
                self.memory_structural[label] = replacing_data

    @property
    def g_func(self):  # , g_func: Union[Callable, type(None)] = None
        assert '0' in self.memory_default.keys()  # Check if we are working with the grid cache
        return self._g_func(self.get_all()[1])

    @g_func.setter
    def g_func(self, function: Callable):
        self._g_func = function

    def add_base_matrix(self, label):
        assert label in self.memory_default.keys()
        self.base_tensors.append(label)

    def set_boundaries(self, boundary_width: Union[int, list, tuple]):
        """
        Setting the number of unaccounted elements at the edges
        """
        assert '0' in self.memory_default.keys(), 'Boundaries should be specified for grid cache.'
        shape = self.get('0')[1].shape
        if isinstance(boundary_width, int):
            if any([elem <= 2*boundary_width for elem in shape]):
                raise IndexError(f'Mismatching shapes: boundary of {boundary_width} does not fit data of shape {shape}')
        elif isinstance(boundary_width, (list, tuple)):
            if any([elem <= 2*boundary_width[idx] for idx, elem in enumerate(shape)]):
                raise IndexError(f'Mismatching shapes: boundary of {boundary_width} does not fit data of shape {shape}')                
        else:
            raise TypeError(f'Incorrect type of boundaries: {type(boundary_width)}, instead of expected int or list/tuple')

        self.boundary_width = boundary_width

    def memory_usage_properties(self, obj_test_case=None, mem_for_cache_frac=None, mem_for_cache_abs=None):
        """
        Method for setting of memory using in algorithm's process

        Args:
            obj_test_case (`ndarray`): referntial tensor to evaluate memory consuption by tensors equation search
            mem_for_cache_frac (`int`): memory available for cache (in fraction of RAM). The default - None.
            mem_for_cache_abs (`int`): memory available for cache (in byte). The default - None.
        
        Returns:
            None
        """
        assert not (mem_for_cache_frac is None and mem_for_cache_abs is None), 'Avalable memory space not defined'
        assert obj_test_case is not None or len(self.memory_default) > 0, 'Method needs sample of stored matrix to evaluate memory allocation'
        if mem_for_cache_abs is None:
            self.available_mem = mem_for_cache_frac / 100. * psutil.virtual_memory().total  # Allocated memory for tensor storage, bytes
        else:
            self.available_mem = mem_for_cache_abs

        assert self.available_mem < psutil.virtual_memory().available

        if len(self.memory_default) == 0:
            assert obj_test_case is not None
            self.max_allowed_tensors = int(np.floor(self.available_mem/obj_test_case.nbytes)/2)
        else:
            self.max_allowed_tensors = int(np.floor(self.available_mem/self.memory_default[np.random.choice(list(self.memory_default.keys()))].nbytes))

        eps = 1e-7
        if np.abs(self.available_mem) < eps:
            print('The memory can not containg any tensor even if it is entirely free (This message can not appear)')

    def clear(self, full=False):
        if full:
            del self.memory_default, self.memory_normalized, self.memory_structural, self.base_tensors
            self.memory_default = dict()
            self.memory_normalized = dict()
            self.memory_structural = dict()
            self.base_tensors = []
        else:
            memory_new = dict()
            memory_new_norm = dict()
            memory_new_structural = dict()
            for key in self.base_tensors:
                memory_new[key] = self.memory_default[key]
                memory_new_norm[key] = self.memory_normalized[key]
                memory_new_structural[key] = self.memory_structural[key]

            del self.memory_default, self.memory_normalized, self.memory_structural
            self.memory_default = memory_new
            self.memory_normalized = memory_new_norm
            self.memory_structural = memory_new_structural

    def change_variables(self, increment, increment_structral=None):
        '''
            Additional regression in the search process, run on the structural data, is required to set the
            increment_structral tensor. ToDo!
        '''
        raise DeprecationWarning('No need to change variables in current version of EPDE.')
        assert not (increment_structral is None and not all(self.structural_and_base_merged)), 'Not all structural data taken from the default, and the increment for structural was not sent'

        random_key = list(self.memory_default.keys())[0]
        increment = np.reshape(increment, newshape=self.memory_default[random_key].shape)
        del self.memory_normalized
        self.memory_default = {key: self.memory_default[key] for key in self.base_tensors}  # deepcopy(self.base_tensors)
        self.memory_structural = {key: self.memory_structural[key] for key in self.base_tensors}  # deepcopy(self.base_tensors_structural)
        self.memory_normalized = dict()
        for key in self.memory_default.keys():
            assert np.all(self.memory_default[key].shape == increment.shape)
            self.memory_default[key] = self.memory_default[key] - increment
            if not self.structural_and_base_merged[key]:
                self.memory_structural[key] = self.memory_structural[key] - increment_structral

    def add(self, label, tensor, normalized: bool = False, structural: bool = False,
            indication: bool = False):
        '''
        Method for addition of a new tensor into the cache. Returns True if there was enough memory and the tensor was save, and False otherwise.
        '''
        assert not (normalized and structural), 'The added matrix can not be simultaneously normalized and structural. Possibly, bug in token/term saving'
        if normalized:
            if self.max_allowed_tensors is None:
                self.memory_usage_properties(obj_test_case=tensor, mem_for_cache_frac=5)
            if (len(self.memory_normalized) + len(self.memory_default) + len(self.memory_structural) < self.max_allowed_tensors and
                label not in self.memory_normalized.keys()):
                self.memory_normalized[label] = tensor
                if indication:
                    print('Enough space for saved normalized term ', label, tensor.nbytes)
                return True
            elif label in self.memory_normalized.keys():
                assert np.all(np.isclose(self.memory_normalized[label], tensor))
                if indication:
                    print('The term already present in normalized cache, no addition required', label, tensor.nbytes)
                return True
            else:
                if indication:
                    print('Not enough space for term ', label, tensor.nbytes, 'Can save only', self.max_allowed_tensors, 'tensors. While already uploaded ', len(self.memory_normalized) + len(self.memory_default) + len(self.memory_structural))
                return False
        elif structural:
            raise NotImplementedError('The structural data must be added with cache.use_structural method')
        else:
            if self.max_allowed_tensors is None:
                self.memory_usage_properties(obj_test_case=tensor, mem_for_cache_frac=5)
            if (len(self.memory_normalized) + len(self.memory_default) + len(self.memory_structural) < self.max_allowed_tensors and
                label not in self.memory_default.keys()):
                self.memory_default[label] = tensor
                if indication:
                    print('Enough space for saved unnormalized term ', label, tensor.nbytes)
                return True
            elif label in self.memory_default.keys():
                assert np.all(np.isclose(self.memory_default[label], tensor))
                if indication:
                    print('The term already present in unnormalized cache, no addition required', label, tensor.nbytes)
                return True
            else:
                if indication:
                    print('Not enough space for term ', label, tensor.nbytes)
                return False

    def delete_entry(self, entry_label):
        if entry_label not in self.memory_default.keys():
            raise ValueError('deleted element already not in memory')
        del self.memory_default[entry_label]
        try:
            del self.memory_structural[entry_label]
        except KeyError:
            pass
        try:
            del self.memory_normalized[entry_label]
        except KeyError:
            pass

    def get(self, label, normalized=False, structural=False, saved_as=None):
        assert not (normalized and structural), 'The added matrix can not be simultaneously normalized and scaled'
#        assert not scaled or self.scale_used, 'Trying to add scaled data, while the cache it not allowed to get it'
        if label is None:
            print(self.memory_default.keys())
            return np.random.choice(list(self.memory_default.values()))
        if normalized:
            try:
                return self.memory_normalized[label]
            except KeyError:
                print('memory keys: ', self.memory_normalized.keys())
                print('fetched label:', label, ' prev. known as ', saved_as)
                raise KeyError('Can not fetch tensor from cache with normalied data')
        elif structural:
            try:
                return self.memory_default[label] if self.structural_and_base_merged[label] else self.memory_structural[label]
            except KeyError:
                print('structural', structural)
                print('self.structural_and_base_merged', self.structural_and_base_merged.keys())
                print('self.memory_default', self.memory_default.keys())
                if self.structural_and_base_merged[label]:
                    print('memory keys (structural data taken from the default): ', self.memory_default.keys())
                else:
                    print('memory keys: ', self.memory_structural.keys())
                print('fetched label:', label, ' prev. known as ', saved_as)
                raise KeyError('Can not fetch tensor from cache with normalied data')
        else:
            try:
                return self.memory_default[label]
            except KeyError:
                print('memory keys: ', self.memory_default.keys())
                print('fetched label:', label, ' prev. known as ', saved_as)
                raise KeyError('Can not fetch tensor from cache with non-normalied data')

    def get_all(self, normalized=False, structural=False):
        if normalized:
            processed_mem = self.memory_normalized
        elif structural:
            processed_mem = self.memory_structural
        else:
            processed_mem = self.memory_default

        keys = []
        tensors = []
        for key, value in processed_mem.items():
            keys.append(key)
            tensors.append(value)

        return keys, tensors

    def __contains__(self, obj):
        '''
        Valid input type:
            'label' (checked in unnormalized data); ('label1', normalized), where normalized is bool (T if norm, else F);
            np.ndarray of values (checked in unnormalized data); (np.ndarray, normalized), where normalized is bool
            (T if norm, else F) and np.ndarray is np.ndarray of tensor values. Does not support scaled vals
        '''
        if (type(obj) == tuple or type(obj) == list) and type(obj[0]) == str:
            return obj in self.memory_default.keys()
        elif (type(obj) == tuple or type(obj) == list) and type(obj[0]) == tuple and type(obj[1]) == bool:
            if obj[1]:
                return obj[0] in self.memory_normalized.keys()
            else:
                return obj[0] in self.memory_default.keys()
        elif type(obj) == np.ndarray:
            try:
                return np.any([np.all(obj == entry_values) for entry_values in self.memory_default.values()])
            except:
                return False
        elif (type(obj) == tuple or type(obj) == list) and type(obj[0]) == np.ndarray and type(obj[1]) == bool:
            try:
                if obj[1]:
                    return np.any([np.all(obj[0] == entry_values) for entry_values in self.memory_normalized.values()])
                else:
                    return np.any([np.all(obj[0] == entry_values) for entry_values in self.memory_default.values()])
            except:
                return False
        else:
            raise NotImplementedError('Invalid format of function input to check, if the object is in cache')

#    def __iter__(self):
#        for key in self.memory_default.keys()
    def prune_tensors(self, pruner, mem_to_process: list = ['default', 'structural', 'normalized']):
        mem_arranged = {'default': self.memory_default,
                            'structural': self.memory_structural,
                            'normalized': self.memory_normalized}

        for key in self.memory_default.keys():
            for mem_type in mem_to_process:
                try:
                    mem_arranged[mem_type][key] = pruner.prune(mem_arranged[mem_type][key])
                except (NameError, KeyError) as e:
                    pass

    @property
    def consumed_memory(self):
        memsize = np.sum([value.nbytes for _, value in self.memory_default.items()])
        memsize += np.sum([value.nbytes for _, value in self.memory_normalized.items()])
        for label, merged_state in self.structural_and_base_merged.items():
            if not merged_state: memsize += self.memory_structural[label].nbytes
        return memsize


def upload_complex_token(label: str, params_values: OrderedDict, evaluator, tensor_cache: Cache, grid_cache: Cache):
    # label_completed = (label, (1.0,))
    # value = evaluator.apply(self)
    # global_var.tensor_cache.add(self.cache_label, value, structural = False)
    try:
        evaluation_function = evaluator.evaluation_functions[label]
    except TypeError:
        evaluation_function = evaluator.evaluation_functions
    _, grids = grid_cache.get_all()
    grid_function = np.vectorize(lambda args: evaluation_function(*args, **params_values))
    indexes_vect = np.empty_like(grids[0], dtype=object)
    for tensor_idx, _ in np.ndenumerate(grids[0]):
        indexes_vect[tensor_idx] = tuple([grid[tensor_idx] for grid in grids])

    label_completed = (label, tuple(params_values.values()))
    tensor_cache.add(label_completed, grid_function(indexes_vect))
