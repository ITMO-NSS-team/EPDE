'''
The cache object is introduced to reduce memory usage by storing the values of terms/factors of the discovered equations.

Functions:
    upload_simple_tokens: uploads the basic factor into the cache with its value in ndimensional numpy.array
    download_variable: download a variable from the disc by its and its derivatives file names, select axis for time (for normalization purposes) & cut values near area boundary

Objects:
    Cache: see object description (tbd)

The recommended way to declare the cache object isto declare it as a global variable:
    >>> import src.globals as global_var
    >>> global_var.cache.memoryUsageProperties(obj_test_case=XX, mem_for_cache_frac = 25) #  XX - np.ndarray from np.meshgrid, mem_for_cache_frac - max part of memory to be used for cache, %
    >>> print(global_var.cache.consumed_memory)

'''

# !/usr/bin/env python3
# -*- coding: utf-8 -*-

from warnings import warn
import numpy as np
import psutil

from typing import Union, Callable, List, Literal

import torch
from collections import OrderedDict
try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable


def uploadSimpleTokens(labels, cache, tensors, 
                       deriv_codes: List = None, grid_setting = False):
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
    if deriv_codes is not None and len(deriv_codes) != len(labels):
        print(deriv_codes, labels)
        raise ValueError('Incorrect number of deriv codes passed, expected ')
    
    for idx, label in enumerate(labels):
        if grid_setting:
            label_completed = label
            deriv_code = None
        else:
            label_completed = (label, (1.0,))
            deriv_code = None if deriv_codes is None else deriv_codes[idx]
        cache.add(label_completed, tensors[idx], deriv_code = deriv_code)
        cache.add_base_matrix(label_completed)


def uploadGrids(grids, cache):
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
    uploadSimpleTokens(labels=labels, cache=cache, tensors=tensors, grid_setting=True)


def prepareVarTensor(var_tensor, derivs_tensor, time_axis):
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
    result = np.ones((derivs_tensor.shape[-1], ) + tuple([shape for shape in var_tensor.shape]))

    increment = 0
    if derivs_tensor.ndim == 2:
        for i_outer in range(0, derivs_tensor.shape[1]):
            result[i_outer+increment, ...] = np.moveaxis(derivs_tensor[:, i_outer].reshape(initial_shape),
                                                         source=time_axis, destination=0)
    else:
        for i_outer in range(0, derivs_tensor.shape[-1]):
            assert derivs_tensor.ndim == var_tensor.ndim + 1, 'The shape of tensor of derivatives does not correspond '
            result[i_outer+increment, ...] = np.moveaxis(derivs_tensor[..., i_outer],
                                                         source=time_axis, destination=0)
    return result



class Cache(object):
    """Class for keeping values of terms/factors of equations.

    Args:
        max_allowed_tensors (`int`): limitation on the number of allowed tensors to load into the cache.
        memory_default (`dict`): key - name of tensor (tuple - (name_of_term, params)), value - derivative. Objects without changes after evolutional step
        memory_normalized (`dict`): key - name of tensor (tuple - (name_of_term, params)), value - derivative. Objects with normalize
    """
    def __init__(self, device: Literal['cpu', 'cuda'] = 'cpu', backend: Literal['torch', 'numpy'] = 'numpy'):
        if device == 'cuda' and backend == 'numpy':
            raise NotImplementedError('Numpy operations for graphics cards is not implemented.')
            # TODO: implement cupy versions of conventional numpy operations

        self._device = device
        self._backend = backend

        self.max_allowed_tensors = None

        self.initMemory()

        self._deriv_codes = [] # Elements of this list must be tuples with the first element - 
                               # deriv code ([var, term]) like ([1], [0]) for dy/dt in LV, and the second - cache label in 
                               # standard form ('dy/dx1', (1.0,))

    def initMemory(self) -> None:
        self.memory_default    = dict() # In this dict, keys should be IDs of domains, arguments - matching types of tensors/arrays
        self.memory_normalized = dict() # 'numpy' : dict()
        # self.memory_anns       = dict() # Maybe, initialize storage for trained NNs

    @staticmethod
    def getMemLen(self, mem_dict: dict) -> int:
        return sum([len(elem) for elem in mem_dict])

    def attrsFromDict(self, attributes, except_attrs: dict = {}) -> None:
        except_attrs['obj_type'] = None
        self.__dict__ = {key : item for key, item in attributes.items()
                         if key not in except_attrs.keys}
        for key, elem in except_attrs.items():
            if elem is not None:
                self.__dict__[key] = elem

    def memoryUsageProperties(self, obj_test_case=None, mem_for_cache_frac=None, mem_for_cache_abs=None):
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
        assert obj_test_case is not None or self.getMemLen(self.memory_default) > 0, \
             'Method needs sample of stored matrix  to evaluate memory allocation'
        if mem_for_cache_abs is None:
            self.available_mem = mem_for_cache_frac / 100. * psutil.virtual_memory().total  # Allocated memory for tensor storage, bytes
        else:
            self.available_mem = mem_for_cache_abs

        assert self.available_mem < psutil.virtual_memory().available

        if self.getMemLen(self.memory_default) == 0:
            assert obj_test_case is not None
            self.max_allowed_tensors = int(np.floor(self.available_mem/obj_test_case.nbytes)/2)
        else:
            # key = np.random.choice(list(self.memory_default['numpy'].keys()))
            key = (self.memory_default[next(iter(self.memory_default.keys()))])
            self.max_allowed_tensors = int(np.floor(self.available_mem/
                                                    next(iter(self.memory_default))[key].nbytes))

        eps = 1e-7
        if np.abs(self.available_mem) < eps:
            print('The memory can not containg any tensor even if it is entirely free (This message can not appear)')

    def clear(self, full=False) -> None:
        raise NotImplementedError('Depricated method! Cache is not expected to be cleared.')

    def add(self, label, tensor, subcache_ID: int, normalized: bool = False,
            deriv_code = None, indication: bool = False) -> bool:
        '''
        Method for addition of a new tensor into the cache.
        Returns True if there was enough memory and the tensor was save, and False otherwise.
        '''
        if subcache_ID not in self.memory_default.keys():
            self.memory_default[subcache_ID] = {}
        if subcache_ID not in self.memory_normalized.keys():
            self.memory_normalized[subcache_ID] = {}

        if deriv_code is not None:
            self._deriv_codes.append((deriv_code, label))
        
        if normalized:
            if self.max_allowed_tensors is None:
                self.memoryUsageProperties(obj_test_case=tensor, mem_for_cache_frac=5)
            if ((self.getMemLen(self.memory_normalized) + # ADD FUNCTION TO RETURN TOTAL LEN OF THE SPECIFIC MEMORY
                 self.getMemLen(self.memory_default)) < self.max_allowed_tensors and
                label not in self.memory_normalized[subcache_ID].keys()):
                self.memory_normalized[subcache_ID][label] = tensor
                if indication:
                    print('Enough space for saved normalized term ', label, tensor.nbytes)
                return True
            elif label in self.memory_normalized[subcache_ID].keys():
                if indication:
                    print('The term already present in normalized cache, no addition required', label, tensor.nbytes)
                return True
            else:
                if indication:
                    print('Not enough space for term ', label, tensor.nbytes, 'Can save only', self.max_allowed_tensors, 
                          'tensors. While already uploaded ', self.getMemLen(self.memory_normalized) + self.getMemLen(self.memory_default))
                return False
        else:
            if self.max_allowed_tensors is None:
                self.memoryUsageProperties(obj_test_case=tensor, mem_for_cache_frac=5)
            if ((self.getMemLen(self.memory_normalized) +
                 self.getMemLen(self.memory_default)) < self.max_allowed_tensors and
                label not in self.memory_default[subcache_ID].keys()):
                self.memory_default[subcache_ID][label] = tensor
                if indication:
                    print('Enough space for saved unnormalized term ', label, tensor.nbytes)
                return True
            elif label in self.memory_default[subcache_ID].keys():
                if indication:
                    print('The term already present in unnormalized cache, no addition required', label, tensor.nbytes)
                return True
            else:
                if indication:
                    print('Not enough space for term ', label, tensor.nbytes)
                return False

    def deleteEntry(self, entry_label):
        print(f'Deleting {entry_label} from all memory dicts in cache.')
        if not all(entry_label in memdict.keys() for memdict in self.memory_default.values()):
            raise ValueError('deleted element already not in memory')
        
        for memkey in self.memory_default.keys():
            del self.memory_default[memkey][entry_label]

            try:
                del self.memory_normalized[memkey][entry_label]
            except KeyError:
                pass    

    def get(self, label: tuple = None, subcache_ID: int = None, normalized: bool = False, deriv_code = None) -> np.ndarray:
        if subcache_ID is None:
            subcache_ID = next(iter(self.memory_default.keys()))
        
        if deriv_code is not None: # Condition not verified in rework due to looking plausible
            assert label is None, 'Get method got both deriv_code and explicit label'
            label = [elem[1] for elem in self._deriv_codes if elem[0] == deriv_code][0]

        if label is None:
            return np.random.choice(list(next(iter(self.memory_default)).values()))
        
        if normalized:
            return self.memory_normalized[subcache_ID][label]              
        else:
            return self.memory_default[subcache_ID][label]

    def get_all(self, normalized: bool = False, subcache_ID: int = None):
        if subcache_ID is None:
            subcache_ID = next(iter(self.memory_default.keys()))

        if normalized:
            processed_mem = self.memory_normalized[subcache_ID]
        else:
            processed_mem = self.memory_default[subcache_ID]

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
        
        Method returns True if only the required entry is present in every subcache.
        '''
        # subcache_IDs = list(self.memory_default.keys())

        if isinstance(obj, (tuple, list)) and isinstance(obj[0], str):
            return all([obj in memdict.keys() for memdict in self.memory_default.values()])

        elif isinstance(obj, (tuple, list)) and isinstance(obj[0], tuple, frozenset) and isinstance(obj[1], bool):
            if obj[1]:
                return all([obj[0] in memdict.keys() for memdict in self.memory_normalized.values()])
            else:
                return all([obj[0] in memdict.keys() for memdict in self.memory_default.values()])

        elif isinstance(obj, np.ndarray):
            try:
                return all([any([np.isclose(obj, entry_values) for entry_values in memdict.values()])
                            for memdict in self.memory_default.values()])
            except:
                return False

        elif type(obj) == torch.Tensor:
            raise NotImplementedError('Depricated support of torch tensors.')

        elif isinstance(obj, (tuple, list)) and isinstance(obj[0], np.ndarray) and isinstance(obj[1], bool):
            try:
                if obj[1]:
                    return all([any([np.isclose(obj[0], entry_values) for entry_values in memdict.values()]) 
                                for memdict in self.memory_normalized.values()])
                else:
                    return all([any([np.isclose(obj[0], entry_values) for entry_values in memdict.values()]) 
                                for memdict in self.memory_default.values()])
            except:
                return False
            
        elif isinstance(obj, (tuple, list)) and isinstance(obj[0], torch.Tensor) and isinstance(obj[1], bool):
            raise NotImplementedError('Depricated support of torch tensors.')
    
        else:
            raise NotImplementedError('Invalid format of function input to check, if the object is in cache')

    def prune_tensors(self, pruner, subcache_ID: int, mem_to_process: List[str] = ['default', 'normalized']):

        if 'default' in mem_to_process:
            for array_key in self.memory_default[subcache_ID].keys():
                try:
                    self.memory_default[subcache_ID][array_key] = pruner.prune(self.memory_default[subcache_ID][array_key])
                except (NameError, KeyError) as e:
                    pass
        
        if 'normalized' in mem_to_process:
            for array_key in self.memory_normalized[subcache_ID].keys():
                try:
                    self.memory_normalized[subcache_ID][array_key] = pruner.prune(self.memory_normalized[subcache_ID][array_key])
                except (NameError, KeyError) as e:
                    pass

    @property
    def consumed_memory(self):
        memsize = 0

        for subcache_idx in self.memory_default.keys():
            assert subcache_idx in self.memory_normalized.keys()
            memsize += sum([value.nbytes for value in self.memory_default[subcache_idx].values()])
            memsize += sum([value.nbytes for value in self.memory_normalized[subcache_idx].values()])

        return memsize


def upload_complex_token(label: str, params_values: OrderedDict, evaluator, tensor_cache: Cache, grid_cache: Cache):
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