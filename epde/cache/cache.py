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

# !/usr/bin/env python3
# -*- coding: utf-8 -*-

from warnings import warn
import numpy as np
import psutil
from functools import partial

from typing import Union, Callable, List

import torch
from copy import deepcopy
from collections import OrderedDict
try:
    from collections.abc import Iterable
except ImportError:
    from collections import Iterable


def upload_simple_tokens(labels, cache, tensors, deriv_codes: List = None, 
                         grid_setting=False):
    """
    Uploads basic factors (tokens) into the cache, associating them with their numerical values.
    
        This function prepares the elementary building blocks for constructing more complex equation terms.
        It iterates through provided labels and their corresponding tensor values, storing them within the cache.
        The cache is later used to efficiently evaluate and manipulate these factors during the equation discovery process.
    
        Args:
            labels (list or numpy.ndarray): A list of string names representing the coefficients or variables.
            cache (Cache): The cache object used for storing and retrieving factor values.
            tensors (numpy.ndarray): A multi-dimensional array containing the numerical values for each coefficient.
                The shape should be (n, m, ...), where n is the number of coefficients.
            deriv_codes (List, optional): List of derivative codes associated with each label. Defaults to None.
            grid_setting (bool, optional): A flag indicating whether the labels are part of a grid setting. Defaults to False.
    
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


def upload_grids(grids, cache):
    """
    Saves grid data into the cache for subsequent equation discovery.
    
    This method prepares grid data by converting it into a suitable format 
    and storing it within the cache. This allows the evolutionary algorithm 
    to efficiently access and utilize the grid data during the equation 
    search process. The grid data represents the independent variable space 
    over which the solution is defined.
    
    Args:
        grids (`list|tuple|numpy.ndarray`): Grid data to be stored. Can be a list/tuple of grids or a single numpy array.
        cache (`Cache`): The cache object where the grid data will be stored.
    
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
    Transforms the input variable and its derivatives to a format suitable for equation discovery. The time axis is moved to the first position to align the data for subsequent processing by the evolutionary algorithm. This rearrangement ensures that the temporal dependencies are properly considered when searching for the best-fitting differential equation.
    
        Args:
            var_tensor (numpy.ndarray): The input variable tensor.
            derivs_tensor (numpy.ndarray): The tensor containing the derivatives of the variable.
            time_axis (int): The axis representing time in the input tensors.
    
        Returns:
            numpy.ndarray: A reshaped tensor with the time axis moved to the first position, ready for equation discovery.
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

def switch_format(inp: Union[torch.Tensor, np.ndarray], device = 'cpu'):
    """
    Converts between NumPy arrays and PyTorch tensors to ensure compatibility between different computational stages within the EPDE framework.
    
        This function facilitates seamless data exchange between numerical solvers (often requiring NumPy arrays) and the evolutionary search process (leveraging PyTorch tensors for GPU acceleration and automatic differentiation).
        
        Args:
            inp: The input NumPy array or PyTorch tensor.
            device: The device to move the tensor to (only relevant when converting from NumPy to Torch). Defaults to 'cpu'.
        
        Returns:
            Union[torch.Tensor, np.ndarray]: The converted NumPy array or PyTorch tensor.
    """
    if isinstance(inp, np.ndarray):
        return torch.from_numpy(inp).to(device) # TODO: add device selection
    elif isinstance(inp, torch.Tensor):
        if device == 'cpu':
            return inp.detach().numpy()
        else:
            return inp.detach().cpu().numpy()


def download_variable(var_filename, deriv_filename, time_axis):
    """
    Downloads variable data and its derivatives, then prepares a tensor of tokens.
    
    This function loads variable data and its corresponding derivatives from specified files,
    processes them, and combines them into a tensor suitable for equation discovery.
    The resulting tensor serves as input for the evolutionary process of identifying
    the underlying differential equation.
    
    Args:
        var_filename (str): The filename of the variable data.
        deriv_filename (str): The filename of the derivative data.
        time_axis (int): The axis representing the time dimension.
    
    Returns:
        np.ndarray: A tensor of tokens representing the processed variable and derivative data.
    """
    var = np.load(var_filename)
    derivs = np.load(deriv_filename)
    tokens_tensor = prepare_var_tensor(var, derivs, time_axis)
    return tokens_tensor


class Cache(object):
    """
    Class for efficiently storing and retrieving intermediate calculation results, enhancing performance by avoiding redundant computations.
    
    
    Args:
        max_allowed_tensors (`int`): limitation on the number of allowed tensors to load into the cache.
        memory_default (`dict`): key - name of tensor (tuple - (name_of_term, params)), value - derivative. Objects without changes after evolutional step
        memory_normalized (`dict`): key - name of tensor (tuple - (name_of_term, params)), value - derivative. Objects with normalize
        memory_structural (`dict`): key - name of tensor (tuple - (name_of_term, params)), value - derivative. NOT USED ДОПИСАТЬ ПРОСМОТРЕВ КОД
    """

    def __init__(self, device = 'cpu'):
        """
        Initializes the MemoryAnalyzer class.
        
                Sets up the memory analyzer to track and manage memory usage during the equation discovery process. This initialization prepares the data structures needed to store memory consumption data for different types of tensors, allowing for efficient memory management and optimization of the search for governing equations. By tracking memory usage, the system can avoid memory overflow and improve the efficiency of the equation discovery process.
        
                Args:
                  device (str): The device being used for computation (e.g., 'cpu', 'cuda').
        
                Fields:
                  _device (str): The device being used for computation.
                  max_allowed_tensors (None): Maximum number of tensors allowed (currently not set).
                  memory_default (dict): Stores default memory usage for torch and numpy tensors.
                  memory_normalized (dict): Stores normalized memory usage for torch and numpy tensors.
                  memory_structural (dict): Stores structural memory usage for torch and numpy tensors.
                  memory_anns (dict): Stores memory annotations.
                  mem_prop_set (bool): Flag indicating if memory properties have been set.
                  base_tensors (list): Stores non-normalized tensors.
                  structural_and_base_merged (dict): Stores merged structural and base tensor information.
                  _deriv_codes (list): Stores derivative codes and corresponding cache labels.
        
                Returns:
                  None
        """
        self._device = device
        self.max_allowed_tensors = None
        
        self.memory_default = {'torch' : dict(), 'numpy' : dict()} # TODO: add separate cache for torch tensors and numpy
        self.memory_normalized = {'torch' : dict(), 'numpy' : dict()}
        self.memory_structural = {'torch' : dict(), 'numpy' : dict()}
        self.memory_anns = dict()
        
        self.mem_prop_set = False 
        self.base_tensors = []  # storage of non-normalized tensors, that will not be affected by change of variables
        self.structural_and_base_merged = dict()
        self._deriv_codes = [] # Elements of this list must be tuples with the first element - 
                               # deriv code ([var, term]) like ([1], [0]) for dy/dt in LV, and the second - cache label in 
                               # standard form ('dy/dx1', (1.0,))

    def attrs_from_dict(self, attributes, except_attrs: dict = {}):
        """
        Populates the cache object's attributes from a dictionary, excluding specified keys to maintain a consistent and manageable cache state.
        
                This method initializes the object's attributes from the provided dictionary,
                optionally excluding keys specified in the `except_attrs` dictionary.
                If `except_attrs` contains keys with non-None values, those keys
                and values are explicitly set as attributes on the object after
                processing the main `attributes` dictionary. This ensures that certain
                attributes, if provided, always take precedence, allowing for fine-grained
                control over the cache's configuration.
        
                Args:
                    attributes: A dictionary containing attribute names and their values,
                        used to populate the cache object's state.
                    except_attrs: A dictionary of attributes to exclude from
                        the `attributes` dictionary during initial population.
                        If a key exists in this dictionary and its value is not None, it
                        will be explicitly set as an attribute after the initial population
                        step.
        
                Returns:
                    None.  The method modifies the object's attributes in place, updating
                    the cache object's internal state.
        """
        except_attrs['obj_type'] = None
        self.__dict__ = {key : item for key, item in attributes.items()
                         if key not in except_attrs.keys}
        for key, elem in except_attrs.items():
            if elem is not None:
                self.__dict__[key] = elem

    def use_structural(self, use_base_data=True, label=None, replacing_data=None):
        """
        Uses structural data, potentially replacing base data, to refine equation discovery.
        
                This method allows the user to specify whether to use the initial data as a starting point for structural exploration or to replace it entirely with new data. This is crucial for exploring different equation structures and identifying the best fit for the observed data.
        
                Args:
                    use_base_data: A boolean flag. If True, the method uses the initial data stored in the cache as a basis for structural data. If False, it uses the `replacing_data` instead.
                    label: An optional label. If None, the method applies the structural changes to all base data. If a tuple, it applies the changes only to the data associated with the given label.
                    replacing_data: Optional data to replace the base data. If `label` is None, this must be a dictionary where keys are memory keys (tuples) and values are NumPy arrays. If `label` is a tuple, this must be a NumPy array representing the replacement data for that specific label.
        
                Returns:
                    None
        """
        assert use_base_data or replacing_data is not None, 'Structural data must be declared with base data or by additional tensors.'
        # print('Called `use_structural`, expect caches to alter')
        if label is None:
            if use_base_data:
                self.memory_structural['numpy'] = {key: val for key, val in self.memory_default['numpy'].items()}
                try:
                    for key in self.memory_structural['numpy'].keys():
                        self.structural_and_base_merged[key] = True
                except AttributeError as e:
                    print(f"Error in class Cache {e}")
            else:
                if type(replacing_data) != dict:
                    raise TypeError('Replacing data shall be set with dict of format: tuple - memory key: np.ndarray ')
                if np.any([type(entry) != np.ndarray for entry in replacing_data.values()]):
                    raise TypeError('Replacing data shall be set with dict of format: tuple - memory key: np.ndarray ')
                if replacing_data.keys() != self.memory_default['numpy'].keys():
                    print(replacing_data.keys(), self.memory_default['numpy'].keys())
                    raise ValueError('Labels for the new structural data do not with the baseline data ones.')
                if np.any([entry.shape != self.memory_default['numpy'][label].shape for label, entry in replacing_data.items()]):
                    print([(entry.shape, self.memory_default['numpy'][label].shape) for label, entry in replacing_data.items()])
                    raise ValueError('Shapes of tensors in new structural data do not match their counterparts in the base data')
                for key in self.memory_default['numpy'].keys():
                    self.structural_and_base_merged[label] = False
                self.memory_structural = replacing_data
        elif isinstance(label, tuple):
            if use_base_data:
                # print(self.memory_default['numpy'].keys())
                replacing_data = self.memory_default['numpy'][label]
                self.structural_and_base_merged[label] = True
                if label not in self.memory_default['numpy'].keys():
                    self.add(label=label, tensor=replacing_data)
            else:
                if replacing_data is None:
                    raise ValueError('Got no replacing data, when expected!')
                self.structural_and_base_merged[label] = False
                if type(replacing_data) != np.ndarray:
                    raise TypeError('Replacing data with provided label shall be set with np.ndarray ')
                if label in self.memory_default['numpy'].keys():
                    if replacing_data.shape != self.memory_default['numpy'][label].shape:
                        raise ValueError('Shapes of tensors in new structural data do not match their counterparts in the base data')
                self.memory_structural['numpy'][label] = replacing_data

    @property
    def g_func(self):  # , g_func: Union[Callable, type(None)] = None
        try:
        """
        Calculates and returns the result of the g-function.
        
                This method dynamically selects and applies the g-function. If a grid cache
                is available, it computes the g-function based on the cached data,
                facilitating efficient equation discovery by reusing previously computed
                values. Otherwise, it returns the pre-computed g-function, ensuring
                that the equation discovery process can proceed even without cached data.
        
                Args:
                    self: The instance of the Cache class.
        
                Returns:
                    The result of the g-function, which can be either a NumPy array
                    or a list, depending on whether the grid cache is utilized.
        """
            assert '0' in self.memory_default['numpy'].keys()  # Check if we are working with the grid cache
            return self._g_func(self.get_all()[1])
        except TypeError:
            assert isinstance(self._g_func, (np.ndarray, list))
            return self._g_func

    @g_func.setter
    def g_func(self, function: Union[Callable, np.ndarray, list]):
        """
        Sets the function to be used for calculating the 'g' term.
        
        The 'g' term represents a component of the discovered differential equation.
        Setting this function allows the algorithm to explore different mathematical
        representations for this component during the equation discovery process.
        
        Args:
            function (Callable | np.ndarray | list): The function to be used for the 'g' term.
                It can be a callable (e.g., a Python function), a NumPy array representing
                discrete values, or a list of values.
        
        Returns:
            None
        """
        self._g_func = function

    def add_base_matrix(self, label):
        """
        Adds a label to the list of base tensors, ensuring it exists in the memory.
        
                This function is crucial for tracking the fundamental matrices used in constructing more complex equation terms. By maintaining a record of these base matrices, the system can efficiently manage and reuse them during the equation discovery process.
        
                Args:
                    label (str): The label of the base matrix to add. This label must correspond to a key in the 'numpy' section of the default memory.
        
                Returns:
                    None
        """
        assert label in self.memory_default['numpy'].keys()
        self.base_tensors.append(label)

    def set_boundaries(self, boundary_width: Union[int, list, tuple]):
        """
        Sets the boundary width, defining the region near the edges of the data that is excluded from equation discovery.
        
                This is crucial for mitigating boundary effects and ensuring the discovered equations accurately represent the underlying dynamics within the domain, rather than being influenced by artificial constraints at the edges.
        
                Args:
                    boundary_width (Union[int, list, tuple]): The width of the boundary region. If an integer is provided, the same width is applied to all dimensions. If a list or tuple is provided, it specifies the width for each dimension.
        
                Raises:
                    AssertionError: If the cache is not initialized for grid data.
                    IndexError: If the boundary width is too large for the data shape.
                    TypeError: If the boundary_width is not an int, list, or tuple.
        """
        assert '0' in self.memory_default['numpy'].keys(), 'Boundaries should be specified for grid cache.'
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
        Method for configuring memory usage for equation discovery.
        
                This method sets the available memory for caching tensors, which influences the complexity of equations that can be explored.
        
                Args:
                    obj_test_case (`ndarray`): A sample tensor used to estimate the memory footprint of potential equation terms.
                    mem_for_cache_frac (`int`): The fraction of total RAM to allocate for the cache (as a percentage). Defaults to None.
                    mem_for_cache_abs (`int`): The absolute amount of memory (in bytes) to allocate for the cache. Defaults to None.
        
                Returns:
                    None
        
                Why:
                    Configuring memory usage allows the algorithm to efficiently manage resources when searching for differential equations. By setting memory limits, we control the size and number of tensors that can be stored, which directly impacts the complexity of equations that can be discovered.
        """
        assert not (mem_for_cache_frac is None and mem_for_cache_abs is None), 'Avalable memory space not defined'
        assert obj_test_case is not None or len(self.memory_default['numpy']) > 0, 'Method needs sample of stored matrix  to evaluate memory allocation'
        if mem_for_cache_abs is None:
            self.available_mem = mem_for_cache_frac / 100. * psutil.virtual_memory().total  # Allocated memory for tensor storage, bytes
        else:
            self.available_mem = mem_for_cache_abs

        assert self.available_mem < psutil.virtual_memory().available

        if len(self.memory_default['numpy']) == 0:
            assert obj_test_case is not None
            self.max_allowed_tensors = int(np.floor(self.available_mem/obj_test_case.nbytes)/2)
        else:
            key = np.random.choice(list(self.memory_default['numpy'].keys()))
            self.max_allowed_tensors = int(np.floor(self.available_mem/
                                                    self.memory_default['numpy'][key].nbytes))

        eps = 1e-7
        if np.abs(self.available_mem) < eps:
            print('The memory can not containg any tensor even if it is entirely free (This message can not appear)')

    def clear(self, full=False):
        """
        Clears the stored derivative codes and cached tensors.
        
                This method is crucial for managing memory and ensuring that the equation discovery process remains efficient. By clearing the cache, we prevent the accumulation of unnecessary data, which can slow down the search for governing equations.
        
                Args:
                    self: The instance of the Cache class.
                    full (bool, optional): If True, clears all cached tensors, including base tensors. If False, only derivative codes are cleared, and base tensors are retained. Defaults to False.
        
                Returns:
                    None
        """
        self._deriv_codes = []
        print('Clearing cache')
        if full:
            del self.memory_default, self.memory_normalized, self.memory_structural, self.base_tensors
            self.memory_default = {'torch' : dict(), 'numpy' : dict()}
            self.memory_normalized = {'torch' : dict(), 'numpy' : dict()}
            self.memory_structural = {'torch' : dict(), 'numpy' : dict()}
            self.base_tensors = []
        else:
            new_memory_default = {'torch' : dict(), 'numpy' : dict()}
            new_memory_normalized = {'torch' : dict(), 'numpy' : dict()}
            new_memory_structural = {'torch' : dict(), 'numpy' : dict()}
            for key in self.base_tensors:
                new_memory_default['torch'][key] = self.get(key, False, False, None, True)
                new_memory_default['numpy'][key] = self.get(key, False, False, None, False)

                new_memory_normalized['torch'][key] = self.get(key, True, False, None, True)
                new_memory_normalized['numpy'][key] = self.get(key, True, False, None, False)

                new_memory_structural['torch'][key] = self.get(key, False, True, None, True)
                new_memory_structural['numpy'][key] = self.get(key, False, True, None, False)

            del self.memory_default, self.memory_normalized, self.memory_structural
            self.memory_default = new_memory_default
            self.memory_normalized = new_memory_normalized
            self.memory_structural = new_memory_structural

    def add(self, label, tensor, normalized: bool = False, structural: bool = False,
            deriv_code = None, indication: bool = False):
        """
        Adds a tensor to the cache, categorizing it as normalized or default based on the provided flags.
        
                This method attempts to store the given tensor in the appropriate memory cache (normalized or default) based on its properties and available space.
                The addition is skipped if the cache already contains tensor with provided label.
                The method ensures that the cache does not exceed its maximum allowed tensor limit.
        
                Args:
                    label (str): A unique identifier for the tensor.
                    tensor (torch.Tensor or numpy.ndarray): The tensor to be added to the cache.
                    normalized (bool, optional): Indicates whether the tensor is normalized. Defaults to False.
                    structural (bool, optional): Indicates whether the tensor is structural. Defaults to False.
                    deriv_code (object, optional): Derivative code associated with the tensor. Defaults to None.
                    indication (bool, optional): Enables print statements for debugging. Defaults to False.
        
                Returns:
                    bool: True if the tensor was successfully added to the cache, False otherwise.
        """
        # print(deriv_code)
        if deriv_code is not None:
            self._deriv_codes.append((deriv_code, label))
        assert not (normalized and structural), 'The added matrix can not be simultaneously normalized and structural. Possibly, bug in token/term saving'
        type_key = 'torch' if isinstance(tensor, torch.Tensor) else 'numpy'
        if normalized:
            if self.max_allowed_tensors is None:
                self.memory_usage_properties(obj_test_case=tensor, mem_for_cache_frac=5)
            if ((len(self.memory_normalized[type_key]) + len(self.memory_default[type_key]) +
                 len(self.memory_structural[type_key])) < self.max_allowed_tensors and
                label not in self.memory_normalized[type_key].keys()):
                self.memory_normalized[type_key][label] = tensor
                if indication:
                    print('Enough space for saved normalized term ', label, tensor.nbytes)
                return True
            elif label in self.memory_normalized[type_key].keys():
                if indication:
                    print('The term already present in normalized cache, no addition required', label, tensor.nbytes)
                return True
            else:
                if indication:
                    print('Not enough space for term ', label, tensor.nbytes, 'Can save only', self.max_allowed_tensors, 
                          'tensors. While already uploaded ', len(self.memory_normalized) + len(self.memory_default) + len(self.memory_structural))
                return False
        elif structural:
            raise NotImplementedError('The structural data must be added with cache.use_structural method')
        else:
            if self.max_allowed_tensors is None:
                self.memory_usage_properties(obj_test_case=tensor, mem_for_cache_frac=5)
            if ((len(self.memory_normalized[type_key]) + len(self.memory_default[type_key]) +
                 len(self.memory_structural[type_key])) < self.max_allowed_tensors and
                label not in self.memory_default[type_key].keys()):
                self.memory_default[type_key][label] = tensor
                if indication:
                    print('Enough space for saved unnormalized term ', label, tensor.nbytes)
                return True
            elif label in self.memory_default[type_key].keys():
                if indication:
                    print('The term already present in unnormalized cache, no addition required', label, tensor.nbytes)
                return True
            else:
                if indication:
                    print('Not enough space for term ', label, tensor.nbytes)
                return False

    def delete_entry(self, entry_label):
        """
        Deletes a specific data entry from all cache levels (default, structural, and normalized).
        
                This ensures data consistency across different representations used during the equation discovery process. By removing the entry from all levels, we prevent outdated or inconsistent data from influencing the search for governing differential equations.
        
                Args:
                    entry_label (str): The label of the data entry to delete.
        
                Returns:
                    None
        
                Raises:
                    ValueError: If the specified `entry_label` is not found in the default memory. This indicates a potential issue with data management or a request to delete a non-existent entry.
        """
        print(f'Deleting {entry_label} from cache!')
        if entry_label not in self.memory_default["numpy"].keys():
            raise ValueError('deleted element already not in memory')
        del self.memory_default["numpy"][entry_label]
        try:
            del self.memory_structural["numpy"][entry_label]
        except KeyError:
            pass
        try:
            del self.memory_normalized["numpy"][entry_label]
        except KeyError:
            pass

    def get(self, label, normalized=False, structural=False, saved_as=None, torch_mode: bool = False, deriv_code = None):
        """
        Retrieves a matrix from the cache based on the provided label and flags.
        
                This method is central to accessing and managing the discovered equation components.
                It allows retrieval of matrices in different formats (numpy or torch), and with different scaling
                (normalized or structural), ensuring that the correct representation is used in the equation discovery process.
                The method also handles on-the-fly format conversion if the matrix is available in a different format.
        
                Args:
                    label: The label of the matrix to retrieve.
                    normalized: A boolean indicating whether to retrieve the normalized version of the matrix.
                    structural: A boolean indicating whether to retrieve the structurally scaled version of the matrix.
                    saved_as: The format the matrix was saved as.
                    torch_mode: A boolean indicating whether to return a torch tensor or a numpy array.
                    deriv_code: Derivative code to look up the label.
        
                Returns:
                    The retrieved matrix, either as a numpy array or a torch tensor.
        """
        assert not (normalized and structural), 'The added matrix can not be simultaneously normalized and scaled'
        type_key, other, other_bool = ('torch', 'numpy', False) if torch_mode else ('numpy', 'torch', True)
        if deriv_code is not None:
            label = [elem[1] for elem in self._deriv_codes if elem[0] == deriv_code][0]

        if label is None:
            # print(self.memory_default[type_key].keys())
            return np.random.choice(list(self.memory_default[type_key].values()))
        if normalized:
            if label not in self.memory_normalized[type_key] and label in self.memory_normalized[other]:
                self.memory_normalized[type_key][label] = switch_format(self.get(label, normalized,
                                                                                  structural, saved_as, other_bool), 
                                                                        device = self._device)
            return self.memory_normalized[type_key][label]
        elif structural:
            if self.structural_and_base_merged[label]:
                return self.get(label, normalized, False, saved_as, torch_mode)
            else:
                if label not in self.memory_structural[type_key] and label in self.memory_structural[other]:
                    self.memory_structural[type_key][label] = switch_format(self.get(label, normalized, 
                                                                                     structural, saved_as, other_bool),
                                                                            device = self._device)
                # print('keys in mem_struct:', type_key, self.memory_structural[type_key].keys())
                return self.memory_structural[type_key][label]                
        else:
            if label not in self.memory_default[type_key] and label in self.memory_default[other]:
                self.memory_default[type_key][label] = switch_format(self.get(label, normalized, 
                                                                              structural, saved_as, other_bool),
                                                                     device = self._device)
            return self.memory_default[type_key][label]

    def get_all(self, normalized=False, structural=False, mode: str = 'numpy'):
        """
        Returns all stored keys and their corresponding tensors.
        
                This method retrieves data from both primary and secondary memory formats (NumPy and Torch),
                ensuring all available data is included regardless of the current primary format. It also
                handles necessary format conversions to maintain consistency.
        
                Args:
                    normalized (bool, optional): If True, retrieve data from the normalized memory. Defaults to False.
                    structural (bool, optional): If True, retrieve data from the structural memory. Defaults to False.
                    mode (str, optional): The primary memory format ('numpy' or 'torch'). Defaults to 'numpy'.
        
                Returns:
                    tuple[list, list]: A tuple containing:
                        - A list of keys representing the identified equation terms.
                        - A list of tensors representing the corresponding data for each term,
                          ensuring data from both memory formats are included and converted to
                          the appropriate format.
        
                Why:
                    This method is essential for accessing all discovered equation terms and their
                    corresponding data, regardless of the storage format. It ensures that the
                    evolutionary algorithm has a complete view of the candidate equations and
                    their performance, facilitating the equation discovery process.
        """
        other = 'torch' if mode == 'numpy' else 'numpy'
        if normalized:
            processed_mem = self.memory_normalized[mode]
            other_mem = self.memory_normalized[other]
        elif structural:
            processed_mem = self.memory_structural[mode]
            other_mem = self.memory_structural[other]
        else:
            processed_mem = self.memory_default[mode]
            other_mem = self.memory_default[other]

        keys = []
        tensors = []
        for key, value in processed_mem.items():
            keys.append(key)
            tensors.append(value)

        for key, value in other_mem.items():
            if key not in processed_mem.keys():
                keys.append(key)
                tensors.append(switch_format(value, device = self._device))

        return keys, tensors

    def __contains__(self, obj):
        """
        Checks if a given object is present in the cache, considering both normalized and unnormalized data. This is essential for efficiently reusing previously computed results when exploring the solution space of differential equations.
        
                Args:
                    obj (str, tuple, list, np.ndarray, torch.Tensor): The object to check for presence in the cache.
                        It can be a label (string), a tuple/list containing a label and a boolean indicating normalization,
                        a NumPy array, a PyTorch tensor, or a tuple/list containing a NumPy array/PyTorch tensor and a boolean
                        indicating normalization.
        
                Returns:
                    bool: True if the object is found in the cache, False otherwise.
        
                Raises:
                    NotImplementedError: If the input object's format is not supported.
        """
        if (type(obj) == tuple or type(obj) == list) and type(obj[0]) == str:
            return (obj in self.memory_default['numpy'].keys()) or (obj in self.memory_default['torch'].keys()) 
        elif (type(obj) == tuple or type(obj) == list) and type(obj[0]) == tuple and type(obj[1]) == bool:
            if obj[1]:
                return (obj[0] in self.memory_normalized['numpy'].keys()) or (obj[0] in self.memory_normalized['torch'].keys())
            else:
                return (obj[0] in self.memory_default['numpy'].keys()) or (obj[0] in self.memory_default['torch'].keys())
        elif type(obj) == np.ndarray:
            try:
                return np.any([np.all(obj == entry_values) for entry_values in self.memory_default['numpy'].values()])
            except:
                return False
        elif type(obj) == torch.Tensor:
            try:
                return np.any([np.all(obj == entry_values) for entry_values in self.memory_default['torch'].values()])
            except:
                return False            
        elif (type(obj) == tuple or type(obj) == list) and type(obj[0]) == np.ndarray and type(obj[1]) == bool:
            try:
                if obj[1]:
                    return np.any([np.all(obj[0] == entry_values) for entry_values in self.memory_normalized['numpy'].values()])
                else:
                    return np.any([np.all(obj[0] == entry_values) for entry_values in self.memory_default['numpy'].values()])
            except:
                return False
        elif (type(obj) == tuple or type(obj) == list) and type(obj[0]) == torch.Tensor and type(obj[1]) == bool:
            try:
                if obj[1]:
                    return np.any([np.all(obj[0] == entry_values) for entry_values in self.memory_normalized['torch'].values()])
                else:
                    return np.any([np.all(obj[0] == entry_values) for entry_values in self.memory_default['torch'].values()])
            except:
                return False            
        else:
            raise NotImplementedError('Invalid format of function input to check, if the object is in cache')

#    def __iter__(self):
#        for key in self.memory_default.keys()
    def prune_tensors(self, pruner, mem_to_process: list = ['default', 'structural', 'normalized'], 
                      torch_mode: bool = False):
        """
        Prunes stored tensors to reduce memory footprint based on the provided pruner.
        
                This method iterates through specified memory types (default, structural, normalized)
                and applies the provided pruner to each tensor associated with a key. This helps in
                reducing the computational cost by removing unnecessary information, thereby optimizing
                the search for governing differential equations.
        
                Args:
                    pruner: The pruner object used to prune the tensors.
                    mem_to_process: A list of memory types to process (default, structural, normalized).
                    torch_mode: A boolean indicating whether to use torch mode or numpy mode.
        
                Returns:
                    None. The method modifies the memory tensors in place.
        """
        mode = 'torch' if torch_mode else 'numpy'
        mem_arranged = {'default': self.memory_default,
                        'structural': self.memory_structural,
                        'normalized': self.memory_normalized}

        for key in self.memory_default[mode].keys():
            for mem_type in mem_to_process:
                try:
                    mem_arranged[mode][mem_type][key] = pruner.prune(mem_arranged[mode][mem_type][key])
                except (NameError, KeyError) as e:
                    pass

    @property
    def consumed_memory(self):
        """
        Return the total memory consumed by NumPy arrays across different memory scopes.
        
                This property calculates the total memory footprint of NumPy arrays
                stored within the cache. It aggregates the memory usage from the
                'memory_default', 'memory_normalized', and 'memory_structural'
                dictionaries, focusing specifically on 'numpy' entries. This is
                crucial for monitoring memory usage during the equation discovery
                process, especially when dealing with large datasets or complex
                equation structures. By tracking memory consumption, we can optimize
                the performance and prevent memory-related issues during the search
                for governing differential equations.
        
                Args:
                    self: The Cache instance.
        
                Returns:
                    int: The total memory consumed by NumPy arrays, in bytes.
        """
        memsize = np.sum([value.nbytes for _, value in self.memory_default['numpy'].items()])
        memsize += np.sum([value.nbytes for _, value in self.memory_normalized['numpy'].items()])
        for label, merged_state in self.structural_and_base_merged.items():
            if not merged_state: memsize += self.memory_structural['numpy'][label].nbytes
        return memsize


def upload_complex_token(label: str, params_values: OrderedDict, evaluator, tensor_cache: Cache, grid_cache: Cache):
    """
    Uploads a complex token's evaluated tensor to the cache.
    
        This method constructs a tensor by evaluating a predefined function
        (identified by the label) over a grid of input values. These input
        grids are retrieved from the grid cache, and the resulting tensor
        is stored in the tensor cache, enabling later use for equation discovery.
    
        Args:
            label (str): The label identifying the evaluation function to use for tensor generation.
            params_values (OrderedDict): An ordered dictionary of parameter values to be
                passed to the evaluation function.
            evaluator: An object containing the evaluation functions.
            tensor_cache (Cache): The cache for storing the resulting tensors.
            grid_cache (Cache): The cache containing the grids of input values.
    
        Returns:
            None: This method does not return any value. The generated tensor is stored in `tensor_cache`.
    
        WHY: This method is crucial for constructing the building blocks of potential differential equations. By evaluating predefined functions over a grid of values, it generates tensors that represent different terms or operations within an equation. These tensors are then used in the equation discovery process to find the best-fitting model for the given data.
    """
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

# class EquationsCache(object):
#     '''
#     Cache to keep the information about already discovered equations. Getting equation objectives values will reduce the unnecessary
#     computations, that may occur if the EPDE repeatedly generates the same equation.
#     '''
#     def __init__(self):
#         self._saved_equations = set()

#     @staticmethod
#     def parse_input(self, equation):
#         return 