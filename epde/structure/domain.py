import numpy as np

from typing import Callable, List, Union
from functools import singledispatchmethod

from epde.cache.cache import Cache
from epde.decorators import BoundaryExclusion
from epde.preprocessing.domain_pruning import DomainPruner


class Subdomain(object):
    def __init__(self, grid: Union[np.ndarray, List[np.ndarray]]):
        self._g_func = None
        self._grid_cache = None
        self._token_cache = None
        if isinstance(grid, np.ndarray): # or (isinstance(grid, (list, tuple)) and len(grid) == 1):
            self._dim = 1
            self._grids = [grid,]
        elif isinstance(grid, (tuple, list)):
            self._dim = len(grid)
            self._grids = grid

    def upload_g_func(self, g_func: Union[Callable, np.ndarray, list] = None):
        def baseline_exp_function(grids):
            def uniformize(data):
                temp = -(data - np.mean(data))**2
                if np.min(temp) == np.max(temp):
                    return np.ones_like(temp)
                else:
                    return (temp - np.min(temp)) / (np.max(temp) - np.min(temp))

            exponent_partial = np.array([uniformize(grid) for grid in grids])
            exponent = np.multiply.reduce(exponent_partial, axis=0)
            return exponent

        if isinstance(g_func, (np.ndarray, list)):
            self._g_func = g_func
        elif isinstance(g_func, Callable):
                decorator = BoundaryExclusion(boundary_width=self.boundary_width)
                if isinstance(g_func, (Callable, np.ndarray, list)):
                    self._g_func = decorator(baseline_exp_function)
                else:
                    self._g_func = decorator(g_func)

    @property
    def pruned(self):
        return self._pruner.prune(self._grid)

    def set_boundaries(self, boundary_width: Union[int, list, tuple]):
        """
        Setting the number of unaccounted elements at the edges
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

    def set_cache(self, cache: Cache, token_cache: bool):
        if token_cache:
            self._token_cache = cache
            if self._g_func is not None:
                self._token_cache.g_func = self._g_func
        else:
            self._grid_cache = cache
            if self._g_func is not None:
                self._grid_cache.g_func = self._g_func


class Domain(object): # inheritance from cache objects?
    def __init__(self, subdomains: List[Subdomain]):   # grids: List[Union[Union[List[np.ndarray], np.ndarray]]]):
        self.subdomains = subdomains

    def set_pruner(self, pivotal_tensor: np.ndarray = None, pruner: DomainPruner = None,
                   threshold : float = 1e-5, division_fractions = 3,
                   rectangular : bool = True):
        """
        Method for select only subdomains with variable dynamics.

        Args:
            pivotal_tensor_label (`np.ndarray`): 
                Pattern that guides the domain pruning will be cutting areas, where values of the 
                `pivotal_tensor` are closed to zero.
            pruner (`DomainPruner`): 
                Custom object for selecting domain region by pruning out areas with no dynamics.
            threshold (`float`): optional, default - 1e-5
                The boundary at which values are considered zero.
            division_fractions (`int`): optional, default - 3
                Number of fraction for each axis (if this is integer than all axis are dividing by same fractions).
            rectangular (`bool`): default - True
                Flag indecating that area is rectangle.
                
        Returns:
            None
        """
        
        for subdomain in self.subdomains:
            subdomain.set_pruner(pivotal_tensor, pruner, threshold, division_fractions, rectangular)

    def set_boundaries(self, boundary_width: Union[int, list]):
        """
        Setting the number of unaccounted elements at the edges
        """

        for subdomain in self.subdomains:
            subdomain.set_boundaries(boundary_width)

    @singledispatchmethod
    def set_cache(self, cache: Union[Cache, List[Cache]], token_cache: bool = True):
        pass
    
    @set_cache.register
    def _(self, cache: Cache, token_cache: bool = True):
        for subdomain in self.subdomains:
            subdomain.set_cache(cache, token_cache)

    @set_cache.register
    def _(self, cache: List[Cache], token_cache: bool = True):
        assert len(self.subdomains) == len(cache), 'Mismatching lengths of subdomains and caches'
        for domain_idx, subcache in enumerate(cache):
            self.subdomains[domain_idx].set_cache(subcache, token_cache)