import numpy as np
import torch
from collections import OrderedDict

from typing import Callable, Tuple, List, Union, Literal
from functools import singledispatchmethod

from epde.cache.cache import upload_grids

from epde.cache.cache import Cache, prepare_var_tensor, upload_simple_tokens
from epde.decorators import BoundaryExclusion
from epde.preprocessing.domain_pruning import DomainPruner

from epde.supplementary import define_derivatives
from epde.evaluators import simple_function_evaluator # , trigonometric_evaluator

from epde.interface.prepared_tokens import DataPolynomials
from epde.interface.type_checks import *
from epde.interface.token_family import TFPool, TokenFamily

from epde.preprocessing.preprocessor import ConcretePrepBuilder, PreprocessingPipe


RELATIVE_BC_LOC_DIR = {0: (), 1: (0.,), 2: (0., 1.),
                       3: (0., 0., 1.), 4: (0., 0., 1., 1.)}

# class Subdomain(object):
#     def __init__(self, grid: Union[np.ndarray, List[np.ndarray]]):
#         self._g_func = None
#         self._grid_cache = None
#         self._token_cache = None
#         if isinstance(grid, np.ndarray): # or (isinstance(grid, (list, tuple)) and len(grid) == 1):
#             self._dim = 1
#             self._grids = [grid,]
#         elif isinstance(grid, (tuple, list)):
#             self._dim = len(grid)
#             self._grids = grid

    # def upload_g_func(self, g_func: Union[Callable, np.ndarray, list] = None):
    #     def baseline_exp_function(grids):
    #         def uniformize(data):
    #             temp = -(data - np.mean(data))**2
    #             if np.min(temp) == np.max(temp):
    #                 return np.ones_like(temp)
    #             else:
    #                 return (temp - np.min(temp)) / (np.max(temp) - np.min(temp))

    #         exponent_partial = np.array([uniformize(grid) for grid in grids])
    #         exponent = np.multiply.reduce(exponent_partial, axis=0)
    #         return exponent

    #     if isinstance(g_func, (np.ndarray, list)):
    #         self._g_func = g_func
    #     elif isinstance(g_func, Callable):
    #             decorator = BoundaryExclusion(boundary_width=self.boundary_width)
    #             if isinstance(g_func, (Callable, np.ndarray, list)):
    #                 self._g_func = decorator(baseline_exp_function)
    #             else:
    #                 self._g_func = decorator(g_func)

    # @property
    # def pruned(self):
    #     return self._pruner.prune(self._grid)

    # def set_boundaries(self, boundary_width: Union[int, list, tuple]):
    #     """
    #     Setting the number of unaccounted elements at the edges
    #     """
        
    #     assert '0' in self.memory_default['numpy'].keys(), 'Boundaries should be specified for grid cache.'
    #     shape = self.get('0')[1].shape
    #     if isinstance(boundary_width, int):
    #         if any([elem <= 2*boundary_width for elem in shape]):
    #             raise IndexError(f'Mismatching shapes: boundary of {boundary_width} does not fit data of shape {shape}')
    #     elif isinstance(boundary_width, (list, tuple)):
    #         if any([elem <= 2*boundary_width[idx] for idx, elem in enumerate(shape)]):
    #             raise IndexError(f'Mismatching shapes: boundary of {boundary_width} does not fit data of shape {shape}')                
    #     else:
    #         raise TypeError(f'Incorrect type of boundaries: {type(boundary_width)}, instead of expected int or list/tuple')

    #     self.boundary_width = boundary_width

    # def getBCLocation(self, bc_info):
    #     return ...

    # def __call__(self, function: Callable, mode: Literal['numpy', 'torch'] = 'numpy') -> Union[torch.Tensor, np.ndarray]:
    #     # Type checks on function signature?
    #     return function(self._grid_cache)

    # def set_cache(self, cache: Cache, token_cache: bool):
    #     if token_cache:
    #         self._token_cache = cache
    #         if self._g_func is not None:
    #             self._token_cache.g_func = self._g_func
    #     else:
    #         self._grid_cache = cache
    #         if self._g_func is not None:
    #             self._grid_cache.g_func = self._g_func

class InputDataEntry(object):
    """
    Class for keeping input data

    Attributes:
        var_name (`str`): name of input data dependent variable
        data_tensor (`np.ndarray`): value of the input data
        names (`list`): keys for derivatides
        d_orders (`list`): keys for derivatides on `int` format for `solver`
        derivatives (`np.ndarray`): values of derivatives
        deriv_properties (`dict`): settings of derivatives
    """
    def __init__(self, var_name: str, var_idx: int, data_tensor: Union[List[np.ndarray], np.ndarray], boundary):
        self.var_name = var_name
        self.var_idx = var_idx 
        if isinstance(data_tensor, np.ndarray):
            check_nparray(data_tensor) 
            self.ndim = data_tensor.ndim
        elif isinstance(data_tensor, list):
            [check_nparray(tensor) for tensor in data_tensor]
            assert all([data_tensor[0].ndim == tensor.ndim for tensor in data_tensor]), 'Mismatching dimensionalities of data tensors.'
            self.ndim = data_tensor[0].ndim
        self.data_tensor = data_tensor
        self.boundary = boundary


    def setDerivatives(self, preprocesser: PreprocessingPipe, deriv_tensors: Union[list, np.ndarray] = None,
                        max_order: Union[list, tuple, int] = 1, grid: list = []):
        """
        Method for setting derivatives ot calculate derivatives from data

        Args:
            preprocesser (`PreprocessingPipe`): operator for preprocessing data (smooting and calculating derivatives)
            deriv_tensor (`np.ndarray`): values of derivatives
            max_order (`list`|`tuple`|`int`): order for derivatives
            grid: value of grid

        Returns:
            None
        """
        deriv_names, deriv_orders = define_derivatives(self.var_name, dimensionality=self.ndim,
                                                       max_order=max_order)

        self.names = deriv_names
        self.d_orders = deriv_orders

        if deriv_tensors is None and isinstance(self.data_tensor, np.ndarray):
            self.data_tensor, self.derivatives = preprocesser.run(self.data_tensor, grid=grid,
                                                                  max_order=max_order)
            self.deriv_properties = {'max order': max_order,
                                     'dimensionality': self.data_tensor.ndim}
        elif deriv_tensors is None and isinstance(self.data_tensor, list):
            if isinstance(grid[0], np.ndarray):
                raise ValueError('A single set of grids passed for multiple samples mode.')
            data_tensors, derivatives = [], []
            for samp_idx, sample in enumerate(self.data_tensor):
                processed_data, derivs = preprocesser.run(sample, grid=grid[samp_idx],
                                                          max_order=max_order)
                data_tensors.append(processed_data)
                derivatives.append(derivs)
            self.data_tensor = np.concatenate(data_tensors, axis = 0) # TODO: stack data_tensors with the time axis in the correct wa
            self.derivatives = np.concatenate(derivatives, axis=0) # TODO: check the correct
            self.deriv_properties = {'max order': max_order,
                                     'dimensionality': self.data_tensor.ndim}

        elif deriv_tensors is not None and isinstance(self.data_tensor, list):
            self.data_tensor = np.concatenate(self.data_tensor, axis = 0)

            print(f'Concatenating arrays of len {len(deriv_tensors)}')
            self.derivatives = np.concatenate(deriv_tensors, axis = 0)
            self.deriv_properties = {'max order': max_order,
                                     'dimensionality': self.data_tensor.ndim}            
        else:
            self.derivatives = deriv_tensors
            self.deriv_properties = {'max order': max_order,
                                     'dimensionality': self.data_tensor.ndim}

    # def use_global_cache(self): # , var_idx: int, deriv_codes: list
    #     """
    #     Method for add calculated derivatives in the cache
    #     """
    #     var_idx = self.var_idx
    #     deriv_codes = self.d_orders
    #     self.data_tensor = self.data_tensor[self.boundary != 0]
    #     self.derivatives = np.array([derivative[self.boundary.flatten() != 0] for derivative in self.derivatives.T]).T
    #     derivs_stacked = prepare_var_tensor(self.data_tensor, self.derivatives, 
    #                                         time_axis=global_var.time_axis)
    #     deriv_codes = [(var_idx, code) for code in deriv_codes]

    #     try:
    #         upload_simple_tokens(self.names, global_var.tensor_cache, derivs_stacked, 
    #                              deriv_codes=deriv_codes)
    #         upload_simple_tokens([self.var_name,], global_var.tensor_cache, [self.data_tensor,],
    #                              deriv_codes=[(var_idx, [None,]),])
    #         upload_simple_tokens([self.var_name,], global_var.initial_data_cache, [self.data_tensor,])

    #     except AttributeError:
    #         raise NameError('Cache has not been declared before tensor addition.')
    #     print(f'Size of linked labels is {len(global_var.tensor_cache._deriv_codes)}')
    #     global_var.tensor_cache.use_structural()

    @staticmethod
    def latex_form(label, **params):
        '''
        Parameters
        ----------
        label : str
            label of the token, for which we construct the latex form.
        **params : dict
            dictionary with parameter labels as keys and tuple of parameter values 
            and their output text forms as values.

        Returns
        -------
        form : str
            LaTeX-styled text form of token.
        '''            
        if '/' in label:
            label = label[:label.find('x')+1] + '_' + label[label.find('x')+1:]
            label = label.replace('d', r'\partial ').replace('/', r'}{')
            label = r'\frac{' + label + r'}'
                            
        if params['power'][0] > 1:
            label = r'\left(' + label + r'\right)^{{{0}}}'.format(params["power"][1])
        return label

    def create_derivs_family(self, max_deriv_power: int = 1):
        self._derivs_family = TokenFamily(token_type=f'deriv of {self.var_name}', variable = self.var_name, 
                                          family_of_derivs=True)
        
        self._derivs_family.set_latex_form_constructor(self.latex_form)
        self._derivs_family.set_status(demands_equation=True, unique_specific_token=False,
                                       unique_token_type=False, s_and_d_merged=False,
                                       meaningful=True)
        self._derivs_family.set_params(self.names, OrderedDict([('power', (1, max_deriv_power))]),
                                      {'power': 0}, self.d_orders)
        self._derivs_family.set_evaluator(simple_function_evaluator)

    def create_polynomial_family(self, max_power):
        polynomials = DataPolynomials(self.var_name, max_power = max_power)
        self._polynomial_family = polynomials.token_family

    def get_families(self):
        return [self._polynomial_family, self._derivs_family]

    def matched_derivs(self, max_order: int = 1, time_axis: int = 0):
        derivs_stacked = prepare_var_tensor(self.data_tensor, self.derivatives, 
                                            time_axis = time_axis)
        # print(f'Creating matched derivs: {[[self.var_idx, key, len(key) <= max_order] for idx, 
        #                                    key in enumerate(self.d_orders)]}')
        # print(f'From {self.d_orders}')
        return [[self.var_idx, key, derivs_stacked[idx, ...]] for idx, key in enumerate(self.d_orders)
                if len(key) <= max_order]


class Domain(object): # inheritance from cache objects?
    def __init__(self, 
                 grids: Union[np.ndarray, Tuple[np.ndarray], List[np.ndarray]], 
                 grid_cache: Cache,
                 time_axis: int = 0):
        self._time_axis = time_axis
        
        self._g_func = None
        self._g_func_flat_cache = None
        self._g_func_mask_cache = None        
        
        self._grid_cache = None
        self._tensor_cache = None

        self._pruner = None

        if isinstance(grids, np.ndarray): # or (isinstance(grid, (list, tuple)) and len(grid) == 1):
            self._dim = 1
            grids = [grids,]
        elif isinstance(grids, (tuple, list)):
            self._dim = len(grids)
        
        self._grid_cache = grid_cache
        upload_grids(grids, grid_cache)

    def set_pruner(self, pivotal_tensor: np.ndarray, pruner: DomainPruner = None,
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

        if pruner is not None:
            self.pruner = pruner
        else:
            self.pruner = DomainPruner(domain_selector_kwargs={'threshold': threshold})  

        if not self.pruner.bds_init:
            self.pruner.get_boundaries(pivotal_tensor, division_fractions=division_fractions,
                                       rectangular=rectangular)


    def setBoundaries(self, boundary_width: Union[int, List[int]]) -> None:
        """
        Setting the number of unaccounted elements at the edges
        """
        if isinstance(boundary_width, int):
            self._boundary_width = [boundary_width,] * self._dim
        elif isinstance(boundary_width, (list, tuple)):
            assert all([isinstance(width, int) for width in boundary_width]), 'Boundary width has to be an int.'
            self._boundary_width = boundary_width

    def prune(self) -> None:
        raise NotImplementedError('Pruning is not implemented yet.')

    @staticmethod
    def getBCIdxs(tensor_shape, axis, rel_loc) -> Tuple[np.ndarray]:
        return tuple(np.meshgrid(*[np.arange(shape) if dim_idx != axis else min(int(rel_loc * shape), shape-1)
                                   for dim_idx, shape in enumerate(tensor_shape)], indexing='ij'))

    def getGrids(self, mode: Literal['full', 'solver'] = 'full') -> List[np.ndarray]:
        _, grids = self._grid_cache.get_all()

        if mode == 'solver':
            for grid_idx in range(len(grids)):
                for dim_idx, bnd in enumerate(self._boundary_width):
                    grids[grid_idx] = np.take(a       = grids[grid_idx], 
                                              indices = np.r_[bnd : grids[grid_idx].shape[dim_idx]-bnd],
                                              axis    = dim_idx)
        return grids

    def getCachedTensor(self, label: tuple = None, subcache_ID: int = None,
                        normalized: bool = False, deriv_code = None) -> np.ndarray:
        if self._tensor_cache is None:
            raise RuntimeError("Trying to call tensors, linked to a domain, before specializing the cache for the domain.")
        return self._tensor_cache.get(label, subcache_ID, normalized, deriv_code)

    @property
    def g_func(self) -> np.ndarray:
        try:
            return self._g_func(self.getGrids(mode = 'full'))
        except TypeError:
            assert isinstance(self._g_func, (np.ndarray, list))
            return self._g_func

    @g_func.setter
    def g_func(self, function: Union[Callable, np.ndarray, list]) -> None:
        self._g_func = function
        self._g_func_flat_cache = None
        self._g_func_mask_cache = None

    @property
    def g_func_flat(self) -> np.ndarray:
        if self._g_func_flat_cache is None:
            self._g_func_flat_cache = self.g_func.reshape(-1)
        return self._g_func_flat_cache

    @property
    def g_func_mask(self) -> np.ndarray:
        if self._g_func_mask_cache is None:
            self._g_func_mask_cache = self.g_func != 0
        return self._g_func_mask_cache 

    def getBCLocation(self, bc_info: dict) -> Tuple[Tuple[np.ndarray], List[np.ndarray]]:
        '''
        Method to get bc locations as indexes of numpy ndarrays, or torch Tensors.

        Args:
            bc_info (dict):
                Information about the boundary condition, which location is to be retrieved.
                Example of such info dict: {'axis': 0, 'order': 1, 'var': 0, 'loc': 0}, which denotes that the equation requires 
                0-th (i.e. original function) of the first (0-th) variable along the first (0-th) axis.

        Returns:
            (indexes, coords) (Tuple[List[np.ndarray]]):
                tuple: first element - indexes of the points, that are employed in the BC formulation, 
                second - coordinates of these BC evaluation points.  
        '''

        grids = self.getGrids()
        bc_idxs = self.getBCIdxs(grids[0].shape, bc_info['axis'], bc_info['loc'])

        bc_locs = []
        for grid in grids:
            bc_locs.append(grid[bc_idxs])

        return (bc_idxs, bc_locs)

    def addEntry(self, input_entry: InputDataEntry) -> None:
        deriv_codes = [(input_entry.var_idx, code) for code in input_entry.d_orders]

        derivatives = np.array([derivative[self.g_func_mask] for derivative in input_entry.derivatives.T]).T
        derivs_stacked = prepare_var_tensor(input_entry.data_tensor, input_entry.derivatives, 
                                            time_axis = self._time_axis)
        
        try:
            upload_simple_tokens(input_entry.names, self._tensor_cache, derivs_stacked, 
                                 deriv_codes=deriv_codes)
            upload_simple_tokens([input_entry.var_name,], self._tensor_cache, [input_entry.data_tensor,],
                                 deriv_codes=[(input_entry.var_idx, [None,]),])
            upload_simple_tokens([input_entry.var_name,], self._tensor_cache, [input_entry.data_tensor,])

        except AttributeError:
            raise NameError('Cache has not been declared before tensor addition.')        
        


class Sample(object):
    def __init__(self, domain: Domain): #  tensor_cache: Cache,
        # self._tensors = tensor_cache
        self._domain  = domain

    def setSample(self, entry: InputDataEntry, preprocessor_pipeline: PreprocessingPipe = None,
                  derivs: np.ndarray = None, max_deriv_order: Union[int, List[int], Tuple[int]] = 1,
                  data_fun_pow: int = 1, deriv_fun_pow: int = 1) -> tuple:
        '''
        Returns tuple: first element is a list  matched labled for derivatives
        '''
        entry.setDerivatives(preprocesser = preprocessor_pipeline, deriv_tensors = derivs,
                             grid = self._domain.getGrids(), max_order = max_deriv_order)
        
        self._domain.addEntry(entry)

        entry.create_derivs_family(max_deriv_power=deriv_fun_pow)
        entry.create_polynomial_family(max_power=data_fun_pow)

        return entry.get_families(), entry.matched_derivs(max_order = 2)

    

        # pass # Add logic to create



    # @singledispatchmethod
    # def set_cache(self, cache: Union[Cache, List[Cache]], token_cache: bool = True):
    #     pass
    
    # @set_cache.register
    # def _(self, cache: Cache, token_cache: bool = True):
    #     for subdomain in self.subdomains:
    #         subdomain.set_cache(cache, token_cache)

    # @set_cache.register
    # def _(self, cache: List[Cache], token_cache: bool = True):
    #     assert len(self.subdomains) == len(cache), 'Mismatching lengths of subdomains and caches'
    #     for domain_idx, subcache in enumerate(cache):
    #         self.subdomains[domain_idx].set_cache(subcache, token_cache)
