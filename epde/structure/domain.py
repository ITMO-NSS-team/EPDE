import numpy as np
import torch
from collections import OrderedDict

from typing import Callable, Tuple, List, Union, Literal, Dict
from functools import singledispatchmethod, singledispatch

from epde.cache.cache_refactored import Cache, prepareVarTensor, uploadSimpleTokens, uploadGrids
from epde.decorators import BoundaryExclusion
from epde.preprocessing.domain_pruning import DomainPruner

from epde.supplementary import define_derivatives
from epde.evaluators import simple_function_evaluator # , trigonometric_evaluator

from epde.interface.prepared_tokens import DataPolynomials, CustomTokens
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

class InputEntry(object):
    def __init__(self, var_name: str, data_tensor: np.ndarray):
        self.names = var_name
        self.var_name = var_name

        if isinstance(data_tensor, np.ndarray):
            check_nparray(data_tensor) 
            self.ndim = data_tensor.ndim
        else:
            raise TypeError(f'data_tensor arg. in InputEntry.__init__(...) must be a np.ndarray.')
        
        self.data_tensor = data_tensor

    def setDerivatives(self, preprocesser: PreprocessingPipe, deriv_tensors: Union[list, np.ndarray] = None,
                        max_order: Union[list, tuple, int] = 1, grid: list = []):
        """
        Method for setting derivatives ot calculate derivatives from data

        Args:
            preprocesser (`PreprocessingPipe`): operator for preprocessing data (smooting and calculating derivatives).
                Typically, we use the one, initialized in EpdeSearch objects: EpdeSearch.preprocessor_pipeline
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
            raise NotImplementedError("Depricated method: used to be a way to create multi-sample setup")
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


class VariableEntry(InputEntry):
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
    def __init__(self, var_name: str, var_idx: int, data_tensor: np.ndarray): # Union[List[np.ndarray], np.ndarray]
        super().__init__(var_name, data_tensor)
        self.var_idx = var_idx

    # def use_global_cache(self): # , var_idx: int, deriv_codes: list
    #     """
    #     Method for add calculated derivatives in the cache
    #     """
    #     var_idx = self.var_idx
    #     deriv_codes = self.d_orders
    #     self.data_tensor = self.data_tensor[self.boundary != 0]
    #     self.derivatives = np.array([derivative[self.boundary.flatten() != 0] for derivative in self.derivatives.T]).T
    #     derivs_stacked = prepareVarTensor(self.data_tensor, self.derivatives, 
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
        derivs_stacked = prepareVarTensor(self.data_tensor, self.derivatives, 
                                            time_axis = time_axis)
        # print(f'Creating matched derivs: {[[self.var_idx, key, len(key) <= max_order] for idx, 
        #                                    key in enumerate(self.d_orders)]}')
        # print(f'From {self.d_orders}')
        return [(self.var_idx, key, derivs_stacked[idx, ...]) for idx, key in enumerate(self.d_orders)
                if len(key) <= max_order]


class Domain(object): # inheritance from cache objects?
    def __init__(self, 
                 grids: Union[np.ndarray, Tuple[np.ndarray], List[np.ndarray]], 
                 grid_cache: Cache,
                 time_axis: int = 0,
                 ID: int = 0,
                 boundary: Union[int, List[int], Tuple[int]] = 0):
        self._ID = ID
        self._time_axis = time_axis
        
        self._g_func = None
        self._g_func_flat_cache = None
        self._g_func_mask_cache = None        
        
        self._pruner = None
        self.inner_shape = None

        if isinstance(grids, np.ndarray): # or (isinstance(grid, (list, tuple)) and len(grid) == 1):
            self._dim = 1
            grids = [grids,]
        elif isinstance(grids, (tuple, list)):
            self._dim = len(grids)
        
        self._grid_cache: Cache = grid_cache
        uploadGrids(grids, grid_cache, domain_id = self._ID)
        self.setBoundaries(boundary)

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

    @property
    def ID(self) -> int:
        return self._ID

    def prune(self) -> None:
        raise NotImplementedError('Pruning is not implemented yet.')

    @staticmethod
    def getBCIdxs(tensor_shape, axis, rel_loc) -> Tuple[np.ndarray]:
        return tuple(np.meshgrid(*[np.arange(shape) if dim_idx != axis else min(int(rel_loc * shape), shape-1)
                                   for dim_idx, shape in enumerate(tensor_shape)], indexing='ij'))

    def get(self, key) -> np.ndarray:
        return self._grid_cache.get(label = key, subcache_ID = self._ID)

    def getGrids(self, mode: Literal['full', 'solver'] = 'full') -> List[np.ndarray]:
        _, grids = self._grid_cache.get_all(subcache_ID = self._ID)

        if mode == 'solver':
            for grid_idx in range(len(grids)):
                for dim_idx, bnd in enumerate(self.boundary_width_per_axis):
                    grids[grid_idx] = np.take(a       = grids[grid_idx], 
                                              indices = np.r_[bnd : grids[grid_idx].shape[dim_idx]-bnd],
                                              axis    = dim_idx)
        return grids

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

    @property
    def g_func_mask_flat(self) -> np.ndarray:
        """``g_func_mask`` flattened, for tensors held as (points, channels).

        Derivatives come back from the preprocessor flat -- ``(n_points,
        n_derivs)`` -- while the data tensor keeps the grid shape, so the two
        need different forms of the same mask. They coincide in 1-D, which is
        why indexing a derivative column with the grid-shaped mask worked for
        ODEs and raised ``IndexError: too many indices`` for every 2-D system.
        """
        return self.g_func_mask.reshape(-1)

    @property
    def g_func_masked_val(self) -> np.ndarray:
        return self.g_func[self.g_func_mask]

    @property
    def g_func_masked_flat(self) -> np.ndarray:
        return self.g_func_masked_val.reshape(-1)

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

    def setBoundaries(self, boundary_width: Union[int, list, tuple]):
        """
        Setting the number of unaccounted elements at the edges
        """
        # ``self.get`` reads THIS domain's subcache. Reading the cache
        # directly (``self._grid_cache.get('0')``) hit the default subcache
        # instead, so with several domains registered the fit-check validated
        # the width against whichever grid happened to be uploaded last --
        # and, for a scalar width, derived the per-axis form from that grid's
        # dimensionality rather than this one's.
        shape = self.get('0').shape

        self.initial_shape = shape
        if isinstance(boundary_width, int):
            if any([elem <= 2*boundary_width for elem in shape]):
                raise IndexError(f'Mismatching shapes: boundary of {boundary_width} does not fit data of shape {shape}')
        elif isinstance(boundary_width, (list, tuple)):
            if any([elem <= 2*boundary_width[idx] for idx, elem in enumerate(shape)]):
                raise IndexError(f'Mismatching shapes: boundary of {boundary_width} does not fit data of shape {shape}')                
        else:
            raise TypeError(f'Incorrect type of boundaries: {type(boundary_width)}, instead of expected int or list/tuple')

        self.boundary_width = boundary_width
        # Per-axis form, for callers that need to index by dimension.
        # ``getGrids(mode='solver')`` read a ``_boundary_width`` attribute that
        # was never assigned (AttributeError on every call), and iterating the
        # scalar form would have failed even after fixing the name.
        self.boundary_width_per_axis = ([boundary_width] * len(shape)
                                        if isinstance(boundary_width, int)
                                        else list(boundary_width))
        self.inner_shape = np.array(shape) - 2 * np.array(self.boundary_width_per_axis)
        print(f'Set domain {self} with inner shape of {self.inner_shape}')


@singledispatch
def addEntryToCache(input_entry, domain: Domain, cache: Cache, trajectory_id: int = 0):
    raise NotImplementedError('Trying to call default addEntryToCache function.')

@addEntryToCache.register
def _(input_entry: InputEntry, domain: Domain, cache: Cache, trajectory_id: int = 0):
    try:
        uploadSimpleTokens(labels = input_entry.names, cache = cache, # tensors = , 
                           subcache_ID = trajectory_id) # , deriv_codes = deriv_codes

    except AttributeError:
        raise NameError('Cache has not been declared before tensor addition.')      

@addEntryToCache.register
def _(input_entry: VariableEntry, domain: Domain, cache: Cache, trajectory_id: int = 0):
    deriv_codes = [(input_entry.var_idx, code) for code in input_entry.d_orders]

    # Flat mask: derivatives are (n_points, n_derivs); the data tensor below
    # keeps the grid shape and takes the grid-shaped mask.
    derivatives = np.array([derivative[domain.g_func_mask_flat]
                            for derivative in input_entry.derivatives.T]).T
    # print(f'In addEntryToCache: {derivatives}')
    derivs_stacked = prepareVarTensor(input_entry.data_tensor[domain.g_func_mask], derivatives, time_axis = domain._time_axis)
    
    try:
        uploadSimpleTokens(labels = input_entry.names, cache = cache, tensors = derivs_stacked, 
                           subcache_ID = trajectory_id, deriv_codes = deriv_codes)
        uploadSimpleTokens(labels = [input_entry.var_name,], cache = cache, tensors = [input_entry.data_tensor[domain.g_func_mask],],
                           subcache_ID = trajectory_id, deriv_codes=[(input_entry.var_idx, [None,]),])
        uploadSimpleTokens(labels = [input_entry.var_name,], cache = cache, tensors = [input_entry.data_tensor[domain.g_func_mask],])

    except AttributeError:
        raise NameError('Cache has not been declared before tensor addition.')      


class Trajectory(object): # Pass around in evo. operators or init in globals. Use instead of caches!
    """One data sample attached to a domain.

    A trajectory carries WHAT is differentiated and evaluated -- the tensors,
    the domain they live on, the pipeline that differentiates them -- but not
    HOW MANY derivatives or powers the search may use. Those describe the token
    pool, which is one shared structure however many trajectories feed it
    (``create_pool`` reads its families off ``data[0]``), so they belong to
    ``create_pool``/``fit`` and arrive here through :meth:`build`. Trajectories
    differ from one another in evaluation only.
    """

    def __init__(self, entries: Union[List[VariableEntry], Dict[str, np.ndarray]],
                 domain: Domain, cache: Cache,
                 cache_id = None, # ID: int = 0,
                 preprocessor_pipeline: PreprocessingPipe = None,
                 additional_cached_tokens: Dict[CustomTokens, Dict[str, np.ndarray]] = None,
                 derivs: Union[List[np.ndarray], np.ndarray] = None):
        self._data_tokens: List[TokenFamily]    = None

        self._domain: Domain = domain
        self._cache_id: int  = cache_id
        self._cache: Cache   = cache

        # Resolved now and kept, so that a later set_preprocessor() does not
        # retroactively change how already-registered data is differentiated.
        self._preprocessor_pipeline = preprocessor_pipeline
        self._derivs = derivs
        self._additional_cached_tokens = additional_cached_tokens

        if additional_cached_tokens is not None and not isinstance(additional_cached_tokens, dict):
            raise TypeError(f'additional_cached_tokens were passed incorrectly, expected dict, got {type(additional_cached_tokens)}')

        if isinstance(entries, list):
            for entry in entries:
                assert isinstance(entry, VariableEntry), \
                    'Entries have to be passed as list of VariableEntry objects or specific dicts.'
            self._entries = list(entries)

        elif isinstance(entries, dict):
            self._entries = []
            for key_idx, key in enumerate(entries.keys()):
                assert isinstance(key, str) and isinstance(entries[key], np.ndarray), \
                    'Dict of entries has to have str as keys and np.ndarrays as values.'
                self._entries.append(VariableEntry(key, key_idx, entries[key]))
        else:
            raise TypeError(f'Incorrect type of entries inputs for Trajectory: \
                              expected list[VariableEntry] or dict[str, np.ndarray], instead got {type(entries)}.')

        # Available before build(): pool invalidation and the legacy shim both
        # need to know which variables a trajectory carries.
        self.variable_names = [entry.var_name for entry in self._entries]

        self.families, self.base_derivs = [], []
        # Recorded by build(), not left at zero: EpdeSearch keys its
        # pool-invalidation check on this, and a constant made that check
        # meaningless -- a pool built for max_deriv_order=1 would be reused
        # for a request of 3.
        self.max_deriv_order = None
        self._built = None

    def build(self, max_deriv_order: Union[int, List[int], Tuple[int]] = 1,
              data_fun_pow: int = 1, deriv_fun_pow: int = 1) -> 'Trajectory':
        """Differentiate the data and derive the token families from it.

        Called by ``create_pool`` with the search-level orders and powers, so
        every trajectory in one pool is described by the same families. Calling
        it again with the same arguments is a no-op; calling it with different
        ones recomputes, which is what makes a pool rebuild honest.
        """
        request = (max_deriv_order, data_fun_pow, deriv_fun_pow)
        if self._built == request:
            return self

        self.max_deriv_order = max_deriv_order
        self.families, self.base_derivs = [], []
        for entry in self._entries:
            fam, bd = self.setEntry(entry, self._preprocessor_pipeline,
                                    self._derivs, max_deriv_order,
                                    data_fun_pow, deriv_fun_pow)
            self.families.extend(fam); self.base_derivs.extend(bd)

        if isinstance(self._additional_cached_tokens, dict):
            for token_type, tokens_val in self._additional_cached_tokens.items():
                self.uploadTokenTensors(token_type, tokens_val)

        self._built = request
        return self

    def uploadTokenTensors(self, family, tensors: Dict[str, np.ndarray]) -> None:
        """Put a token family's declared tensors into THIS trajectory's subcache.

        The tensors arrive on the full grid, like the variable data, and are
        masked the same way -- ``addEntryToCache`` does exactly this for the
        variable itself. Without the mask they keep the untrimmed shape and no
        longer line up with anything else in the cache.
        """
        masked = {label: np.asarray(tensor)[self._domain.g_func_mask]
                  for label, tensor in tensors.items()}
        family.upload(trajectory = masked, cache = self._cache,
                      traj_id = self._cache_id)

    @property
    def built(self) -> bool:
        return self._built is not None

    @property
    def tokens(self):
        if self._built is None:
            raise RuntimeError(
                'Trajectory.tokens was read before build(): the token families '
                'follow from max_deriv_order/data_fun_pow/deriv_fun_pow, which '
                'are create_pool/fit arguments. Pass this trajectory to '
                'create_pool(...) rather than reading its families directly.')
        return self.families # _data_tokens

    def setEntry(self, entry: VariableEntry, preprocessor_pipeline: PreprocessingPipe = None,
                 derivs: np.ndarray = None, max_deriv_order: Union[int, List[int], Tuple[int]] = 1,
                 data_fun_pow: int = 1, deriv_fun_pow: int = 1) -> tuple:
        '''
        Returns tuple: first element is a list  matched labled for derivatives
        '''
        entry.setDerivatives(preprocesser = preprocessor_pipeline, deriv_tensors = derivs,
                             grid = self._domain.getGrids(), max_order = max_deriv_order)
        
        addEntryToCache(entry, self._domain, self._cache, self.ID)

        entry.create_derivs_family(max_deriv_power=deriv_fun_pow)
        entry.create_polynomial_family(max_power=data_fun_pow)

        return entry.get_families(), entry.matched_derivs(max_order = 2)

    @property
    def inner_shape(self) -> np.ndarray:        # Tuple[int]:
        return self._domain.inner_shape

    def checkShapes(self, domain: Domain) -> bool:
        sample_tensor = self._cache.get(subcache_ID = self._cache_id)
        print('In check shapes: ', domain.inner_shape, type(domain.inner_shape), sample_tensor.shape, type(sample_tensor.shape))
        return (domain.inner_shape == sample_tensor.shape)

    @property
    def ID(self) -> int:
        return self._cache_id

    def grids(self, mode: Literal['full', 'solver'] = 'full') -> np.ndarray:
        """``mode='solver'`` trims ``boundary_width`` off every axis -- the
        INNER domain the cached tensors and ``Equation.evaluate`` live on.
        ``Domain.getGrids`` has always offered it; the trajectory accessors
        did not pass it through, so the solver was handed full grids and
        indexed inner-domain values with full-grid indices."""
        return self._domain.getGrids(mode=mode)
    
    def gFuncs(self, mode: Literal['d', 'f', 'm', 'dmf', 'dm'] = 'd') -> np.ndarray:
        match mode:
            case 'd':
                return self._domain.g_func
            case 'f':
                return self._domain.g_func_flat
            case 'm':
                return self._domain.g_func_mask
            case 'dmf':
                return self._domain.g_func_masked_flat
            case 'dm':
                return self._domain.g_func_masked_val
            case _:
                raise NotImplementedError("Incorrect mode for calling g_func of the domain.")

    def get(self, label, normalized=False, saved_as=None, deriv_code = None) -> np.ndarray:
        return self._cache.get(label = label,
                               subcache_ID = self._cache_id,
                               normalized = normalized,
                               # saved_as = saved_as,
                               deriv_code = deriv_code)
        
    def add(self, label, tensor: np.ndarray, normalized: bool = False,
            deriv_code = None, indication: bool = False) -> bool:
        return self._cache.add(label, tensor, self._cache_id, normalized, deriv_code, indication)

    # def __call__(self, *args, **kwargs):
    #     return global_var.tensor_cache.get(*args, **kwargs)

    # def load(self, label):
    #     global_var.tensor_cache()

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

class TrajectoriesManager(object):
    def __init__(self):
        # self.exists = True
        self.reset()

    def reset(self):
        self._traj: Dict[int, Trajectory] = dict()
        self._domains: Dict[int, Domain] = dict()

    def addTrajectory(self, trajectory: Trajectory, domain: Domain):
        print(f'Added trajectory {trajectory.ID} to exist. traj: {self._traj.keys()}')
        # The guard is against two DIFFERENT samples claiming one cache id --
        # re-registering the same object is not that, and it happens whenever
        # a pool is rebuilt for new orders or powers (create_pool called twice
        # with the same trajectories), which used to raise here.
        if (trajectory.ID in self._traj.keys()
                and self._traj[trajectory.ID] is not trajectory):
            raise KeyError(f'Trajectory with key {trajectory.ID}')

        self._traj[trajectory.ID] = trajectory
        if domain._ID not in self._domains.keys():
            self._domains[domain._ID] = domain
        else:
            assert trajectory.checkShapes(self._domains[domain._ID])

    @property
    def inner_shapes(self):
        return {key: trajectory.inner_shape for key, trajectory in self._traj.items()}

    def __getitem__(self, key) -> np.ndarray:
        if key not in self._traj:
            raise KeyError(f"Missing key {key} from trajectories: {self._traj.keys}")
        
        return self._traj[key]

    def __contains__(self, obj):
        '''
        Valid input type:
            'label' (checked in unnormalized data); ('label1', normalized), where normalized is bool (T if norm, else F);
            np.ndarray of values (checked in unnormalized data); (np.ndarray, normalized), where normalized is bool
            (T if norm, else F) and np.ndarray is np.ndarray of tensor values. Does not support scaled vals
        '''
        return all([obj in trajectory._cache for trajectory in self._traj.values()])

    @property
    def trajecatoryIDs(self) -> List[int]:
        return [key for key in self._traj.keys()]

    def grids(self, mode: Literal['full', 'solver'] = 'full') -> Dict[int, np.ndarray]:
        return {key: trajectory.grids(mode) for key, trajectory in self._traj.items()}

    @property
    def grid_keys(self) -> list:
        # A bit too hard-coded as: 
        traj_ID = self.trajecatoryIDs[0]
        return self._traj[traj_ID]._domain._grid_cache.getKeys()

    @property
    def ndim(self) -> int:
        traj_ID = self.trajecatoryIDs[0]
        return self._traj[traj_ID]._domain._grid_cache.get(label = None).ndim

    def gFunc(self, mode_key: Literal['d', 'f', 'm', 'dmf', 'dm']) -> Dict[int, np.ndarray]:
        return {key: trajectory.gFuncs(mode_key) for key, trajectory in self._traj.items()}

    def get(self, label, normalized=False, saved_as=None, deriv_code = None, sample_key: int = None, ) -> Dict[int, np.ndarray]:
        '''
        Method, mirroring eponymous method of the cache, aimed on calling the method for all samples.
        '''
        # for trajectory in self._traj.values():
        #     print(label, trajectory.get(label, normalized, saved_as, deriv_code).shape)
        if sample_key is None:
            return {key : trajectory.get(label, normalized, saved_as, deriv_code) for key, trajectory in self._traj.items()}
        else:
            return {sample_key: self._traj[sample_key].get(label, normalized, saved_as, deriv_code)}
    
    def getSingleTrajectory(self, label, traj_key, normalized=False, saved_as=None, deriv_code = None) -> Dict[int, np.ndarray]:
        '''
        Method, mirroring eponymous method of the cache, aimed on calling the method for all samples.
        '''
        return self._traj[traj_key].get(label, normalized, saved_as, deriv_code)


    def add(self, label, tensors: Dict[int, np.ndarray], normalized: bool = False,
            deriv_code = None, indication: bool = False) -> bool:
        success_flag: bool = True

        for key in self._traj.keys():
            if key not in tensors:
                raise RuntimeError(f'Missing tensor for sample {key} from tensor dict argument.')
            sf_new = self._traj[key].add(label, tensors[key], normalized = normalized, 
                                         deriv_code = deriv_code, indication = indication)
            success_flag = (success_flag and sf_new)

        return success_flag
    
