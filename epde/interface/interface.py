"""

Inteface objects for EPDE framework

Contains:
---------

**InputDataEntry** class, containing logic for preparing the input data for the equation search,
such as initialization of neccessary token families and derivatives calculation.

**EpdeSearch** class for main interactions between the user and the framework.

"""
import pickle
import numpy as np
import torch

from copy import deepcopy
from typing import Union, Callable, List, Tuple
from collections import OrderedDict
from functools import reduce, singledispatchmethod

import epde.globals as global_var

from epde.optimizers.builder import StrategyBuilder
from epde.optimizers.builder import OptimizationPatternDirector

from epde.optimizers.moeadd.moeadd import *
from epde.optimizers.moeadd.supplementary import *
from epde.optimizers.moeadd.strategy import MOEADDDirector
from epde.optimizers.moeadd.strategy_elems import MOEADDSectorProcesser

from epde.optimizers.single_criterion.optimizer import EvolutionaryStrategy, SimpleOptimizer, Population
from epde.optimizers.single_criterion.strategy import BaselineDirector
from epde.optimizers.single_criterion.supplementary import simple_sorting

from epde.preprocessing.domain_pruning import DomainPruner
from epde.operators.utils.default_parameter_loader import EvolutionaryParams

from epde.decorators import BoundaryExclusion

from epde.evaluators import simple_function_evaluator, trigonometric_evaluator
from epde.supplementary import define_derivatives
from epde.cache.cache import upload_simple_tokens, upload_grids, prepare_var_tensor

from epde.preprocessing.preprocessor_setups import PreprocessorSetup
from epde.preprocessing.preprocessor import ConcretePrepBuilder, PreprocessingPipe

from epde.structure.main_structures import Equation, SoEq

from epde.interface.token_family import TFPool, TokenFamily
from epde.interface.type_checks import *
from epde.interface.prepared_tokens import PreparedTokens, CustomTokens, DataPolynomials
from epde.integrate import BoundaryConditions, SolverAdapter, SystemSolverInterface

class InputDataEntry(object):
    """
    Represents a single entry of input data. This class is designed to hold and manage individual data points used within the equation discovery process.
    
    
        Attributes:
            var_name (`str`): name of input data dependent variable
            data_tensor (`np.ndarray`): value of the input data
            names (`list`): keys for derivatides
            d_orders (`list`): keys for derivatides on `int` format for `solver`
            derivatives (`np.ndarray`): values of derivatives
            deriv_properties (`dict`): settings of derivatives
    """

    def __init__(self, var_name: str, var_idx: int, data_tensor: Union[List[np.ndarray], np.ndarray]):
        """
        Initializes a DataTensor object, associating a variable with its corresponding data.
        
                This method creates a `DataTensor` instance, linking a variable name and index to its numerical representation.
                The data tensor's dimensionality is also inferred and validated to ensure consistency. This association is crucial
                for representing the data in a structured format suitable for equation discovery.
        
                Args:
                    var_name (str): The name of the variable.
                    var_idx (int): The index of the variable.
                    data_tensor (Union[List[np.ndarray], np.ndarray]): The data tensor, which can be a NumPy array or a list of NumPy arrays.
        
                Returns:
                    None
        
                Class Fields:
                    var_name (str): The name of the variable associated with the data tensor.
                    var_idx (int): The index of the variable associated with the data tensor.
                    ndim (int): The dimensionality of the data tensor.
                    data_tensor (Union[List[np.ndarray], np.ndarray]): The actual data tensor, which can be a single NumPy array or a list of NumPy arrays.
        """
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


    def set_derivatives(self, preprocesser: PreprocessingPipe, deriv_tensors: Union[list, np.ndarray] = None,
                        max_order: Union[list, tuple, int] = 1, grid: list = []):
        """
        Method for associating derivative information with the input data.
        
                This method prepares the data for the equation discovery process by calculating or assigning derivative values.
                It ensures that the data is properly formatted and that the necessary derivative information is available for
                the subsequent steps of equation learning.
        
                Args:
                    preprocesser (`PreprocessingPipe`): Operator for preprocessing data, including smoothing and derivative calculation.
                    deriv_tensors (`np.ndarray`, optional): Pre-calculated derivative values. If None, derivatives are computed using the preprocesser.
                    max_order (`list`|`tuple`|`int`): The maximum order of derivatives to be calculated or used.
                    grid (`list`): Grid values corresponding to the data points, used for derivative calculation.
        
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

    def use_global_cache(self): # , var_idx: int, deriv_codes: list
        """
        Method for add calculated derivatives in the cache
        """
        var_idx = self.var_idx
        deriv_codes = self.d_orders
        derivs_stacked = prepare_var_tensor(self.data_tensor, self.derivatives, 
                                            time_axis=global_var.time_axis)
        deriv_codes = [(var_idx, code) for code in deriv_codes]

        try:
        """
        Stores precomputed derivative tensors in a global cache for efficient reuse.
        
        This method stores the calculated derivatives and the original data tensor
        in a global cache, indexed by variable name and derivative order. This
        avoids redundant computations when the same derivatives are needed
        multiple times during the equation discovery process. The method also
        prepares the data by adjusting the time axis for consistency.
        
        Args:
            self: The InputDataEntry instance containing the data and derivatives.
        
        Returns:
            None. The method updates the global tensor and initial data caches.
        """
            upload_simple_tokens(self.names, global_var.tensor_cache, derivs_stacked, 
                                 deriv_codes=deriv_codes)
            upload_simple_tokens([self.var_name,], global_var.tensor_cache, [self.data_tensor,],
                                 deriv_codes=[(var_idx, [None,]),])
            upload_simple_tokens([self.var_name,], global_var.initial_data_cache, [self.data_tensor,])

        except AttributeError:
            raise NameError('Cache has not been declared before tensor addition.')
        print(f'Size of linked labels is {len(global_var.tensor_cache._deriv_codes)}')
        global_var.tensor_cache.use_structural()

    @staticmethod
    def latex_form(label, **params):
        """
        Constructs a LaTeX-formatted representation of a token, incorporating its label and any associated parameters like power.
        
                This ensures proper display of mathematical expressions, especially when dealing with derivatives and exponents.
        
                Args:
                    label (str): The label of the token to be formatted.
                    **params (dict): A dictionary containing parameter labels as keys and tuples of (value, LaTeX-formatted string) as values.  Crucially expects a 'power' key.
        
                Returns:
                    str: The LaTeX-formatted string representation of the token.
                
                Why:
                    This method is essential for generating human-readable and mathematically correct representations of the discovered equation terms. The LaTeX format ensures that derivatives, fractions, and exponents are displayed properly, aiding in the interpretation and validation of the identified equations.
        """
        if '/' in label:
            label = label[:label.find('x')+1] + '_' + label[label.find('x')+1:]
            label = label.replace('d', r'\partial ').replace('/', r'}{')
            label = r'\frac{' + label + r'}'
                            
        if params['power'][0] > 1:
            label = r'\left(' + label + r'\right)^{{{0}}}'.format(params["power"][1])
        return label

    def create_derivs_family(self, max_deriv_power: int = 1):
        """
        Creates a family of tokens representing the derivatives of this token.
        
                This method generates a `TokenFamily` instance to manage the derivatives
                of the current token, configuring its LaTeX representation, status flags,
                derivative parameters, and evaluation function. This is crucial for
                representing and manipulating derivative terms within the equation discovery process.
        
                Args:
                    max_deriv_power: The maximum order of derivative to include in the family.
        
                Returns:
                    None.
        
                Initializes:
                    _derivs_family (TokenFamily): A TokenFamily instance representing the derivatives
                        of the current token. It is initialized with the token type as
                        'deriv of {self.var_name}', the variable name, and the
                        `family_of_derivs` flag set to True. The latex form constructor is set using `self.latex_form`.
                        The status is set to `demands_equation=True`, `unique_specific_token=False`,
                        `unique_token_type=False`, `s_and_d_merged=False`, and `meaningful=True`.
                        The parameters are set using `self.names`, an OrderedDict with 'power'
                        ranging from 1 to `max_deriv_power`, a default power of 0, and `self.d_orders`.
                        The evaluator is set to `simple_function_evaluator`.
        """
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
        """
        Creates a family of polynomial tokens for representing the input data.
        
                This method generates a set of polynomial tokens based on the input variable name
                and a specified maximum power. These tokens are then stored as the polynomial family,
                which will be used to construct candidate differential equations.
                
                Args:
                    max_power: The maximum power to which the input variable will be raised in the polynomial tokens.
        
                Returns:
                    None. The polynomial family is stored internally for later use in equation discovery.
        """
        polynomials = DataPolynomials(self.var_name, max_power = max_power)
        self._polynomial_family = polynomials.token_family

    def get_families(self):
        """
        Returns the polynomial and derivative families associated with this data entry.
        
        These families define the basis functions and derivative operators used to construct candidate differential equations.
        
        Args:
            None
        
        Returns:
            list: A list containing the polynomial family and the derivative family.
        """
        return [self._polynomial_family, self._derivs_family]

    def matched_derivs(self, max_order = 1):
        """
        Returns a list of derivatives matched to a maximum order for equation discovery.
        
                This method filters the derivatives associated with the variable,
                selecting those with an order less than or equal to `max_order`.
                These derivatives are then prepared for use in the equation discovery process.
                The derivatives are stacked using `prepare_var_tensor` to prepare them for use.
                This ensures that only relevant terms, up to the specified order,
                are considered when constructing candidate equations.
        
                Args:
                    max_order: The maximum order of derivatives to include. Defaults to 1.
        
                Returns:
                    A list of lists, where each inner list contains:
                        - The variable index (`self.var_idx`).
                        - The derivative order (`key`).
                        - The derivative tensor (`derivs_stacked[idx, ...]`).
                    Only derivatives with an order less than or equal to `max_order` are included.
        """
        derivs_stacked = prepare_var_tensor(self.data_tensor, self.derivatives, 
                                            time_axis=global_var.time_axis)
        # print(f'Creating matched derivs: {[[self.var_idx, key, len(key) <= max_order] for idx, 
        #                                    key in enumerate(self.d_orders)]}')
        # print(f'From {self.d_orders}')
        return [[self.var_idx, key, derivs_stacked[idx, ...]] for idx, key in enumerate(self.d_orders)
                if len(key) <= max_order]

def simple_selector(sorted_neighbors, number_of_neighbors=4):
    """
    Selects the most relevant neighbors from a ranked list to refine equation discovery.
    
        This function takes a list of neighbors, assumed to be sorted by relevance
        (e.g., based on some error metric), and returns a subset of the top neighbors.
        This selection is crucial for focusing the search on the most promising equation candidates
        and improving the efficiency of the equation discovery process.
    
        Args:
            sorted_neighbors (list): A list of neighbors, sorted by relevance.
            number_of_neighbors (int, optional): The number of top neighbors to select. Defaults to 4.
    
        Returns:
            list: A list containing the first `number_of_neighbors` elements from the
                  `sorted_neighbors` list, representing the most relevant neighbors.
    """
    return sorted_neighbors[:number_of_neighbors]

class EpdeSearch(object):
    """
    Intialization of the epde search object. Here, the user can declare the properties of the 
        search mechainsm by defining evolutionary search strategy.
    
        Attributes:
            multiobjective_mode (`bool`): set mode of multiobjective optimization during equation search
            preprocessor_set (`bool`): flag about using defined algorithm for preprocessing input data
            director (`OptimizationPatternDirector`): optional
                Pre-defined director, responsible for construction of multi-objective evolutionary optimization
                strategy; shall not be interfered with unless for very specific tasks.
            director_params (`dict`): optionals
                Contains parameters for evolutionary operator builder / construction director, that
                can be passed to individual operators. Keys shall be 'variation_params', 'mutation_params',
                'pareto_combiner_params', 'pareto_updater_params'.
            search_conducted (`bool`): flag that the equation was searched 
            optimizer_init_params (`dict`): parameters for optimization algorithm initialization
            optimizer_exec_params (`dict`): parameters for execution algorithm of optimization
            optimizer (`OptimizationPatternDirector`): the strategy of the evolutionary algorithm
    """

    def __init__(self, multiobjective_mode: bool = True, use_pic = False, use_default_strategy: bool = True, director=None, 
                 director_params: dict = {'variation_params': {}, 'mutation_params': {},
                                          'pareto_combiner_params': {}, 'pareto_updater_params': {}}, 
                 time_axis: int = 0, define_domain: bool = True, function_form=None, boundary: int = 0,
                 use_solver: bool = False, verbose_params: dict = {'show_iter_idx' : True}, 
                 coordinate_tensors=None, memory_for_cache=15, prune_domain: bool = False,
                 pivotal_tensor_label=None, pruner=None, threshold: float = 1e-2, 
                 division_fractions=3, rectangular: bool = True, 
                 params_filename: str = None, device: str = 'cpu'):
        """
        Initializes the EPDE search object with specified parameters.
        
                This method sets up the search environment, including the evolutionary strategy,
                domain properties, and optimization mode (single or multi-objective). It prepares
                the system for discovering differential equations from data by configuring the
                search space and optimization algorithms.
        
                Args:
                    multiobjective_mode (`bool`): optional, default True
                        Flag indicating whether to use multi-objective optimization (MOEADD). If False,
                        single-objective optimization is used. Multi-objective optimization allows to find
                        a set of equations that are good in terms of different criteria, such as accuracy and complexity.
                    use_pic (`bool`): optional, default False
                        Flag indicating whether to use Physics Informed learning paradigm.
                    use_default_strategy (`bool`): optional, default True
                        If True, the default evolutionary strategy is used. If False, a user-defined
                        strategy must be provided via the `director` argument. The evolutionary strategy
                        defines how the search for equations is conducted.
                    director (`object`): optional
                        A custom director object that implements a specific evolutionary strategy.
                        Required if `use_default_strategy` is False.
                    director_params (`dict`): optional
                        Parameters for the director object, such as variation, mutation, and Pareto
                        combination/update parameters.
                    time_axis (`int`): optional, default 0
                        The index of the time axis in the input data and coordinate tensors. Used for
                        calculating time derivatives.
                    define_domain (`bool`): optional, default True
                        If True, the domain properties (coordinate tensors, memory cache, boundary) are
                        initialized. Disabling this is useful when the domain is already defined.
                    function_form (`callable`): optional
                        An auxiliary function used in the weak derivative definition. The default is a
                        negative square function centered in the domain.
                    boundary (`int` or `tuple/list of integers`): optional, default 0
                        The width of the boundary region to exclude from the equation discovery process.
                    use_solver (`bool`): optional, default False
                        If True, an automatic partial differential equation solver is used to evaluate the
                        fitness of candidate solutions. This can improve the accuracy of the discovered equations.
                    verbose_params (`dict`): optional
                        A dictionary of parameters controlling the verbosity of the algorithm's output.
                    coordinate_tensors (`list of np.ndarrays`): optional
                        The values of the coordinates at the grid nodes where the function values are known.
                        For 1D problems, this is a NumPy array. For higher-dimensional problems, this can
                        be generated using `numpy.meshgrid`. If None, tensors are created as ranges with a
                        step of 1 between nodes.
                    memory_for_cache (`int` or `float`): optional, default 15
                        An estimate of the memory (in MB) that can be used to cache pre-evaluated tensors.
                        Caching can speed up the equation discovery process.
                    prune_domain (`bool`): optional, default False
                        If True, subdomains with no dynamics are pruned from the data. This can improve
                        the efficiency of the search.
                    pivotal_tensor_label (`str`): optional
                        The label of the tensor used to identify subdomains with no dynamics. The default
                        is 'du/dt', where 't' is the time axis.
                    pruner (`object`): optional
                        A custom pruner object that removes subdomains with no dynamics.
                    threshold (`float`): optional, default 1e-2
                        The threshold for determining whether a pivotal tensor value is considered zero.
                        Used by the pruner.
                    division_fractions (`int`): optional, default 3
                        The number of subdomains along each axis used for domain pruning.
                    rectangular (`bool`): optional, default True
                        If True, entire lines of subdomains along an axis can be removed if all values
                        within them are zero.
                    params_filename (`str`): optional
                        Path to a JSON file containing evolutionary parameters. If None, default parameters
                        are used based on the optimization mode.
                    device (`str`): optional, default 'cpu'
                        The device to use for computations (e.g., 'cpu' or 'cuda').
        
                Returns:
                    None
        """
        self._device = device
        self.multiobjective_mode = multiobjective_mode
        self._use_pic = use_pic

        global_var.set_time_axis(time_axis)
        global_var.init_verbose(**verbose_params)
        self.preprocessor_set = False

        if define_domain:
            if coordinate_tensors is None:
                raise ValueError('Grids can not be empty during calculations.')
            self.set_domain_properties(coordinate_tensors, memory_for_cache, boundary, function_form=function_form,
                                       prune_domain=prune_domain, pruner=pruner, pivotal_tensor_label=pivotal_tensor_label,
                                       threshold=threshold, division_fractions=division_fractions,
                                       rectangular=rectangular)

        self._mode_info = {'criteria': 'multi objective' if multiobjective_mode else 'single objective',
                           'solver_fitness': use_solver}

        # Here we initialize a singleton object with evolutionary params. It is used in operators' initialization.
        EvolutionaryParams.reset()
        evo_param = EvolutionaryParams(parameter_file = params_filename, mode = self._mode_info['criteria'])

        if director is not None and not use_default_strategy:
            self.director = director
        elif director is None and use_default_strategy:
            if self.multiobjective_mode:
                self.director = MOEADDDirector()
                builder = StrategyBuilder(MOEADDSectorProcesser)
            else:
                self.director = BaselineDirector()
                builder = StrategyBuilder(EvolutionaryStrategy)
            self.director.builder = builder
            self.director.use_baseline(use_solver=self._mode_info['solver_fitness'], 
                                       use_pic=self._use_pic, params=director_params)
        else:
            raise NotImplementedError('Wrong arguments passed during the epde search initialization')

        if self.multiobjective_mode:    
            self.set_moeadd_params()
        else:
            self.set_singleobjective_params()

        self.pool = None
        self.search_conducted = False

    def set_memory_properties(self, example_tensor, mem_for_cache_frac=None, mem_for_cache_abs=None):
        """
        Sets the memory constraints for caching intermediate tensor results during the equation search. This is crucial for balancing computational speed and memory usage, especially when dealing with large datasets or complex equation spaces. By limiting the memory footprint of the tensor cache, the search process can avoid memory overflow errors and maintain efficiency.
        
                Args:
                    example_tensor (`ndarray`): A representative tensor used to estimate the memory consumption of other tensors during the equation search process.
                    mem_for_cache_frac (`float`, optional): The fraction of total RAM to allocate for caching tensors. Defaults to None.
                    mem_for_cache_abs (`int`, optional): The absolute amount of memory (in bytes) to allocate for caching tensors. Defaults to None.
        
                Returns:
                    None
        """
        if global_var.grid_cache is not None:
            if mem_for_cache_frac is not None:
                mem_for_cache_frac = int(mem_for_cache_frac/2.)
            else:
                mem_for_cache_abs = int(mem_for_cache_abs/2.)
        global_var.tensor_cache.memory_usage_properties(example_tensor, mem_for_cache_frac, mem_for_cache_abs)

    def set_moeadd_params(self, population_size: int = 6, solution_params: dict = {},
                          delta: float = 1/50., neighbors_number: int = 3,
                          nds_method: Callable = fast_non_dominated_sorting,
                          ndl_update_method: Callable = ndl_update,
                          subregion_mating_limitation: float = .95,
                          PBI_penalty: float = 1., training_epochs: int = 100,
                          neighborhood_selector: Callable = simple_selector,
                          neighborhood_selector_params: tuple = (4,)):
        """
        Sets the parameters for the multi-objective evolutionary algorithm used to discover differential equations. Default values are set during the initialization of the `EpdeSearch` object. These parameters control the search process, influencing the diversity and convergence of solutions representing potential equation structures.
        
                Args:
                    population_size (`int`): optional
                        The size of the population of solutions (equation candidates) created during the multi-objective optimization. Default is 6.
                    solution_params (`dict`): optional
                        A dictionary containing additional parameters to be passed to newly created solutions. This allows for customization of individual equation candidates.
                    delta (`float`): optional
                        Parameter controlling the uniform spacing between weight vectors. `H = 1 / delta` should be an integer, representing the number of divisions along an objective coordinate axis.
                    neighbors_number (`int`): *> 0*, optional
                        The number of neighboring weight vectors considered during evolutionary operations. This influences the exploration of the solution space by considering similar equation structures.
                    nds_method (`callable`): optional, default ``moeadd.moeadd_supplementary.fast_non_dominated_sorting``
                        The method used for non-dominated sorting of candidate solutions. The default method is implemented according to the article *K. Deb, A. Pratap, S. Agarwal, and T. Meyarivan, “A fast and elitist multiobjective genetic algorithm: NSGA-II,” IEEE Trans. Evol. Comput., vol. 6, no. 2, pp. 182–197, Apr. 2002.*
                    ndl_update_method (`callable`): optional, default ``moeadd.moeadd_supplementary.NDL_update``
                        The method for adding a new solution point into the objective functions space, minimizing the recalculation of non-dominated levels for the entire population. The default method is taken from *K. Li, K. Deb, Q. Zhang, and S. Kwong, “Efficient non-domination level update approach for steady-state evolutionary multiobjective optimization,” Dept. Electr. Comput. Eng., Michigan State Univ., East Lansing, MI, USA, Tech. Rep. COIN No. 2014014, 2014.*
                    neighborhood_selector (`callable`): optional
                        The method for finding "close neighbors" of a vector with a proximity list. The baseline example, presented in ``moeadd.moeadd_stc.simple_selector``, selects n-adjacent ones.
                    subregion_mating_limitation (`float`): optional
                        The probability of mating selection being limited only to selected subregions (adjacent to the weight vector domain).  Value should be in the range :math:`\delta \in [0., 1.)`.
                    neighborhood_selector_params (`tuple|list`): optional
                        An iterable passed to the `neighborhood_selector` as an argument. Use `None` if no additional arguments are required.
                    training_epochs (`int`): optional
                        The maximum number of iterations during which the optimization will be performed. The optimization stops if the algorithm converges to a single Pareto frontier.
                    PBI_penalty (`float`): optional
                        The penalty parameter used in penalty-based intersection calculation. Default value is 1.
        
                Returns:
                    None
        """
        self.optimizer_init_params = {'weights_num': population_size, 'pop_size': population_size,
                              'delta': delta, 'neighbors_number': neighbors_number,
                              'solution_params': solution_params,
                              'nds_method' : nds_method, 
                              'ndl_update' : ndl_update_method}
        
        self.optimizer_exec_params = {'epochs' : training_epochs}
 
    def set_singleobjective_params(self, population_size: int = 4, solution_params: dict = {}, 
                                   sorting_method: Callable = simple_sorting, training_epochs: int = 50):
        """
        Sets the parameters required for the single-objective evolutionary search.
        
                This configuration step is crucial for tailoring the search process to the specific problem, 
                allowing control over the population size, solution generation, sorting strategies, and training duration.
                These parameters influence how the algorithm explores the space of possible differential equations 
                and converges towards an optimal solution.
        
                Args:
                    population_size (`int`): optional, default - 4
                        Size of population.  A larger population may improve exploration but increases computational cost.
                    solution_params (`dict`):
                        Parameters guiding the creation of candidate solutions (equation structures).  
                        These parameters define the building blocks and constraints for generating differential equations.
                    sorting_method(`callable`): optional, default - `simple_sorting`
                        Method for sorting individuals in the population based on their fitness.  
                        The sorting method determines how the evolutionary algorithm selects promising equation structures for reproduction and refinement.
                    training_epochs (`int`): optional, default - 50
                        Maximum number of iterations for the optimization process.  
                        This parameter limits the computational time spent searching for the best equation.
        
                Returns:
                    None
        """
        self.optimizer_init_params = {'pop_size' : population_size, 'solution_params': solution_params,
                                      'sorting_method' : sorting_method}
        
        self.optimizer_exec_params = {'epochs' : training_epochs}        
    
    def domain_pruning(self, pivotal_tensor_label = None, pruner = None, 
                             threshold : float = 1e-5, division_fractions = 3, 
                             rectangular : bool = True):
        """
        Method for refining the search domain by focusing on regions exhibiting significant dynamic behavior.
        
                This method identifies and isolates areas within the domain where the dynamics, as indicated by the pivotal tensor, are most pronounced. By discarding regions with minimal change, the search for governing equations becomes more efficient and accurate.
        
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
            if pivotal_tensor_label is None:
                pivotal_tensor_label = ('du/dx'+str(global_var.time_axis + 1), (1.0,))
            elif isinstance(pivotal_tensor_label, str):
                pivotal_tensor_label = (pivotal_tensor_label, (1.0,))
            elif not isinstance(pivotal_tensor_label, tuple):
                raise TypeError('Label of the pivotal tensor shall be declared with str or tuple')

            pivotal_tensor = global_var.tensor_cache.get(pivotal_tensor_label)
            self.pruner.get_boundaries(pivotal_tensor, division_fractions=division_fractions,
                                       rectangular=rectangular)

        global_var.tensor_cache.prune_tensors(self.pruner)
        if global_var.grid_cache is not None:
            global_var.grid_cache.prune_tensors(self.pruner)

    def _create_caches(self, coordinate_tensors, memory_for_cache):
        """
        Creates and initializes caches for storing tensors, which are essential for efficient equation discovery.
        
                The caches are used to store coordinate tensors and other intermediate results, 
                allowing for faster access and manipulation during the evolutionary search process.
                This pre-allocation of memory optimizes performance by reducing the overhead of repeated memory allocation and deallocation.
                
                Args:
                    coordinate_tensors (`np.ndarray|list`):
                        Grid values, passed as a single `np.ndarray` or a list of `np.ndarray`'s. These represent the independent variables of the differential equation.
                    memory_for_cache (`float`): 
                        Fraction of available memory to be used for caching tensors. This parameter controls the trade-off between memory usage and computational speed.
                
                Returns:
                    None
        """
        global_var.init_caches(set_grids=True, device=self._device)
        example = coordinate_tensors if isinstance(coordinate_tensors, np.ndarray) else coordinate_tensors[0]
        self.set_memory_properties(example_tensor=example, mem_for_cache_frac=memory_for_cache)
        upload_grids(coordinate_tensors, global_var.initial_data_cache)
        upload_grids(coordinate_tensors, global_var.grid_cache)

    def set_boundaries(self, boundary_width: Union[int, list]):
        """
        Sets the boundary conditions for the computational grid. This ensures that the equation discovery process accounts for edge effects and avoids overfitting to boundary artifacts.
        
                Args:
                    boundary_width (Union[int, list]): The width of the boundary region to be excluded from consideration during equation discovery. Can be an integer for uniform width or a list for variable widths along different dimensions.
        
                Returns:
                    None
        """
        global_var.grid_cache.set_boundaries(boundary_width=boundary_width)

    def _upload_g_func(self, function_form: Union[Callable, np.ndarray, list] = None):
        """
        Loads and prepares a test function for weak derivative calculations. This function is crucial for evaluating the fitness of candidate equations by projecting them onto a suitable function space.
        
                Args:
                    function_form (`callable`, or `np.ndarray`, or `list[np.ndarray]`, optional):
                        The test function to be used. It can be a callable function, a NumPy array, or a list of NumPy arrays. If `None`, a default inverse polynomial function with a maximum at the domain center is used. Defaults to `None`.
        
                Returns:
                    None
        
                Raises:
                    NameError: If the grid cache has not been initialized.
        
                Why:
                    This method sets up the test function used in the weak formulation of derivatives. The weak formulation allows the framework to handle noisy or incomplete data by integrating the equation against a test function, effectively smoothing the derivatives. The choice of test function can influence the accuracy and stability of the equation discovery process.
        """
        if isinstance(function_form, (np.ndarray, list)):
            global_var.grid_cache.g_func = function_form
        else:
            try:
                decorator = BoundaryExclusion(boundary_width=global_var.grid_cache.boundary_width)
                if function_form is None:
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

                    global_var.grid_cache.g_func = decorator(baseline_exp_function)
                else:
                    global_var.grid_cache.g_func = decorator(function_form)

            except NameError:
                raise NameError('Cache for grids has not been initilized yet!')

    def set_domain_properties(self, coordinate_tensors, memory_for_cache, boundary_width: Union[int, list],
                              function_form: Callable = None, prune_domain: bool = False,
                              pivotal_tensor_label=None, pruner=None, threshold: float = 1e-5,
                              division_fractions: int = 3, rectangular: bool = True):
        """
        Sets the properties of the domain to be processed, including boundary widths, a test function, and whether to prune the domain. This configuration is crucial for tailoring the search space and improving the efficiency of the equation discovery process by focusing on dynamically relevant regions.
        
                Parameters
                ----------
                coordinate_tensors : list|np.ndarrays, optional
                    Values of the coordinates on the grid nodes with studied functions values. In case of 1D-problem,
                    that will be ``numpy.array``, while the parameter for higher dimensionality problems can be set from
                    ``numpy.meshgrid`` function.
                memory_for_cache : int
                    Allowed amount of memory (in percentage) for data storage.
                boundary_width : int|list
                    The number of unaccountable elements at the edges of the domain.
                function_form : callable, optional
                    Testing function connected to the weak derivative notion, the default value is None, that 
                    corresponds with the product of normalized inverse square functions of the coordinates, 
                    centered at the middle of the domain.
                prune_domain : bool
                    Flag, enabling area cropping by removing subdomains with constant values, default - False.
                pivotal_tensor_label : np.ndarray
                    Pattern that guides the domain pruning, the default is None.
                pruner : DomainPruner
                    Object for selecting domain region, the default is None.
                threshold : float, optional
                    The boundary at which values are considered zero, the default is 1e-5.
                division_fractions : int, optional
                    Number of fraction for each axis (if this is integer than all axis are dividing by 
                    same fractions), the default is 3.
                rectangular : bool, optional
                    Flag indecating that crop subdomains are rectangle, default - True.
        
                Returns
                -------
                None.
        """
        self._create_caches(coordinate_tensors=coordinate_tensors, memory_for_cache=memory_for_cache)
        if prune_domain:
            self.domain_pruning(pivotal_tensor_label, pruner, threshold, division_fractions, rectangular)
        self.set_boundaries(boundary_width)
        self._upload_g_func(function_form)

    def set_preprocessor(self, preprocessor_pipeline: PreprocessingPipe = None,
                         default_preprocessor_type: str = 'poly', preprocessor_kwargs: dict = {}):
        """
        Specifies the preprocessor to be used for data preparation and derivative calculation. 
        The preprocessor smooths the input data and computes derivatives, which are essential 
        for constructing candidate equation terms.
        
        Args:
            preprocessor_pipeline (PreprocessingPipe, optional): A custom pipeline of preprocessing 
                operators. If provided, this pipeline will be used directly. Defaults to None.
            default_preprocessor_type (str, optional): A key to select a pre-defined preprocessing 
                pipeline. Options include 'poly' (Savitsky-Golay filtering), 'ANN' (neural network 
                approximation with finite-difference differentiation), 'spectral' (spectral differentiation),
                and 'FD' (finite-difference differentiation). Defaults to 'poly'.
            preprocessor_kwargs (dict, optional): Keyword arguments to configure the selected 
                preprocessor. If not provided, default parameters for the chosen preprocessor type 
                will be used. Defaults to {}.
        
        Returns:
            None
        """
        if preprocessor_pipeline is None:
            setup = PreprocessorSetup()
            builder = ConcretePrepBuilder()
            setup.builder = builder

            if default_preprocessor_type == 'ANN':
                setup.build_ANN_preprocessing(**preprocessor_kwargs)
            elif default_preprocessor_type == 'poly':
                setup.build_poly_diff_preprocessing(**preprocessor_kwargs)
            elif default_preprocessor_type == 'spectral':
                setup.build_spectral_preprocessing(**preprocessor_kwargs)
            elif default_preprocessor_type == 'FD':
                setup.build_FD_preprocessing(**preprocessor_kwargs)
            else:
                raise NotImplementedError('Incorrect default preprocessor type. Only ANN, spectral or poly are allowed.')
            preprocessor_pipeline = setup.builder.prep_pipeline

        if 'max_order' not in preprocessor_pipeline.deriv_calculator_kwargs.keys():
            preprocessor_pipeline.deriv_calculator_kwargs['max_order'] = None

        self.preprocessor_set = True
        self.preprocessor_pipeline = preprocessor_pipeline

    def create_pool(self, data: Union[np.ndarray, list, tuple], variable_names=['u',],
                    derivs=None, max_deriv_order=1, additional_tokens=[],
                    data_fun_pow: int = 1, deriv_fun_pow: int = 1, grid: list = None,
                    data_nn: torch.nn.Sequential = None, fourier_layers: bool = True,
                    fourier_params: dict = {'L' : [4,], 'M' : [3,]}):
        """
        Create a pool of token families representing elementary functions and their derivatives.
        
                This pool serves as the building block for constructing candidate equations, allowing the search algorithm to explore different combinations of functions and derivatives to identify the governing equation.
        
                Args:
                    data (np.ndarray | list of np.ndarrays | tuple of np.ndarrays): Input data as a single array or a list/tuple of arrays, where each array represents a different variable.
                    variable_names (list, optional): Names of the variables corresponding to the data arrays. Defaults to ['u'].
                    derivs (list, optional): Pre-computed derivatives for each variable. If None, derivatives are computed internally. Defaults to None.
                    max_deriv_order (int, optional): Maximum order of derivatives to compute. Defaults to 1.
                    additional_tokens (list, optional): List of additional token families or prepared tokens to include in the pool. Defaults to [].
                    data_fun_pow (int, optional): Maximum power of the input data to include in the pool. Defaults to 1.
                    deriv_fun_pow (int, optional): Maximum power of the derivatives to include in the pool. Defaults to 1.
                    grid (list, optional): Grid on which the data is defined. If None, the global grid cache is used. Defaults to None.
                    data_nn (torch.nn.Sequential, optional): Pre-trained neural network for data representation. Defaults to None.
                    fourier_layers (bool, optional): Whether to use Fourier layers in the neural network. Defaults to True.
                    fourier_params (dict, optional): Parameters for the Fourier layers. Defaults to {'L' : [4,], 'M' : [3,]}.
        
                Returns:
                    None
        """
        grid = grid if grid is not None else global_var.grid_cache.get_all()[1]

        self.pool_params = {'variable_names' : variable_names, 'max_deriv_order' : max_deriv_order,
                            'additional_tokens' : [family.token_family.ftype for family in additional_tokens]}

        if isinstance(data, np.ndarray):
            data = [data,]

        if derivs is None:
            if len(data) != len(variable_names):
                msg = f'Mismatching nums of data tensors {len(data)} and the names of the variables { len(variable_names)}'
                raise ValueError(msg)
        else:
            if not (len(data) == len(variable_names) == len(derivs)):
                raise ValueError('Mismatching lengths of data tensors, names of the variables and passed derivatives')

        if not self.preprocessor_set:
            self.set_preprocessor()

        data_tokens = []
        if self._mode_info['solver_fitness']: 
            base_derivs = []
            
        for data_elem_idx, data_tensor in enumerate(data):
            entry = InputDataEntry(var_name=variable_names[data_elem_idx], var_idx=data_elem_idx,
                                   data_tensor=data_tensor)
            derivs_tensor = derivs[data_elem_idx] if derivs is not None else None
            entry.set_derivatives(preprocesser=self.preprocessor_pipeline, deriv_tensors=derivs_tensor,
                                  grid=grid, max_order=max_deriv_order)
            entry.use_global_cache()

            self.save_derivatives(variable=variable_names[data_elem_idx], deriv=entry.derivatives)  
            entry.create_derivs_family(max_deriv_power=deriv_fun_pow)
            entry.create_polynomial_family(max_power=data_fun_pow)
            if self._mode_info['solver_fitness']:
                base_derivs.extend(entry.matched_derivs(max_order = 2)) # TODO: add setup of Sobolev learning order
                
            data_tokens.extend(entry.get_families())

        if self._mode_info['solver_fitness']:
            if data_nn is not None:
                print('Using pre-trained ANN')
                global_var.reset_data_repr_nn(data = data, derivs = base_derivs, train = False, 
                                              grids = grid, predefined_ann = data_nn, device = self._device)
            else:
                epochs_max = 1e4
                global_var.reset_data_repr_nn(data = data, derivs = base_derivs, epochs_max=epochs_max,
                                              grids = grid, predefined_ann = None, device = self._device, 
                                              use_fourier = fourier_layers, fourier_params = fourier_params)

        if isinstance(additional_tokens, list):
            if not all([isinstance(tf, (TokenFamily, PreparedTokens)) for tf in additional_tokens]):
                raise TypeError(f'Incorrect type of additional tokens: expected list or TokenFamily/Prepared_tokens - obj, instead got list of {type(additional_tokens[0])}')
        elif isinstance(additional_tokens, (TokenFamily, PreparedTokens)):
            additional_tokens = [additional_tokens,]
        else:
            print(isinstance(additional_tokens, PreparedTokens))
            raise TypeError(f'Incorrect type of additional tokens: expected list or TokenFamily/Prepared_tokens - obj, instead got {type(additional_tokens)}')
        self.pool = TFPool(data_tokens + [tf if isinstance(tf, TokenFamily) else tf.token_family
                                          for tf in additional_tokens])
        print(f'The cardinality of defined token pool is {self.pool.families_cardinality()}')
        print(f'Among them, the pool contains {self.pool.families_cardinality(meaningful_only=True)}')
        for family in self.pool.families:
            family.chech_constancy()
        
    def save_derivatives(self, variable:str, deriv:np.ndarray):
        """
        Passes the computed derivatives of a variable to the internal storage. These derivatives are essential components for constructing candidate equations.
        
                Args:
                    variable (str): Key identifying the variable for which the derivatives are being stored.
                    deriv (np.ndarray): Array of derivatives. It should be shaped as (n, m), where n represents the number of derivative terms (e.g., first and second-order derivatives along different axes), and m is the number of data points.
        
                Returns:
                    None
        """
        try:
            self._derivatives
        except AttributeError:
            self._derivatives = {}
        self._derivatives[variable] = deriv

    @property
    def saved_derivaties(self):
        """
        Return the precomputed derivatives of the input data.
        
                These derivatives are essential for constructing candidate equations. If they haven't been computed yet, a warning is issued, as equation discovery relies on these values.
        
                Args:
                    self: The object instance.
        
                Returns:
                    The precomputed derivatives, or None if they haven't been calculated.
        """
        try:
            return self._derivatives
        except AttributeError:
            print('Trying to get derivatives before their calculation. Call EPDESearch.create_pool() to calculate derivatives')
            return None

    def fit(self, data: Union[np.ndarray, list, tuple] = None, equation_terms_max_number=6,
            equation_factors_max_number=1, variable_names=['u',], eq_sparsity_interval=(1e-4, 2.5), 
            derivs=None, max_deriv_order=1, additional_tokens = None, data_fun_pow: int = 1, deriv_fun_pow: int = 1,
            optimizer: Union[SimpleOptimizer, MOEADDOptimizer] = None, pool: TFPool = None,
            population: List[SoEq] = None, data_nn = None, 
            fourier_layers: bool = False, fourier_params: dict = {'L' : [4,], 'M' : [3,]}):
        """
        Fit the EPDE search algorithm to identify differential equations that best describe the provided data.
        
                This method orchestrates the search process, leveraging evolutionary algorithms and optimization techniques
                to explore the space of possible equation structures. It initializes the search pool, configures the
                optimizer, and executes the optimization loop to find equations that accurately represent the underlying
                dynamics of the data.
        
                Parameters
                ----------
                data  : np.ndarray | list | tuple, optional
                    Values of modeled variables. If the variable is single (i.e. deriving a single equation),
                    it can be passed as the numpy.ndarray or as the list/tuple with a single element;
                    multiple variables are not supported yet, use older interfaces. Default value is None, but it 
                    shall be used only for retraining, when the pool argument is passed.
                equation_terms_max_number  : int, optional
                    The maximum number of terms, present in the derived equations, the default is 6.
                equation_factors_max_number : int, optional
                    The maximum number of factors (token functions; real-valued coefficients are not counted here),
                    present in terms of the equaton, the default is 1.
                variable_names : list | str, optional
                    Names of the independent variables, passed into search mechanism. Length of the list must correspond
                    to the number of np.ndarrays, sent with in ``data`` parameter. In case of system of differential equation discovery, 
                    all variables shall be named here, default - ``['u',]``, representing a single variable *u*.
                eq_sparsity_interval : tuple, optional
                    The left and right boundaries of interval with sparse regression values. Undirectly influences the 
                    number of active terms in the equation, the default is ``(1e-4, 2.5)``.
                derivs : list or list of lists of np.ndarrays, optional
                    Pre-computed values of derivatives. If ``None`` is passed, the derivatives are calculated in the
                    method. Recommended to use, if the computations of derivatives take too long. For further information
                    about using data, prepared in advance, check ``epde.preprocessing.derivatives.preprocess_derivatives`` 
                    function, default - None.
                max_deriv_order : int | list | tuple, optional
                    Highest order of calculated derivatives, the default is 1.
                additional_tokens : list of TokenFamily or Prepared_tokens, optional
                    Additional tokens, that would be used to construct the equations among the main variables and their
                    derivatives. Objects of this list must be of type ``epde.interface.token_family.TokenFamily`` or
                    of ``epde.interface.prepared_tokens.Prepared_tokens`` subclasses types. The default is None.
                data_fun_pow : int, optional
                    Maximum power of token, the default is 1.
                optimizer : SimpleOptimizer | MOEADDOptimizer, optional
                    Pre-defined optimizer, that will be used during evolution. Shall correspond with the mode 
                    (single- and multiobjective). The default is None, matching no use of pre-defined optimizer.
                pool : TFPool, optional
                    Pool of tokens, that can be explicitly passed. The default is None, matching no use of passed pool.
                population : List[SoEq], optional
                    Population of candidate equatons, that can be optionally passed in explicit form. The type of objects
                    must match the optimization algorithm: epde.optimizers.single_criterion.optimizer.Population for 
                    single-objective mode and epde.optimizers.moeadd.moeadd.ParetoLevels for multiobjective optimization.
                    The default is None, specifing no passed population.
                data_nn : TYPE, optional
                    DESCRIPTION. The default is None.
                fourier_layers : bool, optional
                    DESCRIPTION. The default is False.
                fourier_params : dict, optional
                    DESCRIPTION. The default is {'L' : [4,], 'M' : [3,]}.
        
                Returns
                -------
                None.
                    The method modifies the internal state of the EPDE search object, storing the identified equations.
        """
        # TODO: ADD EXPLICITLY SENT POPULATION PROCESSING
        if additional_tokens is None:
            additional_tokens = []        
        cur_params = {'variable_names' : variable_names, 'max_deriv_order' : max_deriv_order,
                      'additional_tokens' : [family.token_family.ftype for family in additional_tokens]}

        if pool is None:
            if self.pool == None or self.pool_params != cur_params:
                if data is None:
                    raise ValueError('Data has to be specified beforehand or passed in fit as an argument.')
                self.create_pool(data = data, variable_names=variable_names, 
                                 derivs=derivs, max_deriv_order=max_deriv_order, 
                                 additional_tokens=additional_tokens, 
                                 data_fun_pow = data_fun_pow, deriv_fun_pow = deriv_fun_pow, 
                                 data_nn = data_nn, fourier_layers=fourier_layers, fourier_params=fourier_params)
        else:
            self.pool = pool; self.pool_params = cur_params

        self.optimizer_init_params['population_instruct'] = {"pool": self.pool,
                                                             "terms_number": equation_terms_max_number,
                                                             "max_factors_in_term": equation_factors_max_number,
                                                             "sparsity_interval": eq_sparsity_interval,
                                                             "use_pic": self._use_pic}
        
        if optimizer is None:
            self.optimizer = self._create_optimizer(self.multiobjective_mode, self.optimizer_init_params, 
                                                    self.director, population, self._use_pic)
        else:
            self.optimizer = optimizer
            
        self.optimizer.optimize(**self.optimizer_exec_params)
        
        print('The optimization has been conducted.')
        self.search_conducted = True



    @staticmethod
    def _create_optimizer(multiobjective_mode: bool, optimizer_init_params: dict,
                          opt_strategy_director: OptimizationPatternDirector,
                          population: List[SoEq] = None, use_pic: bool = False):
        """
        Creates and configures an optimizer instance tailored for equation discovery.
        
                This method selects and configures an optimizer based on whether the search
                involves multiple objectives (e.g., accuracy and complexity) or a single
                objective. It then sets up the optimization strategy to guide the search
                for the best equation structure.
        
                Args:
                    multiobjective_mode: A boolean indicating whether to use multi-objective optimization.
                    optimizer_init_params: A dictionary containing initialization parameters for the optimizer.
                    opt_strategy_director: An `OptimizationPatternDirector` instance to set the optimization strategy.
                    population: A list of `SoEq` objects representing the initial population (optional).
                    use_pic: A boolean indicating whether to use a specific configuration for best solution values.
        
                Returns:
                    The created and configured optimizer instance.
        
                Why:
                    This method is responsible for setting up the optimization process, which is a crucial step in discovering
                    differential equations from data. The choice of optimizer and its configuration directly impact the
                    effectiveness and efficiency of the equation search.
        """
        if multiobjective_mode:
            optimizer_init_params['passed_population'] = population
            optimizer = MOEADDOptimizer(**optimizer_init_params)

            # if best_sol_vals is None:
            best_sol_vals = [0., 0.] if use_pic else [0., 1.]
            # best_sol_vals = [0., 0.] if use_pic else [0., 1.]

            same_obj_count = sum([1 for token_family in optimizer_init_params['population_instruct']['pool'].families
                                  if token_family.status['demands_equation']])
            best_obj = np.concatenate([np.full(same_obj_count, fill_value = fval) for fval in best_sol_vals])
            print('best_obj', len(best_obj))
            optimizer.pass_best_objectives(*best_obj)
        else:
            optimizer_init_params['passed_population'] = population
            optimizer = SimpleOptimizer(**optimizer_init_params)
        
        optimizer.set_strategy(opt_strategy_director)        
        return optimizer

    @property
    def _resulting_population(self):
        """
        Returns the population of equation candidates after the evolutionary search process.
        
                This population represents the set of equations that the algorithm has identified as potentially
                describing the underlying dynamics of the system.  The structure of the returned list depends
                on whether the search was conducted in multiobjective mode.
        
                Returns:
                    list: The resulting population. If in multiobjective mode, returns the Pareto levels,
                        representing a set of non-dominated solutions balancing multiple objectives.
                        Otherwise, returns the population from the optimizer, representing the final
                        set of equation candidates after the search.
        
                Raises:
                    AttributeError: If the search has not been conducted yet (``self.fit`` method not called).
                        The population is only available after the evolutionary search has been completed.
        """
        if not self.search_conducted:
            raise AttributeError('Pareto set of the best equations has not been discovered. Use ``self.fit`` method.')
        if self.multiobjective_mode:
            return self.optimizer.pareto_levels.levels
        else:
            return self.optimizer.population.population
    
    def equations(self, only_print : bool = True, only_str = False, num = 1):
        """
        Method for retrieving or displaying the discovered differential equations.
        
                This method provides access to the final set of equations identified by the algorithm. 
                It allows the user to either print these equations for immediate inspection or retrieve them 
                as a data structure for further analysis or use in other parts of a workflow. The number of 
                equations returned or printed can be controlled. This is useful to inspect the best solutions 
                found by the algorithm.
        
                Args:
                    only_print (bool, optional): If True, the equations are printed to the console. 
                        If False, the equations are returned as a list. Defaults to True.
                    only_str (bool, optional): If True, returns only string representation of equations. Defaults to False.
                    num (int, optional): The maximum number of equations to return or print. Defaults to 1.
        
                Returns:
                    None: If `only_print` is True.
                    list: A list of `num` best equations found by the algorithm. If `only_str` is True, returns list of strings.
        """
        if self.multiobjective_mode:
            if only_print:
                for idx in range(min(num, len(self._resulting_population))):
                    print('\n')
                    print(f'{idx}-th non-dominated level')    
                    print('\n')                
                    [print(f'{solution.text_form} , with objective function values of {solution.obj_fun} \n')  
                    for solution in self._resulting_population[idx]]
            else:
                if only_str:
                    eqs = []
                    for idx in range(min(num, len(self._resulting_population))):
                        eqs.append([solution.text_form for solution in self._resulting_population[idx]])
                    return eqs
                else:
                    return self._resulting_population[:num]
        else:
            if only_print:
                [print(f'{solution.text_form} , with objective function values of {solution.obj_fun} \n')  
                 for solution in self._resulting_population[:num]]
            else:
                if only_str:
                    return [solution.text_form for solution in self._resulting_population[:num]]
                else:
                    return self._resulting_population[:num]

    def solver_forms(self, grids: list = None, num: int = 1):
        """
        Converts the discovered equation systems into a solver-compatible format.
        
                This method prepares the identified equation systems for numerical solution by transforming them into a format readily usable by numerical solvers.  This involves converting the symbolic representation of the equations into a form that can be efficiently evaluated. The number of returned systems is controlled by the `num` parameter.
        
                Args:
                    grids (list, optional): Spatial and temporal grids on which the solution is to be evaluated. Defaults to None.
                    num (int, optional): The number of top-performing equation systems to convert. Defaults to 1.
        
                Returns:
                    list: A list of equation systems, each formatted for use with a numerical solver.  If multi-objective mode is enabled, the list contains sub-lists, each representing a Pareto-optimal set of systems.
        """
        forms = []
        if self.multiobjective_mode:
            for level in self._resulting_population[:min(num, len(self._resulting_population))]:
                temp = []
                for sys in level: #self.resulting_population[idx]:
                    temp.append(SystemSolverInterface(sys, device=self._device).form(grids=grids))
                forms.append(temp)
        else:
            for sys in self._resulting_population[:min(num, len(self._resulting_population))]:
                forms.append(SystemSolverInterface(sys, device=self._device).form(grids=grids))
        return forms

    @property
    def cache(self):
        """
        Returns the cached grid and tensor to avoid redundant computations.
        
                The grid cache stores previously computed grids, and the tensor cache
                stores corresponding tensor representations. If a grid has already been
                computed and stored, this method retrieves it along with its tensor,
                preventing the need for recomputation. This optimization is crucial
                for improving the efficiency of the equation discovery process.
        
                Args:
                    self: The object instance.
        
                Returns:
                    Tuple[Any, Any]: A tuple containing the grid cache and the tensor cache.
                    Returns (global_var.grid_cache, global_var.tensor_cache) if
                    global_var.grid_cache is not None, otherwise (None, global_var.tensor_cache).
        """
        if global_var.grid_cache is not None:
            return global_var.grid_cache, global_var.tensor_cache
        else:
            return None, global_var.tensor_cache

    def get_equations_by_complexity(self, complexity : Union[float, list]):
        """
        Retrieve equations that strike a balance between accuracy and simplicity, as quantified by their complexity. This is particularly useful for exploring the trade-off between model fit and parsimony when analyzing the solution space.
        
                Args:
                    complexity (float | list of floats): The desired complexity level(s) for the equations.  For systems of equations, provide a list of complexity values corresponding to each equation in the system.
        
                Returns:
                    list of ``epde.structure.main_structures.SoEq objects``: A list of equation objects that match the specified complexity criteria. These equations represent potential solutions that balance model fit and simplicity.
        """
        return self.optimizer.pareto_levels.get_by_complexity(complexity)

    def predict(self, system : SoEq, boundary_conditions: BoundaryConditions = None, grid : list = None, data = None,
                system_file: str = None, mode: str = 'NN', compiling_params: dict = {}, optimizer_params: dict = {},
                cache_params: dict = {}, early_stopping_params: dict = {}, plotting_params: dict = {}, 
                training_params: dict = {}, use_cache: bool = False, use_fourier: bool = False, 
                fourier_params: dict = None, net = None, use_adaptive_lambdas: bool = False):
        """
        Predicts the state by solving the discovered equation or system of equations. This leverages a solver implementation adapted from https://github.com/ITMO-NSS-team/torch_DE_solver to find a numerical solution. This is a crucial step in the EPDE workflow, allowing us to validate the discovered equations against the original data and generate predictions.
        
                Parameters
                ----------
                system : SoEq
                    Object containing the system (or a single equation as a system of one equation) to solve.
                boundary_conditions : BoundaryConditions, optional
                    Boundary condition objects, should match the order of differential equations due to no internal checks.
                    Over/underdefined solution can happen if the number of conditions is incorrect. The default value is None,
                    matching automatic construction of the required Dirichlet BC from data.
                grid : list, optional
                    Grids, defining Cartesian coordinates, on which the equations will be solved. The default is None, specifying
                    the use of grids stored in cache during equation learning.
                data : TYPE, optional
                    Dataset, from which the boundary conditions can be automatically created. The default is None, making use of
                    the training datasets stored in cache during equation training.
                system_file : str, optional
                    Filename for the pickled equation/system of equations. If passed, **system** can be None. The default is None, meaning no equation.
                mode : str, optional
                    Key, defining used method of the automatic DE solution. Supported methods: 'NN', 'mat' and 'autodiff'. The default is 'NN'.
                compiling_params : dict, optional
                    Parameters for the equation compiling stage. The default is {}.
                optimizer_params : dict, optional
                    Parameters for the optimization algorithm used in the solver. The default is {}.
                cache_params : dict, optional
                    Parameters for caching intermediate results. The default is {}.
                early_stopping_params : dict, optional
                    Parameters for early stopping during the solver's training. The default is {}.
                plotting_params : dict, optional
                    Parameters for plotting the solution. The default is {}.
                training_params : dict, optional
                    Parameters for the solver's training process. The default is {}.
                use_cache : bool, optional
                    Flag indicating whether to use cached results. The default is False.
                use_fourier : bool, optional
                    Flag indicating whether to use Fourier features. The default is False.
                fourier_params : dict, optional
                    Parameters for the Fourier features. The default is None.
                net : torch.nn.Module, optional
                    A pre-trained neural network to use as a solver. The default is None.
                use_adaptive_lambdas : bool, optional
                    Flag indicating whether to use adaptive lambdas. The default is False.
        
                Raises
                ------
                ValueError
                    If no system is provided, either directly or via a file.
        
                Returns
                -------
                solution_model : torch.nn.Module
                    The trained model that represents the solution to the discovered equation.
        """
        
        if system is not None:
            print('Using explicitly sent system of equations.')
        elif system_file is not None:
            assert '.pickle' in system_file
            print('Loading equation from pickled file.')

            system = pickle.load(file=system_file)
        else:
            raise ValueError('Missing system, that was not passed in any form.')
        
        if grid is None:
            grid = global_var.grid_cache.get_all()[1]
        
        adapter = SolverAdapter(net = net, use_cache = use_cache) # var_number = len(system.vars_to_describe), 
        
        # Setting various adapater parameters
        adapter.set_compiling_params(**compiling_params)
        
        adapter.set_optimizer_params(**optimizer_params)
        
        adapter.set_cache_params(**cache_params)
        
        adapter.set_early_stopping_params(**early_stopping_params)
        
        adapter.set_plotting_params(**plotting_params)
        
        adapter.set_training_params(**training_params)
        
        adapter.change_parameter('mode', mode, param_dict_key = 'compiling_params')
        print(f'grid.shape is {grid[0].shape}')
        solution_model = adapter.solve_epde_system(system = system, grids = grid, data = data, 
                                                   boundary_conditions = boundary_conditions, 
                                                   mode = mode, use_cache = use_cache, 
                                                   use_fourier = use_fourier, fourier_params = fourier_params,
                                                   use_adaptive_lambdas = use_adaptive_lambdas)
        return solution_model

    def visualize_solutions(self, dimensions:list = [0, 1], **visulaizer_kwargs) -> None:
        '''
        Plot the discovered equations using matplotlib.
        
        By default, the method plots only the Pareto-optimal equations from the population.
        Annotations of the candidate equations are rendered using LaTeX. This visualization
        helps in understanding the trade-offs between different equation structures and their
        fit to the data, allowing for informed selection of the most appropriate model.
        
        Args:
            dimensions (list, optional): The dimensions to plot. Defaults to [0, 1].
            **visulaizer_kwargs: Keyword arguments to pass to the plotting function.
        
        Returns:
            None
        '''
        if self.multiobjective_mode:
            self.optimizer.plot_pareto(dimensions=dimensions, **visulaizer_kwargs)
        else:
            raise NotImplementedError('Solution visualization is implemented only for multiobjective mode.')
            
            
class ExperimentCombiner(object):
    """
    Combines results from multiple experiments to find the best solutions.
    
        Class Methods:
        - create_best:
    """

    def __init__(self, candidates: Union[ParetoLevels, List[SoEq], List[ParetoLevels]]):
        """
        Initializes the ComplexityHandler, preparing complexity data for equation discovery.
        
                This method takes a list of candidate solutions, extracts their complexities,
                and organizes them to facilitate efficient searching of the equation space.
                The extracted complexities are stored alongside their corresponding candidate
                solutions, and then sorted to enable efficient comparison during the
                equation discovery process.
        
                Args:
                    candidates: A list of candidate solutions, which can be either
                        ParetoLevels objects, SoEq objects, or a list of ParetoLevels objects.
        
                Returns:
                    None.
        
                Class Fields:
                    complexity_matched (List[Tuple[Any, List[Any]]]): A list of tuples, where each tuple contains
                        a candidate solution and a list of its complexities.
                    ordered_complexities (List[List[Any]]): A list of lists, where each inner list contains
                        the sorted unique complexities for a specific complexity metric.
        
                WHY:
                This initialization is crucial for efficiently exploring the equation space. By
                extracting and organizing complexities, the system can effectively compare
                different candidate solutions based on their complexity profiles, guiding the
                search towards simpler and more accurate equations.
        """
        self.complexity_matched = self.get_complexities(candidates)
        complexity_sets = [set() for i in self.complexity_matched[0][1]]
        for eq, complexity in self.complexity_matched:
            for idx, compl in enumerate(complexity):
                complexity_sets[idx].add(compl)
        self.ordered_complexities = [sorted(compl_set) for compl_set in complexity_sets]
        
    @singledispatchmethod
    def get_complexities(self, candidates) -> list:
        """
        Calculate the complexity of each candidate equation.
        
        This method assesses the complexity of candidate equations,
        providing a basis for balancing model accuracy with parsimony
        during the equation discovery process. More complex equations
        might fit the data better but are also more prone to overfitting.
        
        Args:
            candidates: A list of candidate equations.
        
        Returns:
            A list of complexities, where each element corresponds to the
            complexity of the respective candidate equation.
        
        Raises:
            NotImplementedError: Always raised, indicating that the method
                should be implemented in a subclass.
        """
        raise NotImplementedError('Incorrect type of equations to parse')

    @get_complexities.register
    def _(self, candidates: list) -> list:
        """
        Processes a list of candidate solutions to extract complexity information.
        
                This method prepares candidate solutions for further analysis by extracting
                relevant complexity measures. The type of the candidate solutions determines
                how the complexity is assessed. It supports `ParetoLevels` and `SoEq` types.
                The extracted complexity information is used to evaluate and compare different
                equation candidates during the evolutionary search process.
        
                Args:
                  candidates: A list of candidate solutions. The type of the elements
                    determines the processing logic.
        
                Returns:
                  A list containing complexity information extracted from the candidates.
                  The format of the returned list depends on the type of the candidates:
                    - If candidates are of type `ParetoLevels`, returns a list of complexities.
                    - If candidates are of type `SoEq`, returns a list of tuples, where each
                      tuple contains a candidate and its complexity objectives.
        
                Raises:
                    ValueError: If the type of the candidate is not supported.
        """
        if isinstance(candidates[0], ParetoLevels):
            return reduce(lambda x, y: x.append(y), [self.get_complexities(pareto_level) for 
                                                    pareto_level in candidates], [])
        elif isinstance(candidates[0], SoEq):
            # Here we assume, that the number of objectives is even, having quality 
            # and complexity for each equation
            compl_objs_num = int(candidates[0].obj_fun.size/2)
            # print(compl_objs_num)
            return [(candidate, candidate.obj_fun[-compl_objs_num:]) for candidate in candidates]
        else:
            raise ValueError(f'Incorrect type of the equation, got {type(candidates[0])}')
        
    @get_complexities.register
    def _(self, candidates: ParetoLevels) -> list:
        """
        Combines complexities from all Pareto levels into a single list.
        
                This method aggregates the complexities associated with each Pareto level
                present in the provided `candidates` object. By consolidating these
                complexities, the method facilitates a comprehensive evaluation of the
                candidate solutions discovered during the equation search process. This
                aggregation is crucial for assessing the overall complexity of the
                identified equations and selecting the most parsimonious models.
        
                Args:
                    candidates (ParetoLevels): A ParetoLevels object containing the Pareto levels to process.
        
                Returns:
                    list: A list containing the combined complexities from all Pareto levels.
        """
        eqs = reduce(lambda x, y: x.append(y), [self.get_complexities(level)  for 
                                                level in candidates.levels], [])
        return eqs
        
    def create_best_for_complexity(self, complexity: tuple, pool: TFPool):
        """
        Creates a compound equation by selecting the best candidate equations for each dimension's complexity level.
        
                This method aims to construct an equation that accurately represents the system by choosing the most suitable equation for each dimension based on the provided complexity constraints. It iterates through the specified complexity levels, identifies the best candidate equation for each, and combines them into a single, comprehensive equation. This approach allows the framework to adapt to varying levels of complexity across different dimensions of the problem.
        
                Args:
                    complexity: A tuple representing the desired complexity levels for each dimension. A value of `None` indicates that the best equation should be selected regardless of complexity for that dimension.
                    pool: A TFPool object (not used in the provided code).
        
                Returns:
                    The compound equation created from the best candidate equations, representing the discovered relationship between variables.
        """
        vars_to_describe = self.complexity_matched[0][0].vars_to_describe # Get dependent variables
        
        best_eqs = []
        for idx, elem in enumerate(complexity):
            if elem is not None:
                relaxed_compl = [None,]*len(complexity)
                relaxed_compl[idx] = elem
                candidates = [candidate for candidate, _ in self.complexity_matched 
                             if candidate.matches_complexitiy(relaxed_compl)]
                best_candidate = sorted(candidates, key=lambda x: x.obj_fun[idx])[0]
                # best_eqs.append(best_candidate.vals[vars_to_describe[idx]])
            else:
                best_candidate = sorted([candidate for candidate, _ in self.complexity_matched], 
                                        key=lambda x: x.obj_fun[idx])[0]
            best_eqs.append(best_candidate.vals[vars_to_describe[idx]])
        compound_equation = deepcopy(self.complexity_matched[0][0])
        compound_equation.create(passed_equations = best_eqs)
        return compound_equation
    
    def create_best(self, pool: TFPool):
        """
        Creates the best program variant based on the highest complexities.
        
        It retrieves the highest complexity value for each program in the
        ordered complexities list and then uses these values to create the
        best program variant. This is done to prioritize more complex and potentially more accurate models
        identified during the equation discovery process.
        
        Args:
            pool: The TFPool object used for program creation.
        
        Returns:
            The best program variant created based on the highest complexities.
        """
        best_qualities_compl = [complexities[-1] for complexities in self.ordered_complexities]
        return self.create_best_for_complexity(best_qualities_compl, pool)
    
class EpdeMultisample(EpdeSearch):
    """
    Multisample EPDE class for equation discovery from multiple datasets.
    
            This class extends the EPDE framework to handle multiple data samples,
            allowing for equation discovery across different datasets. It provides
            methods for setting domain properties, uploading test functions, setting
            data samples, and fitting the EPDE search algorithm.
    
            Class Methods:
            - __init__
            - set_domain_properties
            - _upload_g_func
            - set_samples
            - fit
    
            Attributes:
            - use_default_strategy: True (base and recommended value), if the default evolutionary strategy will be used, 
                        False if the user-defined strategy will be passed further. Otherwise, the search will 
                        not be conducted.
            - time_axis: Indicator of time axis in data and grids. Used in normalization for regressions.
            - function_form: Auxilary function, used in the weak derivative definition. Default function is negative square function 
                        with maximum values in the center of the domain.
            - boundary: Boundary width for the domain. Boundary points will be ignored for the purposes of equation discovery
            - use_solver: Allow use of the automaic partial differential solver to evaluate fitness of the candidate solutions.
            - dimensionality: Dimensionality of the problem. ! Currently you should pass value, reduced by one !
            - verbose_params: Description, of algorithm details, that will be demonstrated to the user. Usual
            - memory_for_cache: Rough estimation of the memory, which can be used for cache of pre-evaluated tensors during the equation
            - prune_domain: If ``True``, subdomains with no dynamics will be pruned from data. Default value: ``False``.
            - pivotal_tensor_label: Indicator, according to which token data will be pruned. Default value - ``'du/dt'``, where 
                        ``t`` is selected as a time axis from ``time_axis`` parameter.
            - pruner: Pruner object, which will remove subdomains with no dynamics i.e. with derivative 
                        identically equal to zero.
            - threshold: Pruner parameter, indicating the boundary of interval in which the pivotal tensor values are 
                        considered as zeros. Default value: 1e-2
            - division_fractions: Number of subdomains along each axis, defining the division of the domain for pruning.
                        Default value: 3
            - rectangular: A line of subdomains along an axis can be removed if all values inside them are identical to zero.
    """

    def __init__(self, data_samples : List[List], multiobjective_mode: bool = True, 
                 use_default_strategy: bool = True, director=None, 
                 director_params: dict = {'variation_params': {}, 'mutation_params': {},
                                           'pareto_combiner_params': {}, 'pareto_updater_params': {}}, 
                 time_axis: int = 0, function_form=None, boundary: int = 0, 
                 use_solver: bool = False, verbose_params: dict = {'show_iter_idx' : True},
                 memory_for_cache=5, prune_domain: bool = False, 
                 pivotal_tensor_label=None, pruner=None, threshold: float = 1e-2, 
                 division_fractions=3, rectangular: bool = True, params_filename: str = None):
        """
        Initializes the EpdeMultisample class, configuring the equation discovery process across multiple data samples.
        
                This class extends the functionality of the base EPDE framework to handle multiple datasets, 
                allowing for more robust and generalizable equation discovery. It preprocesses and stacks 
                grids from multiple samples to define the problem domain.
        
                Args:
                    data_samples (List[List]): A list of data samples, where each sample is a list containing grids and corresponding data.
                    multiobjective_mode (bool): Whether to use multiobjective optimization. Defaults to True.
                    use_default_strategy (bool): If True, uses the default evolutionary strategy. If False, a user-defined strategy is expected. Defaults to True.
                    director: An optional director object to manage the evolutionary process.
                    director_params (dict): Parameters for the director, including variation, mutation, pareto combiner, and pareto updater settings.
                    time_axis (int): The index of the time axis in the data. Defaults to 0.
                    function_form (callable): An auxiliary function used in the weak derivative definition.
                    boundary (int|tuple/list of integers): The boundary width for the domain; boundary points are ignored during equation discovery. Defaults to 0.
                    use_solver (bool): Whether to use an automatic partial differential equation solver to evaluate candidate solutions. Defaults to False.
                    verbose_params (dict): Parameters controlling the verbosity of the algorithm's output. Defaults to {'show_iter_idx' : True}.
                    memory_for_cache (int|float): An estimate of the memory to use for caching pre-evaluated tensors. Defaults to 5.
                    prune_domain (bool): If True, subdomains with no dynamics are removed from the data. Defaults to False.
                    pivotal_tensor_label (str): The label of the tensor used to identify subdomains with dynamics. Defaults to None.
                    pruner: An optional pruner object to remove subdomains with no dynamics.
                    threshold (float): The threshold for determining when pivotal tensor values are considered zero for pruning. Defaults to 1e-2.
                    division_fractions (int): The number of subdomains along each axis for pruning. Defaults to 3.
                    rectangular (bool): If True, entire lines of subdomains with zero values can be removed. Defaults to True.
                    params_filename (str): Path to the file with custom parameters. Defaults to None.
        
                Returns:
                    None
        """
        super().__init__(multiobjective_mode = multiobjective_mode, use_default_strategy = use_default_strategy, 
                         director = director, director_params = director_params, time_axis = time_axis,
                         define_domain = False, function_form = function_form, boundary = boundary, 
                         use_solver = use_solver, verbose_params = verbose_params,
                         coordinate_tensors = None, memory_for_cache = memory_for_cache, prune_domain = prune_domain, 
                         pivotal_tensor_label = pivotal_tensor_label, pruner = pruner, threshold = threshold, 
                         division_fractions = division_fractions, rectangular = rectangular, 
                         params_filename = params_filename)
        self._memory_for_cache = memory_for_cache
        self._boundary = boundary
        self._function_form = function_form

        grids = [sample[0] for sample in data_samples]
        # print('grids shape is', [(type(subgrid), len(subgrid)) for subgrid in grids])

        subgrids = [list() for var_grid in grids[0]]
        for sample_grids in grids:
            for idx, var_grid in enumerate(sample_grids):
                subgrids[idx].append(var_grid)

        grids_stacked = [np.concatenate(var_grid) for var_grid in subgrids]
        self.set_domain_properties(grids_stacked, self._memory_for_cache, self._boundary, self._function_form)

        global_var.grid_cache.g_func = np.concatenate([self.g_func(grid) for grid in grids])

        # Domain will not be set properly in init, thus a separate initialization is necessary

    def set_domain_properties(self, coordinate_tensors, memory_for_cache, boundary_width: Union[int, list],
                              function_form: Callable = None, prune_domain: bool = False,
                              pivotal_tensor_label=None, pruner=None, threshold: float = 1e-5,
                              division_fractions: int = 3, rectangular: bool = True):
        """
        Sets the properties of the domain for equation discovery.
        
                This method configures the domain by setting boundaries, pruning regions with minimal dynamics,
                and defining a test function. These properties are crucial for focusing the equation search
                on the most relevant areas of the domain and for evaluating candidate equations effectively.
        
                Args:
                    coordinate_tensors (list):
                        List of tensors representing the coordinates of the domain.
                    memory_for_cache (int):
                        Allowed amount of memory (in percentage) for data storage.
                    boundary_width (int | list):
                        The number of unaccountable elements at the edges of the domain.
                    function_form (callable, optional):
                        Testing function connected to the weak derivative notion. The default value is None, which
                        corresponds to the product of normalized inverse square functions of the coordinates,
                        centered at the middle of the domain.
                    prune_domain (bool, optional):
                        Flag enabling area cropping by removing subdomains with constant values. Default is False.
                    pivotal_tensor_label (np.ndarray, optional):
                        Pattern that guides the domain pruning. The default is None.
                    pruner (DomainPruner, optional):
                        Object for selecting domain region. The default is None.
                    threshold (float, optional):
                        The boundary at which values are considered zero. The default is 1e-5.
                    division_fractions (int, optional):
                        Number of fractions for each axis (if this is an integer, then all axes are divided by the
                        same fraction). The default is 3.
                    rectangular (bool, optional):
                        Flag indicating that crop subdomains are rectangular. Default is True.
        
                Returns:
                    None
        """
        # raise NotImplementedError('In ensemble mode the domain is set in `set_samples` method.')
    
        # assert self.coodinate_tensors is not None, 'Coordinate tensors for the sample have to be set beforehand.'
        self._create_caches(coordinate_tensors=coordinate_tensors, memory_for_cache=memory_for_cache)
        if prune_domain:
            self.domain_pruning(pivotal_tensor_label, pruner, threshold, division_fractions, rectangular)
        self.set_boundaries(boundary_width)

        # TODO$
        self._upload_g_func(function_form)

    def _upload_g_func(self, function_form: Union[Callable, np.ndarray, list] = None, boundary_width: int = None):
        """
        Loading a test function for weak derivative calculations.
        
                This function prepares a test function, crucial for evaluating equation fitness using weak derivatives. 
                Unlike single-equation approaches, the test function is stored for later application across multiple equation candidates, 
                allowing for a more comprehensive and efficient search for the best model.
        
                Args:
                    function_form (`callable`, `np.ndarray`, or `list`):
                        The test function itself. If a callable is provided, it's used directly (or a default is created if None). 
                        If an array or list of arrays is given, it's used as the test function directly.
                    boundary_width (`int`, optional):
                        Specifies the width of the boundary region to exclude when evaluating the test function. 
                        Defaults to the boundary width defined in the global grid cache.
        
                Returns:
                    None: The test function is stored internally for later use in equation evaluation.
        """
        boundary_width = boundary_width if boundary_width is not None else global_var.grid_cache.boundary_width
        if isinstance(function_form, (np.ndarray, list)):
            self.g_func = function_form
        else:
            try:
                decorator = BoundaryExclusion(boundary_width=boundary_width)
                if function_form is None:
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

                    self.g_func = decorator(baseline_exp_function)
                else:
                    self.g_func = decorator(function_form)

            except NameError:
                raise NameError('Cache for grids has not been initilized yet!')


    def set_samples(self, data_samples: List[List], sample_derivs: List[List[np.ndarray]] = None, var_names: List[str] = ['u',], 
                    max_deriv_orders: Union[int, list[int]] = 1, additional_tokens: list = [], data_fun_pow: int = 1,
                    deriv_fun_pow: int = 1):
        """
        Sets the data samples and their derivatives to construct a pool of equation terms.
                
                This method prepares the data by extracting grid and data values from the input samples.
                It then organizes this information to create a comprehensive pool of tokens,
                which are the building blocks for constructing candidate differential equations.
                This setup is crucial for the subsequent symbolic regression process, where the
                algorithm searches for the best equation structure that fits the data.
                The method supports both single and multiple dependent variables, accommodating
                various data formats such as NumPy arrays or lists/tuples of NumPy arrays.
                
                Args:
                    data_samples: A list of lists, where each inner list contains the grid
                        and data values for a sample. The data values can be a NumPy array
                        or a list/tuple of NumPy arrays for multiple dependent variables.
                    sample_derivs: A list of lists containing the derivatives of the
                        data samples. Each inner list corresponds to a sample and
                        contains NumPy arrays representing the derivatives. Defaults to None.
                    var_names: A list of strings representing the names of the dependent
                        variables. Defaults to ['u'].
                    max_deriv_orders: An integer or a list of integers specifying the
                        maximum order of derivatives to be considered for each variable.
                        Defaults to 1.
                    additional_tokens: A list of additional tokens to be included in the
                        token pool. Defaults to [].
                    data_fun_pow: An integer representing the power to which the data
                        is raised when creating tokens. Defaults to 1.
                    deriv_fun_pow: An integer representing the power to which the
                        derivatives are raised when creating tokens. Defaults to 1.
                
                Returns:
                    None.
        """
        if isinstance(data_samples[0][1], np.ndarray):
            data_comb = [sample[1] for sample in data_samples]
            print('Samples are np.ndarrays somehow')
        elif isinstance(data_samples[0][1], tuple) or isinstance(data_samples[0][1], list):
            data_comb = []
            assert all([isinstance(sample_var, np.ndarray) for sample_var in data_samples[0][1]]), f'Samples must be passed as \
                a list of multiple numpy ndarrays, if the equations are derived for mutiple dependent variables.'
            print(f'Presumably we have {len(data_samples[0][1])} dependent variables')
            for var_idx in range(len(data_samples[0][1])):
                data_comb.append([sample[1][var_idx] for sample in data_samples])

        grids = [sample[0] for sample in data_samples]
                
        # subgrids = [list() for var_grid in grids[0]]
        # for sample_grids in grids:
        #     for idx, var_grid in sample_grids:
        #         subgrids[idx].append(var_grid)

        # grids_stacked = [np.concatenate(var_grid) for var_grid in subgrids]

        self.create_pool(data = data_comb, variable_names = var_names, derivs = sample_derivs, 
                         max_deriv_order = max_deriv_orders, additional_tokens = additional_tokens,
                         data_fun_pow = data_fun_pow, deriv_fun_pow=deriv_fun_pow, grid = grids) # Implement sample-wise differentiation.

        # for sample in data_samples[1:]:
            # if multi_var_mode:
                # pass
        # TODO: calculated derivatives, combine them into single arrays to correctly create tokens. 
    
        
    # def set_derivatives(self, variable:str, deriv:np.ndarray):
    #     '''
    #     Pass the derivatives of a variable as a np.ndarray.
    
    #     Parameters
    #     ----------
    #     variable : str
    #         Key for the variable to have the derivatives set.
    #     deriv : np.ndarray
    #         Arrays of derivatives. Have to be shaped as (n, m), where n is the number of passed derivatives 
    #         (for example, when you differentiate the dataset once for the first axis, and up to the second order for 
    #          the second, and you have no mixed derivatives, *n = 3*), and m is the number of data points in the domain.

    #     Returns
    #     -------
    #     None.
    #     '''
    #     try:
    #         self._derivatives
    #     except AttributeError:
    #         self._derivatives = {}
    #     self._derivatives[variable] = deriv


    def fit(self, samples: List[Tuple], equation_terms_max_number=6, equation_factors_max_number=1, variable_names=['u',], 
            eq_sparsity_interval=(1e-4, 2.5), derivs=None, max_deriv_order=1, additional_tokens=[], 
            data_fun_pow: int = 1, deriv_fun_pow: int = 1, optimizer: Union[SimpleOptimizer, MOEADDOptimizer] = None, 
            pool: TFPool = None, population: Union[ParetoLevels, Population] = None):
        """
        Fit epde search algorithm to obtain differential equations, describing passed data.
                This method orchestrates the equation discovery process by initializing the search space,
                setting up the optimization algorithm, and running the evolutionary search. It leverages
                the provided data samples and parameter settings to identify candidate equations that
                effectively model the underlying dynamics.
        
                Parameters
                ----------
                samples : List[Tuple]
                    List of data samples, where each sample is a tuple containing the values of the
                    modeled variables.
                equation_terms_max_number : int, optional
                    The maximum number of terms allowed in the derived equations, the default is 6.
                equation_factors_max_number : int, optional
                    The maximum number of factors (token functions; real-valued coefficients are not counted here)
                    allowed in each term of the equation, the default is 1.
                variable_names : list | str, optional
                    Names of the independent variables, passed into search mechanism. Length of the list must correspond
                    to the number of np.ndarrays, sent with in ``data`` parameter. In case of system of differential equation discovery, 
                    all variables shall be named here, default - ``['u',]``, representing a single variable *u*.
                eq_sparsity_interval : tuple, optional
                    The left and right boundaries of interval with sparse regression values. Undirectly influences the 
                    number of active terms in the equation, the default is ``(1e-4, 2.5)``.
                derivs : list or list of lists of np.ndarrays, optional
                    Pre-computed values of derivatives. If ``None`` is passed, the derivatives are calculated in the
                    method. Recommended to use, if the computations of derivatives take too long. For further information
                    about using data, prepared in advance, check ``epde.preprocessing.derivatives.preprocess_derivatives`` 
                    function, default - None.
                max_deriv_order : int | list | tuple, optional
                    Highest order of calculated derivatives, the default is 1.
                additional_tokens : list of TokenFamily or Prepared_tokens, optional
                    Additional tokens, that would be used to construct the equations among the main variables and their
                    derivatives. Objects of this list must be of type ``epde.interface.token_family.TokenFamily`` or
                    of ``epde.interface.prepared_tokens.Prepared_tokens`` subclasses types. The default is None.
                data_fun_pow : int, optional
                    Maximum power of token, the default is 1.
                deriv_fun_pow : int, optional
                    Maximum power of derivative token, the default is 1.
                optimizer : SimpleOptimizer | MOEADDOptimizer, optional
                    Pre-defined optimizer, that will be used during evolution. Shall correspond with the mode 
                    (single- and multiobjective). The default is None, matching no use of pre-defined optimizer.
                pool : TFPool, optional
                    Pool of tokens, that can be explicitly passed. The default is None, matching no use of passed pool.
                population : Population | ParetoLevels, optional
                    Population of candidate equatons, that can be optionally passed in explicit form. The type of objects
                    must match the optimization algorithm: epde.optimizers.single_criterion.optimizer.Population for 
                    single-objective mode and epde.optimizers.moeadd.moeadd.ParetoLevels for multiobjective optimization.
                    The default is None, specifing no passed population.
            
                Returns
                -------
                None.
                    The method does not return any value. The discovered equations and related information
                    are stored within the class instance.
        """
        # TODO: ADD EXPLICITLY SENT POPULATION PROCESSING
        cur_params = {'variable_names' : variable_names, 'max_deriv_order' : max_deriv_order,
                      'additional_tokens' : [family.token_family.ftype for family in additional_tokens]}

        # if pool is None:
        #     if self.pool == None or self.pool_params != cur_params:
        #         if data is None:
        #             raise ValueError('Data has to be specified beforehand or passed in fit as an argument.')
        #         self.create_pool(data = data, variable_names=variable_names, 
        #                          derivs=derivs, max_deriv_order=max_deriv_order, 
        #                          additional_tokens=additional_tokens, 
        #                          data_fun_pow=data_fun_pow)
        # else:
        #     self.pool = pool; self.pool_params = cur_params
        if pool is None:
            self.set_samples(samples, sample_derivs=derivs, var_names = variable_names, max_deriv_orders = max_deriv_order, 
                             additional_tokens = additional_tokens, data_fun_pow = data_fun_pow, deriv_fun_pow=deriv_fun_pow)
        else:
            self.pool = pool; self.pool_params = cur_params

        self.optimizer_init_params['population_instruct'] = {"pool": self.pool, "terms_number": equation_terms_max_number,
                                                             "max_factors_in_term": equation_factors_max_number,
                                                             "sparsity_interval": eq_sparsity_interval}
        
        if optimizer is None:
            self.optimizer = self._create_optimizer(self.multiobjective_mode, self.optimizer_init_params, 
                                                    self.director)
        else:
            self.optimizer = optimizer
            
        self.optimizer.optimize(**self.optimizer_exec_params)
        
        print('The optimization has been conducted.')
        self.search_conducted = True    
