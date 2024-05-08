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
from epde.interface.solver_integration import BoundaryConditions, SolverAdapter, SystemSolverInterface

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
    def __init__(self, var_name: str, data_tensor: Union[List[np.ndarray], np.ndarray]):
        self.var_name = var_name
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

    def use_global_cache(self):
        """
        Method for add calculated derivatives in the cache
        """
        derivs_stacked = prepare_var_tensor(self.data_tensor, self.derivatives, time_axis=global_var.time_axis)

        try:
            upload_simple_tokens(self.names, global_var.tensor_cache, derivs_stacked)
            upload_simple_tokens([self.var_name,], global_var.tensor_cache, [self.data_tensor,])            
            upload_simple_tokens([self.var_name,], global_var.initial_data_cache, [self.data_tensor,])

        except AttributeError:
            raise NameError('Cache has not been declared before tensor addition.')

        global_var.tensor_cache.use_structural()

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
        self._derivs_family.set_evaluator(simple_function_evaluator, [])        

    def create_polynomial_family(self, max_power):
        polynomials = DataPolynomials(self.var_name, max_power = max_power)
        self._polynomial_family = polynomials.token_family

    def get_families(self):
        return [self._polynomial_family, self._derivs_family]

def simple_selector(sorted_neighbors, number_of_neighbors=4):
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
    def __init__(self, multiobjective_mode: bool = True, use_default_strategy: bool = True, director=None, 
                 director_params: dict = {'variation_params': {}, 'mutation_params': {},
                                           'pareto_combiner_params': {}, 'pareto_updater_params': {}}, 
                 time_axis: int = 0, define_domain: bool = True, function_form=None, boundary: int = 0, 
                 use_solver: bool = False, dimensionality: int = 1, verbose_params: dict = {'show_iter_idx' : True}, 
                 coordinate_tensors=None, memory_for_cache=5, prune_domain: bool = False, 
                 pivotal_tensor_label=None, pruner=None, threshold: float = 1e-2, 
                 division_fractions=3, rectangular: bool = True, params_filename: str = None):
        """
        Args:
            use_default_strategy (`bool`): optional
                True (base and recommended value), if the default evolutionary strategy will be used, 
                False if the user-defined strategy will be passed further. Otherwise, the search will 
                not be conducted.  
            time_axis (`int`): optional
                Indicator of time axis in data and grids. Used in normalization for regressions.
            define_domain (`bool`): optional
                Indicator, showing that if the domain will be set in the initialization of the search objects.
                For more details view ``epde_search.set_domain_properties`` method.
            function_form (`callable`): optional
                Auxilary function, used in the weak derivative definition. Default function is negative square function 
                with maximum values in the center of the domain.
            boundary (`int|tuple/list of integers`): optional
                Boundary width for the domain. Boundary points will be ignored for the purposes of equation discovery
            use_solver (`bool`): optional
                Allow use of the automaic partial differential solver to evaluate fitness of the candidate solutions.
            dimensionality (`int`): optional
                Dimensionality of the problem. ! Currently you should pass value, reduced by one !
            verbose_params (`dict`): optional
                Description, of algorithm details, that will be demonstrated to the user. Usual
            coordinate_tensors (`list of np.ndarrays`): optional
                Values of the coordinates on the grid nodes with studied functions values. In case of 1D-problem, 
                that will be ``numpy.array``, while the parameter for higher dimensionality problems can be set from 
                ``numpy.meshgrid`` function. With None, the tensors will be created as ranges with step of 1 between 
                nodes. Defalut value: None.
            memory_for_cache (`int|float`): optional
                Rough estimation of the memory, which can be used for cache of pre-evaluated tensors during the equation
            prune_domain (`bool`): optional
                If ``True``, subdomains with no dynamics will be pruned from data. Default value: ``False``.
            pivotal_tensor_label (`str`): optional
                Indicator, according to which token data will be pruned. Default value - ``'du/dt'``, where 
                ``t`` is selected as a time axis from ``time_axis`` parameter.
            pruner (`object`): optional
                Pruner object, which will remove subdomains with no dynamics i.e. with derivative 
                identically equal to zero.
            threshold (`float`): optional
                Pruner parameter, indicating the boundary of interval in which the pivotal tensor values are 
                considered as zeros. Default value: 1e-2
            division_fractions (`int`): optional
                Number of subdomains along each axis, defining the division of the domain for pruning.
                Default value: 3
            rectangular(`bool`): optional
                A line of subdomains along an axis can be removed if all values inside them are identical to zero.
        """
        self.multiobjective_mode = multiobjective_mode
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

        if multiobjective_mode:
            mode_key = 'multi objective'
        else:
            mode_key = 'single objective'
        
        # Here we initialize a singleton object with evolutionary params. It is used in operators' initialization.
        EvolutionaryParams.reset()
        evo_param = EvolutionaryParams(parameter_file = params_filename, mode = mode_key)

        if use_solver:
            global_var.dimensionality = dimensionality

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
            self.director.use_baseline(params=director_params)
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
        Setting properties for using memory to cache

        Args:
            example_tensor (`ndarray`): referntial tensor to evaluate memory consuption by tensors equation search
            mem_for_cache_frac (`int`): optional
                memory available for cache (in fraction of RAM). The default - None.
            mem_for_cache_abs (`int`): optional
                memory available for cache (in byte). The default - None.

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
        Setting the parameters of the multiobjective evolutionary algorithm. declaration of
        the default values is held in the initialization of EpdeSearch object.

        Args:
            population_size (`int`): optional
                The size of the population of solutions, created during MO - optimization, default 6.
            solution_params (`dict`): optional
                Dictionary, containing additional parameters to be sent into the newly created solutions.
            delta (`float`): optional
                parameter of uniform spacing between the weight vectors; *H = 1 / delta*
                should be integer - a number of divisions along an objective coordinate axis.
            neighbors_number (`int`): *> 0*, optional
                number of neighboring weight vectors to be considered during the operation
                of evolutionary operators as the "neighbors" of the processed sectors.
            nds_method (`callable`): optional, default ``moeadd.moeadd_supplementary.fast_non_dominated_sorting``
                Method of non-dominated sorting of the candidate solutions. The default method is implemented according to the article
                *K. Deb, A. Pratap, S. Agarwal, and T. Meyarivan, “A fast and elitist
                multiobjective genetic algorithm: NSGA-II,” IEEE Trans. Evol. Comput.,
                vol. 6, no. 2, pp. 182–197, Apr. 2002.*
            ndl_update (`callable`): optional, defalut ``moeadd.moeadd_supplementary.NDL_update``
                Method of adding a new solution point into the objective functions space, introduced
                to minimize the recalculation of the non-dominated levels for the entire population.
                The default method was taken from the *K. Li, K. Deb, Q. Zhang, and S. Kwong, “Efficient non-domination level
                update approach for steady-state evolutionary multiobjective optimization,”
                Dept. Electr. Comput. Eng., Michigan State Univ., East Lansing,
                MI, USA, Tech. Rep. COIN No. 2014014, 2014.*
            neighborhood_selector (`callable`): optional
                Method of finding "close neighbors" of the vector with proximity list.
                The baseline example of the selector, presented in
                ``moeadd.moeadd_stc.simple_selector``, selects n-adjacent ones.
            subregion_mating_limitation (`float`): optional
                The probability of mating selection to be limited only to the selected
                subregions (adjacent to the weight vector domain).:math:`\delta \in [0., 1.)
            neighborhood_selector_params (`tuple|list`): optional
                Iterable, which will be passed into neighborhood_selector, as
                an arugument. *None*, is no additional arguments are required inside
                the selector.
            training_epochs (`int`): optional
                Maximum number of iterations, during that the optimization will be held.
                Note, that if the algorithm converges to a single Pareto frontier,
                the optimization is stopped.
            PBI_penalty (`float`):  optional
                The penalty parameter, used in penalty based intersection
                calculation, defalut value is 1.

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
        Setting parameters for singelobjective optimization.

        Args:
            population_size (`int`): optional, default - 4
                Size of population.
            solution_params (`dict`):
                Parameters, guiding candidate solution creation.
            sorting_method(`callable`): optional, default - `simple_sorting`
                Method for sorting of individs in population.
            trainig_epochs (`int`): optional, default - 50
                Maximum number of iterations, during that the optimization will be held.
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
        Creating caches for keeping tensors during EPDE search.
        
        Args:
            coordinate_tensors (`np.ndarray|list`):
                Grid values, passed as a single `np.ndarray` or a list of `np.ndarray`'s.
            memory_for_cache (`int`): 
                allowed amount of memory for data storage
        
        Returns:
            None
        """
        global_var.init_caches(set_grids=True)
        example = coordinate_tensors if isinstance(coordinate_tensors, np.ndarray) else coordinate_tensors[0]
        self.set_memory_properties(example_tensor=example, mem_for_cache_frac=memory_for_cache)
        upload_grids(coordinate_tensors, global_var.initial_data_cache)
        upload_grids(coordinate_tensors, global_var.grid_cache)

    def set_boundaries(self, boundary_width: Union[int, list]):
        """
        Setting the number of unaccountable elements at the edges into cache with saved grid.
        """
        global_var.grid_cache.set_boundaries(boundary_width=boundary_width)

    def _upload_g_func(self, function_form: Union[Callable, np.ndarray, list] = None):
        """
        Loading testing function connected to the weak derivative notion.

        Args:
            function_form (`callable`, or `np.ndarray`, or `list[np.ndarray]`)
                Test function, default using inverse polynomial with max in the domain center.

        Returns:
            None 
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
        Setting properties for processing considered domain, such as removing areas with no dynamics,
        and setting bounderes. Can be used for uploading test function.

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
        '''
        Specification of preprocessor, devoted to smoothing the raw input data and 
        calculating the derivatives.
    
        Parameters
        ----------
        preprocessor_pipeline : PreprocessingPipe, optional
            Pipeline of operators, aimed on preparing all necessary data for equation discovery.
        default_preprocessor_type : str, optional
            Key for selection of pre-defined preprocessors: **'poly'** matches Savitsky-Golay filtering, 'ANN' if for 
            neural network data approximation and further finite-difference differentiation, 'spectral' for 
            spectral differentiation. The default is 'poly'.
        preprocessor_kwargs : dict, optional
            Keyword arguments for preprocessor setup and operation. The default is an empty dictionary, corresponding to 
            all default parameters of the preprocessors.
    
        Returns
        -------
        None.
    
        '''
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
                    data_fun_pow: int = 1, deriv_fun_pow: int = 1, grid: list = None):
        '''
        Create pool of tokens to represent elementary functions, that can be included in equations.
        
        Args:
            data : np.ndarray | list of np.ndarrays | tuple of np.ndarrays
            
        '''
        grid = grid if grid is not None else global_var.grid_cache.get_all()[1]

        self.pool_params = {'variable_names' : variable_names, 'max_deriv_order' : max_deriv_order,
                            'additional_tokens' : [family.token_family.ftype for family in additional_tokens]}
        # assert (isinstance(derivs, list) and isinstance(derivs[0], np.ndarray)) or derivs is None
        # TODO: add better checks
        if isinstance(data, np.ndarray):
            data = [data,]

        if derivs is None:
            if len(data) != len(variable_names):
                print(len(data), len(variable_names))
                raise ValueError('Mismatching lengths of data tensors and the names of the variables')
        else:
            if not (len(data) == len(variable_names) == len(derivs)):
                raise ValueError('Mismatching lengths of data tensors, names of the variables and passed derivatives')

        if not self.preprocessor_set:
            self.set_preprocessor()

        data_tokens = []
        
        for data_elem_idx, data_tensor in enumerate(data):
            # TODO: add more relevant checks
            # assert isinstance(data_tensor, np.ndarray), 'Input data must be in format of numpy ndarrays or iterable (list or tuple) of numpy arrays'
            entry = InputDataEntry(var_name=variable_names[data_elem_idx],
                                   data_tensor=data_tensor)
            derivs_tensor = derivs[data_elem_idx] if derivs is not None else None
            entry.set_derivatives(preprocesser=self.preprocessor_pipeline, deriv_tensors=derivs_tensor,
                                  grid=grid, max_order=max_deriv_order) # Diff with appropriate method
            entry.use_global_cache()

            self.set_derivatives(variable=variable_names[data_elem_idx], deriv=entry.derivatives)  
            entry.create_derivs_family(max_deriv_power=deriv_fun_pow)
            entry.create_polynomial_family(max_power=data_fun_pow)
            
            data_tokens.extend(entry.get_families())

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
        
    def set_derivatives(self, variable:str, deriv:np.ndarray):
        '''
        Pass the derivatives of a variable as a np.ndarray.
    
        Parameters
        ----------
        variable : str
            Key for the variable to have the derivatives set.
        deriv : np.ndarray
            Arrays of derivatives. Have to be shaped as (n, m), where n is the number of passed derivatives 
            (for example, when you differentiate the dataset once for the first axis, and up to the second order for 
             the second, and you have no mixed derivatives, *n = 3*), and m is the number of data points in the domain.

        Returns
        -------
        None.
        '''
        try:
            self._derivatives
        except AttributeError:
            self._derivatives = {}
        self._derivatives[variable] = deriv

    @property
    def saved_derivaties(self):
        try:
            return self._derivatives
        except AttributeError:
            print('Trying to get derivatives before their calculation. Call EPDESearch.create_pool() to calculate derivatives')
            return None

    def fit(self, data: Union[np.ndarray, list, tuple] = None, equation_terms_max_number=6,
            equation_factors_max_number=1, variable_names=['u',], eq_sparsity_interval=(1e-4, 2.5), 
            derivs=None, max_deriv_order=1, additional_tokens=[], data_fun_pow: int = 1, deriv_fun_pow: int = 1,
            optimizer: Union[SimpleOptimizer, MOEADDOptimizer] = None, pool: TFPool = None,
            population: Union[ParetoLevels, Population] = None):
        """
        Fit epde search algorithm to obtain differential equations, describing passed data.

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
        field_smooth : bool, optional
            Parameter, if the input variable fields shall be smoothed to avoid the errors. If the data is
            assumed to be noiseless, shall be set to False, otherwise - True, the default - False.
        memory_for_cache : int | float, optional
            Limit for the cache (in fraction of the memory) for precomputed tensor values to be stored:
            if int, will be considered as the percentage of the entire memory, and if float,
            then as a fraction of memory, the default is 5.
        data_fun_pow : int, optional
            Maximum power of token, the default is 1.
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
        """
        # TODO: ADD EXPLICITLY SENT POPULATION PROCESSING
        cur_params = {'variable_names' : variable_names, 'max_deriv_order' : max_deriv_order,
                      'additional_tokens' : [family.token_family.ftype for family in additional_tokens]}

        if pool is None:
            if self.pool == None or self.pool_params != cur_params:
                if data is None:
                    raise ValueError('Data has to be specified beforehand or passed in fit as an argument.')
                self.create_pool(data = data, variable_names=variable_names, 
                                 derivs=derivs, max_deriv_order=max_deriv_order, 
                                 additional_tokens=additional_tokens, 
                                 data_fun_pow=data_fun_pow, deriv_fun_pow = deriv_fun_pow)
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



    @staticmethod
    def _create_optimizer(multiobjective_mode:bool, optimizer_init_params:dict, 
                          opt_strategy_director:OptimizationPatternDirector):
        if multiobjective_mode:
            optimizer = MOEADDOptimizer(**optimizer_init_params)
                            
            best_obj = np.concatenate((np.zeros(shape=len([1 for token_family in optimizer_init_params['population_instruct']['pool'].families 
                                                           if token_family.status['demands_equation']])),
                                       np.ones(shape=len([1 for token_family in optimizer_init_params['population_instruct']['pool'].families 
                                                          if token_family.status['demands_equation']]))))
            print('best_obj', len(best_obj))
            optimizer.pass_best_objectives(*best_obj)            
        else:
            optimizer = SimpleOptimizer(**optimizer_init_params)
        
        optimizer.set_strategy(opt_strategy_director)        
        return optimizer

    @property
    def _resulting_population(self):
        if not self.search_conducted:
            raise AttributeError('Pareto set of the best equations has not been discovered. Use ``self.fit`` method.')
        if self.multiobjective_mode:
            return self.optimizer.pareto_levels.levels
        else:
            return self.optimizer.population.population
    
    def equations(self, only_print : bool = True, only_str = False, num = 1):
        """
        Method for print or getting results of searching differential equation

        Parameters
        ----------
        only_print : `bool`, optional
            Flag about action (print ot get) for results, the default is True.
        Num : `int`, optional
            Number of results for return or printing, the default is 1.

        Returns:
            None, when `only_print` == True
            resulting equations from population, when `only_print` == False  
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
        '''
        Method returns solver forms of the equations in a form of Python list.

        Returns:
            system form, suitable for solver
        '''
        forms = []
        if self.multiobjective_mode:
            for level in self._resulting_population[:min(num, len(self._resulting_population))]:
                temp = []
                for sys in level: #self.resulting_population[idx]:
                    temp.append(SystemSolverInterface(sys).form(grids=grids))
                forms.append(temp)
        else:
            for sys in self._resulting_population[:min(num, len(self._resulting_population))]:
                forms.append(SystemSolverInterface(sys).form(grids=grids))
        return forms

    @property
    def cache(self):
        if global_var.grid_cache is not None:
            return global_var.grid_cache, global_var.tensor_cache
        else:
            return None, global_var.tensor_cache

    def get_equations_by_complexity(self, complexity : Union[float, list]):
        '''
        Get equations with desired complexity. Works best with ``EpdeSearch.visualize_solutions(...)``

        Parameters
        ----------
        complexity : float | list of floats
            The complexity metric of the desited equation. For systems of equations shall be passed as the list of complexities.

        Returns
        -------
        list of ``epde.structure.main_structures.SoEq objects``.
        '''
        return self.optimizer.pareto_levels.get_by_complexity(complexity)

    def predict(self, system : SoEq, boundary_conditions: BoundaryConditions = None, grid : list = None, data = None,
                system_file: str = None, mode: str = 'NN', compiling_params: dict = {}, optimizer_params: dict = {},
                cache_params: dict = {}, early_stopping_params: dict = {}, plotting_params: dict = {}, 
                training_params: dict = {}, use_cache: bool = False, use_fourier: bool = False, 
                fourier_params: dict = None, net = None, use_adaptive_lambdas: bool = False):
        '''
        Predict state by automatically solving discovered equation or system. Employs solver implementation, adapted from 
        https://github.com/ITMO-NSS-team/torch_DE_solver.  

        Parameters
        ----------
        system : SoEq
            Object, containing the system (or a single equation as a system of one equation) to solve. 
        boundary_conditions : BoundaryConditions, optional
            Boundary condition objects, should match the order of differential equations due to no internal checks. 
            Over/underdefined solution can happen, if the number of conditions is incorrect. The default value is None, 
            matching automatic construction of the required Dirichlet BC from data. 
        grid : list of np.ndarrays, optional
            Grids, defining Cartesian coordinates, on which the equations will be solved. The default is None, specifing 
            the use of grids, stored in cache during equation learning.
        data : TYPE, optional
            Dataset, from which the boundary conditions can be automatically created. The default is None, making use of
            the training datasets, stored in cache during equation training.
        system_file : str, optional
            Filename for the pickled equation/system of equations. If passed, **system** can be None. The default is None, meaning no equation.
        solver_kwargs : dict, optional
            Parameters of the solver. The default is {'use_cache' : True}, with that no  
        mode : TYPE, optional
            Key, defining used method of the automatic DE solution. Supported methods: 'NN', 'mat' and 'autodiff'. The default is 'NN'.

        Raises
        ------
        ValueError
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.

        '''
        
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
        Plot discovered equation, using matplotlib tools. By default the method plots only the Pareto-optimal 
        equations from the population. Furthermore, the annotate of the candidate equations are made with LaTeX toolkit. 
        '''
        if self.multiobjective_mode:
            self.optimizer.plot_pareto(dimensions=dimensions, **visulaizer_kwargs)
        else:
            raise NotImplementedError('Solution visualization is implemented only for multiobjective mode.')
            
            
class ExperimentCombiner(object):
    def __init__(self, candidates: Union[ParetoLevels, List[SoEq], List[ParetoLevels]]):
        self.complexity_matched = self.get_complexities(candidates)
        complexity_sets = [set() for i in self.complexity_matched[0][1]]
        for eq, complexity in self.complexity_matched:
            for idx, compl in enumerate(complexity):
                complexity_sets[idx].add(compl)
        self.ordered_complexities = [sorted(compl_set) for compl_set in complexity_sets]
        
    @singledispatchmethod
    def get_complexities(self, candidates) -> list:
        raise NotImplementedError('Incorrect type of equations to parse')

    @get_complexities.register
    def _(self, candidates: list) -> list:
        if isinstance(candidates[0], ParetoLevels):
            return reduce(lambda x, y: x.append(y), [self.get_complexities(pareto_level) for 
                                                    pareto_level in candidates], [])
        elif isinstance(candidates[0], SoEq):
            # Here we assume, that the number of objectives is even, having quality 
            # and complexity for each equation
            compl_objs_num = candidates[0].obj_fun.size/2
            return [(candidate, candidate.obj_fun[-compl_objs_num:]) for candidate in candidates]
        else:
            raise ValueError('Incorrect type of the equation')
        
    @get_complexities.register
    def _(self, candidates: ParetoLevels) -> list:
        eqs = reduce(lambda x, y: x.append(y), [self.get_complexities(level)  for 
                                                level in candidates.levels], [])
        return eqs
        
    def create_best_for_complexity(self, complexity: tuple, pool: TFPool):
        vars_to_describe = self.complexity_matched[0][0].vars_to_describe # Get dependent variables
        
        best_eqs = []
        for idx, elem in enumerate(complexity):
            if elem is not None:
                relaxed_compl = [None,]*len(complexity)
                relaxed_compl[idx] = elem
                candidates = [candidate for candidate, _ in self.complexity_matched 
                             if candidate.matches_complexitiy(relaxed_compl)]
                best_candidate = sorted(candidates, lambda x: x.obj_fun[idx])[0]
                # best_eqs.append(best_candidate.vals[vars_to_describe[idx]])
            else:
                best_candidate = sorted([candidate for candidate, _ in self.complexity_matched], 
                                       lambda x: x.obj_fun[idx])[0]
            best_eqs.append(best_candidate.vals[vars_to_describe[idx]])
        compound_equation = deepcopy(self.complexity_matched[0][0])
        compound_equation.create(passed_equations = best_eqs)
        return compound_equation
    
    def create_best(self, pool: TFPool):
        best_qualities_compl = [complexities[-1] for complexities in self.ordered_complexities]
        return self.create_best_for_complexity(best_qualities_compl, pool)
    
class EpdeMultisample(EpdeSearch):
    def __init__(self, data_samples : List[List], multiobjective_mode: bool = True, 
                 use_default_strategy: bool = True, director=None, 
                 director_params: dict = {'variation_params': {}, 'mutation_params': {},
                                           'pareto_combiner_params': {}, 'pareto_updater_params': {}}, 
                 time_axis: int = 0, function_form=None, boundary: int = 0, 
                 use_solver: bool = False, dimensionality: int = 1, verbose_params: dict = {'show_iter_idx' : True}, 
                 memory_for_cache=5, prune_domain: bool = False, 
                 pivotal_tensor_label=None, pruner=None, threshold: float = 1e-2, 
                 division_fractions=3, rectangular: bool = True, params_filename: str = None):
        """
        Args:
            use_default_strategy (`bool`): optional
                True (base and recommended value), if the default evolutionary strategy will be used, 
                False if the user-defined strategy will be passed further. Otherwise, the search will 
                not be conducted.  
            time_axis (`int`): optional
                Indicator of time axis in data and grids. Used in normalization for regressions.
            function_form (`callable`): optional
                Auxilary function, used in the weak derivative definition. Default function is negative square function 
                with maximum values in the center of the domain.
            boundary (`int|tuple/list of integers`): optional
                Boundary width for the domain. Boundary points will be ignored for the purposes of equation discovery
            use_solver (`bool`): optional
                Allow use of the automaic partial differential solver to evaluate fitness of the candidate solutions.
            dimensionality (`int`): optional
                Dimensionality of the problem. ! Currently you should pass value, reduced by one !
            verbose_params (`dict`): optional
                Description, of algorithm details, that will be demonstrated to the user. Usual
            memory_for_cache (`int|float`): optional
                Rough estimation of the memory, which can be used for cache of pre-evaluated tensors during the equation
            prune_domain (`bool`): optional
                If ``True``, subdomains with no dynamics will be pruned from data. Default value: ``False``.
            pivotal_tensor_label (`str`): optional
                Indicator, according to which token data will be pruned. Default value - ``'du/dt'``, where 
                ``t`` is selected as a time axis from ``time_axis`` parameter.
            pruner (`object`): optional
                Pruner object, which will remove subdomains with no dynamics i.e. with derivative 
                identically equal to zero.
            threshold (`float`): optional
                Pruner parameter, indicating the boundary of interval in which the pivotal tensor values are 
                considered as zeros. Default value: 1e-2
            division_fractions (`int`): optional
                Number of subdomains along each axis, defining the division of the domain for pruning.
                Default value: 3
            rectangular(`bool`): optional
                A line of subdomains along an axis can be removed if all values inside them are identical to zero.
        """
        super().__init__(multiobjective_mode = multiobjective_mode, use_default_strategy = use_default_strategy, 
                         director = director, director_params = director_params, time_axis = time_axis,
                         define_domain = False, function_form = function_form, boundary = boundary, 
                         use_solver = use_solver, dimensionality = dimensionality, verbose_params = verbose_params, 
                         coordinate_tensors = None, memory_for_cache = memory_for_cache, prune_domain = prune_domain, 
                         pivotal_tensor_label = pivotal_tensor_label, pruner = pruner, threshold = threshold, 
                         division_fractions = division_fractions, rectangular = rectangular, 
                         params_filename = params_filename)
        self._memory_for_cache = memory_for_cache
        self._boundary = boundary
        self._function_form = function_form

        grids = [sample[0] for sample in data_samples]
        print('grids shape is', [(type(subgrid), len(subgrid)) for subgrid in grids])

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
        Setting properties for processing considered domain, such as removing areas with no dynamics,
        and setting bounderes. Can be used for uploading test function. In enseble equation learning can not 
        take coordinates as the argument.

        Parameters
        ----------
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
        Loading testing function connected to the weak derivative notion. In contrast to a single equation
        discovery approach the testing function is not immediately stored in cache, but saved to be used 
        later and applied to equations.

        Args:
            function_form (`callable`, or `np.ndarray`, or `list[np.ndarray]`)
                Test function, default using inverse polynomial with max in the domain center.

        Returns:
            None 
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
        field_smooth : bool, optional
            Parameter, if the input variable fields shall be smoothed to avoid the errors. If the data is
            assumed to be noiseless, shall be set to False, otherwise - True, the default - False.
        memory_for_cache : int | float, optional
            Limit for the cache (in fraction of the memory) for precomputed tensor values to be stored:
            if int, will be considered as the percentage of the entire memory, and if float,
            then as a fraction of memory, the default is 5.
        data_fun_pow : int, optional
            Maximum power of token, the default is 1.
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