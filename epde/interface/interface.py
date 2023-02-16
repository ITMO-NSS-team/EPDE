# !/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 15:55:12 2021

@author: mike_ubuntu
"""
import time
import pickle
import numpy as np
from typing import Union, Callable
from collections import OrderedDict
import warnings

import epde.globals as global_var

from epde.optimizers.moeadd.moeadd import *
from epde.optimizers.moeadd.supplementary import *
from epde.optimizers.single_criterion.supplementary import simple_sorting
# from epde.optimizers.moeadd.strategy_elems import SectorProcesserBuilder

from epde.preprocessing.DomainPruning import DomainPruner
from epde.operators.utils.default_parameter_loader import EvolutionaryParams

from epde.optimizers.builder import StrategyBuilder
from epde.optimizers.single_criterion.optimizer import EvolutionaryStrategy, SimpleOptimizer
from epde.optimizers.moeadd.strategy_elems import MOEADDSectorProcesser
from epde.decorators import BoundaryExclusion

from epde.optimizers.single_criterion.population_constr import SystemsPopulationConstructor as SOSystemPopConstr
from epde.optimizers.moeadd.population_constr import SystemsPopulationConstructor as MOEADDSystemPopConstr
from epde.optimizers.single_criterion.strategy import BaselineDirector
from epde.optimizers.moeadd.strategy import MOEADDDirector

from epde.evaluators import simple_function_evaluator, trigonometric_evaluator
from epde.supplementary import define_derivatives
from epde.cache.cache import upload_simple_tokens, upload_grids, prepare_var_tensor #, np_ndarray_section

from epde.preprocessing.preprocessor_setups import PreprocessorSetup
from epde.preprocessing.preprocessor import ConcretePrepBuilder, PreprocessingPipe

from epde.structure.main_structures import Equation, SoEq

from epde.interface.token_family import TFPool, TokenFamily
from epde.interface.type_checks import *
from epde.interface.prepared_tokens import PreparedTokens, CustomTokens
from epde.interface.solver_integration import BoundaryConditions, SolverAdapter

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
    def __init__(self, var_name: str, data_tensor: np.ndarray):
        self.var_name = var_name
        check_nparray(data_tensor)
        self.data_tensor = data_tensor

    def set_derivatives(self, preprocesser: PreprocessingPipe, deriv_tensors=None,
                        max_order: Union[list, tuple, int] = 1, grid: list = []):
        """
        Method for setting derivatives ot calculate derivatives from data 
        """
        deriv_names, deriv_orders = define_derivatives(self.var_name, dimensionality=self.data_tensor.ndim,
                                                       max_order=max_order)

        self.names = deriv_names
        self.d_orders = deriv_orders

        if deriv_tensors is None:
            self.data_tensor, self.derivatives = preprocesser.run(self.data_tensor, grid=grid,
                                                                  max_order=max_order)
            self.deriv_properties = {'max order': max_order,
                                     'dimensionality': self.data_tensor.ndim}
        else:
            self.derivatives = deriv_tensors
            self.deriv_properties = {'max order': max_order,
                                     'dimensionality': self.data_tensor.ndim}

    def use_global_cache(self):
        # print(f'self.data_tensor: {self.data_tensor.shape}')
        # print(f'self.derivatives: {self.derivatives.shape}')
        derivs_stacked = prepare_var_tensor(self.data_tensor, self.derivatives, time_axis=global_var.time_axis)
        # print(f'derivs_stacked: {derivs_stacked.shape}')

        # raise Exception('ZUL LUL')
        try:
            upload_simple_tokens(self.names, global_var.tensor_cache, derivs_stacked)
            upload_simple_tokens([self.var_name,], global_var.initial_data_cache, [self.data_tensor,])

        except AttributeError:
            raise NameError('Cache has not been declared before tensor addition.')

        global_var.tensor_cache.use_structural()


def simple_selector(sorted_neighbors, number_of_neighbors=4):
    return sorted_neighbors[:number_of_neighbors]


class EpdeSearch(object):
    '''
    
    Intialization of the epde search object. Here, the user can declare the properties of the 
    search mechainsm by defining evolutionary search strategy.
    
    Parameters:
    --------------
    
    use_default_strategy : bool, optional
        True (base and recommended value), if the default evolutionary strategy will be used, 
        False if the user-defined strategy will be passed further. Otherwise, the search will 
        not be conducted.
        
    director : OptimizationPatternDirector, optional
        User-defined director, responsible for construction of multi-objective evolutionary optimization
        strategy; shall not be interfered with unless for very specific tasks.
        
    director_params : dict, optional
        Contains parameters for evolutionary operator builder / construction director, that
        can be passed to individual operators. Keys shall be 'variation_params', 'mutation_params',
        'pareto_combiner_params', 'pareto_updater_params'.
        
    time_axis : int, optional
        Indicator of time axis in data and grids. Used in normalization for regressions.
        
    define_domain : bool, optional
        Indicator, showing that if the domain will be set in the initialization of the search objects.
        For more details view ``epde_search.set_domain_properties`` method.
        
    function_form : callable, optional
        Auxilary function, used in the weak derivative definition. Default function is negative square function 
        with maximum values in the center of the domain.
        
    boundary : int or tuple/list of integers, optional
        Boundary width for the domain. Boundary points will be ignored for the purposes of equation discovery
        
    use_solver : bool, optional
        Allow use of the automaic partial differential solver to evaluate fitness of the candidate solutions. 
        
    dimensionality : int, optional
        Dimensionality of the problem. ! Currently you should pass value, reduced by one !
    
    verbose_params : dict, optional
        Description, of algorithm details, that will be demonstrated to the user. Usual
        
    coordinate_tensors : list of np.ndarrays, optional
        Values of the coordinates on the grid nodes with studied functions values. In case of 1D-problem, 
        that will be ``numpy.array``, while the parameter for higher dimensionality problems can be set from 
        ``numpy.meshgrid`` function. With None, the tensors will be created as ranges with step of 1 between 
        nodes. Defalut value: None. 
    
    memory_for_cache : int or float, optional
        Rough estimation of the memory, which can be used for cache of pre-evaluated tensors during the equation
    
    prune_domain : bool, optional
        If ``True``, subdomains with no dynamics will be pruned from data. Default value: ``False``.
        
    pivotal_tensor_label : str, optional
        Indicator, according to which token data will be pruned. Default value - ``'du/dt'``, where 
        ``t`` is selected as a time axis from ``time_axis`` parameter.
        
    pruner : object, optional
        Pruner object, which will remove subdomains with no dynamics i.e. with derivative 
        identically equal to zero.
    
    threshold : float, optional
        Pruner parameter, indicating the boundary of interval in which the pivotal tensor values are 
        considered as zeros. Default value: 1e-2
        
    division_fractions : int, optional
        Number of subdomains along each axis, defining the division of the domain for pruning.
        Default value: 3
    
    rectangular: bool, optional
        A line of subdomains along an axis can be removed if all values inside them are identical to zero.        
    
    '''    
    def __init__(self, multiobjective_mode: bool = True, use_default_strategy: bool = True, director=None, 
                 director_params : dict = {'variation_params' : {}, 'mutation_params' : {},
                                           'pareto_combiner_params' : {}, 'pareto_updater_params' : {}}, 
                 time_axis: int = 0, define_domain: bool = True, function_form=None, boundary: int = 0, 
                 use_solver: bool = False, dimensionality: int = 1, verbose_params: dict = {}, 
                 coordinate_tensors=None, memory_for_cache=5, prune_domain: bool = False, 
                 pivotal_tensor_label=None, pruner=None, threshold: float = 1e-2, 
                 division_fractions=3, rectangular: bool = True):
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

        if use_solver:
            global_var.dimensionality = dimensionality

        if director is not None and not use_default_strategy:
            self.director = director
        elif director is None and self.multiobjective_mode and use_default_strategy:
            self.director = MOEADDDirector()
            self.director.builder = StrategyBuilder(MOEADDSectorProcesser)
            
            self.director.use_baseline(variation_params=director_params['variation_params'],
                                       mutation_params=director_params['mutation_params'],
                                       pareto_combiner_params=director_params['pareto_combiner_params'],
                                       pareto_updater_params=director_params['pareto_updater_params'])
        elif director is None and not self.multiobjective_mode and use_default_strategy:
            self.director = BaselineDirector()
            # print('ZUL')
            self.director.builder = StrategyBuilder(EvolutionaryStrategy)
            
            self.director.use_baseline(variation_params       = director_params['variation_params'], 
                                       mutation_params        = director_params['mutation_params'])

        else:
            raise NotImplementedError('Wrong arguments passed during the epde search initialization')
        if self.multiobjective_mode:    
            self.set_moeadd_params()
        else:
            self.set_singleobjective_params()
        self.search_conducted = False

    def set_memory_properties(self, example_tensor, mem_for_cache_frac=None, mem_for_cache_abs=None):
        '''

        Parameters
        ----------
        example_tensor : TYPE
            DESCRIPTION.
        mem_for_cache_frac : TYPE, optional
            DESCRIPTION. The default is None.
        mem_for_cache_abs : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None.

        '''
        if global_var.grid_cache is not None:
            if mem_for_cache_frac is not None:
                mem_for_cache_frac = int(mem_for_cache_frac/2.)
            else:
                mem_for_cache_abs = int(mem_for_cache_abs/2.)
            global_var.grid_cache.memory_usage_properties(example_tensor, mem_for_cache_frac, mem_for_cache_abs)
        global_var.tensor_cache.memory_usage_properties(example_tensor, mem_for_cache_frac, mem_for_cache_abs )

    def set_moeadd_params(self, population_size: int = 6, solution_params: dict = {},
                          delta: float = 1/50., neighbors_number: int = 3,
                          nds_method: Callable = fast_non_dominated_sorting,
                          ndl_update_method: Callable = ndl_update,
                          subregion_mating_limitation: float = .95,
                          PBI_penalty: float = 1., training_epochs: int = 100,
                          neighborhood_selector: Callable = simple_selector,
                          neighborhood_selector_params: tuple = (4,)):
        '''

        Setting the parameters of the multiobjective evolutionary algorithm. declaration of
        the default values is held in the initialization of EpdeSearch object.

        Parameters:
        ------------

        population_size: int, optional
            The size of the population of solutions, created during MO - optimization, default 6.

        solution_params: dict, optional
            Dictionary, containing additional parameters to be sent into the newly created
            solutions.

        delta: float, optional
            parameter of uniform spacing between the weight vectors; *H = 1 / delta*
            should be integer - a number of divisions along an objective coordinate axis.

        neighbors_number: int, *> 0*, optional
            number of neighboring weight vectors to be considered during the operation
            of evolutionary operators as the "neighbors" of the processed sectors.

        nds_method: function, optional, default ``moeadd.moeadd_supplementary.fast_non_dominated_sorting``
            Method of non-dominated sorting of the candidate solutions. The default method is implemented according to the article
            *K. Deb, A. Pratap, S. Agarwal, and T. Meyarivan, “A fast and elitist
            multiobjective genetic algorithm: NSGA-II,” IEEE Trans. Evol. Comput.,
            vol. 6, no. 2, pp. 182–197, Apr. 2002.*

        ndl_update: function, optional, defalut ``moeadd.moeadd_supplementary.NDL_update``
            Method of adding a new solution point into the objective functions space, introduced
            to minimize the recalculation of the non-dominated levels for the entire population.
            The default method was taken from the *K. Li, K. Deb, Q. Zhang, and S. Kwong, “Efficient non-domination level
            update approach for steady-state evolutionary multiobjective optimization,”
            Dept. Electr. Comput. Eng., Michigan State Univ., East Lansing,
            MI, USA, Tech. Rep. COIN No. 2014014, 2014.*


        neighborhood_selector: function, optional
            Method of finding "close neighbors" of the vector with proximity list.
            The baseline example of the selector, presented in
            ``moeadd.moeadd_stc.simple_selector``, selects n-adjacent ones.

        subregion_mating_limitation: float, optional
            The probability of mating selection to be limited only to the selected
            subregions (adjacent to the weight vector domain).:math:`\delta \in [0., 1.)

        neighborhood_selector_params: tuple/list or None, optional
            Iterable, which will be passed into neighborhood_selector, as
            an arugument. *None*, is no additional arguments are required inside
            the selector.

        epochs: int, optional
            Maximum number of iterations, during that the optimization will be held.
            Note, that if the algorithm converges to a single Pareto frontier,
            the optimization is stopped.

        PBI_penalty:  float, optional
            The penalty parameter, used in penalty based intersection
            calculation, defalut value is 1.

        '''
        self.optimizer_init_params = {'weights_num': population_size, 'pop_size': population_size,
                              'delta': delta, 'neighbors_number': neighbors_number,
                              'solution_params': solution_params,
                              'nds_method' : nds_method, 
                              'ndl_update' : ndl_update_method}
        
        self.optimizer_exec_params = {'epochs' : training_epochs}
 
    def set_singleobjective_params(self, population_size: int = 4, solution_params: dict = {}, 
                                   sorting_method: Callable = simple_sorting, training_epochs: int = 50):
        self.optimizer_init_params = {'pop_size' : population_size, 'solution_params': solution_params,
                                      'sorting_method' : sorting_method}
        
        self.optimizer_exec_params = {'epochs' : training_epochs}        
    
    def domain_pruning(self, pivotal_tensor_label = None, pruner = None, 
                             threshold : float = 1e-5, division_fractions = 3, 
                             rectangular : bool = True):
        
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

    def create_caches(self, coordinate_tensors, memory_for_cache):
        global_var.init_caches(set_grids=True)
        example = coordinate_tensors if isinstance(coordinate_tensors, np.ndarray) else coordinate_tensors[0]
        self.set_memory_properties(example_tensor=example, mem_for_cache_frac=memory_for_cache)
        upload_grids(coordinate_tensors, global_var.initial_data_cache)
        upload_grids(coordinate_tensors, global_var.grid_cache)

    def set_boundaries(self, boundary_width: Union[int, list]):
        global_var.grid_cache.set_boundaries(boundary_width=boundary_width)

    def upload_g_func(self, function_form: Callable = None):
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
        self.create_caches(coordinate_tensors=coordinate_tensors, memory_for_cache=memory_for_cache)
        if prune_domain:
            self.domain_pruning(pivotal_tensor_label, pruner, threshold, division_fractions, rectangular)
        self.set_boundaries(boundary_width)
        self.upload_g_func(function_form)

    def set_preprocessor(self, preprocessor_pipeline: PreprocessingPipe = None,
                         default_preprocessor_type: str = 'poly',
                         preprocessor_kwargs: dict = {}):
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
            else:
                raise NotImplementedError('Incorrect default preprocessor type. Only ANN or poly are allowed.')
            preprocessor_pipeline = setup.builder.prep_pipeline

        if 'max_order' not in preprocessor_pipeline.deriv_calculator_kwargs.keys():
            preprocessor_pipeline.deriv_calculator_kwargs['max_order'] = None

        self.preprocessor_set = True
        self.preprocessor_pipeline = preprocessor_pipeline

    def create_pool(self, data: Union[np.ndarray, list, tuple], variable_names=['u',],
                    derivs=None, max_deriv_order=1, additional_tokens=[],
                    data_fun_pow: int = 1):
        # method = 'ANN', method_kwargs: dict = {},
        assert (isinstance(derivs, list) and isinstance(derivs[0], np.ndarray)) or derivs is None
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
            assert isinstance(data_tensor, np.ndarray), 'Input data must be in format of numpy ndarrays or iterable (list or tuple) of numpy arrays'
            entry = InputDataEntry(var_name=variable_names[data_elem_idx],
                                     data_tensor=data_tensor)
            derivs_tensor = derivs[data_elem_idx] if derivs is not None else None
            entry.set_derivatives(preprocesser=self.preprocessor_pipeline, deriv_tensors=derivs_tensor,
                                  grid=global_var.grid_cache.get_all()[1], max_order=max_deriv_order)
            entry.use_global_cache()

            print(f'creating TokenFamily entry {entry.var_name}')
            time.sleep(10)
            entry_token_family = TokenFamily(entry.var_name, family_of_derivs=True)
            entry_token_family.set_status(demands_equation=True, unique_specific_token=False,
                                          unique_token_type=False, s_and_d_merged=False,
                                          meaningful=True)
            entry_token_family.set_params(entry.names, OrderedDict([('power', (1, data_fun_pow))]),
                                          {'power': 0}, entry.d_orders)
            entry_token_family.set_evaluator(simple_function_evaluator, [])

            print(entry_token_family.tokens)
            data_tokens.append(entry_token_family)

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

    def fit(self, data: Union[np.ndarray, list, tuple], equation_terms_max_number=6,
            equation_factors_max_number=1, variable_names=['u',], eq_sparsity_interval=(1e-4, 2.5),
            derivs=None, max_deriv_order=1,
            additional_tokens=[], coordinate_tensors=None, memory_for_cache=5,
            prune_domain: bool = False, pivotal_tensor_label=None, pruner=None,
            threshold: float = 1e-2, division_fractions=3, rectangular: bool = True,
            data_fun_pow: int = 1):
        '''

        Fit epde search algorithm to obtain differential equations, describing passed data.

        Parameters:
        -------------

        data: np.ndarray, list or tuple,
            Values of modeled variables. If the variable is single (i.e. deriving a single equation),
            it can be passed as the numpy.ndarray or as the list/tuple with a single element;
            multiple variables are not supported yet, use older interfaces.

        time_axis: int, optional,
            Index of the axis in the data, representing time. Default value: 0.

        equation_terms_max_number: int, optional,
            The maximum number of terms, present in the derived equations. Default value: 6.

        equation_factors_max_number: int, optional,
            The maximum number of factors (token functions; real-valued coefficients are not counted here),
            present in terms of the equaton. Default value: 1.

        variable_names: list of str, optional,
            Names of the independent variables, passed into search mechanism. Length of the list must correspond
            to the number of np.ndarrays, sent with in ``data`` parameter. In case of system of differential equation discovery, 
            all variables shall be named here. Defalut value: ``['u',]``, representinga single variable *u*.

        eq_sparsity_interval: tuple of two float values, optional,
            The left and right boundaries of interval with sparse regression values. Influence the number of
            terms in the equation. Default value: ``(1e-4, 2.5)``.

        derivs: list of lists of np.ndarrays or None, optional,
            Pre-computed values of derivatives. If ``None`` is passed, the derivatives are calculated in the
            method. Recommended to use, if the computations of derivatives take too long. For further information
            about using data, prepared in advance, check ``epde.preprocessing.derivatives.preprocess_derivatives`` function.
            Default value: None.

        max_deriv_order: int or tuple/list, optional,
            Highest order of calculated derivatives. Default value: 1.

        additional_tokens: None or list of ``TokenFamily`` or ``Prepared_tokens`` objects, optional
            Additional tokens, that would be used to construct the equations among the main variables and their
            derivatives. Objects of this list must be of type ``epde.interface.token_family.TokenFamily`` or
            of ``epde.interface.prepared_tokens.Prepared_tokens`` subclasses types. Default value: None.

        coordinate_tensors: list of np.ndarrays, optional
            Values of the coordinates on the grid nodes with studied functions values. In case of 1D-problem,
            that will be ``numpy.array``, while the parameter for higher dimensionality problems can be set from
            ``numpy.meshgrid`` function. With None, the tensors will be created as ranges with step of 1 between
            nodes. Defalut value: None.

        field_smooth: bool, optional
            Parameter, if the input variable fields shall be smoothed to avoid the errors. If the data is
            assumed to be noiseless, shall be set to False, otherwise - True. Default value: False.

        memory_for_cache: int or float, optional
            Limit for the cache (in fraction of the memory) for precomputed tensor values to be stored:
            if int, will be considered as the percentage of the entire memory, and if float,
            then as a fraction of memory.

        data_fun_pow: int
            Maximum power of token,

        '''
        self.create_pool(data = data, variable_names=variable_names, 
                         derivs=derivs, max_deriv_order=max_deriv_order, 
                         additional_tokens=additional_tokens, 
                         data_fun_pow=data_fun_pow)        
        if self.multiobjective_mode:
            self.fit_multiobjective(data, equation_terms_max_number,
                                    equation_factors_max_number, variable_names,
                                    eq_sparsity_interval, derivs, max_deriv_order,
                                    additional_tokens, memory_for_cache, prune_domain, 
                                    pivotal_tensor_label, pruner, threshold, division_fractions,
                                    rectangular, data_fun_pow)
        else:
            self.fit_singleobjective(data, equation_terms_max_number,
                                     equation_factors_max_number, variable_names,
                                     eq_sparsity_interval, derivs, max_deriv_order,
                                     additional_tokens, memory_for_cache, prune_domain, 
                                     pivotal_tensor_label, pruner, threshold, division_fractions,
                                     rectangular, data_fun_pow)
            
    def fit_multiobjective(self, data : Union[np.ndarray, list, tuple], equation_terms_max_number = 6,
            equation_factors_max_number = 1, variable_names = ['u',], eq_sparsity_interval = (1e-4, 2.5), 
            derivs = None, max_deriv_order = 1, additional_tokens = [], memory_for_cache = 5,
            prune_domain : bool = False, pivotal_tensor_label = None, pruner = None, 
            threshold : float = 1e-2, division_fractions = 3, rectangular : bool = True, 
            data_fun_pow : int = 1):    
        pop_constructor = MOEADDSystemPopConstr(pool = self.pool, terms_number = equation_terms_max_number, 
                                                max_factors_in_term = equation_factors_max_number,
                                                sparsity_interval = eq_sparsity_interval)

        self.optimizer_init_params['pop_constructor'] = pop_constructor
        self.optimizer = MOEADDOptimizer(**self.optimizer_init_params)
        
        evo_operator_builder = self.director.builder
        evo_operator_builder.assemble(True)
        evo_operator = evo_operator_builder.processer

        self.optimizer.set_sector_processer(processer=evo_operator)
        best_obj = np.concatenate((np.zeros(shape=len([1 for token_family in self.pool.families if token_family.status['demands_equation']])),
                                   np.ones(shape=len([1 for token_family in self.pool.families if token_family.status['demands_equation']]))))
        print('best_obj', len(best_obj))
        self.optimizer.pass_best_objectives(*best_obj)
        self.optimizer.optimize(**self.optimizer_exec_params)

        print('The optimization has been conducted.')
        self.search_conducted = True
        
    def fit_singleobjective(self, data : Union[np.ndarray, list, tuple], equation_terms_max_number = 6,
                            equation_factors_max_number = 1, variable_names = ['u',], eq_sparsity_interval = (1e-4, 2.5), 
                            derivs = None, max_deriv_order = 1, additional_tokens = [], memory_for_cache = 5,
                            prune_domain : bool = False, pivotal_tensor_label = None, pruner = None, 
                            threshold : float = 1e-2, division_fractions = 3, rectangular : bool = True, 
                            data_fun_pow : int = 1):
        pop_constructor = SOSystemPopConstr(pool = self.pool, terms_number = equation_terms_max_number, 
                                            max_factors_in_term = equation_factors_max_number,
                                            sparsity_interval = eq_sparsity_interval)
        # self.optimizer_params['pop_constructor']
        self.optimizer_init_params['pop_constructor'] = pop_constructor
        self.optimizer = SimpleOptimizer(**self.optimizer_init_params)        

        # TODO: Somehow generalize
        evo_operator_builder = self.director.builder
        evo_operator_builder.assemble(True)
        evo_operator = evo_operator_builder.processer
        self.optimizer.set_strategy(strategy = evo_operator)

        self.optimizer.optimize(**self.optimizer_exec_params)

        print('The optimization has been conducted.')
        self.search_conducted = True

    @property
    def resulting_population(self):
        if not self.search_conducted:
            raise AttributeError('Pareto set of the best equations has not been discovered. Use ``self.fit`` method.')
        if self.multiobjective_mode:
            return self.optimizer.pareto_levels.levels
        else:
            return self.optimizer.population.population
            
    
    def equation_search_results(self, only_print : bool = True, num = 1):
        if self.multiobjective_mode:
            if only_print:
                for idx in range(min(num, len(self.resulting_population))):
                    print('\n')
                    print(f'{idx}-th non-dominated level')    
                    print('\n')                
                    [print(f'{solution.text_form} , with objective function values of {solution.obj_fun} \n')  
                    for solution in self.resulting_population[idx]]
            else:
                return self.resulting_population[:num]
        else:
            if only_print:
                [print(f'{solution.text_form} , with objective function values of {solution.obj_fun} \n')  
                 for solution in self.resulting_population[:num]]
            else:
                return self.resulting_population[:num]

    def solver_forms(self):
        '''
        Method returns solver forms of the equations on 1-st non-dominated levels in a form of Python list.
        '''
        sf = []
        for system in self.resulting_population[0]:
            if len(system.structure) > 1:
                raise NotImplementedError('Currently, only "systems", containing a single equations, can be passed to solvers.')
            sf.append(system.structure[0].solver_form())
        return sf

    @property
    def cache(self):
        if global_var.grid_cache is not None:
            return global_var.grid_cache, global_var.tensor_cache
        else:
            return None, global_var.tensor_cache

    def get_equations_by_complexity(self, complexity : Union[int, list]):
        return self.optimizer.pareto_levels.get_by_complexity(complexity)

    def predict(self, system: SoEq, boundary_conditions: BoundaryConditions, grid: list = None, 
                system_file: str = None, solver_kwargs: dict={'model' : None, 'use_cache' : True}):
        solver_kwargs['dim'] = len(global_var.grid_cache.get_all()[1])
        # solver_kwargs['dim']
        
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
        
        adapter = SolverAdapter(var_number = len(system.vars_to_describe))
        solution_model = adapter.solve_epde_system(system = system, grids = grid, 
                                                   boundary_conditions = boundary_conditions)
        return solution_model(adapter.convert_grid(grid)).detach().numpy()
