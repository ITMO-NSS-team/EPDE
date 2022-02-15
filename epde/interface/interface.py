#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul  6 15:55:12 2021

@author: mike_ubuntu
"""
import numpy as np
from typing import Union, Callable
from collections import OrderedDict
import warnings

import epde.globals as global_var

from epde.moeadd.moeadd import *
from epde.moeadd.moeadd_supplementary import *

from epde.prep.DomainPruning import Domain_Pruner

from epde.operators.ea_stop_criteria import Stop_condition, Iteration_limit
import epde.operators.sys_search_operators as operators

from epde.evaluators import simple_function_evaluator, trigonometric_evaluator
from epde.supplementary import Define_Derivatives
from epde.cache.cache import upload_simple_tokens, upload_grids, prepare_var_tensor, np_ndarray_section
from epde.prep.derivatives import Preprocess_derivatives
from epde.eq_search_strategy import Strategy_director, Strategy_director_solver
from epde.structure import Equation

from epde.interface.token_family import TF_Pool, TokenFamily
from epde.interface.type_checks import *
from epde.interface.prepared_tokens import Prepared_tokens, Custom_tokens

class Input_data_entry(object):
    def __init__(self, var_name, data_tensor, coord_tensors = None):
        self.var_name = var_name
        check_nparray(data_tensor)
        if coord_tensors is not None:
            check_nparray_iterable(coord_tensors)
            if any([tensor.shape != data_tensor.shape for tensor in coord_tensors]):
                print('data_tensor.shape', data_tensor.shape, 'tensors:', 
                      [tensor.shape for tensor in coord_tensors])
                raise ValueError('mismatching shapes of coordinate tensors and input data')
            self.coord_tensors = coord_tensors
        else:
            axes = []
            for ax_idx in range(data_tensor.ndim):
                axes.append(np.linspace(0., 1., data_tensor.shape[ax_idx]))
            self.coord_tensors = np.meshgrid(*axes)
        self.data_tensor = data_tensor
        
    def set_derivatives(self, deriv_tensors = None, method = 'ANN', max_order = 1, method_kwargs = {}):

        deriv_names, deriv_orders = Define_Derivatives(self.var_name, dimensionality=self.data_tensor.ndim, 
                                                       max_order = max_order)

        self.names = deriv_names # coord_names +  Define_Derivatives(self.var_name, dimensionality=self.data_tensor.ndim, 
                                    #                  max_order = max_order)
        self.d_orders = deriv_orders
        if deriv_tensors is None:
            method_kwargs['max_order'] = max_order
            if self.coord_tensors is not None and 'grid' not in method_kwargs.keys():
                method_kwargs['grid'] = self.coord_tensors
            _, self.derivatives = Preprocess_derivatives(self.data_tensor, method=method, 
                                                         method_kwargs=method_kwargs)
            self.deriv_properties = {'max order' : max_order,
                                     'dimensionality' : self.data_tensor.ndim}
        else:
            self.derivatives = deriv_tensors
            self.deriv_properties = {'max order' : max_order,
                                     'dimensionality' : self.data_tensor.ndim}
            
    def use_global_cache(self, grids_as_tokens = True, set_grids = True, memory_for_cache = 5, 
                         boundary : int = 0):

        print(type(self.data_tensor), type(self.derivatives))
        derivs_stacked = prepare_var_tensor(self.data_tensor, self.derivatives,
                                            time_axis = global_var.time_axis, boundary = boundary) 
#                                            axes = self.coord_tensors)
        if isinstance(self.coord_tensors, (list, tuple)):
            coord_tensors_cut = []
            for tensor in self.coord_tensors:
                coord_tensors_cut.append(np_ndarray_section(tensor, boundary = boundary))
        elif isinstance(self.coord_tensors, np.ndarray):
            coord_tensors_cut = np_ndarray_section(self.coord_tensors, boundary = boundary)
        else:
            raise TypeError('Coordinate tensors are presented in format, other than np.ndarray or list/tuple of np.ndarray`s')
        
        try:       
            upload_simple_tokens(self.names, global_var.tensor_cache, derivs_stacked)
            upload_simple_tokens(['u',], global_var.initial_data_cache, [self.data_tensor,])            
            if set_grids: 
                memory_for_cache = int(memory_for_cache/2)
                upload_grids(self.coord_tensors, global_var.initial_data_cache)
                upload_grids(coord_tensors_cut, global_var.grid_cache)
                print(f'completed grid cache with {len(global_var.grid_cache.memory_default)} tensors with labels {global_var.grid_cache.memory_default.keys()}')

        except AttributeError:         
            print('Somehow, cache had not been initialized')
            print(self.names, derivs_stacked.shape)            
            print(global_var.tensor_cache.memory_default.keys())
            global_var.init_caches(set_grids = set_grids)
            if set_grids: 
                print('setting grids')
                memory_for_cache = int(memory_for_cache/2)
                global_var.grid_cache.memory_usage_properties(obj_test_case = self.data_tensor,
                                                                mem_for_cache_frac = memory_for_cache)
                upload_grids(self.coord_tensors, global_var.initial_data_cache)
                upload_grids(coord_tensors_cut, global_var.grid_cache)
                print(f'completed grid cache with {len(global_var.grid_cache.memory_default)} tensors with labels {global_var.grid_cache.memory_default.keys()}')
            global_var.tensor_cache.memory_usage_properties(obj_test_case = self.data_tensor,
                                                            mem_for_cache_frac = memory_for_cache)
            print(self.names, derivs_stacked.shape)
            upload_simple_tokens(self.names, global_var.tensor_cache, derivs_stacked)
            upload_simple_tokens(['u',], global_var.initial_data_cache, [self.data_tensor,])

        global_var.tensor_cache.use_structural()
    
def simple_selector(sorted_neighbors, number_of_neighbors = 4):
    return sorted_neighbors[:number_of_neighbors]
     
class epde_search(object):
    def __init__(self, use_default_strategy : bool = True, eq_search_stop_criterion : Stop_condition = Iteration_limit,
                 director = None, equation_type : set = {'PDE', 'derivatives only'}, time_axis : int = 0, 
                 init_cache : bool = True, example_tensor_shape : Union[tuple, list] = (1000,), 
                 set_grids : bool = True, eq_search_iter : int = 300, use_solver : bool = False, 
                 dimensionality : int = 1, verbose_params : dict = {}):
        '''
        
        Intialization of the epde search object. Here, the user can declare the properties of the 
        search mechainsm by defining evolutionary search strategy.
        
        Parameters:
        --------------
        
        use_default_strategy : bool, optional
            True (base and recommended value), if the default evolutionary strategy shall be used, 
            False if the user - defined strategy will be passed further.
        
        eq_search_stop_criterion : epde.operators.ea_stop_criteria.Stop_condition object, optional
            The stop condition for the evolutionary search of the equation. Default value 
            represents loop over 300 evolutionary epochs.
            
        director : obj, optional
            User-defined director, responsible for construction of evolutionary strategy; 
            shall be declared for very specific tasks.
        
        equation_type : set of str, optional
            Will define equation type, TBD later.
        
        '''
        global_var.set_time_axis(time_axis)
        global_var.init_verbose(**verbose_params)

        if init_cache:
            global_var.init_caches(set_grids)

        if use_solver:
            global_var.dimensionality = dimensionality

        if director is not None and not use_default_strategy:
            self.director = director
        elif director is None and use_default_strategy:
            if use_solver:
                self.director = Strategy_director_solver(eq_search_stop_criterion, {'limit' : eq_search_iter})#, 
                                                         # dimensionality=dimensionality)
            else:
                self.director = Strategy_director(eq_search_stop_criterion, {'limit' : eq_search_iter})
            self.director.strategy_assembly()
        else: 
            raise NotImplementedError('Wrong arguments passed during the epde search initialization')
        self.set_moeadd_params()
        self.search_conducted = False
        
            
    def set_memory_properties(self, example_tensor, mem_for_cache_frac = None, mem_for_cache_abs = None):
        if global_var.grid_cache is not None:
            if mem_for_cache_frac is not None:
                mem_for_cache_frac = int(mem_for_cache_frac/2.)
            else:
                mem_for_cache_abs = int(mem_for_cache_abs/2.)
            global_var.tensor_cache.memory_usage_properties(example_tensor, mem_for_cache_frac, mem_for_cache_abs)            
        global_var.tensor_cache.memory_usage_properties(example_tensor, mem_for_cache_frac, mem_for_cache_abs )

    
    def set_moeadd_params(self, population_size : int = 6, solution_params : dict = {}, 
                          delta : float = 1/50., neighbors_number : int = 3, 
                          NDS_method : Callable = fast_non_dominated_sorting, 
                          NDL_update_method : Callable = NDL_update,
                          subregion_mating_limitation : float = .95, 
                          PBI_penalty : float = 1., training_epochs : int = 100, 
                          neighborhood_selector : Callable = simple_selector, 
                          neighborhood_selector_params : tuple = (4,)):
        '''        
        
        Setting the parameters of the multiobjective evolutionary algorithm. declaration of 
        the default values is held in the initialization of epde_search object.
        
        Parameters:
        ------------
        
        population_size : int, optional
            The size of the population of solutions, created during MO - optimization, default 6.
            
        solution_params : dict, optional
            Dictionary, containing additional parameters to be sent into the newly created 
            solutions.
        
        delta : float, optional
            parameter of uniform spacing between the weight vectors; *H = 1 / delta*        
            should be integer - a number of divisions along an objective coordinate axis.
            
        neighbors_number : int, *> 0*, optional
            number of neighboring weight vectors to be considered during the operation 
            of evolutionary operators as the "neighbors" of the processed sectors.
            
        NDS_method : function, optional, default ``moeadd.moeadd_supplementary.fast_non_dominated_sorting``
            Method of non-dominated sorting of the candidate solutions. The default method is implemented according to the article 
            *K. Deb, A. Pratap, S. Agarwal, and T. Meyarivan, “A fast and elitist
            multiobjective genetic algorithm: NSGA-II,” IEEE Trans. Evol. Comput.,
            vol. 6, no. 2, pp. 182–197, Apr. 2002.*
            
        NDL_update : function, optional, defalut ``moeadd.moeadd_supplementary.NDL_update``
            Method of adding a new solution point into the objective functions space, introduced 
            to minimize the recalculation of the non-dominated levels for the entire population. 
            The default method was taken from the *K. Li, K. Deb, Q. Zhang, and S. Kwong, “Efficient non-domination level
            update approach for steady-state evolutionary multiobjective optimization,” 
            Dept. Electr. Comput. Eng., Michigan State Univ., East Lansing,
            MI, USA, Tech. Rep. COIN No. 2014014, 2014.*        
            
        
        neighborhood_selector : function, optional
            Method of finding "close neighbors" of the vector with proximity list.
            The baseline example of the selector, presented in 
            ``moeadd.moeadd_stc.simple_selector``, selects n-adjacent ones.
            
        subregion_mating_limitation : float, optional
            The probability of mating selection to be limited only to the selected
            subregions (adjacent to the weight vector domain). :math:`\delta \in [0., 1.)
        
        neighborhood_selector_params : tuple/list or None, optional
            Iterable, which will be passed into neighborhood_selector, as 
            an arugument. *None*, is no additional arguments are required inside
            the selector.
            
        epochs : int, optional
            Maximum number of iterations, during that the optimization will be held.
            Note, that if the algorithm converges to a single Pareto frontier, 
            the optimization is stopped.
            
        PBI_penalty :  float, optional
            The penalty parameter, used in penalty based intersection 
            calculation, defalut value is 1.        
        
        '''    
        self.moeadd_params = {'weights_num' : population_size, 'pop_size' : population_size, 
                              'delta' : delta, 'neighbors_number' : neighbors_number, 
                              'solution_params': solution_params,
                              'NDS_method' : NDS_method, 
                              'NDL_update' : NDL_update_method}
        
        self.moeadd_optimization_params = {'neighborhood_selector' : neighborhood_selector,
                                           'neighborhood_selector_params' : neighborhood_selector_params,
                                           'delta' : subregion_mating_limitation, 
                                           'epochs' : training_epochs, 
                                           'PBI_penalty' : PBI_penalty}
 
    def set_domain_pruning(self, pivotal_tensor_label = None, pruner = None, 
                             threshold : float = 1e-5, division_fractions = 3, 
                             rectangular : bool = True):
        if pruner is not None:
            self.pruner = pruner
        else:
            self.pruner = Domain_Pruner(domain_selector_kwargs={'threshold' : threshold})
        if not self.pruner.bds_init:
            if pivotal_tensor_label is None:
                pivotal_tensor_label = ('du/dx'+str(global_var.time_axis + 1), (1.0,))
            elif isinstance(pivotal_tensor_label, str):
                pivotal_tensor_label = (pivotal_tensor_label, (1.0,))
            elif not isinstance(pivotal_tensor_label, tuple):
                raise TypeError('Label of the pivotal tensor shall be declared with str or tuple')
                
            pivotal_tensor = global_var.tensor_cache.get(pivotal_tensor_label)
            self.pruner.get_boundaries(pivotal_tensor, division_fractions = division_fractions,
                                       rectangular = rectangular)
        
        global_var.tensor_cache.prune_tensors(self.pruner)
        if global_var.grid_cache is not None:
            global_var.grid_cache.prune_tensors(self.pruner)
    
    def create_pool(self, data : Union[np.ndarray, list, tuple], time_axis : int = 0, boundary : int = 0, 
                    variable_names = ['u',], derivs = None, method = 'ANN', method_kwargs : dict = {},
                    max_deriv_order = 1, additional_tokens = [], coordinate_tensors = None,
                    memory_for_cache = 5, prune_domain : bool = False,
                    pivotal_tensor_label = None, pruner = None, threshold : float = 1e-2, 
                    division_fractions = 3, rectangular : bool = True, data_fun_pow : int = 1):
        assert (isinstance(derivs, list) and isinstance(derivs[0], np.ndarray)) or derivs is None
        if isinstance(data, np.ndarray):
            data = [data,]

        set_grids = coordinate_tensors is not None
        set_grids_among_tokens = coordinate_tensors is not None
        if derivs is None:
            if len(data) != len(variable_names):
                print(len(data), len(variable_names))
                raise ValueError('Mismatching lengths of data tensors and the names of the variables')
        else:
            if not (len(data) == len(variable_names) == len(derivs)): 
                raise ValueError('Mismatching lengths of data tensors, names of the variables and passed derivatives')            
        data_tokens = []
        for data_elem_idx, data_tensor in enumerate(data):
            assert isinstance(data_tensor, np.ndarray), 'Input data must be in format of numpy ndarrays or iterable (list or tuple) of numpy arrays'
            entry = Input_data_entry(var_name = variable_names[data_elem_idx],
                                     data_tensor = data_tensor, 
                                     coord_tensors = coordinate_tensors)
            derivs_tensor = derivs[data_elem_idx] if derivs is not None else None
            entry.set_derivatives(deriv_tensors = derivs_tensor, method = method, max_order = max_deriv_order, 
                                  method_kwargs=method_kwargs)
            print(f'set grids parameter is {set_grids}')
            entry.use_global_cache(grids_as_tokens = set_grids_among_tokens,
                                   set_grids=set_grids, memory_for_cache=memory_for_cache, 
                                   boundary=boundary)
            set_grids = False; set_grids_among_tokens = False
            
            entry_token_family = TokenFamily(entry.var_name, family_of_derivs = True)
            entry_token_family.set_status(unique_specific_token=False, unique_token_type=False, 
                                 s_and_d_merged = False, meaningful = True, 
                                 unique_for_right_part = False)     
            entry_token_family.set_params(entry.names, OrderedDict([('power', (1, data_fun_pow))]),
                                          {'power' : 0}, entry.d_orders)
            entry_token_family.set_evaluator(simple_function_evaluator, [])
                
            print(entry_token_family.tokens)
            data_tokens.append(entry_token_family)
        if prune_domain:            
            self.set_domain_pruning(pivotal_tensor_label, pruner, threshold, division_fractions, rectangular)
            
        if isinstance(additional_tokens, list):
            if not all([isinstance(tf, (TokenFamily, Prepared_tokens)) for tf in additional_tokens]):
                raise TypeError(f'Incorrect type of additional tokens: expected list or TokenFamily/Prepared_tokens - obj, instead got list of {type(additional_tokens[0])}')                
        elif isinstance(additional_tokens, (TokenFamily, Prepared_tokens)):
            additional_tokens = [additional_tokens,]
        else:
            print(isinstance(additional_tokens, Prepared_tokens))
            raise TypeError(f'Incorrect type of additional tokens: expected list or TokenFamily/Prepared_tokens - obj, instead got {type(additional_tokens)}')
        self.pool = TF_Pool(data_tokens + [tf if isinstance(tf, TokenFamily) else tf.token_family 
                                      for tf in additional_tokens])
        print(f'The cardinality of defined token pool is {self.pool.families_cardinality()}')
    
    def fit(self, data : Union[np.ndarray, list, tuple], time_axis : int = 0, boundary : int = 0, 
            equation_terms_max_number = 6, equation_factors_max_number = 1, variable_names = ['u',], 
            eq_sparsity_interval = (1e-4, 2.5), derivs = None, max_deriv_order = 1, 
            deriv_method = 'ANN', deriv_method_kwargs : dict = {},
            additional_tokens = [], coordinate_tensors = None, memory_for_cache = 5,
            prune_domain : bool = False, pivotal_tensor_label = None, pruner = None, 
            threshold : float = 1e-2, division_fractions = 3, rectangular : bool = True, 
            data_fun_pow : int = 1):
        '''
        
        Fit epde search algorithm to obtain differential equations, describing passed data.
        
        Parameters:
        -------------
        
        data : np.ndarray, list or tuple,
            Values of modeled variables. If the variable is single (i.e. deriving a single equation),
            it can be passed as the numpy.ndarray or as the list/tuple with a single element;
            multiple variables are not supported yet, use older interfaces.
            
        time_axis : int, optional,
            Index of the axis in the data, representing time. Default value: 0.
            
        boundary : int, optional,
            The number of grid nodes, considered as the boundary of the studied domain,
            that are excluded during equation search to avoid wrong results, linked to 
            errors in derivatives estimation. Default value: 0.
            
        equation_terms_max_number : int, optional,
            The maximum number of terms, present in the derived equations. Default value: 6.
            
        equation_factors_max_number : int, optional,
            The maximum number of factors (token functions; real-valued coefficients are not counted here), 
            present in terms of the equaton. Default value: 1.
            
        variable_names : list of str, optional,
            Names of the independent variables, passed into search mechanism. Length of the list must correspond
            to the number of np.ndarrays, sent with in ``data`` parameter. Defalut value: ``['u',]``, representing 
            a single variable *u*.
            
        eq_sparsity_interval : tuple of two float values, optional,
            The left and right boundaries of interval with sparse regression values. Influence the number of 
            terms in the equation. Default value: ``(1e-4, 2.5)``.
            
        derivs : list of lists of np.ndarrays or None, optional,
            Pre-computed values of derivatives. If ``None`` is passed, the derivatives are calculated in the 
            method. Recommended to use, if the computations of derivatives take too long. For further information 
            about using data, prepared in advance, check ``epde.prep.derivatives.Preprocess_derivatives`` function.
            Default value: None.
            
        max_deriv_order : int or tuple/list, optional,
            Highest order of calculated derivatives. Default value: 1.
            
        additional_tokens : None or list of ``TokenFamily`` or ``Prepared_tokens`` objects, optional
            Additional tokens, that would be used to construct the equations among the main variables and their 
            derivatives. Objects of this list must be of type ``epde.interface.token_family.TokenFamily`` or 
            of ``epde.interface.prepared_tokens.Prepared_tokens`` subclasses types. Default value: None.
        
        coordinate_tensors : list of np.ndarrays, optional
            Values of the coordinates on the grid nodes with studied functions values. In case of 1D-problem, 
            that will be ``numpy.array``, while the parameter for higher dimensionality problems can be set from 
            ``numpy.meshgrid`` function. With None, the tensors will be created as ranges with step of 1 between 
            nodes. Defalut value: None. 
            
        field_smooth : bool, optional
            Parameter, if the input variable fields shall be smoothed to avoid the errors. If the data is 
            assumed to be noiseless, shall be set to False, otherwise - True. Default value: False.

        memory_for_cache : int or float, optional
            Limit for the cache (in fraction of the memory) for precomputed tensor values to be stored: 
            if int, will be considered as the percentage of the entire memory, and if float, 
            then as a fraction of memory.
            
        data_fun_pow : int
            Maximum power of token,
            
        '''
        if equation_terms_max_number < self.moeadd_params['pop_size']:
            self.moeadd_params['pop_size'] = equation_terms_max_number
            self.moeadd_params['weights_num'] = equation_terms_max_number
        
        self.create_pool(data = data, time_axis=time_axis, boundary=boundary, variable_names=variable_names, 
                         derivs=derivs, method=deriv_method, method_kwargs=deriv_method_kwargs, 
                         max_deriv_order=max_deriv_order, additional_tokens=additional_tokens, 
                         coordinate_tensors=coordinate_tensors, memory_for_cache=memory_for_cache, 
                         prune_domain=prune_domain, pivotal_tensor_label=pivotal_tensor_label, 
                         pruner=pruner, threshold=threshold, division_fractions=division_fractions, 
                         rectangular=rectangular, data_fun_pow=data_fun_pow)
        
        pop_constructor = operators.Systems_population_constructor(pool = self.pool, terms_number = equation_terms_max_number, 
                                                               max_factors_in_term=equation_factors_max_number, 
                                                               eq_search_evo=self.director.constructor.strategy,
                                                               sparsity_interval = eq_sparsity_interval)

        self.moeadd_params['pop_constructor'] = pop_constructor
        self.optimizer = moeadd_optimizer(**self.moeadd_params)
        evo_operator = operators.sys_search_evolutionary_operator(operators.mixing_xover, 
                                                                  operators.gaussian_mutation)
        self.optimizer.set_evolutionary(operator=evo_operator)        
        best_obj = np.concatenate((np.ones([1,]),
                                  np.zeros(shape=len([1 for token_family in self.pool.families if token_family.status['meaningful']]))))  
        self.optimizer.pass_best_objectives(*best_obj)
        self.optimizer.optimize(**self.moeadd_optimization_params)

        print('The optimization has been conducted.')
        self.search_conducted = True
        
    @property
    def equations_pareto_frontier(self):
        if not self.search_conducted:
            raise AttributeError('Pareto set of the best equations has not been discovered. Use ``self.fit`` method.')
        return self.optimizer.pareto_levels.levels
    
    def equation_search_results(self, only_print : bool = True, level_num = 1):
        if only_print:
            for idx in range(min(level_num, len(self.equations_pareto_frontier))):
                print('\n')
                print(f'{idx}-th non-dominated level')    
                print('\n')                
                [print(f'{solution.text_form} , with objective function values of {solution.obj_fun} \n')  
                for solution in self.equations_pareto_frontier[idx]]
        else:
            return self.optimizer.pareto_levels.levels[:level_num]
        
    def solver_forms(self):
        '''
        Method returns solver forms of the equations on 1-st non-dominated levels in a form of Python list.
        '''
        sf = []
        for system in self.equations_pareto_frontier[0]:
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

