"""

Inteface objects for EPDE framework

Contains:
---------

**VariableEntry** class, containing logic for preparing the input data for the equation search,
such as initialization of neccessary token families and derivatives calculation.

**EpdeSearch** class for main interactions between the user and the framework.

"""
import inspect
import pickle
import warnings
import numpy as np
import torch

from copy import deepcopy
from typing import Union, Callable, List, Tuple, Dict
from collections import OrderedDict
from functools import reduce, singledispatchmethod

import epde.globals as global_var
from epde.operators.common.objectives import ideal_point
from epde.interface.search_config import (UNSET, build_tokens, collect_overrides,
                                          load_search_config, set_active_config)
from epde.interface.legacy_api import (LEGACY_DATA_KEYS, LEGACY_INIT_KEYS,
                                       reject_removed, split_legacy, warn_legacy)

from epde import _loop_stats

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
from epde.cache.cache_refactored import Cache # upload_simple_tokens, upload_grids, prepareVarTensor, 

from epde.preprocessing.preprocessor_setups import PreprocessorSetup
from epde.preprocessing.preprocessor import ConcretePrepBuilder, PreprocessingPipe

from epde.structure.domain import VariableEntry, Trajectory, TrajectoriesManager, Domain
from epde.structure.main_structures import Equation, SoEq

from epde.interface.token_family import TFPool, TokenFamily
from epde.interface.type_checks import *
from epde.interface.prepared_tokens import PreparedTokens, CustomTokens, DataPolynomials
from epde.integrate import BoundaryConditions, SolverAdapter, SystemSolverInterface


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
    def __init__(self, config=None, *, director=None,
                 multiobjective_mode=UNSET, second_objective=UNSET,
                 complexity_metric=UNSET, instability_metric=UNSET,
                 single_objective_metric=UNSET, discrepancy_metric=UNSET,
                 anchor_on_residual=UNSET, sparsity_cls=UNSET,
                 sparsity_kwargs=UNSET,
                 use_solver=UNSET, solver_backend=UNSET, device=UNSET,
                 boundary_width=UNSET, time_axis=UNSET,
                 default_preprocessor_type=UNSET, preprocessor_kwargs=UNSET,
                 max_deriv_order=UNSET,
                 data_fun_pow=UNSET, deriv_fun_pow=UNSET,
                 equation_terms_max_number=UNSET,
                 equation_factors_max_number=UNSET,
                 rps_amplification_cap=UNSET, tokens=UNSET,
                 population_size=UNSET, training_epochs=UNSET,
                 neighbors_number=UNSET, PBI_penalty=UNSET,
                 subregion_mating_limitation=UNSET, solution_params=UNSET,
                 director_params=UNSET, operators=UNSET,
                 memory_for_cache=UNSET, verbose_params=UNSET,
                 params_filename=UNSET, **solver_kwargs):
        """Build a search from the grouped configuration.

        Every setting has a default in
        ``epde/interface/parameters/default_search_config.json``, grouped by
        concern. Resolution order is strictly::

            built-in JSON  <  ``config``  <  the keyword arguments here

        so a keyword always wins, whatever group its key belongs to --
        ``EpdeSearch(use_solver=True)`` enables the solver without the caller
        needing to know that ``use_solver`` lives under ``solver``. Pass
        ``None`` explicitly to override a file value back to "unset"; omit an
        argument entirely to leave the resolved value alone.

        Args:
            config: path to a JSON/YAML search config, a nested
                ``{group: {key: value}}`` dict, or None for the shipped
                defaults. See :mod:`epde.interface.search_config`.
            director: a pre-built optimization strategy director. When given
                it is used as-is and the ``objectives``/``solver`` groups do
                not assemble one. (This replaces the former
                ``use_default_strategy`` flag, which carried no information
                ``director is None`` did not already carry, and whose
                ``director=None, use_default_strategy=False`` combination
                raised ``NotImplementedError``.)
            **solver_kwargs: the remaining ``solver`` group keys
                (``pinn_loss_mult``, ``error_metric``, ``deepxde_config``,
                ``mode``, ``use_cache``, ``use_fourier``, ``fourier_params``,
                ``use_adaptive_lambdas`` and ``predict``'s six ``*_params``
                dicts). Unknown names raise ``ValueError`` naming the valid
                parameters by group.

        Note:
            The objective settings are process globals, so the last search
            constructed wins for the process -- a long-standing trade-off,
            unchanged here.
        """
        reject_removed(solver_kwargs)
        legacy, solver_kwargs = split_legacy(solver_kwargs, LEGACY_INIT_KEYS)
        warn_legacy('EpdeSearch(...)', legacy,
                    'Build the region with createDomain(grids, boundary_width=...) '
                    'and attach data with createTrajectory(...).')

        overrides = collect_overrides(
            multiobjective_mode=multiobjective_mode,
            second_objective=second_objective,
            complexity_metric=complexity_metric,
            instability_metric=instability_metric,
            single_objective_metric=single_objective_metric,
            discrepancy_metric=discrepancy_metric,
            anchor_on_residual=anchor_on_residual,
            sparsity_cls=sparsity_cls, sparsity_kwargs=sparsity_kwargs,
            use_solver=use_solver, solver_backend=solver_backend, device=device,
            boundary_width=boundary_width, time_axis=time_axis,
            default_preprocessor_type=default_preprocessor_type,
            preprocessor_kwargs=preprocessor_kwargs,
            max_deriv_order=max_deriv_order,
            data_fun_pow=data_fun_pow, deriv_fun_pow=deriv_fun_pow,
            equation_terms_max_number=equation_terms_max_number,
            equation_factors_max_number=equation_factors_max_number,
            rps_amplification_cap=rps_amplification_cap, tokens=tokens,
            population_size=population_size, training_epochs=training_epochs,
            neighbors_number=neighbors_number, PBI_penalty=PBI_penalty,
            subregion_mating_limitation=subregion_mating_limitation,
            solution_params=solution_params, director_params=director_params,
            operators=operators, memory_for_cache=memory_for_cache,
            verbose_params=verbose_params, params_filename=params_filename)
        overrides.update(solver_kwargs)
        self._config = cfg = load_search_config(config, overrides)

        self.multiobjective_mode = cfg.objectives.multiobjective_mode

        # Must be published BEFORE the director is built: ``use_baseline``
        # resolves the second Pareto axis at assembly time, and the fillers it
        # builds are fixed from then on.
        set_active_config(cfg)

        global_var.init_verbose(**cfg.runtime.verbose_params)

        self.preprocessor_set = False
        self._cache_mem_set = False
        self._mem_for_cache = None
        self._g_func = None

        self._create_caches(cfg.runtime.memory_for_cache)

        criteria = 'multi objective' if self.multiobjective_mode else 'single objective'

        # Singleton, read during operator initialization below.
        EvolutionaryParams.reset()
        EvolutionaryParams(parameter_file=cfg.runtime.params_filename, mode=criteria)

        if director is not None:
            self.director = director
        else:
            if self.multiobjective_mode:
                self.director = MOEADDDirector()
                builder = StrategyBuilder(MOEADDSectorProcesser)
            else:
                self.director = BaselineDirector()
                builder = StrategyBuilder(EvolutionaryStrategy)
            self.director.builder = builder
            # One merged override dict rather than two splats: the solver
            # keys and evolution.operators both target the same
            # add_base_param_to_operator path, and naming a key in both would
            # otherwise be a "got multiple values" TypeError. An explicit
            # operators entry wins, being the more specific statement.
            operator_overrides = self._solver_operator_overrides(cfg)
            operator_overrides.update(cfg.evolution.operators)
            self.director.use_baseline(
                use_solver=cfg.solver.use_solver,
                second_objective=cfg.objectives.second_objective,
                solver_backend=cfg.solver.solver_backend,
                params=cfg.evolution.director_params,
                sparsity_cls=cfg.objectives.sparsity_cls,
                sparsity_kwargs=cfg.objectives.sparsity_kwargs,
                **operator_overrides)

        # The axis the director actually assembled -- a user-supplied director
        # has not been through use_baseline, so fall back to the global. Every
        # later consumer (population_instruct, the ideal point) reads this,
        # never the raw constructor argument.
        self._second_objective = (getattr(self.director, 'second_objective', None)
                                  or cfg.objectives.second_objective)

        if self.multiobjective_mode:
            self.set_moeadd_params()
        else:
            self.set_singleobjective_params()

        self.pool = None
        self.search_conducted = False

        # Legacy: a search used to own exactly one implicit domain, built from
        # the constructor's coordinate_tensors. Keep it so the old
        # create_pool/fit forms have something to attach trajectories to.
        self._legacy_domain = None
        if legacy.get('coordinate_tensors') is not None:
            boundary = legacy.get('boundary', UNSET)
            self._legacy_domain = self.createDomain(
                legacy['coordinate_tensors'],
                boundary_width=UNSET if boundary is None else boundary,
                gfunction=legacy.get('function_form'))[1]

    @property
    def config(self):
        """The resolved :class:`SearchConfig` backing this search."""
        return self._config

    @staticmethod
    def _solver_operator_overrides(cfg) -> dict:
        """The ``solver`` keys that are also operator parameters.

        ``pinn_loss_mult`` / ``error_metric`` / ``deepxde_config`` are declared
        in the ``SolverBasedFitness`` block of the operator JSON, so
        ``use_baseline``'s existing ``**kwargs`` -> ``add_base_param_to_operator``
        path already forwards them. Passing them here makes the ``solver``
        group authoritative while the JSON keeps supplying the fallback.

        Forwarded only for a solver run: ``add_base_param_to_operator`` only
        adopts keys the target operator's JSON block declares, and
        ``SolverFreeFitness`` declares none of these.
        """
        if not cfg.solver.use_solver:
            return {}
        return {'pinn_loss_mult': cfg.solver.pinn_loss_mult,
                'error_metric': cfg.solver.error_metric,
                'deepxde_config': cfg.solver.deepxde_config}

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
        if self._cache_mem_set:
            return
        if mem_for_cache_frac is None and mem_for_cache_abs is None:
            raise ValueError(
                'set_memory_properties needs either mem_for_cache_frac or '
                'mem_for_cache_abs; both were None. (It used to reach '
                'int(None/2.) and raise TypeError here.)')
        if global_var.grid_cache is not None:
            # The grid cache holds a second copy of comparable size, so the
            # tensor cache gets half the budget.
            if mem_for_cache_frac is not None:
                mem_for_cache_frac = int(mem_for_cache_frac / 2.)
            else:
                mem_for_cache_abs = int(mem_for_cache_abs / 2.)
        global_var.tensor_cache.memoryUsageProperties(example_tensor, mem_for_cache_frac, mem_for_cache_abs)
        self._cache_mem_set = True

    def set_moeadd_params(self, population_size=UNSET, solution_params=UNSET,
                          neighbors_number=UNSET,
                          nds_method: Callable = fast_non_dominated_sorting,
                          ndl_update_method: Callable = ndl_update,
                          subregion_mating_limitation=UNSET,
                          PBI_penalty=UNSET, training_epochs=UNSET,
                          early_stopping_callback: Callable = None):
        r"""
        Setting the parameters of the multiobjective evolutionary algorithm. declaration of
        the default values is held in the initialization of EpdeSearch object.

        Args:
            population_size (`int`): optional
                The size of the population of solutions, created during MO - optimization, default 6.
            solution_params (`dict`): optional
                Dictionary, containing additional parameters to be sent into the newly created solutions.
            H (`float`): optional
                parameter of uniform spacing between the weight vectors; *H = 1 / delta*
                should be integer - a number of divisions along an objective coordinate axis.
                NOTE: currently ignored — the optimizer always uses *H = population_size - 1*,
                which (for the two-objective weight space used by EPDE) keeps the number of
                Das-Dennis weight vectors equal to the population size, as the MOEA/DD paper
                requires (N solutions <-> N weight vectors).
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
            subregion_mating_limitation (`float`): optional
                The probability of mating selection to be limited only to the selected
                subregions (adjacent to the weight vector domain). :math:`\delta \in [0., 1.)`,
                default value is 0.9, as in the MOEA/DD paper.
            training_epochs (`int`): optional
                Maximum number of iterations, during that the optimization will be held.
                Note, that if the algorithm converges to a single Pareto frontier,
                the optimization is stopped.
            PBI_penalty (`float`):  optional
                The penalty parameter :math:`\\theta`, used in penalty based intersection
                calculation, default value is 5.0, as in the MOEA/DD paper.

        Returns:
            None
        """
        evolution = self._config.evolution
        if population_size is UNSET:
            population_size = evolution.population_size
        if solution_params is UNSET:
            solution_params = evolution.solution_params
        if neighbors_number is UNSET:
            neighbors_number = evolution.neighbors_number
        if subregion_mating_limitation is UNSET:
            subregion_mating_limitation = evolution.subregion_mating_limitation
        if PBI_penalty is UNSET:
            PBI_penalty = evolution.PBI_penalty
        if training_epochs is UNSET:
            training_epochs = evolution.training_epochs

        # H is always population_size - 1: for the two-objective weight space
        # EPDE uses, that keeps the Das-Dennis weight vectors equal in number
        # to the population, as MOEA/DD requires. It used to be a parameter
        # that was accepted and then immediately overwritten by exactly this.
        self.optimizer_init_params = {'pop_size': population_size,
                              'H': population_size-1, 'neighbors_number': neighbors_number,
                              'solution_params': solution_params,
                              'nds_method' : nds_method,
                              'ndl_update' : ndl_update_method}

        self.optimizer_exec_params = {'epochs' : training_epochs,
                                      'early_stopping_callback' : early_stopping_callback}

        # Forward the user-facing MOEA/DD parameters to the operators of the
        # strategy assembled in __init__ (previously these arguments were
        # accepted but silently dropped, so the JSON defaults always applied).
        director = getattr(self, 'director', None)
        if director is not None and director.builder is not None:
            blocks = director.builder.blocks_labeled
            if 'selection' in blocks:
                selection = blocks['selection']._operator
                selection.params['delta'] = subregion_mating_limitation
                # The operator-level neighbourhood count and the optimizer-level
                # one are the same concept; the operator half used to be
                # reachable only from the operator JSON, so the two could
                # silently disagree.
                neighborhood = selection.suboperators['neighborhood_selector']
                if 'number_of_neighbors' in neighborhood.params:
                    neighborhood.params['number_of_neighbors'] = neighbors_number
            if 'pareto_updater_compl' in blocks:
                pareto_updater = blocks['pareto_updater_compl']._operator
                pareto_updater.suboperators['pareto_level_updater'].params['PBI_penalty'] = PBI_penalty

    def set_singleobjective_params(self, population_size=UNSET, solution_params=UNSET,
                                   sorting_method: Callable = simple_sorting,
                                   training_epochs=UNSET):
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
        evolution = self._config.evolution
        if population_size is UNSET:
            population_size = evolution.population_size
        if solution_params is UNSET:
            solution_params = evolution.solution_params
        if training_epochs is UNSET:
            training_epochs = evolution.training_epochs

        self.optimizer_init_params = {'pop_size' : population_size, 'solution_params': solution_params,
                                      'sorting_method' : sorting_method}
        
        self.optimizer_exec_params = {'epochs' : training_epochs}        


    def _create_caches(self, memory_for_cache: Union[int, float]): # , coordinate_tensors, memory_for_cache):
        """
        Creating caches for keeping tensors during EPDE search.
        
        Args:
            None
        
        Returns:
            None
        """
        # No ``device``: the caches are numpy, and ``Cache.__init__`` raises
        # NotImplementedError for device='cuda' with the numpy backend, which
        # made EpdeSearch(device='cuda') unconstructible. The GPU is only ever
        # used by the solver, so the device lives in the ``solver`` group and
        # reaches SystemSolverInterface, not the caches. Cache's own
        # device/backend parameters stay, for the cupy work.
        global_var.init_caches(set_grids=True)
        self._mem_for_cache = memory_for_cache
        print(f'Set self._mem_for_cache as {self._mem_for_cache}')

        # example = coordinate_tensors if isinstance(coordinate_tensors, np.ndarray) else coordinate_tensors[0]
        # self.set_memory_properties(example_tensor=example, mem_for_cache_frac=memory_for_cache)
        # upload_grids(coordinate_tensors, global_var.initial_data_cache)
        # upload_grids(coordinate_tensors, global_var.grid_cache)

    def createDomain(self, grids: Union[np.ndarray, Tuple[np.ndarray], List[np.ndarray]],
                     time_axis=UNSET, ID: int = 0, gfunction: Callable = None,
                     boundary_width=UNSET) -> Tuple[int, Domain]:
        """Register a sampled region. Omitted arguments come from the
        ``domain`` config group.

        ``boundary_width`` is deliberately used twice below -- for the domain's
        own bookkeeping and for the test function's ``BoundaryExclusion`` width.
        They are one concept and must agree: ``setBoundaries`` only records
        ``inner_shape``, while the points are actually excluded once, by
        ``g_func_mask``. If the two ever diverged, ``inner_shape`` would
        misdescribe the masked data.
        """
        domain_cfg = self._config.domain
        if time_axis is UNSET:
            time_axis = domain_cfg.time_axis
        if boundary_width is UNSET:
            boundary_width = domain_cfg.boundary_width

        self.set_memory_properties(grids[0], mem_for_cache_frac=self._mem_for_cache)

        domain = Domain(grids, self.grid_cache, time_axis, ID, boundary=boundary_width)
        domain.g_func = self._get_g_func(gfunction, boundary_width)
        # TODO: consider implementing domain.set_pruner()
        
        return domain.ID, domain
    
    def createTrajectory(self, entries: Union[List[VariableEntry], Dict[str, np.ndarray]],
                         domain: Domain, cache_id = None,
                         derivs: Union[List[np.ndarray], np.ndarray] = None,
                         cached_token_tensors: Dict[CustomTokens, Dict[str, np.ndarray]] = None,
                         preprocessor: PreprocessingPipe = None) -> Tuple[int, Trajectory]:
        """Attach data to a domain.

        A trajectory is a data sample: the tensors, the domain they live on,
        any pre-computed ``derivs`` and the pipeline that differentiates them.
        It deliberately does NOT carry ``max_deriv_order``, ``data_fun_pow`` or
        ``deriv_fun_pow`` -- those describe the token pool, which is a single
        structure shared by every trajectory feeding it, so they are
        ``create_pool``/``fit`` arguments. Trajectories differ in evaluation
        only, which is also why declarative token families (``GridTokens``,
        ``CacheStoredTokens``, ...) stay on ``create_pool``/``fit`` as well.

        ``cached_token_tensors`` was called ``additional_tokens``, which
        collided with the same name on ``create_pool``/``fit`` -- there it is a
        list of token FAMILIES, here a ``{CustomTokens: {label: tensor}}``
        upload map. The new name matches the ``Trajectory`` parameter it
        forwards to.
        """
        if isinstance(entries, list):
            self.set_memory_properties(entries[0].data_tensor, mem_for_cache_frac=self._mem_for_cache)
        elif isinstance(entries, dict):
            self.set_memory_properties(list(entries.values())[0], mem_for_cache_frac=self._mem_for_cache)

        if preprocessor is None:
            # Guard the auto-setup, not the argument: an explicitly supplied
            # pipeline used to trip the ``preprocessor_set`` assert on a fresh
            # search, because nothing had set the flag.
            if not self.preprocessor_set:
                self.set_preprocessor()
            preprocessor = self.preprocessor_pipeline

        trajectory = Trajectory(entries, domain, self.tensor_cache, cache_id,
                                preprocessor, cached_token_tensors, derivs)
        # trajectory.
        return trajectory.ID, trajectory

    @property
    def grid_cache(self) -> Cache:
        return global_var.grid_cache
    
    @property
    def tensor_cache(self) -> Cache:
        return global_var.tensor_cache

    def set_boundaries(self, boundary_width: Union[int, list]):
        """
        Setting the number of unaccountable elements at the edges into cache with saved grid.
        """
        raise NotImplementedError('Method has been depricated!')
        # global_var.grid_cache.set_boundaries(boundary_width=boundary_width)

    @staticmethod
    def _get_g_func(function_form: Union[Callable, np.ndarray, list] = None, boundary_width: int = 5): # self, 
        """
        Loading testing function connected to the weak derivative notion.

        Args:
            function_form (`callable`, or `np.ndarray`, or `list[np.ndarray]`)
                Test function, default using inverse polynomial with max in the domain center.

        Returns:
            None
        """
        if isinstance(function_form, (np.ndarray, list)):
            return function_form
        else:
            try:
                bound_excl_decorator = BoundaryExclusion(boundary_width=boundary_width)
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

                    def return_ones(grids):
                        ones_partial = np.array([np.ones_like(grid) for grid in grids])
                        ones = np.multiply.reduce(ones_partial, axis=0)
                        return ones
                    
                    # global_var.grid_cache.g_func = bound_excl_decorator(baseline_exp_function)
                    return bound_excl_decorator(return_ones)
                else:
                    return bound_excl_decorator(function_form)

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
        raise NotImplementedError('Method depricated!')
        self._create_caches(coordinate_tensors=coordinate_tensors, memory_for_cache=memory_for_cache)
        if prune_domain:
            self.domain_pruning(pivotal_tensor_label, pruner, threshold, division_fractions, rectangular)
        self.set_boundaries(boundary_width)
        self._upload_g_func(function_form)

    def set_preprocessor(self, preprocessor_pipeline: PreprocessingPipe = None,
                         default_preprocessor_type=UNSET, preprocessor_kwargs=UNSET):
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
        if default_preprocessor_type is UNSET:
            default_preprocessor_type = self._config.preprocessing.default_preprocessor_type
        if preprocessor_kwargs is UNSET:
            preprocessor_kwargs = self._config.preprocessing.preprocessor_kwargs

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
                raise NotImplementedError(
                    f'Incorrect default preprocessor type {default_preprocessor_type!r}. '
                    "Allowed: 'poly', 'ANN', 'spectral', 'FD'.")
            preprocessor_pipeline = setup.builder.prep_pipeline

        if 'max_order' not in preprocessor_pipeline.deriv_calculator_kwargs.keys():
            preprocessor_pipeline.deriv_calculator_kwargs['max_order'] = None

        self.preprocessor_set = True
        self.preprocessor_pipeline = preprocessor_pipeline

    def create_pool(self, data: Union[Trajectory, List[Trajectory]],
                    additional_tokens=None, max_deriv_order=UNSET,
                    data_fun_pow=UNSET, deriv_fun_pow=UNSET, **legacy_kwargs):
        '''
        Create pool of tokens to represent elementary functions, that can be included in equations.

        Args:
            data : Trajectory | list of Trajectory
            additional_tokens : token families to add beside the ones derived
                from the data. Declarative families (``GridTokens``,
                ``CacheStoredTokens``, ``TrigonometricTokens``, ...) belong
                here rather than on a trajectory: the pool is one structure,
                and trajectories differ in evaluation only.
            max_deriv_order : highest derivative order to compute and to offer
                as tokens. Defaults to ``preprocessing.max_deriv_order``.
            data_fun_pow, deriv_fun_pow : highest powers the variable and
                derivative token families accept. Default to
                ``search_space.data_fun_pow`` / ``deriv_fun_pow``.

        ``max_deriv_order``/``data_fun_pow``/``deriv_fun_pow`` describe the
        pool, not the sample, so they are resolved here and pushed into every
        trajectory through ``Trajectory.build``.
        '''
        # if isinstance(data, Trajectory):
        #     data = [data,]

        max_deriv_order, data_fun_pow, deriv_fun_pow = self._pool_structure(
            max_deriv_order, data_fun_pow, deriv_fun_pow)

        additional_tokens = self._resolve_token_families(additional_tokens)
        data = self._as_trajectories(data, legacy_kwargs, 'create_pool')
        assert isinstance(data, list), f'On this stage, data has to be a list of Trajectory objs., instead got {type(data)}.'
        for trajectory in data:
            assert isinstance(trajectory, Trajectory), \
                f'Individual trajectories have to be passed as Trajectory objects, instead got {type(trajectory)}.'
            trajectory.build(max_deriv_order, data_fun_pow, deriv_fun_pow)
            # A family that ships its own tensors (CacheStoredTokens) declares
            # them once, with the family, because a family is pool structure.
            # Evaluation is per trajectory, so the tensors land in every
            # trajectory's subcache -- which is where the evaluator reads them
            # (samples_manager.get walks the trajectories).
            for family in additional_tokens:
                tensors = getattr(family, 'token_tensors', None)
                if tensors:
                    trajectory.uploadTokenTensors(family, tensors)
        self.pool_params = cur_params = self._pool_params(
            data, additional_tokens, max_deriv_order, data_fun_pow, deriv_fun_pow)

        if isinstance(data, Trajectory):
            data_tokens = data.families
            base_derivs = data.base_derivs
        elif isinstance(data, (list, tuple)):
            # data_tokens = []
            # for entry in data:
            assert isinstance(data[0], Trajectory), \
                f'Individual trajectories have to be passed as Trajectory objects, instead got {type(data[0])}.'
            data_tokens = data[0].families
            base_derivs = data[0].base_derivs

        # if isinstance(data, np.ndarray):
        #     data = [data,]

        # if derivs is None:
        #     if len(data) != len(variable_names):
        #         msg = f'Mismatching nums of data tensors {len(data)} and the names of the variables { len(variable_names)}'
        #         raise ValueError(msg)
        # else:
        #     if not (len(data) == len(variable_names) == len(derivs)):
        #         raise ValueError('Mismatching lengths of data tensors, names of the variables and passed derivatives')

        # if not self.preprocessor_set:
        #     self.set_preprocessor()

        # if self._mode_info['solver_fitness']: 
        #     base_derivs = []
            
        # for data_elem_idx, data_tensor in enumerate(data):
        #     entry = VariableEntry(var_name=variable_names[data_elem_idx], var_idx=data_elem_idx,
        #                            data_tensor=data_tensor, boundary=self.cache[0].g_func)
        #     derivs_tensor = derivs[data_elem_idx] if derivs is not None else None
        #     entry.setDerivatives(preprocesser=self.preprocessor_pipeline, deriv_tensors=derivs_tensor,
        #                           grid=grid, max_order=max_deriv_order)
        #     # entry.use_global_cache()

        #     self.save_derivatives(variable=variable_names[data_elem_idx], deriv=entry.derivatives)  
        #     entry.create_derivs_family(max_deriv_power=deriv_fun_pow)
        #     entry.create_polynomial_family(max_power=data_fun_pow)
        #     if self._mode_info['solver_fitness']:
        #         base_derivs.extend(entry.matched_derivs(max_order = 2)) # TODO: add setup of Sobolev learning order
                
        #     data_tokens.extend(entry.get_families())

        # TODO: refactor! It is neccessary. Each sample must have a separate ANN?
        if self._config.solver.use_solver:
            warnings.warn('Missing code for ANN pretraining!')
        
        #     if data_nn is not None:
        #         print('Using pre-trained ANN')
        #         global_var.reset_data_repr_nn(data = data, derivs = base_derivs, train = False, 
        #                                       grids = grid, predefined_ann = data_nn, device = self._device)
        #     else:
        #         pass
        #         # epochs_max = 1e5 # 1e4
                # global_var.reset_data_repr_nn(data = data, derivs = base_derivs, epochs_max=ann_epochs_max,
                #                               grids = grid, predefined_ann = None, device = self._device,
                #                               use_fourier = fourier_layers, fourier_params = fourier_params)

        for traj in data:
            global_var.samples_manager.addTrajectory(traj, domain = traj._domain)

        bad = [tf for tf in additional_tokens
               if not isinstance(tf, (TokenFamily, PreparedTokens))]
        if bad:
            raise TypeError('Incorrect type of additional tokens: expected '
                            f'TokenFamily/PreparedTokens objects, got {type(bad[0])}.')
        self.pool = TFPool(data[0].tokens + [tf if isinstance(tf, TokenFamily) else tf.token_family
                                             for tf in additional_tokens])
        #TODO: add check, if all trajectories have the same tokens 
        print(f'The cardinality of defined token pool is {self.pool.families_cardinality()}')
        print(f'Among them, the pool contains {self.pool.families_cardinality(meaningful_only=True)}')
        for family in self.pool.families:
            family.chech_constancy()
        
    def _as_trajectories(self, data, legacy_kwargs: dict, where: str):
        """Accept the pre-refactor data form and return Trajectory objects.

        Old form::

            search.fit(data=[x, y], variable_names=['u', 'v'],
                       max_deriv_order=(1,), data_fun_pow=1, ...)

        i.e. raw arrays plus the names, against the single implicit domain the
        constructor built from ``coordinate_tensors``. New form: build a
        Trajectory yourself with ``createTrajectory({'u': x, 'v': y}, domain)``
        and pass that. Only the ARRAYS move; ``max_deriv_order`` and the two
        powers describe the pool rather than the sample and stay right where
        they are, as ``fit``/``create_pool`` arguments.
        """
        reject_removed(legacy_kwargs)
        legacy, unknown = split_legacy(legacy_kwargs, LEGACY_DATA_KEYS)
        if unknown:
            raise TypeError('{0}() got unexpected keyword argument(s) {1}.'.format(
                where, sorted(unknown)))

        already_new = (data is None or isinstance(data, Trajectory) or
                       (isinstance(data, (list, tuple)) and data and
                        isinstance(data[0], Trajectory)))
        if already_new:
            warn_legacy('{0}(...)'.format(where), legacy,
                        'These are createTrajectory arguments now and are '
                        'ignored here, because the data was passed as '
                        'Trajectory objects that already carry them.',
                        stacklevel=4)
            return data

        warn_legacy('{0}(...)'.format(where), legacy,
                    "Build the data with createTrajectory({'u': array}, domain) "
                    'and pass the resulting Trajectory instead of raw arrays.',
                    stacklevel=4)

        domain = self._legacy_domain
        if domain is None:
            raise ValueError(
                '{0}() was given raw arrays, which is the pre-domain_refactor '
                'form, but this search has no domain to attach them to. Either '
                'pass coordinate_tensors=... to EpdeSearch (the old form), or '
                'build the data with createDomain/createTrajectory (the current '
                'one).'.format(where))

        arrays = [data] if isinstance(data, np.ndarray) else list(data)
        names = legacy.get('variable_names') or ['u']
        if len(names) != len(arrays):
            raise ValueError(
                'Mismatching numbers of data tensors ({0}) and variable names '
                '({1}).'.format(len(arrays), len(names)))

        forwarded = {key: legacy[key] for key in ('derivs',) if key in legacy}
        return [self.createTrajectory(dict(zip(names, arrays)), domain,
                                      cache_id=0, **forwarded)[1]]

    def _pool_structure(self, max_deriv_order, data_fun_pow, deriv_fun_pow):
        """Resolve the three settings that describe the pool, not the sample."""
        if max_deriv_order is UNSET:
            max_deriv_order = self._config.preprocessing.max_deriv_order
        if data_fun_pow is UNSET:
            data_fun_pow = self._config.search_space.data_fun_pow
        if deriv_fun_pow is UNSET:
            deriv_fun_pow = self._config.search_space.deriv_fun_pow
        return max_deriv_order, data_fun_pow, deriv_fun_pow

    @staticmethod
    def _pool_params(data, additional_tokens, max_deriv_order=None,
                     data_fun_pow=None, deriv_fun_pow=None) -> dict:
        """The key that decides whether an existing pool can be reused.

        ``create_pool`` used to compute this and only print it, leaving
        ``self.pool_params`` unset -- so a SECOND ``fit`` on the same object
        raised AttributeError on the very check meant to protect it.

        The orders and powers are taken from the REQUEST rather than off the
        trajectory, because they are what the caller asked the pool to be
        built for; all three change which token families exist, so all three
        have to invalidate it.

        ``ftype`` is read off either a bare ``TokenFamily`` or a
        ``PreparedTokens`` wrapper: ``create_pool`` accepts both, while this
        key used to assume the wrapper and raise AttributeError on the former.
        """
        if data is None:
            return None
        return {'variable_names': data[0].variable_names,
                'max_deriv_order': max_deriv_order,
                'data_fun_pow': data_fun_pow,
                'deriv_fun_pow': deriv_fun_pow,
                'additional_tokens': [
                    (family if isinstance(family, TokenFamily)
                     else family.token_family).ftype
                    for family in additional_tokens]}

    def _resolve_token_families(self, additional_tokens) -> list:
        """Configured token families plus the ones passed at the call site.

        The kwarg APPENDS rather than replacing, which is a deliberate
        exception to the "a kwarg replaces wholesale" rule elsewhere: the two
        sources carry disjoint kinds. Declarative families
        (``TrigonometricTokens``, ``GridTokens``, ...) can live in the config;
        the ones that need tensors or callables (``CacheStoredTokens``,
        ``CustomTokens``, ``ExternalDerivativesTokens``, ...) cannot, so
        replacing would make ``search_space.tokens`` useless for exactly the
        scripts that need it. Set ``tokens: []`` in the config to drop the
        configured families.
        """
        families = build_tokens(self._config.search_space.tokens)
        if additional_tokens is None:
            return families
        if isinstance(additional_tokens, (TokenFamily, PreparedTokens)):
            additional_tokens = [additional_tokens]
        return families + list(additional_tokens)

    def save_derivatives(self, variable:str, deriv: Dict[int, np.ndarray]):
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

    @_loop_stats.timed('EpdeSearch.fit')
    def fit(self, data: Union[Trajectory, List[Trajectory]] = None,
            equation_terms_max_number=UNSET, equation_factors_max_number=UNSET,
            eq_sparsity_interval=UNSET, additional_tokens=None,
            max_deriv_order=UNSET, data_fun_pow=UNSET, deriv_fun_pow=UNSET,
            optimizer: Union[SimpleOptimizer, MOEADDOptimizer] = None, pool: TFPool = None,
            population: List[SoEq] = None, **legacy_kwargs):
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
            Interval the ``('sparsity', var)`` metaparameter -- the legacy LASSO
            ``alpha`` -- is seeded from, log-uniformly, when the population is
            created. Only ``LASSOSparsity`` reads that value, so the interval is
            the sparse-regression operator's parameter and its default comes
            from ``LASSOSparsity.initial_sparsity_interval`` ((1e-4, 2.5)),
            not from the search-space configuration. It is an initial range
            only: metaparameter mutation and crossover move alpha outside it
            during the search. Under the default VWSR sparsity, which derives
            its penalties from the data, passing this warns and changes nothing.
        max_deriv_order : int | list | tuple, optional
            Highest order of calculated derivatives, and therefore the highest
            derivative offered as a token. It describes the POOL, not any one
            sample, so it lives here rather than on ``createTrajectory``, and
            is applied to every trajectory passed in ``data``. Default comes
            from ``preprocessing.max_deriv_order`` (1).
        additional_tokens : list of TokenFamily or Prepared_tokens, optional
            Additional tokens, that would be used to construct the equations among the main variables and their
            derivatives. Objects of this list must be of type ``epde.interface.token_family.TokenFamily`` or
            of ``epde.interface.prepared_tokens.Prepared_tokens`` subclasses types. The default is None.
            Like the orders above these describe the pool, so they are passed
            here rather than attached to a trajectory.
        data_fun_pow : int, optional
            Maximum power of the variable token family. Default comes from
            ``search_space.data_fun_pow`` (1).
        deriv_fun_pow : int, optional
            Maximum power of the derivative token families. Default comes from
            ``search_space.deriv_fun_pow`` (1).
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
        search_space = self._config.search_space
        if equation_terms_max_number is UNSET:
            equation_terms_max_number = search_space.equation_terms_max_number
        if equation_factors_max_number is UNSET:
            equation_factors_max_number = search_space.equation_factors_max_number
        # The seeding interval belongs to the sparsity operator rather than
        # to the space of equations being searched, so it is read from the
        # operator: objectives.sparsity_kwargs if configured there, else the
        # class default. Equal ends on the CLASS are that operator saying it
        # does not tune a sparsity constant at all, which makes an explicitly
        # passed interval inert -- worth a word, because it looks like it is
        # doing something.
        from epde.operators.common.sparsity import initial_sparsity_interval
        objectives = self._config.objectives
        sparsity_cls = objectives.sparsity_cls
        class_interval = initial_sparsity_interval(sparsity_cls)
        configured = objectives.sparsity_kwargs.get('initial_sparsity_interval')
        if eq_sparsity_interval is UNSET:
            eq_sparsity_interval = (class_interval if configured is None
                                    else tuple(configured))
        elif class_interval[0] == class_interval[1]:
            warnings.warn(
                'eq_sparsity_interval seeds the LASSO alpha metaparameter, and '
                '{0} never reads it -- the value is ignored. Pass '
                "sparsity_cls='lasso' to run the pipeline it configures.".format(
                    getattr(sparsity_cls, '__name__', sparsity_cls)),
                global_var.EPDEUsageWarning, stacklevel=3)   # 3: @timed wrapper

        max_deriv_order, data_fun_pow, deriv_fun_pow = self._pool_structure(
            max_deriv_order, data_fun_pow, deriv_fun_pow)

        additional_tokens = self._resolve_token_families(additional_tokens)
        data = self._as_trajectories(data, legacy_kwargs, 'fit')

        if isinstance(data, Trajectory):
            data = [data,]
        if data is None and pool is None:
            raise ValueError('Data has to be specified beforehand or passed in '
                             'fit as an argument.')

        cur_params = self._pool_params(data, additional_tokens, max_deriv_order,
                                       data_fun_pow, deriv_fun_pow)

        if pool is None:
            if self.pool is None or self.pool_params != cur_params:
                self.create_pool(data = data, additional_tokens=additional_tokens,
                                 max_deriv_order=max_deriv_order,
                                 data_fun_pow=data_fun_pow,
                                 deriv_fun_pow=deriv_fun_pow)
        else:
            self.pool = pool; self.pool_params = cur_params

        self._run_optimization(equation_terms_max_number, equation_factors_max_number,
                               eq_sparsity_interval, optimizer, population)



    def _run_optimization(self, equation_terms_max_number, equation_factors_max_number,
                          eq_sparsity_interval, optimizer=None, population=None):
        """Build the optimizer for the current pool and run it."""
        self.optimizer_init_params['population_instruct'] = {
            "pool": self.pool,
            "terms_number": equation_terms_max_number,
            "max_factors_in_term": equation_factors_max_number,
            "sparsity_interval": eq_sparsity_interval,
            "second_objective": self._second_objective}

        if optimizer is None:
            self.optimizer = self._create_optimizer(
                self.multiobjective_mode, self.optimizer_init_params,
                self.director, population, self._second_objective)
        else:
            self.optimizer = optimizer

        # Pass only the exec params this optimizer's ``optimize`` accepts:
        # SimpleOptimizer.optimize has no ``early_stopping_callback`` (a
        # MOEA/D-only exec param that set_moeadd_params leaves behind when an
        # EpdeSearch built multiobjective-by-default is run single-objective).
        _exec_keys = set(inspect.signature(self.optimizer.optimize).parameters) - {'self'}
        _exec_params = {k: v for k, v in self.optimizer_exec_params.items() if k in _exec_keys}
        self.optimizer.optimize(**_exec_params)

        print('The optimization has been conducted.')
        self.search_conducted = True

        if self._config.runtime.free_tensor_cache_after_fit:
            # The evaluated-term cache is the bulk of the memory and is
            # rebuildable, so it goes as soon as the search stops needing it.
            # The grid, sample and initial-data caches stay: equations(),
            # predict(), solver_forms() and visualize_solutions() all read
            # them AFTER fit returns, so releasing those here would break most
            # of the post-fit API. ``close()`` releases everything.
            global_var.release_tensor_cache()

    @staticmethod
    def _create_optimizer(multiobjective_mode: bool, optimizer_init_params: dict,
                          opt_strategy_director: OptimizationPatternDirector,
                          population: List[SoEq] = None,
                          second_objective: str = None):
        if multiobjective_mode:
            # Lockstep site #3 of the selectable second axis. The ideal point
            # is ASKED OF THE OBJECTIVES rather than written here: ``[0., 1.]``
            # was never "the ideal when use_pic is off", it is Complexity's
            # ideal, because the least complex equation has one factor. With
            # the value living on ``EquationObjective.ideal_value``, a new
            # second-axis objective declares its own optimum and this function
            # does not change -- and the ideal can no longer disagree with the
            # axis actually assembled.
            axes = ('discrepancy',
                    second_objective or self._config.objectives.second_objective)
            best_sol_vals = ideal_point(axes)
            optimizer_init_params['best_sol_vals'] = best_sol_vals
            optimizer_init_params['passed_population'] = population
            optimizer = MOEADDOptimizer(**optimizer_init_params)
            optimizer.pass_best_objectives(*best_sol_vals)
        else:
            optimizer_init_params['passed_population'] = population
            # Pass only the params SimpleOptimizer accepts. ``optimizer_init_params``
            # can still carry MOEA/D-only keys (e.g. ``H``, ``nds_method``) if
            # ``set_moeadd_params`` ran earlier -- which it does in EpdeSearch's
            # default (multiobjective) __init__ before a switch to single-objective.
            # Selecting by SimpleOptimizer's signature keeps this robust to how the
            # params dict was populated.
            so_keys = set(inspect.signature(SimpleOptimizer.__init__).parameters) - {'self'}
            so_params = {k: v for k, v in optimizer_init_params.items() if k in so_keys}
            optimizer = SimpleOptimizer(**so_params)

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

    def solver_forms(self, grids: list = None, num: int = 1, sample_key: int = None):
        '''
        Method returns solver forms of the equations in a form of Python list.

        Args:
            grids (`list`): optional
                Grids to state the forms on. When omitted they are taken from
                the trajectory named by ``sample_key``.
            num (`int`): optional
                How many Pareto levels (multiobjective) or solutions
                (single-objective) to convert.
            sample_key (`int`): optional
                Which trajectory's grid the forms are built on. Defaults to the
                first registered one. A search fitted on several samples has one
                grid per sample, so there is no single "the" grid to fall back
                to -- ``SystemSolverInterface.form`` requires the choice, and
                this argument used to be missing entirely, making every call
                raise ``TypeError``.

        Returns:
            system form, suitable for solver
        '''
        if sample_key is None:
            sample_ids = global_var.samples_manager.trajecatoryIDs
            if not sample_ids:
                raise RuntimeError(
                    'No trajectory is registered, so there is no grid to state '
                    'the solver form on. Run fit (or create_pool) first.')
            sample_key = sample_ids[0]

        device = self._config.solver.device
        forms = []
        if self.multiobjective_mode:
            for level in self._resulting_population[:min(num, len(self._resulting_population))]:
                temp = []
                for sys in level: #self.resulting_population[idx]:
                    temp.append(SystemSolverInterface(sys, device=device).form(
                        domain_key=sample_key, grids=grids))
                forms.append(temp)
        else:
            for sys in self._resulting_population[:min(num, len(self._resulting_population))]:
                forms.append(SystemSolverInterface(sys, device=device).form(
                    domain_key=sample_key, grids=grids))
        return forms

    @property
    def cache(self):
        if global_var.grid_cache is not None:
            return global_var.grid_cache, global_var.tensor_cache
        else:
            return None, global_var.tensor_cache

    def close(self):
        """Release every cached tensor held for this search.

        The caches are process-level, so this ends the *process's* search
        state, not just this object's: after it, ``predict``, ``solver_forms``
        and ``visualize_solutions`` have no data to work from. Call it when the
        results have been read out, or use the context-manager form::

            with EpdeSearch(...) as search:
                search.fit(...)
                eqs = search.equations(only_print=False)

        ``equations()`` keeps working either way -- it reads the recorded
        population, not the caches.
        """
        global_var.delete_cache()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

    @property
    def pareto_history(self):
        """Per-epoch Pareto-level-0 snapshots, populated during ``fit``.

        Returns a list of length ``training_epochs``; each element is a
        list of ``{'text_form': str, 'obj_fun': list}`` dicts -- one per
        solution on the non-dominated front at the end of that epoch.
        Empty list when the optimizer hasn't been run or doesn't track
        epoch history (e.g. single-objective mode).
        """
        return getattr(self.optimizer, '_pareto_history', [])

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

    def predict(self, system : SoEq = None, boundary_conditions: BoundaryConditions = None,
                grid : list = None, data = None, system_file: str = None, net = None,
                mode=UNSET, compiling_params=UNSET, optimizer_params=UNSET,
                cache_params=UNSET, early_stopping_params=UNSET,
                plotting_params=UNSET, training_params=UNSET, use_cache=UNSET,
                use_fourier=UNSET, fourier_params=UNSET,
                use_adaptive_lambdas=UNSET):
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
        
        solver_cfg = self._config.solver
        if mode is UNSET:
            mode = solver_cfg.mode
        if compiling_params is UNSET:
            compiling_params = solver_cfg.compiling_params
        if optimizer_params is UNSET:
            optimizer_params = solver_cfg.optimizer_params
        if cache_params is UNSET:
            cache_params = solver_cfg.cache_params
        if early_stopping_params is UNSET:
            early_stopping_params = solver_cfg.early_stopping_params
        if plotting_params is UNSET:
            plotting_params = solver_cfg.plotting_params
        if training_params is UNSET:
            training_params = solver_cfg.training_params
        if use_cache is UNSET:
            use_cache = solver_cfg.use_cache
        if use_fourier is UNSET:
            use_fourier = solver_cfg.use_fourier
        if fourier_params is UNSET:
            fourier_params = solver_cfg.fourier_params
        if use_adaptive_lambdas is UNSET:
            use_adaptive_lambdas = solver_cfg.use_adaptive_lambdas

        if system is not None:
            print('Using explicitly sent system of equations.')
        elif system_file is not None:
            assert '.pickle' in system_file
            print('Loading equation from pickled file.')

            with open(system_file, 'rb') as handle:
                system = pickle.load(handle)
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
        
        # ``mode`` reaches the adapter exactly once, here. It used to be both
        # written into compiling_params AND passed again to solve_epde_system,
        # so a caller's own compiling_params['mode'] was silently overwritten.
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
            return self.optimizer.plot_pareto(dimensions=dimensions, **visulaizer_kwargs)
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
            compl_objs_num = int(candidates[0].obj_fun.size/2)
            # print(compl_objs_num)
            return [(candidate, candidate.obj_fun[-compl_objs_num:]) for candidate in candidates]
        else:
            raise ValueError(f'Incorrect type of the equation, got {type(candidates[0])}')
        
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
        best_qualities_compl = [complexities[-1] for complexities in self.ordered_complexities]
        return self.create_best_for_complexity(best_qualities_compl, pool)
    
class EpdeMultisample(EpdeSearch):
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
        # self.set_domain_properties(grids_stacked, self._memory_for_cache, self._boundary, self._function_form)

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
        raise NotImplementedError('This method has been depricated!')
    
        # assert self.coodinate_tensors is not None, 'Coordinate tensors for the sample have to be set beforehand.'
        # self._create_caches(coordinate_tensors=coordinate_tensors, memory_for_cache=memory_for_cache)
        # if prune_domain:
        #     self.domain_pruning(pivotal_tensor_label, pruner, threshold, division_fractions, rectangular)
        # self.set_boundaries(boundary_width)

        # # TODO$
        # self._upload_g_func(function_form)

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
            
        # Pass only the exec params this optimizer's ``optimize`` accepts:
        # SimpleOptimizer.optimize has no ``early_stopping_callback`` (a
        # MOEA/D-only exec param that set_moeadd_params leaves behind when an
        # EpdeSearch built multiobjective-by-default is run single-objective).
        _exec_keys = set(inspect.signature(self.optimizer.optimize).parameters) - {'self'}
        _exec_params = {k: v for k, v in self.optimizer_exec_params.items() if k in _exec_keys}
        self.optimizer.optimize(**_exec_params)
        
        print('The optimization has been conducted.')
        self.search_conducted = True    
