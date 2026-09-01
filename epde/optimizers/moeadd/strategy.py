#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 12:54:02 2022

@author: maslyaev
"""

import numpy as np
from functools import partial

import epde.globals as global_var
from epde.interface.search_config import active_config

from epde.operators.utils.operator_mappers import map_operator_between_levels, OperatorCondition
from epde.operators.utils.template import add_base_param_to_operator

from epde.operators.multiobjective.selections import MOEADDSelection
from epde.operators.multiobjective.variation import get_basic_variation
from epde.operators.common.fitness import SolverFreeFitness, SolverBasedFitness
from epde.operators.common.objectives import Complexity, Discrepancy, Instability
from epde.operators.common.right_part_selection import RandomRHPSelector, EqRightPartSelector, SoEqRightPartSelector

from epde.operators.multiobjective.moeadd_specific import get_pareto_levels_updater, SimpleNeighborSelector, get_initial_sorter
from epde.operators.common.sparsity import (LASSOSparsity, VWSRSparsity,
                                            build_sparsity_operator)
from epde.operators.common.coeff_calculation import LinRegBasedCoeffsEquation

from epde.optimizers.builder import add_sequential_operators, OptimizationPatternDirector, StrategyBuilder
from epde.optimizers.moeadd.strategy_elems import MOEADDSectorProcesser

class MOEADDDirector(OptimizationPatternDirector):
    """
    Class for creating strategy builder of multicriterian optimization
    """
    def use_baseline(self, use_solver: bool = False, second_objective: str = None,
                     sparsity_cls=None, sparsity_kwargs: dict = None,
                     variation_params : dict = {}, mutation_params : dict = {},
                     sorter_params : dict = {}, pareto_combiner_params : dict = {},
                     pareto_updater_params : dict = {}, params : dict = None,
                     solver_backend : str = 'autograd', **kwargs):
        # ``params`` is the bundled director_params dict the interface passes;
        # per key it wins over the individually named ``*_params`` (kept for
        # direct callers). Declaring it as a named parameter also keeps the
        # bundle out of ``kwargs``, which doubles as the operator-parameter
        # override dict for ``add_base_param_to_operator``.
        if params:
            variation_params = params.get('variation_params', variation_params)
            mutation_params = params.get('mutation_params', mutation_params)
            sorter_params = params.get('sorter_params', sorter_params)
            pareto_combiner_params = params.get('pareto_combiner_params', pareto_combiner_params)
            pareto_updater_params = params.get('pareto_updater_params', pareto_updater_params)
        add_kwarg_to_operator = partial(add_base_param_to_operator, target_dict = kwargs)

        def _resolved_second() -> str:
            """The second Pareto axis, resolved once for this assembly.

            ``None`` defers to ``objectives.second_objective``. Both fitness
            branches below and ``self.second_objective`` (which
            ``EpdeSearch._create_optimizer`` reads to build the ideal point)
            go through this, so the filler set and the utopia point cannot
            disagree.
            """
            if second_objective is not None:
                return second_objective
            return active_config().objectives.second_objective

        # Recorded so the optimizer can derive its ideal point from the axes
        # that were actually assembled, rather than re-deriving them.
        self.second_objective = _resolved_second()

        def _solver_free_fitness(second: str = None, metric: str = None):
            """Assemble a SolverFreeFitness host. The primary Discrepancy
            filler is built BARE -- it resolves its option from the search
            configuration (``objectives.discrepancy_metric``, published by
            ``EpdeSearch.__init__``) at compute time; ``metric`` is passed
            only for fixed-role hosts (the RPS-sweep's 'l2_relative').
            ``second`` names the second-axis family filler -- Instability
            or Complexity -- or None (discrepancy only, the RPS shape)."""
            disc = Discrepancy(metric)
            objectives = [disc]
            if second == 'instability':
                objectives.append(Instability())
            elif second == 'complexity':
                objectives.append(Complexity())
            return SolverFreeFitness(['penalty_coeff'], objectives=objectives, primary=disc)

        neighborhood_selector = SimpleNeighborSelector(['number_of_neighbors'])
        add_kwarg_to_operator(operator = neighborhood_selector)

        selection = MOEADDSelection(['delta', 'parents_fraction'])
        add_kwarg_to_operator(operator = selection)
        selection.set_suboperators({'neighborhood_selector' : neighborhood_selector})

        variation = get_basic_variation(variation_params)

        # right_part_selector = RandomRHPSelector()
        right_part_selector = EqRightPartSelector()

        sparsity = build_sparsity_operator(sparsity_cls, sparsity_kwargs)
        coeff_calc = LinRegBasedCoeffsEquation()

        if use_solver:
            # ``second_objective`` selects the second axis (instability in
            # place of the baseline complexity) -- via the same resolver used
            # by the solver-free branch, the SoEq axis registration and the
            # MOEA/D ideal point. The backend (and the backend-implied primary
            # discrepancy option) is chosen by ``solver_backend``,
            # independently. A complexity second axis needs no filler -- the
            # ``equation_complexity`` reader computes lazily.
            second = (Instability()
                      if _resolved_second() == 'instability'
                      else None)
            if solver_backend == 'deepxde':
                primary = Discrepancy('deepxde')
            elif solver_backend == 'autograd':
                primary = Discrepancy('solver_l2')
            else:
                raise ValueError(f'Unknown solver_backend {solver_backend!r}: '
                                 "expected 'autograd' or 'deepxde'.")
            objectives = [primary] if second is None else [primary, second]
            fitness = SolverBasedFitness(['penalty_coeff', 'pinn_loss_mult'],
                                         objectives=objectives, primary=primary,
                                         backend=solver_backend,
                                         masked=False)

            sparsity_c = map_operator_between_levels(sparsity, 'gene level', 'chromosome level')
        else:
            # Lockstep site #1 of the selectable second axis (the others:
            # SoEq.use_default_multiobjective_function and the MOEA/D ideal
            # point) -- the same resolver at all three keeps the computed
            # filler set, the registered axis readers and the utopia point
            # coherent by construction.
            fitness = _solver_free_fitness(_resolved_second())
        add_kwarg_to_operator(operator = fitness)

        # ``sparsity`` and ``coeff_calc`` are wired onto the RIGHT-PART
        # SELECTOR, not onto the fitness host: the host is a pure scorer, and
        # RPS is what fits each candidate target before scoring it (and the
        # only operator that prunes). Both are GENE-level here, matching
        # EqRightPartSelector's own level -- the chromosome-level ``sparsity_c``
        # below is a separate object, for the dormant legacy-LASSO hook on
        # OffspringUpdater.
        fitness_cond = lambda x: not getattr(x, 'fitness_calculated')
        if use_solver:
            # The RPS term-sweep must never solve: use a lightweight
            # solver-free WAPE fitness as the right-part fitness instead.
            fitness_lightweight = _solver_free_fitness(metric='l2_relative')
            add_kwarg_to_operator(operator = fitness_lightweight)
            right_part_selector.set_suboperators({'fitness_calculation' : fitness_lightweight,
                                                  'sparsity' : sparsity,
                                                  'coeff_calc' : coeff_calc})

            fitness = OperatorCondition(fitness, fitness_cond)
        else:
            right_part_selector.set_suboperators({'fitness_calculation' : fitness,
                                                  'sparsity' : sparsity,
                                                  'coeff_calc' : coeff_calc})
            fitness = map_operator_between_levels(fitness, 'gene level', 'chromosome level',
                                                  objective_condition=fitness_cond)

            sparsity_c = map_operator_between_levels(sparsity, 'gene level', 'chromosome level')

        # Chromosome-level RPS that resolves system degeneracy: after the
        # per-equation sweeps it rerolls any equation whose ACTIVE structure
        # coincides with another equation's (the same law rearranged), while
        # allowing legitimate cross-equation coupling terms. The SoEq comes
        # out degeneracy-clean by construction.
        sys_rps_inner = SoEqRightPartSelector()
        sys_rps_inner.set_suboperators({'eq_right_part_selector': right_part_selector})
        rps_cond = lambda x: any([not elem_eq.right_part_selected for elem_eq in x.vals])
        sys_rps = OperatorCondition(sys_rps_inner, rps_cond)

        # Separate mutation from population updater for better customization.
        initial_sorter = get_initial_sorter(right_part_selector = sys_rps, chromosome_fitness = fitness, 
                                            sorter_params = sorter_params)
        population_updater = get_pareto_levels_updater(right_part_selector = sys_rps, chromosome_fitness = fitness,
                                                       sparsity=sparsity_c,
                                                       constrained = False, mutation_params = mutation_params, 
                                                       pl_updater_params = pareto_updater_params, 
                                                       combiner_params = pareto_combiner_params)

        self.builder = add_sequential_operators(self.builder, [('initial_sorter', initial_sorter),
                                                               # ('pareto_updater_initial', population_updater),
                                                               ('selection', selection),
                                                               ('variation', variation),
                                                               ('pareto_updater_compl', population_updater)])
    
    def use_constrained_eq_search(self):
        raise NotImplementedError('No constraints have been implemented yest')
