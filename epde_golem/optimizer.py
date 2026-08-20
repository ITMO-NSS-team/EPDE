"""GOLEM-backed drop-in replacement for EPDE's native evolutionary optimizers.

``GolemEpdeOptimizer`` presents the exact interface ``EpdeSearch`` expects from
``MOEADDOptimizer`` (multi-objective) and ``SimpleOptimizer`` (single-objective)
-- ``set_strategy``, ``optimize(epochs=...)``, ``pareto_levels`` / ``population``
-- but the population engine underneath is GOLEM's ``EvoGraphOptimizer``.

What is *shared* with the native run (so the comparison isolates the engine):
    * initial population        -- ``SystemsPopulationConstructor``
    * chromosome representation -- ``SoEq``
    * mutation                  -- ``SystemMutation``
    * crossover                 -- ``ChromosomeCrossover``
    * evaluation                -- right-part selection -> sparsity -> LinReg
                                   coefficients -> objective readers

What *differs* (the thing under test):
    * selection: SPEA-2 (GOLEM) vs PBI decomposition over Das-Dennis weight
      vectors with a Gale-Shapley solution/weight marriage (EPDE);
    * survival: unbounded Pareto archive + elitism (GOLEM) vs per-sector
      crowding-based replacement of the PBI-worst neighbour (EPDE);
    * schedule: adaptive population size / operator probabilities and an
      optional multi-armed-bandit mutation agent (GOLEM) vs a fixed schedule.
"""

import datetime
import logging
from typing import List, Optional, Union

import numpy as np

from golem.core.optimisers.genetic.gp_optimizer import EvoGraphOptimizer
from golem.core.optimisers.genetic.gp_params import GPAlgorithmParameters
from golem.core.optimisers.genetic.operators.elitism import ElitismTypesEnum
from golem.core.optimisers.genetic.operators.inheritance import GeneticSchemeTypesEnum
from golem.core.optimisers.genetic.operators.selection import SelectionTypesEnum
from golem.core.optimisers.adaptive.operator_agent import MutationAgentTypeEnum
from golem.core.optimisers.optimization_parameters import GraphRequirements
from golem.core.optimisers.optimizer import GraphGenerationParams

import epde.globals as global_var
from epde.interface.interface import EpdeSearch
from epde.optimizers.moeadd.moeadd import ParetoLevels
from epde.optimizers.moeadd.population_constr import SystemsPopulationConstructor
from epde.optimizers.moeadd.supplementary import fast_non_dominated_sorting, ndl_update

from .graph import soeq_to_graph
from .objective import SystemEvaluator, build_objective
from .operators import (make_mutation, make_add_term_mutation,
                        make_drop_term_mutation, make_equation_reroll_mutation,
                        make_sparsity_mutation, make_crossover, soeq_is_valid)


def _unwrap_to_chromosome_level(operator):
    """Peel ``OperatorMapper`` layers until a chromosome-level operator.

    EPDE's directors wrap the same operator at whatever level the surrounding
    pipeline needs (gene -> chromosome -> population). GOLEM drives one
    candidate system at a time, i.e. the chromosome level, so the wrappers
    above it have to come off.
    """
    while 'chromosome level' not in getattr(operator, 'operator_tags', set()):
        subops = getattr(getattr(operator, 'suboperators', None), 'suboperators', {})
        if 'to_map' not in subops:
            raise ValueError(f'Cannot reach the chromosome level of {operator!r}')
        operator = operator.suboperators['to_map']
    return operator


def extract_epde_operators(director) -> dict:
    """Pull the assembled EPDE operators out of the strategy director.

    Reusing these *instances* -- rather than re-assembling equivalents --
    guarantees both engines run byte-identical domain operators with
    byte-identical parameters. Both of EPDE's directors are supported: the
    multi-objective ``MOEADDDirector`` and the single-objective
    ``BaselineDirector``, which label and nest their blocks differently.
    """
    blocks = director.builder.blocks_labeled
    if 'initial_sorter' in blocks:                       # MOEADDDirector
        initial_sorter = blocks['initial_sorter']._operator
        offspring_updater = blocks['pareto_updater_compl']._operator
        variation = blocks['variation']._operator
        return {
            'right_part_selector': initial_sorter.suboperators['right_part_selector'],
            'chromosome_fitness': initial_sorter.suboperators['chromosome_fitness'],
            'chromosome_mutation': offspring_updater.suboperators['chromosome_mutation'],
            'chromosome_crossover': variation.suboperators['chromosome_crossover'],
        }
    # BaselineDirector: every stage is mapped up to the population level, and
    # its mutation additionally carries an "only elites" element condition
    # that would make GOLEM's mutation a no-op -- so unwrap to the bare
    # chromosome-level operators.
    return {
        'right_part_selector': _unwrap_to_chromosome_level(
            blocks['right part selection 1']._operator),
        'chromosome_fitness': _unwrap_to_chromosome_level(
            blocks['fitness evaluation 1']._operator),
        'chromosome_mutation': _unwrap_to_chromosome_level(
            blocks['mutation']._operator),
        'chromosome_crossover': blocks['variation']._operator.suboperators[
            'chromosome_crossover'],
    }


class _GolemParetoLevels(ParetoLevels):
    """A ``ParetoLevels`` facade over a GOLEM result population.

    ``EpdeSearch`` reads results through ``optimizer.pareto_levels.levels``;
    building a real ``ParetoLevels`` (with EPDE's own non-dominated sorting)
    from the GOLEM archive keeps every downstream consumer -- ``equations()``,
    ``plot_pareto``, ``get_by_complexity`` -- working unchanged.
    """

    def __init__(self, population: List):
        super().__init__(population=[], weights=np.ones((1, 1)),
                         sorting_method=fast_non_dominated_sorting,
                         update_method=ndl_update)
        self.population = list(population)
        self.levels = self.sort() if self.population else [[]]


class GolemEpdeOptimizer:
    """EPDE equation search driven by GOLEM's evolutionary engine.

    Multi-objective by default (SPEA-2 over the discrepancy/second-axis front);
    pass ``multiobjective=False`` for the single-objective mode, where GOLEM
    runs tournament selection over EPDE's scalar objective.
    """

    #: Extra knobs; override per-run via the ``golem_params`` dict.
    golem_defaults = dict(
        # steady_state, NOT generational: in multi-objective mode GOLEM
        # disables elitism, and the generational scheme keeps the offspring
        # unconditionally, so that combination runs with no survival selection
        # whatsoever (see the ablation in results/ and the warning added to
        # GPAlgorithmParameters.__post_init__).
        genetic_scheme=GeneticSchemeTypesEnum.steady_state,
        elitism_type=ElitismTypesEnum.keep_n_best,
        adaptive_mutation_type=MutationAgentTypeEnum.default,
        adaptive_depth=False,
        mutation_prob=0.8,
        crossover_prob=0.8,
        max_pop_size=None,
        keep_history=False,
        show_progress=False,
        use_add_term_mutation=True,
        # Extra structural actions beyond EPDE's own term-replacement
        # mutation. 'add' grows, 'drop' shrinks, 'reroll' restarts one
        # equation in place, 'sparsity' moves along the complexity axis
        # without touching the structure. Each is a separate GOLEM action, so
        # an adaptive agent can learn which pays off when.
        extra_mutations=('add',),
        # Environmental (survival) selection: 'spea2' is GOLEM's default for
        # multi-objective runs, 'nsga2' the rank+crowding alternative added
        # for this work.
        selection=None,
        # Mating-pool selection. None keeps GOLEM's historical behaviour, in
        # which reproduction applies no selection pressure at all.
        mating_selection=None,
        # Memoize evaluated chromosomes by (structure, metaparameters). 0
        # disables. EPDE's own loop skips the fitness call for a structure it
        # has already seen; GOLEM has no equivalent.
        fitness_cache_size=0,
        # Split the evaluation budget across N independent searches and union
        # their archives. GOLEM has no restart mechanism; on this domain the
        # true equation reaches the front early and is then displaced, so
        # several short runs beat one long one at equal budget.
        restarts=1,
        timeout_minutes=None,
        # How often GOLEM replaces structurally identical individuals with
        # fresh ones (-1 disables). EPDE's MOEA/D keeps diversity through its
        # weight sectors and an explicit duplicate history; this is GOLEM's
        # counterpart and matters a lot on this domain, where the discrepancy
        # objective happily collapses the whole population onto one form.
        structural_diversity_check=5,
        # Budget-matched runs: stop once ``eval_counter()`` (defaults to the
        # optimizer's own system-evaluation counter) reaches ``eval_budget``.
        # Checked once per generation, so the budget can be overshot by at
        # most one generation -- report the achieved count, not the target.
        eval_budget=None,
        eval_counter=None,
        collect_garbage=False,
        # How many times a degenerate initial candidate is re-rolled before
        # being seeded anyway. Mirrors EPDE's ``uniqueness_attempt_limit``
        # policy for the hard-reject case.
        initial_reroll_attempts=4,
        # GOLEM logs two INFO banners plus a bandit dump per generation; on
        # short EPDE runs that console traffic is a measurable share of the
        # wall clock, so quieten it by default.
        log_level=logging.WARNING,
    )

    def __init__(self, population_instruct, pop_size, solution_params,
                 H: int = None, neighbors_number: int = None,
                 nds_method=fast_non_dominated_sorting, ndl_update=ndl_update,
                 passed_population: Union[List, ParetoLevels] = None,
                 best_sol_vals=None, weights_assigner: str = 'marriage',
                 sorting_method=None, multiobjective: bool = True,
                 golem_params: Optional[dict] = None):
        self.pop_size = pop_size
        self.solution_params = solution_params if solution_params is not None else {}
        self.best_sol_vals = best_sol_vals
        self.multiobjective = multiobjective
        self.sorting_method = sorting_method
        self.golem_params = dict(self.golem_defaults)
        if golem_params:
            self.golem_params.update(golem_params)

        if multiobjective:
            constructor_cls = SystemsPopulationConstructor
        else:
            from epde.optimizers.single_criterion.population_constr import (
                SystemsPopulationConstructor as SOConstructor)
            constructor_cls = SOConstructor
        self._pop_constructor = constructor_cls(**population_instruct)
        self._initial_systems = self._build_initial_population(passed_population)

        # Objective layout mirrors ``SoEq.obj_fun``: the per-equation readers
        # are flattened, so a system of n equations with k objective families
        # yields n * k values.
        variables = list(self._initial_systems[0].vars_to_describe)
        if multiobjective:
            second = global_var.resolve_second_objective(
                population_instruct.get('use_pic', True))
            families = ['discrepancy', second]
        else:
            families = [getattr(global_var, 'single_objective_metric', 'discrepancy')]
        if len(variables) > 1:
            self.metric_names = [f'{fam}:{var}' for fam in families for var in variables]
        else:
            self.metric_names = list(families)

        self.epde_operators = None
        self.evaluator = None
        self.optimiser = None
        self.history = None
        self._pareto_levels = None
        self._population = None
        self._epoch_snapshots = []
        self._population_snapshots = []

    # ------------------------------------------------------------------ setup

    def _build_initial_population(self, passed_population) -> List:
        if isinstance(passed_population, ParetoLevels):
            population = list(passed_population.population) or list(
                passed_population.unplaced_candidates)
        elif hasattr(passed_population, 'population'):        # EPDE Population
            population = list(passed_population.population)
        else:
            population = list(passed_population) if passed_population else []
        for solution in population:
            if hasattr(self._pop_constructor, 'applyToPassed'):
                self._pop_constructor.applyToPassed(solution, **self.solution_params)
        for _ in range(self.pop_size - len(population)):
            candidate = self._pop_constructor.create(**self.solution_params)
            for equation in candidate.vals:
                while len(equation.terms_labels) != len(equation.structure):
                    candidate.vals[equation.main_var_to_explain].randomize()
                    candidate.vals[equation.main_var_to_explain].reset_saved_state()
            population.append(candidate)
        return population

    def _vet_initial_population(self):
        """Re-roll initial candidates the fitness chain flags degenerate.

        EPDE's own initialiser (``InitialParetoLevelSorting``) treats
        degeneracy as a HARD reject -- a chromosome whose sparse regression
        collapsed it to ``LOSS_NAN_VAL`` is never seeded -- and duplicates as a
        SOFT one. The hard reject is a property of the problem, not of MOEA/D,
        so the GOLEM arm applies it too; otherwise it would start from a
        measurably worse population than the native arm and the comparison
        would be measuring initialisation, not search. Duplicates are left
        alone: MOEA/D needs one distinct solution per weight vector, GOLEM does
        not, and it has its own structural-diversity mechanism.

        The evaluations spent here are counted like any others, so the two arms
        remain budget-comparable.
        """
        from epde.operators.multiobjective.moeadd_specific import has_degenerate_equation
        attempts_limit = self.golem_params['initial_reroll_attempts']
        graphs = []
        for candidate in self._initial_systems:
            values = self.evaluator.evaluate_system(candidate, fresh=True)
            for _ in range(attempts_limit):
                if not has_degenerate_equation(candidate):
                    break
                candidate.create()
                values = self.evaluator.evaluate_system(candidate, fresh=True)
            # Hand GOLEM a graph whose cached objective vector is already
            # filled, so the vetting evaluation is not paid for twice.
            graph = soeq_to_graph(candidate)
            graph.obj_values = tuple(values)
            graphs.append(graph)
        return graphs

    def pass_best_objectives(self, *args) -> None:
        """Kept for interface parity with ``MOEADDOptimizer``; GOLEM's
        selection is decomposition-free, so an ideal point is not needed."""
        self.best_sol_vals = list(args)

    def set_strategy(self, strategy_director):
        strategy_director.builder.assemble(True)
        self.epde_operators = extract_epde_operators(strategy_director)

    # -------------------------------------------------------------- execution

    def _build_optimiser(self, epochs: int, budget_target=None):
        from golem.core.log import Log
        Log().reset_logging_level(self.golem_params['log_level'])

        ops = self.epde_operators
        self.evaluator = SystemEvaluator(ops['right_part_selector'],
                                         ops['chromosome_fitness'],
                                         n_objectives=len(self.metric_names),
                                         metric_names=self.metric_names,
                                         cache_size=self.golem_params['fitness_cache_size'])
        multi = self.multiobjective and len(self.metric_names) > 1
        objective = build_objective(self.evaluator, self.metric_names, multi)

        mutations = [make_mutation(ops['chromosome_mutation'])]
        extra = set(self.golem_params['extra_mutations'] or ())
        if self.golem_params['use_add_term_mutation']:
            extra.add('add')
        if 'add' in extra:
            mutations.append(make_add_term_mutation())
        if 'drop' in extra:
            mutations.append(make_drop_term_mutation())
        if 'reroll' in extra:
            mutations.append(make_equation_reroll_mutation())
        if 'sparsity' in extra:
            subops = getattr(ops['chromosome_mutation'], 'suboperators', None)
            param_mutation = (subops['param_mutation']
                              if subops is not None and 'param_mutation' in subops.keys()
                              else None)
            if param_mutation is not None:
                mutations.append(make_sparsity_mutation(param_mutation))

        selection_types = [self.golem_params['selection']] if self.golem_params['selection'] \
            else [SelectionTypesEnum.spea2 if multi else SelectionTypesEnum.tournament]
        mating_selection_types = ([self.golem_params['mating_selection']]
                                  if self.golem_params['mating_selection'] else None)

        gp_params = GPAlgorithmParameters(
            multi_objective=multi,
            pop_size=self.pop_size,
            max_pop_size=self.golem_params['max_pop_size'] or self.pop_size,
            mutation_prob=self.golem_params['mutation_prob'],
            crossover_prob=self.golem_params['crossover_prob'],
            variable_mutation_num=False,
            mutation_types=mutations,
            crossover_types=[make_crossover(ops['chromosome_crossover'])],
            selection_types=selection_types,
            mating_selection_types=mating_selection_types,
            elitism_type=self.golem_params['elitism_type'],
            genetic_scheme_type=self.golem_params['genetic_scheme'],
            adaptive_mutation_type=self.golem_params['adaptive_mutation_type'],
            adaptive_depth=self.golem_params['adaptive_depth'],
            structural_diversity_frequency_check=self.golem_params['structural_diversity_check'],
        )

        timeout = (datetime.timedelta(minutes=self.golem_params['timeout_minutes'])
                   if self.golem_params['timeout_minutes'] else None)
        requirements = GraphRequirements(
            num_of_generations=epochs,
            timeout=timeout,
            early_stopping_iterations=None,
            early_stopping_timeout=None,
            keep_n_best=self.pop_size,
            keep_history=self.golem_params['keep_history'],
            history_dir=None,
            show_progress=self.golem_params['show_progress'],
            n_jobs=1,
            parallelization_mode='sequential',
            # The EPDE objective keeps a large, long-lived working set alive
            # (token pool, cached derivative tensors, torch). A full collection
            # after every population costs ~0.1 s here -- more than the
            # evaluations it follows -- and frees nothing that matters, since
            # the chromosomes are dropped by reference counting.
            collect_garbage=self.golem_params['collect_garbage'],
        )
        graph_gen_params = GraphGenerationParams(
            adapter=None,               # graphs already carry the domain payload
            rules_for_constraint=[soeq_is_valid],
        )

        initial_graphs = self._vet_initial_population()
        self.optimiser = EvoGraphOptimizer(objective, initial_graphs, requirements,
                                           graph_gen_params, gp_params)
        self.optimiser.set_iteration_callback(self._on_generation)

        if budget_target:
            counter = self._budget_counter()
            self.optimiser.stop_optimization.add_condition(
                lambda: counter() >= budget_target,
                'Optimisation stopped: evaluation budget exhausted')
        return objective

    def _budget_counter(self):
        return self.golem_params['eval_counter'] or \
            (lambda: self.evaluator.n_evaluations)

    def _on_generation(self, population, optimiser):
        # GOLEM's ``optimise`` finishes by overwriting ``optimiser.population``
        # with the Pareto archive ('final_choices'), so the last *evolved*
        # population is unreachable afterwards. Keep the last two snapshots
        # and use the earlier one as the population the run actually ended on.
        self._population_snapshots.append(list(population))
        del self._population_snapshots[:-2]
        front = optimiser.generations.best_individuals
        self._epoch_snapshots.append([
            {'text_form': ind.graph.soeq.text_form,
             'obj_fun': list(ind.fitness.values)}
            for ind in front])

    def optimize(self, epochs: int = 100, early_stopping_callback=None):
        restarts = max(1, int(self.golem_params['restarts'] or 1))
        if restarts > 1:
            return self._optimize_with_restarts(epochs, restarts)
        objective = self._build_optimiser(epochs, self.golem_params['eval_budget'])
        self.optimiser.optimise(objective)
        self.history = self.optimiser.history
        self._pareto_levels = _GolemParetoLevels(self._collect_systems())
        self._finalize_single_objective()

    def _optimize_with_restarts(self, epochs: int, restarts: int):
        """Spend the budget on several independent short searches.

        Motivated by the measured behaviour of this domain: the true equation
        reaches the non-dominated front in nearly every run, but is then
        displaced by lower-discrepancy, less parsimonious forms as the search
        converges (see the 'recovery: ever' versus 'recovery: front' columns in
        results/). Short runs therefore recover it more reliably than long
        ones -- so K short runs, whose archives are unioned at the end, beat
        one long run at the same total budget.

        GOLEM has no restart mechanism of its own: ``optimise`` runs one
        population to the stop condition and returns.
        """
        counter = self._budget_counter()
        budget = self.golem_params['eval_budget']
        collected, seen = [], set()
        for run_idx in range(restarts):
            if run_idx:
                # A restart is only a restart if the population is new.
                self._initial_systems = self._build_initial_population(None)
            target = (counter() + max(1, budget // restarts)) if budget else None
            objective = self._build_optimiser(max(1, epochs // restarts) if not budget else epochs,
                                              target)
            self.optimiser.optimise(objective)
            self.history = self.optimiser.history
            for soeq in self._collect_systems():
                key = soeq.equations_labels
                if key not in seen:
                    seen.add(key)
                    collected.append(soeq)
            self._population_snapshots = []
            if budget and counter() >= budget:
                break
        self._pareto_levels = _GolemParetoLevels(collected)
        self._finalize_single_objective()

    def _collect_systems(self):
        """The systems one ``optimise`` call produced: archive + last population.

        Both, not just the archive: the archive holds only mutually
        non-dominated systems, while the population also holds the diverse,
        dominated-but-parsimonious candidates a user still wants to see. This
        mirrors what EPDE's MOEA/D returns -- its ``levels[0]`` plus the rest
        of ``population``.
        """
        last_population = (self._population_snapshots[0]
                           if len(self._population_snapshots) > 1
                           else (self._population_snapshots[-1]
                                 if self._population_snapshots else []))
        systems, seen = [], set()
        for individual in list(self.optimiser.generations.best_individuals) + \
                list(last_population):
            soeq = individual.graph.soeq
            key = soeq.equations_labels
            if key in seen:
                continue
            seen.add(key)
            systems.append(soeq)
        return systems

    def _finalize_single_objective(self):
        if self.multiobjective:
            return
        from epde.optimizers.single_criterion.optimizer import Population
        from epde.optimizers.single_criterion.supplementary import simple_sorting
        self._population = Population(
            elements=list(self._pareto_levels.population),
            sorting_method=self.sorting_method or simple_sorting)
        self._population.sorted()

    # ----------------------------------------------------------------- output

    @property
    def pareto_levels(self) -> ParetoLevels:
        if self._pareto_levels is None:
            raise AttributeError('Call optimize() before reading the results.')
        return self._pareto_levels

    @property
    def population(self):
        """Single-objective result, in the shape ``EpdeSearch`` expects."""
        if getattr(self, '_population', None) is None:
            raise AttributeError('Call optimize() before reading the results.')
        return self._population

    @property
    def pareto_history(self):
        return self._epoch_snapshots

    def get_hist(self, best_only: bool = True):
        return self._epoch_snapshots

    def plot_pareto(self, dimensions: list, **kwargs):
        from epde.optimizers.moeadd.vis import ParetoVisualizer
        return ParetoVisualizer(self.pareto_levels).plot_pareto_per_equation(
            dimensions=tuple(dimensions), **kwargs)


class EpdeGolemSearch(EpdeSearch):
    """``EpdeSearch`` whose evolutionary engine is GOLEM.

    Everything else -- preprocessing, token-pool construction, the strategy
    director and its operators, result reporting -- is inherited unchanged.
    Both of EPDE's modes are covered: multi-objective (replacing
    ``MOEADDOptimizer``) and single-objective (replacing ``SimpleOptimizer``).
    """

    def __init__(self, *args, golem_params: Optional[dict] = None, **kwargs):
        self.golem_params = dict(golem_params or {})
        super().__init__(*args, **kwargs)

    def _create_optimizer(self, multiobjective_mode: bool, optimizer_init_params: dict,
                          opt_strategy_director, population=None, use_pic: bool = False):
        init_params = dict(optimizer_init_params)
        init_params['passed_population'] = population
        init_params['golem_params'] = self.golem_params
        init_params['multiobjective'] = multiobjective_mode
        if multiobjective_mode:
            init_params['best_sol_vals'] = EpdeSearch._resolve_ideal_point(use_pic)
        optimizer = GolemEpdeOptimizer(**init_params)
        optimizer.set_strategy(opt_strategy_director)
        return optimizer
