from epde.optimizers.builder import add_sequential_operators, OptimizationPatternDirector, StrategyBuilder
from epde.optimizers.single_criterion.optimizer import EvolutionaryStrategy
from functools import partial

from epde.operators.utils.operator_mappers import map_operator_between_levels
from epde.operators.utils.template import add_base_param_to_operator

from epde.operators.common.right_part_selection import (EqRightPartSelector,
                                                        SoEqRightPartSelector)
from epde.operators.common.fitness import SolverFreeFitness
from epde.operators.common.objectives import Discrepancy, Instability
import epde.globals as global_var
from epde.interface.search_config import active_config
from epde.operators.common.sparsity import (LASSOSparsity, VWSRSparsity,
                                            build_sparsity_operator)
from epde.operators.common.coeff_calculation import LinRegBasedCoeffsEquation
from epde.operators.singleobjective.mutations import get_singleobjective_mutation
from epde.operators.singleobjective.variation import get_singleobjective_variation
from epde.operators.singleobjective.selections import RouletteWheelSelection
from epde.operators.singleobjective.so_specific import SizeRestriction, FractionElitism

class BaselineDirector(OptimizationPatternDirector):
    def __init__(self) -> None:
        super().__init__()
        self.builder = StrategyBuilder(EvolutionaryStrategy)

    def use_baseline(self, params: dict, sparsity_cls=None,
                     sparsity_kwargs: dict = None, **kwargs):
        variation_params = params.get('variation_params', {})
        mutation_params = params.get('mutation_params', {})

        add_kwarg_to_operator = partial(add_base_param_to_operator, target_dict = kwargs)

        elitism = FractionElitism()

        mutation = get_singleobjective_mutation(mutation_params = mutation_params)
        elitism_cond_for_mutation = lambda candidate: candidate.elite
        mutation = map_operator_between_levels(mutation, 'chromosome level', 'population level', 
                                               element_condition = elitism_cond_for_mutation)

        variation = get_singleobjective_variation(variation_params = variation_params)
        # variation = map_operator_between_levels(variation, 'chromosome level', 'population level')

        selection = RouletteWheelSelection(['parents_fraction'])
        add_kwarg_to_operator(operator = selection)

        # Honours the configured sparsity operator: this branch used to
        # hardcode VWSR, so EpdeSearch(multiobjective_mode=False,
        # sparsity_cls='lasso') silently ran the other estimator.
        sparsity = build_sparsity_operator(sparsity_cls, sparsity_kwargs)
        coeff_calc = LinRegBasedCoeffsEquation()
        # Single-objective fitness: discrepancy is always computed (it
        # drives diagnostics and any right-part work); when the configured
        # single_objective_metric is 'instability', the instability filler
        # is added too so equation_terms_stability has a value to read.
        # Which attribute the optimizer actually minimises is chosen by
        # SoEq.use_default_singleobjective_function (the objective reader).
        #
        # Built BARE, exactly as MOEA/D builds it, so the discrepancy option
        # resolves from the search configuration at compute time. This branch
        # used to hardcode a separate WAPE-only filler, which made
        # EpdeSearch(multiobjective_mode=False, discrepancy_metric=...) a
        # silent no-op -- the one objective setting that never reached the
        # single-objective search.
        disc = Discrepancy()
        objectives = [disc]
        if active_config().objectives.single_objective_metric == 'instability':
            objectives.append(Instability())
        eq_fitness = SolverFreeFitness(['penalty_coeff'], objectives=objectives, primary=disc)
        add_kwarg_to_operator(operator = eq_fitness)
        # No suboperators on the host: it scores an already-fitted equation.
        # ``sparsity`` / ``coeff_calc`` go onto the right-part selector below,
        # which fits every candidate target before asking for its score.

        fitness_cond = lambda x: not getattr(x, 'fitness_calculated')
        sys_fitness = map_operator_between_levels(eq_fitness, 'gene level', 'chromosome level', 
                                                  objective_condition = fitness_cond)
        pop_fitness = map_operator_between_levels(sys_fitness, 'chromosome level', 'population level') # TODO: edit in operator_mappers.py 
        
        # Fitness-based right-part selection with zero-term pruning, the
        # same machinery MOEA/D uses (EqRightPartSelector fits each candidate
        # target with ``sparsity`` + ``coeff_calc``, ranks them by the
        # discrepancy ``eq_fitness`` returns with force_out_of_place=True,
        # then remove_zero_terms prunes the winner). This
        # replaces the old RandomRHPSelector: random selection left the
        # structure unpruned, so the in-place fitness saw the full feature
        # matrix against a sparse weight vector (shape mismatch). The
        # pruned structure keeps features aligned with weights_final, and
        # using the data-driven RPS also makes the single-objective search
        # consistent with the multi-objective one.
        rps_cond = lambda x: any([not elem_eq.right_part_selected for elem_eq in x.vals])
        eq_right_part_selector = EqRightPartSelector()
        eq_right_part_selector.set_suboperators({'fitness_calculation': eq_fitness,
                                                 'sparsity': sparsity,
                                                 'coeff_calc': coeff_calc})
        sys_rps_inner = SoEqRightPartSelector()
        sys_rps_inner.set_suboperators({'eq_right_part_selector': eq_right_part_selector})
        # rps_cond is a per-chromosome predicate; on a chromosome->population
        # map it is the element_condition (objective_condition would be
        # handed the population object, which has no ``.vals``).
        pop_rps = map_operator_between_levels(sys_rps_inner, 'chromosome level', 'population level',
                                              element_condition = rps_cond) # TODO: edit in operator_mappers.py

        population_pruner = SizeRestriction()

        self.builder = add_sequential_operators(self.builder, [('right part selection 1', pop_rps),
                                                               ('fitness evaluation 1', pop_fitness),
                                                               ('selection', selection),
                                                               ('variation', variation),
                                                               ('right part selection 2', pop_rps),
                                                               ('fitness evaluation 2', pop_fitness),
                                                               ('elitism', elitism),
                                                               ('mutation', mutation), 
                                                               ('right part selection 3', pop_rps),
                                                               ('fitness evaluation 3', pop_fitness),
                                                               ('size restriction', population_pruner)]) 
                                                               # TODO: assess the correctness of the pipe element return and general linkage
