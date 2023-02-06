from epde.optimizers.builder import add_sequential_operators, OptimizationPatternDirector
from functools import partial

from epde.operators.utils.operator_mappers import map_operator_between_levels
from epde.operators.utils.template import add_param_to_operator

from epde.operators.singleobjective.right_part_selection import RandomRHPSelector
from epde.operators.singleobjective.fitness import L2Fitness
from epde.operators.singleobjective.sparsity import LASSOSparsity
from epde.operators.singleobjective.coeff_calculation import LinRegBasedCoeffsEquation
from epde.operators.singleobjective.mutations import get_singleobjective_mutation
from epde.operators.singleobjective.variation import get_singleobjective_variation
from epde.operators.singleobjective.selections import RouletteWheelSelection
from epde.operators.singleobjective.so_specific import SizeRestriction

class BaselineDirector(OptimizationPatternDirector):
    def use_baseline(self, variation_params: dict = {}, mutation_params: dict = {}, **kwargs):
        add_kwarg_to_operator = partial(add_param_to_operator, target_dict = kwargs)

        mutation = get_singleobjective_mutation(mutation_params = mutation_params)
        
        variation = get_singleobjective_variation(variation_params = variation_params)
        selection = RouletteWheelSelection(['parents_fraction'])
        add_kwarg_to_operator(operator = selection, labeled_base_val = {'parents_fraction' : 0.4})



        sparsity = LASSOSparsity()
        coeff_calc = LinRegBasedCoeffsEquation()
        eq_fitness = L2Fitness(['penalty_coeff'])
        add_kwarg_to_operator(operator = eq_fitness, labeled_base_val = {'penalty_coeff' : 0.2})
        eq_fitness.set_suboperators({'sparsity' : sparsity, 'coeff_calc' : coeff_calc})

        fitness_cond = lambda x: not getattr(x, 'fitness_calculated')
        sys_fitness = map_operator_between_levels(eq_fitness, 'gene level', 'chromosome level', fitness_cond)
        pop_fitness = map_operator_between_levels(sys_fitness, 'chromosome level', 'population level') # TODO: edit in operator_mappers.py 
        
        # Check for optimality of operator application
        rps_cond = lambda x: any([not elem_eq.right_part_selected for elem_eq in x.vals])
        right_part_selector = RandomRHPSelector()
        sys_rps = map_operator_between_levels(right_part_selector, 'gene level', 'chromosome level', rps_cond)
        pop_rps = map_operator_between_levels(sys_rps, 'chromosome level', 'population level') # TODO: edit in operator_mappers.py 

        population_pruner = SizeRestriction()

        self.builder = add_sequential_operators(self.builder, [('right part selection 1', pop_rps),
                                                               ('fitness evaluation 1', pop_fitness),
                                                               ('selection', selection),
                                                               ('variation', variation),
                                                               ('mutation', mutation), 
                                                               ('right part selection 2', pop_rps),
                                                               ('fitness evaluation 2', pop_fitness),
                                                               ('size restriction', population_pruner)]) 
                                                               # TODO: assess the correctness of the pipe element return and general linkage