import numpy as np

from epde.operators.utils.template import CompoundOperator
from epde.optimizers.single_criterion.optimizer import Population

class SizeRestriction(CompoundOperator):
    def apply(self, objective: Population, arguments: dict):
        self_args, subop_args = self.parse_suboperator_args(arguments = arguments)    
        objective.population = objective.sort()[:objective.length]
        return objective

    def use_default_tags(self):
        self._tags = {'size restriction', 'population level', 'no suboperators', 'standard'}    

class FractionElitism(CompoundOperator):
    def apply(self, objective: Population, arguments: dict):
        self_args, subop_args = self.parse_suboperator_args(arguments = arguments)    

#
#        if isinstance(self.params['elite_fraction'], float):
#            assert self.params['elite_fraction'] <= 1 and self.params['elite_fraction'] >= 0
#            fraction = int(np.ceil(self.params['elite_fraction'] * len(objective.population)))
        for idx, elem in enumerate(objective.population):
            if idx == 0:
                setattr(elem, 'elite', 'immutable')
#            if idx < fraction:
#                setattr(elem, 'elite', 'elite')
            else:
                setattr(elem, 'elite', 'non-elite')
                
        # print('in elitism')
        # print([[(eq.fitness_calculated, eq.text_form) for eq in candidate.vals] 
               # for candidate in objective])
        return objective
    
    @property
    def operator_tags(self):
        return {'elitism', 'population level', 'auxilary', 'no suboperators', 'standard'}        