from epde.operators.utils.template import CompoundOperator
from epde.optimizers.single_criterion.optimizer import Population

class SizeRestriction(CompoundOperator):
    def apply(self, objective: Population, arguments: dict):
        self_args, subop_args = self.parse_suboperator_args(arguments = arguments)    
        objective.population = objective.sort()[:objective.length]
        return objective

    def use_default_tags(self):
        self._tags = {'size restriction', 'population level', 'no suboperators', 'standard'}    