import numpy as np

import epde.globals as global_var
from epde.operators.utils.template import CompoundOperator
from epde.optimizers.single_criterion.optimizer import Population

class SizeRestriction(CompoundOperator):
    key = 'SizeRestriction'

    def apply(self, objective: Population, arguments: dict):
        self_args, subop_args = self.parse_suboperator_args(arguments = arguments)          
        objective.population = objective.sort()[:objective.length]
        global_var.history.add([eq.fitness_value  for eq in objective.population[0]][0])        
        return objective

    def use_default_tags(self):
        self._tags = {'size restriction', 'population level', 'no suboperators', 'standard'}    

class ResamplingRightPartSelector(CompoundOperator):
    '''
    Chromosome-level right-part selection of the single-objective search that
    redraws what it declines.

    ``EqRightPartSelector`` declines an equation it cannot fit instead of
    redrawing it in place, as it once did (up to its outer cap of 50 passes
    per call). This operator gives the single-objective search that redraw
    back, from the population side: it runs the system selector
    (``right_part_selector``), and when an equation leaves it DECLINED it
    scores the candidate (``chromosome_fitness``) and hands it to the shared,
    budgeted resample of the multi-objective search
    (``moeadd_specific.regenerate_degenerate_equations``, up to
    ``RESAMPLE_BUDGET`` redraws per declined equation). With
    ``max_passes=0``, a fit the fitness host stamps is not redrawn: this
    search never regenerated those, and they still sort last
    (``supplementary.is_stamped``). Beside such a fit a declined equation is
    not redrawn either: the candidate sorts last whatever the redraws give,
    so it is given up at once (recorded ``exhausted`` in the loop stats). A
    candidate whose redraws all decline keeps its stamp and sorts last too.

    Applied only where selection runs (an equation not selected yet), as the
    in-place redraw was: a candidate that already went through it is not
    redrawn again on later passes over the population. A candidate with no
    declined equation leaves exactly as the selector left it, unscored, for
    the population's fitness block.
    '''
    key = 'ResamplingRightPartSelector'

    def apply(self, objective, arguments: dict):
        from epde.operators.common.right_part_selection import _rps_declined
        from epde.operators.multiobjective.moeadd_specific import regenerate_degenerate_equations
        self_args, subop_args = self.parse_suboperator_args(arguments = arguments)
        selector = self.suboperators['right_part_selector']
        fitness = self.suboperators['chromosome_fitness']
        selector.apply(objective, subop_args['right_part_selector'])
        if not any(_rps_declined(eq) for eq in objective.vals):
            return
        fitness.apply(objective, subop_args['chromosome_fitness'])
        regenerate_degenerate_equations(objective, selector, fitness,
                                        subop_args['right_part_selector'],
                                        subop_args['chromosome_fitness'], max_passes=0)

    def use_default_tags(self):
        self._tags = {'right part selection', 'chromosome level', 'contains suboperators', 'inplace'}

class FractionElitism(CompoundOperator):
    key = 'FractionElitism'

    def apply(self, objective: Population, arguments: dict):
        self_args, subop_args = self.parse_suboperator_args(arguments = arguments)    

        objective.population = objective.sort()
        for idx, elem in enumerate(objective.population):
            if idx == 0:
                setattr(elem, 'elite', 'immutable')
            else:
                setattr(elem, 'elite', 'non-elite')
                
        return objective
    
    @property
    def operator_tags(self):
        return {'elitism', 'population level', 'auxilary', 'no suboperators', 'standard'}        