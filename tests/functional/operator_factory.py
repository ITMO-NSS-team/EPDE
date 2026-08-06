from epde.operators.common.coeff_calculation import LinRegBasedCoeffsEquation
from epde.operators.common.sparsity import LASSOSparsity
from epde.operators.utils.operator_mappers import map_operator_between_levels
from epde.operators.utils.template import CompoundOperator
from epde.operators.common.fitness import SolverFreeFitness, SolverBasedFitness
from epde.operators.common.objectives import Discrepancy, Instability


class FitnessOperatorFactory:
    """Build a fitness operator for the functional test harness.

    The historical operator names ("L2LRFitness", "PIC",
    "DeepXDEBasedFitness") are kept as test fixtures and mapped onto the
    new host operators:

    * ``L2LRFitness``        -> SolverFreeFitness(WAPE discrepancy + instability)
    * ``PIC``                -> SolverBasedFitness(autograd, masked, PIC error + instability)
    * ``DeepXDEBasedFitness``-> SolverBasedFitness(deepxde, error + instability)
    """

    @staticmethod
    def create(name: str, params: dict) -> CompoundOperator:
        if name == 'L2LRFitness':
            disc = Discrepancy('wape')
            operator = SolverFreeFitness(list(params.keys()),
                                         objectives=[disc, Instability()], primary=disc)
            sparsity = LASSOSparsity()
            coeff_calc = LinRegBasedCoeffsEquation()
        elif name == 'PIC':
            primary = Discrepancy('pic')
            operator = SolverBasedFitness(list(params.keys()), objectives=[primary],
                                          primary=primary, instability=Instability(),
                                          backend='autograd', masked=True)
            sparsity = map_operator_between_levels(LASSOSparsity(), 'gene level', 'chromosome level')
            coeff_calc = map_operator_between_levels(LinRegBasedCoeffsEquation(), 'gene level', 'chromosome level')
        elif name == 'DeepXDEBasedFitness':
            primary = Discrepancy('deepxde')
            operator = SolverBasedFitness(list(params.keys()), objectives=[primary],
                                          primary=primary, instability=Instability(),
                                          backend='deepxde')
            sparsity = LASSOSparsity()
            coeff_calc = LinRegBasedCoeffsEquation()
        else:
            raise ValueError(f"Unknown operator: {name}")

        operator.set_suboperators({
            "sparsity": sparsity,
            "coeff_calc": coeff_calc,
        })
        operator.params = params

        if 'chromosome level' not in operator._tags:
            fitness_cond = lambda x: not getattr(x, "fitness_calculated", False)
            operator = map_operator_between_levels(
                operator,
                'gene level',
                "chromosome level",
                objective_condition=fitness_cond,
            )
        return operator
