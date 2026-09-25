import pytest

from tests.functional.operator_factory import FitnessOperatorFactory
from tests.functional.scenarios.burgers.burgers import BurgersTest
from epde.operators.utils.default_parameter_loader import EvolutionaryParams

operator_params_l2lr = EvolutionaryParams().get_default_params_for_operator(
    "DiscrepancyBasedFitnessWithCV"
)

operator_params_deepxde = EvolutionaryParams().get_default_params_for_operator(
    'DeepXDEBasedFitness'
)

ALL_CASES = [
    ("DeepXDEBasedFitness", operator_params_deepxde),
    ("L2LRFitness", operator_params_l2lr),
]

@pytest.mark.functional
@pytest.mark.parametrize("operator_name, params", ALL_CASES)
def test_burgers_sindy(operator_name, params, runtime_options):
    import epde.globals as global_var
    global_var.solution_guess_nn = None
    if operator_name not in runtime_options["operators"]:
        pytest.skip(f"{operator_name} skipped by --operators")

    operator = FitnessOperatorFactory.create(operator_name, params)
    scenario = BurgersTest(noise_level=0)

    if runtime_options["discovery"]:
        search_obj = scenario.make_search()
        search_obj, elapsed = scenario.run_sindy_discovery(
            search_obj,
            report_dir=runtime_options["report_dir"] if runtime_options["report"] else None,
            operator_name=operator_name,
        )
        assert elapsed < 600
    else:
        ok, elapsed = scenario.run(operator)
        assert ok
        assert elapsed < 60