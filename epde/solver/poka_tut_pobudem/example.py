import numpy as np
import copy
from scipy.integrate import solve_ivp
from epde.interface.interface import EpdeSearch
from epde.interface.equation_translator import translate_equation
from epde.structure.main_structures import SoEq, Chromosome
from epde.interface.token_family import TFPool
from epde.operators.utils.default_parameter_loader import EvolutionaryParams
from epde.solver.factory import SolverFactory

def create_equation_from_str(eq_str, target_var, base_pool, all_vars):
    families_copy = [copy.deepcopy(fam) for fam in base_pool.families]
    for fam in families_copy:
        if hasattr(fam, 'variable') and fam.variable is not None and fam.variable != target_var:
            fam.status['demands_equation'] = False
    temp_pool = TFPool(families_copy)
    soeq = translate_equation(eq_str, temp_pool, all_vars=[target_var])
    return soeq.vals[target_var]

def lv_rhs(t, y):
    u, v = y
    alpha, beta, gamma, delta = 2/3, 4/3, 1.0, 1.0
    du = alpha * u - beta * u * v
    dv = delta * u * v - gamma * v
    return [du, dv]

t = np.linspace(0, 20, 200)
sol_ref = solve_ivp(lv_rhs, (0,20), [1.0,1.0], t_eval=t, method='RK45', rtol=1e-6, atol=1e-9)
exact_u, exact_v = sol_ref.y

data_u = exact_u + 0.01 * np.random.normal(size=exact_u.shape)
data_v = exact_v + 0.01 * np.random.normal(size=exact_v.shape)

search = EpdeSearch(
    use_solver=False,
    multiobjective_mode=True,
    coordinate_tensors=[t],
    verbose_params={'show_iter_idx': False},
    device='cpu'
)
search.set_preprocessor(default_preprocessor_type='FD', preprocessor_kwargs={})
search.create_pool(data=[data_u, data_v], variable_names=['u', 'v'], max_deriv_order=1, additional_tokens=[])

correct_eqs = [
    '0.6666666666666666 * u{power: 1.0} + -1.3333333333333333 * u{power: 1.0} * v{power: 1.0} = du/dx0{power: 1.0}',
    '1.0 * u{power: 1.0} * v{power: 1.0} + -1.0 * v{power: 1.0} = dv/dx0{power: 1.0}'
]

eq_u = create_equation_from_str(correct_eqs[0], 'u', search.pool, ['u', 'v'])
eq_u.main_var_to_explain = 'u'
eq_u.weights_internal = np.ones(len(eq_u.structure) - 1)
eq_u.weights_internal_evald = True
eq_u.weights_final_evald = True

eq_v = create_equation_from_str(correct_eqs[1], 'v', search.pool, ['u', 'v'])
eq_v.main_var_to_explain = 'v'
eq_v.weights_internal = np.ones(len(eq_v.structure) - 1)
eq_v.weights_internal_evald = True
eq_v.weights_final_evald = True

system = SoEq(search.pool, {})
system.vals = Chromosome({'u': eq_u, 'v': eq_v}, {})
system.moeadd_set = True

def solve_with_solver(solver_type, solver_config, system, t, data):
    adapter = SolverFactory.create(solver_type, **solver_config)
    solutions, loss = adapter.solve(system, [t], data)
    return solutions, loss


print("=" * 50)
print("Classical ODE solver (RK45)")
print("=" * 50)

solver_config_classical = {
    "method": "RK45",
    "rtol": 1e-6,
    "atol": 1e-9,
    "rhs": lv_rhs,
    "y0": [1.0, 1.0]
}
solutions_cl, loss_cl = solve_with_solver("classical_ode", solver_config_classical, system, t, [data_u, data_v])
print(f"Loss (RMSE): {loss_cl:.6f}")
print(f"Max error u: {np.max(np.abs(solutions_cl[0] - exact_u)):.6f}")
print(f"Max error v: {np.max(np.abs(solutions_cl[1] - exact_v)):.6f}")

print("\n" + "=" * 50)
print("DeepXDE (PINN) solver")
print("=" * 50)

solver_config_deepxde = EvolutionaryParams().get_default_params_for_operator('DeepXDEBasedFitness')
solutions_dx, loss_dx = solve_with_solver("deepxde", solver_config_deepxde, system, t, [data_u, data_v])
print(f"Loss (RMSE): {loss_dx:.6f}")
print(f"Max error u: {np.max(np.abs(solutions_dx[0] - exact_u)):.6f}")
print(f"Max error v: {np.max(np.abs(solutions_dx[1] - exact_v)):.6f}")