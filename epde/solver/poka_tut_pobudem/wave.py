import numpy as np
import copy
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

def solve_with_solver(solver_type, solver_config, system, grids, data):
    adapter = SolverFactory.create(solver_type, **solver_config)
    solutions, loss = adapter.solve(system, grids, data)
    return solutions, loss

# ----------------------------------------------------------------------
# Волновое уравнение (PDE) – только DeepXDE
# ----------------------------------------------------------------------
print("=" * 60)
print("Wave equation (PDE) – DeepXDE PINN solver")
print("=" * 60)

# Генерация данных (аналитическое решение)
nx, nt = 50, 50
x = np.linspace(0, 1, nx)
t = np.linspace(0, 2, nt)
X_grid, T_grid = np.meshgrid(t, x, indexing='ij')
exact = np.sin(np.pi * X_grid) * np.cos(np.pi * T_grid)
data_wave = exact + 0.01 * np.random.normal(size=exact.shape)

# Создание пула EPDE
search_wave = EpdeSearch(
    use_solver=False,
    coordinate_tensors=(T_grid, X_grid),
    verbose_params={'show_iter_idx': False},
    device='cpu'
)
search_wave.set_preprocessor(default_preprocessor_type='FD', preprocessor_kwargs={})
search_wave.create_pool(
    data=data_wave,
    variable_names=['u'],
    max_deriv_order=(2, 2),
    additional_tokens=[]
)

# Уравнение волновое
eq_str = '1.0 * d^2u/dx1^2{power: 1.0} = d^2u/dx0^2{power: 1.0}'
soeq_wave = translate_equation(eq_str, search_wave.pool, all_vars=['u'])
eq_wave = soeq_wave.vals['u']
eq_wave.main_var_to_explain = 'u'
eq_wave.weights_internal = np.ones(len(eq_wave.structure) - 1)
eq_wave.weights_internal_evald = True
eq_wave.weights_final_evald = True

system_wave = SoEq(search_wave.pool, {})
system_wave.vals = Chromosome({'u': eq_wave}, {})
system_wave.moeadd_set = True

# Конфигурация DeepXDE (увеличиваем параметры для PDE)
solver_config_deepxde = EvolutionaryParams().get_default_params_for_operator('DeepXDEBasedFitness')
solver_config_deepxde['num_domain'] = 2000
solver_config_deepxde['num_boundary'] = 500
solver_config_deepxde['num_initial'] = 500
solver_config_deepxde['epochs'] = 3000

solutions_wave, loss_wave = solve_with_solver(
    "deepxde",
    solver_config_deepxde,
    system_wave,
    [T_grid, X_grid],
    [data_wave.flatten()]
)

soln_wave = solutions_wave[0].reshape(data_wave.shape)
print(f"Loss (RMSE): {loss_wave:.6f}")
print(f"Max error: {np.max(np.abs(soln_wave - exact)):.6f}")