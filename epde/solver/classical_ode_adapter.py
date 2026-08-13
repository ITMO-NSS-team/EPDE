import numpy as np
from scipy.integrate import solve_ivp
from epde.solver.base import BaseSolverAdapter
from epde.solver.ode_converter import ODEToFirstOrder
from epde.structure.main_structures import Equation, SoEq
import epde.globals as global_var
import sympy as sp


class ClassicalODEAdapter(BaseSolverAdapter):
    def __init__(self, **config):
        self.method = config.get('method', 'RK45')
        self.rtol = config.get('rtol', 1e-6)
        self.atol = config.get('atol', 1e-9)
        self.rhs = config.get('rhs')
        self.y0 = config.get('y0')
        self._auto_rhs = self.rhs is None
        self._converter = ODEToFirstOrder() if self._auto_rhs else None

    def _get_order(self, equation):
        if not self._auto_rhs:
            return None
        expr = self._converter._build_expression(equation, equation.main_var_to_explain)
        u = sp.Function(equation.main_var_to_explain)(self._converter.t)
        max_order = 0
        for term in sp.preorder_traversal(expr):
            if isinstance(term, sp.Derivative) and term.args[0] == u:
                order = term.args[1][1]
                if order > max_order:
                    max_order = order
        return max_order

    def _get_initial_conditions(self, equation, data):
        if self.y0 is not None:
            return self.y0
        order = self._get_order(equation)
        if order == 1:
            return [data[0]]
        else:
            raise ValueError(f"Для уравнения порядка {order} необходимо явно задать y0 в конфигурации.")

    def solve(self, equation_or_system, grids, data):
        print("[DEBUG] ClassicalODEAdapter.solve called")
        print(f"[DEBUG] equation_or_system type: {type(equation_or_system)}")
        print(f"[DEBUG] data type: {type(data)}")
        if isinstance(data, (list, tuple)):
            print(f"[DEBUG] data length: {len(data)}")
            for i, d in enumerate(data):
                print(f"[DEBUG] data[{i}].shape: {d.shape}")
        else:
            print(f"[DEBUG] data.shape: {data.shape}")

        if len(grids) != 1:
            raise ValueError("ClassicalODEAdapter работает только с 1D временной сеткой.")
        mask = global_var.grid_cache.g_func_mask
        t_full = grids[0].flatten()
        t_masked = t_full[mask]
        t_span = (t_masked.min(), t_masked.max())
        t_eval = t_masked

        # Определяем правую часть и начальные условия
        if isinstance(equation_or_system, Equation):
            if self.rhs is not None:
                rhs = self.rhs
                y0 = self.y0
                if y0 is None:
                    raise ValueError("Для явной rhs необходимо указать y0.")
            else:
                rhs = self._converter.equation_to_rhs(equation_or_system, equation_or_system.main_var_to_explain)
                y0 = self._get_initial_conditions(equation_or_system, data)
        elif isinstance(equation_or_system, SoEq):
            # Для системы: rhs и y0 должны быть заданы явно
            if self.rhs is None:
                raise NotImplementedError(
                    "Автоматическое преобразование систем ОДУ пока не поддерживается. Укажите rhs в конфигурации.")
            rhs = self.rhs
            y0 = self.y0
            if y0 is None:
                raise ValueError("Для системы необходимо указать y0 в конфигурации.")
        else:
            raise TypeError("Unsupported equation type")

        sol = solve_ivp(rhs, t_span, y0, method=self.method,
                        t_eval=t_eval, rtol=self.rtol, atol=self.atol)
        if not sol.success:
            raise RuntimeError(f"ODE solver failed: {sol.message}")

        solutions = [sol.y[i] for i in range(sol.y.shape[0])]

        print(f"[DEBUG] solutions length: {len(solutions)}")
        for i, s in enumerate(solutions):
            print(f"[DEBUG] solutions[{i}].shape: {s.shape}")
        # data может быть списком массивов (для SoEq) или одним массивом (для Equation)
        if isinstance(data, (list, tuple)):
            loss = np.mean([np.sqrt(np.mean((solutions[i] - data[i]) ** 2)) for i in range(len(solutions))])
        else:
            loss = np.sqrt(np.mean((solutions[0] - data) ** 2))
        return solutions, loss

    def get_requirements(self):
        return {
            'form': 'explicit_ode',
            'needs_initial_conditions': True,
            'needs_boundary_conditions': False,
            'supports_multisample': False
        }
