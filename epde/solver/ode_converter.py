import sympy as sp
import numpy as np
from typing import Callable
from epde.structure.main_structures import Equation

class ODEToFirstOrder:
    def __init__(self, t_symbol: str = 't'):
        self.t = sp.Symbol(t_symbol, real=True)
        self.debug = True  # можно выключить после отладки

    def _build_expression(self, eq: Equation, var_name: str) -> sp.Expr:
        u = sp.Function(var_name)(self.t)
        expr = 0
        if self.debug:
            print(f"[DEBUG] Building expression for equation, target_idx={eq.target_idx}")
        for term_idx, term in enumerate(eq.structure):
            coeff = eq.weights_final[term_idx] if term_idx < len(eq.weights_final) else 1.0
            term_expr = 1
            if self.debug:
                print(f"  Term {term_idx}: {term.name}, coeff={coeff}")
            for factor in term.structure:
                deriv_code = getattr(factor, "deriv_code", None)
                is_deriv = factor.is_deriv and deriv_code is not None and len(deriv_code) > 0 and not all(v is None for v in deriv_code)
                if self.debug:
                    print(f"    Factor: is_deriv={factor.is_deriv}, variable={getattr(factor, 'variable', None)}, deriv_code={deriv_code}, params={factor.params}, is_deriv_flag={is_deriv}")
                if is_deriv:
                    # Настоящая производная (deriv_code содержит числа, не None)
                    order = len(deriv_code)
                    deriv = sp.Derivative(u, (self.t, order))
                    power = factor.params[-1] if factor.params else 1.0
                    term_expr *= deriv ** power
                    if self.debug:
                        print(f"      Derivative order {order}, power {power}, deriv={deriv}")
                else:
                    # Не производная – переменная или константа
                    if hasattr(factor, 'variable') and factor.variable is not None:
                        power = factor.params[-1] if factor.params else 1.0
                        term_expr *= u ** power
                        if self.debug:
                            print(f"      Variable {factor.variable}, power {power}")
                    else:
                        if hasattr(factor, 'params') and len(factor.params) > 0:
                            term_expr *= sp.Float(factor.params[-1])
                            if self.debug:
                                print(f"      Constant {factor.params[-1]}")
                        else:
                            if self.debug:
                                print(f"      Unknown factor, ignored")
            if term_idx == eq.target_idx:
                expr -= coeff * term_expr
                if self.debug:
                    print(f"  Target term, adding -{coeff} * {term_expr}")
            else:
                expr += coeff * term_expr
                if self.debug:
                    print(f"  Adding +{coeff} * {term_expr}")
        if self.debug:
            print(f"  Total expression: {expr}")
        return expr

    def equation_to_rhs(self, eq: Equation, var_name: str) -> Callable:
        expr = self._build_expression(eq, var_name)
        u = sp.Function(var_name)(self.t)

        derivs = [arg for arg in sp.preorder_traversal(expr) if isinstance(arg, sp.Derivative) and arg.args[0] == u]
        if not derivs:
            raise ValueError("Нет производных")
        max_order = max(d.args[1][1] for d in derivs)
        if self.debug:
            print(f"[DEBUG] max_order = {max_order}")

        u_deriv = sp.Derivative(u, (self.t, max_order))
        sol = sp.solve(expr, u_deriv)
        if self.debug:
            print(f"[DEBUG] sol = {sol}")
        if not sol:
            raise ValueError(f"Не удалось выразить {u_deriv} из уравнения {expr}")
        rhs_expr = sol[0]

        y_sym = [sp.Symbol(f'{var_name}_{i}', real=True) for i in range(max_order)]
        subs = {u: y_sym[0]}
        for i in range(1, max_order):
            subs[sp.Derivative(u, (self.t, i))] = y_sym[i]
        rhs_expr = rhs_expr.subs(subs)
        if self.debug:
            print(f"[DEBUG] rhs_expr after substitution = {rhs_expr}")

        rhs_expr = rhs_expr.replace(lambda x: isinstance(x, sp.Derivative), lambda x: 0)
        rhs_expr = sp.simplify(rhs_expr)
        if self.debug:
            print(f"[DEBUG] rhs_expr after derivative replacement = {rhs_expr}")

        derivatives = [y_sym[i+1] if i+1 < max_order else rhs_expr for i in range(max_order)]
        if self.debug:
            print(f"[DEBUG] derivatives = {derivatives}")

        rhs_func_args = sp.lambdify([self.t] + y_sym, derivatives, modules='numpy')
        def rhs_wrapper(t, y):
            if self.debug:
                print(f"[DEBUG] rhs_wrapper t={t}, y={y}")
            res = rhs_func_args(t, *y)
            if self.debug:
                print(f"[DEBUG] rhs_wrapper res = {res}")
            return np.array(res, dtype=float)
        return rhs_wrapper