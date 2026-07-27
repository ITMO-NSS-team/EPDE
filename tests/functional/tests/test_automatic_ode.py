import pytest
import numpy as np
from epde.solver.classical_ode_adapter import ClassicalODEAdapter
from epde.interface.interface import EpdeSearch
from epde.interface.equation_translator import translate_equation

def test_automatic_logistic():
    t = np.linspace(0, 5, 100)
    exact = 1 / (1 + (1/0.1 - 1) * np.exp(-2 * t))
    data = exact + 0.01 * np.random.normal(size=t.shape)

    search = EpdeSearch(
        use_solver=False,
        coordinate_tensors=[t],
        verbose_params={'show_iter_idx': False},
        device='cpu'
    )
    search.set_preprocessor(default_preprocessor_type='FD', preprocessor_kwargs={})
    search.create_pool(data=data, variable_names=['u'], max_deriv_order=1, data_fun_pow=2, additional_tokens=[])

    eq_str = '2.0 * u{power: 1.0} + -2.0 * u{power: 2.0} = du/dx0{power: 1.0}'
    soeq = translate_equation(eq_str, search.pool, all_vars=['u'])
    eq = soeq.vals['u']

    eq.target_idx = 2
    eq.weights_final = [2.0, -2.0, 1.0]
    eq.weights_final_evald = True

    adapter = ClassicalODEAdapter(y0=[0.1])
    solutions, loss = adapter.solve(eq, [t], data)
    assert np.allclose(solutions[0], exact, rtol=1e-2)

def test_automatic_oscillator():
    t = np.linspace(0, 2*np.pi, 100)
    exact = np.cos(t)
    data = exact + 0.01 * np.random.normal(size=t.shape)

    search = EpdeSearch(
        use_solver=False,
        coordinate_tensors=[t],
        verbose_params={'show_iter_idx': False},
        device='cpu'
    )
    search.set_preprocessor(default_preprocessor_type='FD', preprocessor_kwargs={})
    search.create_pool(data=data, variable_names=['u'], max_deriv_order=2, additional_tokens=[])

    eq_str = '-1.0 * u{power: 1.0} = d^2u/dx0^2{power: 1.0}'
    soeq = translate_equation(eq_str, search.pool, all_vars=['u'])
    eq = soeq.vals['u']

    eq.target_idx = 1
    eq.weights_final = [-1.0, 1.0]
    eq.weights_final_evald = True

    adapter = ClassicalODEAdapter(y0=[1.0, 0.0])
    solutions, loss = adapter.solve(eq, [t], data)
    assert np.allclose(solutions[0], exact, rtol=1e-2)

def test_trigonometric_rhs():
    def rhs(t, y):
        return [np.cos(t)]
    t = np.linspace(0, 2*np.pi, 100)
    exact = np.sin(t)
    data = exact + 0.01 * np.random.normal(size=t.shape)

    search = EpdeSearch(
        use_solver=False,
        coordinate_tensors=[t],
        verbose_params={'show_iter_idx': False},
        device='cpu'
    )
    search.set_preprocessor(default_preprocessor_type='FD', preprocessor_kwargs={})
    search.create_pool(data=data, variable_names=['u'], max_deriv_order=1, additional_tokens=[])
    eq_str = '0.0 = du/dx0{power: 1.0}'
    soeq = translate_equation(eq_str, search.pool, all_vars=['u'])
    eq = soeq.vals['u']
    eq.weights_final = [0.0, 1.0]
    eq.weights_final_evald = True

    adapter = ClassicalODEAdapter(rhs=rhs, y0=[0.0])
    solutions, loss = adapter.solve(eq, [t], data)
    assert np.allclose(solutions[0], exact, rtol=1e-1, atol=1e-2)

if __name__ == "__main__":
    pytest.main([__file__])