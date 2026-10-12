"""Side-by-side reconstruction on known laws, without evolutionary searches."""
import sys
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
from pathlib import Path

PIC = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PIC), str(PIC.parent.parent)]

from epde_bench import datasets, load_config, resolve_for_problem
from epde_bench.reconstruction import equation_fields, parse_equation, r2_score, simulate_ode


class ParseTests(unittest.TestCase):
    def test_terms_constant_and_target(self):
        terms, constant, target, coef = parse_equation(
            '-4.0 * u{power: 1.0} + 1.5e+00 * x{power: 1.0, dim: 0.0} + 0.25 = d^2u/dx0^2{power: 1.0}')
        self.assertEqual([c for c, _ in terms], [-4.0, 1.5])
        self.assertEqual(constant, 0.25)
        self.assertEqual(target[0][0], 'd^2u/dx0^2')
        self.assertEqual(coef, 1.0)

    def test_compact_sum_keeps_scientific_exponent(self):
        terms, constant, target, coef = parse_equation(
            '1e+05*u{power:1.0}+2*u{power:2.0}=du/dx0{power:1.0}')
        self.assertEqual([c for c, _ in terms], [100000.0, 2.0])

    def test_system_brace_prefix(self):
        terms, constant, target, _ = parse_equation(
            '/ -20.0 * v{power: 1.0} * u{power: 1.0} + 20.0 * u{power: 1.0} + 0.0 = du/dx0{power: 1.0}')
        self.assertEqual([c for c, _ in terms], [-20.0, 20.0])
        self.assertEqual(parse_equation('\\ 1.0 * u{power: 1.0} = dv/dx0{power: 1.0}')[2][0][0], 'dv/dx0')


class KnownLawTests(unittest.TestCase):
    def setUp(self):
        import matplotlib
        matplotlib.use('Agg')

    def problem(self, name):
        p = datasets.load(name)
        return p, resolve_for_problem(load_config(name), p)

    def test_ode_law_reproduces_the_data(self):
        p, search = self.problem('ode')
        observed, predicted = equation_fields(p, p.truth[0], search)
        self.assertGreater(r2_score(observed, predicted), 0.99)
        t, simulated, data = simulate_ode(p, p.truth, search)
        self.assertGreater(r2_score(data['u'], simulated['u']), 0.99)

    def test_system_is_integrated_jointly(self):
        p, search = self.problem('lv')
        t, simulated, data = simulate_ode(p, p.truth, search)
        for var in ('u', 'v'):
            self.assertGreater(r2_score(data[var], simulated[var]), 0.99)

    def test_failed_integration_has_no_success_score(self):
        from epde_bench.reconstruction import plot_reconstruction
        p, search = self.problem('ode')
        fake = SimpleNamespace(success=False, message='integration stopped',
                               t=np.array([1.0]), y=np.array([[2.0], [0.0]]))
        with patch('scipy.integrate.solve_ivp', return_value=fake):
            result = simulate_ode(p, p.truth, search)
            self.assertFalse(result.success)
            self.assertLess(result.coverage, 1.0)
            fig, scores = plot_reconstruction(p, p.truth, search)
            self.assertTrue(np.isnan(scores['trajectory u']))
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_time_tail_is_integrated_without_reanchoring(self):
        p, search = self.problem('ode')
        result = simulate_ode(p, p.truth, search, start_fraction=0.8)
        self.assertTrue(result.success)
        t, simulated, observed = result
        self.assertLess(len(t), p.shape[0] // 3)
        self.assertGreater(r2_score(observed['u'], simulated['u']), 0.98)

    def test_independent_baseline_does_not_build_native_polynomial_pipeline(self):
        from epde_bench.problem import Problem, mesh
        t = np.linspace(0, 1, 9)
        p = Problem('small', 'Small', 'ode', 'synthetic', mesh(t), {'u': np.exp(-t)}, ('t',))
        search = {'preprocessing': {'default_preprocessor_type': 'poly',
                 'preprocessor_kwargs': {'polynomial_window': 51, 'poly_order': 4, 'mp_poolsize': 1},
                 'max_deriv_order': [1]}, 'domain': {'boundary_width': 1},
                 '_reconstruction_differentiation': 'finite_difference'}
        result = simulate_ode(p, ['-1.0 * u{power:1.0} = du/dx0{power:1.0}'], search)
        self.assertTrue(result.success)
        self.assertGreater(r2_score(result.observed['u'], result.simulated['u']), .999)

    def test_diffusion_uses_nearest_neighbour_second_derivative(self):
        from epde_bench.problem import Problem
        from epde_bench.reconstruction import simulate_pde
        x, t = np.linspace(0, np.pi, 51), np.linspace(0, .001, 5)
        decay = -4 * np.sin(24 * np.pi / 100)**2 / (x[1]-x[0])**2
        tt, xx = np.meshgrid(t, x, indexing='ij')
        field = np.exp(decay * tt) * np.sin(24 * xx)
        p = Problem('heat_modes', 'Heat modes', 'pde_1d', 'synthetic', (tt, xx),
                    {'u': field}, ('t', 'x'))
        result = simulate_pde(p, ['1.0 * d^2u/dx1^2{power:1.0} = du/dx0{power:1.0}'], {})
        self.assertTrue(result.success)
        np.testing.assert_allclose(result.simulated['u'], field, atol=1e-6)

    def test_pde_requires_a_finite_time_interval(self):
        from epde_bench.problem import Problem
        from epde_bench.reconstruction import simulate_pde
        tt, xx = np.meshgrid([0.0], np.arange(5.), indexing='ij')
        p = Problem('one', 'One', 'pde_1d', 'synthetic', (tt, xx),
                    {'u': np.ones((1, 5))}, ('t', 'x'))
        with self.assertRaisesRegex(ValueError, 'two time'):
            simulate_pde(p, ['0.0 = du/dx0{power: 1.0}'], {})

    def test_explicit_heat_equation_is_solved_in_space(self):
        from epde_bench.problem import Problem
        from epde_bench.reconstruction import simulate_pde
        t, x = np.linspace(0, .2, 21), np.linspace(0, np.pi, 41)
        tt, xx = np.meshgrid(t, x, indexing='ij')
        p = Problem('heat', 'Heat', 'pde_1d', 'synthetic', (tt, xx),
                    {'u': np.exp(-tt) * np.sin(xx)}, ('t', 'x'))
        cfg = {'preprocessing': {'default_preprocessor_type': 'FD'},
               'search_space': {'max_deriv_order': [1, 2]}, 'domain': {'boundary': 0}}
        result = simulate_pde(p, ['1.0 * d^2u/dx1^2{power: 1.0} = du/dx0{power: 1.0}'], cfg)
        self.assertTrue(result.success)
        self.assertGreater(r2_score(p.data['u'], result.simulated['u']), 0.999)

    def test_implicit_form_is_not_integrated(self):
        p, search = self.problem('ode')
        implicit = ['1.0 * u{power: 1.0} * d^2u/dx0^2{power: 1.0} = du/dx0{power: 1.0}']
        self.assertIsNone(simulate_ode(p, implicit, search))

    def test_pde_fields_have_interior_shape(self):
        p, search = self.problem('burgers')
        observed, predicted = equation_fields(p, p.truth[0], search)
        self.assertEqual(observed.shape, predicted.shape)
        self.assertGreater(r2_score(observed, predicted), 0.99)


if __name__ == '__main__':
    unittest.main()
