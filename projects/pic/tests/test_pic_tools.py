"""Regression checks for PIC tooling; no evolutionary search is required."""
import contextlib
import io
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

PIC = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PIC), str(PIC.parent.parent)]
from epde_bench import env
env.pin_blas_threads(1)
import numpy as np
from epde_bench.problem import Problem, mesh
from epde_bench.metrics import _parse_factor
from epde_bench.truthcheck import _factor_arrays
from epde_bench.config import load_config, resolve_for_problem


def fixture():
    t = np.linspace(0, 2, 121)
    y = np.sin(3*t) + np.random.default_rng(2).normal(0, .02, t.size)
    return Problem('test', 'test', 'ode', 'synthetic', mesh(t), {'u': y}, ('t',))


class DerivativeTests(unittest.TestCase):
    def assert_matches_pipeline(self, method, kwargs):
        from epde.preprocessing.preprocessor import ConcretePrepBuilder
        from epde.preprocessing.preprocessor_setups import PreprocessorSetup
        p = fixture()
        cfg = {'preprocessing': {'default_preprocessor_type': method,
               'preprocessor_kwargs': kwargs, 'max_deriv_order': [2]},
               'search_space': {'data_fun_pow': 2, 'deriv_fun_pow': 2}}
        setup = PreprocessorSetup(); setup.builder = ConcretePrepBuilder()
        if method == 'FD': setup.build_FD_preprocessing(**kwargs)
        else: setup.build_poly_diff_preprocessing(**kwargs)
        values, derivatives = setup.builder.prep_pipeline.run(p.data['u'], list(p.grids), [2])
        arrays = _factor_arrays(p, p.data, cfg)
        for token, expected in [('u{power: 1.0}', values),
                                ('u{power: 2.0}', values**2),
                                ('du/dx0{power: 1.0}', derivatives[:, 0]),
                                ('d^2u/dx0^2{power: 1.0}', derivatives[:, 1]),
                                ('d^2u/dx0^2{power: 2.0}', derivatives[:, 1]**2)]:
            np.testing.assert_allclose(arrays[_parse_factor(token)], expected, rtol=1e-12, atol=1e-12)

    def test_fd_matches_epde_with_noise(self):
        self.assert_matches_pipeline('FD', {})

    def test_poly_matches_epde_with_smoothing(self):
        self.assert_matches_pipeline('poly', {'polynomial_window': 15, 'poly_order': 4,
                                   'mp_poolsize': 1, 'use_smoothing': True, 'sigma': 1})

    def test_supplied_derivatives_are_used_for_every_variable(self):
        p = fixture(); p.data['v'] = 2*p.data['u']
        p.derivs = [np.full((121, 2), 7.), np.full((121, 2), 11.)]
        p.deriv_orders = [2] if p.derivs is not None else None
        cfg = {'preprocessing': {'max_deriv_order': [2], 'default_preprocessor_type': 'FD'},
               'search_space': {'data_fun_pow': 1, 'deriv_fun_pow': 1}}
        a = _factor_arrays(p, p.data, cfg)
        np.testing.assert_array_equal(a[_parse_factor('du/dx0{power: 1.0}')], 7.)
        np.testing.assert_array_equal(a[_parse_factor('d^2v/dx0^2{power: 1.0}')], 11.)

    def test_noise_on_supplied_derivatives_is_rejected(self):
        p = fixture(); p.derivs = [np.zeros((121, 2))]
        with self.assertRaisesRegex(ValueError, 'pre-computed derivatives'):
            p.noisy(1, 0)

    def test_replacing_fields_with_supplied_derivatives_is_rejected(self):
        from epde_bench.preprocessing import validate_data
        p = fixture(); p.derivs = [np.zeros((121, 2))]
        with self.assertRaisesRegex(ValueError, 'pre-computed derivatives'):
            validate_data(p, {'u': p.data['u'] + .1})

    def test_bad_derivative_stack_shape_is_rejected(self):
        p = fixture(); p.derivs = [np.ones((121, 1))]
        p.deriv_orders = [2] if p.derivs is not None else None
        cfg = {'preprocessing': {'max_deriv_order': [2], 'default_preprocessor_type': 'FD'},
               'search_space': {'data_fun_pow': 1, 'deriv_fun_pow': 1}}
        with self.assertRaisesRegex(ValueError, 'derivatives must be finite'):
            _factor_arrays(p, p.data, cfg)

    def test_supplied_orders_are_checked_by_check_and_run(self):
        from epde_bench.runner import build_search
        p = fixture(); p.derivs = [np.ones((121, 2))]; p.deriv_orders = [2]
        cfg = {'preprocessing': {'max_deriv_order': [1]}}
        for action in (_factor_arrays, build_search):
            with self.subTest(action=action.__name__), self.assertRaisesRegex(ValueError, 'orders'):
                action(p, p.data, cfg) if action is _factor_arrays else action(p, cfg)

    def test_same_size_derivative_axis_reordering_is_rejected(self):
        from epde_bench.runner import build_search
        p = fixture(); p.grids = mesh(np.arange(3.), np.arange(4.), np.arange(5.))
        p.data = {'u': np.ones((3, 4, 5))}; p.derivs = [np.ones((60, 3))]
        p.deriv_orders = [1, 1, 1]
        cfg = {'preprocessing': {'max_deriv_order': [2, 1, 0]}, 'domain': {'boundary_width': 0}}
        for action in (_factor_arrays, build_search):
            with self.subTest(action=action.__name__), self.assertRaisesRegex(ValueError, 'orders'):
                action(p, p.data, cfg) if action is _factor_arrays else action(p, cfg)

    def test_run_rejects_nonfinite_supplied_derivatives(self):
        from epde_bench.runner import build_search
        p = fixture(); p.derivs = [np.ones((121, 2))]; p.derivs[0][0, 0] = np.nan
        p.deriv_orders = [2]
        with self.assertRaisesRegex(ValueError, 'derivatives must be finite'):
            build_search(p, {'preprocessing': {'max_deriv_order': [2]}})

    def test_nonfinite_observation_is_rejected(self):
        from epde_bench.runner import build_search
        p = fixture(); p.data['u'][0] = np.nan
        cfg = resolve_for_problem(load_config('ode'), p)
        with self.assertRaisesRegex(ValueError, 'NaN'):
            build_search(p, cfg)

    def test_nonuniform_fd_uses_coordinates(self):
        p = fixture(); t = np.linspace(0, 1, 121)**2
        p.grids = mesh(t); p.data = {'u': t**3}
        p.deriv_orders = [2] if p.derivs is not None else None
        cfg = {'preprocessing': {'max_deriv_order': [2], 'default_preprocessor_type': 'FD'},
               'search_space': {'data_fun_pow': 1, 'deriv_fun_pow': 1}}
        a = _factor_arrays(p, p.data, cfg)
        expected = np.gradient(np.gradient(t**3, t), t)
        np.testing.assert_allclose(a[_parse_factor('d^2u/dx0^2{power: 1.0}')], expected)


class InterfaceTests(unittest.TestCase):
    def test_check_respects_variant_and_overrides(self):
        import bench
        with patch('epde_bench.truthcheck.check', return_value=(None, [])) as check, \
             patch('epde_bench.truthcheck.print_check'):
            bench.main(['check', 'ode', '--noise', '0', '--variant', 'poly',
                        '--set', 'search.preprocessing.preprocessor_kwargs.polynomial_window=15'])
        self.assertEqual(check.call_args.kwargs['variant'], 'poly')
        self.assertEqual(check.call_args.kwargs['overrides']['search']['preprocessing']
                         ['preprocessor_kwargs']['polynomial_window'], 15)

    def test_script_check_forwards_configuration(self):
        import bench
        from epde_bench.script import main
        with patch.object(bench, 'main', return_value=0) as run:
            main(['ode'], ['--check', '--variant', 'poly', '--set', 'search.evolution.training_epochs=2'])
        args = run.call_args.args[0]
        self.assertIn('--variant', args); self.assertIn('poly', args)
        self.assertIn('--set', args); self.assertIn('search.evolution.training_epochs=2', args)

    def test_explicit_missing_config_is_rejected(self):
        from tempfile import TemporaryDirectory
        with TemporaryDirectory() as folder:
            with self.assertRaises(FileNotFoundError):
                load_config('ode', config_path=Path(folder)/'missing.yaml')

    def test_scalar_derivative_order_must_be_an_integer(self):
        from epde_bench.preprocessing import derivative_orders
        for value in (1.5, -0.5):
            with self.subTest(value=value), self.assertRaises(ValueError):
                derivative_orders({'preprocessing': {'max_deriv_order': value}}, 1)

    def test_omitted_boundary_uses_epde_default(self):
        from epde_bench.library import _interior
        from epde.interface.search_config import load_search_config
        p = fixture(); width = load_search_config({}).domain.boundary_width
        self.assertEqual(_interior(p, {}), (slice(width, p.shape[0]-width),))

    def test_empty_interior_is_rejected(self):
        from epde_bench.library import _interior
        p = fixture()
        with self.assertRaisesRegex(ValueError, 'boundary'):
            _interior(p, {'domain': {'boundary_width': 61}})

    def test_largest_ocean_rectangle_has_only_finite_values(self):
        from epde_bench.datasets import _finite_rectangle
        mask = np.ones((6, 8), dtype=bool); mask[:2, :3] = False; mask[4:, 6:] = False
        ys, xs = _finite_rectangle(mask)
        self.assertTrue(mask[ys, xs].all())
        # Brute force is independent of the implementation and small here.
        best = max((y1-y0)*(x1-x0) for y0 in range(6) for y1 in range(y0+1, 7)
                   for x0 in range(8) for x1 in range(x0+1, 9) if mask[y0:y1, x0:x1].all())
        self.assertEqual(mask[ys, xs].size, best)

    def test_ocean_rectangle_rejects_empty_mask(self):
        from epde_bench.datasets import _finite_rectangle
        with self.assertRaisesRegex(ValueError, 'finite'):
            _finite_rectangle(np.zeros((3, 4), dtype=bool))


@unittest.skipUnless((PIC/'data/jhtdb/jhtdb_pilot_plane.npz').exists(), 'JHTDB data absent')
class JHTDBTests(unittest.TestCase):
    def test_required_couplings_can_be_generated(self):
        from epde_bench.datasets import load
        from epde_bench.runner import build_search
        p = load('jhtdb_plane'); cfg = resolve_for_problem(load_config(p.name), p)
        with contextlib.redirect_stdout(io.StringIO()):
            search, trajectory, families = build_search(p, cfg)
            search.create_pool(data=[trajectory], additional_tokens=families)
        lookup = {label: f for f in search.pool.families for label in f.tokens}
        for label in ('u_z', 'v_z'):
            self.assertIsNot(lookup['w'], lookup[label])
            self.assertTrue(lookup['w'].status['meaningful'] or lookup[label].status['meaningful'])
        a = _factor_arrays(p, p.data, cfg)
        np.testing.assert_array_equal(a[_parse_factor('du/dx1{power: 1.0}')],
                                      p.derivs[0][:, 1].reshape(p.shape))
        from epde_bench.truthcheck import check_equation
        from epde_bench.library import _interior
        for equation in p.truth:
            self.assertGreater(check_equation(equation, a, _interior(p, cfg))['r2'], .99)


if __name__ == '__main__':
    unittest.main()
