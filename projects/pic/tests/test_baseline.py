"""Baseline representation and shared-preprocessing regressions; no EPDE fit."""
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
from epde_bench import baselines
from epde_bench.config import resolve_for_problem
from epde_bench.library import _blocks
from epde_bench.problem import Problem, mesh


def fixture(name='test'):
    t = np.linspace(0, 2, 121)
    return Problem(name, name, 'ode', 'synthetic', mesh(t),
                   {'u': np.exp(-t)}, ('t',))


def config(method='FD', kwargs=None):
    return {'search': {'domain': {'boundary_width': 5},
            'preprocessing': {'max_deriv_order': [1],
                              'default_preprocessor_type': method,
                              'preprocessor_kwargs': kwargs or {}},
            'search_space': {'data_fun_pow': 1, 'deriv_fun_pow': 1,
                             'equation_terms_max_number': 3}}}


class SupportTests(unittest.TestCase):
    def test_unrepresentable_problems_stop_before_preprocessing_or_fit(self):
        error = getattr(baselines, 'UnsupportedProblem', None)
        self.assertIsNotNone(error, 'unsupported equations need a distinct outcome')
        for name, reason in [('pde_divide', 'x'), ('ns', 'continuity')]:
            with self.subTest(name=name), patch.object(baselines, '_blocks') as blocks, \
                 patch.object(baselines, '_fit') as fit:
                p = fixture(name)
                with self.assertRaisesRegex(error, reason) as caught:
                    baselines.run_pysindy(p, p.data, {})
                self.assertEqual(caught.exception.problem, name)
                self.assertTrue(caught.exception.reason)
                blocks.assert_not_called()
                fit.assert_not_called()

    def test_support_classifier_does_not_use_truth(self):
        classifier = getattr(baselines, 'unsupported_reason', None)
        self.assertIsNotNone(classifier, 'support must be queryable before a run')
        for name in ('pde_divide', 'ns'):
            self.assertTrue(classifier(fixture(name)))
        p = fixture('ode')
        p.truth = ['arbitrary text never used to build the baseline']
        self.assertIsNone(classifier(p))


class PreprocessingTests(unittest.TestCase):
    def test_default_fit_uses_supplied_derivatives_instead_of_redifferentiating(self):
        p = fixture()
        # Deliberately differs from d(exp(-t))/dt: proves the supplied target
        # reaches regression, rather than merely comparing two near-equal FDs.
        p.derivs = [(7 * p.data['u']).reshape(-1, 1)]
        p.deriv_orders = [1]
        front, objectives, seconds, info = baselines.run_pysindy(p, p.data, config())
        equation = front[0][0]
        coefficient = float(equation.split(' * ')[0])
        self.assertAlmostEqual(coefficient, 7., places=4)
        self.assertIn('u{power: 1.0} = du/dx0{power: 1.0}', equation)
        self.assertLess(objectives[0][0], 1e-8)
        self.assertGreaterEqual(seconds, 0.)
        self.assertEqual(info['differentiation'], 'epde')

    def test_native_smoothed_fields_and_derivatives_reach_the_fit(self):
        from epde_bench.preprocessing import prepare_data
        p = fixture()
        p.data['u'] += np.random.default_rng(4).normal(0, .03, p.shape)
        cfg = config('poly', {'polynomial_window': 15, 'poly_order': 4,
                     'mp_poolsize': 1, 'use_smoothing': True, 'sigma': 1})
        search = resolve_for_problem(cfg, p)
        processed, derivatives = prepare_data(p, p.data, search)
        inner, train, val = baselines._rows(p, search, 0)
        expected_x = processed['u'][inner].ravel()[train]
        expected_y = derivatives['u'][:, 0].reshape(p.shape)[inner].ravel()[train]
        all_y = derivatives['u'][:, 0].reshape(p.shape)[inner].ravel()
        scale = np.std(all_y[np.concatenate([train, val])]) or 1.
        samples = []

        def capture(X_tr, y_tr, X_va, y_va, threshold, normalize):
            samples.append((X_tr.copy(), y_tr.copy()))
            return {'coef': np.array([-1.]), 'k': 1, 'mse': .01}

        with contextlib.redirect_stdout(io.StringIO()), patch.object(baselines, '_fit', capture):
            baselines.run_pysindy(p, p.data, cfg)
        np.testing.assert_allclose(samples[0][0][:, 0], expected_x, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(samples[0][1], expected_y / scale, rtol=1e-12, atol=1e-12)
        self.assertGreater(np.linalg.norm(processed['u'] - p.data['u']), 0.)

    def test_explicit_independent_differentiation_remains_available(self):
        p = fixture()
        cfg = config()
        cfg['pysindy'] = {'differentiation': 'finite_difference'}
        front, _, _, info = baselines.run_pysindy(p, p.data, cfg)
        self.assertAlmostEqual(float(front[0][0].split(' * ')[0]), -1., places=3)
        self.assertEqual(info['differentiation'], 'finite_difference')

    def test_independent_differentiation_rejects_supplied_derivative_protocol(self):
        p = fixture()
        p.derivs = [np.ones((121, 1))]
        p.deriv_orders = [1]
        cfg = config()
        cfg['pysindy'] = {'differentiation': 'finite_difference'}
        with self.assertRaisesRegex(ValueError, 'supplied derivatives'):
            baselines.run_pysindy(p, p.data, cfg)

    def test_unknown_differentiation_is_rejected(self):
        p = fixture()
        cfg = config()
        cfg['pysindy'] = {'differentiation': 'typo'}
        with self.assertRaisesRegex(ValueError, 'differentiation'):
            baselines.run_pysindy(p, p.data, cfg)


class TransportLibraryTests(unittest.TestCase):
    def assert_transport_available(self, problem, search):
        blocks, _ = _blocks(problem, problem.data, search, method='epde')
        columns = dict(baselines._columns(blocks, lhs_order=1))
        self.assertEqual(baselines._n_columns(blocks), len(columns))
        for label in ('u_z', 'v_z'):
            name = f'w{{power: 1.0}} * {label}{{power: 1.0}}'
            np.testing.assert_array_equal(columns[name],
                problem.named_arrays['w'][0] * problem.named_arrays[label][0])

    def test_out_of_plane_transport_is_available_on_a_small_fixture(self):
        p = fixture('jhtdb_plane')
        p.data['v'] = 2 * p.data['u']
        p.derivs = [np.ones((121, 1)), np.full((121, 1), 2.)]
        p.deriv_orders = [1]
        w, uz, vz = (np.full(p.shape, value) for value in (3., 5., 7.))
        p.token_groups = [('ns_oop_velocity', {'w': w}, True),
                          ('ns_oop_u_gradient', {'u_z': uz}, False),
                          ('ns_oop_v_gradient', {'v_z': vz}, False)]
        self.assert_transport_available(p, resolve_for_problem(config(), p))

    @unittest.skipUnless((PIC/'data/jhtdb/jhtdb_pilot_plane.npz').exists(), 'JHTDB data absent')
    def test_out_of_plane_transport_is_available_from_the_actual_loader(self):
        from epde_bench.config import load_config
        from epde_bench.datasets import load
        p = load('jhtdb_plane')
        self.assert_transport_available(p, resolve_for_problem(load_config(p.name), p))


if __name__ == '__main__':
    unittest.main()
