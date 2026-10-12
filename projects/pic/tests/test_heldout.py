"""Chronological forecast protocol: future values cannot affect training."""
import sys
import json
import unittest
from pathlib import Path
from unittest.mock import patch
from dataclasses import replace
import numpy as np

PIC = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PIC), str(PIC.parent.parent)]
from epde_bench.problem import Problem, mesh


class HeldoutTests(unittest.TestCase):
    def test_constant_interval_has_json_safe_undefined_r2(self):
        from epde_bench.heldout import _scores
        values = np.ones(5)
        scores = _scores({'u': values}, {'u': values}, True)
        self.assertEqual(scores['u']['rmse'], 0)
        self.assertIsNone(scores['u']['r2'])
        json.dumps(scores, allow_nan=False)

    def setUp(self):
        t = np.linspace(0, 2, 101)
        self.problem = Problem('decay', 'Decay', 'ode', 'synthetic', mesh(t),
                               {'u': np.exp(-t)}, ('t',))
        self.cfg = {'domain': {'boundary_width': 2},
                    'preprocessing': {'default_preprocessor_type': 'FD', 'max_deriv_order': [1]}}

    def test_future_mutation_cannot_change_training_selection_or_forecast(self):
        from epde_bench.heldout import chronological_split, evaluate_candidates, forecast_ode
        changed = replace(self.problem, data={'u': self.problem.data['u'].copy()})
        changed.data['u'][70:] += 100 * np.arange(31)
        candidates = [['-1.0 * u{power:1.0} = du/dx0{power:1.0}'],
                      ['-2.0 * u{power:1.0} = du/dx0{power:1.0}']]
        train_a, split = chronological_split(self.problem, .7)
        train_b, _ = chronological_split(changed, .7)
        np.testing.assert_array_equal(train_a.data['u'], train_b.data['u'])
        result_a = evaluate_candidates(train_a, candidates, self.cfg)
        result_b = evaluate_candidates(train_b, candidates, self.cfg)
        self.assertEqual(result_a, result_b)
        system = result_a['selected']
        forecast_a = forecast_ode(train_a, system, self.cfg, self.problem.grids[0][split:])
        forecast_b = forecast_ode(train_b, system, self.cfg, changed.grids[0][split:])
        np.testing.assert_array_equal(forecast_a.simulated['u'], forecast_b.simulated['u'])
        self.assertTrue(forecast_a.success)
        np.testing.assert_allclose(forecast_a.simulated['u'], self.problem.data['u'][split:], atol=1e-5)

    def test_supplied_derivatives_are_rejected_before_slicing(self):
        from epde_bench.heldout import chronological_split
        with self.assertRaisesRegex(ValueError, 'derivative'):
            chronological_split(replace(self.problem, derivs=[np.zeros((101, 1))], deriv_orders=[1]))

    def test_highest_derivative_can_be_reoriented_without_refitting(self):
        from epde_bench.heldout import explicit_system
        system = explicit_system(self.problem,
                  ['-0.5 * du/dx0{power:1.0} + 0.0 = u{power:1.0}'])
        self.assertIsNotNone(system)
        from epde_bench.reconstruction import parse_equation
        terms, constant, target, coef = parse_equation(system[0])
        self.assertEqual(target[0][0], 'du/dx0')
        self.assertEqual(terms[0][0], -2.0)

    def test_runner_builds_search_only_after_split_and_closes_it(self):
        from epde_bench.heldout import run_heldout
        seen = []
        class Search:
            closed = False
            def close(self):
                self.closed = True
        search = Search()
        def discover(train, cfg):
            seen.append(train.data['u'].copy())
            return search, [['-1.0 * u{power:1.0} = du/dx0{power:1.0}']], [[0.0]], .1
        with patch('epde_bench.heldout.load', return_value=self.problem), \
             patch('epde_bench.heldout.discover', side_effect=discover):
            record = run_heldout('ode', overrides={'search': self.cfg})
        self.assertEqual(len(seen[0]), 70)
        self.assertTrue(search.closed)
        self.assertEqual(record['test']['n_points'], 31)
        self.assertTrue(record['test']['integration_success'])
        self.assertGreater(record['test']['metrics']['u']['r2'], .999)


class LifecycleTests(unittest.TestCase):
    def test_fit_failure_closes_search_before_exception_propagates(self):
        from epde_bench.heldout import discover
        class FailingSearch:
            closed = False
            def fit(self, **kwargs):
                raise RuntimeError('fit failed')
            def close(self):
                self.closed = True
        search = FailingSearch()
        with patch('epde_bench.heldout.build_search', return_value=(search, object(), [])):
            with self.assertRaisesRegex(RuntimeError, 'fit failed'):
                discover(None, {})
        self.assertTrue(search.closed)
