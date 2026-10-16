"""Run records distinguish unsupported experiments from search errors."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
PIC = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PIC), str(PIC.parent.parent)]
from epde_bench import runner
import numpy as np
from epde_bench.problem import Problem, mesh


def fixture(name, supplied=False):
    t = np.linspace(0, 2, 121)
    p = Problem(name, name, "ode", "synthetic", mesh(t),
                {"u": np.exp(-t)}, ("t",))
    if supplied:
        p.derivs = [(-p.data["u"]).reshape(-1, 1)]
        p.deriv_orders = [1]
    return p


class RunRecordTests(unittest.TestCase):
    def test_run_records_content_identity(self):
        with patch.object(runner, 'load', return_value=fixture('ode')), \
             patch.object(runner, 'discover', return_value=(None, [], [], .01)):
            record = runner.run_one('ode', noise=0, seed=2)
        self.assertEqual(record['status'], 'ok')
        self.assertIn('identity', record)
        self.assertEqual(record['identity']['job'], dict(dataset='ode', variant='default', noise=0., seed=2))

    def test_unsupported_baseline_is_a_distinct_record(self):
        with patch.object(runner, 'load', return_value=fixture('pde_divide')):
            record = runner.run_one('pde_divide', variant='pysindy')
        self.assertEqual(record['status'], 'unsupported')
        self.assertTrue(record['reason'])
        self.assertNotIn('metrics', record)

    def test_supplied_derivative_noise_is_an_unsupported_protocol(self):
        with patch.object(runner, 'load', return_value=fixture('jhtdb_plane', supplied=True)):
            record = runner.run_one('jhtdb_plane', noise=1)
        self.assertEqual(record['status'], 'unsupported')
        self.assertIn('noise', record['reason'])

if __name__ == '__main__': unittest.main()
