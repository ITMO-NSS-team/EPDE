"""The short evolution lesson executes the actual optimizer and records its front."""
import importlib
import math
import sys
import unittest
from pathlib import Path

PIC = Path(__file__).resolve().parents[1]
APP = PIC / 'app'
sys.path[:0] = [str(APP), str(PIC), str(PIC.parent.parent)]
from epde_bench import env
env.pin_blas_threads(1)


class OptimizerDemoTests(unittest.TestCase):
    def test_live_run_observes_epochs_and_recovers_the_known_oscillator(self):
        try:
            module = importlib.import_module('epde_bench.optimizer_demo')
        except ModuleNotFoundError:
            module = None
        self.assertIsNotNone(module, 'the live optimizer lesson needs a reusable runner')
        result = module.run_optimizer_demo()
        self.assertEqual([entry['epoch'] for entry in result['history']], [1, 2, 3])
        self.assertEqual([row['Epoch'] for row in result['rows']], [1, 2, 3])
        self.assertGreaterEqual(result['seconds'], 0.)
        for entry in result['history']:
            self.assertTrue(entry['front'])
            for solution in entry['front']:
                self.assertTrue(all(math.isfinite(v) for v in solution['obj_fun']))
                self.assertIn('=', solution['text_form'])
        from epde_bench.metrics import coefficient_error_best
        truth = ['-1.0625 * u{power: 1.0} + -0.5 * du/dx0{power: 1.0} '
                 '= d^2u/dx0^2{power: 1.0}']
        self.assertTrue(result['equations'])
        self.assertLess(coefficient_error_best(result['equations'], [truth]), 1e-10)


if __name__ == '__main__':
    unittest.main()
