import sys
import unittest
from pathlib import Path
import numpy as np
PIC = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PIC), str(PIC.parent.parent)]
from epde_bench.problem import Problem, mesh
from epde_bench.application import ballbeam_validation

class ApplicationTests(unittest.TestCase):
    def fixture(self):
        t = np.linspace(0, 20, 401)
        return Problem('ballbeam', 'fixture', 'ode', 'synthetic', mesh(t), {'y': 3*t*t}, ('t',),
                       token_groups=[('forcing', {'u_in': np.ones(t.size)*2}, True)])
    def test_coefficient_and_heldout_trajectory(self):
        result = ballbeam_validation(self.fixture())
        self.assertAlmostEqual(result['coefficient'], 3., places=9)
        self.assertLess(result['acceleration']['rmse'], 1e-9)
        self.assertLess(result['trajectory']['rmse'], 1e-9)
        self.assertGreater(result['test_start'], result['train_stop'])
    def test_test_changes_do_not_change_training_coefficient(self):
        p = self.fixture(); before = ballbeam_validation(p)
        p.data['y'][before['split_index']:] += np.linspace(0, 5, len(p.data['y'])-before['split_index'])**3
        after = ballbeam_validation(p)
        self.assertEqual(before['coefficient'], after['coefficient'])
        self.assertGreater(after['trajectory']['rmse'], 1.)
    def test_short_segments_are_rejected(self):
        with self.assertRaises(ValueError): ballbeam_validation(self.fixture(), boundary=100)

if __name__ == '__main__': unittest.main()
