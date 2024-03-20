import unittest
import logging

import numpy as np

from epde.optimizers.moeadd.solution_template import MOEADDSolution
from epde.optimizers.moeadd.supplementary import check_dominance, ndl_update, \
    fast_non_dominated_sorting, slow_non_dominated_sorting
from epde.optimizers.moeadd.moeadd import MOEADDOptimizer, ParetoLevels


level = logging.critical
logging.getLogger("base_moeadd").setLevel(level)

class DummySolutionClass(MOEADDSolution):
    def __hash__(self):
        return np.random.randint(0, 1e4)

class TestBaseSolution(unittest.TestCase):
    def setUp(self) -> None:
        dummy_obj_funs = [lambda x: np.max(np.abs(x)),
                          lambda x: np.min(np.abs(x)),
                          lambda x: x.size - np.count_nonzero(x)]
        self.solution_1 = DummySolutionClass(x = np.full(shape = 4, fill_value = 1), obj_funs = dummy_obj_funs)
        self.solution_2 = DummySolutionClass(x = np.array([0, 1, 2, 3]), obj_funs = dummy_obj_funs)
        self.solution_3 = DummySolutionClass(x = np.full(shape = 4, fill_value = 2), obj_funs = dummy_obj_funs)
        self.solution_4 = DummySolutionClass(x = np.full(shape = 4, fill_value = 1) + np.random.uniform(1e-9, 1e-8, size = 4),
                                             obj_funs = dummy_obj_funs)

    def test_basic_functionality(self):
        weights = [np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([0, 0, 1]), np.array([1., 1., 4.])/6.]
        self.assertEqual(self.solution_1.get_domain(weights = weights), 4, 
                         msg = 'Incorrect placement of the solution to domain')

    def test_dominance(self):
        self.assertTrue(check_dominance(self.solution_1, self.solution_3), 
                        msg = 'Expected dominance not detected.')
        self.assertFalse(check_dominance(self.solution_1, self.solution_2), 
                         msg = 'Got dominance between non-dominant individuals.')
        self.assertFalse(check_dominance(self.solution_1, self.solution_4), 
                         msg = 'Insignificant differences are still considered during dominance check.')

    def test_


class TestMOEADD(unittest.TestCase):
    def setUp(self) -> None:
        self.optimizer = MOEADDOptimizer() 

    def test_dominance_method(self):

        