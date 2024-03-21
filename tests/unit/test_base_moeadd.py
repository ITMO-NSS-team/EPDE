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


class ArrayLikeSolution(MOEADDSolution):
    def __init__(self, x, obj_funs):
        super().__init__(x, obj_funs)
        self.x_size_sqrt = np.sqrt(self.vals.size)
    
    @property
    def obj_fun(self):
        if self.precomputed_value: 
            return self._obj_fun
        else:
            self._obj_fun = np.fromiter(map(lambda obj_fun: obj_fun(self.vals, self.x_size_sqrt), self.obj_funs),
                                        dtype = float)
            self.precomputed_value = True
            return self._obj_fun

    def __eq__(self, other):
        if isinstance(other, type(self)):
            epsilon = 1e-9
            return all([abs(self.vals[0] - other.vals[0]) < epsilon,
                        abs(self.vals[1] - other.vals[1]) < epsilon])
        else:
            return NotImplemented
    
    def __hash__(self):
        return hash(tuple(self.vals))


class TestBaseSolution(unittest.TestCase):
    def setUp(self):
        dummy_obj_funs = [lambda x: np.max(np.abs(x)),
                          lambda x: np.min(np.abs(x)),
                          lambda x: np.count_nonzero(x)]
        self.solution_1 = DummySolutionClass(x = np.full(shape = 4, fill_value = 1), obj_funs = dummy_obj_funs)
        self.solution_2 = DummySolutionClass(x = np.array([0, 1, 2, 3]), obj_funs = dummy_obj_funs)
        self.solution_3 = DummySolutionClass(x = np.full(shape = 4, fill_value = 2), obj_funs = dummy_obj_funs)
        self.solution_4 = DummySolutionClass(x = np.full(shape = 4, fill_value = 1) + np.random.uniform(1e-9, 1e-8, 
                                                                                                        size = 4),
                                             obj_funs = dummy_obj_funs)

    def test_basic_functionality(self):
        weights = [np.array([1, 0, 0]),
                   np.array([0, 1, 0]),
                   np.array([0, 0, 1]),
                   np.array([1., 1., 4.])/6.]
        
        self.assertEqual(self.solution_1.get_domain(weights = weights), 3,
                         msg = 'Incorrect placement of the solution to domain.')
        self.assertTrue(self.solution_1.precomputed_domain, msg = 'Domain saving did not work.')
        self.assertEqual(self.solution_1.crossover_times(), 0,
                         msg = 'Individual is incorrectly marked as crossover-selected.')
        self.solution_1.incr_counter()
        self.assertEqual(self.solution_1.crossover_times(), 1,
                         msg = 'Individual is not marked as crossover-selected, while it should be.')
        
        self.assertTrue(check_dominance(self.solution_1, self.solution_3), 
                        msg = 'Expected dominance not detected.')
        self.assertFalse(check_dominance(self.solution_1, self.solution_2), 
                         msg = 'Got dominance between non-dominant individuals.')
        self.assertFalse(check_dominance(self.solution_1, self.solution_4), 
                         msg = 'Insignificant differences are still considered during dominance check.')


class TestMOEADD(unittest.TestCase):
    def setUp(self) -> None:
        
        self.optimizer = MOEADDOptimizer()

    def test_dominance_method(self):

        