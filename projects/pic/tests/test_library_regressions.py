"""Forcing must be representable by discovery and sparse regression."""
import sys
import unittest
from pathlib import Path

PIC = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(PIC), str(PIC.parent.parent)]
from epde_bench import env
env.pin_blas_threads(1)
import numpy as np
from epde_bench.baselines import _columns
from epde_bench.config import load_config, resolve_for_problem, split_bench_tokens
from epde_bench.library import _blocks
from epde_bench.problem import Problem, mesh


def fixture():
    t = np.linspace(0, 2, 31)
    p = Problem('duffing', 'Duffing', 'ode', 'synthetic', mesh(t),
                {'u': np.sin(t)}, ('t',))
    p.derivs = [np.column_stack((np.cos(t), -np.sin(t)))]
    p.deriv_orders = [2]
    return p


class TrigonometricLibraryTests(unittest.TestCase):
    def test_configured_duffing_has_standalone_cosine_in_both_libraries(self):
        p = fixture()
        search = resolve_for_problem(load_config('duffing'), p)
        blocks, _ = _blocks(p, p.data, search)
        columns = dict(_columns(blocks, lhs_order=2))
        cosine = 'cos{power: 1.0, freq: 1.0, dim: 0.0}'
        self.assertIn(cosine, columns, 'forcing cos(t) must stand alone')
        np.testing.assert_array_equal(columns[cosine], np.cos(p.grids[0]))
        _, families = split_bench_tokens(search)
        trig = next(f for f in families if 'cos' in f.token_family.tokens)
        self.assertTrue(trig.token_family.status['meaningful'])

    def test_library_honors_meaningful_flag_with_default_false(self):
        p = fixture()
        for flag in (True, False, None):
            with self.subTest(flag=flag):
                search = resolve_for_problem(load_config('duffing'), p)
                spec = {'family': 'fixed_trigonometric', 'freq': 1.0}
                if flag is not None:
                    spec['meaningful'] = flag
                search['search_space']['tokens'] = [spec]
                blocks, _ = _blocks(p, p.data, search)
                columns = dict(_columns(blocks, lhs_order=2))
                cosine = 'cos{power: 1.0, freq: 1.0, dim: 0.0}'
                self.assertEqual(cosine in columns, flag is True)
                product = 'u{power: 1.0} * ' + cosine
                np.testing.assert_array_equal(columns[product], p.data['u'] * np.cos(p.grids[0]))


if __name__ == '__main__':
    unittest.main()
