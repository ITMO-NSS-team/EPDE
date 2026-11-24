from types import SimpleNamespace
import unittest

import numpy as np
from measurements import FAMILIES, LEVEL, RHO, observe, rms


def problem(**fields):
    dims = next(iter(fields.values())).ndim
    return SimpleNamespace(data=fields, derivs=None, axis_names=('t', 'x', 'y')[:dims])


class MeasurementTests(unittest.TestCase):
    def test_spike_counts_and_energy(self):
        for shape in [(21,), (250,), (19, 21), (7, 8, 9)]:
            values = np.arange(np.prod(shape), dtype=float).reshape(shape)
            fields, meta = observe(problem(u=values), LEVEL, 17, family='spikes')
            noise = fields['u'] - values
            self.assertEqual(np.count_nonzero(noise), max(1, int(values.size * 0.01 + 0.5)))
            self.assertAlmostEqual(rms(noise), .05 * values.std())
            self.assertAlmostEqual(rms(noise), meta['field_rms']['u'])

    def test_separable_stationary_covariance(self):
        values = np.arange(100 * 90 * 80, dtype=float).reshape(100, 90, 80)
        fields, meta = observe(problem(u=values), LEVEL, 35, family='correlated')
        noise = (fields['u'] - values) / meta['target_sigma']['u']
        self.assertAlmostEqual(rms(noise), 1., delta=.045)
        for axis in range(3):
            z = np.moveaxis(noise, axis, 0)
            corr = np.corrcoef(z[1:].ravel(), z[:-1].ravel())[0, 1]
            self.assertAlmostEqual(corr, RHO, delta=.02)
        diagonal = np.corrcoef(noise[1:, 1:, :].ravel(), noise[:-1, :-1, :].ravel())[0, 1]
        self.assertAlmostEqual(diagonal, RHO**2, delta=.025)
        # Check the very first sample of a 1-D record across independent realizations.
        fields = problem(u=np.arange(12, dtype=float))
        first = [observe(fields, LEVEL, s, family='correlated')[0]['u'][0] for s in range(2000)]
        self.assertAlmostEqual(rms(first) / (.05 * fields.data['u'].std()), 1., delta=.05)

    def test_signal_dependent_rms(self):
        values = np.tile([0., -1., 4.], 50_000)
        fields, meta = observe(problem(u=values), LEVEL, 59, family='signal_dependent')
        noise = fields['u'] - values
        np.testing.assert_array_equal(noise[::3], 0.)
        self.assertAlmostEqual(rms(noise[2::3]) / rms(noise[1::3]), 4., delta=.07)
        self.assertAlmostEqual(rms(noise) / meta['target_sigma']['u'], 1., delta=.02)

    def test_field_and_search_rng_independence_and_replay(self):
        values = np.arange(3072, dtype=float).reshape(24, 128)
        p = problem(u=values, v=values.copy(), constant=np.full(values.shape, .1), zero=np.zeros_like(values))
        initial = np.random.get_state()
        for family in FAMILIES:
            first, meta = observe(p, LEVEL, 37, family=family)
            again, replay = observe(p, LEVEL, 37, family=family)
            changed, _ = observe(p, LEVEL, 38, family=family)
            self.assertEqual(meta, replay)
            for field in first:
                np.testing.assert_array_equal(first[field], again[field])
                self.assertFalse(np.shares_memory(first[field], p.data[field]))
            self.assertFalse(np.array_equal(first['u'], first['v']))
            self.assertFalse(np.array_equal(first['u'], changed['u']))
            for field in ('constant', 'zero'):
                np.testing.assert_array_equal(first[field], p.data[field])
                self.assertEqual(meta['field_rms'][field], 0.)
        current = np.random.get_state()
        np.testing.assert_array_equal(current[1], initial[1])
        self.assertEqual(current[2:], initial[2:])

    def test_unsupported_inputs(self):
        p = problem(u=np.arange(20.))
        for family, level in [('unknown', LEVEL), ('spikes', -4)]:
            with self.assertRaises(ValueError):
                observe(p, level, 17, family=family)
        p.derivs = ['precomputed']
        with self.assertRaises(ValueError):
            observe(p, LEVEL, 17, family='correlated')


if __name__ == '__main__':
    unittest.main()
